"""Cold Start API — CAMEO training and cold-start prediction."""

from __future__ import annotations

import logging
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.cold_start.cameo.artifacts import artifact_path, clear_artifact_cache, is_trained, load_validation_summary
from app.cold_start.cameo.trainer import CameoTrainingError, train_cameo
from app.cold_start.cold_start_service import (
    ColdStartService,
    DrugCatalogEmptyError,
    DrugReceiptsEmptyError,
    invalidate_library_embedding_cache,
)
from app.cold_start.schemas import (
    CameoAccuracySummary,
    CameoPerDrugAccuracy,
    ColdStartPredictRequest,
    ColdStartPredictResponse,
    ColdStartStatusResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cold-start", tags=["Cold Start"])


class TrainCameoResponse(BaseModel):
    status: str
    drugs_trained_on: int
    artifact_path: str
    accuracy_summary: Optional[CameoAccuracySummary] = None


# Backward-compatible alias for dashboard training controls
TrainEmbedderResponse = TrainCameoResponse
TrainMamlResponse = TrainCameoResponse


def _accuracy_summary_from_validation(raw: Optional[dict]) -> Optional[CameoAccuracySummary]:
    if not raw:
        return None
    per_drug = [
        CameoPerDrugAccuracy(
            drug_code=str(row.get("drug_code", "")),
            sb_class=row.get("sb_class"),
            accuracy_pct=row.get("accuracy_pct"),
            wape=row.get("wape"),
        )
        for row in raw.get("per_drug") or []
    ]
    return CameoAccuracySummary(
        n_historical_library_drugs=int(raw.get("n_historical_library_drugs") or 0),
        n_coldstart_test_drugs=int(raw.get("n_coldstart_test_drugs") or 0),
        forecast_horizon_weeks=int(raw.get("forecast_horizon_weeks") or 20),
        pooled_accuracy_pct=dict(raw.get("pooled_accuracy_pct") or {}),
        median_accuracy_pct=dict(raw.get("median_accuracy_pct") or {}),
        win_rate_pct=dict(raw.get("win_rate_pct") or {}),
        per_sb_class_pooled_accuracy_pct=dict(raw.get("per_sb_class_pooled_accuracy_pct") or {}),
        empirical_conformal_coverage_pct=raw.get("empirical_conformal_coverage_pct"),
        conformal_half_width_weekly=raw.get("conformal_half_width_weekly"),
        cameo_vs_analogous_wape_improvement_pct=raw.get("cameo_vs_analogous_wape_improvement_pct"),
        smooth_pooled_accuracy_pct=raw.get("smooth_pooled_accuracy_pct"),
        operational_pooled_accuracy_pct=raw.get("operational_pooled_accuracy_pct"),
        non_lumpy_operational_pooled_accuracy_pct=raw.get("non_lumpy_operational_pooled_accuracy_pct"),
        zero_demand_test_drugs=int(raw.get("zero_demand_test_drugs") or 0),
        per_drug=per_drug,
        note=raw.get("note"),
    )


@router.get(
    "/status",
    response_model=ColdStartStatusResponse,
    summary="CAMEO training status and hold-out validation accuracy",
)
def cold_start_status() -> ColdStartStatusResponse:
    if not is_trained():
        return ColdStartStatusResponse(model_ready=False)

    validation = load_validation_summary()
    trained_at: str | None = None
    drugs_trained_on: int | None = None
    try:
        import json
        import os

        with open(artifact_path(), encoding="utf-8") as handle:
            payload = json.load(handle)
        trained_at = payload.get("trained_at")
        drugs_trained_on = len(payload.get("hist_drug_codes") or [])
    except (OSError, json.JSONDecodeError):
        pass

    return ColdStartStatusResponse(
        model_ready=True,
        drugs_trained_on=drugs_trained_on,
        trained_at=trained_at,
        accuracy_summary=_accuracy_summary_from_validation(validation),
    )


@router.post(
    "/train-cameo",
    response_model=TrainCameoResponse,
    summary="Train CAMEO metric-learning cold-start model",
)
def train_cameo_endpoint(
    db: Annotated[Session, Depends(get_db)],
) -> TrainCameoResponse:
    try:
        artifacts, drugs_trained_on = train_cameo(db)
    except CameoTrainingError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    clear_artifact_cache()
    invalidate_library_embedding_cache()
    path = artifact_path()
    logger.info("CAMEO trained on %d drugs; artifact=%s", drugs_trained_on, path)
    return TrainCameoResponse(
        status="ok",
        drugs_trained_on=drugs_trained_on,
        artifact_path=path,
        accuracy_summary=_accuracy_summary_from_validation(artifacts.validation_summary),
    )


@router.post(
    "/train-embedder",
    response_model=TrainEmbedderResponse,
    summary="Train CAMEO model (legacy alias)",
    deprecated=True,
)
def train_embedder(
    db: Annotated[Session, Depends(get_db)],
) -> TrainEmbedderResponse:
    return train_cameo_endpoint(db)


@router.post(
    "/train-maml",
    response_model=TrainMamlResponse,
    summary="No-op legacy endpoint — CAMEO replaces MAML",
    deprecated=True,
)
def train_maml_endpoint(
    db: Annotated[Session, Depends(get_db)],
) -> TrainMamlResponse:
    return train_cameo_endpoint(db)


@router.post(
    "/predict",
    response_model=ColdStartPredictResponse,
    summary="CAMEO cold-start demand forecast for a new drug",
)
def predict_cold_start(
    request: ColdStartPredictRequest,
    db: Annotated[Session, Depends(get_db)],
) -> ColdStartPredictResponse:
    service = ColdStartService(db)

    try:
        return service.predict(request)
    except RuntimeError as exc:
        if "not trained yet" in str(exc).lower():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="CAMEO not ready. Train via POST /cold-start/train-cameo first.",
            ) from exc
        raise
    except (DrugReceiptsEmptyError, DrugCatalogEmptyError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
