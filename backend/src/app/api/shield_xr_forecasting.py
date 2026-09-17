"""SHIELD-XR training, inference, and performance API."""

from __future__ import annotations

import logging
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.forecasting.performance_monitoring import get_performance_monitoring
from app.forecasting.schemas import (
    ForecastRequest,
    ForecastResponse,
    PaginatedReceiptDrugSearchResponse,
    PerformanceMonitoringResponse,
    ReceiptDrugOption,
    ShieldXRAccuracySummary,
    ShieldXRStatusResponse,
    TrainForecastingRequest,
    TrainStatusResponse,
)
from app.forecasting.shield_xr.artifacts import is_trained, load_training_meta
from app.forecasting.shield_xr.forecaster import ShieldXRForecaster, ShieldXRNotTrainedError
from app.forecasting.shield_xr.trainer import (
    SHIELD_XR_MODELS,
    ShieldXRTrainer,
    resolve_training_drug_codes,
)
from app.services.enriched_demand import enriched_data_exists, search_enriched_drug_codes

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/forecasting", tags=["Forecasting"])

MIN_DRUG_SEARCH_LENGTH = 3
DEFAULT_DRUG_SEARCH_PAGE_SIZE = 10


def _paginate(total: int, page: int, page_size: int) -> int:
    if total <= 0:
        return 0
    return (total + page_size - 1) // page_size


def _accuracy_summary_from_meta(meta: dict) -> ShieldXRAccuracySummary:
    per_sb = meta.get("per_sb_class_weekly_accuracy") or {}
    return ShieldXRAccuracySummary(
        daily_ensemble_accuracy_pct=_pct(meta.get("test_accuracy")),
        hospital_weekly_accuracy_pct=_pct(meta.get("hospital_weekly_accuracy")),
        reconciled_weekly_per_drug_mean_accuracy_pct=_pct(
            meta.get("reconciled_weekly_per_drug_mean_accuracy"),
        ),
        reconciled_weekly_per_drug_median_accuracy_pct=_pct(
            meta.get("reconciled_weekly_per_drug_median_accuracy"),
        ),
        volume_weighted_weekly_accuracy_pct=_pct(meta.get("volume_weighted_weekly_accuracy")),
        non_lumpy_weekly_accuracy_pct=_pct(meta.get("non_lumpy_weekly_accuracy")),
        hybrid_abc_combined_accuracy_pct=_pct(meta.get("hybrid_abc_combined_accuracy")),
        per_sb_class_weekly_accuracy_pct={
            str(key): round(float(value) * 100, 2) for key, value in per_sb.items()
        },
    )


def _pct(value: object) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if numeric <= 1.0:
        return round(numeric * 100, 2)
    return round(numeric, 2)


@router.get(
    "/status",
    response_model=ShieldXRStatusResponse,
    summary="SHIELD-XR training status and hold-out accuracy",
)
def shield_xr_status() -> ShieldXRStatusResponse:
    ready = is_trained()
    if not ready:
        return ShieldXRStatusResponse(models_ready=False)

    meta = load_training_meta()
    return ShieldXRStatusResponse(
        models_ready=True,
        training_run_id=meta.get("training_run_id"),
        train_end=meta.get("train_end"),
        valid_end=meta.get("valid_end"),
        n_skus=meta.get("n_skus"),
        accuracy_summary=_accuracy_summary_from_meta(meta),
    )


@router.post(
    "/train",
    response_model=TrainStatusResponse,
    summary="Train SHIELD-XR forecasting models",
)
def train_forecasting_models(
    request: TrainForecastingRequest,
    db: Annotated[Session, Depends(get_db)],
) -> TrainStatusResponse:
    if not enriched_data_exists(db):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "No enriched training data found. Upload hospital Excel exports "
                "via Data Ingestion before training."
            ),
        )

    drug_codes = resolve_training_drug_codes(db, request.drug_codes)
    if not drug_codes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No drug codes found in hospital_daily_demand_enriched.",
        )

    trainer = ShieldXRTrainer()
    try:
        response = trainer.train_all(
            drug_codes=drug_codes,
            models_to_train=request.models or SHIELD_XR_MODELS,
            db_session=db,
            force_retrain=request.force_retrain,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    logger.info(
        "SHIELD-XR training complete: drugs=%d run=%s weekly_acc=%s",
        response.drugs_trained,
        response.training_run_id,
        response.accuracy_summary.get("hospital_weekly_accuracy_pct"),
    )
    return response


@router.get(
    "/performance",
    response_model=PerformanceMonitoringResponse,
    summary="SHIELD-XR model performance metrics",
)
def performance_monitoring(
    db: Annotated[Session, Depends(get_db)],
    drug_code: Annotated[Optional[str], Query()] = None,
    model_name: Annotated[Optional[str], Query()] = None,
    training_run_id: Annotated[Optional[str], Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> PerformanceMonitoringResponse:
    return get_performance_monitoring(
        db,
        drug_code=drug_code,
        model_name=model_name or "shield_xr_ensemble",
        training_run_id=training_run_id,
        limit=limit,
    )


@router.get(
    "/enriched-drugs/search",
    response_model=PaginatedReceiptDrugSearchResponse,
    summary="Search drug codes in the enriched hospital demand panel",
)
def search_enriched_drugs_endpoint(
    q: Annotated[str, Query(min_length=MIN_DRUG_SEARCH_LENGTH)],
    db: Annotated[Session, Depends(get_db)],
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=50)] = DEFAULT_DRUG_SEARCH_PAGE_SIZE,
) -> PaginatedReceiptDrugSearchResponse:
    rows, total = search_enriched_drug_codes(db, q, page=page, page_size=page_size)
    return PaginatedReceiptDrugSearchResponse(
        items=[ReceiptDrugOption(**row) for row in rows],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=_paginate(total, page, page_size),
    )


@router.post(
    "/predict",
    response_model=ForecastResponse,
    summary="SHIELD-XR per-drug demand forecast",
)
def predict_drug_demand(
    request: ForecastRequest,
    db: Annotated[Session, Depends(get_db)],
) -> ForecastResponse:
    forecaster = ShieldXRForecaster()
    try:
        return forecaster.forecast(
            drug_code=request.drug_code,
            horizon_days=request.horizon_days,
            center_syn_id=request.center_syn_id,
            db_session=db,
        )
    except ShieldXRNotTrainedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
