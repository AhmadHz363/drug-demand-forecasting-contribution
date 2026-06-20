"""Cold Start API — training endpoints and cold-start prediction."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.cold_start.autoencoder.embedder import clear_model_cache
from app.cold_start.autoencoder.trainer import _artifact_path, train_autoencoder
from app.cold_start.receipt_adapter import load_receipt_drug_metadata
from app.cold_start.cold_start_service import (
    ColdStartService,
    DrugCatalogEmptyError,
    DrugReceiptsEmptyError,
    invalidate_library_embedding_cache,
)
from app.cold_start.drug_metadata_encoder import encode_drug_metadata
from app.cold_start.maml.trainer import artifact_path as maml_artifact_path
from app.cold_start.maml.trainer import train_maml
from app.cold_start.schemas import ColdStartPredictRequest, ColdStartPredictResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cold-start", tags=["Cold Start"])


class TrainEmbedderResponse(BaseModel):
    status: str
    drugs_trained_on: int
    artifact_path: str


class TrainMamlResponse(BaseModel):
    status: str
    artifact_path: str
    tasks_trained_on: int


@router.post(
    "/train-embedder",
    response_model=TrainEmbedderResponse,
    summary="Train drug metadata autoencoder",
    response_description="Training summary and path to saved weights.",
)
def train_embedder(
    db: Annotated[Session, Depends(get_db)],
) -> TrainEmbedderResponse:
    # TODO: restrict to admin role
    metadata_list = load_receipt_drug_metadata(db)
    if len(metadata_list) < 5:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Need at least 5 drugs in drug_receipts before training.",
        )

    encoded_drugs = [encode_drug_metadata(meta) for meta in metadata_list]

    try:
        train_autoencoder(encoded_drugs)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    clear_model_cache()
    invalidate_library_embedding_cache()

    artifact_path = _artifact_path()
    logger.info(
        "Autoencoder trained on %d drugs; artifact=%s",
        len(metadata_list),
        artifact_path,
    )
    return TrainEmbedderResponse(
        status="ok",
        drugs_trained_on=len(metadata_list),
        artifact_path=artifact_path,
    )


@router.post(
    "/train-maml",
    response_model=TrainMamlResponse,
    summary="Meta-train MAML base forecaster",
    response_description="Training summary and path to saved MAML weights.",
)
def train_maml_endpoint(
    db: Annotated[Session, Depends(get_db)],
) -> TrainMamlResponse:
    # TODO: restrict to admin role
    try:
        _, tasks_trained_on = train_maml(db)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    artifact_path = maml_artifact_path()
    logger.info(
        "MAML trained on %d eligible drugs; artifact=%s",
        tasks_trained_on,
        artifact_path,
    )
    return TrainMamlResponse(
        status="ok",
        artifact_path=artifact_path,
        tasks_trained_on=tasks_trained_on,
    )


@router.post(
    "/predict",
    response_model=ColdStartPredictResponse,
    summary="Cold-start demand forecast for a new drug",
    response_description="Embedding, neighbours, graduation stage, and forecast intervals.",
)
def predict_cold_start(
    request: ColdStartPredictRequest,
    db: Annotated[Session, Depends(get_db)],
) -> ColdStartPredictResponse:
    service = ColdStartService(db)

    try:
        return service.predict(request)
    except RuntimeError as exc:
        if "Autoencoder not trained yet" in str(exc):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Embedder not ready. Train the autoencoder first.",
            ) from exc
        raise
    except DrugReceiptsEmptyError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except DrugCatalogEmptyError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
