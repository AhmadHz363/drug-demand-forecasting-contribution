"""Forecasting API — model training and demand prediction."""

from __future__ import annotations

import logging
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.forecasting.inference.forecaster import (
    DrugForecaster,
    DrugNotInCatalogError,
    NoTrainedModelsError,
    validate_forecast_quantiles,
)
from app.forecasting.performance_monitoring import get_performance_monitoring
from app.forecasting.schemas import (
    BatchForecastRequest,
    ForecastRequest,
    ForecastResponse,
    HoldoutDateSuggestions,
    HoldoutResponse,
    HoldoutValidationRequest,
    ModelWeightBreakdown,
    PaginatedDrugCodeSearchResponse,
    PaginatedReceiptDrugSearchResponse,
    PerformanceMonitoringResponse,
    ReceiptDrugOption,
    TrainForecastingRequest,
    TrainStatusResponse,
)
from app.forecasting.training.holdout import run_holdout_validation
from app.forecasting.training.holdout_dates import compute_holdout_date_suggestions
from app.forecasting.training.trainer import ForecastingTrainer
from app.services.demand_aggregation import (
    get_distinct_drug_codes_from_receipts,
    get_receipt_date_bounds,
    search_receipt_drug_codes,
)
from app.services.forecast_drug_lookup import search_forecasted_drug_codes

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/forecasting", tags=["Forecasting"])

MIN_DRUG_SEARCH_LENGTH = 3
DEFAULT_DRUG_SEARCH_PAGE_SIZE = 10


def _paginate(total: int, page: int, page_size: int) -> int:
    if total <= 0:
        return 0
    return (total + page_size - 1) // page_size


@router.get(
    "/receipt-drug-codes/search",
    response_model=PaginatedReceiptDrugSearchResponse,
    summary="Search drug codes in drug_receipts",
)
def search_receipt_drug_codes_endpoint(
    q: Annotated[str, Query(min_length=MIN_DRUG_SEARCH_LENGTH)],
    db: Annotated[Session, Depends(get_db)],
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=50)] = DEFAULT_DRUG_SEARCH_PAGE_SIZE,
) -> PaginatedReceiptDrugSearchResponse:
    rows, total = search_receipt_drug_codes(db, q, page=page, page_size=page_size)
    return PaginatedReceiptDrugSearchResponse(
        items=[ReceiptDrugOption(**row) for row in rows],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=_paginate(total, page, page_size),
    )


@router.get(
    "/forecasted-drugs/search",
    response_model=PaginatedDrugCodeSearchResponse,
    summary="Search drug codes that have forecast results",
)
def search_forecasted_drugs_endpoint(
    db: Annotated[Session, Depends(get_db)],
    q: Annotated[Optional[str], Query(min_length=0)] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=50)] = DEFAULT_DRUG_SEARCH_PAGE_SIZE,
) -> PaginatedDrugCodeSearchResponse:
    normalized = (q or "").strip()
    if normalized and len(normalized) < MIN_DRUG_SEARCH_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Search query must be at least {MIN_DRUG_SEARCH_LENGTH} characters.",
        )

    items, total = search_forecasted_drug_codes(
        db,
        normalized or None,
        page=page,
        page_size=page_size,
    )
    return PaginatedDrugCodeSearchResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=_paginate(total, page, page_size),
    )


@router.post(
    "/train",
    response_model=TrainStatusResponse,
    summary="Train forecasting models",
    response_description="Training summary with sMAPE and saved artifact paths.",
)
def train_forecasting_models(
    request: TrainForecastingRequest,
    db: Annotated[Session, Depends(get_db)],
) -> TrainStatusResponse:
    # TODO: make async with Celery task for production
    if request.drug_codes is None:
        drug_codes = get_distinct_drug_codes_from_receipts(db)
    else:
        drug_codes = request.drug_codes

    if not drug_codes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No drug codes to train. Provide drug_codes or load drug_receipts data.",
        )

    trainer = ForecastingTrainer()
    try:
        response = trainer.train_all(
            drug_codes=drug_codes,
            models_to_train=request.models,
            db_session=db,
            force_retrain=request.force_retrain,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    logger.info(
        "Forecasting training complete: drugs=%d models=%s run=%s",
        response.drugs_trained,
        response.models_trained,
        response.training_run_id,
    )
    return response


@router.get(
    "/performance",
    response_model=PerformanceMonitoringResponse,
    summary="Model health metrics from training runs",
    response_description=(
        "Recent walk-forward sMAPE/MASE, coverage, weight snapshots, and drift flags."
    ),
)
def get_forecasting_performance(
    db: Annotated[Session, Depends(get_db)],
    drug_code: Annotated[Optional[str], Query(min_length=1)] = None,
    model_name: Annotated[Optional[str], Query(min_length=1)] = None,
    training_run_id: Annotated[Optional[str], Query(min_length=1)] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> PerformanceMonitoringResponse:
    return get_performance_monitoring(
        db,
        drug_code=drug_code.strip() if drug_code else None,
        model_name=model_name.strip() if model_name else None,
        training_run_id=training_run_id.strip() if training_run_id else None,
        limit=limit,
    )


@router.post(
    "/predict",
    response_model=ForecastResponse,
    summary="Forecast demand for a single drug",
    response_description="Calibrated quantile forecast with optional SHAP and attention.",
)
def predict_drug_demand(
    request: ForecastRequest,
    db: Annotated[Session, Depends(get_db)],
) -> ForecastResponse:
    forecaster = DrugForecaster()
    try:
        response = forecaster.forecast(
            drug_code=request.drug_code,
            horizon_days=request.horizon_days,
            center_syn_id=request.center_syn_id,
            include_shap=request.include_shap,
            include_attention=request.include_attention,
            db_session=db,
        )
    except DrugNotInCatalogError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except NoTrainedModelsError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    validate_forecast_quantiles(response)
    return response


@router.post(
    "/predict-batch",
    response_model=list[ForecastResponse],
    summary="Forecast demand for multiple drugs",
    response_description="One ForecastResponse per drug; failures include an error field.",
)
def predict_batch(
    request: BatchForecastRequest,
    db: Annotated[Session, Depends(get_db)],
) -> list[ForecastResponse]:
    forecaster = DrugForecaster()
    responses: list[ForecastResponse] = []

    for drug_code in request.drug_codes:
        try:
            response = forecaster.forecast(
                drug_code=drug_code,
                horizon_days=request.horizon_days,
                center_syn_id=None,
                include_shap=request.include_shap,
                include_attention=False,
                db_session=db,
            )
            validate_forecast_quantiles(response)
            responses.append(response)
        except DrugNotInCatalogError as exc:
            responses.append(
                ForecastResponse(
                    drug_code=drug_code,
                    center_syn_id=None,
                    horizon_days=request.horizon_days,
                    model_weights=ModelWeightBreakdown(sarima=0.0, lgbm=0.0, classical=0.0),
                    forecast=[],
                    uncertainty_note="",
                    error=str(exc),
                )
            )
        except NoTrainedModelsError as exc:
            responses.append(
                ForecastResponse(
                    drug_code=drug_code,
                    center_syn_id=None,
                    horizon_days=request.horizon_days,
                    model_weights=ModelWeightBreakdown(sarima=0.0, lgbm=0.0, classical=0.0),
                    forecast=[],
                    uncertainty_note="",
                    error=str(exc),
                )
            )
        except Exception as exc:
            logger.exception("Batch forecast failed for %s", drug_code)
            responses.append(
                ForecastResponse(
                    drug_code=drug_code,
                    center_syn_id=None,
                    horizon_days=request.horizon_days,
                    model_weights=ModelWeightBreakdown(sarima=0.0, lgbm=0.0, classical=0.0),
                    forecast=[],
                    uncertainty_note="",
                    error=str(exc),
                )
            )

    return responses


@router.get(
    "/holdout/date-suggestions",
    response_model=HoldoutDateSuggestions,
    summary="Suggested hold-out train/test windows for a drug",
)
def holdout_date_suggestions(
    drug_code: Annotated[str, Query(min_length=1)],
    db: Annotated[Session, Depends(get_db)],
) -> HoldoutDateSuggestions:
    code = drug_code.strip()
    data_start, data_end = get_receipt_date_bounds(db, code)
    if data_start is None or data_end is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No receipt history found for drug {code}.",
        )
    return compute_holdout_date_suggestions(data_start, data_end)


@router.post(
    "/holdout",
    response_model=HoldoutResponse,
    summary="Hold-out validation — compare forecasts to actual demand",
    response_description=(
        "Trains on history through train_end, forecasts the test window, "
        "and returns actual vs predicted time series with accuracy metrics."
    ),
)
def holdout_validation(
    request: HoldoutValidationRequest,
    db: Annotated[Session, Depends(get_db)],
) -> HoldoutResponse:
    try:
        return run_holdout_validation(
            drug_code=request.drug_code,
            db_session=db,
            train_end=request.train_end,
            test_start=request.test_start,
            test_end=request.test_end,
            models=request.models,
            center_syn_id=request.center_syn_id,
            train_start=request.train_start,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
