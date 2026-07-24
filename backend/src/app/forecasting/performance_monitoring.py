"""Query and assemble model health metrics for monitoring (Phase 3)."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.forecasting.drift_detection import assess_metric_drift
from app.forecasting.schemas import ModelPerformanceRow, PerformanceMonitoringResponse
from app.models.forecast_result import ForecastResult
from app.models.model_performance import ModelPerformance


def _latest_weight_snapshot(
    db_session: Session,
    drug_code: str,
) -> Optional[tuple[float, float, float, datetime]]:
    row = (
        db_session.query(ForecastResult)
        .filter(ForecastResult.drug_code == drug_code)
        .order_by(desc(ForecastResult.generated_at))
        .first()
    )
    if row is None:
        return None
    return (
        float(row.model_weight_sarima),
        float(row.model_weight_lgbm),
        float(row.model_weight_classical),
        row.generated_at,
    )


def _previous_run_metrics(
    db_session: Session,
    drug_code: str,
    model_name: str,
    current_run_id: Optional[str],
) -> Optional[ModelPerformance]:
    query = (
        db_session.query(ModelPerformance)
        .filter(
            ModelPerformance.drug_code == drug_code,
            ModelPerformance.model_name == model_name,
        )
        .order_by(desc(ModelPerformance.evaluated_at))
    )
    rows = query.limit(5).all()
    for row in rows:
        if current_run_id and row.training_run_id == current_run_id:
            continue
        return row
    return None


def get_performance_monitoring(
    db_session: Session,
    *,
    drug_code: Optional[str] = None,
    model_name: Optional[str] = None,
    training_run_id: Optional[str] = None,
    limit: int = 100,
) -> PerformanceMonitoringResponse:
    """Return recent model_performance rows with drift vs prior run."""
    query = db_session.query(ModelPerformance).order_by(desc(ModelPerformance.evaluated_at))
    if drug_code:
        query = query.filter(ModelPerformance.drug_code == drug_code)
    if model_name:
        query = query.filter(ModelPerformance.model_name == model_name)
    if training_run_id:
        query = query.filter(ModelPerformance.training_run_id == training_run_id)

    rows = query.limit(limit).all()
    items: list[ModelPerformanceRow] = []

    for row in rows:
        prev = _previous_run_metrics(
            db_session,
            row.drug_code,
            row.model_name,
            row.training_run_id,
        )
        drift = assess_metric_drift(
            float(prev.smape) if prev else None,
            float(row.smape),
            float(prev.mase) if prev and prev.mase is not None else None,
            float(row.mase) if row.mase is not None else None,
        )
        weights = _latest_weight_snapshot(db_session, row.drug_code)

        items.append(
            ModelPerformanceRow(
                drug_code=row.drug_code,
                model_name=row.model_name,
                smape=float(row.smape),
                smape_normal_supply=(
                    float(row.smape_normal_supply)
                    if row.smape_normal_supply is not None
                    else None
                ),
                mase=float(row.mase) if row.mase is not None else None,
                mase_normal_supply=(
                    float(row.mase_normal_supply)
                    if row.mase_normal_supply is not None
                    else None
                ),
                coverage_90=float(row.coverage_90),
                demand_segment=row.demand_segment,
                data_quality_status=row.data_quality_status,
                training_run_id=row.training_run_id,
                weight_sarima=weights[0] if weights else None,
                weight_lgbm=weights[1] if weights else None,
                weight_classical=weights[2] if weights else None,
                weights_as_of=weights[3] if weights else None,
                smape_drift_pct=drift.smape_delta_pct,
                mase_drift_pct=drift.mase_delta_pct,
                drift_detected=drift.has_drift,
                evaluated_at=row.evaluated_at,
            )
        )

    return PerformanceMonitoringResponse(items=items, total=len(items))
