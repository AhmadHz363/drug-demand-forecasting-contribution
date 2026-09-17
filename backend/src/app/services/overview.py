"""Aggregate KPIs for the system overview endpoint."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from app.cold_start.constants import (
    ARTIFACTS_DIR,
    BLEND_UNTIL,
    COLD_START_ONLY_BELOW,
)
from app.models.category import Category
from app.models.drug import Drug
from app.models.drug_receipt import DrugReceipt
from app.models.forecast_result import ForecastResult
from app.models.model_performance import ModelPerformance
from app.schemas.overview import (
    CategoryRegistryKPIs,
    CategorySummary,
    CenterSummary,
    ColdStartKPIs,
    DrugRegistryKPIs,
    DrugSummary,
    ForecastingKPIs,
    GraduationDistribution,
    IngestionKPIs,
    ModelPerformanceSummary,
    OverviewResponse,
    TimeSeriesPoint,
)

_AUTOENCODER_ARTIFACT = os.path.join(ARTIFACTS_DIR, "autoencoder.pt")
_MAML_ARTIFACT = os.path.join(ARTIFACTS_DIR, "maml_base.pt")

TOP_N = 5
TREND_MONTHS = 18
TOP_CENTERS = 8


def _ingestion_kpis(db: Session) -> IngestionKPIs:
    row = db.query(
        func.count(DrugReceipt.id),
        func.count(func.distinct(DrugReceipt.drug_code)),
        func.count(func.distinct(DrugReceipt.center_syn_id)),
        func.min(DrugReceipt.receipt_date),
        func.max(DrugReceipt.receipt_date),
        func.coalesce(func.sum(DrugReceipt.quantity), 0.0),
    ).one()

    month_bucket = func.date_trunc("month", DrugReceipt.receipt_date)
    trend_rows = (
        db.query(
            month_bucket.label("month"),
            func.count(DrugReceipt.id).label("receipt_count"),
            func.coalesce(func.sum(DrugReceipt.quantity), 0.0).label("total_quantity"),
        )
        .group_by(month_bucket)
        .order_by(month_bucket.desc())
        .limit(TREND_MONTHS)
        .all()
    )
    trend_rows = list(reversed(trend_rows))
    receipt_trend = [
        TimeSeriesPoint(
            period=month.strftime("%Y-%m") if month else "",
            receipt_count=int(count or 0),
            total_quantity=float(qty or 0.0),
        )
        for month, count, qty in trend_rows
        if month is not None
    ]

    center_rows = (
        db.query(
            DrugReceipt.center_syn_id,
            func.count(DrugReceipt.id).label("receipt_count"),
        )
        .filter(DrugReceipt.center_syn_id.isnot(None))
        .group_by(DrugReceipt.center_syn_id)
        .order_by(desc("receipt_count"))
        .limit(TOP_CENTERS)
        .all()
    )
    top_centers = [
        CenterSummary(
            center_syn_id=str(center_id),
            receipt_count=int(count or 0),
        )
        for center_id, count in center_rows
        if center_id
    ]

    return IngestionKPIs(
        total_receipt_rows=int(row[0] or 0),
        distinct_drug_codes=int(row[1] or 0),
        distinct_centers=int(row[2] or 0),
        first_receipt_date=row[3],
        last_receipt_date=row[4],
        total_quantity=float(row[5] or 0.0),
        receipt_trend=receipt_trend,
        top_centers=top_centers,
    )


def _drug_registry_kpis(db: Session) -> DrugRegistryKPIs:
    total_drugs = db.query(func.count(Drug.id)).scalar() or 0

    # Graduation distribution: count distinct receipt days per drug_code
    obs_rows = (
        db.query(
            DrugReceipt.drug_code,
            func.count(func.distinct(DrugReceipt.receipt_date)).label("obs_count"),
        )
        .group_by(DrugReceipt.drug_code)
        .all()
    )
    obs_by_code = {row.drug_code: int(row.obs_count) for row in obs_rows}

    cold_start_only = 0
    blended = 0
    full_ensemble = 0

    # Every known drug is counted; drugs with no receipts default to cold_start_only
    all_codes_query = db.query(Drug.drug_code).all()
    for (code,) in all_codes_query:
        obs = obs_by_code.get(code, 0)
        if obs < COLD_START_ONLY_BELOW:
            cold_start_only += 1
        elif obs < BLEND_UNTIL:
            blended += 1
        else:
            full_ensemble += 1

    top_drug_rows = (
        db.query(Drug)
        .order_by(desc(Drug.receipt_count))
        .limit(TOP_N)
        .all()
    )
    top_drugs = [
        DrugSummary(
            drug_code=d.drug_code,
            drug_name=d.drug_name,
            drug_category=d.drug_category,
            receipt_count=d.receipt_count,
        )
        for d in top_drug_rows
    ]

    return DrugRegistryKPIs(
        total_drugs=total_drugs,
        graduation_distribution=GraduationDistribution(
            cold_start_only=cold_start_only,
            blended=blended,
            full_ensemble=full_ensemble,
        ),
        top_drugs=top_drugs,
    )


def _category_registry_kpis(db: Session) -> CategoryRegistryKPIs:
    total_categories = db.query(func.count(Category.id)).scalar() or 0

    drug_count_sub = (
        db.query(
            DrugReceipt.category_id,
            func.count(func.distinct(DrugReceipt.drug_id)).label("drug_count"),
        )
        .filter(DrugReceipt.category_id.isnot(None))
        .group_by(DrugReceipt.category_id)
        .subquery()
    )

    top_cat_rows = (
        db.query(
            Category.category_code,
            Category.name,
            func.coalesce(drug_count_sub.c.drug_count, 0).label("drug_count"),
            Category.receipt_count,
        )
        .outerjoin(drug_count_sub, Category.id == drug_count_sub.c.category_id)
        .order_by(func.coalesce(drug_count_sub.c.drug_count, 0).desc())
        .limit(TOP_N)
        .all()
    )
    top_categories = [
        CategorySummary(
            category_code=row.category_code,
            name=row.name,
            drug_count=int(row.drug_count),
            receipt_count=int(row.receipt_count),
        )
        for row in top_cat_rows
    ]

    return CategoryRegistryKPIs(
        total_categories=total_categories,
        top_categories=top_categories,
    )


def _cold_start_kpis() -> ColdStartKPIs:
    return ColdStartKPIs(
        embedder_trained=os.path.isfile(_AUTOENCODER_ARTIFACT),
        maml_trained=os.path.isfile(_MAML_ARTIFACT),
    )


def _forecasting_kpis(db: Session) -> ForecastingKPIs:
    total_forecasted = (
        db.query(func.count(func.distinct(ForecastResult.drug_code))).scalar() or 0
    )

    perf_stats = (
        db.query(
            func.avg(ModelPerformance.smape).label("avg_smape"),
            func.avg(ModelPerformance.coverage_90).label("avg_coverage_90"),
            func.count(ModelPerformance.id).label("total"),
        )
        .one()
    )

    models_evaluated: list[str] = [
        row[0]
        for row in db.query(ModelPerformance.model_name)
        .distinct()
        .order_by(ModelPerformance.model_name.asc())
        .all()
    ]

    latest_run: Optional[str] = None
    latest_row = (
        db.query(ModelPerformance.training_run_id)
        .filter(ModelPerformance.training_run_id.isnot(None))
        .order_by(desc(ModelPerformance.evaluated_at))
        .first()
    )
    if latest_row:
        latest_run = latest_row[0]

    perf_by_model = (
        db.query(
            ModelPerformance.model_name,
            func.avg(ModelPerformance.smape).label("avg_smape"),
            func.avg(ModelPerformance.coverage_90).label("avg_coverage_90"),
            func.count(ModelPerformance.id).label("record_count"),
        )
        .group_by(ModelPerformance.model_name)
        .order_by(ModelPerformance.model_name.asc())
        .all()
    )
    model_performance = [
        ModelPerformanceSummary(
            model_name=row.model_name,
            avg_smape=round(float(row.avg_smape or 0.0), 4),
            avg_coverage_90=round(float(row.avg_coverage_90 or 0.0), 4),
            record_count=int(row.record_count or 0),
        )
        for row in perf_by_model
    ]

    return ForecastingKPIs(
        total_forecasted_drugs=int(total_forecasted),
        avg_smape=round(float(perf_stats.avg_smape), 4) if perf_stats.avg_smape else None,
        avg_coverage_90=(
            round(float(perf_stats.avg_coverage_90), 4)
            if perf_stats.avg_coverage_90
            else None
        ),
        models_evaluated=models_evaluated,
        latest_training_run_id=latest_run,
        total_performance_records=int(perf_stats.total or 0),
        model_performance=model_performance,
    )


def get_overview(db: Session) -> OverviewResponse:
    """Compile all KPIs for the system overview in a single DB session."""
    return OverviewResponse(
        ingestion=_ingestion_kpis(db),
        drug_registry=_drug_registry_kpis(db),
        category_registry=_category_registry_kpis(db),
        cold_start=_cold_start_kpis(),
        forecasting=_forecasting_kpis(db),
        generated_at=datetime.now(tz=timezone.utc),
    )
