"""Pydantic schemas for the system overview endpoint."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel


class DrugSummary(BaseModel):
    drug_code: str
    drug_name: Optional[str] = None
    drug_category: Optional[str] = None
    receipt_count: int


class TimeSeriesPoint(BaseModel):
    """Monthly receipt volume for trend charts."""

    period: str
    receipt_count: int
    total_quantity: float


class CenterSummary(BaseModel):
    center_syn_id: str
    receipt_count: int


class ModelPerformanceSummary(BaseModel):
    model_name: str
    avg_smape: float
    avg_coverage_90: float
    record_count: int


class CategorySummary(BaseModel):
    category_code: str
    name: Optional[str] = None
    drug_count: int
    receipt_count: int


class GraduationDistribution(BaseModel):
    """How many drugs fall into each cold-start graduation tier."""

    cold_start_only: int
    blended: int
    full_ensemble: int


class IngestionKPIs(BaseModel):
    total_receipt_rows: int
    distinct_drug_codes: int
    distinct_centers: int
    first_receipt_date: Optional[date] = None
    last_receipt_date: Optional[date] = None
    total_quantity: float
    receipt_trend: list[TimeSeriesPoint]
    top_centers: list[CenterSummary]


class DrugRegistryKPIs(BaseModel):
    total_drugs: int
    graduation_distribution: GraduationDistribution
    top_drugs: list[DrugSummary]


class CategoryRegistryKPIs(BaseModel):
    total_categories: int
    top_categories: list[CategorySummary]


class ColdStartKPIs(BaseModel):
    embedder_trained: bool
    maml_trained: bool


class ForecastingKPIs(BaseModel):
    total_forecasted_drugs: int
    avg_smape: Optional[float] = None
    avg_coverage_90: Optional[float] = None
    models_evaluated: list[str]
    latest_training_run_id: Optional[str] = None
    total_performance_records: int
    model_performance: list[ModelPerformanceSummary]


class OverviewResponse(BaseModel):
    ingestion: IngestionKPIs
    drug_registry: DrugRegistryKPIs
    category_registry: CategoryRegistryKPIs
    cold_start: ColdStartKPIs
    forecasting: ForecastingKPIs
    generated_at: datetime
