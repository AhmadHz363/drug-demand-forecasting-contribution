"""Pydantic schemas for Forecasting API input and output."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field

from app.forecasting.constants import FORECAST_HORIZON


class ForecastRequest(BaseModel):
    drug_code: str
    horizon_days: int = Field(default=7, ge=1, le=FORECAST_HORIZON)
    center_syn_id: Optional[str] = None
    include_shap: bool = False
    include_attention: bool = False


class TrainForecastingRequest(BaseModel):
    drug_codes: Optional[list[str]] = None
    models: list[str] = ["sarima", "lgbm", "tft"]
    force_retrain: bool = False


class DailyForecastPoint(BaseModel):
    date: date
    p5: Optional[float] = None
    p10: float
    p50: float
    p90: float
    p95: Optional[float] = None
    recommended_quantity: Optional[float] = Field(
        default=None,
        description="Criticality-aware stocking target (newsvendor operating quantile).",
    )


class HistoryPoint(BaseModel):
    date: date
    quantity: float


class ShapFeature(BaseModel):
    feature_name: str
    shap_value: float
    feature_value: float


class AttentionWeight(BaseModel):
    week_offset: int
    weight: float


class ModelWeightBreakdown(BaseModel):
    sarima: float
    lgbm: float
    tft: float


class InferenceHealth(BaseModel):
    """Paths used during ensemble inference — for monitoring fallback rates."""

    demand_segment: str
    used_stacking: bool
    used_conformal: bool
    used_spread_fallback: bool


class ForecastResponse(BaseModel):
    drug_code: str
    center_syn_id: Optional[str]
    horizon_days: int
    model_weights: ModelWeightBreakdown
    forecast: list[DailyForecastPoint]
    history: list[HistoryPoint] = Field(default_factory=list)
    shap_features: Optional[list[ShapFeature]] = None
    attention_weights: Optional[list[AttentionWeight]] = None
    uncertainty_note: str
    smape_last_validation: Optional[float] = None
    ven_class: Optional[str] = Field(
        default=None,
        description="VEN criticality class (V/E/N) from drug catalog.",
    )
    operating_quantile: Optional[float] = Field(
        default=None,
        description="Newsvendor critical fractile used for stocking recommendations.",
    )
    recommended_quantity_total: Optional[float] = Field(
        default=None,
        description="Sum of per-day recommended quantities over the forecast horizon.",
    )
    inference_health: Optional[InferenceHealth] = None
    error: Optional[str] = None


class BatchForecastRequest(BaseModel):
    drug_codes: list[str] = Field(..., min_length=1, max_length=70)
    horizon_days: int = Field(default=7, ge=1, le=FORECAST_HORIZON)
    include_shap: bool = False


class DrugQualitySummary(BaseModel):
    drug_code: str
    status: str
    reasons: list[str] = Field(default_factory=list)


class TrainStatusResponse(BaseModel):
    status: str
    training_run_id: str = ""
    drugs_trained: int
    models_trained: list[str]
    smape_summary: dict[str, float]
    artifacts_saved: list[str]
    skipped_drugs: list[DrugQualitySummary] = Field(default_factory=list)
    flagged_drugs: list[DrugQualitySummary] = Field(default_factory=list)
    drift_alerts: list[str] = Field(
        default_factory=list,
        description="Drugs whose metrics degraded vs the prior training run.",
    )


class ModelPerformanceRow(BaseModel):
    drug_code: str
    model_name: str
    smape: float
    mase: Optional[float] = None
    coverage_90: float
    demand_segment: Optional[str] = None
    data_quality_status: Optional[str] = None
    training_run_id: Optional[str] = None
    weight_sarima: Optional[float] = None
    weight_lgbm: Optional[float] = None
    weight_tft: Optional[float] = None
    weights_as_of: Optional[datetime] = None
    smape_drift_pct: Optional[float] = None
    mase_drift_pct: Optional[float] = None
    drift_detected: bool = False
    evaluated_at: datetime


class PerformanceMonitoringResponse(BaseModel):
    items: list[ModelPerformanceRow]
    total: int


class ReceiptDrugOption(BaseModel):
    drug_code: str
    drug_name: Optional[str] = None


class PaginatedDrugCodeSearchResponse(BaseModel):
    items: list[str]
    total: int
    page: int
    page_size: int
    total_pages: int


class PaginatedReceiptDrugSearchResponse(BaseModel):
    items: list[ReceiptDrugOption]
    total: int
    page: int
    page_size: int
    total_pages: int


class PeriodRange(BaseModel):
    start: date
    end: date


class HoldoutDateDefaults(BaseModel):
    train_end: date
    test_start: date
    test_end: date


class HoldoutDateSuggestions(BaseModel):
    data_start: date
    data_end: date
    defaults: HoldoutDateDefaults
    train_end_options: list[date]
    test_start_options: list[date]
    test_end_options: list[date]


class HoldoutValidationRequest(BaseModel):
    drug_code: str
    train_end: date = Field(default=date(2025, 12, 31))
    test_start: date = Field(default=date(2026, 1, 1))
    test_end: date = Field(default=date(2026, 3, 31))
    train_start: Optional[date] = None
    center_syn_id: Optional[str] = None
    models: list[str] = ["sarima", "lgbm", "tft"]


class HoldoutMetrics(BaseModel):
    smape: float
    mae: float
    coverage_90: float
    accuracy_pct: float = Field(
        description="Secondary score from sMAPE (0–100%); may distort near-zero demand.",
    )
    mase: float = 0.0
    rmsse: float = 0.0
    pinball_p10: float = 0.0
    pinball_p50: float = 0.0
    pinball_p90: float = 0.0
    accuracy_skill_pct: float = Field(
        default=0.0,
        description="Primary skill score: max(0, (1−MASE)×100) vs weekly seasonal naive.",
    )


class HoldoutSeriesPoint(BaseModel):
    date: date
    actual: float
    sarima_p50: Optional[float] = None
    lgbm_p50: Optional[float] = None
    tft_p50: Optional[float] = None
    ensemble_p50: Optional[float] = None
    ensemble_p10: Optional[float] = None
    ensemble_p90: Optional[float] = None


class HoldoutResponse(BaseModel):
    drug_code: str
    center_syn_id: Optional[str]
    train_period: PeriodRange
    test_period: PeriodRange
    models_evaluated: list[str]
    model_errors: dict[str, str]
    metrics: dict[str, HoldoutMetrics]
    demand_segment: str = Field(
        description="Syntetos–Boylan demand pattern segment for this drug.",
    )
    total_accuracy_pct: float = Field(
        description="Ensemble sMAPE-derived accuracy (0–100%) over the hold-out window.",
    )
    total_accuracy_skill_pct: float = Field(
        description="Ensemble MASE skill score (0–100%) vs weekly seasonal naive.",
    )
    model_weights: ModelWeightBreakdown
    series: list[HoldoutSeriesPoint]
