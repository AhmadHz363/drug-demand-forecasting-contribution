"""Pydantic schemas for Cold Start API input and output."""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class DrugMetadataInput(BaseModel):
    """CAMEO attribute vector — mirrors notebook Section 6 fields."""

    drug_code: str
    drug_name: str
    generic_name: Optional[str] = None
    drug_class: str
    dosage_form: str
    strength: str
    route_of_administration: str
    pregnancy_category: str = "MISSING"
    availability: str = "Prescription"
    indications: str = ""
    side_effects: str = ""
    contraindications: str = ""


class PharmacistEstimate(BaseModel):
    weekly_units: float = Field(..., gt=0, description="Pharmacist's manual weekly demand estimate")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ColdStartPredictRequest(BaseModel):
    drug_metadata: DrugMetadataInput
    pharmacist_estimate: Optional[PharmacistEstimate] = None
    forecast_horizon_days: int = Field(default=7, ge=1, le=140)


class DailyForecast(BaseModel):
    date: date
    p10: float
    p50: float
    p90: float


class ColdStartPredictResponse(BaseModel):
    drug_code: str
    stage_used: str
    observation_count: int
    embedding: List[float]
    nearest_neighbours: List[str]
    similarity_scores: List[float]
    forecast: List[DailyForecast]
    uncertainty_note: str
    conformal_half_width_weekly: Optional[float] = None
    drift_alarms: int = 0


class CameoPerDrugAccuracy(BaseModel):
    drug_code: str
    sb_class: Optional[str] = None
    accuracy_pct: Optional[float] = None
    wape: Optional[float] = None


class CameoAccuracySummary(BaseModel):
    n_historical_library_drugs: int = 0
    n_coldstart_test_drugs: int = 0
    forecast_horizon_weeks: int = 20
    pooled_accuracy_pct: Dict[str, float] = Field(default_factory=dict)
    median_accuracy_pct: Dict[str, Optional[float]] = Field(default_factory=dict)
    win_rate_pct: Dict[str, float] = Field(default_factory=dict)
    per_sb_class_pooled_accuracy_pct: Dict[str, float] = Field(default_factory=dict)
    empirical_conformal_coverage_pct: Optional[float] = None
    conformal_half_width_weekly: Optional[float] = None
    cameo_vs_analogous_wape_improvement_pct: Optional[float] = None
    smooth_pooled_accuracy_pct: Optional[float] = None
    operational_pooled_accuracy_pct: Optional[float] = None
    non_lumpy_operational_pooled_accuracy_pct: Optional[float] = None
    zero_demand_test_drugs: int = 0
    per_drug: List[CameoPerDrugAccuracy] = Field(default_factory=list)
    note: Optional[str] = None


class ColdStartStatusResponse(BaseModel):
    model_ready: bool
    drugs_trained_on: Optional[int] = None
    trained_at: Optional[str] = None
    accuracy_summary: Optional[CameoAccuracySummary] = None
