"""Pydantic schemas for Cold Start API input and output."""

from __future__ import annotations

from datetime import date
from typing import List, Optional

from pydantic import BaseModel, Field


class DrugMetadataInput(BaseModel):
    """Metadata fields required to embed a drug (maps to drug_catalog columns)."""

    drug_code: str
    drug_name: str
    therapeutic_class: str
    atc_category: str
    pharmaceutical_form: str
    ven_class: str
    abc_class: str
    unit_price_tier: int = Field(..., ge=1, le=5)
    requires_refrigeration: bool
    is_controlled_substance: bool
    average_shelf_life_days: int = Field(..., gt=0)
    route_of_administration: str


class PharmacistEstimate(BaseModel):
    weekly_units: float = Field(..., gt=0, description="Pharmacist's manual weekly demand estimate")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ColdStartPredictRequest(BaseModel):
    drug_metadata: DrugMetadataInput
    pharmacist_estimate: Optional[PharmacistEstimate] = None
    forecast_horizon_days: int = Field(default=7, ge=1, le=30)


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
