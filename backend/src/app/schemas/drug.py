"""Schemas for the unique drugs registry API."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict


class DrugItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    drug_code: str
    drug_name: Optional[str] = None
    drug_category: Optional[str] = None
    receipt_count: int
    created_at: datetime
    updated_at: datetime


class PaginatedDrugListResponse(BaseModel):
    items: list[DrugItem]
    total: int
    page: int
    page_size: int
    total_pages: int


class DailyDemandPoint(BaseModel):
    date: date
    quantity: float


GraduationStageLiteral = Literal["cold_start_only", "blended", "full_ensemble"]


class DrugDetailResponse(BaseModel):
    id: int
    drug_code: str
    drug_name: Optional[str] = None
    drug_category: Optional[str] = None
    receipt_count: int
    created_at: datetime
    updated_at: datetime
    total_quantity: float
    distinct_receipt_days: int
    first_receipt_date: Optional[date] = None
    last_receipt_date: Optional[date] = None
    center_count: int
    avg_daily_quantity: Optional[float] = None
    observation_count: int
    graduation_stage: GraduationStageLiteral
    lookback_days: int
    demand_series: list[DailyDemandPoint]
