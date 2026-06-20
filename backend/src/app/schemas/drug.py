"""Schemas for the unique drugs registry API."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

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
