"""Schemas for the unique categories registry API."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class CategoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    category_code: str
    name: Optional[str] = None
    receipt_count: int
    drug_count: int
    created_at: datetime
    updated_at: datetime


class PaginatedCategoryListResponse(BaseModel):
    items: list[CategoryItem]
    total: int
    page: int
    page_size: int
    total_pages: int
