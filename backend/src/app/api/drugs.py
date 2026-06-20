"""Unique drugs registry — list and search drugs derived from receipts."""

from __future__ import annotations

from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.schemas.drug import DrugItem, PaginatedDrugListResponse
from app.services.drug_registry import search_drugs

router = APIRouter(prefix="/drugs", tags=["drugs"])

DEFAULT_PAGE_SIZE = 20


def _paginate(total: int, page: int, page_size: int) -> int:
    if total <= 0:
        return 0
    return (total + page_size - 1) // page_size


@router.get(
    "",
    response_model=PaginatedDrugListResponse,
    summary="List unique drugs from receipt ingestion",
)
def list_drugs(
    db: Annotated[Session, Depends(get_db)],
    q: Annotated[Optional[str], Query(min_length=0)] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = DEFAULT_PAGE_SIZE,
) -> PaginatedDrugListResponse:
    rows, total = search_drugs(db, q, page=page, page_size=page_size)
    return PaginatedDrugListResponse(
        items=[DrugItem.model_validate(row) for row in rows],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=_paginate(total, page, page_size),
    )
