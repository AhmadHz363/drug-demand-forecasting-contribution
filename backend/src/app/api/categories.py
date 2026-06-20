"""Unique categories registry — list and search categories derived from receipts."""

from __future__ import annotations

from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.schemas.category import CategoryItem, PaginatedCategoryListResponse
from app.services.category_registry import search_categories

router = APIRouter(prefix="/categories", tags=["categories"])

DEFAULT_PAGE_SIZE = 20


def _paginate(total: int, page: int, page_size: int) -> int:
    if total <= 0:
        return 0
    return (total + page_size - 1) // page_size


@router.get(
    "",
    response_model=PaginatedCategoryListResponse,
    summary="List unique categories from receipt ingestion",
)
def list_categories(
    db: Annotated[Session, Depends(get_db)],
    q: Annotated[Optional[str], Query(min_length=0)] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = DEFAULT_PAGE_SIZE,
) -> PaginatedCategoryListResponse:
    rows, total = search_categories(db, q, page=page, page_size=page_size)
    return PaginatedCategoryListResponse(
        items=[CategoryItem.model_validate(row) for row in rows],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=_paginate(total, page, page_size),
    )
