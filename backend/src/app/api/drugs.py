"""Unique drugs registry — list and search drugs derived from receipts."""

from __future__ import annotations

from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.schemas.drug import DailyDemandPoint, DrugDetailResponse, DrugItem, PaginatedDrugListResponse
from app.services.drug_registry import get_drug_detail, search_drugs

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
        items=[
            DrugItem(
                id=row["drug"].id,
                drug_code=row["drug"].drug_code,
                drug_name=row["drug"].drug_name,
                drug_category=row["drug"].drug_category,
                receipt_count=row["drug"].receipt_count,
                distinct_receipt_days=row["distinct_receipt_days"],
                first_receipt_date=row["first_receipt_date"],
                last_receipt_date=row["last_receipt_date"],
                created_at=row["drug"].created_at,
                updated_at=row["drug"].updated_at,
            )
            for row in rows
        ],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=_paginate(total, page, page_size),
    )


@router.get(
    "/{drug_code}",
    response_model=DrugDetailResponse,
    summary="Drug detail with receipt stats and demand history",
)
def get_drug(
    drug_code: str,
    db: Annotated[Session, Depends(get_db)],
    lookback_days: Annotated[int, Query(ge=7, le=3650)] = 365,
) -> DrugDetailResponse:
    detail = get_drug_detail(db, drug_code, lookback_days=lookback_days)
    if detail is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Drug {drug_code!r} not found in registry.",
        )

    drug = detail["drug"]
    return DrugDetailResponse(
        id=drug.id,
        drug_code=drug.drug_code,
        drug_name=drug.drug_name,
        drug_category=drug.drug_category,
        receipt_count=drug.receipt_count,
        created_at=drug.created_at,
        updated_at=drug.updated_at,
        total_quantity=detail["total_quantity"],
        distinct_receipt_days=detail["distinct_receipt_days"],
        first_receipt_date=detail["first_receipt_date"],
        last_receipt_date=detail["last_receipt_date"],
        center_count=detail["center_count"],
        avg_daily_quantity=detail["avg_daily_quantity"],
        observation_count=detail["observation_count"],
        graduation_stage=detail["graduation_stage"],
        lookback_days=detail["lookback_days"],
        demand_series=[DailyDemandPoint.model_validate(row) for row in detail["demand_series"]],
    )
