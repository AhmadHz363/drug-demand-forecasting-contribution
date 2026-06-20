"""Lookup helpers for drugs referenced in forecast results."""

from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from app.models.forecast_result import ForecastResult


def search_forecasted_drug_codes(
    db_session: Session,
    query: Optional[str],
    *,
    page: int = 1,
    page_size: int = 10,
) -> tuple[list[str], int]:
    """Paginated distinct drug codes that appear in ``forecast_results``."""
    normalized = (query or "").strip()
    grouped = db_session.query(ForecastResult.drug_code).group_by(ForecastResult.drug_code)
    if normalized:
        grouped = grouped.filter(ForecastResult.drug_code.ilike(f"%{normalized}%"))

    total = grouped.count()
    rows = (
        grouped.order_by(ForecastResult.drug_code.asc())
        .offset(max(page - 1, 0) * page_size)
        .limit(page_size)
        .all()
    )
    return [row[0] for row in rows], total
