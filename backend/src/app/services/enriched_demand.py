"""Query helpers for the enriched hospital daily demand panel."""

from __future__ import annotations

from datetime import date
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.hospital_daily_demand_enriched import HospitalDailyDemandEnriched


def enriched_data_exists(db_session: Session) -> bool:
    return (
        db_session.query(func.count(HospitalDailyDemandEnriched.id)).scalar() or 0
    ) > 0


def get_distinct_drug_codes_from_enriched(
    db_session: Session,
    *,
    min_distinct_days: int = 1,
) -> list[str]:
    query = (
        db_session.query(HospitalDailyDemandEnriched.drug_code)
        .group_by(HospitalDailyDemandEnriched.drug_code)
        .having(
            func.count(func.distinct(HospitalDailyDemandEnriched.demand_date)) >= min_distinct_days,
        )
        .order_by(HospitalDailyDemandEnriched.drug_code.asc())
    )
    return [row[0] for row in query.all()]


def get_enriched_date_bounds(
    db_session: Session,
    drug_code: Optional[str] = None,
) -> tuple[Optional[date], Optional[date]]:
    query = db_session.query(
        func.min(HospitalDailyDemandEnriched.demand_date),
        func.max(HospitalDailyDemandEnriched.demand_date),
    )
    if drug_code:
        query = query.filter(HospitalDailyDemandEnriched.drug_code == drug_code)
    start, end = query.one()
    return start, end


def drug_has_enriched_history(db_session: Session, drug_code: str) -> bool:
    count = (
        db_session.query(func.count(HospitalDailyDemandEnriched.id))
        .filter(HospitalDailyDemandEnriched.drug_code == drug_code)
        .scalar()
    )
    return int(count or 0) > 0


def search_enriched_drug_codes(
    db_session: Session,
    query_text: str,
    *,
    page: int = 1,
    page_size: int = 10,
) -> tuple[list[dict[str, str | None]], int]:
    pattern = f"%{query_text.strip()}%"
    grouped = (
        db_session.query(
            HospitalDailyDemandEnriched.drug_code,
            func.max(HospitalDailyDemandEnriched.article).label("drug_name"),
        )
        .filter(
            HospitalDailyDemandEnriched.drug_code.ilike(pattern)
            | HospitalDailyDemandEnriched.article.ilike(pattern),
        )
        .group_by(HospitalDailyDemandEnriched.drug_code)
    )
    total = grouped.count()
    rows = (
        grouped.order_by(HospitalDailyDemandEnriched.drug_code.asc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    items = [{"drug_code": row.drug_code, "drug_name": row.drug_name} for row in rows]
    return items, total
