"""Aggregate pharmacy receipt lines into daily drug demand.

Source of truth for training/inference demand history is ``drug_receipts``.
``daily_drug_demand`` is a materialized aggregate refreshed from receipts.

Daily totals are **net inpatient consumption** (sales − returns − cancel sales).
Inter-department transfers and other ledger movements are excluded.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

from sqlalchemy import case, cast, Float, func, or_
from sqlalchemy.orm import Session

from app.models.daily_drug_demand import DailyDrugDemand
from app.models.drug_receipt import DrugReceipt
from app.services.import_coverage import (
    ensure_default_coverage_from_receipt_bounds,
    get_merged_coverage_periods,
    is_date_covered,
    latest_coverage_segment,
)
from app.services.movement_demand import (
    PATIENT_RETURN_OR_CANCEL_MOVEMENTS,
    PATIENT_SALE_MOVEMENTS,
    clip_daily_demand,
    patient_demand_contribution,
)

logger = logging.getLogger(__name__)


def _movement_number_as_int_expr():
    """Cast varchar MOV# values like ``5.0`` / ``5`` to integers for filtering."""
    return cast(func.nullif(func.trim(DrugReceipt.movement_number), ""), Float)


def _patient_contribution_expr():
    """SQL expression: role-based signed contribution matching ``patient_demand_contribution``."""
    mov = _movement_number_as_int_expr()
    qty = func.coalesce(DrugReceipt.quantity, 0.0)
    abs_qty = func.abs(qty)
    return case(
        (mov.in_(sorted(PATIENT_SALE_MOVEMENTS)), abs_qty),
        (mov.in_(sorted(PATIENT_RETURN_OR_CANCEL_MOVEMENTS)), -abs_qty),
        (DrugReceipt.movement_number.is_(None), qty),  # legacy rows without MOV#
        (func.trim(DrugReceipt.movement_number) == "", qty),
        else_=0.0,
    )


def _aggregate_query(
    db_session: Session,
    *,
    drug_codes: Optional[list[str]] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    center_syn_id: Optional[str] = None,
):
    contrib = _patient_contribution_expr()
    query = db_session.query(
        DrugReceipt.drug_code,
        DrugReceipt.receipt_date.label("demand_date"),
        func.coalesce(func.sum(contrib), 0.0).label("total_quantity"),
    )
    if drug_codes:
        query = query.filter(DrugReceipt.drug_code.in_(drug_codes))
    if start_date is not None:
        query = query.filter(DrugReceipt.receipt_date >= start_date)
    if end_date is not None:
        query = query.filter(DrugReceipt.receipt_date <= end_date)
    if center_syn_id is not None:
        query = query.filter(DrugReceipt.center_syn_id == center_syn_id)
    return query.group_by(DrugReceipt.drug_code, DrugReceipt.receipt_date)


def aggregate_daily_demand_rows(
    db_session: Session,
    drug_code: str,
    start_date: date,
    end_date: date,
    center_syn_id: Optional[str] = None,
) -> list[tuple[date, float]]:
    """Return daily inpatient demand totals for one drug from ``drug_receipts``."""
    rows = (
        _aggregate_query(
            db_session,
            drug_codes=[drug_code],
            start_date=start_date,
            end_date=end_date,
            center_syn_id=center_syn_id,
        )
        .order_by(DrugReceipt.receipt_date.asc())
        .all()
    )
    return [
        (row.demand_date, clip_daily_demand(float(row.total_quantity)))
        for row in rows
    ]


def get_receipt_date_bounds(
    db_session: Session,
    drug_code: str,
    center_syn_id: Optional[str] = None,
) -> tuple[Optional[date], Optional[date]]:
    """Min/max ``receipt_date`` for a drug in ``drug_receipts``."""
    query = db_session.query(
        func.min(DrugReceipt.receipt_date),
        func.max(DrugReceipt.receipt_date),
    ).filter(DrugReceipt.drug_code == drug_code)
    if center_syn_id is not None:
        query = query.filter(DrugReceipt.center_syn_id == center_syn_id)
    start_date, end_date = query.one()
    return start_date, end_date


def get_global_receipt_date_bounds(
    db_session: Session,
) -> tuple[Optional[date], Optional[date], int]:
    """Hospital-wide receipt min/max dates and row count."""
    min_date, max_date, row_count = db_session.query(
        func.min(DrugReceipt.receipt_date),
        func.max(DrugReceipt.receipt_date),
        func.count(DrugReceipt.id),
    ).one()
    return min_date, max_date, int(row_count or 0)


def get_import_coverage_periods(db_session: Session) -> list[tuple[date, date]]:
    """Hospital/file coverage intervals (not per-drug sparse activity)."""
    min_date, max_date, row_count = get_global_receipt_date_bounds(db_session)
    ensure_default_coverage_from_receipt_bounds(
        db_session,
        min_date=min_date,
        max_date=max_date,
        row_count=row_count,
    )
    return get_merged_coverage_periods(db_session)


def get_latest_coverage_segment(db_session: Session) -> Optional[tuple[date, date]]:
    periods = get_import_coverage_periods(db_session)
    if not periods:
        return latest_coverage_segment(db_session)
    return periods[-1]


def resolve_covered_history_range(
    db_session: Session,
    drug_code: str,
    center_syn_id: Optional[str] = None,
    *,
    prefer_latest_segment: bool = False,
) -> tuple[Optional[date], Optional[date]]:
    """
    Resolve the feature-engineering date range for a drug.

    When ``prefer_latest_segment`` is True (SARIMA/classical), clip to the
    latest contiguous hospital coverage segment intersected with the drug's
    receipt bounds. Otherwise return the full drug receipt span (LightGBM may
    use all covered segments with gap resets).
    """
    drug_start, drug_end = get_receipt_date_bounds(db_session, drug_code, center_syn_id)
    if drug_start is None or drug_end is None:
        return None, None
    if not prefer_latest_segment:
        return drug_start, drug_end
    segment = get_latest_coverage_segment(db_session)
    if segment is None:
        return drug_start, drug_end
    seg_start, seg_end = segment
    start = max(drug_start, seg_start)
    end = min(drug_end, seg_end)
    if end < start:
        return drug_start, drug_end
    return start, end


def date_is_covered(db_session: Session, day: date) -> bool:
    return is_date_covered(day, get_import_coverage_periods(db_session))


def count_distinct_receipt_days(
    db_session: Session,
    drug_code: str,
    center_syn_id: Optional[str] = None,
) -> int:
    """Number of distinct ``receipt_date`` values for a drug in ``drug_receipts``."""
    query = db_session.query(func.count(func.distinct(DrugReceipt.receipt_date))).filter(
        DrugReceipt.drug_code == drug_code
    )
    if center_syn_id is not None:
        query = query.filter(DrugReceipt.center_syn_id == center_syn_id)
    return int(query.scalar() or 0)


def drug_has_receipt_history(
    db_session: Session,
    drug_code: str,
    center_syn_id: Optional[str] = None,
) -> bool:
    """Return True when the drug has at least one row in ``drug_receipts``."""
    return count_distinct_receipt_days(db_session, drug_code, center_syn_id) > 0


def get_distinct_drug_codes_from_receipts(
    db_session: Session,
    *,
    min_distinct_days: Optional[int] = None,
) -> list[str]:
    """Distinct ``drug_code`` values present in ``drug_receipts``."""
    query = db_session.query(DrugReceipt.drug_code)
    if min_distinct_days is not None:
        query = (
            query.group_by(DrugReceipt.drug_code)
            .having(func.count(func.distinct(DrugReceipt.receipt_date)) >= min_distinct_days)
        )
    else:
        query = query.distinct()
    rows = query.order_by(DrugReceipt.drug_code.asc()).all()
    return [row[0] for row in rows]


def search_receipt_drug_codes(
    db_session: Session,
    query: str,
    *,
    page: int = 1,
    page_size: int = 10,
) -> tuple[list[dict[str, str | None]], int]:
    """Search distinct drugs in ``drug_receipts`` by code or name."""
    normalized = query.strip()
    pattern = f"%{normalized}%"
    grouped = (
        db_session.query(
            DrugReceipt.drug_code,
            func.max(DrugReceipt.drug_name).label("drug_name"),
        )
        .filter(
            or_(
                DrugReceipt.drug_code.ilike(pattern),
                DrugReceipt.drug_name.ilike(pattern),
            )
        )
        .group_by(DrugReceipt.drug_code)
    )

    total = grouped.count()
    rows = (
        grouped.order_by(DrugReceipt.drug_code.asc())
        .offset(max(page - 1, 0) * page_size)
        .limit(page_size)
        .all()
    )
    return [
        {"drug_code": row.drug_code, "drug_name": row.drug_name}
        for row in rows
    ], total


def load_demand_series_from_receipts(
    db_session: Session,
    drug_code: str,
    *,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    center_syn_id: Optional[str] = None,
) -> list[tuple[date, float]]:
    """Daily ``(date, quantity)`` pairs aggregated from ``drug_receipts``."""
    bounds_start, bounds_end = get_receipt_date_bounds(
        db_session,
        drug_code,
        center_syn_id,
    )
    if bounds_start is None or bounds_end is None:
        return []

    effective_start = start_date or bounds_start
    effective_end = end_date or bounds_end
    if effective_end < effective_start:
        return []

    return aggregate_daily_demand_rows(
        db_session,
        drug_code,
        effective_start,
        effective_end,
        center_syn_id=center_syn_id,
    )


def load_recent_quantities_from_receipts(
    db_session: Session,
    drug_code: str,
    *,
    lookback_days: int = 90,
    center_syn_id: Optional[str] = None,
) -> list[float]:
    """Recent daily quantities for a drug, newest first, from ``drug_receipts``."""
    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days - 1)
    rows = aggregate_daily_demand_rows(
        db_session,
        drug_code,
        start_date,
        end_date,
        center_syn_id=center_syn_id,
    )
    quantities = [qty for _, qty in reversed(rows)]
    return quantities[:lookback_days]


def aggregate_receipt_rows_in_memory(
    rows: list[dict],
) -> dict[tuple[str, date], float]:
    """Aggregate receipt dicts to ``(drug_code, date) → clipped patient demand``."""
    totals: dict[tuple[str, date], float] = {}
    for row in rows:
        code = str(row["drug_code"]).strip()
        demand_date = row["receipt_date"]
        contrib = patient_demand_contribution(row.get("movement_number"), row.get("quantity"))
        key = (code, demand_date)
        totals[key] = totals.get(key, 0.0) + contrib
    return {key: clip_daily_demand(value) for key, value in totals.items()}


def sync_daily_demand_from_receipts(
    db_session: Session,
    *,
    drug_codes: Optional[list[str]] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> int:
    """
    Refresh ``daily_drug_demand`` rows from ``drug_receipts``.

    Aggregates across all centers (``center_syn_id`` is not stored on
    ``daily_drug_demand``). For center-specific views, query receipts directly.
    """
    aggregates = _aggregate_query(
        db_session,
        drug_codes=drug_codes,
        start_date=start_date,
        end_date=end_date,
        center_syn_id=None,
    ).all()

    codes_to_refresh = drug_codes or sorted({row.drug_code for row in aggregates})
    if not codes_to_refresh:
        logger.info("No receipt rows found to sync into daily_drug_demand")
        return 0

    delete_query = db_session.query(DailyDrugDemand).filter(
        DailyDrugDemand.drug_code.in_(codes_to_refresh)
    )
    if start_date is not None:
        delete_query = delete_query.filter(DailyDrugDemand.demand_date >= start_date)
    if end_date is not None:
        delete_query = delete_query.filter(DailyDrugDemand.demand_date <= end_date)
    delete_query.delete(synchronize_session=False)

    payload = [
        {
            "drug_code": row.drug_code,
            "demand_date": row.demand_date,
            "total_quantity": clip_daily_demand(float(row.total_quantity)),
        }
        for row in aggregates
    ]
    if payload:
        db_session.bulk_insert_mappings(DailyDrugDemand, payload)
    db_session.flush()
    logger.info(
        "Synced %d daily_drug_demand rows from drug_receipts for %d drug(s)",
        len(payload),
        len(codes_to_refresh),
    )
    return len(payload)


def ensure_daily_demand_materialized(
    db_session: Session,
    drug_code: str,
    center_syn_id: Optional[str] = None,
) -> None:
    """Materialize receipt aggregates when using the global (all-centers) demand table."""
    if center_syn_id is None:
        sync_daily_demand_from_receipts(db_session, drug_codes=[drug_code])
