"""Upsert and search unique drugs from receipt ingestion."""

from __future__ import annotations

from collections import Counter
from datetime import date, timedelta
from typing import Optional

from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app.cold_start.graduation import decide_stage, get_observation_count
from app.models.drug import Drug
from app.models.drug_receipt import DrugReceipt
from app.services.category_registry import attach_category_ids, upsert_categories_from_receipt_rows
from app.services.demand_aggregation import load_demand_series_from_receipts


def upsert_drugs_from_receipt_rows(
    db_session: Session,
    rows: list[dict],
) -> dict[str, int]:
    """
    Insert new drugs or increment ``receipt_count`` for existing ones.

    Returns a mapping of ``drug_code`` → ``drug_id`` for linking receipt rows.
    """
    if not rows:
        return {}

    counts: Counter[str] = Counter()
    metadata_by_code: dict[str, dict[str, str | None]] = {}

    for row in rows:
        code = str(row["drug_code"]).strip()
        counts[code] += 1
        if code not in metadata_by_code:
            metadata_by_code[code] = {
                "drug_name": row.get("drug_name"),
                "drug_category": row.get("drug_category"),
            }
            continue
        current = metadata_by_code[code]
        if not current.get("drug_name") and row.get("drug_name"):
            current["drug_name"] = row["drug_name"]
        if not current.get("drug_category") and row.get("drug_category"):
            current["drug_category"] = row["drug_category"]

    code_to_id: dict[str, int] = {}
    for code, count in counts.items():
        existing = (
            db_session.query(Drug)
            .filter(Drug.drug_code == code)
            .one_or_none()
        )
        if existing is not None:
            existing.receipt_count += count
            code_to_id[code] = existing.id
            continue

        info = metadata_by_code[code]
        drug = Drug(
            drug_code=code,
            drug_name=info.get("drug_name"),
            drug_category=info.get("drug_category"),
            receipt_count=count,
        )
        db_session.add(drug)
        db_session.flush()
        code_to_id[code] = drug.id

    return code_to_id


def attach_drug_ids(rows: list[dict], code_to_id: dict[str, int]) -> None:
    """Mutate receipt row dicts in place with the resolved ``drug_id``."""
    for row in rows:
        code = str(row["drug_code"]).strip()
        drug_id = code_to_id.get(code)
        if drug_id is None:
            raise ValueError(f"No drug_id resolved for drug_code={code!r}")
        row["drug_id"] = drug_id


def insert_receipt_rows(db_session: Session, rows: list[dict]) -> None:
    """Upsert drugs/categories and bulk-insert receipt rows with foreign keys."""
    code_to_id = upsert_drugs_from_receipt_rows(db_session, rows)
    attach_drug_ids(rows, code_to_id)
    category_code_to_id = upsert_categories_from_receipt_rows(db_session, rows)
    attach_category_ids(rows, category_code_to_id)
    db_session.bulk_insert_mappings(DrugReceipt, rows)


def search_drugs(
    db_session: Session,
    query: Optional[str] = None,
    *,
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[Drug], int]:
    """Paginated search over the unique drugs table."""
    base = db_session.query(Drug)
    normalized = (query or "").strip()
    if normalized:
        pattern = f"%{normalized}%"
        base = base.filter(
            or_(
                Drug.drug_code.ilike(pattern),
                Drug.drug_name.ilike(pattern),
                Drug.drug_category.ilike(pattern),
            )
        )

    total = base.with_entities(func.count(Drug.id)).scalar() or 0
    rows = (
        base.order_by(Drug.drug_code.asc())
        .offset(max(page - 1, 0) * page_size)
        .limit(page_size)
        .all()
    )
    return rows, total


def get_drug_by_code(db_session: Session, drug_code: str) -> Drug | None:
    """Return the registry row for a drug code, if present."""
    normalized = drug_code.strip()
    if not normalized:
        return None
    return (
        db_session.query(Drug)
        .filter(Drug.drug_code == normalized)
        .one_or_none()
    )


def get_drug_detail(
    db_session: Session,
    drug_code: str,
    *,
    lookback_days: int = 365,
) -> dict | None:
    """Aggregate registry metadata, receipt stats, and recent demand history."""
    drug = get_drug_by_code(db_session, drug_code)
    if drug is None:
        return None

    stats = (
        db_session.query(
            func.coalesce(func.sum(DrugReceipt.quantity), 0.0),
            func.count(func.distinct(DrugReceipt.receipt_date)),
            func.min(DrugReceipt.receipt_date),
            func.max(DrugReceipt.receipt_date),
            func.count(func.distinct(DrugReceipt.center_syn_id)),
        )
        .filter(DrugReceipt.drug_code == drug.drug_code)
        .one()
    )

    total_quantity = float(stats[0] or 0.0)
    distinct_receipt_days = int(stats[1] or 0)
    first_receipt_date = stats[2]
    last_receipt_date = stats[3]
    center_count = int(stats[4] or 0)

    if first_receipt_date is not None and last_receipt_date is not None:
        window_start = max(
            last_receipt_date - timedelta(days=lookback_days - 1),
            first_receipt_date,
        )
        series_rows = load_demand_series_from_receipts(
            db_session,
            drug.drug_code,
            start_date=window_start,
            end_date=last_receipt_date,
        )
    else:
        series_rows = []

    observation_count = get_observation_count(drug.drug_code, db_session)
    avg_daily_quantity = (
        total_quantity / distinct_receipt_days if distinct_receipt_days > 0 else None
    )

    return {
        "drug": drug,
        "total_quantity": total_quantity,
        "distinct_receipt_days": distinct_receipt_days,
        "first_receipt_date": first_receipt_date,
        "last_receipt_date": last_receipt_date,
        "center_count": center_count,
        "avg_daily_quantity": avg_daily_quantity,
        "observation_count": observation_count,
        "graduation_stage": decide_stage(observation_count),
        "lookback_days": lookback_days,
        "demand_series": [
            {"date": demand_date, "quantity": quantity}
            for demand_date, quantity in series_rows
        ],
    }
