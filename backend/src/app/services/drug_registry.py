"""Upsert and search unique drugs from receipt ingestion."""

from __future__ import annotations

from collections import Counter
from typing import Optional

from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app.models.drug import Drug
from app.models.drug_receipt import DrugReceipt
from app.services.category_registry import attach_category_ids, upsert_categories_from_receipt_rows


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
