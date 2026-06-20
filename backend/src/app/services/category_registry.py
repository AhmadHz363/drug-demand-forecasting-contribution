"""Upsert and search unique categories from receipt ingestion."""

from __future__ import annotations

from collections import Counter
from typing import Any, Optional

from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app.models.category import Category
from app.models.drug_receipt import DrugReceipt


def normalize_category_code(value: Any) -> Optional[str]:
    """Return a stripped category code or None when absent."""
    if value is None:
        return None
    code = str(value).strip()
    return code or None


def upsert_categories_from_receipt_rows(
    db_session: Session,
    rows: list[dict],
) -> dict[str, int]:
    """
    Insert new categories or increment ``receipt_count`` for existing ones.

    Returns a mapping of ``category_code`` → ``category_id`` for linking receipt rows.
    """
    if not rows:
        return {}

    counts: Counter[str] = Counter()
    for row in rows:
        code = normalize_category_code(row.get("drug_category"))
        if code is None:
            continue
        counts[code] += 1

    if not counts:
        return {}

    code_to_id: dict[str, int] = {}
    for code, count in counts.items():
        existing = (
            db_session.query(Category)
            .filter(Category.category_code == code)
            .one_or_none()
        )
        if existing is not None:
            existing.receipt_count += count
            code_to_id[code] = existing.id
            continue

        category = Category(
            category_code=code,
            name=None,
            receipt_count=count,
        )
        db_session.add(category)
        db_session.flush()
        code_to_id[code] = category.id

    return code_to_id


def attach_category_ids(rows: list[dict], code_to_id: dict[str, int]) -> None:
    """Mutate receipt row dicts in place with the resolved ``category_id`` when present."""
    for row in rows:
        code = normalize_category_code(row.get("drug_category"))
        if code is None:
            continue
        category_id = code_to_id.get(code)
        if category_id is None:
            raise ValueError(f"No category_id resolved for drug_category={code!r}")
        row["category_id"] = category_id


def search_categories(
    db_session: Session,
    query: Optional[str] = None,
    *,
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[dict], int]:
    """Paginated search over categories with distinct drug counts from receipts."""
    drug_counts = (
        db_session.query(
            DrugReceipt.category_id.label("category_id"),
            func.count(func.distinct(DrugReceipt.drug_id)).label("drug_count"),
        )
        .filter(DrugReceipt.category_id.isnot(None))
        .group_by(DrugReceipt.category_id)
        .subquery()
    )

    base = db_session.query(
        Category,
        func.coalesce(drug_counts.c.drug_count, 0).label("drug_count"),
    ).outerjoin(drug_counts, Category.id == drug_counts.c.category_id)

    normalized = (query or "").strip()
    if normalized:
        pattern = f"%{normalized}%"
        base = base.filter(
            or_(
                Category.category_code.ilike(pattern),
                Category.name.ilike(pattern),
            )
        )

    count_query = db_session.query(func.count(Category.id))
    if normalized:
        pattern = f"%{normalized}%"
        count_query = count_query.filter(
            or_(
                Category.category_code.ilike(pattern),
                Category.name.ilike(pattern),
            )
        )
    total = count_query.scalar() or 0

    rows = (
        base.order_by(Category.category_code.asc())
        .offset(max(page - 1, 0) * page_size)
        .limit(page_size)
        .all()
    )
    return [
        {
            "id": category.id,
            "category_code": category.category_code,
            "name": category.name,
            "receipt_count": category.receipt_count,
            "drug_count": int(drug_count),
            "created_at": category.created_at,
            "updated_at": category.updated_at,
        }
        for category, drug_count in rows
    ], total
