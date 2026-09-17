"""Populate receipt registry tables from persisted hospital_receipt_raw rows."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import func, text
from sqlalchemy.orm import Session

from app.models.category import Category
from app.models.drug import Drug
from app.models.drug_receipt import DrugReceipt
from app.models.hospital_receipt_raw import HospitalReceiptRaw
from app.services.category_registry import normalize_category_code
from app.services.hospital_demand_enrichment import parse_hospital_date

logger = logging.getLogger(__name__)

_RECEIPT_BATCH_SIZE = 2000


@dataclass
class RegistryBackfillOutcome:
    categories_upserted: int
    drugs_upserted: int
    receipts_inserted: int
    receipts_skipped: int


def _raw_row_to_receipt_dict(
    row: HospitalReceiptRaw,
    *,
    drug_id: int,
    category_id: Optional[int],
) -> Optional[dict]:
    code = (row.code or "").strip()
    if not code:
        return None

    receipt_date = parse_hospital_date(row.date_raw)
    if receipt_date is None:
        return None

    category_code = normalize_category_code(row.cat)
    receipt_id = str(row.doc).strip() if row.doc is not None else str(row.id)

    return {
        "drug_id": drug_id,
        "category_id": category_id,
        "receipt_id": receipt_id,
        "line_count": row.line,
        "drug_category": category_code,
        "center_receipt": str(row.c_r) if row.c_r is not None else None,
        "receipt_date": receipt_date,
        "movement_number": str(row.mov_num) if row.mov_num is not None else None,
        "movement_type": row.mov_des,
        "drug_code": code,
        "drug_name": row.article,
        "month": row.m,
        "center_syn_id": str(row.c_s) if row.c_s is not None else None,
        "quantity": float(row.qty) if row.qty is not None else None,
        "unit_price": float(row.u_p) if row.u_p is not None else None,
        "total_price": float(row.t_p) if row.t_p is not None else None,
        "admission_date": parse_hospital_date(row.ad_date_raw),
        "room_number": row.r,
        "bed_number": row.u,
        "doctor_name": row.dr,
    }


def backfill_registry_from_hospital_receipts(
    db_session: Session,
    *,
    reset: bool = False,
) -> RegistryBackfillOutcome:
    """
    Derive unique categories and drugs from ``hospital_receipt_raw`` and load
    ``drug_receipts`` with foreign keys linking each line to its drug and category.
    """
    if reset:
        db_session.query(DrugReceipt).delete()
        db_session.query(Drug).delete()
        db_session.query(Category).delete()
        db_session.commit()

    category_stats = db_session.execute(
        text(
            """
            SELECT cat::text AS category_code, COUNT(*) AS receipt_count
            FROM hospital_receipt_raw
            WHERE cat IS NOT NULL
            GROUP BY cat
            """
        )
    ).mappings().all()

    categories_upserted = 0
    for row in category_stats:
        code = str(row["category_code"]).strip()
        count = int(row["receipt_count"])
        existing = (
            db_session.query(Category)
            .filter(Category.category_code == code)
            .one_or_none()
        )
        if existing is None:
            db_session.add(
                Category(
                    category_code=code,
                    name=None,
                    receipt_count=count,
                )
            )
            categories_upserted += 1
        else:
            existing.receipt_count = count
    db_session.flush()

    drug_stats = db_session.execute(
        text(
            """
            SELECT
                TRIM(code) AS drug_code,
                MAX(article) AS drug_name,
                MAX(cat::text) AS drug_category,
                COUNT(*) AS receipt_count
            FROM hospital_receipt_raw
            WHERE code IS NOT NULL AND TRIM(code) <> ''
            GROUP BY TRIM(code)
            """
        )
    ).mappings().all()

    drugs_upserted = 0
    for row in drug_stats:
        code = str(row["drug_code"]).strip()
        count = int(row["receipt_count"])
        existing = (
            db_session.query(Drug)
            .filter(Drug.drug_code == code)
            .one_or_none()
        )
        if existing is None:
            db_session.add(
                Drug(
                    drug_code=code,
                    drug_name=row["drug_name"],
                    drug_category=row["drug_category"],
                    receipt_count=count,
                )
            )
            drugs_upserted += 1
        else:
            existing.drug_name = row["drug_name"]
            existing.drug_category = row["drug_category"]
            existing.receipt_count = count
    db_session.commit()

    drug_id_by_code = {
        drug.drug_code: drug.id
        for drug in db_session.query(Drug.drug_code, Drug.id).all()
    }
    category_id_by_code = {
        category.category_code: category.id
        for category in db_session.query(Category.category_code, Category.id).all()
    }

    receipts_inserted = 0
    receipts_skipped = 0
    batch: list[dict] = []
    last_id = 0
    total_raw = db_session.query(func.count(HospitalReceiptRaw.id)).scalar() or 0
    processed = 0

    while True:
        rows = (
            db_session.query(HospitalReceiptRaw)
            .filter(HospitalReceiptRaw.id > last_id)
            .order_by(HospitalReceiptRaw.id.asc())
            .limit(_RECEIPT_BATCH_SIZE)
            .all()
        )
        if not rows:
            break

        for row in rows:
            processed += 1
            code = (row.code or "").strip()
            drug_id = drug_id_by_code.get(code)
            if drug_id is None:
                receipts_skipped += 1
                continue

            category_code = normalize_category_code(row.cat)
            category_id = category_id_by_code.get(category_code) if category_code else None
            receipt = _raw_row_to_receipt_dict(
                row,
                drug_id=drug_id,
                category_id=category_id,
            )
            if receipt is None:
                receipts_skipped += 1
                continue

            batch.append(receipt)

        last_id = rows[-1].id
        if batch:
            db_session.bulk_insert_mappings(DrugReceipt, batch)
            db_session.commit()
            receipts_inserted += len(batch)
            batch.clear()

        if processed % 50000 == 0 or processed >= total_raw:
            logger.info(
                "Registry backfill progress receipts=%s/%s raw_rows=%s",
                receipts_inserted,
                total_raw,
                processed,
            )

    if batch:
        db_session.bulk_insert_mappings(DrugReceipt, batch)
        db_session.commit()
        receipts_inserted += len(batch)

    logger.info(
        "Registry backfill complete categories=%s drugs=%s receipts=%s skipped=%s",
        categories_upserted,
        drugs_upserted,
        receipts_inserted,
        receipts_skipped,
    )
    return RegistryBackfillOutcome(
        categories_upserted=categories_upserted,
        drugs_upserted=drugs_upserted,
        receipts_inserted=receipts_inserted,
        receipts_skipped=receipts_skipped,
    )
