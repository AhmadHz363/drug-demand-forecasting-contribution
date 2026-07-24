from __future__ import annotations

import logging
import re
from datetime import date, datetime
from io import BytesIO
from collections.abc import Mapping
from typing import Any, Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.models.drug_receipt import DrugReceipt
from app.schemas.receipt_upload import ReceiptRowError
from app.services.demand_aggregation import sync_daily_demand_from_receipts
from app.services.category_registry import attach_category_ids, upsert_categories_from_receipt_rows
from app.services.drug_registry import attach_drug_ids, upsert_drugs_from_receipt_rows
from app.services.import_coverage import content_hash_bytes, record_import_coverage
from app.services.column_mapping import (
    DATE_FIELDS,
    EXCEL_TO_DB_COLUMNS,
    REQUIRED_CANONICAL_EXCEL_COLUMNS,
    FLOAT_FIELDS,
    INTEGER_FIELDS,
    STRING_FIELDS,
)

logger = logging.getLogger(__name__)

_REQUIRED_ROW_FIELDS = frozenset({"receipt_id", "drug_code", "receipt_date"})
_RECEIPT_INSERT_BATCH_SIZE = 500


class IngestionOutcome:
    __slots__ = ("inserted_rows", "failed_rows", "errors")

    def __init__(
        self,
        inserted_rows: int,
        failed_rows: int,
        errors: list[ReceiptRowError],
    ) -> None:
        self.inserted_rows = inserted_rows
        self.failed_rows = failed_rows
        self.errors = errors


def validate_excel_columns(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Ensure minimum columns exist after `_canonicalize_excel_headers` (Code, Date). Doc is optional."""
    actual = {str(c).strip() for c in df.columns}
    missing = sorted(REQUIRED_CANONICAL_EXCEL_COLUMNS - actual)
    if missing:
        return False, missing
    return True, []


def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _header_norm_key(label: str) -> str:
    return " ".join(label.split()).upper()


def _header_flags(labels: list[str]) -> tuple[bool, bool]:
    """Detect layout so a lone `MOV` column maps to Mov# vs Mov correctly."""
    keys = {_header_norm_key(x) for x in labels}
    has_mov_num = "MOV#" in keys or "MOV #" in keys
    has_mov_des = any("MOV" in k and "DES" in k for k in keys)
    return has_mov_num, has_mov_des


def _resolve_raw_header_to_canonical(raw: str, has_mov_num: bool, has_mov_des: bool) -> Optional[str]:
    """Map export-specific labels (e.g. DOC, MOV DES) to keys in `EXCEL_TO_DB_COLUMNS`."""
    k = _header_norm_key(raw)
    if k in ("MOV DES", "MOV_DES", "MOV.DES") or ("MOV" in k and "DES" in k):
        return "Mov"
    if k in ("MOV#", "MOV #"):
        return "Mov#"
    if k == "MOV":
        if has_mov_num:
            return "Mov"
        return "Mov#"
    for canon in EXCEL_TO_DB_COLUMNS:
        if _header_norm_key(canon) == k:
            return canon
    return None


def _canonicalize_excel_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Strip headers and fold aliases (case, MOV/MOV DES, etc.) to canonical Excel keys."""
    out = df.copy()
    stripped = [str(c).strip() for c in out.columns]
    has_mov_num, has_mov_des = _header_flags(stripped)
    out.columns = [
        _resolve_raw_header_to_canonical(s, has_mov_num, has_mov_des) or s for s in stripped
    ]
    return out


def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {excel: db for excel, db in EXCEL_TO_DB_COLUMNS.items() if excel in df.columns}
    subset = df[list(rename.keys())].copy()
    subset = subset.rename(columns=rename)
    if "receipt_id" not in subset.columns:
        subset = subset.copy()
        subset["receipt_id"] = np.nan
    return subset


def _needs_synthetic_receipt_id(value: Any) -> bool:
    """True when Doc / receipt_id is absent or empty so we can assign an incremental id."""
    if value is None:
        return True
    if value is pd.NaT:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _empty_to_none(value: Any) -> Optional[Any]:
    if value is None:
        return None
    if isinstance(value, str):
        return None if not value.strip() else value
    if value is pd.NaT:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _coerce_int(raw: Any, field: str) -> tuple[Optional[int], Optional[str]]:
    """Parse integers; negative values are allowed (same as source Excel)."""
    v = _empty_to_none(raw)
    if v is None:
        return None, None
    if isinstance(v, bool):
        return None, f"{field}: expected integer, got boolean"
    try:
        if isinstance(v, (float, np.floating)) and not float(v).is_integer():
            return int(v), None
        return int(v), None
    except (TypeError, ValueError, OverflowError) as e:
        return None, f"{field}: cannot parse integer ({e!s})"


def _coerce_float(raw: Any, field: str) -> tuple[Optional[float], Optional[str]]:
    """Parse decimals; negative values are allowed (e.g. quantity or price adjustments)."""
    v = _empty_to_none(raw)
    if v is None:
        return None, None
    try:
        return float(v), None
    except (TypeError, ValueError) as e:
        return None, f"{field}: cannot parse number ({e!s})"


def _coerce_date(raw: Any, field: str) -> tuple[Optional[date], Optional[str]]:
    """Parse hospital ledger dates (DD/MM/YY with irregular whitespace)."""
    v = _empty_to_none(raw)
    if v is None:
        return None, None
    if isinstance(v, datetime):
        return v.date(), None
    if isinstance(v, date):
        return v, None
    if isinstance(v, str):
        v = re.sub(r"\s+", "", v.strip())
    try:
        ts = pd.NaT
        if isinstance(v, str):
            for fmt in ("%d/%m/%y", "%d/%m/%Y", "%Y-%m-%d"):
                ts = pd.to_datetime(v, format=fmt, errors="coerce")
                if not pd.isna(ts):
                    break
            if pd.isna(ts):
                # Lebanese hospital exports are day-first; never use US month-first.
                ts = pd.to_datetime(v, errors="coerce", dayfirst=True)
        else:
            ts = pd.to_datetime(v, errors="coerce", dayfirst=True)
        if pd.isna(ts):
            return None, f"{field}: invalid date"
        dt = ts.to_pydatetime()
        return dt.date(), None
    except Exception as e:  # noqa: BLE001 — log and treat as parse failure
        return None, f"{field}: invalid date ({e!s})"


def _coerce_str(raw: Any) -> Optional[str]:
    v = _empty_to_none(raw)
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _build_row_dict(row: Mapping[str, Any]) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    data: dict[str, Any] = {}
    errors: list[str] = []

    for field in row:
        raw = row.get(field)
        if field in INTEGER_FIELDS:
            val, err = _coerce_int(raw, field)
            if err:
                errors.append(err)
            data[field] = val
        elif field in FLOAT_FIELDS:
            val, err = _coerce_float(raw, field)
            if err:
                errors.append(err)
            data[field] = val
        elif field in DATE_FIELDS:
            val, err = _coerce_date(raw, field)
            if err:
                errors.append(err)
            data[field] = val
        elif field in STRING_FIELDS:
            data[field] = _coerce_str(raw)
        else:
            data[field] = raw

    for req in _REQUIRED_ROW_FIELDS:
        if data.get(req) is None:
            errors.append(f"Missing required value for {req}")

    if errors:
        return None, "; ".join(errors)

    rec_id = data.get("receipt_id")
    code = data.get("drug_code")
    assert rec_id is not None and code is not None
    data["receipt_id"] = str(rec_id).strip()
    data["drug_code"] = str(code).strip()

    return data, None


def _dedupe_key(record: dict[str, Any]) -> tuple[Any, ...]:
    """Identity for a receipt line — includes LINE so multi-line docs are kept."""
    line = record.get("line_count")
    mov = record.get("movement_number")
    return (
        str(record["receipt_id"]).strip(),
        line if line is not None else "",
        str(mov).strip() if mov is not None else "",
        str(record["drug_code"]).strip(),
        record["receipt_date"],
    )


def _dedupe_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: set[tuple[Any, ...]] = set()
    unique: list[dict[str, Any]] = []
    duplicates = 0
    for r in records:
        key = _dedupe_key(r)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique.append(r)
    return unique, duplicates


def _delete_existing_receipt_lines(db: Session, records: list[dict[str, Any]]) -> int:
    """Remove prior rows matching upload identity so re-uploads are idempotent."""
    if not records:
        return 0
    deleted = 0
    # Chunk to avoid oversized IN clauses on large hospital exports.
    chunk_size = 500
    for offset in range(0, len(records), chunk_size):
        chunk = records[offset : offset + chunk_size]
        receipt_ids = sorted({str(r["receipt_id"]).strip() for r in chunk})
        existing = (
            db.query(DrugReceipt)
            .filter(DrugReceipt.receipt_id.in_(receipt_ids))
            .all()
        )
        if not existing:
            continue
        wanted = {_dedupe_key(r) for r in chunk}
        to_delete = [
            row
            for row in existing
            if (
                str(row.receipt_id).strip(),
                row.line_count if row.line_count is not None else "",
                str(row.movement_number).strip() if row.movement_number is not None else "",
                str(row.drug_code).strip(),
                row.receipt_date,
            )
            in wanted
        ]
        for row in to_delete:
            db.delete(row)
            deleted += 1
    if deleted:
        db.flush()
        logger.info("Removed %s existing receipt lines before re-insert (idempotent upload)", deleted)
    return deleted


_ACCEPTED_EXTENSIONS = (".xlsx", ".xls", ".csv")


def _load_receipt_dataframe(file_bytes: bytes, *, filename: str) -> pd.DataFrame:
    lower = filename.lower()
    bio = BytesIO(file_bytes)
    if lower.endswith(".csv"):
        return pd.read_csv(bio)
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(bio, engine="openpyxl")
    raise ValueError(
        f"Unsupported file type. Accepted extensions: {', '.join(_ACCEPTED_EXTENSIONS)}",
    )


def ingest_receipt_file(
    file_bytes: bytes,
    db: Session,
    *,
    filename: str = "upload.xlsx",
) -> IngestionOutcome:
    """
    Validate, clean, deduplicate receipt rows from a spreadsheet and persist them.

    Supports ``.xlsx``, ``.xls``, and ``.csv`` exports with the same column layout.
    Invalid rows are logged and collected without aborting processing of other rows.
    All inserts run in one transaction : rollback when the bulk insert fails.
    """
    errors: list[ReceiptRowError] = []
    parsed_rows: list[dict[str, Any]] = []

    try:
        df = _load_receipt_dataframe(file_bytes, filename=filename)
        logger.info(
            "Loaded receipt file filename=%s rows=%s cols=%s",
            filename,
            len(df),
            len(df.columns),
        )
    except Exception as e:
        logger.exception("Failed to read receipt file filename=%s", filename)
        err = [ReceiptRowError(row_index=0, message=str(e))]
        return IngestionOutcome(0, len(err), err)

    df = _normalize_headers(df)
    df = _canonicalize_excel_headers(df)

    ok, missing = validate_excel_columns(df)
    if not ok:
        msg = "Missing columns: " + ", ".join(missing)
        logger.warning(msg)
        errors = [ReceiptRowError(row_index=0, message=msg)]
        return IngestionOutcome(0, len(errors), errors)

    mapped = _map_columns(df)

    excel_row_offset = 2  # 1-based Excel row assuming single header row
    duplicates_in_file = 0
    synthetic_receipt_seq = 0

    for pos, raw_row in enumerate(mapped.to_dict("records"), start=excel_row_offset):
        row = dict(raw_row)
        rid = row.get("receipt_id")
        if _needs_synthetic_receipt_id(rid):
            synthetic_receipt_seq += 1
            row["receipt_id"] = str(synthetic_receipt_seq)

        rec, msg = _build_row_dict(row)
        if rec is None:
            err = ReceiptRowError(row_index=pos, message=msg or "Invalid row")
            errors.append(err)
            logger.warning("Invalid row skipped: row_index=%s reason=%s", pos, err.message)
            continue
        parsed_rows.append(rec)

    parsed_rows, extra_dup_count = _dedupe_records(parsed_rows)
    duplicates_in_file += extra_dup_count
    if synthetic_receipt_seq:
        logger.info("Assigned incremental receipt_id for rows missing Doc: %s rows", synthetic_receipt_seq)
    if duplicates_in_file:
        logger.info(
            "Removed duplicate rows (same receipt_id, line, movement, drug_code, receipt_date): %s",
            duplicates_in_file,
        )

    if not parsed_rows:
        logger.warning("No valid rows after cleaning; duplicates_removed=%s", duplicates_in_file)
        return IngestionOutcome(inserted_rows=0, failed_rows=len(errors), errors=errors)

    try:
        replaced = _delete_existing_receipt_lines(db, parsed_rows)
        code_to_id = upsert_drugs_from_receipt_rows(db, parsed_rows)
        attach_drug_ids(parsed_rows, code_to_id)
        category_code_to_id = upsert_categories_from_receipt_rows(db, parsed_rows)
        attach_category_ids(parsed_rows, category_code_to_id)
        # Commit registry first so large exports can recover mid-file.
        db.commit()
        inserted = 0
        for offset in range(0, len(parsed_rows), _RECEIPT_INSERT_BATCH_SIZE):
            batch = parsed_rows[offset : offset + _RECEIPT_INSERT_BATCH_SIZE]
            db.bulk_insert_mappings(DrugReceipt, batch)
            db.commit()
            inserted += len(batch)
            if offset == 0 or (offset // _RECEIPT_INSERT_BATCH_SIZE) % 50 == 0:
                logger.info(
                    "Receipt insert progress filename=%s rows=%s/%s",
                    filename,
                    inserted,
                    len(parsed_rows),
                )
        affected_codes = sorted({row["drug_code"] for row in parsed_rows})
        sync_daily_demand_from_receipts(db, drug_codes=affected_codes)
        receipt_dates = [row["receipt_date"] for row in parsed_rows]
        centers = {
            str(row["center_syn_id"]).strip()
            for row in parsed_rows
            if row.get("center_syn_id") is not None and str(row["center_syn_id"]).strip()
        }
        record_import_coverage(
            db,
            filename=filename,
            content_hash=content_hash_bytes(file_bytes),
            min_receipt_date=min(receipt_dates),
            max_receipt_date=max(receipt_dates),
            row_count=inserted,
            center_scope=",".join(sorted(centers)) if centers else None,
            notes=f"ingest replaced={replaced} failed_rows={len(errors)}",
        )
        db.commit()
        logger.info(
            "Inserted %s drug receipt rows (replaced=%s); failed=%s; duplicates_skipped=%s; synced %d drug(s) to daily_drug_demand",
            inserted,
            replaced,
            len(errors),
            duplicates_in_file,
            len(affected_codes),
        )
        return IngestionOutcome(
            inserted_rows=inserted,
            failed_rows=len(errors),
            errors=errors,
        )
    except Exception as e:
        logger.exception("Database insert failed; rolling back batch")
        db.rollback()
        errors.append(
            ReceiptRowError(
                row_index=0,
                message=f"Bulk insert aborted; transaction rolled back: {e}",
            ),
        )
        return IngestionOutcome(inserted_rows=0, failed_rows=len(errors), errors=errors)


def ingest_receipt_excel(file_bytes: bytes, db: Session) -> IngestionOutcome:
    """Backward-compatible wrapper for Excel-only uploads."""
    return ingest_receipt_file(file_bytes, db, filename="upload.xlsx")
