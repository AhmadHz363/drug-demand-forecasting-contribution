"""Ingest hospital pharmacy Excel exports into raw + enriched training tables."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.models.hospital_daily_demand_enriched import HospitalDailyDemandEnriched
from app.models.hospital_receipt_raw import HospitalReceiptRaw
from app.schemas.receipt_upload import ReceiptRowError
from app.services.hospital_demand_enrichment import build_enriched_daily_panel

logger = logging.getLogger(__name__)

_RAW_INSERT_BATCH_SIZE = 1000
_ENRICHED_INSERT_BATCH_SIZE = 2000

_EXCEL_TO_RAW: dict[str, str] = {
    "DOC": "doc",
    "LINE": "line",
    "CAT": "cat",
    "C.R": "c_r",
    "DATE": "date_raw",
    "MOV#": "mov_num",
    "Mov des": "mov_des",
    "CODE": "code",
    "ARTICLE": "article",
    "M": "m",
    "C.S": "c_s",
    "QTY": "qty",
    "U.P": "u_p",
    "T.P": "t_p",
    "MRN": "mrn",
    "AD DATE": "ad_date_raw",
    "R": "r",
    "U": "u",
    "AGE": "age_raw",
    "DR": "dr",
}


@dataclass
class HospitalIngestionOutcome:
    raw_inserted_rows: int
    raw_failed_rows: int
    enriched_inserted_rows: int
    filtered_out_rows: int
    errors: list[ReceiptRowError]

    @property
    def inserted_rows(self) -> int:
        return self.raw_inserted_rows

    @property
    def failed_rows(self) -> int:
        return self.raw_failed_rows


def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]
    return out


def _cell_to_raw_text(value: Any) -> Optional[str]:
    if value is None or value is pd.NaT:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return None
        return ts.strftime("%Y-%m-%d")
    text = str(value).strip()
    return text or None


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None or value is pd.NaT:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return None


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None or value is pd.NaT:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_excel_sheets(file_bytes: bytes, *, filename: str) -> dict[str, pd.DataFrame]:
    lower = filename.lower()
    bio = BytesIO(file_bytes)
    if lower.endswith(".csv"):
        return {"Sheet1": pd.read_csv(bio)}
    if lower.endswith((".xlsx", ".xls")):
        workbook = pd.ExcelFile(bio, engine="openpyxl")
        return {sheet: workbook.parse(sheet) for sheet in workbook.sheet_names}
    raise ValueError("Only `.xlsx`, `.xls`, and `.csv` files are accepted")


def _header_norm_key(label: str) -> str:
    return " ".join(label.split()).upper()


def _canonicalize_hospital_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Fold common export header aliases to the hospital raw column names."""
    alias_map = {
        "DOC": "DOC",
        "LINE": "LINE",
        "CAT": "CAT",
        "C.R": "C.R",
        "CR": "C.R",
        "DATE": "DATE",
        "MOV#": "MOV#",
        "MOV #": "MOV#",
        "MOV DES": "Mov des",
        "MOV_DES": "Mov des",
        "MOV.DES": "Mov des",
        "CODE": "CODE",
        "ARTICLE": "ARTICLE",
        "M": "M",
        "C.S": "C.S",
        "CS": "C.S",
        "QTY": "QTY",
        "U.P": "U.P",
        "UP": "U.P",
        "T.P": "T.P",
        "TP": "T.P",
        "MRN": "MRN",
        "AD DATE": "AD DATE",
        "AD": "AD DATE",
        "R": "R",
        "U": "U",
        "AGE": "AGE",
        "DR": "DR",
    }

    out = df.copy()
    renamed: list[str] = []
    for raw in [str(col).strip() for col in out.columns]:
        key = _header_norm_key(raw)
        if key in alias_map:
            renamed.append(alias_map[key])
            continue
        if "MOV" in key and "DES" in key:
            renamed.append("Mov des")
            continue
        renamed.append(raw)
    out.columns = renamed
    return out


def _missing_required_columns(df: pd.DataFrame) -> list[str]:
    required = {"DATE", "Mov des", "CODE", "QTY"}
    return sorted(required - set(df.columns))


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NaT:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _sanitize_enriched_record(record: dict[str, Any], *, source_file: str) -> dict[str, Any]:
    """Convert pandas NaN/NaT to SQL NULL and coerce DB column types."""
    nullable_int_cols = {"cat", "top1_cr", "top2_cr", "top3_cr"}
    nullable_float_cols = {"top_cr_drug", "top_cs_drug", "age_mean", "age_median"}
    nullable_str_cols = {"article", "top_dr_drug"}
    int_cols = {
        "demand",
        "n_patients_drug",
        "n_doctors_drug",
        "n_unique_patients",
        "n_admissions",
        "n_unique_doctors",
        "n_unique_cr",
        "n_unique_cs",
        "n_transactions",
        "n_demand_txns",
        "hospital_total_demand",
    }
    float_cols = {"top1_cr_share", "top2_cr_share", "top3_cr_share"}

    row: dict[str, Any] = {"source_file": source_file}
    for key, value in record.items():
        if _is_missing(value):
            row[key] = None
            continue
        if key in nullable_int_cols:
            row[key] = int(value)
        elif key in int_cols:
            row[key] = int(value)
        elif key in nullable_float_cols | float_cols:
            row[key] = float(value)
        elif key in nullable_str_cols:
            text = str(value).strip()
            row[key] = text or None
        else:
            row[key] = value
    return row


def _delete_existing_for_source(db: Session, source_file: str) -> None:
    db.query(HospitalDailyDemandEnriched).filter(
        HospitalDailyDemandEnriched.source_file == source_file,
    ).delete(synchronize_session=False)
    db.query(HospitalReceiptRaw).filter(
        HospitalReceiptRaw.source_file == source_file,
    ).delete(synchronize_session=False)
    db.flush()


def _build_raw_row(
    row: dict[str, Any],
    *,
    source_file: str,
    source_sheet: str,
    excel_row_number: int,
) -> dict[str, Any]:
    mapped: dict[str, Any] = {
        "source_file": source_file,
        "source_sheet": source_sheet,
        "excel_row_number": excel_row_number,
    }
    for excel_col, db_col in _EXCEL_TO_RAW.items():
        raw_value = row.get(excel_col)
        if db_col in {"date_raw", "ad_date_raw", "age_raw", "mov_des", "code", "article", "mrn", "r", "u", "dr"}:
            mapped[db_col] = _cell_to_raw_text(raw_value)
        elif db_col in {"doc"}:
            mapped[db_col] = _coerce_optional_int(raw_value)
        elif db_col in {"line", "cat", "c_r", "mov_num", "m", "c_s"}:
            mapped[db_col] = _coerce_optional_int(raw_value)
        else:
            mapped[db_col] = _coerce_optional_float(raw_value)
    return mapped


def ingest_hospital_receipt_file(
    file_bytes: bytes,
    db: Session,
    *,
    filename: str = "upload.xlsx",
) -> HospitalIngestionOutcome:
    """Persist raw Excel rows, then build and store the cleaned enriched training panel."""
    errors: list[ReceiptRowError] = []
    raw_rows: list[dict[str, Any]] = []
    combined_frames: list[pd.DataFrame] = []

    try:
        sheets = _load_excel_sheets(file_bytes, filename=filename)
    except Exception as exc:
        logger.exception("Failed to read hospital receipt file filename=%s", filename)
        return HospitalIngestionOutcome(
            raw_inserted_rows=0,
            raw_failed_rows=1,
            enriched_inserted_rows=0,
            filtered_out_rows=0,
            errors=[ReceiptRowError(row_index=0, message=str(exc))],
        )

    excel_row_offset = 2
    for sheet_name, df in sheets.items():
        normalized = _canonicalize_hospital_headers(_normalize_headers(df))
        missing = _missing_required_columns(normalized)
        if missing:
            logger.info(
                "Skipping sheet %s in %s — missing columns: %s",
                sheet_name,
                filename,
                ", ".join(missing),
            )
            continue

        combined_frames.append(normalized)
        for pos, raw_row in enumerate(normalized.to_dict("records"), start=excel_row_offset):
            raw_rows.append(
                _build_raw_row(
                    raw_row,
                    source_file=filename,
                    source_sheet=sheet_name,
                    excel_row_number=pos,
                ),
            )

    if not raw_rows:
        return HospitalIngestionOutcome(
            raw_inserted_rows=0,
            raw_failed_rows=0,
            enriched_inserted_rows=0,
            filtered_out_rows=0,
            errors=errors,
        )

    try:
        _delete_existing_for_source(db, filename)

        raw_inserted = 0
        for offset in range(0, len(raw_rows), _RAW_INSERT_BATCH_SIZE):
            batch = raw_rows[offset : offset + _RAW_INSERT_BATCH_SIZE]
            db.bulk_insert_mappings(HospitalReceiptRaw, batch)
            db.commit()
            raw_inserted += len(batch)

        combined = pd.concat(combined_frames, ignore_index=True)
        enriched_df, filtered_out_rows = build_enriched_daily_panel(combined)
        enriched_inserted = 0

        if not enriched_df.empty:
            enriched_records = [
                _sanitize_enriched_record(record, source_file=filename)
                for record in enriched_df.to_dict("records")
            ]

            for offset in range(0, len(enriched_records), _ENRICHED_INSERT_BATCH_SIZE):
                batch = enriched_records[offset : offset + _ENRICHED_INSERT_BATCH_SIZE]
                db.bulk_insert_mappings(HospitalDailyDemandEnriched, batch)
                db.commit()
                enriched_inserted += len(batch)

        logger.info(
            "Hospital ingest complete filename=%s raw=%s enriched=%s filtered_out=%s failed=%s",
            filename,
            raw_inserted,
            enriched_inserted,
            filtered_out_rows,
            len(errors),
        )
        return HospitalIngestionOutcome(
            raw_inserted_rows=raw_inserted,
            raw_failed_rows=len(errors),
            enriched_inserted_rows=enriched_inserted,
            filtered_out_rows=filtered_out_rows,
            errors=errors,
        )
    except Exception as exc:
        logger.exception("Hospital ingest failed filename=%s", filename)
        db.rollback()
        errors.append(
            ReceiptRowError(
                row_index=0,
                message=f"Ingestion aborted; transaction rolled back: {exc}",
            ),
        )
        return HospitalIngestionOutcome(
            raw_inserted_rows=0,
            raw_failed_rows=len(errors),
            enriched_inserted_rows=0,
            filtered_out_rows=0,
            errors=errors,
        )
