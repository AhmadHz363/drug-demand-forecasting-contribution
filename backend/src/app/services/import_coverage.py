"""Record and query hospital receipt import coverage periods."""

from __future__ import annotations

import hashlib
import logging
from datetime import date, timedelta
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.import_coverage import ImportCoverage

logger = logging.getLogger(__name__)

# Gaps longer than this between coverage periods are treated as uncovered ledger.
COVERAGE_GAP_THRESHOLD_DAYS = 60


def content_hash_bytes(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def record_import_coverage(
    db_session: Session,
    *,
    filename: str,
    content_hash: str,
    min_receipt_date: date,
    max_receipt_date: date,
    row_count: int,
    center_scope: Optional[str] = None,
    notes: Optional[str] = None,
) -> ImportCoverage:
    """Upsert a successful import coverage row keyed by content hash."""
    existing = (
        db_session.query(ImportCoverage)
        .filter(ImportCoverage.content_hash == content_hash)
        .one_or_none()
    )
    if existing is not None:
        existing.filename = filename
        existing.min_receipt_date = min_receipt_date
        existing.max_receipt_date = max_receipt_date
        existing.row_count = row_count
        existing.center_scope = center_scope
        existing.status = "success"
        existing.notes = notes
        db_session.flush()
        return existing

    row = ImportCoverage(
        filename=filename,
        content_hash=content_hash,
        min_receipt_date=min_receipt_date,
        max_receipt_date=max_receipt_date,
        row_count=row_count,
        center_scope=center_scope,
        status="success",
        notes=notes,
    )
    db_session.add(row)
    db_session.flush()
    logger.info(
        "Recorded import coverage filename=%s rows=%s span=%s..%s",
        filename,
        row_count,
        min_receipt_date,
        max_receipt_date,
    )
    return row


def backfill_coverage_from_receipts(
    db_session: Session,
    *,
    filename: str,
    content_hash: str,
    min_receipt_date: date,
    max_receipt_date: date,
    row_count: int,
    notes: str = "backfill",
) -> ImportCoverage:
    return record_import_coverage(
        db_session,
        filename=filename,
        content_hash=content_hash,
        min_receipt_date=min_receipt_date,
        max_receipt_date=max_receipt_date,
        row_count=row_count,
        notes=notes,
    )


def list_coverage_periods(db_session: Session) -> list[tuple[date, date]]:
    """Return successful coverage intervals sorted by start date."""
    rows = (
        db_session.query(ImportCoverage.min_receipt_date, ImportCoverage.max_receipt_date)
        .filter(ImportCoverage.status == "success")
        .order_by(ImportCoverage.min_receipt_date.asc())
        .all()
    )
    return [(row[0], row[1]) for row in rows]


def merge_coverage_periods(
    periods: list[tuple[date, date]],
    *,
    gap_threshold_days: int = COVERAGE_GAP_THRESHOLD_DAYS,
) -> list[tuple[date, date]]:
    """Merge overlapping/adjacent coverage periods; split on large gaps."""
    if not periods:
        return []
    ordered = sorted(periods, key=lambda p: (p[0], p[1]))
    merged: list[list[date]] = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        cur_start, cur_end = merged[-1]
        gap = (start - cur_end).days
        if gap <= gap_threshold_days:
            merged[-1][1] = max(cur_end, end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def get_merged_coverage_periods(
    db_session: Session,
    *,
    gap_threshold_days: int = COVERAGE_GAP_THRESHOLD_DAYS,
) -> list[tuple[date, date]]:
    return merge_coverage_periods(
        list_coverage_periods(db_session),
        gap_threshold_days=gap_threshold_days,
    )


def latest_coverage_segment(
    db_session: Session,
    *,
    gap_threshold_days: int = COVERAGE_GAP_THRESHOLD_DAYS,
) -> Optional[tuple[date, date]]:
    segments = get_merged_coverage_periods(
        db_session,
        gap_threshold_days=gap_threshold_days,
    )
    if not segments:
        return None
    return segments[-1]


def is_date_covered(
    day: date,
    periods: list[tuple[date, date]],
) -> bool:
    return any(start <= day <= end for start, end in periods)


def coverage_mask_for_dates(
    dates: list[date],
    periods: list[tuple[date, date]],
) -> list[bool]:
    return [is_date_covered(d, periods) for d in dates]


def ensure_default_coverage_from_receipt_bounds(
    db_session: Session,
    *,
    min_date: Optional[date],
    max_date: Optional[date],
    row_count: int,
) -> None:
    """If no coverage rows exist, synthesize one from current receipt bounds."""
    if min_date is None or max_date is None:
        return
    existing = db_session.query(func.count(ImportCoverage.id)).scalar() or 0
    if existing > 0:
        return
    synthetic_hash = hashlib.sha256(
        f"backfill:{min_date.isoformat()}:{max_date.isoformat()}:{row_count}".encode()
    ).hexdigest()
    record_import_coverage(
        db_session,
        filename="backfill_from_receipts",
        content_hash=synthetic_hash,
        min_receipt_date=min_date,
        max_receipt_date=max_date,
        row_count=row_count,
        notes="Auto-backfill from existing drug_receipts bounds",
    )
    db_session.commit()
