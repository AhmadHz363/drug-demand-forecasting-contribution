#!/usr/bin/env python3
"""Clean re-ingest of hospital pharmacy Excel exports into drug_receipts.

Usage (from backend/):
  PYTHONPATH=src .venv/bin/python scripts/reingest_hospital_excel.py \\
    --files "/Users/.../hospital 2023.xlsx" "/Users/.../hospital 2024.xlsx"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from sqlalchemy import text

from app.core.database import SessionLocal, engine
from app.services.receipt_ingestion import ingest_receipt_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("reingest_hospital_excel")


def _truncate_receipt_tables() -> None:
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE daily_drug_demand RESTART IDENTITY CASCADE"))
        conn.execute(text("TRUNCATE TABLE drug_receipts RESTART IDENTITY CASCADE"))
    logger.info("Truncated daily_drug_demand and drug_receipts")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="Hospital Excel/CSV receipt exports in chronological order",
    )
    parser.add_argument(
        "--no-truncate",
        action="store_true",
        help="Skip truncate (relies on idempotent per-line replace)",
    )
    args = parser.parse_args(argv)

    paths = [Path(p).expanduser().resolve() for p in args.files]
    for path in paths:
        if not path.is_file():
            logger.error("File not found: %s", path)
            return 1

    if not args.no_truncate:
        _truncate_receipt_tables()

    db = SessionLocal()
    try:
        for path in paths:
            logger.info("Ingesting %s (%.1f MB)", path.name, path.stat().st_size / 1e6)
            outcome = ingest_receipt_file(path.read_bytes(), db, filename=path.name)
            logger.info(
                "Done %s: inserted=%s failed=%s",
                path.name,
                outcome.inserted_rows,
                outcome.failed_rows,
            )
            if outcome.failed_rows:
                for err in outcome.errors[:10]:
                    logger.warning("row %s: %s", err.row_index, err.message)
                if outcome.inserted_rows == 0:
                    return 1
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
