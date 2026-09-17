#!/usr/bin/env python3
"""
Populate categories, drugs, and drug_receipts from hospital_receipt_raw.

Usage (from backend/):
  source .venv/bin/activate
  export PYTHONPATH=src
  python scripts/backfill_registry_from_hospital_receipts.py [--reset]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from app.core.database import SessionLocal
from app.services.registry_backfill import backfill_registry_from_hospital_receipts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill receipt registry tables from hospital_receipt_raw",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear categories, drugs, and drug_receipts before backfill",
    )
    args = parser.parse_args()

    db = SessionLocal()
    try:
        outcome = backfill_registry_from_hospital_receipts(db, reset=args.reset)
        print(
            f"Backfill complete: {outcome.categories_upserted} categories, "
            f"{outcome.drugs_upserted} drugs, {outcome.receipts_inserted} receipt lines "
            f"({outcome.receipts_skipped} skipped)."
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
