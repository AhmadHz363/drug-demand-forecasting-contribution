#!/usr/bin/env python3
"""
Load drugs_training_synthetic_filled_with_codes.xlsx into the ``cameo_drugs`` table.

Usage (from backend/):
  source bin/activate
  export PYTHONPATH=src
  python scripts/seed_drugs_from_excel.py [--reset] [--file PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from app.core.database import SessionLocal
from app.models.cameo_drug import CameoDrug

DEFAULT_XLSX = Path.home() / "Downloads" / "drugs_training_synthetic_filled_with_codes.xlsx"


def _clean(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def _int_or_zero(value) -> int:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0
    return int(value)


def load_frames(path: Path) -> pd.DataFrame:
    drug_info = pd.read_excel(path, sheet_name="drug_info_filled")
    match_report = pd.read_excel(path, sheet_name="match_report")
    merged = drug_info.merge(match_report, on="drug_code", how="left", suffixes=("", "_mr"))
    return merged


def seed_drugs(db, merged: pd.DataFrame, *, reset: bool) -> int:
    if reset:
        db.query(CameoDrug).delete()
        db.commit()

    inserted = 0
    batch = 0
    for _, row in merged.iterrows():
        code = _clean(row.get("drug_code"))
        if not code:
            continue

        existing = db.query(CameoDrug).filter(CameoDrug.drug_code == code).one_or_none()
        payload = {
            "generic_name": _clean(row.get("Generic Name")),
            "drug_class": _clean(row.get("Drug Class")),
            "indications": _clean(row.get("Indications")),
            "dosage_form": _clean(row.get("Dosage Form")),
            "strength": _clean(row.get("Strength")),
            "route_of_administration": _clean(row.get("Route of Administration")),
            "side_effects": _clean(row.get("Side Effects")),
            "contraindications": _clean(row.get("Contraindications")),
            "interaction_warnings_precautions": _clean(
                row.get("Interaction warnings & Precautions")
            ),
            "storage_conditions": _clean(row.get("Storage Conditions")),
            "pregnancy_category": _clean(row.get("Pregnancy Category")),
            "reference": _clean(row.get("Reference")),
            "availability": _clean(row.get("Availability")),
            "input_id": int(row["input_id"]) if pd.notna(row.get("input_id")) else None,
            "input_drug_name": _clean(row.get("input_drug_name")),
            "resolved_generic": _clean(row.get("resolved_generic")),
            "matched_source_generic": _clean(row.get("matched_source_generic")),
            "match_status": _clean(row.get("status")),
            "match_method": _clean(row.get("method")),
            "receipt_count": _int_or_zero(row.get("receipt_count")),
            "synthetic_cells_filled": _int_or_zero(row.get("synthetic_cells_filled")),
            "training_quality_note": _clean(row.get("training_quality_note")),
        }

        if existing is None:
            db.add(CameoDrug(drug_code=code, **payload))
            inserted += 1
        else:
            for key, value in payload.items():
                setattr(existing, key, value)

        batch += 1
        if batch % 200 == 0:
            db.commit()

    db.commit()
    total = db.query(CameoDrug).count()
    matched = (
        db.query(CameoDrug)
        .filter(CameoDrug.match_status == "matched_source")
        .count()
    )
    print(f"Upserted {inserted} new rows; drugs table now has {total} rows ({matched} matched_source).")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed drugs table from training Excel")
    parser.add_argument("--reset", action="store_true", help="Delete existing drugs before import")
    parser.add_argument("--file", type=Path, default=DEFAULT_XLSX, help="Path to Excel workbook")
    args = parser.parse_args()

    if not args.file.exists():
        raise SystemExit(f"Excel file not found: {args.file}")

    merged = load_frames(args.file)
    print(f"Loaded {len(merged)} rows from {args.file.name}")

    db = SessionLocal()
    try:
        seed_drugs(db, merged, reset=args.reset)
    finally:
        db.close()


if __name__ == "__main__":
    main()
