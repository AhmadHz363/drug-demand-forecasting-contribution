#!/usr/bin/env python3
"""
Seed drug_catalog from a CSV export.

Usage (from backend/):
  source bin/activate
  export PYTHONPATH=src
  python scripts/seed_drug_catalog.py --csv /path/to/drug_catalog_10k_db_ready.csv
  python scripts/seed_drug_catalog.py --csv /path/to/file.csv --reset
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.models.drug_catalog import DrugCatalog

CATALOG_COLUMNS = (
    "drug_code",
    "drug_name",
    "therapeutic_class",
    "atc_category",
    "pharmaceutical_form",
    "ven_class",
    "abc_class",
    "unit_price_tier",
    "requires_refrigeration",
    "is_controlled_substance",
    "average_shelf_life_days",
    "route_of_administration",
)

BATCH_SIZE = 1000


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid boolean: {value!r}")


def parse_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None

    # Normalize short timezone offsets like +03 -> +03:00 for fromisoformat.
    if len(text) >= 3 and text[-3] in {"+", "-"} and text[-2:].isdigit():
        text = f"{text}:00"

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        for fmt in (
            "%Y-%m-%d %H:%M:%S.%f%z",
            "%Y-%m-%d %H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
        ):
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"invalid timestamp: {value!r}")

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def row_to_payload(row: dict[str, str]) -> dict:
    payload: dict = {
        "drug_code": row["drug_code"].strip(),
        "drug_name": row["drug_name"].strip(),
        "therapeutic_class": row["therapeutic_class"].strip(),
        "atc_category": row["atc_category"].strip(),
        "pharmaceutical_form": row["pharmaceutical_form"].strip(),
        "ven_class": row["ven_class"].strip(),
        "abc_class": row["abc_class"].strip(),
        "unit_price_tier": int(row["unit_price_tier"]),
        "requires_refrigeration": parse_bool(row["requires_refrigeration"]),
        "is_controlled_substance": parse_bool(row["is_controlled_substance"]),
        "average_shelf_life_days": int(row["average_shelf_life_days"]),
        "route_of_administration": row["route_of_administration"].strip(),
    }

    created_at = parse_timestamp(row.get("created_at"))
    if created_at is not None:
        payload["created_at"] = created_at

    updated_at = parse_timestamp(row.get("updated_at"))
    if updated_at is not None:
        payload["updated_at"] = updated_at

    return payload


def load_csv_rows(csv_path: Path) -> tuple[list[dict], int]:
    seen_codes: set[str] = set()
    payloads: list[dict] = []
    skipped_duplicates = 0

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(CATALOG_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        for line_no, row in enumerate(reader, start=2):
            drug_code = row["drug_code"].strip()
            if drug_code in seen_codes:
                skipped_duplicates += 1
                continue
            seen_codes.add(drug_code)

            try:
                payloads.append(row_to_payload(row))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid row at line {line_no}: {exc}") from exc

    return payloads, skipped_duplicates


def reset_catalog(db: Session) -> None:
    db.query(DrugCatalog).delete(synchronize_session=False)
    db.commit()


def seed_catalog_from_csv(
    db: Session,
    *,
    csv_path: Path,
    reset: bool = False,
) -> tuple[int, int, int]:
    payloads, csv_duplicates = load_csv_rows(csv_path)

    if reset:
        reset_catalog(db)
        existing_codes: set[str] = set()
    else:
        existing_codes = {
            code for (code,) in db.query(DrugCatalog.drug_code).all()
        }

    to_insert = [row for row in payloads if row["drug_code"] not in existing_codes]
    skipped_existing = len(payloads) - len(to_insert)

    for offset in range(0, len(to_insert), BATCH_SIZE):
        batch = to_insert[offset : offset + BATCH_SIZE]
        db.bulk_insert_mappings(DrugCatalog, batch)

    db.commit()
    return len(to_insert), skipped_existing, csv_duplicates


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed drug_catalog from CSV.")
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to drug_catalog CSV file",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete all existing drug_catalog rows before seeding",
    )
    args = parser.parse_args()

    if not args.csv.is_file():
        print(f"CSV file not found: {args.csv}", file=sys.stderr)
        return 1

    db = SessionLocal()
    try:
        inserted, skipped_existing, csv_duplicates = seed_catalog_from_csv(
            db,
            csv_path=args.csv,
            reset=args.reset,
        )
        total = db.query(DrugCatalog).count()
        print(
            f"Inserted {inserted} rows "
            f"(skipped {skipped_existing} existing, "
            f"{csv_duplicates} duplicate codes in CSV). "
            f"drug_catalog now has {total} rows."
        )
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
