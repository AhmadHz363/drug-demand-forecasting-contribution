#!/usr/bin/env python3
"""
Run all Step 1 Cold Start validation scenarios.

Uses sample_drug_receipts_100.csv (first 100 DB rows) + live PostgreSQL when available.

Usage (from backend/):
  source bin/activate
  export PYTHONPATH=src
  python scripts/validate_cold_start_step1.py
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

import csv

# Ensure src is on path when run as script
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pydantic import ValidationError

from app.cold_start.constants import ARTIFACTS_DIR, METADATA_INPUT_DIM
from app.cold_start.drug_metadata_encoder import encode_drug_metadata
from app.cold_start.schemas import (
    ColdStartPredictRequest,
    DrugMetadataInput,
    PharmacistEstimate,
)

FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/sample_drug_receipts_100.csv"
CEFTRIAXONE_VECTOR = [0.0, 0.36, 0.1818, 1.0, 1.0, 0.75, 0.0, 0.0, 0.39, 0.125, 1.0, 1.0]

passed = 0
failed = 0
skipped = 0


def ok(name: str) -> None:
    global passed
    passed += 1
    print(f"  PASS  {name}")


def fail(name: str, detail: str) -> None:
    global failed
    failed += 1
    print(f"  FAIL  {name}: {detail}")


def skip(name: str, reason: str) -> None:
    global skipped
    skipped += 1
    print(f"  SKIP  {name}: {reason}")


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        ok(name)
    else:
        fail(name, detail or "assertion failed")


def sample_drug(**overrides) -> DrugMetadataInput:
    base = dict(
        drug_code="NEW-001",
        drug_name="Ceftriaxone 1g",
        therapeutic_class="antibiotic",
        atc_category="J01",
        pharmaceutical_form="injection",
        ven_class="V",
        abc_class="A",
        unit_price_tier=4,
        requires_refrigeration=False,
        is_controlled_substance=False,
        average_shelf_life_days=730,
        route_of_administration="iv",
    )
    base.update(overrides)
    return DrugMetadataInput(**base)


def db_available() -> bool:
    from sqlalchemy import text
    from sqlalchemy.exc import SQLAlchemyError

    from app.core.database import engine

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError:
        return False


def _load_csv_rows() -> list[dict[str, str]]:
    with FIXTURE.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _run_db_checks(rows: list[dict[str, str]] | None) -> None:
    from sqlalchemy import inspect, text

    from app.cold_start.catalog_adapter import drug_catalog_to_metadata
    from app.core.database import SessionLocal, engine
    from app.models.drug_catalog import DrugCatalog
    from app.models.drug_receipt import DrugReceipt

    with engine.connect() as conn:
        version = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
        check("alembic at 20260505_0002", version == "20260505_0002", str(version))

        cols = {c["name"] for c in inspect(engine).get_columns("drug_catalog")}
        expected = {
            "id", "drug_code", "drug_name", "therapeutic_class", "atc_category",
            "pharmaceutical_form", "ven_class", "abc_class", "unit_price_tier",
            "requires_refrigeration", "is_controlled_substance", "average_shelf_life_days",
            "route_of_administration", "created_at", "updated_at",
        }
        check("drug_catalog 15 columns", cols == expected, f"missing {expected - cols}")

        n = conn.execute(text("SELECT COUNT(*) FROM drug_receipts")).scalar()
        check("drug_receipts count >= 100", n >= 100, str(n))

    if not rows:
        return

    row0 = rows[0]
    db = SessionLocal()
    try:
        rec = db.get(DrugReceipt, int(row0["id"]))
        match = (
            rec is not None
            and rec.drug_code == str(row0["drug_code"])
            and rec.receipt_date == date.fromisoformat(str(row0["receipt_date"]))
        )
        check("csv first row matches DB", match)

        codes = list({r["drug_code"] for r in rows})[:50]
        found = {
            r[0]
            for r in db.query(DrugReceipt.drug_code)
            .filter(DrugReceipt.drug_code.in_(codes))
            .distinct()
            .all()
        }
        check("sample drug codes exist in DB (subset 50)", len(found) > 0)

        test_code = "TEST-VALIDATE-STEP1"
        meta = sample_drug(drug_code=test_code, drug_name="Validation Drug")
        cat = DrugCatalog(
            drug_code=test_code,
            drug_name=meta.drug_name,
            therapeutic_class=meta.therapeutic_class,
            atc_category=meta.atc_category,
            pharmaceutical_form=meta.pharmaceutical_form,
            ven_class=meta.ven_class,
            abc_class=meta.abc_class,
            unit_price_tier=meta.unit_price_tier,
            requires_refrigeration=meta.requires_refrigeration,
            is_controlled_substance=meta.is_controlled_substance,
            average_shelf_life_days=meta.average_shelf_life_days,
            route_of_administration=meta.route_of_administration,
        )
        db.add(cat)
        db.flush()
        enc = encode_drug_metadata(drug_catalog_to_metadata(cat))
        db.rollback()
        check("catalog adapter round-trip", len(enc) == 12 and all(0 <= x <= 1 for x in enc))
    finally:
        db.close()


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Validate Cold Start Step 1")
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Skip database integration checks (encoder/CSV only)",
    )
    args = parser.parse_args()

    print("Cold Start Step 1 — validation report")
    print("=" * 50)

    # --- CSV fixture ---
    print("\n[1] Sample CSV (100 receipt rows)")
    check("fixture exists", FIXTURE.is_file())
    rows: list[dict[str, str]] | None = None
    if FIXTURE.is_file():
        rows = _load_csv_rows()
        check("csv has 100 rows", len(rows) == 100, f"got {len(rows)}")
        check(
            "required columns",
            {"drug_code", "receipt_date", "receipt_id"}.issubset(rows[0].keys()),
        )
        check("first row id=1 P325096", rows[0]["drug_code"] == "P325096")
        unique_codes = {r["drug_code"] for r in rows}
        check("multiple unique drug codes", len(unique_codes) >= 10, f"got {len(unique_codes)}")
    else:
        skip("csv content checks", "fixture missing")

    # --- Encoder ---
    print("\n[2] Metadata encoder")
    vec = encode_drug_metadata(sample_drug())
    check("vector length 12", len(vec) == 12)
    check("values in [0,1]", all(0 <= v <= 1 for v in vec))
    check("ceftriaxone reference vector", [round(v, 4) for v in vec] == CEFTRIAXONE_VECTOR)
    check("deterministic", encode_drug_metadata(sample_drug()) == vec)
    unknown = encode_drug_metadata(sample_drug(therapeutic_class="exotic_xyz"))
    check("unknown therapeutic_class", len(unknown) == 12 and all(0 <= v <= 1 for v in unknown))
    check("unknown form/route", len(encode_drug_metadata(sample_drug(pharmaceutical_form="x", route_of_administration="y"))) == 12)

    if rows is not None:
        bad = []
        seen: set[str] = set()
        for row in rows:
            if row["drug_code"] in seen:
                continue
            seen.add(row["drug_code"])
            meta = sample_drug(
                drug_code=str(row["drug_code"]),
                drug_name=str(row["drug_name"])[:512],
                therapeutic_class="other",
                atc_category="A01",
                pharmaceutical_form="tablet",
                route_of_administration="oral",
            )
            v = encode_drug_metadata(meta)
            if len(v) != 12 or not all(0 <= x <= 1 for x in v):
                bad.append(row["drug_code"])
        check("encode all sample drug codes", len(bad) == 0, f"failed: {bad[:5]}")

    # --- Pydantic ---
    print("\n[3] Pydantic schemas")
    try:
        ColdStartPredictRequest(
            drug_metadata=sample_drug(),
            pharmacist_estimate=PharmacistEstimate(weekly_units=200, confidence=0.8),
        )
        ok("ColdStartPredictRequest valid")
    except Exception as exc:
        fail("ColdStartPredictRequest valid", str(exc))

    try:
        DrugMetadataInput(**{**sample_drug().model_dump(), "unit_price_tier": 0})
        fail("reject invalid tier", "expected ValidationError")
    except ValidationError:
        ok("reject invalid unit_price_tier")

    # --- Constants ---
    print("\n[4] Constants & artifacts")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    check("ARTIFACTS_DIR exists", os.path.isdir(ARTIFACTS_DIR))
    check("METADATA_INPUT_DIM", METADATA_INPUT_DIM == 12)

    # --- Database ---
    print("\n[5] Database integration")
    if args.no_db:
        skip("all DB checks", "--no-db")
    elif not db_available():
        skip("all DB checks", "DB unavailable")
    else:
        _run_db_checks(rows)

    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
