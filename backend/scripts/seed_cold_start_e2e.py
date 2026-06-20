#!/usr/bin/env python3
"""
Seed drug_catalog and drug_receipts for Cold Start Step 6 integration tests.

Usage (from backend/):
  source bin/activate
  export PYTHONPATH=src
  python scripts/seed_cold_start_e2e.py [--reset]
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import date, timedelta
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.models.daily_drug_demand import DailyDrugDemand
from app.models.drug import Drug
from app.models.drug_catalog import DrugCatalog
from app.models.drug_receipt import DrugReceipt
from app.services.demand_aggregation import sync_daily_demand_from_receipts
from app.services.drug_registry import insert_receipt_rows

E2E_PREFIX = "E2E-"
NEW_DRUG_CODE = "NEW-001"
DEMAND_DAYS = 90

CATALOG_DRUGS: list[dict] = [
    {
        "drug_code": f"{E2E_PREFIX}AMOX",
        "drug_name": "Amoxicillin 500mg Capsule",
        "therapeutic_class": "antibiotic",
        "atc_category": "J01",
        "pharmaceutical_form": "capsule",
        "ven_class": "V",
        "abc_class": "A",
        "unit_price_tier": 2,
        "requires_refrigeration": False,
        "is_controlled_substance": False,
        "average_shelf_life_days": 730,
        "route_of_administration": "oral",
    },
    {
        "drug_code": f"{E2E_PREFIX}PARA",
        "drug_name": "Paracetamol 500mg Tablet",
        "therapeutic_class": "analgesic",
        "atc_category": "N02",
        "pharmaceutical_form": "tablet",
        "ven_class": "N",
        "abc_class": "C",
        "unit_price_tier": 1,
        "requires_refrigeration": False,
        "is_controlled_substance": False,
        "average_shelf_life_days": 1095,
        "route_of_administration": "oral",
    },
    {
        "drug_code": f"{E2E_PREFIX}METF",
        "drug_name": "Metformin 850mg Tablet",
        "therapeutic_class": "antidiabetic",
        "atc_category": "A10",
        "pharmaceutical_form": "tablet",
        "ven_class": "E",
        "abc_class": "B",
        "unit_price_tier": 2,
        "requires_refrigeration": False,
        "is_controlled_substance": False,
        "average_shelf_life_days": 730,
        "route_of_administration": "oral",
    },
    {
        "drug_code": f"{E2E_PREFIX}WARF",
        "drug_name": "Warfarin 5mg Tablet",
        "therapeutic_class": "anticoagulant",
        "atc_category": "B01",
        "pharmaceutical_form": "tablet",
        "ven_class": "V",
        "abc_class": "A",
        "unit_price_tier": 3,
        "requires_refrigeration": False,
        "is_controlled_substance": True,
        "average_shelf_life_days": 730,
        "route_of_administration": "oral",
    },
    {
        "drug_code": f"{E2E_PREFIX}SALB",
        "drug_name": "Salbutamol 100mcg Inhaler",
        "therapeutic_class": "bronchodilator",
        "atc_category": "R03",
        "pharmaceutical_form": "inhaler",
        "ven_class": "V",
        "abc_class": "A",
        "unit_price_tier": 3,
        "requires_refrigeration": False,
        "is_controlled_substance": False,
        "average_shelf_life_days": 365,
        "route_of_administration": "inhalation",
    },
    {
        "drug_code": f"{E2E_PREFIX}OMEP",
        "drug_name": "Omeprazole 20mg Capsule",
        "therapeutic_class": "other",
        "atc_category": "A02",
        "pharmaceutical_form": "capsule",
        "ven_class": "E",
        "abc_class": "B",
        "unit_price_tier": 2,
        "requires_refrigeration": False,
        "is_controlled_substance": False,
        "average_shelf_life_days": 730,
        "route_of_administration": "oral",
    },
    {
        "drug_code": f"{E2E_PREFIX}FURO",
        "drug_name": "Furosemide 40mg Tablet",
        "therapeutic_class": "diuretic",
        "atc_category": "C03",
        "pharmaceutical_form": "tablet",
        "ven_class": "V",
        "abc_class": "A",
        "unit_price_tier": 2,
        "requires_refrigeration": False,
        "is_controlled_substance": False,
        "average_shelf_life_days": 1095,
        "route_of_administration": "oral",
    },
    {
        "drug_code": f"{E2E_PREFIX}DEXA",
        "drug_name": "Dexamethasone 4mg Injection",
        "therapeutic_class": "corticosteroid",
        "atc_category": "H02",
        "pharmaceutical_form": "injection",
        "ven_class": "V",
        "abc_class": "A",
        "unit_price_tier": 4,
        "requires_refrigeration": False,
        "is_controlled_substance": False,
        "average_shelf_life_days": 730,
        "route_of_administration": "iv",
    },
]

NEW_DRUG_PREDICT_BODY: dict = {
    "drug_metadata": {
        "drug_code": NEW_DRUG_CODE,
        "drug_name": "Ceftriaxone 1g Injection",
        "therapeutic_class": "antibiotic",
        "atc_category": "J01",
        "pharmaceutical_form": "injection",
        "ven_class": "V",
        "abc_class": "A",
        "unit_price_tier": 4,
        "requires_refrigeration": False,
        "is_controlled_substance": False,
        "average_shelf_life_days": 730,
        "route_of_administration": "iv",
    },
    "forecast_horizon_days": 7,
}


def e2e_drug_codes() -> list[str]:
    return [drug["drug_code"] for drug in CATALOG_DRUGS]


def generate_demand_series(
    drug_code: str,
    *,
    days: int = DEMAND_DAYS,
    base: float = 50.0,
    amplitude: float = 15.0,
) -> list[tuple[date, float]]:
    """Synthetic weekly sinusoid + deterministic noise per drug."""
    seed = sum(ord(c) for c in drug_code)
    start = date.today() - timedelta(days=days - 1)
    series: list[tuple[date, float]] = []

    for offset in range(days):
        demand_date = start + timedelta(days=offset)
        seasonal = base + amplitude * math.sin(2 * math.pi * offset / 7)
        pseudo_noise = ((seed * (offset + 1) * 9301 + 49297) % 233280) / 233280.0
        noise = (pseudo_noise - 0.5) * 6.0
        quantity = max(1.0, seasonal + noise)
        series.append((demand_date, round(quantity, 2)))

    return series


def clear_e2e_data(db: Session) -> None:
    """Remove integration-test rows (catalog, receipts, demand for E2E drugs and NEW-001)."""
    codes = e2e_drug_codes() + [NEW_DRUG_CODE]
    db.query(DailyDrugDemand).filter(DailyDrugDemand.drug_code.in_(codes)).delete(
        synchronize_session=False
    )
    db.query(DrugReceipt).filter(DrugReceipt.drug_code.in_(codes)).delete(
        synchronize_session=False
    )
    db.query(Drug).filter(Drug.drug_code.in_(codes)).delete(synchronize_session=False)
    db.query(DrugCatalog).filter(DrugCatalog.drug_code.like(f"{E2E_PREFIX}%")).delete(
        synchronize_session=False
    )
    db.commit()


def seed_catalog(db: Session) -> int:
    inserted = 0
    for payload in CATALOG_DRUGS:
        existing = (
            db.query(DrugCatalog)
            .filter(DrugCatalog.drug_code == payload["drug_code"])
            .one_or_none()
        )
        if existing is None:
            db.add(DrugCatalog(**payload))
            inserted += 1
    db.commit()
    return inserted


def seed_demand(db: Session, *, days: int = DEMAND_DAYS) -> int:
    codes = e2e_drug_codes()
    db.query(DrugReceipt).filter(DrugReceipt.drug_code.in_(codes)).delete(
        synchronize_session=False
    )
    db.query(DailyDrugDemand).filter(DailyDrugDemand.drug_code.in_(codes)).delete(
        synchronize_session=False
    )

    receipt_rows: list[dict] = []
    bases = [45, 120, 80, 25, 35, 60, 40, 30]

    for payload, base in zip(CATALOG_DRUGS, bases):
        drug_code = payload["drug_code"]
        for demand_date, quantity in generate_demand_series(
            drug_code,
            days=days,
            base=base,
            amplitude=base * 0.25,
        ):
            receipt_rows.append(
                {
                    "receipt_id": f"{drug_code}-{demand_date.isoformat()}",
                    "drug_code": drug_code,
                    "receipt_date": demand_date,
                    "quantity": quantity,
                }
            )

    insert_receipt_rows(db, receipt_rows)
    db.commit()
    synced = sync_daily_demand_from_receipts(db, drug_codes=codes)
    db.commit()
    return synced


def insert_new_drug_observations(db: Session, count: int) -> None:
    """Insert `count` daily receipt rows for NEW-001 (most recent calendar days)."""
    db.query(DailyDrugDemand).filter(DailyDrugDemand.drug_code == NEW_DRUG_CODE).delete(
        synchronize_session=False
    )
    db.query(DrugReceipt).filter(DrugReceipt.drug_code == NEW_DRUG_CODE).delete(
        synchronize_session=False
    )
    db.query(Drug).filter(Drug.drug_code == NEW_DRUG_CODE).delete(synchronize_session=False)
    start = date.today() - timedelta(days=count - 1)
    receipt_rows = [
        {
            "receipt_id": f"{NEW_DRUG_CODE}-{start + timedelta(days=offset)}",
            "drug_code": NEW_DRUG_CODE,
            "receipt_date": start + timedelta(days=offset),
            "quantity": round(20.0 + offset * 1.5, 2),
        }
        for offset in range(count)
    ]
    insert_receipt_rows(db, receipt_rows)
    db.commit()
    sync_daily_demand_from_receipts(db, drug_codes=[NEW_DRUG_CODE])
    db.commit()


def seed_all(db: Session, *, reset: bool = True) -> tuple[int, int]:
    if reset:
        clear_e2e_data(db)
    catalog_count = seed_catalog(db)
    demand_count = seed_demand(db)
    return catalog_count, demand_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed Cold Start Step 6 integration data.")
    parser.add_argument(
        "--reset",
        action="store_true",
        default=True,
        help="Clear existing E2E rows before seeding (default: true)",
    )
    parser.add_argument(
        "--no-reset",
        action="store_false",
        dest="reset",
        help="Keep existing E2E rows and only add missing catalog entries",
    )
    args = parser.parse_args()

    db = SessionLocal()
    try:
        catalog_added, demand_rows = seed_all(db, reset=args.reset)
        print(
            f"Seeded {len(CATALOG_DRUGS)} catalog drugs "
            f"({catalog_added} newly inserted), {demand_rows} demand rows."
        )
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
