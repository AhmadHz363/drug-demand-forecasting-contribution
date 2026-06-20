#!/usr/bin/env python3
"""
Seed data for Forecasting Step 8 end-to-end integration tests.

Receipt lines are inserted into ``drug_receipts`` (source of truth), then
aggregated into ``daily_drug_demand``:
- 400 days for three primary drugs (SARIMA/LGBM/TFT eligible)
- 120 days for the remaining catalog drugs (>= 90 required)

Usage (from backend/):
  source bin/activate
  export PYTHONPATH=src
  python scripts/seed_forecasting_e2e.py [--reset]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
_SCRIPTS = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.models.daily_drug_demand import DailyDrugDemand
from app.models.drug_receipt import DrugReceipt
from app.models.forecast_result import ForecastResult
from app.models.model_performance import ModelPerformance
from app.models.stockout_flag import StockoutFlag
from app.services.demand_aggregation import sync_daily_demand_from_receipts
from app.services.drug_registry import insert_receipt_rows
from seed_cold_start_e2e import (
    CATALOG_DRUGS,
    E2E_PREFIX,
    clear_e2e_data,
    e2e_drug_codes,
    generate_demand_series,
    seed_catalog,
)

LONG_HISTORY_DRUGS = [f"{E2E_PREFIX}AMOX", f"{E2E_PREFIX}PARA", f"{E2E_PREFIX}METF"]
LONG_DAYS = 400
DEFAULT_DAYS = 120

PRIMARY_DRUG = f"{E2E_PREFIX}AMOX"
BATCH_DRUGS = [f"{E2E_PREFIX}AMOX", f"{E2E_PREFIX}PARA", f"{E2E_PREFIX}METF"]
INVALID_DRUG = "INVALID-CODE"

DEMAND_BASES = {
    f"{E2E_PREFIX}AMOX": 55.0,
    f"{E2E_PREFIX}PARA": 130.0,
    f"{E2E_PREFIX}METF": 85.0,
    f"{E2E_PREFIX}WARF": 28.0,
    f"{E2E_PREFIX}SALB": 38.0,
    f"{E2E_PREFIX}OMEP": 62.0,
    f"{E2E_PREFIX}FURO": 42.0,
    f"{E2E_PREFIX}DEXA": 32.0,
}


def clear_forecasting_outputs(db: Session, drug_codes: list[str]) -> None:
    db.query(ForecastResult).filter(ForecastResult.drug_code.in_(drug_codes)).delete(
        synchronize_session=False
    )
    db.query(ModelPerformance).filter(ModelPerformance.drug_code.in_(drug_codes)).delete(
        synchronize_session=False
    )
    db.query(StockoutFlag).filter(StockoutFlag.drug_code.in_(drug_codes)).delete(
        synchronize_session=False
    )
    db.commit()


def seed_receipt_history(db: Session) -> int:
    codes = e2e_drug_codes()
    db.query(DrugReceipt).filter(DrugReceipt.drug_code.in_(codes)).delete(
        synchronize_session=False
    )
    db.query(DailyDrugDemand).filter(DailyDrugDemand.drug_code.in_(codes)).delete(
        synchronize_session=False
    )

    receipt_rows: list[dict] = []
    for payload in CATALOG_DRUGS:
        drug_code = payload["drug_code"]
        days = LONG_DAYS if drug_code in LONG_HISTORY_DRUGS else DEFAULT_DAYS
        base = DEMAND_BASES.get(drug_code, 50.0)
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


def seed_all(db: Session, *, reset: bool = True) -> tuple[int, int]:
    codes = e2e_drug_codes()
    if reset:
        clear_forecasting_outputs(db, codes)
        clear_e2e_data(db)
    catalog_added = seed_catalog(db)
    demand_rows = seed_receipt_history(db)
    return catalog_added, demand_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed Forecasting Step 8 integration data.")
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
            f"({catalog_added} newly inserted), {demand_rows} daily demand rows "
            f"materialized from drug_receipts."
        )
        print(f"Primary drug: {PRIMARY_DRUG}")
        print(f"Long-history drugs ({LONG_DAYS}d): {', '.join(LONG_HISTORY_DRUGS)}")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
