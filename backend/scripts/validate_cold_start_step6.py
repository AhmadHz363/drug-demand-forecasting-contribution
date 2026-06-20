#!/usr/bin/env python3
"""
Step 6 Cold Start — end-to-end integration validation (no mocking).

Runs the full pipeline: seed → train embedder → train MAML → predict scenarios.

Usage (from backend/):
  source bin/activate
  export PYTHONPATH=src
  alembic upgrade head
  python scripts/validate_cold_start_step6.py [--fast]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
_SCRIPTS = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from fastapi.testclient import TestClient
from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

from app.cold_start.autoencoder.embedder import clear_model_cache
from app.cold_start.constants import ARTIFACTS_DIR
from app.cold_start.maml.adapter import clear_maml_cache
from app.core.database import SessionLocal, engine
from app.main import app
from seed_cold_start_e2e import (
    NEW_DRUG_PREDICT_BODY,
    insert_new_drug_observations,
    seed_all,
)

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


def db_available() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError:
        return False


def table_exists(name: str) -> bool:
    return name in inspect(engine).get_table_names()


def apply_fast_training() -> None:
    import app.cold_start.autoencoder.trainer as ae_trainer
    import app.cold_start.maml.trainer as maml_trainer

    ae_trainer.AUTOENCODER_EPOCHS = 20
    maml_trainer.MAML_EPOCHS = 20


def forecast_intervals_valid(forecast: list[dict]) -> bool:
    for day in forecast:
        if day["p10"] < 0:
            return False
        if not (day["p10"] <= day["p50"] <= day["p90"]):
            return False
    return True


def run_validation(*, fast: bool) -> int:
    global passed, failed, skipped
    passed = failed = skipped = 0

    print("Cold Start Step 6 — End-to-End Integration Validation\n")

    if not db_available():
        skip("database connectivity", "PostgreSQL unavailable")
        print(f"\nSummary: {passed} passed, {failed} failed, {skipped} skipped")
        return 1

    if not table_exists("drug_catalog"):
        fail("drug_catalog table", "Run `alembic upgrade head` first")
        print(f"\nSummary: {passed} passed, {failed} failed, {skipped} skipped")
        return 1

    if not table_exists("drug_receipts"):
        fail("drug_receipts table", "Run `alembic upgrade head` first")
        print(f"\nSummary: {passed} passed, {failed} failed, {skipped} skipped")
        return 1

    if fast:
        print("Using --fast training epochs (20) for embedder and MAML.\n")
        apply_fast_training()

    client = TestClient(app)
    db = SessionLocal()

    try:
        print("Step 6.1 — Seed catalog")
        catalog_added, _ = seed_all(db, reset=True)
        catalog_count = db.execute(
            text("SELECT COUNT(*) FROM drug_catalog WHERE drug_code LIKE 'E2E-%'")
        ).scalar()
        check("catalog has >= 8 E2E drugs", int(catalog_count or 0) >= 8, f"got {catalog_count}")

        print("\nStep 6.2 — Seed receipt history")
        receipt_days = db.execute(
            text(
                "SELECT COUNT(DISTINCT receipt_date) FROM drug_receipts "
                "WHERE drug_code LIKE 'E2E-%'"
            )
        ).scalar()
        check(
            "receipts have >= 90 distinct days for E2E drugs",
            int(receipt_days or 0) >= 90,
            f"got {receipt_days} distinct days",
        )

        print("\nStep 6.3 — Train embedder")
        resp = client.post("/cold-start/train-embedder")
        check("train-embedder HTTP 200", resp.status_code == 200, resp.text)
        if resp.status_code == 200:
            body = resp.json()
            check("drugs_trained_on > 0", body.get("drugs_trained_on", 0) > 0)
            artifact = os.path.join(ARTIFACTS_DIR, "autoencoder.pt")
            check("autoencoder.pt exists", os.path.isfile(artifact), artifact)

        print("\nStep 6.4 — Train MAML")
        resp = client.post("/cold-start/train-maml")
        check("train-maml HTTP 200", resp.status_code == 200, resp.text)
        if resp.status_code == 200:
            body = resp.json()
            check("tasks_trained_on > 0", body.get("tasks_trained_on", 0) > 0)
            artifact = os.path.join(ARTIFACTS_DIR, "maml_base.pt")
            check("maml_base.pt exists", os.path.isfile(artifact), artifact)

        print("\nStep 6.5 — Predict new drug (0 observations)")
        resp = client.post("/cold-start/predict", json=NEW_DRUG_PREDICT_BODY)
        check("predict HTTP 200", resp.status_code == 200, resp.text)
        p50_baseline = None
        if resp.status_code == 200:
            body = resp.json()
            check('stage_used == "cold_start_only"', body.get("stage_used") == "cold_start_only")
            check("embedding length == 32", len(body.get("embedding", [])) == 32)
            check(
                "nearest_neighbours count",
                0 < len(body.get("nearest_neighbours", [])) <= 5,
            )
            check(
                "forecast horizon",
                len(body.get("forecast", [])) == NEW_DRUG_PREDICT_BODY["forecast_horizon_days"],
            )
            check("forecast intervals valid", forecast_intervals_valid(body.get("forecast", [])))
            if body.get("forecast"):
                p50_baseline = body["forecast"][0]["p50"]

        print("\nStep 6.6 — Predict with pharmacist estimate")
        body_with_pharmacist = {
            **NEW_DRUG_PREDICT_BODY,
            "pharmacist_estimate": {"weekly_units": 200, "confidence": 0.8},
        }
        resp = client.post("/cold-start/predict", json=body_with_pharmacist)
        check("predict with pharmacist HTTP 200", resp.status_code == 200, resp.text)
        if resp.status_code == 200 and p50_baseline is not None:
            body = resp.json()
            p50_pharmacist = body["forecast"][0]["p50"]
            check(
                "P50 shifts with pharmacist estimate",
                abs(p50_pharmacist - p50_baseline) > 0.01,
                f"baseline={p50_baseline}, with_pharmacist={p50_pharmacist}",
            )

        print("\nStep 6.7 — Graduation logic")
        insert_new_drug_observations(db, 5)
        resp = client.post("/cold-start/predict", json=NEW_DRUG_PREDICT_BODY)
        check("5 observations → blended", resp.status_code == 200 and resp.json().get("stage_used") == "blended", resp.text)

        insert_new_drug_observations(db, 12)
        resp = client.post("/cold-start/predict", json=NEW_DRUG_PREDICT_BODY)
        check(
            "12 observations → full_ensemble",
            resp.status_code == 200 and resp.json().get("stage_used") == "full_ensemble",
            resp.text,
        )
        if resp.status_code == 200:
            note = resp.json().get("uncertainty_note", "")
            check(
                "full_ensemble uncertainty note",
                "Route to full forecasting ensemble" in note,
                note,
            )

        print("\nStep 6 extras — model reload and latency")
        clear_model_cache()
        clear_maml_cache()
        resp = client.post("/cold-start/predict", json=NEW_DRUG_PREDICT_BODY)
        check("predict after cache clear HTTP 200", resp.status_code == 200, resp.text)

        start = time.perf_counter()
        for _ in range(10):
            resp = client.post("/cold-start/predict", json=NEW_DRUG_PREDICT_BODY)
            if resp.status_code != 200:
                break
        elapsed = time.perf_counter() - start
        check(
            "10 sequential predicts under 2s",
            resp.status_code == 200 and elapsed < 2.0,
            f"elapsed={elapsed:.3f}s status={resp.status_code}",
        )

    finally:
        db.close()

    print(f"\nSummary: {passed} passed, {failed} failed, {skipped} skipped")
    return 0 if failed == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Cold Start Step 6 integration.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use reduced training epochs (20) for embedder and MAML",
    )
    args = parser.parse_args()
    return run_validation(fast=args.fast)


if __name__ == "__main__":
    raise SystemExit(main())
