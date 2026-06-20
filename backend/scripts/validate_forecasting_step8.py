#!/usr/bin/env python3
"""
Forecasting Step 8 — end-to-end integration validation (no mocking).

Runs the full pipeline: seed → feature engineering → correction → train → predict.

Usage (from backend/):
  source bin/activate
  export PYTHONPATH=src
  alembic upgrade head
  python scripts/seed_forecasting_e2e.py
  python scripts/validate_forecasting_step8.py [--fast]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from datetime import date, timedelta
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

from app.core.database import SessionLocal, engine
from app.forecasting.censored_demand.corrector import correct_demand
from app.forecasting.feature_engineering.pipeline import build_feature_matrix
from app.forecasting.inference.forecaster import validate_forecast_quantiles
from app.forecasting.schemas import ForecastResponse
from app.main import app
from seed_cold_start_e2e import E2E_PREFIX
from seed_forecasting_e2e import (
    BATCH_DRUGS,
    INVALID_DRUG,
    LONG_DAYS,
    LONG_HISTORY_DRUGS,
    PRIMARY_DRUG,
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
    import app.forecasting.constants as constants
    import app.forecasting.models.lgbm_model as lgbm_model
    import app.forecasting.models.sarima_model as sarima_model
    import app.forecasting.models.tft_model as tft_model

    constants.LGBM_N_ESTIMATORS = 80
    constants.LGBM_EARLY_STOPPING_ROUNDS = 10
    lgbm_model.LGBM_N_ESTIMATORS = 80
    lgbm_model.LGBM_EARLY_STOPPING_ROUNDS = 10
    sarima_model.SARIMA_MAX_ITER = 50
    tft_model.TFT_MAX_EPOCHS = 1


def patch_artifact_dir(path: str) -> None:
    import app.forecasting.constants as constants
    import app.forecasting.ensemble.conformal as conformal
    import app.forecasting.ensemble.stacking as stacking
    import app.forecasting.models.lgbm_model as lgbm_model
    import app.forecasting.models.sarima_model as sarima_model
    import app.forecasting.models.tft_model as tft_model

    constants.ARTIFACTS_DIR = path
    sarima_model.ARTIFACTS_DIR = path
    lgbm_model.ARTIFACTS_DIR = path
    tft_model.ARTIFACTS_DIR = path
    stacking.ARTIFACTS_DIR = path
    conformal.ARTIFACTS_DIR = path
    os.makedirs(path, exist_ok=True)


def run_validation(*, fast: bool) -> int:
    global passed, failed, skipped
    passed = failed = skipped = 0

    print("Forecasting Step 8 — End-to-End Integration Validation\n")

    if not db_available():
        skip("database connectivity", "PostgreSQL unavailable")
        print(f"\nSummary: {passed} passed, {failed} failed, {skipped} skipped")
        return 1

    required = [
        "drug_catalog",
        "daily_drug_demand",
        "forecast_results",
        "model_performance",
        "stockout_flags",
    ]
    for table in required:
        if not table_exists(table):
            fail(f"{table} table", "Run `alembic upgrade head` first")
            print(f"\nSummary: {passed} passed, {failed} failed, {skipped} skipped")
            return 1

    if fast:
        print("Using --fast training settings (reduced epochs/estimators).\n")
        apply_fast_training()

    artifact_dir = tempfile.mkdtemp(prefix="forecasting_step8_")
    patch_artifact_dir(artifact_dir)
    client = TestClient(app)
    db = SessionLocal()

    try:
        print("Step 8.1 — Seed catalog and demand")
        seed_all(db, reset=True)
        catalog_count = db.execute(
            text("SELECT COUNT(*) FROM drug_catalog WHERE drug_code LIKE :prefix"),
            {"prefix": f"{E2E_PREFIX}%"},
        ).scalar()
        check("catalog has >= 8 E2E drugs", int(catalog_count or 0) >= 8, f"got {catalog_count}")

        for drug_code in LONG_HISTORY_DRUGS:
            days = db.execute(
                text("SELECT COUNT(*) FROM daily_drug_demand WHERE drug_code = :code"),
                {"code": drug_code},
            ).scalar()
            check(
                f"{drug_code} has >= {LONG_DAYS} demand days",
                int(days or 0) >= LONG_DAYS,
                f"got {days}",
            )

        print("\nStep 8.2 — Feature engineering")
        end_date = date.today()
        start_date = end_date - timedelta(days=LONG_DAYS - 1)
        feature_df = build_feature_matrix(PRIMARY_DRUG, None, db, start_date, end_date)
        check("feature matrix has no NaN", feature_df.isnull().sum().sum() == 0)
        check("feature matrix has >= 41 columns", len(feature_df.columns) >= 41)

        print("\nStep 8.3 — Censored demand correction")
        corrected_df = correct_demand(PRIMARY_DRUG, None, db, feature_df)
        check("corrected frame has em_corrected_quantity", "em_corrected_quantity" in corrected_df.columns)
        check("corrected frame has no NaN", corrected_df.isnull().sum().sum() == 0)

        print("\nStep 8.4 — Train forecasting models")
        train_models = ["sarima", "lgbm"] if fast else ["sarima", "lgbm", "tft"]
        resp = client.post(
            "/forecasting/train",
            json={
                "drug_codes": list(LONG_HISTORY_DRUGS),
                "models": train_models,
                "force_retrain": True,
            },
        )
        check("train HTTP 200", resp.status_code == 200, resp.text)
        if resp.status_code == 200:
            body = resp.json()
            check('status == "ok"', body.get("status") == "ok")
            check("drugs_trained > 0", body.get("drugs_trained", 0) > 0)
            smape_values = list(body.get("smape_summary", {}).values())
            check(
                "at least one sMAPE < 50",
                any(value < 50.0 for value in smape_values),
                str(body.get("smape_summary")),
            )

        print("\nStep 8.5 — Single drug predict")
        start = time.perf_counter()
        resp = client.post(
            "/forecasting/predict",
            json={
                "drug_code": PRIMARY_DRUG,
                "horizon_days": 7,
                "include_shap": True,
                "include_attention": "tft" in train_models,
            },
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        check("predict HTTP 200", resp.status_code == 200, resp.text)
        if resp.status_code == 200:
            body = resp.json()
            check("forecast horizon == 7", len(body.get("forecast", [])) == 7)
            check("SHAP features present", bool(body.get("shap_features")))
            try:
                validate_forecast_quantiles(ForecastResponse(**body))
                ok("quantile ordering and weights valid")
            except ValueError as exc:
                fail("quantile ordering and weights valid", str(exc))
            check(
                "single predict latency (<15s CI budget)",
                elapsed_ms < 15000.0,
                f"elapsed={elapsed_ms:.1f}ms",
            )

        print("\nStep 8.6 — Batch predict")
        resp = client.post(
            "/forecasting/predict-batch",
            json={"drug_codes": [*BATCH_DRUGS, INVALID_DRUG], "horizon_days": 14},
        )
        check("predict-batch HTTP 200", resp.status_code == 200, resp.text)
        if resp.status_code == 200:
            payload = resp.json()
            check("batch returns 4 entries", len(payload) == 4)
            check("invalid drug has error", payload[-1].get("error") is not None)
            check(
                "valid drugs succeed",
                all(entry.get("error") is None for entry in payload[:3]),
            )

        print("\nStep 8.7 — forecast_results persistence")
        rows = db.execute(
            text(
                "SELECT drug_code, COUNT(*) FROM forecast_results "
                "WHERE drug_code LIKE :prefix GROUP BY drug_code"
            ),
            {"prefix": f"{E2E_PREFIX}%"},
        ).all()
        check("forecast_results populated", len(rows) > 0, "no rows")

        print("\nStep 8.8 — model_performance sMAPE benchmark")
        perf_rows = db.execute(
            text(
                "SELECT drug_code, model_name, smape FROM model_performance "
                "WHERE drug_code LIKE :prefix ORDER BY smape ASC"
            ),
            {"prefix": f"{E2E_PREFIX}%"},
        ).all()
        check("model_performance has rows", len(perf_rows) > 0)
        if perf_rows:
            best = min(float(row[2]) for row in perf_rows)
            check("best sMAPE < 50", best < 50.0, f"best={best}")
            check("best sMAPE < 20 (target)", best < 20.0, f"best={best}")
            model_names = {row[1] for row in perf_rows}
            check("sarima evaluated", "sarima" in model_names)
            check("lgbm evaluated", "lgbm" in model_names)
            if not fast:
                check("tft evaluated", "tft" in model_names)

        print("\nStep 8 extra — reload artifacts via fresh client")
        fresh = TestClient(app)
        resp = fresh.post(
            "/forecasting/predict",
            json={"drug_code": PRIMARY_DRUG, "horizon_days": 7},
        )
        check("predict after fresh client HTTP 200", resp.status_code == 200, resp.text)

    finally:
        db.close()
        shutil.rmtree(artifact_dir, ignore_errors=True)

    print(f"\nSummary: {passed} passed, {failed} failed, {skipped} skipped")
    return 0 if failed == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Forecasting Step 8 integration.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use reduced training settings and skip TFT",
    )
    args = parser.parse_args()
    return run_validation(fast=args.fast)


if __name__ == "__main__":
    raise SystemExit(main())
