"""
Step 8 Forecasting — end-to-end integration tests (real DB, no mocks).

Skipped when PostgreSQL or forecasting tables are unavailable.
Uses reduced training settings via fixtures for CI-friendly runtime.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

from app.core.database import SessionLocal, engine
from app.forecasting.censored_demand.corrector import correct_demand
from app.forecasting.feature_engineering.pipeline import build_feature_matrix
from app.forecasting.inference.forecaster import validate_forecast_quantiles
from app.forecasting.schemas import ForecastResponse
from app.main import app

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from seed_forecasting_e2e import (  # noqa: E402
    BATCH_DRUGS,
    INVALID_DRUG,
    LONG_DAYS,
    LONG_HISTORY_DRUGS,
    PRIMARY_DRUG,
    seed_all,
)
from seed_cold_start_e2e import E2E_PREFIX  # noqa: E402


def _db_available() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError:
        return False


def _required_tables() -> bool:
    names = set(inspect(engine).get_table_names())
    required = {
        "drug_catalog",
        "daily_drug_demand",
        "forecast_results",
        "model_performance",
        "stockout_flags",
    }
    return required.issubset(names)


pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        not _db_available() or not _required_tables(),
        reason="PostgreSQL with forecasting tables required",
    ),
]


@pytest.fixture(scope="module")
def monkeypatch_module(request):
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    request.addfinalizer(mp.undo)
    return mp


@pytest.fixture(scope="module")
def e2e_artifacts_dir(tmp_path_factory) -> Iterator[str]:
    path = str(tmp_path_factory.mktemp("forecasting_artifacts"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="module", autouse=True)
def patch_artifacts_dir(e2e_artifacts_dir, monkeypatch_module):
    monkeypatch_module.setattr("app.forecasting.constants.ARTIFACTS_DIR", e2e_artifacts_dir)
    monkeypatch_module.setattr(
        "app.forecasting.models.sarima_model.ARTIFACTS_DIR", e2e_artifacts_dir
    )
    monkeypatch_module.setattr(
        "app.forecasting.models.lgbm_model.ARTIFACTS_DIR", e2e_artifacts_dir
    )
    monkeypatch_module.setattr(
        "app.forecasting.models.tft_model.ARTIFACTS_DIR", e2e_artifacts_dir
    )
    monkeypatch_module.setattr(
        "app.forecasting.constants.ARTIFACTS_DIR", e2e_artifacts_dir
    )
    monkeypatch_module.setattr(
        "app.forecasting.ensemble.segment_artifacts.ARTIFACTS_DIR", e2e_artifacts_dir
    )
    os.makedirs(e2e_artifacts_dir, exist_ok=True)


@pytest.fixture(scope="module", autouse=True)
def fast_forecasting_training(monkeypatch_module):
    import app.forecasting.constants as constants
    import app.forecasting.models.lgbm_model as lgbm_model
    import app.forecasting.models.sarima_model as sarima_model
    import app.forecasting.models.tft_model as tft_model

    monkeypatch_module.setattr(constants, "LGBM_N_ESTIMATORS", 80)
    monkeypatch_module.setattr(constants, "LGBM_EARLY_STOPPING_ROUNDS", 10)
    monkeypatch_module.setattr(lgbm_model, "LGBM_N_ESTIMATORS", 80)
    monkeypatch_module.setattr(lgbm_model, "LGBM_EARLY_STOPPING_ROUNDS", 10)
    monkeypatch_module.setattr(sarima_model, "SARIMA_MAX_ITER", 50)
    monkeypatch_module.setattr(tft_model, "TFT_MAX_EPOCHS", 1)


@pytest.fixture(scope="module")
def seeded_db():
    db = SessionLocal()
    seed_all(db, reset=True)
    db.close()


@pytest.fixture(scope="module")
def trained_client(seeded_db):
    client = TestClient(app)
    resp = client.post(
        "/forecasting/train",
        json={
            "drug_codes": list(LONG_HISTORY_DRUGS),
            "models": ["sarima", "lgbm"],
            "force_retrain": True,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "ok"
    assert body["drugs_trained"] > 0
    return client


class TestForecastingStep8Integration:
    def test_prerequisites(self, seeded_db):
        db = SessionLocal()
        try:
            catalog = db.execute(
                text("SELECT COUNT(*) FROM drug_catalog WHERE drug_code LIKE :prefix"),
                {"prefix": f"{E2E_PREFIX}%"},
            ).scalar()
            assert int(catalog or 0) >= 8

            for drug_code in LONG_HISTORY_DRUGS:
                receipt_days = db.execute(
                    text(
                        "SELECT COUNT(DISTINCT receipt_date) FROM drug_receipts "
                        "WHERE drug_code = :code"
                    ),
                    {"code": drug_code},
                ).scalar()
                assert int(receipt_days or 0) >= LONG_DAYS

                days = db.execute(
                    text("SELECT COUNT(*) FROM daily_drug_demand WHERE drug_code = :code"),
                    {"code": drug_code},
                ).scalar()
                assert int(days or 0) >= LONG_DAYS

            for drug_code in [
                f"{E2E_PREFIX}WARF",
                f"{E2E_PREFIX}SALB",
                f"{E2E_PREFIX}OMEP",
                f"{E2E_PREFIX}FURO",
                f"{E2E_PREFIX}DEXA",
            ]:
                days = db.execute(
                    text("SELECT COUNT(*) FROM daily_drug_demand WHERE drug_code = :code"),
                    {"code": drug_code},
                ).scalar()
                assert int(days or 0) >= 90
        finally:
            db.close()

    def test_feature_engineering_pipeline(self, seeded_db):
        db = SessionLocal()
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=LONG_DAYS - 1)
            df = build_feature_matrix(
                PRIMARY_DRUG,
                None,
                db,
                start_date,
                end_date,
            )
            assert df.isnull().sum().sum() == 0
            assert len(df.columns) >= 41
        finally:
            db.close()

    def test_censored_demand_correction(self, seeded_db):
        db = SessionLocal()
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=LONG_DAYS - 1)
            feature_df = build_feature_matrix(
                PRIMARY_DRUG,
                None,
                db,
                start_date,
                end_date,
            )
            corrected_df = correct_demand(PRIMARY_DRUG, None, db, feature_df)
            assert "em_corrected_quantity" in corrected_df.columns
            assert corrected_df.isnull().sum().sum() == 0
        finally:
            db.close()

    def test_train_endpoint(self, trained_client: TestClient):
        resp = trained_client.post(
            "/forecasting/train",
            json={
                "drug_codes": [PRIMARY_DRUG],
                "models": ["sarima"],
                "force_retrain": False,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        smape_values = list(body.get("smape_summary", {}).values())
        assert not smape_values or any(value < 50.0 for value in smape_values)

    def test_predict_single_drug(self, trained_client: TestClient):
        resp = trained_client.post(
            "/forecasting/predict",
            json={
                "drug_code": PRIMARY_DRUG,
                "horizon_days": 7,
                "include_shap": True,
                "include_attention": False,
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert len(body["forecast"]) == 7
        assert body["shap_features"]
        assert len(body["shap_features"]) <= 15
        weights = body["model_weights"]
        assert weights["sarima"] + weights["lgbm"] + weights["tft"] == pytest.approx(1.0)
        validate_forecast_quantiles(ForecastResponse(**body))

    def test_predict_batch_mixed_results(self, trained_client: TestClient):
        resp = trained_client.post(
            "/forecasting/predict-batch",
            json={
                "drug_codes": [*BATCH_DRUGS, INVALID_DRUG],
                "horizon_days": 14,
            },
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert len(payload) == 4
        assert payload[-1]["drug_code"] == INVALID_DRUG
        assert payload[-1]["error"] is not None
        for entry in payload[:3]:
            assert entry["error"] is None
            assert len(entry["forecast"]) == 14

    def test_forecast_results_persisted(self, trained_client: TestClient):
        db = SessionLocal()
        try:
            rows = db.execute(
                text(
                    "SELECT drug_code, COUNT(*) AS n "
                    "FROM forecast_results "
                    "WHERE drug_code LIKE :prefix "
                    "GROUP BY drug_code"
                ),
                {"prefix": f"{E2E_PREFIX}%"},
            ).all()
            assert rows
            for _drug_code, count in rows:
                assert int(count) > 0
        finally:
            db.close()

    def test_model_performance_smape_benchmark(self, trained_client: TestClient):
        db = SessionLocal()
        try:
            rows = db.execute(
                text(
                    "SELECT drug_code, model_name, smape "
                    "FROM model_performance "
                    "WHERE drug_code LIKE :prefix "
                    "ORDER BY smape ASC"
                ),
                {"prefix": f"{E2E_PREFIX}%"},
            ).all()
            assert rows
            best_smape = min(float(row[2]) for row in rows)
            assert best_smape < 50.0
            model_names = {row[1] for row in rows}
            assert "sarima" in model_names
            assert "lgbm" in model_names
        finally:
            db.close()

    def test_predict_after_fresh_client(self, trained_client: TestClient, e2e_artifacts_dir):
        fresh_client = TestClient(app)
        resp = fresh_client.post(
            "/forecasting/predict",
            json={"drug_code": PRIMARY_DRUG, "horizon_days": 7},
        )
        assert resp.status_code == 200, resp.text
        assert os.path.isdir(os.path.join(e2e_artifacts_dir, "sarima"))

    def test_single_predict_latency(self, trained_client: TestClient):
        start = time.perf_counter()
        resp = trained_client.post(
            "/forecasting/predict",
            json={
                "drug_code": PRIMARY_DRUG,
                "horizon_days": 7,
                "include_attention": False,
            },
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert resp.status_code == 200, resp.text
        assert elapsed_ms < 15000.0
