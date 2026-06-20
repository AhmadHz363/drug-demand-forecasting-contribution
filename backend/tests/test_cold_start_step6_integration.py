"""
Step 6 Cold Start — end-to-end integration tests (real DB, no mocks).

Skipped when PostgreSQL is unavailable. Uses --fast training epochs via fixture.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

from app.cold_start.autoencoder.embedder import clear_model_cache
from app.cold_start.constants import ARTIFACTS_DIR
from app.cold_start.maml.adapter import clear_maml_cache
from app.core.database import SessionLocal, engine
from app.main import app

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from seed_cold_start_e2e import (  # noqa: E402
    NEW_DRUG_PREDICT_BODY,
    insert_new_drug_observations,
    seed_all,
)


def _db_available() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError:
        return False


def _required_tables() -> bool:
    names = set(inspect(engine).get_table_names())
    return "drug_receipts" in names


pytestmark = pytest.mark.skipif(
    not _db_available() or not _required_tables(),
    reason="PostgreSQL with drug_receipts required",
)


@pytest.fixture(scope="module", autouse=True)
def fast_training():
    """Reduce training time for integration tests."""
    import app.cold_start.autoencoder.trainer as ae_trainer
    import app.cold_start.maml.trainer as maml_trainer

    old_ae = ae_trainer.AUTOENCODER_EPOCHS
    old_maml = maml_trainer.MAML_EPOCHS
    ae_trainer.AUTOENCODER_EPOCHS = 20
    maml_trainer.MAML_EPOCHS = 20
    yield
    ae_trainer.AUTOENCODER_EPOCHS = old_ae
    maml_trainer.MAML_EPOCHS = old_maml


@pytest.fixture(scope="module")
def e2e_client(fast_training):
    db = SessionLocal()
    seed_all(db, reset=True)
    db.close()

    client = TestClient(app)
    embedder = client.post("/cold-start/train-embedder")
    assert embedder.status_code == 200, embedder.text
    maml = client.post("/cold-start/train-maml")
    assert maml.status_code == 200, maml.text
    return client


class TestColdStartStep6Integration:
    def test_seed_data_counts(self):
        db = SessionLocal()
        try:
            catalog = db.execute(
                text("SELECT COUNT(*) FROM drug_catalog WHERE drug_code LIKE 'E2E-%'")
            ).scalar()
            receipts = db.execute(
                text(
                    "SELECT COUNT(DISTINCT receipt_date) FROM drug_receipts "
                    "WHERE drug_code LIKE 'E2E-%'"
                )
            ).scalar()
            assert int(catalog or 0) >= 8
            assert int(receipts or 0) >= 90
        finally:
            db.close()

    def test_predict_cold_start_only(self, e2e_client: TestClient):
        db = SessionLocal()
        try:
            db.execute(
                text("DELETE FROM drug_receipts WHERE drug_code = 'NEW-001'")
            )
            db.commit()
        finally:
            db.close()

        resp = e2e_client.post("/cold-start/predict", json=NEW_DRUG_PREDICT_BODY)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["stage_used"] == "cold_start_only"
        assert len(body["embedding"]) == 32
        assert 0 < len(body["nearest_neighbours"]) <= 5
        assert len(body["forecast"]) == 7
        for day in body["forecast"]:
            assert day["p10"] >= 0
            assert day["p10"] <= day["p50"] <= day["p90"]

    def test_pharmacist_estimate_shifts_p50(self, e2e_client: TestClient):
        baseline = e2e_client.post("/cold-start/predict", json=NEW_DRUG_PREDICT_BODY)
        assert baseline.status_code == 200
        with_pharmacist = e2e_client.post(
            "/cold-start/predict",
            json={
                **NEW_DRUG_PREDICT_BODY,
                "pharmacist_estimate": {"weekly_units": 200, "confidence": 0.8},
            },
        )
        assert with_pharmacist.status_code == 200
        p50_base = baseline.json()["forecast"][0]["p50"]
        p50_pharm = with_pharmacist.json()["forecast"][0]["p50"]
        assert abs(p50_pharm - p50_base) > 0.01

    def test_graduation_transitions(self, e2e_client: TestClient):
        db = SessionLocal()
        try:
            insert_new_drug_observations(db, 5)
            resp = e2e_client.post("/cold-start/predict", json=NEW_DRUG_PREDICT_BODY)
            assert resp.status_code == 200
            assert resp.json()["stage_used"] == "blended"

            insert_new_drug_observations(db, 12)
            resp = e2e_client.post("/cold-start/predict", json=NEW_DRUG_PREDICT_BODY)
            assert resp.status_code == 200
            body = resp.json()
            assert body["stage_used"] == "full_ensemble"
            assert "Route to full forecasting ensemble" in body["uncertainty_note"]
        finally:
            db.close()

    def test_artifacts_exist_after_training(self):
        assert os.path.isfile(os.path.join(ARTIFACTS_DIR, "autoencoder.pt"))
        assert os.path.isfile(os.path.join(ARTIFACTS_DIR, "maml_base.pt"))

    def test_predict_after_cache_clear(self, e2e_client: TestClient):
        clear_model_cache()
        clear_maml_cache()
        resp = e2e_client.post("/cold-start/predict", json=NEW_DRUG_PREDICT_BODY)
        assert resp.status_code == 200, resp.text

    def test_ten_predicts_under_two_seconds(self, e2e_client: TestClient):
        start = time.perf_counter()
        for _ in range(10):
            resp = e2e_client.post("/cold-start/predict", json=NEW_DRUG_PREDICT_BODY)
            assert resp.status_code == 200
        assert time.perf_counter() - start < 2.0
