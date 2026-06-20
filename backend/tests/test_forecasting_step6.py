"""
Step 6 Forecasting — training orchestrator and train endpoint validation suite.
"""

from __future__ import annotations

from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.forecasting.schemas import TrainStatusResponse
from app.forecasting.training.trainer import ForecastingTrainer
from app.main import app


@pytest.fixture
def artifacts_dir(tmp_path, monkeypatch) -> Iterator[str]:
    path = str(tmp_path / "artifacts")
    monkeypatch.setattr("app.forecasting.models.sarima_model.ARTIFACTS_DIR", path)
    monkeypatch.setattr("app.forecasting.models.lgbm_model.ARTIFACTS_DIR", path)
    monkeypatch.setattr("app.forecasting.models.tft_model.ARTIFACTS_DIR", path)
    monkeypatch.setattr("app.forecasting.ensemble.stacking.ARTIFACTS_DIR", path)
    monkeypatch.setattr("app.forecasting.ensemble.conformal.ARTIFACTS_DIR", path)
    yield path


class TestForecastingTrainer:
    def test_train_all_returns_status_response(self, artifacts_dir):
        trainer = ForecastingTrainer()
        mock_db = MagicMock()
        mock_db.commit = MagicMock()

        corrected_df = MagicMock()
        corrected_df.__len__ = MagicMock(return_value=120)

        def build_side_effect(drug_code, db_session):
            return corrected_df if drug_code == "drug-a" else None

        with (
            patch.object(trainer, "_build_corrected_frame", side_effect=build_side_effect),
            patch.object(
                trainer,
                "_train_single_model",
                return_value=(True, f"{artifacts_dir}/sarima/drug-a.pkl", 12.5, 0.91),
            ),
            patch(
                "app.forecasting.training.trainer.collect_walk_forward_predictions",
                return_value={
                    "sarima": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
                    "actuals": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
                },
            ),
        ):
            response = trainer.train_all(
                drug_codes=["drug-a", "drug-b"],
                models_to_train=["sarima"],
                db_session=mock_db,
                force_retrain=True,
            )

        assert isinstance(response, TrainStatusResponse)
        assert response.status == "ok"
        assert response.drugs_trained == 1
        assert response.models_trained == ["sarima"]
        assert response.smape_summary["sarima"] == pytest.approx(12.5)
        assert mock_db.commit.called

    def test_skips_already_trained_when_force_retrain_false(self, artifacts_dir, caplog):
        import logging

        caplog.set_level(logging.INFO)
        trainer = ForecastingTrainer()
        mock_db = MagicMock()
        mock_db.commit = MagicMock()

        corrected_df = MagicMock()
        corrected_df.__len__ = MagicMock(return_value=120)

        mock_model = MagicMock()
        mock_model.is_trained.return_value = True

        with (
            patch.object(trainer, "_build_corrected_frame", return_value=corrected_df),
            patch.dict(
                "app.forecasting.training.trainer.MODEL_REGISTRY",
                {"sarima": MagicMock(return_value=mock_model)},
            ),
            patch(
                "app.forecasting.training.trainer.collect_walk_forward_predictions",
                return_value=None,
            ),
        ):
            response = trainer.train_all(
                drug_codes=["drug-a"],
                models_to_train=["sarima"],
                db_session=mock_db,
                force_retrain=False,
            )

        assert response.drugs_trained == 0
        assert "already trained" in caplog.text


class TestTrainEndpoint:
    def test_train_endpoint_delegates_to_trainer(self, artifacts_dir):
        client = TestClient(app)
        mock_response = TrainStatusResponse(
            status="ok",
            drugs_trained=2,
            models_trained=["sarima"],
            smape_summary={"sarima": 15.0},
            artifacts_saved=[f"{artifacts_dir}/sarima/a.pkl"],
        )

        with patch(
            "app.api.forecasting.ForecastingTrainer.train_all",
            return_value=mock_response,
        ) as train_all:
            response = client.post(
                "/forecasting/train",
                json={"models": ["sarima"], "force_retrain": True},
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert payload["drugs_trained"] == 2
        train_all.assert_called_once()

    def test_train_endpoint_rejects_empty_catalog(self):
        client = TestClient(app)
        mock_db = MagicMock()
        mock_db.query.return_value.order_by.return_value.all.return_value = []

        app.dependency_overrides.clear()
        from app.api.deps import get_db

        def override_db():
            yield mock_db

        app.dependency_overrides[get_db] = override_db
        try:
            response = client.post(
                "/forecasting/train",
                json={"models": ["sarima"]},
            )
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 422
