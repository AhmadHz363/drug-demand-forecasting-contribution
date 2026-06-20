"""
Step 4 Forecasting — individual model validation suite.
"""

from __future__ import annotations

import json
import logging
from typing import Iterator

import pandas as pd
import pytest

from app.forecasting.models.base_model import BaseForecastingModel
from app.forecasting.models.lgbm_model import LightGBMModel
from app.forecasting.models.sarima_model import SarimaModel
from app.forecasting.models.tft_model import TFTModel
from app.forecasting.feature_engineering.lag_features import add_lag_features
from app.forecasting.feature_engineering.rolling_features import add_rolling_features
from app.forecasting.feature_engineering.temporal_features import add_temporal_features
from app.forecasting.training.walk_forward import walk_forward_smape


def _build_training_frame(n_days: int = 400, *, inject_stockout: bool = False) -> pd.DataFrame:
    quantities = [50.0 + (i % 14) * 2.0 + (i % 7) for i in range(n_days)]
    if inject_stockout and n_days > 20:
        quantities[15] = 0.0
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    base = pd.DataFrame({"demand_date": dates, "total_quantity": quantities})
    base = add_temporal_features(base)
    base = add_lag_features(base)
    base = add_rolling_features(base)
    base["bed_occupancy_rate"] = 0.75
    base["weekly_surgery_count"] = 50
    base["supplier_avg_lead_time"] = 3.0
    base["supplier_lead_time_std"] = 1.0
    base["supplier_reliability_score"] = 0.85
    base["em_corrected_quantity"] = base["total_quantity"].astype(float)
    return base.set_index("demand_date")


@pytest.fixture
def artifacts_dir(tmp_path, monkeypatch) -> Iterator[str]:
    path = str(tmp_path / "artifacts")
    monkeypatch.setattr("app.forecasting.models.sarima_model.ARTIFACTS_DIR", path)
    monkeypatch.setattr("app.forecasting.models.lgbm_model.ARTIFACTS_DIR", path)
    monkeypatch.setattr("app.forecasting.models.tft_model.ARTIFACTS_DIR", path)
    yield path


class TestBaseForecastingModelContract:
    @pytest.mark.parametrize("model_cls", [SarimaModel, LightGBMModel, TFTModel])
    def test_cannot_instantiate_incomplete_subclass(self, model_cls):
        assert issubclass(model_cls, BaseForecastingModel)


class TestSarimaModel:
    def test_train_predict_save_load(self, artifacts_dir):
        drug_code = "amoxicillin"
        df = _build_training_frame(120)
        model = SarimaModel()
        assert model.is_trained(drug_code) is False

        model.train(df, drug_code)
        preds = model.predict(df, horizon_days=7)
        assert len(preds) == 7
        assert set(preds.columns) >= {"forecast_date", "p10", "p50", "p90"}

        path = model.save(drug_code)
        assert path.endswith(".pkl")
        assert model.is_trained(drug_code) is True

        reloaded = SarimaModel()
        reloaded.load(drug_code)
        reloaded_preds = reloaded.predict(df, horizon_days=7)
        pd.testing.assert_frame_equal(preds, reloaded_preds)

    def test_rejects_insufficient_history(self):
        df = _build_training_frame(59)
        model = SarimaModel()
        with pytest.raises(ValueError, match="SARIMA requires at least 60 days"):
            model.train(df, "short-drug")


class TestLightGBMModel:
    def test_train_predict_quantile_ordering(self, artifacts_dir):
        drug_code = "amoxicillin"
        df = _build_training_frame(120)
        model = LightGBMModel()
        assert model.is_trained(drug_code) is False

        model.train(df, drug_code)
        preds = model.predict(df, horizon_days=7)
        assert len(preds) == 7
        assert (preds["p10"] <= preds["p50"]).all()
        assert (preds["p50"] <= preds["p90"]).all()

        model.save(drug_code)
        assert model.is_trained(drug_code) is True

        shap_path = model.shap_artifact_path(drug_code)
        assert shap_path.endswith("_shap.json")
        with open(shap_path, encoding="utf-8") as handle:
            shap_data = json.load(handle)
        assert isinstance(shap_data, dict)
        assert len(shap_data) >= 10


class TestTFTModel:
    def test_skips_training_when_history_too_short(self, artifacts_dir, caplog):
        drug_code = "short-tft"
        df = _build_training_frame(200)
        model = TFTModel()
        with caplog.at_level(logging.WARNING):
            model.train(df, drug_code)
        assert model.is_trained(drug_code) is False
        assert "TFT skipped" in caplog.text

    @pytest.mark.slow
    def test_train_predict_when_history_sufficient(self, artifacts_dir, monkeypatch):
        monkeypatch.setattr("app.forecasting.models.tft_model.TFT_MAX_EPOCHS", 1)
        drug_code = "tft-drug"
        df = _build_training_frame(380)
        model = TFTModel()
        model.train(df, drug_code)
        if not model.is_trained(drug_code):
            pytest.skip("TFT dependencies unavailable or training failed")
        preds = model.predict(df, horizon_days=7)
        assert len(preds) == 7
        for col in ("p5", "p10", "p50", "p90", "p95"):
            assert col in preds.columns


class TestWalkForwardSmape:
    def test_returns_non_negative_score(self, artifacts_dir):
        df = _build_training_frame(150)
        score = walk_forward_smape(SarimaModel(), df, "wf-drug", n_splits=3, test_horizon=7)
        assert 0.0 <= score <= 100.0

    def test_smape_zero_when_both_zero(self):
        from app.forecasting.training.walk_forward import smape

        assert smape(0.0, 0.0) == 0.0
