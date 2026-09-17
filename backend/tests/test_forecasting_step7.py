"""
Step 7 Forecasting — inference pipeline and predict endpoints validation suite.
"""

from __future__ import annotations

from datetime import date
from typing import Iterator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.forecasting.inference.forecaster import (
    DrugForecaster,
    DrugNotInCatalogError,
    NoTrainedModelsError,
    validate_forecast_quantiles,
)
from app.forecasting.schemas import (
    DailyForecastPoint,
    ForecastResponse,
    ModelWeightBreakdown,
)
from app.main import app


def _sample_response(drug_code: str = "drug-a", horizon_days: int = 7) -> ForecastResponse:
    return ForecastResponse(
        drug_code=drug_code,
        center_syn_id=None,
        horizon_days=horizon_days,
        model_weights=ModelWeightBreakdown(sarima=0.4, lgbm=0.6, classical=0.0),
        forecast=[
            DailyForecastPoint(
                date=date(2024, 1, idx),
                p5=8.0 + idx,
                p10=10.0 + idx,
                p50=20.0 + idx,
                p90=30.0 + idx,
                p95=32.0 + idx,
            )
            for idx in range(1, horizon_days + 1)
        ],
        uncertainty_note="test note",
        smape_last_validation=12.5,
    )


class TestValidateForecastQuantiles:
    def test_accepts_valid_response(self):
        validate_forecast_quantiles(_sample_response())

    def test_rejects_invalid_quantile_order(self):
        bad = _sample_response()
        bad.forecast[0].p50 = 5.0
        with pytest.raises(ValueError, match="Invalid quantiles"):
            validate_forecast_quantiles(bad)

    def test_rejects_weights_not_summing_to_one(self):
        bad = _sample_response()
        bad.model_weights = ModelWeightBreakdown(sarima=0.3, lgbm=0.3, classical=0.3)
        with pytest.raises(ValueError, match="Model weights must sum to 1.0"):
            validate_forecast_quantiles(bad)


class TestPredictEndpoint:
    def test_predict_returns_200(self):
        client = TestClient(app)
        response = _sample_response()

        with patch(
            "app.api.forecasting.DrugForecaster.forecast",
            return_value=response,
        ):
            result = client.post(
                "/forecasting/predict",
                json={"drug_code": "drug-a", "horizon_days": 7},
            )

        assert result.status_code == 200
        payload = result.json()
        assert payload["drug_code"] == "drug-a"
        assert len(payload["forecast"]) == 7

    def test_predict_returns_404_for_unknown_drug(self):
        client = TestClient(app)
        with patch(
            "app.api.forecasting.DrugForecaster.forecast",
            side_effect=DrugNotInCatalogError("Drug missing not found in catalog."),
        ):
            result = client.post(
                "/forecasting/predict",
                json={"drug_code": "missing", "horizon_days": 7},
            )
        assert result.status_code == 404

    def test_predict_returns_503_for_untrained_drug(self):
        client = TestClient(app)
        with patch(
            "app.api.forecasting.DrugForecaster.forecast",
            side_effect=NoTrainedModelsError(
                "No trained models found for missing. Call POST /forecasting/train first."
            ),
        ):
            result = client.post(
                "/forecasting/predict",
                json={"drug_code": "missing", "horizon_days": 7},
            )
        assert result.status_code == 503


class TestPredictBatchEndpoint:
    def test_batch_returns_mixed_success_and_error(self):
        client = TestClient(app)
        ok = _sample_response("good-drug")

        def forecast_side_effect(**kwargs):
            if kwargs["drug_code"] == "good-drug":
                return ok
            raise DrugNotInCatalogError("Drug bad-drug not found in catalog.")

        with patch(
            "app.api.forecasting.DrugForecaster.forecast",
            side_effect=lambda **kwargs: forecast_side_effect(**kwargs),
        ):
            result = client.post(
                "/forecasting/predict-batch",
                json={"drug_codes": ["good-drug", "bad-drug"], "horizon_days": 7},
            )

        assert result.status_code == 200
        payload = result.json()
        assert len(payload) == 2
        assert payload[0]["drug_code"] == "good-drug"
        assert payload[0]["error"] is None
        assert payload[1]["error"] is not None


class TestDrugForecasterEnsemble:
    def test_ensemble_forecast_without_artifacts(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "app.forecasting.inference.forecaster.resolve_stacking_path",
            lambda segment: None,
        )
        monkeypatch.setattr(
            "app.forecasting.inference.forecaster.resolve_conformal_path",
            lambda segment: None,
        )

        forecaster = DrugForecaster()
        model_preds = {
            "sarima": np.array([10.0, 12.0]),
            "lgbm": np.array([14.0, 16.0]),
            "classical": np.array([np.nan, np.nan]),
        }
        p5, p10, p50, p90, p95, weights, _health = forecaster._ensemble_forecast(
            model_preds,
            2,
            prediction_cap=100.0,
            demand_segment="smooth",
            history_days=120,
            drug_cv2=0.2,
            classical_available=False,
        )

        assert len(p50) == 2
        assert weights.sarima + weights.lgbm + weights.classical == pytest.approx(1.0)
        assert p5[0] <= p10[0] <= p50[0] <= p90[0] <= p95[0]
