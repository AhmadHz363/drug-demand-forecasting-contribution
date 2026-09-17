"""Hold-out validation tests."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest
from pydantic import ValidationError

from app.forecasting.schemas import HoldoutValidationRequest
from app.forecasting.training.holdout import _compute_metrics, _optional_float
from app.forecasting.evaluation_metrics import accuracy_from_smape, smape


class TestAccuracyFromSmape:
    def test_perfect_prediction(self):
        assert accuracy_from_smape(0.0) == pytest.approx(100.0)

    def test_worst_case(self):
        assert accuracy_from_smape(200.0) == pytest.approx(0.0)

    def test_midpoint(self):
        assert accuracy_from_smape(100.0) == pytest.approx(50.0)


class TestSmape:
    def test_perfect_prediction(self):
        assert smape(100.0, 100.0) == 0.0

    def test_zero_actual_and_predicted(self):
        assert smape(0.0, 0.0) == 0.0

    def test_symmetric_error(self):
        result = smape(100.0, 80.0)
        assert 0.0 < result < 100.0


class TestHoldoutMetrics:
    def test_compute_metrics_perfect(self):
        actuals = np.array([10.0, 20.0, 30.0, 10.0, 20.0, 30.0, 10.0, 20.0])
        predicted = np.array([10.0, 20.0, 30.0, 10.0, 20.0, 30.0, 10.0, 20.0])
        metrics = _compute_metrics(actuals, predicted, predicted * 0.8, predicted * 1.2)
        assert metrics.smape == pytest.approx(0.0)
        assert metrics.mae == pytest.approx(0.0)
        assert metrics.mase == pytest.approx(0.0)
        assert metrics.coverage_90 == pytest.approx(1.0)
        assert metrics.accuracy_pct == pytest.approx(100.0)
        assert metrics.accuracy_skill_pct == pytest.approx(100.0)

    def test_compute_metrics_with_nan_predictions(self):
        actuals = np.array([10.0, 20.0])
        predicted = np.array([12.0, np.nan])
        metrics = _compute_metrics(actuals, predicted)
        assert metrics.mae == pytest.approx(2.0)
        assert metrics.smape == pytest.approx(smape(10.0, 12.0))
        assert metrics.accuracy_pct == pytest.approx(accuracy_from_smape(metrics.smape))


class TestHoldoutSchemas:
    def test_default_date_windows(self):
        req = HoldoutValidationRequest(drug_code="TEST-001")
        assert req.train_end == date(2025, 12, 31)
        assert req.test_start == date(2026, 1, 1)
        assert req.test_end == date(2026, 3, 31)
        assert req.models == ["sarima", "lgbm", "classical"]

    def test_drug_code_required(self):
        with pytest.raises(ValidationError):
            HoldoutValidationRequest()


class TestOptionalFloat:
    def test_nan_returns_none(self):
        assert _optional_float(float("nan")) is None

    def test_value_preserved(self):
        assert _optional_float(42.5) == 42.5
