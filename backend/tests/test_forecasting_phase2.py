"""
Phase 2 forecasting — adaptive model hyperparameters, stacking meta-features,
per-horizon conformal calibration, and recursive drift instrumentation.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from app.forecasting.constants import (
    CV2_THRESHOLD,
    MIN_HISTORY_DAYS_TFT,
    MIN_HISTORY_DAYS_TFT_SHORT,
    SARIMA_FORECAST_SHRINKAGE,
    SARIMA_TRAIN_DAYS,
    SARIMA_TRAIN_DAYS_SHORT,
    STACKING_DISAGREEMENT_CAP_RATIO,
    TFT_MAX_EPOCHS,
    TFT_WALK_FORWARD_EPOCH_RATIO,
    TFT_WALK_FORWARD_MAX_EPOCHS,
)
from app.forecasting.ensemble.conformal import ConformalCalibrator
from app.forecasting.ensemble.stacking import StackingMetaLearner
from app.forecasting.model_adaptation import (
    adaptive_lgbm_anchor_weight,
    adaptive_sarima_shrinkage,
    build_stacking_meta_features,
    lgbm_training_params,
    select_sarima_train_days,
)
from app.forecasting.models.tft_model import _tft_encoder_window, _tft_min_history_days
from app.forecasting.training.walk_forward import (
    measure_lgbm_recursive_drift,
    _walk_forward_tft_epochs,
)


class TestAdaptiveHyperparameters:
    def test_smooth_series_shrink_more(self):
        smooth = adaptive_sarima_shrinkage(0.1)
        volatile = adaptive_sarima_shrinkage(CV2_THRESHOLD * 2)
        assert smooth > SARIMA_FORECAST_SHRINKAGE
        assert volatile < SARIMA_FORECAST_SHRINKAGE

    def test_anchor_weight_decreases_with_volatility(self):
        smooth = adaptive_lgbm_anchor_weight(0.1)
        volatile = adaptive_lgbm_anchor_weight(CV2_THRESHOLD * 2)
        assert smooth > volatile

    def test_short_sarima_window_for_intermittent(self):
        assert select_sarima_train_days("intermittent", 200) == SARIMA_TRAIN_DAYS_SHORT
        assert select_sarima_train_days("smooth", 200) == SARIMA_TRAIN_DAYS

    def test_lgbm_segment_params_regularize_sparse_demand(self):
        sparse = lgbm_training_params("intermittent")
        smooth = lgbm_training_params("smooth")
        assert sparse["num_leaves"] < smooth["num_leaves"]
        assert sparse["max_depth"] <= smooth["max_depth"]


class TestStackingMetaFeatures:
    def test_meta_feature_matrix_shape(self):
        meta = build_stacking_meta_features(
            10,
            demand_segment="lumpy",
            history_days=np.full(10, 120.0),
            recent_cv2=np.linspace(0.1, 1.0, 10),
            tft_available=False,
            horizon_steps=np.arange(1, 11),
        )
        assert meta.shape == (10, 5)

    def test_stacker_uses_meta_features(self, tmp_path, monkeypatch):
        artifacts = str(tmp_path / "artifacts")
        monkeypatch.setattr("app.forecasting.constants.ARTIFACTS_DIR", artifacts)
        monkeypatch.setattr(
            "app.forecasting.ensemble.segment_artifacts.ARTIFACTS_DIR",
            artifacts,
        )
        n = 80
        actuals, sarima, lgbm, tft = _synthetic_predictions(n)
        steps = np.tile(np.arange(1, 8), n // 7 + 1)[:n]
        history = np.full(n, 200.0)
        cv2 = np.full(n, 0.3)

        stacker = StackingMetaLearner(segment="erratic")
        stacker.fit(
            sarima,
            lgbm,
            tft,
            actuals,
            demand_segment="erratic",
            history_days=history,
            recent_cv2=cv2,
            horizon_steps=steps,
        )
        blend, weights = stacker.predict(
            sarima[:5],
            lgbm[:5],
            tft[:5],
            demand_segment="erratic",
            history_days=history[:5],
            recent_cv2=cv2[:5],
            horizon_steps=steps[:5],
            prediction_cap=100.0,
        )
        assert blend.shape == (5,)
        assert weights.sarima + weights.lgbm + weights.tft == pytest.approx(1.0)

    def test_disagreement_cap_constant(self):
        assert STACKING_DISAGREEMENT_CAP_RATIO == 0.5


class TestPerHorizonConformal:
    def test_step_scale_factors_saved(self, tmp_path, monkeypatch):
        artifacts = str(tmp_path / "artifacts")
        monkeypatch.setattr("app.forecasting.constants.ARTIFACTS_DIR", artifacts)
        monkeypatch.setattr(
            "app.forecasting.ensemble.segment_artifacts.ARTIFACTS_DIR",
            artifacts,
        )
        n = 70
        stacked = np.linspace(8.0, 15.0, n)
        actuals = stacked + np.sin(np.arange(n)) * 2.0
        steps = np.tile(np.arange(1, 8), 10)

        calibrator = ConformalCalibrator(segment="smooth")
        calibrator.fit(stacked, actuals, horizon_steps=steps)
        assert len(calibrator._step_scale_factors) >= 2
        saved = calibrator.save()
        assert os.path.isfile(saved)

        reloaded = ConformalCalibrator(segment="smooth")
        reloaded.load()
        assert reloaded._step_scale_factors


class TestRecursiveDrift:
    def test_drift_mae_by_step(self):
        preds = {1: [10.0, 12.0], 7: [10.0, 20.0]}
        actuals = {1: [10.0, 12.0], 7: [10.0, 12.0]}
        drift = measure_lgbm_recursive_drift(preds, actuals)
        assert drift[1] == pytest.approx(0.0)
        assert drift[7] == pytest.approx(4.0)


class TestTftShortHistory:
    def test_min_history_thresholds(self):
        assert _tft_min_history_days(400) == MIN_HISTORY_DAYS_TFT
        assert _tft_min_history_days(200) == MIN_HISTORY_DAYS_TFT_SHORT
        assert _tft_min_history_days(100) == MIN_HISTORY_DAYS_TFT

    def test_encoder_window_shortens(self):
        assert _tft_encoder_window(400) == 365
        short = _tft_encoder_window(200)
        assert 150 <= short <= 180

    def test_walk_forward_epochs_aligned(self):
        epochs = _walk_forward_tft_epochs()
        # TFT_WALK_FORWARD_MAX_EPOCHS is a hard ceiling, so epochs must be
        # at most that value — not at least (the old `max` semantics were wrong).
        assert epochs <= TFT_WALK_FORWARD_MAX_EPOCHS
        assert epochs >= 1


def _synthetic_predictions(n: int, seed: int = 7) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    actuals = rng.gamma(shape=2.0, scale=10.0, size=n)
    sarima = actuals + rng.normal(0, 3.0, size=n)
    lgbm = actuals + rng.normal(0, 2.5, size=n)
    tft = actuals + rng.normal(0, 2.0, size=n)
    return actuals, sarima, lgbm, tft
