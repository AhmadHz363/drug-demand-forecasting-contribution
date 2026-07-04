"""
Phase 1 forecasting — demand segmentation, segment ensembles, walk-forward,
and intermittent-aware metrics.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from app.forecasting.constants import (
    ADI_THRESHOLD,
    CV2_THRESHOLD,
    DEMAND_SEGMENTS,
    GLOBAL_DEMAND_SEGMENT,
    SEGMENT_ENSEMBLE_MIN_SAMPLES,
    WALK_FORWARD_N_SPLITS,
    WALK_FORWARD_TEST_HORIZON,
)
from app.forecasting.demand_segmentation import (
    classify_demand_segment,
    compute_adi_cv2,
)
from app.forecasting.ensemble.conformal import ConformalCalibrator
from app.forecasting.ensemble.segment_artifacts import (
    conformal_artifact_path,
    resolve_conformal_path,
    resolve_stacking_path,
    stacking_artifact_path,
)
from app.forecasting.ensemble.stacking import StackingMetaLearner
from app.forecasting.evaluation_metrics import (
    accuracy_skill_from_mase,
    mase,
    mean_pinball_loss,
    rmsse,
)


class TestDemandSegmentation:
    def test_smooth_segment(self):
        quantities = np.array([10.0, 11.0, 9.0, 10.5, 10.0, 11.0, 9.5] * 8)
        segment = classify_demand_segment(quantities)
        assert segment == "smooth"

    def test_intermittent_segment(self):
        quantities = np.zeros(60)
        quantities[::10] = 5.0
        adi, cv2 = compute_adi_cv2(quantities)
        assert adi > ADI_THRESHOLD
        assert cv2 <= CV2_THRESHOLD
        assert classify_demand_segment(quantities) == "intermittent"

    def test_lumpy_segment(self):
        quantities = np.zeros(60)
        quantities[5] = 5.0
        quantities[25] = 80.0
        quantities[45] = 10.0
        adi, cv2 = compute_adi_cv2(quantities)
        assert adi > ADI_THRESHOLD
        assert cv2 > CV2_THRESHOLD
        assert classify_demand_segment(quantities) == "lumpy"

    def test_all_segments_defined(self):
        assert len(DEMAND_SEGMENTS) == 4


class TestSegmentEnsembleArtifacts:
    def test_save_load_segment_specific_stacker(self, tmp_path, monkeypatch):
        artifacts = str(tmp_path / "artifacts")
        monkeypatch.setattr("app.forecasting.constants.ARTIFACTS_DIR", artifacts)
        monkeypatch.setattr(
            "app.forecasting.ensemble.segment_artifacts.ARTIFACTS_DIR",
            artifacts,
        )
        actuals, sarima, lgbm, tft = _synthetic_predictions(120)
        stacker = StackingMetaLearner(segment="intermittent")
        stacker.fit(sarima, lgbm, tft, actuals)
        saved = stacker.save()
        assert saved == stacking_artifact_path("intermittent")
        assert os.path.isfile(saved)

        reloaded = StackingMetaLearner(segment="intermittent")
        reloaded.load()
        in_blend, in_weights = stacker.predict(sarima[:5], lgbm[:5], tft[:5])
        out_blend, out_weights = reloaded.predict(sarima[:5], lgbm[:5], tft[:5])
        np.testing.assert_allclose(in_blend, out_blend)
        assert in_weights == out_weights

    def test_resolve_stacking_falls_back_to_global(self, tmp_path, monkeypatch):
        artifacts = tmp_path / "artifacts"
        monkeypatch.setattr(
            "app.forecasting.ensemble.segment_artifacts.ARTIFACTS_DIR",
            str(artifacts),
        )
        global_path = stacking_artifact_path(GLOBAL_DEMAND_SEGMENT)
        os.makedirs(os.path.dirname(global_path), exist_ok=True)
        with open(global_path, "wb") as handle:
            handle.write(b"placeholder")

        resolved = resolve_stacking_path("lumpy")
        assert resolved == global_path

    def test_segment_conformal_path(self, tmp_path, monkeypatch):
        artifacts = str(tmp_path / "artifacts")
        monkeypatch.setattr("app.forecasting.constants.ARTIFACTS_DIR", artifacts)
        monkeypatch.setattr(
            "app.forecasting.ensemble.segment_artifacts.ARTIFACTS_DIR",
            artifacts,
        )
        actuals, sarima, lgbm, tft = _synthetic_predictions(180)
        stacker = StackingMetaLearner(segment="smooth")
        stacker.fit(sarima[:120], lgbm[:120], tft[:120], actuals[:120])
        stacked, _ = stacker.predict(sarima[120:], lgbm[120:], tft[120:])

        calibrator = ConformalCalibrator(segment="smooth")
        cal_fit = stacked[:40]
        y_cal = actuals[120:160]
        calibrator.fit(cal_fit, y_cal)
        saved = calibrator.save()
        assert saved == conformal_artifact_path("smooth")
        assert resolve_conformal_path("smooth") == saved


class TestIntermittentAwareMetrics:
    def test_perfect_forecast_skill_is_100(self):
        actuals = np.array([10.0, 12.0, 11.0, 13.0, 10.0, 12.0, 11.0, 13.0])
        predicted = actuals.copy()
        mase_val = mase(actuals, predicted)
        assert mase_val == pytest.approx(0.0)
        assert accuracy_skill_from_mase(mase_val) == pytest.approx(100.0)

    def test_rmsse_and_pinball_positive_for_errors(self):
        actuals = np.array([10.0, 12.0, 11.0, 13.0, 10.0, 12.0, 11.0, 13.0])
        predicted = actuals + 4.0
        assert rmsse(actuals, predicted) > 0.0
        assert mean_pinball_loss(actuals, predicted, 0.5) > 0.0


class TestWalkForwardDefaults:
    def test_default_fold_count_increased(self):
        assert WALK_FORWARD_N_SPLITS >= 5
        assert WALK_FORWARD_TEST_HORIZON == 7

    def test_segment_min_samples_reasonable(self):
        assert SEGMENT_ENSEMBLE_MIN_SAMPLES >= 10


def _synthetic_predictions(n: int, seed: int = 11) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    actuals = rng.gamma(shape=2.0, scale=12.0, size=n)
    sarima = actuals + rng.normal(0, 4.0, size=n)
    lgbm = actuals + rng.normal(0, 3.0, size=n)
    tft = actuals + rng.normal(0, 2.5, size=n)
    return actuals, sarima, lgbm, tft
