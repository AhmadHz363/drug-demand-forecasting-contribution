"""
Step 5 Forecasting — ensemble stacking and conformal calibration validation suite.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pytest

from app.forecasting.ensemble.conformal import ConformalCalibrator, extend_tail_quantiles
from app.forecasting.ensemble.stacking import StackingMetaLearner


def _synthetic_model_predictions(n: int, seed: int = 42) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    actuals = rng.gamma(shape=2.0, scale=20.0, size=n)
    sarima = actuals + rng.normal(0, 8.0, size=n)
    lgbm = actuals + rng.normal(0, 6.0, size=n)
    tft = actuals + rng.normal(0, 5.0, size=n)
    return actuals, sarima, lgbm, tft


@pytest.fixture
def artifacts_dir(tmp_path, monkeypatch) -> Iterator[str]:
    path = str(tmp_path / "artifacts")
    monkeypatch.setattr("app.forecasting.constants.ARTIFACTS_DIR", path)
    monkeypatch.setattr("app.forecasting.ensemble.segment_artifacts.ARTIFACTS_DIR", path)
    yield path


class TestStackingMetaLearner:
    def test_weights_sum_to_one(self):
        actuals, sarima, lgbm, tft = _synthetic_model_predictions(200)
        stacker = StackingMetaLearner()
        stacker.fit(sarima, lgbm, tft, actuals)
        _, weights = stacker.predict(sarima[:10], lgbm[:10], tft[:10])
        total = weights.sarima + weights.lgbm + weights.tft
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_tft_unavailable_renormalizes_sarima_lgbm(self):
        actuals, sarima, lgbm, _ = _synthetic_model_predictions(200)
        tft_nan = np.full_like(sarima, np.nan)
        stacker = StackingMetaLearner()
        stacker.fit(sarima, lgbm, tft_nan, actuals)
        _, weights = stacker.predict(sarima[:5], lgbm[:5], tft_nan[:5])
        assert weights.tft == 0.0
        assert weights.sarima + weights.lgbm == pytest.approx(1.0, abs=1e-6)

    def test_save_load_produces_identical_predictions(self, artifacts_dir):
        actuals, sarima, lgbm, tft = _synthetic_model_predictions(150)
        stacker = StackingMetaLearner()
        stacker.fit(sarima, lgbm, tft, actuals)
        in_memory, weights = stacker.predict(sarima[:7], lgbm[:7], tft[:7])
        stacker.save()

        reloaded = StackingMetaLearner()
        reloaded.load()
        loaded, reloaded_weights = reloaded.predict(sarima[:7], lgbm[:7], tft[:7])
        np.testing.assert_allclose(in_memory, loaded)
        assert weights == reloaded_weights


class TestConformalCalibrator:
    def test_coverage_on_holdout_at_least_88_percent(self):
        actuals, sarima, lgbm, tft = _synthetic_model_predictions(600, seed=7)
        stacker = StackingMetaLearner()
        stacker.fit(sarima[:400], lgbm[:400], tft[:400], actuals[:400])
        stacked_cal, _ = stacker.predict(sarima[400:500], lgbm[400:500], tft[400:500])
        stacked_test, _ = stacker.predict(sarima[500:], lgbm[500:], tft[500:])
        y_cal = actuals[400:500]
        y_test = actuals[500:]

        calibrator = ConformalCalibrator()
        calibrator.fit(stacked_cal, y_cal)
        _, lower, upper = calibrator.predict_interval(stacked_test)
        covered = ((y_test >= lower) & (y_test <= upper)).mean()
        assert covered >= 0.88

    def test_predict_interval_non_negative(self):
        actuals, sarima, lgbm, tft = _synthetic_model_predictions(300)
        stacker = StackingMetaLearner()
        stacker.fit(sarima, lgbm, tft, actuals)
        stacked, _ = stacker.predict(sarima, lgbm, tft)

        calibrator = ConformalCalibrator()
        calibrator.fit(stacked[:200], actuals[:200])
        p50, p10, p90 = calibrator.predict_interval(stacked[200:])
        assert (p50 >= 0).all()
        assert (p10 >= 0).all()
        assert (p90 >= 0).all()

    def test_save_load_is_deterministic(self, artifacts_dir):
        actuals, sarima, lgbm, tft = _synthetic_model_predictions(250)
        stacker = StackingMetaLearner()
        stacker.fit(sarima, lgbm, tft, actuals)
        stacked, _ = stacker.predict(sarima, lgbm, tft)

        calibrator = ConformalCalibrator()
        calibrator.fit(stacked[:150], actuals[:150])
        in_p50, in_p10, in_p90 = calibrator.predict_interval(stacked[150:180])
        calibrator.save()

        reloaded = ConformalCalibrator()
        reloaded.load()
        out_p50, out_p10, out_p90 = reloaded.predict_interval(stacked[150:180])
        np.testing.assert_allclose(in_p50, out_p50)
        np.testing.assert_allclose(in_p10, out_p10)
        np.testing.assert_allclose(in_p90, out_p90)


class TestExtendTailQuantiles:
    def test_quantile_ordering(self):
        p50 = np.array([50.0, 100.0, 30.0])
        p10 = np.array([30.0, 70.0, 20.0])
        p90 = np.array([70.0, 130.0, 40.0])
        p5, p10_out, p50_out, p90_out, p95 = extend_tail_quantiles(p50, p10, p90)
        for idx in range(len(p50)):
            assert p5[idx] <= p10_out[idx] <= p50_out[idx] <= p90_out[idx] <= p95[idx]
            assert p5[idx] >= 0

    def test_artifact_files_exist_after_save(self, artifacts_dir):
        actuals, sarima, lgbm, tft = _synthetic_model_predictions(180)
        stacker = StackingMetaLearner()
        stacker.fit(sarima, lgbm, tft, actuals)
        stacking_path = stacker.save()

        stacked, _ = stacker.predict(sarima, lgbm, tft)
        calibrator = ConformalCalibrator()
        calibrator.fit(stacked[:120], actuals[:120])
        mapie_path = calibrator.save()

        assert stacking_path.endswith("stacking_meta.pkl")
        assert mapie_path.endswith("mapie_wrapper.pkl")
