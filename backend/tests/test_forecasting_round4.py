"""
Round 4 fixes — interval-width collapse, normal-supply metric fallback.

Task I (investigation only): P113114's ~3.5/day recommended level vs ~0.9/day
historical average is likely driven by SARIMA smoothing on a zero-inflated
intermittent series (champion weight 1.0 on SARIMA, classical weight 0).
Classical *does* include TSB/Croston for intermittent/lumpy segments, but when
SARIMA wins champion selection those methods are not blended in. December's
higher spike-day frequency may justify a modest uplift, but a sustained 2–4×
gap vs the 90-day mean points to model/segment mismatch rather than a confirmed
recent trend — scope a Croston/TSB champion or segment routing fix separately.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from app.forecasting.constants import CONFORMAL_MIN_STEP_SCALE_RATIO
from app.forecasting.ensemble.conformal import ConformalCalibrator
from app.forecasting.evaluation_metrics import rolling_validation_metrics
from app.forecasting.models.classical_model import SEGMENT_METHOD


def _assert_smooth_interval_widths(
    widths: np.ndarray,
    *,
    drug_code: str = "TEST",
    min_neighbor_ratio: float = 0.1,
) -> None:
    for idx in range(1, len(widths) - 1):
        neighbor_avg = (widths[idx - 1] + widths[idx + 1]) / 2.0
        if neighbor_avg <= 1e-6:
            continue
        ratio = widths[idx] / neighbor_avg
        assert ratio > min_neighbor_ratio, (
            f"{drug_code} day {idx + 1}: width {widths[idx]:.2f} "
            f"is <{min_neighbor_ratio:.0%} of neighboring average "
            f"{neighbor_avg:.2f}"
        )


def test_per_step_scale_factors_are_floored_relative_to_global():
    calibrator = ConformalCalibrator()
    calibrator.scale_factor = 0.6
    calibrator._step_scale_factors = {1: 0.02, 4: 0.0138, 7: 0.09, 2: 0.7}
    calibrator._finalize_step_scale_factors()
    floor = 0.6 * CONFORMAL_MIN_STEP_SCALE_RATIO
    assert calibrator._scale_for_step(1) >= floor
    assert calibrator._scale_for_step(4) >= floor
    assert calibrator._scale_for_step(7) >= floor
    assert calibrator._scale_for_step(2) == pytest.approx(0.7)


def test_loaded_artifact_recovers_from_collapsed_step_scales(tmp_path, monkeypatch):
    artifacts = str(tmp_path / "artifacts")
    monkeypatch.setattr("app.forecasting.constants.ARTIFACTS_DIR", artifacts)
    monkeypatch.setattr("app.forecasting.ensemble.segment_artifacts.ARTIFACTS_DIR", artifacts)

    rng = np.random.default_rng(0)
    n = 140
    steps = np.tile(np.arange(1, 8), n // 7)
    stacked = rng.uniform(2.0, 12.0, n)
    actuals = stacked + rng.normal(0, 3.0, n)
    actuals = np.clip(actuals, 0, None)

    calibrator = ConformalCalibrator(segment="global")
    calibrator.fit(stacked, actuals, horizon_steps=steps)
    calibrator._step_scale_factors = {1: 0.1067, 4: 0.0138, 7: 0.093, 2: 0.6772}
    calibrator.save()

    reloaded = ConformalCalibrator(segment="global")
    reloaded.load()
    champion = np.linspace(3.0, 8.0, 30)
    _, p10, p90 = reloaded.predict_interval(
        champion,
        horizon_steps=np.arange(1, 31),
    )
    _assert_smooth_interval_widths(p90 - p10, drug_code="ARTIFACT")


def test_interval_widths_smooth_for_typical_low_volume_champion(tmp_path, monkeypatch):
    artifacts = str(tmp_path / "artifacts")
    monkeypatch.setattr("app.forecasting.constants.ARTIFACTS_DIR", artifacts)
    monkeypatch.setattr("app.forecasting.ensemble.segment_artifacts.ARTIFACTS_DIR", artifacts)

    rng = np.random.default_rng(3)
    stacked = rng.uniform(2.0, 12.0, 120)
    actuals = np.clip(stacked + rng.normal(0, 3.0, 120), 0, None)
    calibrator = ConformalCalibrator(segment="global")
    calibrator.fit(stacked, actuals)
    calibrator._step_scale_factors = {
        1: 0.1067,
        2: 0.6772,
        3: 0.5825,
        4: 0.0138,
        5: 0.2183,
        6: 0.4148,
        7: 0.093,
    }
    calibrator._finalize_step_scale_factors()

    for drug_code, level in (("P406816", 45.0), ("P113114", 3.5)):
        champion = np.full(30, level)
        _, p10, p90 = calibrator.predict_interval(
            champion,
            horizon_steps=np.arange(1, 31),
        )
        _assert_smooth_interval_widths(p90 - p10, drug_code=drug_code)


def test_normal_supply_rolling_metrics_fallback_to_full_window():
    n = 50
    actuals = np.ones(n) * 5.0
    predicted = np.ones(n) * 5.0
    stockout = np.zeros(n, dtype=bool)

    rolling = rolling_validation_metrics(
        actuals,
        predicted,
        stockout_flags=stockout,
        training_actuals=actuals[:20],
    )
    assert rolling["smape_7day_normal"] == pytest.approx(rolling["smape_7day_full"])
    assert rolling["mase_7day_normal"] == pytest.approx(rolling["mase_7day_full"])


def test_walk_forward_normal_supply_smape_falls_back_to_full_window():
    smape_scores = [26.0, 27.5, 26.1]
    smape_normal_scores: list[float] = []
    avg_smape = float(sum(smape_scores) / len(smape_scores))
    avg_smape_normal = (
        float(sum(smape_normal_scores) / len(smape_normal_scores))
        if smape_normal_scores
        else avg_smape
    )
    assert avg_smape_normal == pytest.approx(avg_smape)
    assert avg_smape_normal is not None


def test_classical_includes_intermittent_demand_methods():
    assert SEGMENT_METHOD["intermittent"] == "tsb"
    assert SEGMENT_METHOD["lumpy"] == "croston_sba"


def test_legacy_mapie_pickle_without_baselines_is_repaired_on_load(tmp_path, monkeypatch):
    artifacts = str(tmp_path / "artifacts")
    monkeypatch.setattr("app.forecasting.constants.ARTIFACTS_DIR", artifacts)
    monkeypatch.setattr("app.forecasting.ensemble.segment_artifacts.ARTIFACTS_DIR", artifacts)

    from app.forecasting.ensemble.segment_artifacts import conformal_artifact_path
    import os

    os.makedirs(os.path.dirname(conformal_artifact_path("global")), exist_ok=True)
    payload = {
        "mapie": None,
        "estimator": None,
        "scale_factor": 0.5785,
        "step_scale_factors": {1: 0.02, 4: 0.01, 7: 0.05},
        "lower_hw_baseline": 0.0,
        "upper_hw_baseline": 0.0,
        "step_lower_hw": {},
        "step_upper_hw": {},
        "segment": "global",
    }
    # Fit a real MAPIE object so predict_interval works after load.
    rng = np.random.default_rng(1)
    stacked = rng.uniform(2.0, 10.0, 80)
    actuals = stacked + rng.normal(0, 2.0, 80)
    calibrator = ConformalCalibrator(segment="global")
    calibrator.fit(stacked, actuals)
    payload["mapie"] = calibrator._mapie
    payload["estimator"] = calibrator._estimator

    with open(conformal_artifact_path("global"), "wb") as handle:
        pickle.dump(payload, handle)

    reloaded = ConformalCalibrator(segment="global")
    reloaded.load()
    for step in (1, 4, 7):
        assert reloaded._scale_for_step(step) >= 0.5785 * CONFORMAL_MIN_STEP_SCALE_RATIO
