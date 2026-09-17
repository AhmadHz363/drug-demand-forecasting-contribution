"""
Round 5 fixes — sMAPE reliability on sparse series + stacking horizon drift.

Task K finding (P113114 sMAPE 26.7% → 154.8%): the walk-forward hold-out
window is unchanged (5 folds × 7 days = 35 OOF points). The swing is a sMAPE
metric artifact on a zero-inflated intermittent series — small nonzero
forecasts on zero-actual days each contribute up to 200% sMAPE while MASE
stays near baseline (~1.07). No point-forecast regression is implied.

Task L finding (P406816 stacking drift): upward back-half drift is driven
primarily by the SARIMA component's trend extrapolation after stacking blends
it with LGBM/classical. ``dampen_stacked_horizon_drift`` caps growth and
pulls long-horizon steps toward recent level for intermittent/lumpy segments.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.forecasting.constants import (
    STACKING_MAX_HORIZON_GROWTH_INTERMITTENT,
    WALK_FORWARD_N_SPLITS,
    WALK_FORWARD_TEST_HORIZON,
)
from app.forecasting.evaluation_metrics import (
    mase,
    mean_smape,
    smape_unreliable_for_series,
    zero_actual_fraction,
)
from app.forecasting.prediction_bounds import dampen_stacked_horizon_drift


def test_holdout_window_size_unchanged_for_walk_forward():
    assert WALK_FORWARD_N_SPLITS * WALK_FORWARD_TEST_HORIZON == 35


def test_sparse_series_smape_spikes_while_mase_stays_near_baseline():
    """Reproduces P113114-style sMAPE vs MASE disagreement."""
    rng = np.random.default_rng(42)
    n = 35
    actuals = np.zeros(n)
    spike_days = rng.choice(n, size=5, replace=False)
    actuals[spike_days] = rng.uniform(2.0, 8.0, size=5)
    predicted = np.full(n, 3.5)

    sparse_smape = mean_smape(actuals, predicted)
    sparse_mase = mase(actuals, predicted, training_actuals=np.concatenate([[0.0], actuals[:-1]]))

    assert sparse_smape is not None
    assert sparse_smape > 100.0
    assert sparse_mase is not None
    assert sparse_mase < 3.0
    assert smape_unreliable_for_series(actuals, demand_segment="intermittent")
    assert zero_actual_fraction(actuals) > 0.5


def test_intermittent_segment_always_flags_smape_unreliable():
    actuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert smape_unreliable_for_series(actuals, demand_segment="intermittent")
    assert not smape_unreliable_for_series(actuals, demand_segment="smooth")


def test_stacking_drift_dampening_reduces_back_half_trend():
    """Simulates P406816-style SARIMA-led upward drift through the horizon."""
    horizon = 30
    recent_level = 74.0
    sarima = np.linspace(76.0, 130.0, horizon)
    lgbm = np.full(horizon, 74.0)
    classical = np.full(horizon, 72.0)
    weights = np.array([0.40, 0.26, 0.34])
    stacked = (
        weights[0] * sarima + weights[1] * lgbm + weights[2] * classical
    )

    assert stacked[-1] > stacked[0] + 10.0
    assert stacked.sum() > recent_level * horizon * 1.05

    damped = dampen_stacked_horizon_drift(
        stacked,
        demand_segment="intermittent",
        recent_level=recent_level,
        horizon_steps=np.arange(1, horizon + 1),
    )

    assert damped[-1] < stacked[-1]
    assert damped.sum() < stacked.sum()
    assert damped[0] == pytest.approx(stacked[0], rel=0.02)
    back_half_growth = float(damped[-1] - damped[14])
    raw_back_half_growth = float(stacked[-1] - stacked[14])
    assert back_half_growth < raw_back_half_growth * 0.6


def test_stacking_drift_dampening_skips_smooth_segment():
    preds = np.linspace(10.0, 20.0, 10)
    unchanged = dampen_stacked_horizon_drift(
        preds,
        demand_segment="smooth",
        recent_level=12.0,
    )
    np.testing.assert_array_equal(unchanged, preds)


def test_dampening_keeps_30day_total_closer_to_recent_demand():
    """Smoke check — dampening should reduce inflated 30-day totals vs raw stack."""
    horizon = 30
    recent_level = 74.0
    stacked = np.linspace(76.0, 116.0, horizon)
    damped = dampen_stacked_horizon_drift(
        stacked,
        demand_segment="intermittent",
        recent_level=recent_level,
        horizon_steps=np.arange(1, horizon + 1),
    )

    december_total = recent_level * horizon
    assert stacked.sum() > december_total * 1.25
    assert damped.sum() < stacked.sum()
    assert damped.sum() < december_total * 1.20


def test_horizon_growth_cap_matches_constant():
    horizon = 30
    baseline = 80.0
    stacked = baseline * (
        1.0
        + STACKING_MAX_HORIZON_GROWTH_INTERMITTENT
        * np.arange(horizon)
        / max(horizon - 1, 1)
        * 3.0
    )
    damped = dampen_stacked_horizon_drift(
        stacked,
        demand_segment="lumpy",
        recent_level=75.0,
        horizon_steps=np.arange(1, horizon + 1),
    )
    max_allowed = baseline * (1.0 + STACKING_MAX_HORIZON_GROWTH_INTERMITTENT)
    assert float(damped[-1]) <= max_allowed + 1.0
