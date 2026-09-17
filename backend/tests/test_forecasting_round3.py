"""Round 3 validation metrics — MASE, duration stockouts, rolling totals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.forecasting.censored_demand.detector import (
    _flag_duration_stockout_runs,
    detect_stockout_windows,
)
from app.forecasting.constants import STOCKOUT_MIN_RUN_DAYS
from app.forecasting.evaluation_metrics import (
    mase,
    mase_beats_baseline,
    rolling_validation_metrics,
)


def test_mase_uses_training_period_for_scaling():
    training = np.array([10.0, 12.0, 11.0, 13.0, 10.0, 12.0, 11.0, 13.0, 10.0, 12.0])
    actuals = np.array([10.0, 12.0, 11.0, 13.0, 10.0, 12.0, 11.0])
    predicted = actuals.copy()
    assert mase(actuals, predicted, training_actuals=training) == pytest.approx(0.0)
    assert mase_beats_baseline(0.0) is True

    worse = actuals + 3.0
    scaled = mase(actuals, worse, training_actuals=training)
    assert scaled > 1.0
    assert mase_beats_baseline(scaled) is False


def test_rolling_validation_metrics_reduce_noise_vs_daily():
    rng = np.random.default_rng(0)
    n = 70
    actuals = 20 + rng.normal(0, 8, n)
    actuals = np.clip(actuals, 0, None)
    predicted = actuals + rng.normal(0, 6, n)

    rolling = rolling_validation_metrics(actuals, predicted, training_actuals=actuals[:35])
    daily_smape = float(
        np.mean(
            [
                200.0 * abs(a - p) / (abs(a) + abs(p))
                for a, p in zip(actuals[35:], predicted[35:])
                if (abs(a) + abs(p)) > 0
            ]
        )
    )
    assert rolling["smape_30day_full"] is not None
    assert rolling["smape_30day_full"] <= daily_smape + 5.0


def test_normal_supply_metrics_exclude_only_stockout_windows():
    n = 60
    actuals = np.ones(n) * 5.0
    predicted = np.ones(n) * 5.0
    stockout = np.zeros(n, dtype=bool)
    stockout[10:55] = True  # 45-day stockout run

    rolling = rolling_validation_metrics(
        actuals,
        predicted,
        stockout_flags=stockout,
        training_actuals=actuals[:20],
    )
    assert rolling["smape_7day_full"] == pytest.approx(0.0)
    assert rolling["smape_7day_normal"] == pytest.approx(0.0)


def test_duration_stockout_run_threshold():
    near_zero = np.array([False] * 3 + [True] * (STOCKOUT_MIN_RUN_DAYS - 1) + [False])
    assert not _flag_duration_stockout_runs(near_zero).any()

    qualifying = np.array([False] * 2 + [True] * STOCKOUT_MIN_RUN_DAYS + [False])
    flags = _flag_duration_stockout_runs(qualifying)
    assert flags[2 : 2 + STOCKOUT_MIN_RUN_DAYS].all()
    assert not flags[0]
    assert not flags[-1]


def test_detect_stockout_oct_stretch_not_flagged_at_default_threshold():
    # ~3-week quiet period should not qualify at 45-day threshold.
    quantities = [8.0] * 10 + [0.0] * 22 + [9.0] * 10
    df = pd.DataFrame(
        {
            "demand_date": pd.date_range("2024-09-01", periods=len(quantities), freq="D"),
            "total_quantity": quantities,
        }
    )
    result = detect_stockout_windows(df, "P406816-LIKE", db_session=None)
    assert not result["is_stockout"].any()
