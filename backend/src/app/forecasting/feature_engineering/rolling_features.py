"""Rolling statistics and trend features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.forecasting.constants import ROLLING_WINDOWS


def _rolling_slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    y = values.astype(float)
    if np.all(np.isnan(y)):
        return 0.0
    x = np.arange(len(y), dtype=float)
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return 0.0
    coeffs = np.polyfit(x[mask], y[mask], 1)
    return float(coeffs[0])


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append rolling statistics to a DataFrame sorted by ``demand_date``.

    Requires columns: total_quantity
    Rolling windows reset across coverage gaps (gap days treated as NaN).
    """
    out = df.copy()
    covered = (
        out["is_coverage_gap"].astype(int) == 0
        if "is_coverage_gap" in out.columns
        else pd.Series(True, index=out.index)
    )
    series = out["total_quantity"].astype(float).where(covered)
    covered_vals = series.dropna()
    median_qty = float(covered_vals.median()) if not covered_vals.empty else 0.0
    # Preceding rows only (no same-day leakage); NaNs break windows at gaps.
    preceding = series.shift(1)

    for window in ROLLING_WINDOWS:
        rolled = preceding.rolling(window=window, min_periods=7)
        mean_col = f"rolling_mean_{window}d"
        std_col = f"rolling_std_{window}d"
        min_col = f"rolling_min_{window}d"
        max_col = f"rolling_max_{window}d"
        cv_col = f"rolling_cv_{window}d"

        out[mean_col] = rolled.mean().fillna(median_qty)
        out[std_col] = rolled.std().fillna(0.0)
        out[min_col] = rolled.min().fillna(median_qty)
        out[max_col] = rolled.max().fillna(median_qty)
        out[cv_col] = np.where(
            out[mean_col] != 0,
            out[std_col] / out[mean_col],
            0.0,
        )

    out["trend_slope_28d"] = (
        preceding.rolling(window=28, min_periods=7)
        .apply(_rolling_slope, raw=True)
        .fillna(0.0)
    )

    std_28 = out["rolling_std_28d"].replace(0, np.nan)
    out["demand_zscore_28d"] = ((series.fillna(median_qty) - out["rolling_mean_28d"]) / std_28).fillna(0.0)

    for window in ROLLING_WINDOWS:
        rolling_mean = preceding.rolling(window=window, min_periods=7).mean()
        rolling_std = preceding.rolling(window=window, min_periods=7).std()
        threshold = rolling_mean + 2 * rolling_std
        spike_flag = (preceding > threshold).astype(float)
        out[f"spike_count_{window}d"] = (
            spike_flag.rolling(window=window, min_periods=1).sum().fillna(0.0)
        )
        out[f"spike_intensity_{window}d"] = (
            (preceding - rolling_mean)
            .clip(lower=0)
            .rolling(window=window, min_periods=1)
            .max()
            .fillna(0.0)
        )

    return out


def rolling_feature_columns() -> list[str]:
    cols: list[str] = []
    for window in ROLLING_WINDOWS:
        cols.extend(
            [
                f"rolling_mean_{window}d",
                f"rolling_std_{window}d",
                f"rolling_min_{window}d",
                f"rolling_max_{window}d",
                f"rolling_cv_{window}d",
            ]
        )
    cols.extend(["trend_slope_28d", "demand_zscore_28d"])
    for window in ROLLING_WINDOWS:
        cols.extend([f"spike_count_{window}d", f"spike_intensity_{window}d"])
    return cols
