"""Build feature rows for recursive multi-step forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.forecasting.constants import LAG_DAYS, ROLLING_WINDOWS
from app.forecasting.feature_engineering.external_features import (
    DEFAULT_BED_OCCUPANCY,
    DEFAULT_WEEKLY_SURGERY_COUNT,
    external_feature_columns,
)
from app.forecasting.feature_engineering.rolling_features import _rolling_slope
from app.forecasting.feature_engineering.supplier_features import (
    DEFAULT_AVG_LEAD_TIME,
    DEFAULT_LEAD_TIME_STD,
    DEFAULT_RELIABILITY,
    supplier_feature_columns,
)
from app.forecasting.feature_engineering.temporal_features import add_temporal_features


def _rolling_stats(preceding: np.ndarray, window: int, median_qty: float) -> dict[str, float]:
    if preceding.size == 0:
        return {
            "mean": median_qty,
            "std": 0.0,
            "min": median_qty,
            "max": median_qty,
            "cv": 0.0,
        }

    tail = preceding[-window:] if preceding.size >= window else preceding
    mean_val = float(np.mean(tail))
    std_val = float(np.std(tail, ddof=0))
    min_val = float(np.min(tail))
    max_val = float(np.max(tail))
    cv_val = float(std_val / mean_val) if mean_val != 0 else 0.0
    return {
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "cv": cv_val,
    }


def _spike_features(preceding: np.ndarray, window: int) -> dict[str, float]:
    if preceding.size < 2:
        return {f"spike_count_{window}d": 0.0, f"spike_intensity_{window}d": 0.0}

    tail = preceding[-window:] if preceding.size >= window else preceding
    stats_tail = preceding[-(window + 1) : -1] if preceding.size > 1 else preceding[:-1]
    if stats_tail.size < 7:
        rolling_mean = float(np.mean(preceding))
        rolling_std = float(np.std(preceding, ddof=0))
    else:
        stat_window = stats_tail[-window:] if stats_tail.size >= window else stats_tail
        rolling_mean = float(np.mean(stat_window))
        rolling_std = float(np.std(stat_window, ddof=0))

    threshold = rolling_mean + 2 * rolling_std
    spike_count = float(np.sum(tail > threshold))
    intensities = np.clip(tail - rolling_mean, 0, None)
    spike_intensity = float(np.max(intensities)) if intensities.size else 0.0
    return {
        f"spike_count_{window}d": spike_count,
        f"spike_intensity_{window}d": spike_intensity,
    }


def build_recursive_feature_row(
    demand_date: pd.Timestamp,
    quantity_history: np.ndarray,
    *,
    median_qty: float,
    covariate_row: pd.Series | None = None,
    forecast_step: int = 1,
) -> dict[str, float]:
    """
    Build one feature vector for a future demand date using known covariates
    and recursively updated demand history.
    """
    preceding = quantity_history.astype(float)
    row_df = pd.DataFrame({"demand_date": [demand_date.date()]})
    row_df = add_temporal_features(row_df)
    features: dict[str, float] = {
        col: float(row_df.iloc[0][col]) for col in row_df.columns if col != "demand_date"
    }

    gap_mask = False
    for lag in LAG_DAYS:
        col = f"lag_{lag}d"
        if preceding.size >= lag:
            features[col] = float(preceding[-lag])
        else:
            features[col] = median_qty
            gap_mask = True
    features["has_lag_gaps"] = int(gap_mask)

    for window in ROLLING_WINDOWS:
        stats = _rolling_stats(preceding, window, median_qty)
        features[f"rolling_mean_{window}d"] = stats["mean"]
        features[f"rolling_std_{window}d"] = stats["std"]
        features[f"rolling_min_{window}d"] = stats["min"]
        features[f"rolling_max_{window}d"] = stats["max"]
        features[f"rolling_cv_{window}d"] = stats["cv"]
        features.update(_spike_features(preceding, window))

    slope_window = preceding[-28:] if preceding.size >= 28 else preceding
    features["trend_slope_28d"] = _rolling_slope(slope_window)
    std_28 = features["rolling_std_28d"]
    current_qty = float(preceding[-1]) if preceding.size else median_qty
    features["demand_zscore_28d"] = (
        (current_qty - features["rolling_mean_28d"]) / std_28 if std_28 else 0.0
    )

    features["forecast_step"] = float(forecast_step)
    features["forecast_step_sin"] = float(np.sin(2 * np.pi * forecast_step / 30))
    features["forecast_step_cos"] = float(np.cos(2 * np.pi * forecast_step / 30))
    features["demand_filled"] = 1.0

    for col in external_feature_columns():
        if covariate_row is not None and col in covariate_row.index:
            features[col] = float(covariate_row[col])
        else:
            features[col] = (
                DEFAULT_BED_OCCUPANCY
                if col == "bed_occupancy_rate"
                else float(DEFAULT_WEEKLY_SURGERY_COUNT)
            )

    supplier_defaults = {
        "supplier_avg_lead_time": DEFAULT_AVG_LEAD_TIME,
        "supplier_lead_time_std": DEFAULT_LEAD_TIME_STD,
        "supplier_reliability_score": DEFAULT_RELIABILITY,
    }
    for col in supplier_feature_columns():
        if covariate_row is not None and col in covariate_row.index:
            features[col] = float(covariate_row[col])
        else:
            features[col] = supplier_defaults[col]

    # Flag columns added by add_external_features / add_supplier_features during
    # training — they must be present at prediction time so the feature matrix
    # produced here exactly matches self._feature_names stored by LightGBMModel.
    if covariate_row is not None and "external_features_is_default" in covariate_row.index:
        features["external_features_is_default"] = float(covariate_row["external_features_is_default"])
    else:
        features["external_features_is_default"] = 1.0

    if covariate_row is not None and "supplier_features_is_default" in covariate_row.index:
        features["supplier_features_is_default"] = float(covariate_row["supplier_features_is_default"])
    else:
        features["supplier_features_is_default"] = 1.0

    return features
