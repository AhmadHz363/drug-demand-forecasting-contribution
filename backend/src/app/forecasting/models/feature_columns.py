"""Shared feature column helpers for forecasting models."""

from __future__ import annotations

import pandas as pd

from app.forecasting.feature_engineering.lag_features import lag_feature_columns
from app.forecasting.feature_engineering.rolling_features import rolling_feature_columns
from app.forecasting.feature_engineering.supplier_features import supplier_feature_columns
from app.forecasting.feature_engineering.temporal_features import temporal_feature_columns

METADATA_COLUMNS = {
    "total_quantity",
    "observed_quantity",
    "em_corrected_quantity",
    "correction_method",
    "is_stockout",
    "demand_date",
}

LGBM_CATEGORICAL_FEATURES = ["has_lag_gaps"]


def external_feature_columns() -> list[str]:
    return ["bed_occupancy_rate", "weekly_surgery_count"]


def tft_known_reals() -> list[str]:
    return temporal_feature_columns() + external_feature_columns()


def tft_unknown_reals() -> list[str]:
    lag_cols = [c for c in lag_feature_columns() if c != "has_lag_gaps"]
    return lag_cols + rolling_feature_columns()


def tft_static_reals() -> list[str]:
    return supplier_feature_columns()


def lgbm_feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in METADATA_COLUMNS]


def ensure_demand_date_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.name == "demand_date" or isinstance(df.index, pd.DatetimeIndex):
        out = df.copy()
        out.index = pd.to_datetime(out.index)
        out.index.name = "demand_date"
        return out.sort_index()
    if "demand_date" in df.columns:
        out = df.set_index("demand_date")
        out.index = pd.to_datetime(out.index)
        out.index.name = "demand_date"
        return out.sort_index()
    raise ValueError("DataFrame must use demand_date as index or column")


def forecast_dates_from_index(index: pd.Index, horizon_days: int) -> list[pd.Timestamp]:
    last_date = pd.Timestamp(index[-1])
    return [last_date + pd.Timedelta(days=offset) for offset in range(1, horizon_days + 1)]
