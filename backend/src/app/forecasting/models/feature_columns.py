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
    "is_coverage_gap",
    "demand_date",
    "demand_filled",
    "external_features_is_default",
    "supplier_features_is_default",
    "has_census",
    "has_supplier",
}

LGBM_CATEGORICAL_FEATURES = ["has_lag_gaps"]


def external_feature_columns() -> list[str]:
    return ["bed_occupancy_rate", "weekly_surgery_count"]


def _drop_all_nan_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns and not df[col].isna().all()]


def tft_known_reals(df: pd.DataFrame | None = None) -> list[str]:
    cols = temporal_feature_columns() + external_feature_columns()
    if df is None:
        return cols
    return _drop_all_nan_columns(df, cols)


def tft_unknown_reals(df: pd.DataFrame | None = None) -> list[str]:
    lag_cols = [c for c in lag_feature_columns() if c != "has_lag_gaps"]
    cols = lag_cols + rolling_feature_columns()
    if df is None:
        return cols
    return _drop_all_nan_columns(df, cols)


def tft_static_reals(df: pd.DataFrame | None = None) -> list[str]:
    cols = supplier_feature_columns()
    if df is None:
        return cols
    return _drop_all_nan_columns(df, cols)


def lgbm_feature_columns(df: pd.DataFrame) -> list[str]:
    """Model features only — drop metadata and NaN-only default covariates."""
    cols = [col for col in df.columns if col not in METADATA_COLUMNS]
    usable: list[str] = []
    for col in cols:
        series = df[col]
        if col in external_feature_columns() and "external_features_is_default" in df.columns:
            if float(df["external_features_is_default"].astype(float).mean()) >= 0.999:
                continue
        if col in supplier_feature_columns() and "supplier_features_is_default" in df.columns:
            if float(df["supplier_features_is_default"].astype(float).mean()) >= 0.999:
                continue
        if series.isna().all():
            continue
        usable.append(col)
    return usable


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
