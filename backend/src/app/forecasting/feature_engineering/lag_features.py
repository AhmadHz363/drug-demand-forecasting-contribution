"""Lag features — past demand at fixed offsets."""

from __future__ import annotations

import pandas as pd

from app.forecasting.constants import LAG_DAYS


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append lag columns to a DataFrame sorted by ``demand_date``.

    Requires columns: total_quantity
    """
    out = df.copy()
    median_qty = float(out["total_quantity"].median())
    gap_mask = pd.Series(False, index=out.index)

    for lag in LAG_DAYS:
        col = f"lag_{lag}d"
        shifted = out["total_quantity"].shift(lag)
        gap_mask = gap_mask | shifted.isna()
        out[col] = shifted.fillna(median_qty)

    out["has_lag_gaps"] = gap_mask.astype(int)
    return out


def lag_feature_columns() -> list[str]:
    return [f"lag_{lag}d" for lag in LAG_DAYS] + ["has_lag_gaps"]
