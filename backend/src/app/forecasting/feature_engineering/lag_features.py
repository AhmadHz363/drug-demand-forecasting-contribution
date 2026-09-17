"""Lag features — past demand at fixed offsets."""

from __future__ import annotations

import pandas as pd

from app.forecasting.constants import LAG_DAYS


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append lag columns to a DataFrame sorted by ``demand_date``.

    Requires columns: total_quantity
    Resets across coverage gaps: lag_365d is valid only when the prior date
    is covered; otherwise has_lag_gaps=1.
    """
    out = df.copy()
    covered = (
        out["is_coverage_gap"].astype(int) == 0
        if "is_coverage_gap" in out.columns
        else pd.Series(True, index=out.index)
    )
    # Use NaN on gap rows so shifts do not leak zeros across uncovered years.
    qty = out["total_quantity"].astype(float).where(covered)
    covered_qty = qty.dropna()
    median_qty = float(covered_qty.median()) if not covered_qty.empty else 0.0
    gap_mask = pd.Series(False, index=out.index)

    for lag in LAG_DAYS:
        col = f"lag_{lag}d"
        shifted = qty.shift(lag)
        # Also invalid when the source day was a coverage gap.
        source_covered = covered.shift(lag)
        source_covered = source_covered.where(source_covered.notna(), False).astype(bool)
        invalid = shifted.isna() | ~source_covered
        gap_mask = gap_mask | invalid
        out[col] = shifted.where(~invalid, median_qty)

    out["has_lag_gaps"] = gap_mask.astype(int)
    return out


def lag_feature_columns() -> list[str]:
    return [f"lag_{lag}d" for lag in LAG_DAYS] + ["has_lag_gaps"]
