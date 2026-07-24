"""Stockout window detection in demand time series."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.forecasting.constants import (
    STOCKOUT_MIN_RUN_DAYS,
    STOCKOUT_NEAR_ZERO_THRESHOLD,
)

logger = logging.getLogger(__name__)


def _flag_duration_stockout_runs(
    near_zero: np.ndarray,
    *,
    min_run_days: int = STOCKOUT_MIN_RUN_DAYS,
) -> np.ndarray:
    """
    Flag days inside consecutive near-zero runs that meet the minimum duration.

    Isolated or short zero-demand gaps are genuine no-demand and are not flagged.
    """
    flags = np.zeros(len(near_zero), dtype=bool)
    run_start: int | None = None
    for idx, is_low in enumerate(near_zero):
        if is_low:
            if run_start is None:
                run_start = idx
            continue
        if run_start is not None:
            run_len = idx - run_start
            if run_len >= min_run_days:
                flags[run_start:idx] = True
            run_start = None
    if run_start is not None:
        run_len = len(near_zero) - run_start
        if run_len >= min_run_days:
            flags[run_start:] = True
    return flags


def detect_stockout_windows(
    df: pd.DataFrame,
    drug_code: str,
    db_session: Session,
) -> pd.DataFrame:
    """
    Marks rows as stockout periods.

    A stockout is a supply-side outage: an extended consecutive run of zero or
    near-zero demand (see ``STOCKOUT_NEAR_ZERO_THRESHOLD``) lasting at least
    ``STOCKOUT_MIN_RUN_DAYS`` days.  Isolated quiet days are not stockouts.

    Adds column: is_stockout (bool)
    Also stores stockout_rate in df.attrs['stockout_rate'].
    """
    del db_session  # reserved for future inventory-integration hooks

    out = df.copy()
    if "total_quantity" not in out.columns:
        raise ValueError("DataFrame must contain a total_quantity column")

    qty = out["total_quantity"].astype(float)
    if "is_coverage_gap" in out.columns:
        qty = qty.where(out["is_coverage_gap"].astype(int) == 0)

    near_zero = qty.isna() | (qty <= STOCKOUT_NEAR_ZERO_THRESHOLD)
    near_zero = near_zero.fillna(False).to_numpy(dtype=bool)
    out["is_stockout"] = _flag_duration_stockout_runs(near_zero)

    if "is_coverage_gap" in out.columns:
        out.loc[out["is_coverage_gap"].astype(int) == 1, "is_stockout"] = False

    covered = (
        out["is_coverage_gap"].astype(int) == 0
        if "is_coverage_gap" in out.columns
        else pd.Series(True, index=out.index)
    )
    stockout_rate = float(out.loc[covered, "is_stockout"].mean()) if covered.any() else 0.0
    stockout_count = int(out["is_stockout"].sum())
    out.attrs["stockout_rate"] = stockout_rate

    logger.info(
        "Stockout detection for %s: rate=%.4f days=%d (min_run=%d)",
        drug_code,
        stockout_rate,
        stockout_count,
        STOCKOUT_MIN_RUN_DAYS,
    )
    return out
