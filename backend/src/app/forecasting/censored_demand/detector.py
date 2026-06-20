"""Stockout window detection in demand time series."""

from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

_STOCKOUT_WINDOW_DAYS = 14


def detect_stockout_windows(
    df: pd.DataFrame,
    drug_code: str,
    db_session: Session,
) -> pd.DataFrame:
    """
    Marks rows as stockout periods.

    A stockout is defined as: total_quantity == 0 AND the drug had
    non-zero demand in the 14 days before or after (i.e. it is not
    a genuine zero-demand period).

    Adds column: is_stockout (bool)
    Also stores stockout_rate in df.attrs['stockout_rate'].
    """
    del db_session  # reserved for future inventory-integration hooks

    out = df.copy()
    if "total_quantity" not in out.columns:
        raise ValueError("DataFrame must contain a total_quantity column")

    qty = out["total_quantity"].astype(float)
    is_zero = qty == 0
    window_size = 2 * _STOCKOUT_WINDOW_DAYS + 1
    window_sum = qty.rolling(window=window_size, center=True, min_periods=1).sum()
    out["is_stockout"] = is_zero & (window_sum > 0)

    stockout_rate = float(out["is_stockout"].mean())
    stockout_count = int(out["is_stockout"].sum())
    out.attrs["stockout_rate"] = stockout_rate

    logger.info(
        "Stockout detection for %s: rate=%.4f days=%d",
        drug_code,
        stockout_rate,
        stockout_count,
    )
    return out
