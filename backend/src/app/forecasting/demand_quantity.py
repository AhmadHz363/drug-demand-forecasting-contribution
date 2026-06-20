"""Normalize net receipt quantities to consumption demand.

Negative daily totals often reflect inter-department transfers.  Those
movements still consume stock and should be modeled as positive demand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def as_consumption_demand(values: np.ndarray | pd.Series | float) -> np.ndarray:
    """Map net quantity to non-negative consumption (``abs`` of the net)."""
    return np.abs(np.asarray(values, dtype=float))


def apply_consumption_demand(
    df: pd.DataFrame,
    *,
    column: str = "total_quantity",
) -> pd.DataFrame:
    """Return a copy with the target column expressed as consumption demand."""
    if column not in df.columns:
        return df
    out = df.copy()
    out[column] = as_consumption_demand(out[column])
    return out
