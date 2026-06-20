"""Shared forecast evaluation metrics."""

from __future__ import annotations

from collections import Counter

import numpy as np


def smape(actual: float, predicted: float) -> float:
    if actual == 0 and predicted == 0:
        return 0.0
    denominator = abs(actual) + abs(predicted)
    if denominator == 0:
        return 0.0
    return 200.0 * abs(actual - predicted) / denominator


def accuracy_from_smape(smape_val: float) -> float:
    """Convert symmetric MAPE (0–200%) to a 0–100% accuracy score."""
    return max(0.0, min(100.0, 100.0 - smape_val / 2.0))


def build_evaluation_mask(
    actuals: np.ndarray,
    *,
    is_stockout: np.ndarray | None = None,
    sentinel_ratio_threshold: float = 0.05,
) -> np.ndarray:
    """
    Return a boolean mask of rows suitable for hold-out metric computation.

    Excludes censored-demand imputations and repeated median sentinel values.
    """
    actuals = np.asarray(actuals, dtype=float).reshape(-1)
    mask = np.ones(len(actuals), dtype=bool)

    if is_stockout is not None:
        stockout = np.asarray(is_stockout, dtype=bool).reshape(-1)
        if stockout.shape[0] == mask.shape[0]:
            mask &= ~stockout

    if len(actuals) > 0:
        freq = Counter(round(float(a), 4) for a in actuals)
        most_common_val, count = freq.most_common(1)[0]
        if count / len(actuals) > sentinel_ratio_threshold:
            for idx, actual in enumerate(actuals):
                if round(float(actual), 4) == most_common_val:
                    mask[idx] = False

    return mask
