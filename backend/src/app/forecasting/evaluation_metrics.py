"""Shared forecast evaluation metrics."""

from __future__ import annotations

from collections import Counter

import numpy as np

from app.forecasting.constants import SEASONAL_NAIVE_PERIOD


def smape(actual: float, predicted: float) -> float:
    if actual == 0 and predicted == 0:
        return 0.0
    denominator = abs(actual) + abs(predicted)
    if denominator == 0:
        return 0.0
    return 200.0 * abs(actual - predicted) / denominator


def accuracy_from_smape(smape_val: float) -> float:
    """Convert symmetric MAPE (0–200%) to a 0–100% accuracy score (secondary metric)."""
    return max(0.0, min(100.0, 100.0 - smape_val / 2.0))


def seasonal_naive_errors(
    actuals: np.ndarray,
    *,
    seasonal_period: int = SEASONAL_NAIVE_PERIOD,
) -> np.ndarray:
    """Absolute errors of a weekly seasonal-naive one-step baseline."""
    values = np.asarray(actuals, dtype=float).reshape(-1)
    if len(values) <= seasonal_period:
        return np.array([], dtype=float)
    return np.abs(values[seasonal_period:] - values[:-seasonal_period])


def mase(
    actuals: np.ndarray,
    predicted: np.ndarray,
    *,
    seasonal_period: int = SEASONAL_NAIVE_PERIOD,
) -> float:
    """Mean absolute scaled error relative to a seasonal-naive baseline."""
    actual_arr = np.asarray(actuals, dtype=float).reshape(-1)
    pred_arr = np.asarray(predicted, dtype=float).reshape(-1)
    if actual_arr.size == 0:
        return 0.0

    mae = float(np.mean(np.abs(actual_arr - pred_arr)))
    naive_errors = seasonal_naive_errors(actual_arr, seasonal_period=seasonal_period)
    if naive_errors.size == 0:
        return 0.0 if mae == 0.0 else float("inf")

    naive_mae = float(np.mean(naive_errors))
    if naive_mae == 0.0:
        return 0.0 if mae == 0.0 else float("inf")
    return mae / naive_mae


def rmsse(
    actuals: np.ndarray,
    predicted: np.ndarray,
    *,
    seasonal_period: int = SEASONAL_NAIVE_PERIOD,
) -> float:
    """Root mean squared scaled error relative to a seasonal-naive baseline."""
    actual_arr = np.asarray(actuals, dtype=float).reshape(-1)
    pred_arr = np.asarray(predicted, dtype=float).reshape(-1)
    if actual_arr.size == 0:
        return 0.0

    mse = float(np.mean((actual_arr - pred_arr) ** 2))
    naive_errors = seasonal_naive_errors(actual_arr, seasonal_period=seasonal_period)
    if naive_errors.size == 0:
        return 0.0 if mse == 0.0 else float("inf")

    naive_mse = float(np.mean(naive_errors ** 2))
    if naive_mse == 0.0:
        return 0.0 if mse == 0.0 else float("inf")
    return float(np.sqrt(mse / naive_mse))


def pinball_loss(actual: float, predicted: float, quantile: float) -> float:
    """Quantile (pinball) loss for a single observation."""
    error = float(actual) - float(predicted)
    if error >= 0.0:
        return quantile * error
    return (quantile - 1.0) * error


def mean_pinball_loss(
    actuals: np.ndarray,
    predicted: np.ndarray,
    quantile: float,
) -> float:
    actual_arr = np.asarray(actuals, dtype=float).reshape(-1)
    pred_arr = np.asarray(predicted, dtype=float).reshape(-1)
    if actual_arr.size == 0:
        return 0.0
    return float(
        np.mean([pinball_loss(act, pred, quantile) for act, pred in zip(actual_arr, pred_arr)])
    )


def accuracy_skill_from_mase(mase_val: float) -> float:
    """Primary interpretable accuracy: max(0, (1 − MASE) × 100)."""
    if mase_val == float("inf") or mase_val != mase_val:
        return 0.0
    return max(0.0, min(100.0, (1.0 - mase_val) * 100.0))


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
