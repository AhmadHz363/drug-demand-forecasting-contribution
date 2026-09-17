"""Evaluation metrics for SHIELD-XR (WAPE-aligned)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true).sum()
    if denom == 0:
        return float("nan")
    return float(np.abs(y_true - y_pred).sum() / denom)


def accuracy_from_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    value = wape(y_true, y_pred)
    if np.isnan(value):
        return 0.0
    return max(0.0, 1.0 - value)


def smape_from_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Map WAPE to a 0–200 sMAPE-like scale for legacy API compatibility."""
    acc = accuracy_from_wape(y_true, y_pred)
    return max(0.0, (1.0 - acc) * 200.0)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, name: str = "model") -> dict[str, float | str]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    w = wape(y_true, y_pred)
    acc = accuracy_from_wape(y_true, y_pred)
    return {
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "WAPE": w,
        "Accuracy": acc,
    }
