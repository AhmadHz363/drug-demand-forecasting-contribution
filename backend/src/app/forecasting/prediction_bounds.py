"""Shared helpers for clipping unstable forecast spikes."""

from __future__ import annotations

import numpy as np

from app.forecasting.constants import PREDICTION_CAP_RECENT_DAYS


def demand_prediction_cap(
    quantities: np.ndarray,
    *,
    recent_days: int = PREDICTION_CAP_RECENT_DAYS,
) -> float:
    """Upper bound from recent demand — stops TFT spikes from dominating the blend."""
    qty = np.asarray(quantities, dtype=float)
    tail = qty[-recent_days:] if len(qty) > recent_days else qty
    positive = tail[np.isfinite(tail) & (tail > 0)]
    if positive.size == 0:
        finite = qty[np.isfinite(qty)]
        fallback = float(np.max(finite)) if finite.size else 50.0
        return max(fallback, 1.0)
    p95 = float(np.percentile(positive, 95))
    median = float(np.median(positive))
    recent_peak = float(np.max(positive))
    return max(min(p95 * 1.15, recent_peak * 1.2), median * 2.5, 1.0)


def winsorize_predictions(preds: np.ndarray, cap: float) -> np.ndarray:
    return np.clip(np.asarray(preds, dtype=float), 0.0, cap)


def sanitize_ensemble_predictions(
    sarima: np.ndarray,
    lgbm: np.ndarray,
    tft: np.ndarray,
    cap: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clip all model outputs to a demand-derived cap before ensembling."""
    return (
        winsorize_predictions(sarima, cap),
        winsorize_predictions(lgbm, cap),
        winsorize_predictions(tft, cap),
    )
