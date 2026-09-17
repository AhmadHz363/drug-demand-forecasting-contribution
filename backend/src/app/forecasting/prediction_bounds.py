"""Shared helpers for clipping unstable forecast spikes."""

from __future__ import annotations

import numpy as np

from app.forecasting.constants import (
    PREDICTION_CAP_RECENT_DAYS,
    STACKING_HORIZON_DRIFT_START_STEP,
    STACKING_MAX_HORIZON_GROWTH_INTERMITTENT,
)


def demand_prediction_cap(
    quantities: np.ndarray,
    *,
    recent_days: int = PREDICTION_CAP_RECENT_DAYS,
) -> float:
    """Upper bound from recent demand — stops outlier spikes from dominating the blend."""
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


def dampen_stacked_horizon_drift(
    predictions: np.ndarray,
    *,
    demand_segment: str,
    recent_level: float,
    horizon_steps: np.ndarray | None = None,
) -> np.ndarray:
    """
    Limit cumulative upward drift in stacked forecasts for bursty demand.

    SARIMA trend terms can accelerate through the back half of the horizon
    when blended via stacking; pull long-horizon steps toward recent level
    and cap growth relative to the day-1 forecast.
    """
    if demand_segment not in {"intermittent", "lumpy"}:
        return np.asarray(predictions, dtype=float)

    preds = np.asarray(predictions, dtype=float).copy()
    if preds.size == 0:
        return preds

    steps = (
        np.asarray(horizon_steps, dtype=int).reshape(-1)
        if horizon_steps is not None
        else np.arange(1, preds.size + 1, dtype=int)
    )
    anchor = max(float(recent_level), 1e-6)
    baseline = float(preds[0])
    span = max(int(preds.size - 1), 1)
    max_end_ratio = 1.0 + STACKING_MAX_HORIZON_GROWTH_INTERMITTENT

    for idx in range(preds.size):
        step = int(steps[idx])
        if step <= 1:
            continue

        growth_cap = baseline * (1.0 + (max_end_ratio - 1.0) * (step - 1) / span)
        capped = min(float(preds[idx]), growth_cap)

        if step > STACKING_HORIZON_DRIFT_START_STEP:
            blend = min(
                0.55,
                (step - STACKING_HORIZON_DRIFT_START_STEP)
                / max(preds.size - STACKING_HORIZON_DRIFT_START_STEP, 1)
                * 0.55,
            )
            preds[idx] = (1.0 - blend) * capped + blend * anchor
        else:
            preds[idx] = capped

    return np.clip(preds, 0.0, None)


def sanitize_ensemble_predictions(
    sarima: np.ndarray,
    lgbm: np.ndarray,
    classical: np.ndarray,
    cap: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clip all model outputs to a demand-derived cap before ensembling."""
    return (
        winsorize_predictions(sarima, cap),
        winsorize_predictions(lgbm, cap),
        winsorize_predictions(classical, cap),
    )
