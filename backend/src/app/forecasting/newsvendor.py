"""Criticality-aware stocking quantile from VEN class (Phase 3)."""

from __future__ import annotations

import numpy as np

from app.forecasting.constants import NEWSVENDOR_COST_RATIO

DEFAULT_VEN_CLASS = "N"


def resolve_ven_class(raw: str | None) -> str:
    """Normalize VEN class to V, E, or N."""
    if not raw:
        return DEFAULT_VEN_CLASS
    letter = raw.strip().upper()[:1]
    if letter in NEWSVENDOR_COST_RATIO:
        return letter
    return DEFAULT_VEN_CLASS


def operating_quantile(ven_class: str) -> float:
    """
    Newsvendor critical fractile: cost_ratio / (1 + cost_ratio).

    Vital drugs target a higher service level (closer to P90).
    """
    ratio = NEWSVENDOR_COST_RATIO.get(ven_class, NEWSVENDOR_COST_RATIO[DEFAULT_VEN_CLASS])
    return float(ratio / (1.0 + ratio))


def interpolate_forecast_quantile(
    p10: float,
    p50: float,
    p90: float,
    quantile: float,
) -> float:
    """Piecewise-linear interpolation across P10–P50–P90."""
    q = float(np.clip(quantile, 0.0, 1.0))
    p10_v = max(float(p10), 0.0)
    p50_v = max(float(p50), 0.0)
    p90_v = max(float(p90), 0.0)

    if q <= 0.10:
        return p10_v
    if q >= 0.90:
        return p90_v
    if q <= 0.50:
        return p10_v + (p50_v - p10_v) * (q - 0.10) / 0.40
    return p50_v + (p90_v - p50_v) * (q - 0.50) / 0.40


def recommended_quantities(
    p10_values: np.ndarray,
    p50_values: np.ndarray,
    p90_values: np.ndarray,
    ven_class: str,
) -> tuple[float, np.ndarray]:
    """Return operating quantile level and per-day recommended order quantities."""
    op_q = operating_quantile(ven_class)
    recommended = np.array(
        [
            interpolate_forecast_quantile(p10, p50, p90, op_q)
            for p10, p50, p90 in zip(p10_values, p50_values, p90_values)
        ],
        dtype=float,
    )
    return op_q, recommended
