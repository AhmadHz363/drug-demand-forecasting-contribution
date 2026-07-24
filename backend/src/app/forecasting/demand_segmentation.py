"""Syntetos–Boylan demand pattern segmentation (ADI / CV²)."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from app.forecasting.constants import (
    ADI_THRESHOLD,
    CV2_THRESHOLD,
    DEMAND_SEGMENTS,
    SEGMENTATION_LOW_DEMAND_RATIO,
    SEGMENTATION_MIN_DORMANT_RUN_DAYS,
    SEGMENTATION_WEEKLY_DIP_CV2,
)

DemandSegment = Literal["smooth", "intermittent", "erratic", "lumpy"]


def compute_adi_cv2(quantities: np.ndarray) -> tuple[float, float]:
    """
    Compute average inter-demand interval (ADI) and squared CV of non-zero demand.

    ADI = total periods / number of non-zero demand periods.
    CV² = (std / mean)² computed on strictly positive demand days only.
    """
    qty = np.asarray(quantities, dtype=float).reshape(-1)
    n_periods = len(qty)
    if n_periods == 0:
        return 0.0, 0.0

    nonzero = qty > 0.0
    n_nonzero = int(nonzero.sum())
    if n_nonzero == 0:
        return float(n_periods), 0.0

    adi = float(n_periods / n_nonzero)
    nonzero_vals = qty[nonzero]
    mean_nz = float(np.mean(nonzero_vals))
    if mean_nz <= 0.0:
        return adi, 0.0

    cv2 = float((np.std(nonzero_vals, ddof=0) / mean_nz) ** 2)
    return adi, cv2


def _normalize_for_segmentation(quantities: np.ndarray) -> np.ndarray:
    """
    Treat near-zero demand as zero for ADI so stockout/collapse periods
    do not mask bursty intermittent patterns.
    """
    qty = np.asarray(quantities, dtype=float).reshape(-1)
    qty = qty[np.isfinite(qty)]
    if qty.size == 0:
        return qty

    positive = qty > 0.0
    if not positive.any():
        return qty

    median_nz = float(np.median(qty[positive]))
    floor = max(median_nz * SEGMENTATION_LOW_DEMAND_RATIO, 1.0)
    normalized = qty.copy()
    normalized[normalized < floor] = 0.0
    return normalized


def max_consecutive_zero_days(quantities: np.ndarray) -> int:
    """Longest run of zero (or near-zero) demand days."""
    qty = _normalize_for_segmentation(quantities)
    max_run = 0
    current = 0
    for value in qty:
        if value <= 0.0:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def weekly_dow_cv2(quantities: np.ndarray) -> float:
    """
    Coefficient of variation across 7-day buckets (proxy for weekly seasonality).

    High values indicate recurring intra-week dips/spikes even when ADI is low.
    """
    qty = _normalize_for_segmentation(quantities)
    if qty.size < 14:
        return 0.0
    positive = qty > 0.0
    if not positive.any():
        return 0.0

    bucket_means: list[float] = []
    for offset in range(7):
        bucket = qty[offset::7]
        bucket = bucket[bucket > 0.0]
        if bucket.size:
            bucket_means.append(float(np.mean(bucket)))
    if len(bucket_means) < 3:
        return 0.0

    mean_level = float(np.mean(bucket_means))
    if mean_level <= 0.0:
        return 0.0
    return float((np.std(bucket_means, ddof=0) / mean_level) ** 2)


def classify_demand_segment(
    quantities: np.ndarray,
    *,
    adi_threshold: float = ADI_THRESHOLD,
    cv2_threshold: float = CV2_THRESHOLD,
) -> DemandSegment:
    """Bucket demand into smooth / intermittent / erratic / lumpy."""
    normalized = _normalize_for_segmentation(quantities)
    adi, cv2 = compute_adi_cv2(normalized)
    intermittent = adi > adi_threshold
    erratic = cv2 > cv2_threshold

    # Stockout-then-burst histories: long dormant runs are not smooth demand.
    if (
        not intermittent
        and max_consecutive_zero_days(normalized) >= SEGMENTATION_MIN_DORMANT_RUN_DAYS
    ):
        intermittent = True

    # Weekly troughs/spikes visible in day-of-week means -> erratic, not smooth.
    if not intermittent and not erratic:
        if weekly_dow_cv2(normalized) >= SEGMENTATION_WEEKLY_DIP_CV2:
            erratic = True

    if not intermittent and not erratic:
        return "smooth"
    if intermittent and not erratic:
        return "intermittent"
    if not intermittent and erratic:
        return "erratic"
    return "lumpy"


def classify_demand_segment_from_frame(df: pd.DataFrame) -> DemandSegment:
    """Classify using covered observed demand only (exclude gaps and stockouts)."""
    working = df
    if "is_coverage_gap" in df.columns:
        working = df.loc[df["is_coverage_gap"].astype(int) == 0]
        if working.empty:
            working = df
    if "is_stockout" in working.columns:
        non_stockout = working.loc[~working["is_stockout"].astype(bool)]
        if not non_stockout.empty:
            working = non_stockout
    if "observed_quantity" in working.columns:
        series = working["observed_quantity"].astype(float)
    else:
        series = working["total_quantity"].astype(float)
    values = series.replace([np.inf, -np.inf], np.nan).dropna().values
    return classify_demand_segment(values)


def is_valid_demand_segment(segment: str) -> bool:
    return segment in DEMAND_SEGMENTS
