"""Syntetos–Boylan demand pattern segmentation (ADI / CV²)."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from app.forecasting.constants import ADI_THRESHOLD, CV2_THRESHOLD, DEMAND_SEGMENTS

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


def classify_demand_segment(
    quantities: np.ndarray,
    *,
    adi_threshold: float = ADI_THRESHOLD,
    cv2_threshold: float = CV2_THRESHOLD,
) -> DemandSegment:
    """Bucket demand into smooth / intermittent / erratic / lumpy."""
    adi, cv2 = compute_adi_cv2(quantities)
    intermittent = adi > adi_threshold
    erratic = cv2 > cv2_threshold

    if not intermittent and not erratic:
        return "smooth"
    if intermittent and not erratic:
        return "intermittent"
    if not intermittent and erratic:
        return "erratic"
    return "lumpy"


def classify_demand_segment_from_frame(df: pd.DataFrame) -> DemandSegment:
    """Classify using raw receipts when available, else model demand column."""
    if "observed_quantity" in df.columns:
        series = df["observed_quantity"].astype(float)
    else:
        series = df["total_quantity"].astype(float)
    return classify_demand_segment(series.values)


def is_valid_demand_segment(segment: str) -> bool:
    return segment in DEMAND_SEGMENTS
