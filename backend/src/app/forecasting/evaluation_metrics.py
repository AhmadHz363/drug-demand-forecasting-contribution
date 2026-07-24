"""Shared forecast evaluation metrics."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from app.forecasting.constants import (
    ROLLING_VALIDATION_WINDOWS,
    SEASONAL_NAIVE_PERIOD,
    SMAPE_UNRELIABLE_ZERO_FRACTION,
)


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
    training_actuals: np.ndarray | None = None,
    seasonal_period: int = SEASONAL_NAIVE_PERIOD,
) -> float:
    """Mean absolute scaled error relative to a seasonal-naive baseline."""
    actual_arr = np.asarray(actuals, dtype=float).reshape(-1)
    pred_arr = np.asarray(predicted, dtype=float).reshape(-1)
    if actual_arr.size == 0:
        return 0.0

    mae = float(np.mean(np.abs(actual_arr - pred_arr)))
    scale_source = (
        np.asarray(training_actuals, dtype=float).reshape(-1)
        if training_actuals is not None
        else actual_arr
    )
    naive_errors = seasonal_naive_errors(scale_source, seasonal_period=seasonal_period)
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


def finite_mase_or_none(mase_val: float | None) -> float | None:
    """Return MASE for persistence; non-finite values (e.g. inf on sparse series) become None."""
    if mase_val is None or not math.isfinite(mase_val):
        return None
    return float(mase_val)


def build_evaluation_mask(
    actuals: np.ndarray,
    *,
    is_stockout: np.ndarray | None = None,
    is_coverage_gap: np.ndarray | None = None,
    sentinel_ratio_threshold: float = 0.05,
) -> np.ndarray:
    """
    Return a boolean mask of rows suitable for hold-out metric computation.

    Excludes censored-demand imputations, coverage gaps, and repeated median
    sentinel values.
    """
    actuals = np.asarray(actuals, dtype=float).reshape(-1)
    mask = np.ones(len(actuals), dtype=bool)

    if is_coverage_gap is not None:
        gaps = np.asarray(is_coverage_gap, dtype=bool).reshape(-1)
        if gaps.shape[0] == mask.shape[0]:
            mask &= ~gaps

    if is_stockout is not None:
        stockout = np.asarray(is_stockout, dtype=bool).reshape(-1)
        if stockout.shape[0] == mask.shape[0]:
            mask &= ~stockout

    # Non-finite actuals (coverage gaps left as NaN) are never scored.
    mask &= np.isfinite(actuals)

    scored = actuals[mask]
    if len(scored) > 0:
        freq = Counter(round(float(a), 4) for a in scored)
        most_common_val, count = freq.most_common(1)[0]
        if count >= 2 and count / len(scored) > sentinel_ratio_threshold:
            for idx, actual in enumerate(actuals):
                if mask[idx] and round(float(actual), 4) == most_common_val:
                    mask[idx] = False

    return mask


def zero_actual_fraction(actuals: np.ndarray) -> float:
    """Share of validation observations with zero (or near-zero) realized demand."""
    actual_arr = np.asarray(actuals, dtype=float).reshape(-1)
    if actual_arr.size == 0:
        return 0.0
    finite = actual_arr[np.isfinite(actual_arr)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(finite <= 0.0))


def smape_unreliable_for_series(
    actuals: np.ndarray,
    *,
    demand_segment: str | None = None,
    zero_fraction_threshold: float = SMAPE_UNRELIABLE_ZERO_FRACTION,
) -> bool:
    """
    True when sMAPE is likely misleading (zero-inflated intermittent demand).

    Intermittent/lumpy segments always treat sMAPE as supplementary because
    zero-actual days dominate the metric even when MASE remains near baseline.
    """
    if demand_segment in {"intermittent", "lumpy"}:
        return True
    return zero_actual_fraction(actuals) >= zero_fraction_threshold


def primary_validation_metric_for_segment(demand_segment: str | None) -> str:
    """Headline hold-out metric: MASE for sparse segments, MASE elsewhere too."""
    if demand_segment in {"intermittent", "lumpy"}:
        return "mase"
    return "mase"


def validation_metrics_note(
    *,
    demand_segment: str | None,
    zero_fraction: float,
    smape_unreliable: bool,
) -> str | None:
    if not smape_unreliable:
        return None
    if demand_segment in {"intermittent", "lumpy"}:
        return (
            f"MASE is the primary accuracy metric for {demand_segment} demand "
            f"({zero_fraction:.0%} zero-actual days in validation). sMAPE is "
            "shown for reference only — it inflates on zero-actual days when "
            "the forecast is nonzero."
        )
    return (
        f"sMAPE may be unreliable ({zero_fraction:.0%} zero-actual days in "
        "validation). Prefer MASE for this drug."
    )


def mean_smape(actuals: np.ndarray, predicted: np.ndarray) -> float | None:
    """Average symmetric MAPE over aligned daily (or aggregated) points."""
    actual_arr = np.asarray(actuals, dtype=float).reshape(-1)
    pred_arr = np.asarray(predicted, dtype=float).reshape(-1)
    if actual_arr.size == 0:
        return None
    return float(np.mean([smape(float(a), float(p)) for a, p in zip(actual_arr, pred_arr)]))


def rolling_window_totals(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sliding-window sums over *values*.

    Returns (totals, end_indices) where end_indices[i] is the last row index
    included in totals[i].
    """
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size < window:
        return np.array([], dtype=float), np.array([], dtype=int)
    totals = np.convolve(arr, np.ones(window, dtype=float), mode="valid")
    end_indices = np.arange(window - 1, arr.size, dtype=int)
    return totals, end_indices


def _rolling_window_pairs(
    actuals: np.ndarray,
    predicted: np.ndarray,
    *,
    window: int,
    include_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate aligned actual/predicted pairs into sliding-window totals."""
    actual_arr = np.asarray(actuals, dtype=float).reshape(-1)
    pred_arr = np.asarray(predicted, dtype=float).reshape(-1)
    if actual_arr.size != pred_arr.size or actual_arr.size < window:
        return np.array([], dtype=float), np.array([], dtype=float)

    if include_mask is None:
        include_mask = np.ones(actual_arr.size, dtype=bool)
    else:
        include_mask = np.asarray(include_mask, dtype=bool).reshape(-1)
        if include_mask.size != actual_arr.size:
            return np.array([], dtype=float), np.array([], dtype=float)

    act_totals: list[float] = []
    pred_totals: list[float] = []
    for end in range(window - 1, actual_arr.size):
        start = end - window + 1
        if not include_mask[start : end + 1].all():
            continue
        if not np.all(np.isfinite(actual_arr[start : end + 1])):
            continue
        if not np.all(np.isfinite(pred_arr[start : end + 1])):
            continue
        act_totals.append(float(actual_arr[start : end + 1].sum()))
        pred_totals.append(float(pred_arr[start : end + 1].sum()))
    return np.asarray(act_totals, dtype=float), np.asarray(pred_totals, dtype=float)


def rolling_validation_metrics(
    actuals: np.ndarray,
    predicted: np.ndarray,
    *,
    stockout_flags: np.ndarray | None = None,
    training_actuals: np.ndarray | None = None,
    windows: tuple[int, ...] = ROLLING_VALIDATION_WINDOWS,
) -> dict[str, float | None]:
    """
    sMAPE and MASE on sliding 7-day / 30-day demand totals.

    Returns keys like ``smape_7day_full``, ``mase_7day_normal``, etc.
    """
    actual_arr = np.asarray(actuals, dtype=float).reshape(-1)
    pred_arr = np.asarray(predicted, dtype=float).reshape(-1)
    if actual_arr.size != pred_arr.size:
        return {}

    full_mask = np.isfinite(actual_arr) & np.isfinite(pred_arr)
    if stockout_flags is not None:
        stockout = np.asarray(stockout_flags, dtype=bool).reshape(-1)
        if stockout.size == actual_arr.size:
            normal_mask = full_mask & ~stockout
        else:
            normal_mask = full_mask
    else:
        normal_mask = full_mask

    training_rolled: dict[int, np.ndarray] = {}
    if training_actuals is not None:
        train_arr = np.asarray(training_actuals, dtype=float).reshape(-1)
        for window in windows:
            rolled, _ = rolling_window_totals(train_arr, window)
            training_rolled[window] = rolled

    results: dict[str, float | None] = {}
    for window in windows:
        act_full, pred_full = _rolling_window_pairs(
            actual_arr,
            pred_arr,
            window=window,
            include_mask=full_mask,
        )
        act_normal, pred_normal = _rolling_window_pairs(
            actual_arr,
            pred_arr,
            window=window,
            include_mask=normal_mask,
        )

        suffix = f"{window}day"
        results[f"smape_{suffix}_full"] = mean_smape(act_full, pred_full)
        results[f"mase_{suffix}_full"] = (
            finite_mase_or_none(
                mase(
                    act_full,
                    pred_full,
                    training_actuals=training_rolled.get(window),
                    seasonal_period=min(SEASONAL_NAIVE_PERIOD, max(window, 1)),
                )
            )
            if act_full.size
            else None
        )
        results[f"smape_{suffix}_normal"] = mean_smape(act_normal, pred_normal)
        if results[f"smape_{suffix}_normal"] is None:
            results[f"smape_{suffix}_normal"] = results[f"smape_{suffix}_full"]
        results[f"mase_{suffix}_normal"] = (
            finite_mase_or_none(
                mase(
                    act_normal,
                    pred_normal,
                    training_actuals=training_rolled.get(window),
                    seasonal_period=min(SEASONAL_NAIVE_PERIOD, max(window, 1)),
                )
            )
            if act_normal.size
            else None
        )
        if results[f"mase_{suffix}_normal"] is None:
            results[f"mase_{suffix}_normal"] = results[f"mase_{suffix}_full"]
    return results


def mase_beats_baseline(mase_val: float | None) -> bool | None:
    """True when MASE < 1.0 (model beats seasonal-naive baseline)."""
    if mase_val is None or not math.isfinite(mase_val):
        return None
    return mase_val < 1.0
