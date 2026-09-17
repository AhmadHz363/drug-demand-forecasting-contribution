"""Segment- and volatility-aware model hyperparameter adaptation (Phase 2)."""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np
import pandas as pd

from app.forecasting.constants import (
    CV2_THRESHOLD,
    LGBM_EARLY_STOPPING_ROUNDS,
    LGBM_LEARNING_RATE,
    LGBM_MAX_DEPTH,
    LGBM_N_ESTIMATORS,
    LGBM_NUM_LEAVES,
    LGBM_RECURSIVE_ANCHOR_WEIGHT,
    SARIMA_FORECAST_SHRINKAGE,
    SARIMA_ORDER,
    SARIMA_SEASONAL_ORDER,
    SARIMA_TRAIN_DAYS,
    SARIMA_TRAIN_DAYS_SHORT,
)

# Thread-safe per-drug SARIMA order cache.  Populated on first auto_arima call
# and reused for all subsequent walk-forward fold models for the same drug,
# reducing expensive grid searches from O(4 × n_splits) to O(1) per drug.
_sarima_order_cache: dict[str, tuple[tuple, tuple]] = {}
_sarima_order_cache_lock = threading.Lock()

SEGMENT_ENCODING = {
    "smooth": 0.0,
    "intermittent": 1.0,
    "erratic": 2.0,
    "lumpy": 3.0,
    "global": 0.0,
}


def recent_cv2(series: np.ndarray, *, window: int = 28) -> float:
    """Squared CV of non-zero demand in the recent tail."""
    values = np.asarray(series, dtype=float).reshape(-1)
    if values.size == 0:
        return 0.0
    tail = values[-window:] if values.size >= window else values
    nonzero = tail[tail > 0.0]
    if nonzero.size == 0:
        return 0.0
    mean_nz = float(np.mean(nonzero))
    if mean_nz <= 0.0:
        return 0.0
    return float((np.std(nonzero, ddof=0) / mean_nz) ** 2)


def adaptive_sarima_shrinkage(cv2: float) -> float:
    """Smooth series shrink more; volatile/lumpy series shrink less."""
    if cv2 <= CV2_THRESHOLD:
        boost = 1.0 - (cv2 / max(CV2_THRESHOLD, 1e-6))
        return float(min(0.55, SARIMA_FORECAST_SHRINKAGE + 0.15 * boost))
    excess = min((cv2 / max(CV2_THRESHOLD, 1e-6)) - 1.0, 2.0)
    return float(max(0.15, SARIMA_FORECAST_SHRINKAGE - 0.12 * excess))


def adaptive_lgbm_anchor_weight(cv2: float) -> float:
    """High volatility → less anchoring to recent median."""
    if cv2 <= CV2_THRESHOLD:
        return float(min(0.35, LGBM_RECURSIVE_ANCHOR_WEIGHT + 0.05))
    excess = min((cv2 / max(CV2_THRESHOLD, 1e-6)) - 1.0, 2.0)
    return float(max(0.10, LGBM_RECURSIVE_ANCHOR_WEIGHT - 0.08 * excess))


def select_sarima_train_days(
    demand_segment: str,
    series_length: int,
    *,
    cv2: Optional[float] = None,
) -> int:
    """
    Prefer shorter windows for intermittent/lumpy or high-CV² series.
    Smooth series limited to 1 year to avoid over-fitting on old data.
    """
    # Smooth series: max 1 year (365 days)
    # Recent patterns more reliable than very old data for stable demand
    if demand_segment == "smooth":
        limit = min(365, SARIMA_TRAIN_DAYS)
        return min(limit, series_length)
    
    # Intermittent/lumpy: can use up to 2 years
    # Need more history to capture rare demand events
    if demand_segment in {"intermittent", "lumpy"}:
        use_short = cv2 is not None and cv2 > CV2_THRESHOLD
        limit = SARIMA_TRAIN_DAYS_SHORT if use_short else SARIMA_TRAIN_DAYS
        return min(limit, series_length)
    
    # Erratic: middle ground (1.5 years max)
    if demand_segment == "erratic":
        limit = min(545, SARIMA_TRAIN_DAYS)  # ~1.5 years
        return min(limit, series_length)
    
    # Fallback: use constants
    use_short = cv2 is not None and cv2 > CV2_THRESHOLD
    limit = SARIMA_TRAIN_DAYS_SHORT if use_short else SARIMA_TRAIN_DAYS
    return min(limit, series_length)


def lgbm_training_params(demand_segment: str) -> dict[str, int | float]:
    """Stronger regularization for sparse intermittent/lumpy demand."""
    if demand_segment in {"intermittent", "lumpy"}:
        # Slightly relaxed regularization to capture more patterns
        return {
            "num_leaves": 20,
            "max_depth": 5,
            "n_estimators": 400,
            "learning_rate": LGBM_LEARNING_RATE,
            "early_stopping_rounds": LGBM_EARLY_STOPPING_ROUNDS,
        }
    if demand_segment == "erratic":
        return {
            "num_leaves": 26,
            "max_depth": 5,
            "n_estimators": 450,
            "learning_rate": LGBM_LEARNING_RATE,
            "early_stopping_rounds": LGBM_EARLY_STOPPING_ROUNDS,
        }
    return {
        "num_leaves": LGBM_NUM_LEAVES,
        "max_depth": LGBM_MAX_DEPTH,
        "n_estimators": LGBM_N_ESTIMATORS,
        "learning_rate": LGBM_LEARNING_RATE,
        "early_stopping_rounds": LGBM_EARLY_STOPPING_ROUNDS,
    }


def search_sarima_orders(
    series: pd.Series,
    *,
    max_p: int = 2,
    max_q: int = 2,
) -> tuple[tuple[int, int, int], tuple[int, int, int, int]]:
    """Try pmdarima.auto_arima when available; otherwise return defaults."""
    try:
        import pmdarima as pm

        model = pm.auto_arima(
            series,
            seasonal=True,
            m=7,
            max_p=max_p,
            max_q=max_q,
            max_P=1,
            max_Q=1,
            max_d=1,
            max_D=1,
            start_p=0,
            start_q=0,
            information_criterion="aic",
            suppress_warnings=True,
            error_action="ignore",
            stepwise=True,
        )
        order = tuple(int(x) for x in model.order)
        seasonal_order = tuple(int(x) for x in model.seasonal_order)
        return order, seasonal_order
    except ImportError:
        return SARIMA_ORDER, SARIMA_SEASONAL_ORDER
    except Exception:
        return SARIMA_ORDER, SARIMA_SEASONAL_ORDER


def search_sarima_orders_cached(
    drug_code: str,
    series: pd.Series,
    *,
    max_p: int = 2,
    max_q: int = 2,
) -> tuple[tuple[int, int, int], tuple[int, int, int, int]]:
    """
    Cached wrapper around :func:`search_sarima_orders`.

    Returns the previously-discovered best SARIMA order for *drug_code* when
    available, avoiding repeated expensive auto_arima calls during walk-forward
    validation folds.  The cache lives for the lifetime of the process (cleared
    between training runs via :func:`clear_sarima_order_cache`).
    """
    with _sarima_order_cache_lock:
        if drug_code in _sarima_order_cache:
            return _sarima_order_cache[drug_code]

    result = search_sarima_orders(series, max_p=max_p, max_q=max_q)

    with _sarima_order_cache_lock:
        _sarima_order_cache[drug_code] = result

    return result


def clear_sarima_order_cache() -> None:
    """Remove all cached SARIMA orders (call between independent training runs)."""
    with _sarima_order_cache_lock:
        _sarima_order_cache.clear()


def build_stacking_meta_features(
    n_samples: int,
    *,
    demand_segment: str,
    history_days: int | np.ndarray,
    recent_cv2: float | np.ndarray,
    classical_available: bool,
    horizon_steps: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Meta-features for conditional Ridge stacking."""
    segment_code = SEGMENT_ENCODING.get(demand_segment, 0.0) / 3.0

    if isinstance(history_days, np.ndarray):
        hist = np.asarray(history_days, dtype=float).reshape(-1)
        if hist.size != n_samples:
            raise ValueError("history_days length must match n_samples")
        history_norm = np.clip(hist / 365.0, 0.0, 1.0)
    else:
        history_norm = np.full(n_samples, min(float(history_days) / 365.0, 1.0))

    if isinstance(recent_cv2, np.ndarray):
        cv2_arr = np.asarray(recent_cv2, dtype=float).reshape(-1)
        if cv2_arr.size != n_samples:
            raise ValueError("recent_cv2 length must match n_samples")
        cv2_norm = np.clip(cv2_arr / max(CV2_THRESHOLD, 1e-6), 0.0, 3.0) / 3.0
    else:
        cv2_norm = np.full(
            n_samples,
            min(float(recent_cv2) / max(CV2_THRESHOLD, 1e-6), 3.0) / 3.0,
        )

    classical_flag = np.full(n_samples, 1.0 if classical_available else 0.0)
    if horizon_steps is None:
        step_norm = np.full(n_samples, 0.5)
    else:
        steps = np.asarray(horizon_steps, dtype=float).reshape(-1)
        if steps.size != n_samples:
            raise ValueError("horizon_steps length must match n_samples")
        step_norm = np.clip(steps / 30.0, 0.0, 1.0)

    return np.column_stack(
        [
            np.full(n_samples, segment_code),
            history_norm,
            cv2_norm,
            classical_flag,
            step_norm,
        ]
    )


def lgbm_training_target(series_df: pd.DataFrame) -> pd.Series:
    """Prefer EM-corrected demand for training when stockouts were corrected."""
    working = series_df
    if "em_corrected_quantity" in working.columns:
        return working["em_corrected_quantity"].astype(float)
    return working["total_quantity"].astype(float)
