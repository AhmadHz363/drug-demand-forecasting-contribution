"""EM algorithm producing a SARIMA-ready corrected demand series."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from app.forecasting.constants import SARIMA_MAX_ITER, SARIMA_ORDER, SARIMA_SEASONAL_ORDER

logger = logging.getLogger(__name__)

# Relative convergence: stop when mean absolute change drops below this fraction
# of the non-zero demand mean. Avoids artificially tight absolute thresholds for
# high-volume drugs and prevents infinite loops for noisy lumpy demand series.
_EM_RELATIVE_CONVERGENCE_THRESHOLD = 0.005


def _initialize_series(df: pd.DataFrame) -> pd.Series:
    """
    Initialise missing stockout values via linear interpolation then rolling mean.

    Linear interpolation produces smoother initial estimates than a rolling
    mean alone, which cuts iteration count and reduces the risk of SARIMA
    over-fitting to step-function artefacts.
    """
    series = df["total_quantity"].astype(float).copy()
    stockout_mask = df["is_stockout"].astype(bool)

    if not stockout_mask.any():
        return series

    masked = series.copy()
    masked.loc[stockout_mask] = np.nan

    # Linear interpolation handles bounded gaps well; limit=28 prevents
    # interpolating across very long zero-demand windows.
    interpolated = masked.interpolate(method="linear", limit=28, limit_direction="both")
    # Rolling mean fills any residual NaNs (edges or gaps > 28 days).
    rolling_fill = interpolated.fillna(interpolated.rolling(window=14, min_periods=1).mean())
    fallback = float(masked.dropna().mean()) if masked.notna().any() else 0.0
    imputed = rolling_fill.fillna(fallback)

    series.loc[stockout_mask] = imputed.loc[stockout_mask].clip(lower=0.0)
    return series


def _sarima_in_sample_predictions(series: pd.Series) -> pd.Series:
    model = SARIMAX(
        series,
        order=SARIMA_ORDER,
        seasonal_order=SARIMA_SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False, maxiter=SARIMA_MAX_ITER)
    return result.fittedvalues.reindex(series.index)


def em_corrected_series(
    df: pd.DataFrame,
    max_iterations: int = 20,
    convergence_threshold: float | None = None,
) -> pd.Series:
    """
    Runs EM iterations to produce a demand series suitable for SARIMA training.

    Convergence is judged relative to the mean non-zero demand so that the
    threshold self-scales across low- and high-volume drugs.

    Returns a pd.Series of corrected daily demand values (same index as df).
    """
    stockout_mask = df["is_stockout"].astype(bool)
    current = _initialize_series(df)

    if not stockout_mask.any():
        return current

    # Scale the convergence threshold to the mean non-zero demand level.
    nonzero_vals = current.loc[~stockout_mask]
    nonzero_vals = nonzero_vals[nonzero_vals > 0]
    demand_mean = float(nonzero_vals.mean()) if not nonzero_vals.empty else 1.0
    rel_threshold = convergence_threshold if convergence_threshold is not None else (
        _EM_RELATIVE_CONVERGENCE_THRESHOLD * max(demand_mean, 1.0)
    )

    iterations_run = 0
    for iteration in range(1, max_iterations + 1):
        iterations_run = iteration
        try:
            fitted = _sarima_in_sample_predictions(current)
        except Exception as exc:
            logger.warning("EM-SARIMA fit failed at iteration %d: %s", iteration, exc)
            break

        previous = current.loc[stockout_mask].copy()
        current.loc[stockout_mask] = fitted.loc[stockout_mask].clip(lower=0.0)

        mean_abs_change = float((current.loc[stockout_mask] - previous).abs().mean())
        if mean_abs_change < rel_threshold:
            logger.info(
                "EM-SARIMA converged after %d iterations (delta=%.4f, threshold=%.4f)",
                iteration,
                mean_abs_change,
                rel_threshold,
            )
            break
    else:
        logger.info(
            "EM-SARIMA stopped after %d iterations (delta not below threshold=%.4f)",
            iterations_run,
            rel_threshold,
        )

    return current
