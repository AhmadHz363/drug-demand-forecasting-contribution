"""EM algorithm producing a SARIMA-ready corrected demand series."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from app.forecasting.constants import SARIMA_MAX_ITER, SARIMA_ORDER, SARIMA_SEASONAL_ORDER

logger = logging.getLogger(__name__)


def _initialize_series(df: pd.DataFrame) -> pd.Series:
    series = df["total_quantity"].astype(float).copy()
    stockout_mask = df["is_stockout"].astype(bool)

    if not stockout_mask.any():
        return series

    masked = series.copy()
    masked.loc[stockout_mask] = np.nan
    rolling_fill = masked.rolling(window=14, min_periods=1).mean()
    fallback = float(masked.dropna().mean()) if masked.notna().any() else 0.0
    imputed = rolling_fill.fillna(fallback)
    series.loc[stockout_mask] = imputed.loc[stockout_mask]
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
    convergence_threshold: float = 0.01,
) -> pd.Series:
    """
    Runs EM iterations to produce a demand series suitable for SARIMA training.

    Returns a pd.Series of corrected daily demand values (same index as df).
    """
    stockout_mask = df["is_stockout"].astype(bool)
    current = _initialize_series(df)

    if not stockout_mask.any():
        return current

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
        if mean_abs_change < convergence_threshold:
            logger.info("EM-SARIMA converged after %d iterations", iteration)
            break
    else:
        logger.info("EM-SARIMA stopped after %d iterations (threshold not met)", iterations_run)

    return current
