"""Weibull AFT survival correction for high stockout rates."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from app.forecasting.censored_demand.tobit import apply_tobit_correction
from app.forecasting.constants import MIN_STOCKOUT_PERIODS
from app.forecasting.prediction_bounds import demand_prediction_cap, winsorize_predictions

logger = logging.getLogger(__name__)


def apply_weibull_correction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fits a Weibull AFT model to censored demand and imputes stockout rows.

    Adds correction_method = "weibull" for affected rows.
    Falls back to Tobit when fewer than MIN_STOCKOUT_PERIODS stockouts exist.
    """
    try:
        from lifelines import WeibullAFTFitter
    except ImportError as exc:
        raise ImportError(
            "lifelines is required for Weibull censored-demand correction. "
            "Install it with: pip install 'lifelines>=0.27.0' (see requirements.txt)."
        ) from exc

    out = df.copy()
    stockout_mask = out["is_stockout"].astype(bool)
    stockout_count = int(stockout_mask.sum())

    if stockout_count == 0:
        return out

    if stockout_count < MIN_STOCKOUT_PERIODS:
        logger.warning(
            "Only %d stockout periods (< %d) — falling back to Tobit correction",
            stockout_count,
            MIN_STOCKOUT_PERIODS,
        )
        return apply_tobit_correction(out)

    if "correction_method" not in out.columns:
        out["correction_method"] = None

    survival_df = out.copy()
    survival_df["duration"] = survival_df["total_quantity"].astype(float)
    survival_df.loc[stockout_mask, "duration"] = survival_df.loc[
        stockout_mask, "rolling_mean_28d"
    ].astype(float)
    survival_df["observed"] = ~stockout_mask

    survival_df["duration"] = survival_df["duration"].clip(lower=1e-4)

    covariate_cols = [
        col
        for col in ("day_of_week", "month", "rolling_mean_28d")
        if col in survival_df.columns
    ]

    # Compute a demand cap from observed (non-stockout) history so imputed
    # values can never exceed a reasonable multiple of recent demand.
    observed_qty = out.loc[~stockout_mask, "total_quantity"].astype(float).values
    imputation_cap = demand_prediction_cap(observed_qty) if observed_qty.size else None

    try:
        fit_cols = ["duration", "observed", *covariate_cols]
        fit_df = survival_df[fit_cols].astype(float)

        # Standardize covariates to prevent overflow/divide-by-zero in autograd
        # when the Weibull AFT optimizer encounters large or heterogeneous scales.
        covariate_means: dict[str, float] = {}
        covariate_stds: dict[str, float] = {}
        fit_df_scaled = fit_df.copy()
        for col in covariate_cols:
            col_mean = float(fit_df[col].mean())
            col_std = float(fit_df[col].std(ddof=0))
            covariate_means[col] = col_mean
            covariate_stds[col] = col_std if col_std > 1e-8 else 1.0
            fit_df_scaled[col] = (fit_df[col] - covariate_means[col]) / covariate_stds[col]

        fitter = WeibullAFTFitter()
        fitter.fit(
            fit_df_scaled,
            duration_col="duration",
            event_col="observed",
        )
        stockout_rows_scaled = fit_df_scaled.loc[stockout_mask]
        medians = fitter.predict_median(stockout_rows_scaled).astype(float)

        # Guard against NaN/Inf produced by an unstable fit
        if not np.all(np.isfinite(medians.values)):
            raise ValueError("Weibull AFT produced non-finite median predictions")

        imputed = medians.clip(lower=0.0).values
        if imputation_cap is not None:
            imputed = winsorize_predictions(np.asarray(imputed, dtype=float), imputation_cap)

        out.loc[stockout_mask, "total_quantity"] = imputed
        out.loc[stockout_mask, "correction_method"] = "weibull"
    except Exception as exc:
        logger.warning("Weibull AFT fit failed (%s) — falling back to Tobit correction", exc)
        return apply_tobit_correction(out)

    return out
