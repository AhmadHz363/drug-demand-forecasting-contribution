"""Weibull AFT survival correction for high stockout rates."""

from __future__ import annotations

import logging

import pandas as pd

from app.forecasting.censored_demand.tobit import apply_tobit_correction
from app.forecasting.constants import MIN_STOCKOUT_PERIODS

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

    try:
        fit_df = survival_df[["duration", "observed"]].astype(float)
        fitter = WeibullAFTFitter()
        fitter.fit(
            fit_df,
            duration_col="duration",
            event_col="observed",
        )
        stockout_rows = fit_df.loc[stockout_mask]
        medians = fitter.predict_median(stockout_rows).astype(float)
        out.loc[stockout_mask, "total_quantity"] = medians.clip(lower=0.0).values
        out.loc[stockout_mask, "correction_method"] = "weibull"
    except Exception as exc:
        logger.warning("Weibull AFT fit failed (%s) — falling back to Tobit correction", exc)
        return apply_tobit_correction(out)

    return out
