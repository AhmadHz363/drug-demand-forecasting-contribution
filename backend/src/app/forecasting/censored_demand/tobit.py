"""Tobit maximum-likelihood correction for low stockout rates."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def _truncated_normal_mean(mu: float, sigma: float) -> float:
    sigma = max(sigma, 1e-8)
    ratio = mu / sigma
    pdf = stats.norm.pdf(ratio)
    cdf = stats.norm.cdf(ratio)
    if cdf <= 1e-12:
        return max(mu, 0.0)
    return mu + sigma * pdf / cdf


def _neg_log_likelihood(params: np.ndarray, y: np.ndarray, censored: np.ndarray) -> float:
    mu, log_sigma = params
    sigma = float(np.exp(log_sigma)) + 1e-8

    ll = 0.0
    uncensored = ~censored
    if uncensored.any():
        ll += float(np.sum(stats.norm.logpdf(y[uncensored], loc=mu, scale=sigma)))

    if censored.any():
        z = (0.0 - mu) / sigma
        ll += float(np.sum(np.log(stats.norm.cdf(z) + 1e-12)))

    return -ll


def _fallback_imputation(df: pd.DataFrame) -> float:
    non_stockout = df.loc[~df["is_stockout"], "total_quantity"].astype(float)
    if non_stockout.empty:
        return 0.0
    rolling_mean = non_stockout.rolling(window=14, min_periods=1).mean()
    return float(rolling_mean.mean())


def apply_tobit_correction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces total_quantity = 0 (stockout rows only) with Tobit imputed values.

    Non-stockout rows are unchanged. Adds correction_method = "tobit" for
    affected rows.
    """
    out = df.copy()
    if "correction_method" not in out.columns:
        out["correction_method"] = None

    stockout_mask = out["is_stockout"].astype(bool)
    if not stockout_mask.any():
        return out

    y = out["total_quantity"].astype(float).to_numpy()
    censored = stockout_mask.to_numpy()

    y_positive = y[~censored]
    if len(y_positive) == 0:
        imputed = _fallback_imputation(out)
        logger.warning("Tobit MLE skipped — no uncensored observations; using rolling mean fallback")
    else:
        mu0 = float(np.mean(y_positive))
        sigma0 = float(np.std(y_positive)) or 1.0
        x0 = np.array([mu0, np.log(sigma0)])

        result = minimize(
            _neg_log_likelihood,
            x0,
            args=(y, censored),
            method="L-BFGS-B",
        )

        if not result.success:
            imputed = _fallback_imputation(out)
            logger.warning(
                "Tobit optimizer did not converge (%s) — using rolling mean fallback",
                result.message,
            )
            out.loc[stockout_mask, "total_quantity"] = imputed
        else:
            mu_hat, log_sigma_hat = result.x
            sigma_hat = float(np.exp(log_sigma_hat))
            if "rolling_mean_28d" in out.columns:
                local_mu = out.loc[stockout_mask, "rolling_mean_28d"].astype(float).clip(lower=0.0)
                imputed = local_mu.apply(
                    lambda mu: max(_truncated_normal_mean(float(mu), sigma_hat), 0.0),
                )
                out.loc[stockout_mask, "total_quantity"] = imputed.values
            else:
                imputed = max(_truncated_normal_mean(float(mu_hat), sigma_hat), 0.0)
                out.loc[stockout_mask, "total_quantity"] = imputed

    out.loc[stockout_mask, "correction_method"] = "tobit"
    return out
