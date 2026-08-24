"""Censored demand correction pipeline orchestrator."""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

from app.forecasting.demand_quantity import apply_consumption_demand
from app.forecasting.censored_demand.detector import detect_stockout_windows
from app.forecasting.censored_demand.em_sarima import em_corrected_series
from app.forecasting.censored_demand.survival import apply_weibull_correction
from app.forecasting.censored_demand.tobit import apply_tobit_correction
from app.forecasting.constants import STOCKOUT_RATE_THRESHOLD

logger = logging.getLogger(__name__)

DEGENERATE_IMPUTATION_MIN_ROWS = 3


def assert_no_degenerate_stockout_imputation(
    df: pd.DataFrame,
    *,
    min_rows: int = DEGENERATE_IMPUTATION_MIN_ROWS,
) -> None:
    """
    Raise when stockout-corrected demand is byte-identical across unrelated dates.

    Guards against intercept-only survival/Tobit fallbacks leaking a single float
    into the training series for many distinct stockout days.
    """
    if "is_stockout" not in df.columns:
        return

    stockout_mask = df["is_stockout"].astype(bool)
    if int(stockout_mask.sum()) < min_rows:
        return

    imputed = df.loc[stockout_mask, "total_quantity"].astype(float)
    if imputed.nunique() == 1:
        raise ValueError(
            "Degenerate stockout imputation: "
            f"{int(stockout_mask.sum())} corrected rows share identical value "
            f"{float(imputed.iloc[0]):.15g}"
        )


def _log_imputation_stats(
    working: pd.DataFrame,
    drug_code: str,
    stockout_rate: float,
    correction_method: str,
) -> None:
    stockout_mask = working["is_stockout"].astype(bool)
    if not stockout_mask.any():
        return

    imputed = working.loc[stockout_mask, "total_quantity"].astype(float)
    logger.info(
        "Demand imputation stats for %s: rate=%.4f method=%s imputed_min=%.4f "
        "imputed_max=%.4f imputed_std=%.4f rows=%d",
        drug_code,
        stockout_rate,
        correction_method,
        float(imputed.min()),
        float(imputed.max()),
        float(imputed.std(ddof=0)),
        int(stockout_mask.sum()),
    )
    if imputed.nunique() == 1 and int(stockout_mask.sum()) >= DEGENERATE_IMPUTATION_MIN_ROWS:
        logger.warning(
            "Degenerate imputation for %s: %d stockout rows share identical value %.15g",
            drug_code,
            int(stockout_mask.sum()),
            float(imputed.iloc[0]),
        )


def _as_working_frame(feature_df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(feature_df.index, pd.DatetimeIndex) or feature_df.index.name == "demand_date":
        out = feature_df.copy()
        if "demand_date" not in out.columns:
            out = out.reset_index()
            if "index" in out.columns and "demand_date" not in out.columns:
                out = out.rename(columns={"index": "demand_date"})
        return out
    return feature_df.copy()


def _restore_index(feature_df: pd.DataFrame, working: pd.DataFrame) -> pd.DataFrame:
    if isinstance(feature_df.index, pd.DatetimeIndex) or feature_df.index.name == "demand_date":
        result = working.set_index("demand_date")
        result.index.name = "demand_date"
        return result
    return working


def correct_demand(
    drug_code: str,
    center_syn_id: Optional[str],
    db_session: Session,
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Full censored demand correction pipeline (legacy — not used by SHIELD-XR):

    1. Detect stockout windows
    2. Choose correction method based on stockout_rate
    3. Apply correction
    4. Return the feature_df with total_quantity corrected

    Also adds em_corrected_quantity for SARIMA training.
    """
    del center_syn_id
    original_qty = feature_df["total_quantity"].astype(float).copy()
    working = apply_consumption_demand(_as_working_frame(feature_df))
    working["observed_quantity"] = working["total_quantity"].astype(float).copy()

    working = detect_stockout_windows(working, drug_code, db_session)
    stockout_rate = float(working.attrs.get("stockout_rate", working["is_stockout"].mean()))

    if stockout_rate == 0.0:
        working["em_corrected_quantity"] = working["total_quantity"].astype(float)
        result = _restore_index(feature_df, working)
        result["observed_quantity"] = working["observed_quantity"].astype(float).values
        logger.info("No stockouts detected for %s — skipping demand correction", drug_code)
        return result

    if stockout_rate < STOCKOUT_RATE_THRESHOLD:
        working = apply_tobit_correction(working)
        correction_method = "tobit"
    else:
        working = apply_weibull_correction(working)
        stockout_rows = working.loc[working["is_stockout"].astype(bool), "correction_method"]
        correction_method = str(stockout_rows.iloc[0]) if not stockout_rows.empty else "weibull"

    working["em_corrected_quantity"] = em_corrected_series(working)
    _log_imputation_stats(working, drug_code, stockout_rate, correction_method)

    result = _restore_index(feature_df, working)
    result["observed_quantity"] = working["observed_quantity"].astype(float).values
    result["total_quantity"] = working["total_quantity"].astype(float).values
    result["em_corrected_quantity"] = working["em_corrected_quantity"].astype(float).values
    result["is_stockout"] = working["is_stockout"].astype(bool).values
    if "correction_method" in working.columns:
        result["correction_method"] = working["correction_method"].fillna("").astype(str).values

    changed = int((result["total_quantity"].astype(float) != original_qty).sum())
    logger.info(
        "Demand correction for %s complete: rate=%.4f rows_adjusted=%d",
        drug_code,
        stockout_rate,
        changed,
    )
    return result
