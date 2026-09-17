"""Censored demand correction pipeline orchestrator."""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

from app.forecasting.demand_quantity import apply_consumption_demand
from app.forecasting.censored_demand.detector import detect_stockout_windows
from app.forecasting.censored_demand.em_sarima import em_corrected_series
from app.forecasting.censored_demand.survival import apply_weibull_correction
from app.forecasting.censored_demand.tobit import apply_tobit_correction
from app.forecasting.constants import STOCKOUT_RATE_THRESHOLD
from app.models.stockout_flag import StockoutFlag

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


def _flag_date(value: object) -> date:
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


def _write_stockout_flags(
    db_session: Session,
    drug_code: str,
    center_syn_id: Optional[str],
    working: pd.DataFrame,
) -> None:
    stockout_rows = working.loc[working["is_stockout"].astype(bool)]
    for _, row in stockout_rows.iterrows():
        db_session.add(
            StockoutFlag(
                drug_code=drug_code,
                center_syn_id=center_syn_id,
                flag_date=_flag_date(row["demand_date"]),
                observed_quantity=0.0,
                estimated_true_demand=float(row["total_quantity"]),
                correction_method=str(row.get("correction_method") or "none"),
            )
        )
    db_session.flush()


def correct_demand(
    drug_code: str,
    center_syn_id: Optional[str],
    db_session: Session,
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Full censored demand correction pipeline:

    1. Detect stockout windows
    2. Choose correction method based on stockout_rate
    3. Apply correction
    4. Write stockout_flag rows to the stockout_flags table
    5. Return the feature_df with total_quantity corrected

    Also adds em_corrected_quantity for SARIMA training.
    """
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
    _write_stockout_flags(db_session, drug_code, center_syn_id, working)

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
