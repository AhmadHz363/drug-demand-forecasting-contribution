"""Feature engineering pipeline — orchestrates all feature families."""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.forecasting.demand_quantity import apply_consumption_demand
from app.forecasting.feature_engineering.external_features import add_external_features
from app.forecasting.feature_engineering.lag_features import add_lag_features
from app.forecasting.feature_engineering.rolling_features import add_rolling_features
from app.forecasting.feature_engineering.supplier_features import add_supplier_features
from app.forecasting.feature_engineering.temporal_features import add_temporal_features
from app.services.demand_aggregation import (
    aggregate_daily_demand_rows,
    get_import_coverage_periods,
)

logger = logging.getLogger(__name__)


def _coverage_flag_series(
    dates: pd.Series,
    periods: list[tuple[date, date]],
) -> pd.Series:
    if not periods:
        return pd.Series(0, index=dates.index, dtype=int)

    def _covered(day: date) -> bool:
        return any(start <= day <= end for start, end in periods)

    return dates.map(lambda d: 0 if _covered(d) else 1).astype(int)


def _load_demand_series(
    db_session: Session,
    drug_code: str,
    start_date: date,
    end_date: date,
    center_syn_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a daily demand frame with explicit coverage gaps.

    - Days inside a covered import period with no demand → true zero
    - Days outside all import coverage periods → is_coverage_gap=1 (not treated as demand)
    """
    periods = get_import_coverage_periods(db_session)
    rows = aggregate_daily_demand_rows(
        db_session,
        drug_code,
        start_date,
        end_date,
        center_syn_id=center_syn_id,
    )

    full_index = pd.date_range(start=start_date, end=end_date, freq="D")
    if not rows:
        df = pd.DataFrame(
            {
                "demand_date": full_index.date,
                "total_quantity": 0.0,
                "demand_filled": 1,
            }
        )
    else:
        df = pd.DataFrame(rows, columns=["demand_date", "total_quantity"])
        df["demand_date"] = pd.to_datetime(df["demand_date"])
        df["total_quantity"] = df["total_quantity"].astype(float)
        original_dates = set(pd.to_datetime(df["demand_date"]).dt.normalize())
        df = (
            df.set_index("demand_date")
            .reindex(full_index)
            .rename_axis("demand_date")
            .reset_index()
        )
        # Only fill zeros inside covered periods; leave gap days as NaN until flagged.
        date_vals = pd.to_datetime(df["demand_date"]).dt.normalize()
        covered_mask = date_vals.map(
            lambda ts: any(start <= ts.date() <= end for start, end in periods)
        )
        if not periods:
            covered_mask = pd.Series(True, index=df.index)
        df.loc[covered_mask & df["total_quantity"].isna(), "total_quantity"] = 0.0
        df["demand_filled"] = (
            ~date_vals.isin(original_dates) & covered_mask
        ).astype(int)

    df["demand_date"] = pd.to_datetime(df["demand_date"]).dt.date
    df["is_coverage_gap"] = _coverage_flag_series(pd.Series(df["demand_date"]), periods)
    # Uncovered years are excluded from demand modelling, not zero-filled.
    gap_mask = df["is_coverage_gap"].astype(int) == 1
    df.loc[gap_mask, "total_quantity"] = np.nan
    df.loc[gap_mask, "demand_filled"] = 0
    return apply_consumption_demand(df)


def covered_demand_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of rows that are inside imported coverage and usable for metrics."""
    if "is_coverage_gap" in df.columns:
        return df["is_coverage_gap"].astype(int) == 0
    return pd.Series(True, index=df.index)


def filter_covered_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop coverage-gap rows for segmentation, stockout correction, and history counts."""
    mask = covered_demand_mask(df)
    if mask.all():
        return df
    return df.loc[mask].copy()


def build_feature_matrix(
    drug_code: str,
    center_syn_id: Optional[str],
    db_session: Session,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Builds a feature matrix from receipt-derived daily demand.

    Demand history is aggregated directly from ``drug_receipts`` (source of truth).
    Center-specific forecasts aggregate receipts for that center only at query time.

    Coverage-gap days are retained with ``is_coverage_gap=1`` so lag/rolling
    features can reset at segment boundaries; callers that need contiguous
    history should use ``filter_covered_rows``.
    """
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")

    df = _load_demand_series(db_session, drug_code, start_date, end_date, center_syn_id)
    df = df.sort_values("demand_date").reset_index(drop=True)

    # Keep observed zeros for covered days; gap rows stay NaN until lag/rolling reset.
    if "total_quantity" in df.columns:
        covered = df["is_coverage_gap"].astype(int) == 0
        df.loc[covered, "total_quantity"] = df.loc[covered, "total_quantity"].fillna(0.0)

    df = add_temporal_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_external_features(df, db_session, center_syn_id, start_date, end_date)
    df = add_supplier_features(df, drug_code, db_session)

    if "demand_filled" not in df.columns:
        df["demand_filled"] = 0
    df["forecast_step"] = 0
    df["forecast_step_sin"] = 0.0
    df["forecast_step_cos"] = 1.0

    # Do not zero-fill coverage-gap demand or intentionally-missing covariates.
    preserve_nan_cols = {
        "demand_date",
        "is_coverage_gap",
        "total_quantity",
        "observed_quantity",
        "em_corrected_quantity",
        "bed_occupancy_rate",
        "weekly_surgery_count",
        "avg_lead_time_days",
        "lead_time_std_days",
        "reliability_score",
        "supplier_avg_lead_time",
        "supplier_lead_time_std",
        "supplier_reliability_score",
    }
    for col in df.columns:
        if col in preserve_nan_cols:
            continue
        if df[col].dtype.kind in "fc":
            df[col] = df[col].fillna(0)

    df = df.set_index("demand_date")
    logger.debug(
        "Feature matrix for %s: shape=%s columns=%d gaps=%s",
        drug_code,
        df.shape,
        len(df.columns),
        int(df["is_coverage_gap"].sum()) if "is_coverage_gap" in df.columns else 0,
    )
    return df


def build_future_covariates(
    drug_code: str,
    center_syn_id: Optional[str],
    db_session: Session,
    last_date: date,
    horizon_days: int,
) -> pd.DataFrame:
    """
    Known future covariates (calendar, census, supplier) for recursive LGBM forecasts.
    """
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")

    start = pd.Timestamp(last_date) + pd.Timedelta(days=1)
    end = pd.Timestamp(last_date) + pd.Timedelta(days=horizon_days)
    future_dates = pd.date_range(start=start, end=end, freq="D")
    df = pd.DataFrame({"demand_date": future_dates.date})
    df["is_coverage_gap"] = 0
    df = add_temporal_features(df)
    df = add_external_features(
        df,
        db_session,
        center_syn_id,
        start.date(),
        end.date(),
    )
    df = add_supplier_features(df, drug_code, db_session)
    df["forecast_step"] = np.arange(1, horizon_days + 1)
    df["forecast_step_sin"] = np.sin(2 * np.pi * df["forecast_step"] / 30)
    df["forecast_step_cos"] = np.cos(2 * np.pi * df["forecast_step"] / 30)
    df["demand_filled"] = 1
    return df.set_index("demand_date")
