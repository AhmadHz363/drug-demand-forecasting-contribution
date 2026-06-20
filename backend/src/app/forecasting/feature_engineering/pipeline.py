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
from app.services.demand_aggregation import aggregate_daily_demand_rows

logger = logging.getLogger(__name__)


def _load_demand_series(
    db_session: Session,
    drug_code: str,
    start_date: date,
    end_date: date,
    center_syn_id: Optional[str] = None,
) -> pd.DataFrame:
    rows = aggregate_daily_demand_rows(
        db_session,
        drug_code,
        start_date,
        end_date,
        center_syn_id=center_syn_id,
    )

    if not rows:
        full_index = pd.date_range(start=start_date, end=end_date, freq="D")
        return pd.DataFrame(
            {
                "demand_date": full_index.date,
                "total_quantity": 0.0,
                "demand_filled": 1,
            }
        )

    df = pd.DataFrame(rows, columns=["demand_date", "total_quantity"])
    df["demand_date"] = pd.to_datetime(df["demand_date"])
    df["total_quantity"] = df["total_quantity"].astype(float)
    original_dates = df["demand_date"].copy()

    full_index = pd.date_range(start=start_date, end=end_date, freq="D")
    df = (
        df.set_index("demand_date")
        .reindex(full_index, fill_value=0.0)
        .rename_axis("demand_date")
        .reset_index()
    )
    df["demand_filled"] = (~pd.to_datetime(df["demand_date"]).isin(original_dates)).astype(int)
    df["demand_date"] = df["demand_date"].dt.date
    return apply_consumption_demand(df)


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

    Columns guaranteed in output:
    - demand_date (index)
    - total_quantity (non-negative consumption demand; net negatives are abs'd)
    - all temporal features (14 columns)
    - all lag features (4 + 1 gap flag)
    - all rolling features (17 columns)
    - external features (2 columns)
    - supplier features (3 columns)
    Total: ~41 feature columns + target
    """
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")

    df = _load_demand_series(db_session, drug_code, start_date, end_date, center_syn_id)
    df = df.sort_values("demand_date").reset_index(drop=True)

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

    if df.isnull().any().any():
        df = df.fillna(0)

    df = df.set_index("demand_date")
    logger.debug(
        "Feature matrix for %s: shape=%s columns=%d",
        drug_code,
        df.shape,
        len(df.columns),
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
