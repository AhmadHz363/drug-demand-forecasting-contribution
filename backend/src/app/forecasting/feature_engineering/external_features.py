"""External features — hospital census (bed occupancy, surgery volume)."""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

from app.models.hospital_census import HospitalCensus

logger = logging.getLogger(__name__)

DEFAULT_BED_OCCUPANCY = 0.75
DEFAULT_WEEKLY_SURGERY_COUNT = 50


def add_external_features(
    df: pd.DataFrame,
    db_session: Session,
    center_syn_id: Optional[str],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Merge bed occupancy and surgery count onto the demand DataFrame.

    Requires columns: demand_date
    """
    out = df.copy()

    query = (
        db_session.query(
            HospitalCensus.census_date,
            HospitalCensus.bed_occupancy_rate,
            HospitalCensus.weekly_surgery_count,
            HospitalCensus.center_syn_id,
        )
        .filter(
            HospitalCensus.census_date >= start_date,
            HospitalCensus.census_date <= end_date,
        )
    )
    if center_syn_id is not None:
        query = query.filter(HospitalCensus.center_syn_id == center_syn_id)

    rows = query.all()

    if not rows:
        logger.warning("hospital_census table is empty — using default external feature values")
        out["bed_occupancy_rate"] = DEFAULT_BED_OCCUPANCY
        out["weekly_surgery_count"] = DEFAULT_WEEKLY_SURGERY_COUNT
        out["external_features_is_default"] = 1
        return out

    census_df = pd.DataFrame(
        rows,
        columns=["census_date", "bed_occupancy_rate", "weekly_surgery_count", "center_syn_id"],
    )
    census_df["census_date"] = pd.to_datetime(census_df["census_date"]).dt.date

    if center_syn_id is None:
        census_df = (
            census_df.groupby("census_date", as_index=False)
            .agg(
                bed_occupancy_rate=("bed_occupancy_rate", "mean"),
                weekly_surgery_count=("weekly_surgery_count", "mean"),
            )
            .round({"weekly_surgery_count": 0})
        )
        census_df["weekly_surgery_count"] = census_df["weekly_surgery_count"].astype(int)
    else:
        census_df = census_df.drop(columns=["center_syn_id"])

    out["_merge_date"] = pd.to_datetime(out["demand_date"]).dt.date
    merged = out.merge(
        census_df,
        left_on="_merge_date",
        right_on="census_date",
        how="left",
    )
    merged["external_features_is_default"] = merged["bed_occupancy_rate"].isna().astype(int)
    merged["bed_occupancy_rate"] = merged["bed_occupancy_rate"].fillna(DEFAULT_BED_OCCUPANCY)
    merged["weekly_surgery_count"] = (
        merged["weekly_surgery_count"].fillna(DEFAULT_WEEKLY_SURGERY_COUNT).astype(int)
    )
    merged = merged.drop(columns=["_merge_date", "census_date"], errors="ignore")
    return merged


def external_feature_columns() -> list[str]:
    return ["bed_occupancy_rate", "weekly_surgery_count"]
