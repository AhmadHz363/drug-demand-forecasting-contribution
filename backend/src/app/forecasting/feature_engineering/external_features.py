"""External features — hospital census (bed occupancy, surgery volume)."""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

from app.models.hospital_census import HospitalCensus

logger = logging.getLogger(__name__)

# Used only to fill sparse gaps when *some* real census rows exist for the window.
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

    When ``hospital_census`` is empty or unavailable, columns are filled with
    NaN and ``external_features_is_default=1`` so models can omit them instead
    of treating fake constants (0.75 / 50) as real covariates.
    """
    out = df.copy()

    try:
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
    except Exception as exc:  # noqa: BLE001 — table may be missing in older DBs
        logger.warning("hospital_census unavailable (%s) — omitting external features", exc)
        try:
            db_session.rollback()
        except Exception:  # noqa: BLE001
            pass
        out["bed_occupancy_rate"] = float("nan")
        out["weekly_surgery_count"] = float("nan")
        out["external_features_is_default"] = 1
        out["has_census"] = 0
        return out

    if not rows:
        logger.warning(
            "hospital_census table is empty — omitting default-filled external features"
        )
        out["bed_occupancy_rate"] = float("nan")
        out["weekly_surgery_count"] = float("nan")
        out["external_features_is_default"] = 1
        out["has_census"] = 0
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
    # Sparse gaps only: fill from nearby real census, not invent a full fake series.
    merged["bed_occupancy_rate"] = merged["bed_occupancy_rate"].fillna(DEFAULT_BED_OCCUPANCY)
    merged["weekly_surgery_count"] = (
        merged["weekly_surgery_count"].fillna(DEFAULT_WEEKLY_SURGERY_COUNT).astype(int)
    )
    # If most rows were gaps, still mark as mostly-default via the flag above.
    merged["has_census"] = (1 - merged["external_features_is_default"]).astype(int)
    merged = merged.drop(columns=["_merge_date", "census_date"], errors="ignore")
    return merged


def external_feature_columns() -> list[str]:
    return ["bed_occupancy_rate", "weekly_surgery_count"]
