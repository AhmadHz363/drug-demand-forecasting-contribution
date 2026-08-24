"""Build hospital-wide SKU × day panel from enriched training data."""

from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.models.hospital_daily_demand_enriched import HospitalDailyDemandEnriched
from app.services.enriched_demand import get_enriched_date_bounds


def build_hospital_panel(
    db_session: Session,
    *,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    drug_codes: Optional[list[str]] = None,
    center_syn_id: Optional[str] = None,
) -> pd.DataFrame:
    """Return a long-format panel compatible with the SHIELD-XR notebook pipeline."""
    del center_syn_id  # enriched panel is hospital-wide

    min_date, max_date = get_enriched_date_bounds(db_session)
    if min_date is None or max_date is None:
        return pd.DataFrame()

    start = start_date or min_date
    end = end_date or max_date

    query = db_session.query(HospitalDailyDemandEnriched).filter(
        HospitalDailyDemandEnriched.demand_date >= start,
        HospitalDailyDemandEnriched.demand_date <= end,
    )
    if drug_codes:
        query = query.filter(HospitalDailyDemandEnriched.drug_code.in_(drug_codes))

    rows = query.order_by(
        HospitalDailyDemandEnriched.demand_date.asc(),
        HospitalDailyDemandEnriched.drug_code.asc(),
        HospitalDailyDemandEnriched.imported_at.asc(),
    ).all()
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(
        [
            {
                "source_file": row.source_file,
                "imported_at": row.imported_at,
                "CODE": str(row.drug_code),
                "DATE": pd.Timestamp(row.demand_date),
                "demand": float(row.demand),
                "ARTICLE": row.article,
                "CAT": row.cat,
                "n_unique_patients": float(row.n_unique_patients),
                "n_admissions": float(row.n_admissions),
                "n_unique_doctors": float(row.n_unique_doctors),
                "n_unique_CR": float(row.n_unique_cr),
                "n_unique_CS": float(row.n_unique_cs),
                "n_transactions": float(row.n_transactions),
                "n_demand_txns": float(row.n_demand_txns),
                "top1_CR_share": float(row.top1_cr_share or 0.0),
                "top2_CR_share": float(row.top2_cr_share or 0.0),
                "top3_CR_share": float(row.top3_cr_share or 0.0),
                "bed_occupancy_rate": 0.0,
                "weekly_surgery_count": 0.0,
            }
            for row in rows
        ]
    )
    frame = frame.sort_values(["CODE", "DATE", "imported_at"]).drop_duplicates(
        ["CODE", "DATE"],
        keep="last",
    )
    for col in frame.columns:
        if col.startswith("n_") or col.endswith("_share") or col in {"demand"}:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    frame["CAT"] = frame["CAT"].astype(str).replace({"nan": "UNK", "None": "UNK"}).fillna("UNK")
    return frame.sort_values(["CODE", "DATE"]).reset_index(drop=True)


def split_temporal(
    df: pd.DataFrame,
    *,
    train_frac: float = 0.70,
    valid_frac: float = 0.15,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    dates_all = np.sort(df["DATE"].unique())
    n_days = len(dates_all)
    d_train_end = pd.Timestamp(dates_all[int(n_days * train_frac)])
    d_valid_end = pd.Timestamp(dates_all[int(n_days * (train_frac + valid_frac))])
    return d_train_end, d_valid_end
