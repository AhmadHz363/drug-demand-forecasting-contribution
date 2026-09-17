"""Supplier reliability features — per-drug lead time constants."""

from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy.orm import Session

from app.models.supplier_lead_time import SupplierLeadTime

logger = logging.getLogger(__name__)

DEFAULT_AVG_LEAD_TIME = 3.0
DEFAULT_LEAD_TIME_STD = 1.0
DEFAULT_RELIABILITY = 0.85


def add_supplier_features(
    df: pd.DataFrame,
    drug_code: str,
    db_session: Session,
) -> pd.DataFrame:
    """Append constant supplier lead-time features for the drug."""
    out = df.copy()

    row = (
        db_session.query(SupplierLeadTime)
        .filter(SupplierLeadTime.drug_code == drug_code)
        .first()
    )

    if row is None:
        logger.warning(
            "No supplier_lead_times row for drug %s — using default supplier feature values",
            drug_code,
        )
        avg_lead = DEFAULT_AVG_LEAD_TIME
        std_lead = DEFAULT_LEAD_TIME_STD
        reliability = DEFAULT_RELIABILITY
    else:
        avg_lead = float(row.avg_lead_time_days)
        std_lead = float(row.lead_time_std_days)
        reliability = float(row.reliability_score)

    out["supplier_avg_lead_time"] = avg_lead
    out["supplier_lead_time_std"] = std_lead
    out["supplier_reliability_score"] = reliability
    out["supplier_features_is_default"] = 0 if row is not None else 1
    return out


def supplier_feature_columns() -> list[str]:
    return [
        "supplier_avg_lead_time",
        "supplier_lead_time_std",
        "supplier_reliability_score",
    ]
