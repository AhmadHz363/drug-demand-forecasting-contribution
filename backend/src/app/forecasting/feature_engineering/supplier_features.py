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
    """Append supplier lead-time features for the drug.

    When no ``supplier_lead_times`` row exists, values are NaN with
    ``supplier_features_is_default=1`` so models omit fake constants.
    """
    out = df.copy()

    try:
        row = (
            db_session.query(SupplierLeadTime)
            .filter(SupplierLeadTime.drug_code == drug_code)
            .first()
        )
    except Exception as exc:  # noqa: BLE001 — table may be missing
        logger.warning("supplier_lead_times unavailable (%s) — omitting supplier features", exc)
        try:
            db_session.rollback()
        except Exception:  # noqa: BLE001
            pass
        out["supplier_avg_lead_time"] = float("nan")
        out["supplier_lead_time_std"] = float("nan")
        out["supplier_reliability_score"] = float("nan")
        out["supplier_features_is_default"] = 1
        out["has_supplier"] = 0
        return out

    if row is None:
        logger.warning(
            "No supplier_lead_times row for drug %s — omitting default supplier features",
            drug_code,
        )
        out["supplier_avg_lead_time"] = float("nan")
        out["supplier_lead_time_std"] = float("nan")
        out["supplier_reliability_score"] = float("nan")
        out["supplier_features_is_default"] = 1
        out["has_supplier"] = 0
        return out

    out["supplier_avg_lead_time"] = float(row.avg_lead_time_days)
    out["supplier_lead_time_std"] = float(row.lead_time_std_days)
    out["supplier_reliability_score"] = float(row.reliability_score)
    out["supplier_features_is_default"] = 0
    out["has_supplier"] = 1
    return out


def supplier_feature_columns() -> list[str]:
    return [
        "supplier_avg_lead_time",
        "supplier_lead_time_std",
        "supplier_reliability_score",
    ]
