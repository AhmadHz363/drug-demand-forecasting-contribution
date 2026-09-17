"""Supplier reliability features — legacy stub (table removed with SHIELD-XR)."""

from __future__ import annotations

import pandas as pd
from sqlalchemy.orm import Session


def add_supplier_features(
    df: pd.DataFrame,
    drug_code: str,
    db_session: Session,
) -> pd.DataFrame:
    """Legacy hook retained for the old feature pipeline; always omits supplier features."""
    del drug_code, db_session
    out = df.copy()
    out["supplier_avg_lead_time"] = float("nan")
    out["supplier_lead_time_std"] = float("nan")
    out["supplier_reliability_score"] = float("nan")
    out["supplier_features_is_default"] = 1
    out["has_supplier"] = 0
    return out


def supplier_feature_columns() -> list[str]:
    return [
        "supplier_avg_lead_time",
        "supplier_lead_time_std",
        "supplier_reliability_score",
    ]
