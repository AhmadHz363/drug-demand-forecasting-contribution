"""Pre-training data quality assessment and gating (Phase 3)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.forecasting.censored_demand.corrector import assert_no_degenerate_stockout_imputation
from app.forecasting.constants import (
    DATA_QUALITY_MAX_EXTERNAL_DEFAULT_RATE,
    DATA_QUALITY_MAX_SUPPLIER_DEFAULT_RATE,
    DATA_QUALITY_MIN_HISTORY_DAYS,
    DATA_QUALITY_REJECT_DEGENERATE_IMPUTATION,
)

QUALITY_OK = "ok"
QUALITY_FLAGGED = "flagged"
QUALITY_REJECTED = "rejected"


@dataclass
class DataQualityReport:
    status: str
    reasons: list[str] = field(default_factory=list)
    external_default_rate: float = 0.0
    supplier_default_rate: float = 0.0
    history_days: int = 0

    @property
    def should_skip_training(self) -> bool:
        return self.status == QUALITY_REJECTED

    @property
    def is_flagged(self) -> bool:
        return self.status == QUALITY_FLAGGED


def _default_rate(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns or len(df) == 0:
        return 0.0
    return float(df[column].astype(float).mean())


def assess_series_quality(df: pd.DataFrame, drug_code: str) -> DataQualityReport:
    """
    Evaluate whether a corrected demand series is safe to train on.

    Rejects degenerate imputation; flags high fallback-default coverage.
    """
    reasons: list[str] = []
    history_days = len(df)
    external_rate = _default_rate(df, "external_features_is_default")
    supplier_rate = _default_rate(df, "supplier_features_is_default")

    if history_days < DATA_QUALITY_MIN_HISTORY_DAYS:
        reasons.append(
            f"Only {history_days} history days (< {DATA_QUALITY_MIN_HISTORY_DAYS} minimum)"
        )

    if external_rate > DATA_QUALITY_MAX_EXTERNAL_DEFAULT_RATE:
        reasons.append(
            f"External features {external_rate:.0%} default-filled "
            f"(>{DATA_QUALITY_MAX_EXTERNAL_DEFAULT_RATE:.0%})"
        )

    if supplier_rate > DATA_QUALITY_MAX_SUPPLIER_DEFAULT_RATE:
        reasons.append(
            f"Supplier features {supplier_rate:.0%} default-filled "
            f"(>{DATA_QUALITY_MAX_SUPPLIER_DEFAULT_RATE:.0%})"
        )

    degenerate = False
    if DATA_QUALITY_REJECT_DEGENERATE_IMPUTATION:
        try:
            assert_no_degenerate_stockout_imputation(df)
        except ValueError as exc:
            degenerate = True
            reasons.append(str(exc))

    if degenerate:
        status = QUALITY_REJECTED
    elif reasons:
        status = QUALITY_FLAGGED
    else:
        status = QUALITY_OK

    if status == QUALITY_FLAGGED:
        reasons.insert(0, f"Training allowed with review flag for {drug_code}")

    return DataQualityReport(
        status=status,
        reasons=reasons,
        external_default_rate=external_rate,
        supplier_default_rate=supplier_rate,
        history_days=history_days,
    )
