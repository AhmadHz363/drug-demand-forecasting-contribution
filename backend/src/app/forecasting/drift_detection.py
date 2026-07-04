"""Cross-run metric drift detection for model performance (Phase 3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.forecasting.constants import DRIFT_SMAPE_DEGRADATION_PCT, DRIFT_MASE_DEGRADATION_PCT


@dataclass
class DriftAssessment:
    smape_delta_pct: Optional[float] = None
    mase_delta_pct: Optional[float] = None
    smape_degraded: bool = False
    mase_degraded: bool = False

    @property
    def has_drift(self) -> bool:
        return self.smape_degraded or self.mase_degraded


def pct_change(previous: float, current: float) -> Optional[float]:
    """Percent increase from previous to current (None when previous is zero)."""
    if previous == 0.0:
        return None
    return float((current - previous) / abs(previous) * 100.0)


def assess_metric_drift(
    previous_smape: Optional[float],
    current_smape: float,
    previous_mase: Optional[float],
    current_mase: Optional[float],
) -> DriftAssessment:
    """Flag when walk-forward metrics worsen beyond configured thresholds."""
    assessment = DriftAssessment()

    if previous_smape is not None:
        assessment.smape_delta_pct = pct_change(previous_smape, current_smape)
        if assessment.smape_delta_pct is not None:
            assessment.smape_degraded = assessment.smape_delta_pct > DRIFT_SMAPE_DEGRADATION_PCT

    if previous_mase is not None and current_mase is not None:
        assessment.mase_delta_pct = pct_change(previous_mase, current_mase)
        if assessment.mase_delta_pct is not None:
            assessment.mase_degraded = assessment.mase_delta_pct > DRIFT_MASE_DEGRADATION_PCT

    return assessment
