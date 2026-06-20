"""Graduation rule — decide which Cold Start stage handles a drug."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.cold_start.constants import BLEND_UNTIL, COLD_START_ONLY_BELOW
from app.services.demand_aggregation import count_distinct_receipt_days


def get_observation_count(drug_code: str, db_session: Session) -> int:
    """Return distinct receipt days with demand history for this drug."""
    return count_distinct_receipt_days(db_session, drug_code)


def decide_stage(observation_count: int) -> str:
    """
    Return the active forecasting stage for a drug.

    - "cold_start_only"  when observation_count < COLD_START_ONLY_BELOW
    - "blended"          when COLD_START_ONLY_BELOW <= count < BLEND_UNTIL
    - "full_ensemble"    when observation_count >= BLEND_UNTIL
    """
    if observation_count < COLD_START_ONLY_BELOW:
        return "cold_start_only"
    if observation_count < BLEND_UNTIL:
        return "blended"
    return "full_ensemble"
