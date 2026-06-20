"""KNN-weighted baseline demand forecast with uncertainty bounds."""

from __future__ import annotations

import logging
import math
import statistics
from datetime import date, timedelta

from sqlalchemy.orm import Session

from app.cold_start.constants import KNN_K, PHARMACIST_ESTIMATE_WEIGHT
from app.cold_start.schemas import DailyForecast, PharmacistEstimate
from app.services.demand_aggregation import load_recent_quantities_from_receipts

logger = logging.getLogger(__name__)

LOOKBACK_DAYS = 90
MIN_DAYS_REQUIRED = 7
PREDICTION_INTERVAL_Z = 1.28


def _fetch_recent_quantities(db_session: Session, drug_code: str) -> list[float]:
    """Return total_quantity values for the last LOOKBACK_DAYS calendar days."""
    return load_recent_quantities_from_receipts(
        db_session,
        drug_code,
        lookback_days=LOOKBACK_DAYS,
    )


def compute_baseline_forecast(
    neighbours: list[tuple[str, float]],
    db_session: Session,
    horizon_days: int,
    pharmacist_estimate: PharmacistEstimate | None,
) -> tuple[list[DailyForecast], list[tuple[str, float]]]:
    """
    Blend neighbour demand histories into a flat baseline forecast with P10/P50/P90
    intervals for each day in the horizon.

    Walks ``neighbours`` in descending similarity order and uses up to KNN_K drugs
    that have at least MIN_DAYS_REQUIRED distinct receipt days in drug_receipts.
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be at least 1")

    neighbour_stats: list[tuple[float, float, float]] = []
    used_neighbours: list[tuple[str, float]] = []

    for drug_code, similarity in neighbours:
        if len(used_neighbours) >= KNN_K:
            break

        quantities = _fetch_recent_quantities(db_session, drug_code)
        if len(quantities) < MIN_DAYS_REQUIRED:
            logger.warning(
                "Skipping neighbour %s: only %d days of demand data (need %d)",
                drug_code,
                len(quantities),
                MIN_DAYS_REQUIRED,
            )
            continue

        daily_mean = statistics.mean(quantities)
        daily_std = statistics.stdev(quantities) if len(quantities) > 1 else 0.0
        neighbour_stats.append((similarity, daily_mean, daily_std))
        used_neighbours.append((drug_code, similarity))

    if not neighbour_stats:
        if pharmacist_estimate is not None:
            pharmacist_daily = pharmacist_estimate.weekly_units / 7.0
            spread = pharmacist_daily * 0.25 * (1.0 - pharmacist_estimate.confidence)
            start_date = date.today() + timedelta(days=1)
            forecast = [
                DailyForecast(
                    date=start_date + timedelta(days=offset),
                    p10=max(0.0, pharmacist_daily - spread),
                    p50=pharmacist_daily,
                    p90=pharmacist_daily + spread,
                )
                for offset in range(horizon_days)
            ]
            logger.warning(
                "No neighbours with %d+ days of demand; using pharmacist_estimate only",
                MIN_DAYS_REQUIRED,
            )
            return forecast, []

        raise ValueError(
            f"No neighbours with sufficient demand history (need at least "
            f"{MIN_DAYS_REQUIRED} days each in drug_receipts). "
            "Load receipt history for similar drugs, or add a pharmacist_estimate."
        )

    total_similarity = sum(sim for sim, _, _ in neighbour_stats)
    if total_similarity <= 0:
        raise ValueError(
            "Sum of neighbour similarity scores is zero; cannot compute weighted blend."
        )

    knn_estimate = sum(
        (sim / total_similarity) * daily_mean
        for sim, daily_mean, _ in neighbour_stats
    )

    if pharmacist_estimate is not None:
        pharmacist_daily = pharmacist_estimate.weekly_units / 7.0
        final_estimate = (
            (1.0 - PHARMACIST_ESTIMATE_WEIGHT) * knn_estimate
            + PHARMACIST_ESTIMATE_WEIGHT * pharmacist_daily
        )
    else:
        final_estimate = knn_estimate

    pooled_variance = sum(
        (sim / total_similarity) * (daily_std**2)
        for sim, _, daily_std in neighbour_stats
    )
    pooled_std = math.sqrt(pooled_variance)

    p10 = max(0.0, final_estimate - PREDICTION_INTERVAL_Z * pooled_std)
    p50 = final_estimate
    p90 = final_estimate + PREDICTION_INTERVAL_Z * pooled_std

    start_date = date.today() + timedelta(days=1)
    forecast = [
        DailyForecast(
            date=start_date + timedelta(days=offset),
            p10=p10,
            p50=p50,
            p90=p90,
        )
        for offset in range(horizon_days)
    ]
    return forecast, used_neighbours
