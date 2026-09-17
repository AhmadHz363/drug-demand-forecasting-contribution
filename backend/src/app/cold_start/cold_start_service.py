"""Orchestrator — CAMEO cold-start forecasting on real drug attributes + demand."""

from __future__ import annotations

import logging
import math
from datetime import date, timedelta

import numpy as np
from sqlalchemy.orm import Session

from app.cold_start.cameo.artifacts import load_artifacts
from app.cold_start.cameo.metric_net import cameo_init_forecast
from app.cold_start.cameo.panel import load_weekly_series, load_weekly_series_for_drug
from app.cold_start.cameo.pipeline import forecast_cold_start_weeks, run_cameo
from app.cold_start.constants import (
    CAMEO_TOPK,
    COLD_START_ONLY_BELOW,
    PHARMACIST_ESTIMATE_WEIGHT,
)
from app.cold_start.drugs_adapter import get_drug_metadata, load_library_drug_metadata
from app.cold_start.graduation import decide_stage, get_observation_count
from app.cold_start.schemas import ColdStartPredictRequest, ColdStartPredictResponse, DailyForecast

logger = logging.getLogger(__name__)


class DrugsTableEmptyError(ValueError):
    """Raised when the drugs table has no matched_source rows for the library."""


DrugCatalogEmptyError = DrugsTableEmptyError
DrugReceiptsEmptyError = DrugsTableEmptyError


def invalidate_library_embedding_cache() -> None:
    from app.cold_start.cameo.artifacts import clear_artifact_cache

    clear_artifact_cache()


def _weekly_to_daily_forecast(
    weekly_values: np.ndarray,
    horizon_days: int,
    *,
    half_width_weekly: float,
    start_date: date | None = None,
) -> list[DailyForecast]:
    if start_date is None:
        start_date = date.today() + timedelta(days=1)

    daily_half = half_width_weekly / 7.0
    forecasts: list[DailyForecast] = []
    for day_offset in range(horizon_days):
        week_idx = min(day_offset // 7, len(weekly_values) - 1)
        daily_level = float(weekly_values[week_idx]) / 7.0
        forecasts.append(
            DailyForecast(
                date=start_date + timedelta(days=day_offset),
                p10=max(0.0, daily_level - daily_half),
                p50=daily_level,
                p90=daily_level + daily_half,
            )
        )
    return forecasts


def _apply_pharmacist_blend(
    weekly_values: np.ndarray,
    pharmacist_weekly: float,
    confidence: float,
) -> np.ndarray:
    weight = PHARMACIST_ESTIMATE_WEIGHT * confidence
    return (1.0 - weight) * weekly_values + weight * pharmacist_weekly


def _build_uncertainty_note(stage: str, observation_count: int, neighbour_count: int) -> str:
    if stage == "blended":
        return (
            f"CAMEO online tracker active with {observation_count} day(s) of real demand. "
            "Confidence improves as more weekly observations accumulate."
        )
    if neighbour_count < CAMEO_TOPK:
        return (
            f"CAMEO cold-start forecast from {neighbour_count} analog drug(s); "
            "limited library coverage may widen uncertainty."
        )
    return (
        "CAMEO cold-start forecast from metric-learning analogs in the matched-source library. "
        f"No real demand history yet ({observation_count} observation day(s))."
    )


class ColdStartService:
    def __init__(self, db_session: Session) -> None:
        self.db = db_session

    def predict(self, request: ColdStartPredictRequest) -> ColdStartPredictResponse:
        drug = request.drug_metadata
        horizon_days = request.forecast_horizon_days
        horizon_weeks = max(1, math.ceil(horizon_days / 7))

        artifacts = load_artifacts()
        library_metadata = {
            meta.drug_code: meta
            for meta in load_library_drug_metadata(self.db)
            if meta.drug_code in artifacts.hist_drug_codes and meta.drug_code != drug.drug_code
        }
        if not library_metadata:
            raise ValueError(
                "No historical library drugs available after excluding the target drug."
            )

        library_codes = sorted(library_metadata.keys())
        weekly_by_code = load_weekly_series(self.db, library_codes)

        hist_codes: list[str] = []
        hist_series: list[np.ndarray] = []
        hist_metadata = []
        for code in library_codes:
            series = weekly_by_code.get(code)
            if series is None or len(series) < artifacts.min_weeks:
                continue
            hist_codes.append(code)
            hist_series.append(series)
            hist_metadata.append(library_metadata[code])

        if not hist_codes:
            raise ValueError(
                "Historical library drugs lack sufficient weekly demand history. "
                "Ingest hospital_daily_demand_enriched and retrain CAMEO."
            )

        attrs_raw = artifacts.feature_encoder.transform(hist_metadata)
        hist_attrs_scaled = artifacts.scale_attrs(attrs_raw)
        hist_embeds = artifacts.metric_net.embed(hist_attrs_scaled)

        new_attr_raw = artifacts.feature_encoder.transform([drug])[0]
        new_attr_scaled = artifacts.scale_attrs(new_attr_raw.reshape(1, -1))[0]

        init = cameo_init_forecast(
            new_attr_scaled,
            hist_embeds,
            hist_series,
            artifacts.metric_net,
            topk=artifacts.topk,
        )
        neighbours = [(hist_codes[i], float(init.weights[j])) for j, i in enumerate(init.order)]
        embedding = artifacts.metric_net.embed(new_attr_scaled.reshape(1, -1))[0].tolist()

        observation_count = get_observation_count(drug.drug_code, self.db)
        stage = decide_stage(observation_count)

        if stage == "full_ensemble":
            return ColdStartPredictResponse(
                drug_code=drug.drug_code,
                stage_used=stage,
                observation_count=observation_count,
                embedding=embedding,
                nearest_neighbours=[code for code, _ in neighbours],
                similarity_scores=[score for _, score in neighbours],
                forecast=[],
                uncertainty_note=(
                    "This drug has sufficient history. Route to full SHIELD-XR forecasting ensemble."
                ),
                conformal_half_width_weekly=artifacts.conformal_half_width,
            )

        observed_weeks = load_weekly_series_for_drug(self.db, drug.drug_code)
        observed_for_cameo = observed_weeks[: min(len(observed_weeks), horizon_weeks)]
        drift_alarms = 0

        if len(observed_for_cameo) >= COLD_START_ONLY_BELOW:
            weekly_forecast, drift_flags = run_cameo(
                new_attr_scaled,
                hist_embeds,
                hist_series,
                artifacts.metric_net,
                observed_for_cameo,
                topk=artifacts.topk,
            )
            drift_alarms = int(sum(drift_flags))
            if len(weekly_forecast) < horizon_weeks:
                tail_level = weekly_forecast[-1] if len(weekly_forecast) else init.init_level
                tail = np.full(horizon_weeks - len(weekly_forecast), tail_level)
                weekly_forecast = np.concatenate([weekly_forecast, tail])
            else:
                weekly_forecast = weekly_forecast[:horizon_weeks]
        else:
            weekly_forecast, _ = forecast_cold_start_weeks(
                new_attr_scaled,
                hist_embeds,
                hist_series,
                artifacts.metric_net,
                horizon_weeks=horizon_weeks,
                observed_weeks=None,
                topk=artifacts.topk,
            )

        if request.pharmacist_estimate is not None:
            weekly_forecast = _apply_pharmacist_blend(
                weekly_forecast,
                request.pharmacist_estimate.weekly_units,
                request.pharmacist_estimate.confidence,
            )

        daily_forecast = _weekly_to_daily_forecast(
            weekly_forecast,
            horizon_days,
            half_width_weekly=artifacts.conformal_half_width,
        )

        return ColdStartPredictResponse(
            drug_code=drug.drug_code,
            stage_used=stage,
            observation_count=observation_count,
            embedding=embedding,
            nearest_neighbours=[code for code, _ in neighbours],
            similarity_scores=[score for _, score in neighbours],
            forecast=daily_forecast,
            uncertainty_note=_build_uncertainty_note(stage, observation_count, len(neighbours)),
            conformal_half_width_weekly=artifacts.conformal_half_width,
            drift_alarms=drift_alarms,
        )
