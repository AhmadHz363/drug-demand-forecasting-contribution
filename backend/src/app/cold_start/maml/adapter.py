"""Fast MAML adaptation at inference time."""

from __future__ import annotations

import logging
import os
import statistics
from datetime import date, timedelta

import learn2learn as l2l
import torch
import torch.nn as nn

from app.cold_start.constants import (
    ARTIFACTS_DIR,
    BLEND_UNTIL,
    COLD_START_ONLY_BELOW,
    MAML_FORECAST_HORIZON,
    MAML_INNER_LR,
    MAML_INNER_STEPS,
    MAML_SEQUENCE_LEN,
)
from app.cold_start.maml.model import (
    BaseForecaster,
    day_features,
    pad_series_left,
    series_window_to_tensor,
)
from app.cold_start.schemas import DailyForecast

logger = logging.getLogger(__name__)

MAML_ARTIFACT = "maml_base.pt"
PREDICTION_INTERVAL_Z = 1.28

_maml_cache: l2l.algorithms.MAML | None = None


def _artifact_path() -> str:
    return os.path.join(ARTIFACTS_DIR, MAML_ARTIFACT)


def clear_maml_cache() -> None:
    """Reset in-memory MAML cache (e.g. after retraining)."""
    global _maml_cache
    _maml_cache = None


def _load_maml() -> l2l.algorithms.MAML | None:
    global _maml_cache
    if _maml_cache is not None:
        return _maml_cache

    path = _artifact_path()
    if not os.path.isfile(path):
        return None

    base_model = BaseForecaster()
    state_dict = torch.load(path, map_location=torch.device("cpu"), weights_only=True)
    base_model.load_state_dict(state_dict)
    base_model.eval()
    _maml_cache = l2l.algorithms.MAML(base_model, lr=MAML_INNER_LR, first_order=False)
    return _maml_cache


def _blend_alpha(observation_count: int) -> float:
    span = BLEND_UNTIL - COLD_START_ONLY_BELOW
    if span <= 0:
        return 1.0
    return (observation_count - COLD_START_ONLY_BELOW) / span


def _build_adaptation_pairs(
    observations: list[tuple[date, float]],
    mean: float,
    std: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    sorted_obs = sorted(observations, key=lambda item: item[0])
    inputs: list[list[list[float]]] = []
    targets: list[list[float]] = []

    for offset in range(len(sorted_obs)):
        end_input = offset + MAML_SEQUENCE_LEN
        end_target = end_input + MAML_FORECAST_HORIZON
        if end_target > len(sorted_obs):
            break

        window = pad_series_left(sorted_obs[:end_input], MAML_SEQUENCE_LEN, mean)
        target_window = sorted_obs[end_input:end_target]
        inputs.append([day_features(d, q, mean, std) for d, q in window])
        targets.append([((q - mean) / std) for _, q in target_window])

    if not inputs:
        return None

    return (
        torch.tensor(inputs, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
    )


def _run_maml_forecast(
    observations: list[tuple[date, float]],
    horizon_days: int,
) -> list[float]:
    maml = _load_maml()
    if maml is None:
        return []

    sorted_obs = sorted(observations, key=lambda item: item[0])
    quantities = [quantity for _, quantity in sorted_obs]
    mean = statistics.mean(quantities)
    std = statistics.stdev(quantities) if len(quantities) > 1 else max(abs(mean) * 0.1, 1.0)
    if std <= 0:
        std = 1.0

    learner = maml.clone()
    criterion = nn.MSELoss()

    adapt_pairs = _build_adaptation_pairs(sorted_obs, mean, std)
    if adapt_pairs is not None:
        adapt_x, adapt_y = adapt_pairs
        for _ in range(MAML_INNER_STEPS):
            loss = criterion(learner(adapt_x), adapt_y)
            learner.adapt(loss)

    inference_window = pad_series_left(sorted_obs, MAML_SEQUENCE_LEN, mean)
    inference_x = series_window_to_tensor(inference_window, mean, std)

    with torch.no_grad():
        normalized = learner(inference_x).squeeze(0).tolist()

    denormalized = [(value * std) + mean for value in normalized]

    if horizon_days <= len(denormalized):
        return denormalized[:horizon_days]

    extended = list(denormalized)
    last_value = denormalized[-1] if denormalized else mean
    while len(extended) < horizon_days:
        extended.append(last_value)
    return extended


def _observation_uncertainty(observations: list[tuple[date, float]], fallback: float) -> float:
    quantities = [quantity for _, quantity in observations]
    if len(quantities) > 1:
        return statistics.stdev(quantities)
    return fallback


def _maml_daily_forecasts(
    maml_p50: list[float],
    observations: list[tuple[date, float]],
    baseline_forecast: list[DailyForecast],
    start_date: date,
) -> list[DailyForecast]:
    baseline_std = 0.0
    if baseline_forecast:
        sample = baseline_forecast[0]
        baseline_std = max(
            (sample.p90 - sample.p50) / PREDICTION_INTERVAL_Z,
            0.0,
        )

    obs_std = _observation_uncertainty(observations, baseline_std or 0.1)

    forecasts: list[DailyForecast] = []
    for offset, p50 in enumerate(maml_p50):
        p10 = max(0.0, p50 - PREDICTION_INTERVAL_Z * obs_std)
        p90 = p50 + PREDICTION_INTERVAL_Z * obs_std
        forecasts.append(
            DailyForecast(
                date=start_date + timedelta(days=offset),
                p10=p10,
                p50=p50,
                p90=p90,
            )
        )
    return forecasts


def _blend_forecasts(
    baseline: list[DailyForecast],
    maml: list[DailyForecast],
    alpha: float,
) -> list[DailyForecast]:
    blended: list[DailyForecast] = []
    for base_day, maml_day in zip(baseline, maml):
        blended.append(
            DailyForecast(
                date=base_day.date,
                p10=(1.0 - alpha) * base_day.p10 + alpha * maml_day.p10,
                p50=(1.0 - alpha) * base_day.p50 + alpha * maml_day.p50,
                p90=(1.0 - alpha) * base_day.p90 + alpha * maml_day.p90,
            )
        )
    return blended


def adapt_and_forecast(
    real_observations: list[tuple[date, float]],
    horizon_days: int,
    baseline_forecast: list[DailyForecast],
) -> list[DailyForecast]:
    """
    Adapt the meta-trained model on real observations and return a refined forecast.

    Fewer than COLD_START_ONLY_BELOW observations → baseline unchanged.
    COLD_START_ONLY_BELOW <= n < BLEND_UNTIL → blend MAML with baseline.
    n >= BLEND_UNTIL → MAML-only forecast.
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be at least 1")

    observation_count = len(real_observations)
    if observation_count < COLD_START_ONLY_BELOW:
        return baseline_forecast

    if not os.path.isfile(_artifact_path()):
        logger.warning(
            "MAML weights not found at %s; falling back to baseline forecast.",
            _artifact_path(),
        )
        return baseline_forecast

    maml_p50 = _run_maml_forecast(real_observations, horizon_days)
    if not maml_p50:
        logger.warning("MAML forecast unavailable; falling back to baseline forecast.")
        return baseline_forecast

    start_date = (
        baseline_forecast[0].date
        if baseline_forecast
        else date.today() + timedelta(days=1)
    )
    maml_forecast = _maml_daily_forecasts(
        maml_p50,
        real_observations,
        baseline_forecast,
        start_date,
    )

    if len(maml_forecast) < horizon_days:
        last = maml_forecast[-1] if maml_forecast else DailyForecast(
            date=start_date,
            p10=0.0,
            p50=0.0,
            p90=0.0,
        )
        while len(maml_forecast) < horizon_days:
            next_date = start_date + timedelta(days=len(maml_forecast))
            maml_forecast.append(
                DailyForecast(
                    date=next_date,
                    p10=last.p10,
                    p50=last.p50,
                    p90=last.p90,
                )
            )

    if observation_count >= BLEND_UNTIL:
        return maml_forecast[:horizon_days]

    alpha = _blend_alpha(observation_count)
    alpha = max(0.0, min(1.0, alpha))
    baseline_slice = baseline_forecast[:horizon_days]
    maml_slice = maml_forecast[:horizon_days]
    return _blend_forecasts(baseline_slice, maml_slice, alpha)
