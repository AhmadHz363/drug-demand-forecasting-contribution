"""Segment-aware classical forecasting: ETS / Theta / TSB / Croston-SBA."""

from __future__ import annotations

import json
import logging
import os
import pickle
from typing import Any, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.forecasting.theta import ThetaModel

from app.forecasting.constants import (
    ARTIFACTS_DIR,
    CLASSICAL_CROSTON_ALPHA,
    CLASSICAL_SEASONAL_PERIOD,
    CLASSICAL_TSB_ALPHA_D,
    CLASSICAL_TSB_ALPHA_P,
    MIN_HISTORY_DAYS_CLASSICAL,
    PREDICTION_CAP_RECENT_DAYS,
)
from app.forecasting.demand_quantity import as_consumption_demand
from app.forecasting.demand_segmentation import classify_demand_segment_from_frame
from app.forecasting.models.base_model import BaseForecastingModel
from app.forecasting.models.feature_columns import ensure_demand_date_index, forecast_dates_from_index
from app.forecasting.prediction_bounds import demand_prediction_cap

logger = logging.getLogger(__name__)

# Syntetos–Boylan → classical method
SEGMENT_METHOD = {
    "smooth": "ets",
    "erratic": "theta",
    "intermittent": "tsb",
    "lumpy": "croston_sba",
}


def _artifact_path(drug_code: str) -> str:
    return os.path.join(ARTIFACTS_DIR, "classical", f"{drug_code}.pkl")


def _meta_path(drug_code: str) -> str:
    return os.path.join(ARTIFACTS_DIR, "classical", f"{drug_code}_meta.json")


def _target_series(df: pd.DataFrame) -> pd.Series:
    if "em_corrected_quantity" in df.columns:
        return df["em_corrected_quantity"].astype(float)
    return df["total_quantity"].astype(float)


def _covered_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Prefer the latest contiguous covered segment when coverage flags exist."""
    working = ensure_demand_date_index(df)
    if "is_coverage_gap" not in working.columns:
        return working
    covered = working[working["is_coverage_gap"].astype(int) == 0]
    if covered.empty:
        return working
    # Keep the latest contiguous covered run for classical/SARIMA-style models.
    gap_breaks = covered["is_coverage_gap"].astype(int).reindex(working.index).fillna(1)
    # covered already filtered; split on calendar gaps > 1 day
    dates = pd.to_datetime(covered.index)
    if len(dates) < 2:
        return covered
    breaks = np.where(np.diff(dates).astype("timedelta64[D]").astype(int) > 1)[0]
    if breaks.size == 0:
        return covered
    start = int(breaks[-1]) + 1
    return covered.iloc[start:]


def _fit_croston_sba(series: np.ndarray, alpha: float = CLASSICAL_CROSTON_ALPHA) -> dict[str, float]:
    values = np.asarray(series, dtype=float).reshape(-1)
    demand_sizes: list[float] = []
    intervals: list[float] = []
    since_last = 0
    for qty in values:
        since_last += 1
        if qty > 0.0:
            demand_sizes.append(float(qty))
            intervals.append(float(since_last))
            since_last = 0
    if not demand_sizes:
        return {"level": 0.0, "interval": max(float(len(values)), 1.0), "residual_std": 0.0}
    level = demand_sizes[0]
    interval = intervals[0]
    for size, gap in zip(demand_sizes[1:], intervals[1:]):
        level = alpha * size + (1.0 - alpha) * level
        interval = alpha * gap + (1.0 - alpha) * interval
    residuals = np.array(demand_sizes) - level
    residual_std = float(np.std(residuals, ddof=0)) if len(residuals) > 1 else level * 0.2
    return {
        "level": float(level),
        "interval": max(float(interval), 1.0),
        "residual_std": max(residual_std, 1e-6),
        "alpha": float(alpha),
        "sba_factor": float(1.0 - alpha / 2.0),
    }


def _croston_sba_forecast(
    state: dict[str, float],
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rate = state["level"] / state["interval"] * state.get("sba_factor", 1.0 - CLASSICAL_CROSTON_ALPHA / 2.0)
    p50 = np.full(horizon, max(rate, 0.0), dtype=float)
    spread = max(state["residual_std"] / state["interval"], rate * 0.2, 1e-6)
    p10 = np.clip(p50 - 1.28 * spread, 0.0, None)
    p90 = p50 + 1.28 * spread
    return p10, p50, p90


def _fit_tsb(
    series: np.ndarray,
    alpha_p: float = CLASSICAL_TSB_ALPHA_P,
    alpha_d: float = CLASSICAL_TSB_ALPHA_D,
) -> dict[str, float]:
    """Teunter–Syntetos–Babai occurrence/size smoothing."""
    values = np.asarray(series, dtype=float).reshape(-1)
    if values.size == 0:
        return {"p": 0.0, "z": 0.0, "residual_std": 0.0}
    p = 1.0 if values[0] > 0 else 0.0
    z = float(values[0]) if values[0] > 0 else 0.0
    residuals: list[float] = []
    for qty in values[1:]:
        occur = 1.0 if qty > 0.0 else 0.0
        p = alpha_p * occur + (1.0 - alpha_p) * p
        if occur > 0.0:
            residuals.append(qty - z)
            z = alpha_d * float(qty) + (1.0 - alpha_d) * z
    residual_std = float(np.std(residuals, ddof=0)) if len(residuals) > 1 else max(z * 0.2, 1e-6)
    return {
        "p": float(np.clip(p, 0.0, 1.0)),
        "z": float(max(z, 0.0)),
        "residual_std": max(residual_std, 1e-6),
        "alpha_p": float(alpha_p),
        "alpha_d": float(alpha_d),
    }


def _tsb_forecast(state: dict[str, float], horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rate = state["p"] * state["z"]
    p50 = np.full(horizon, max(rate, 0.0), dtype=float)
    spread = max(state["residual_std"] * state["p"], rate * 0.25, 1e-6)
    p10 = np.clip(p50 - 1.28 * spread, 0.0, None)
    p90 = p50 + 1.28 * spread
    return p10, p50, p90


def _ordered_quantiles(p10: np.ndarray, p50: np.ndarray, p90: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stacked = np.column_stack([p10, p50, p90])
    stacked = np.sort(np.clip(stacked, 0.0, None), axis=1)
    return stacked[:, 0], stacked[:, 1], stacked[:, 2]


class ClassicalModel(BaseForecastingModel):
    """Segment-aware classical: smooth→ETS, erratic→Theta, intermittent→TSB, lumpy→Croston-SBA."""

    def __init__(self) -> None:
        self._method: str = "ets"
        self._segment: str = "smooth"
        self._ets_result = None
        self._theta_result = None
        self._intermittent_state: Optional[dict[str, float]] = None
        self._last_index: Optional[pd.DatetimeIndex] = None
        self._residual_std: float = 0.0
        self._prediction_cap: float = 50.0
        self._training_coverage: dict[str, Any] = {}

    def train(self, df: pd.DataFrame, drug_code: str, **kwargs) -> None:
        del kwargs
        self.validate_min_history(df, MIN_HISTORY_DAYS_CLASSICAL, "Classical")
        working = _covered_frame(df)
        if len(working) < MIN_HISTORY_DAYS_CLASSICAL:
            working = ensure_demand_date_index(df)
        segment = classify_demand_segment_from_frame(working)
        self._segment = segment
        self._method = SEGMENT_METHOD.get(segment, "ets")
        series = np.asarray(as_consumption_demand(_target_series(working)), dtype=float)
        self._last_index = working.index
        self._prediction_cap = demand_prediction_cap(
            series,
            recent_days=PREDICTION_CAP_RECENT_DAYS,
        )
        self._ets_result = None
        self._theta_result = None
        self._intermittent_state = None
        coverage_start = working.index.min()
        coverage_end = working.index.max()
        self._training_coverage = {
            "start": str(getattr(coverage_start, "date", lambda: coverage_start)()),
            "end": str(getattr(coverage_end, "date", lambda: coverage_end)()),
            "n_days": int(len(working)),
        }

        if self._method == "croston_sba":
            self._intermittent_state = _fit_croston_sba(series)
            self._residual_std = self._intermittent_state["residual_std"]
        elif self._method == "tsb":
            self._intermittent_state = _fit_tsb(series)
            self._residual_std = self._intermittent_state["residual_std"]
        elif self._method == "theta":
            ets_series = pd.Series(series, index=working.index)
            try:
                theta = ThetaModel(ets_series, period=CLASSICAL_SEASONAL_PERIOD, deseasonalize=True)
                self._theta_result = theta.fit()
                fitted = np.asarray(self._theta_result.fittedvalues, dtype=float)
            except Exception:
                # Fall back to damped ETS when Theta fails on short/odd series.
                self._method = "ets"
                model = ExponentialSmoothing(
                    ets_series,
                    trend="add",
                    damped_trend=True,
                    seasonal=None,
                    initialization_method="estimated",
                )
                self._ets_result = model.fit(optimized=True)
                fitted = np.asarray(self._ets_result.fittedvalues, dtype=float)
            residuals = series[: len(fitted)] - fitted
            self._residual_std = float(np.std(residuals, ddof=0)) if len(residuals) > 1 else 1.0
        else:
            ets_series = pd.Series(series, index=working.index)
            seasonal_periods = CLASSICAL_SEASONAL_PERIOD if len(series) >= 2 * CLASSICAL_SEASONAL_PERIOD else None
            try:
                model = ExponentialSmoothing(
                    ets_series,
                    trend="add",
                    seasonal="add" if seasonal_periods else None,
                    seasonal_periods=seasonal_periods,
                    initialization_method="estimated",
                )
                self._ets_result = model.fit(optimized=True)
            except Exception:
                model = ExponentialSmoothing(
                    ets_series,
                    trend="add",
                    seasonal=None,
                    initialization_method="estimated",
                )
                self._ets_result = model.fit(optimized=True)
            fitted = np.asarray(self._ets_result.fittedvalues, dtype=float)
            residuals = series - fitted
            self._residual_std = float(np.std(residuals, ddof=0)) if len(residuals) > 1 else 1.0

        logger.info(
            "Classical model trained for %s on %d days (segment=%s, method=%s)",
            drug_code,
            len(working),
            self._segment,
            self._method,
        )

    def predict(self, df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
        if self._method in {"croston_sba", "tsb"} and self._intermittent_state is None:
            raise RuntimeError("Classical model is not trained. Call train() or load() first.")
        if self._method == "ets" and self._ets_result is None:
            raise RuntimeError("Classical model is not trained. Call train() or load() first.")
        if self._method == "theta" and self._theta_result is None and self._ets_result is None:
            raise RuntimeError("Classical model is not trained. Call train() or load() first.")

        working = ensure_demand_date_index(df)
        index = self._last_index if self._last_index is not None else working.index
        forecast_dates = forecast_dates_from_index(index, horizon_days)

        if self._method == "croston_sba":
            p10, p50, p90 = _croston_sba_forecast(self._intermittent_state, horizon_days)
        elif self._method == "tsb":
            p10, p50, p90 = _tsb_forecast(self._intermittent_state, horizon_days)
        elif self._method == "theta" and self._theta_result is not None:
            forecast = self._theta_result.forecast(horizon_days)
            p50 = np.clip(np.asarray(forecast, dtype=float), 0.0, None)
            spread = max(self._residual_std, float(np.mean(p50)) * 0.15, 1e-6)
            p10 = np.clip(p50 - 1.28 * spread, 0.0, None)
            p90 = p50 + 1.28 * spread
        else:
            forecast = self._ets_result.forecast(horizon_days)
            p50 = np.clip(np.asarray(forecast, dtype=float), 0.0, None)
            spread = max(self._residual_std, float(np.mean(p50)) * 0.15, 1e-6)
            p10 = np.clip(p50 - 1.28 * spread, 0.0, None)
            p90 = p50 + 1.28 * spread

        p10, p50, p90 = _ordered_quantiles(p10, p50, p90)
        cap = self._prediction_cap
        p10 = np.clip(p10, 0.0, cap)
        p50 = np.clip(p50, 0.0, cap)
        p90 = np.clip(p90, 0.0, cap)
        p10, p50, p90 = _ordered_quantiles(p10, p50, p90)

        return pd.DataFrame(
            {
                "forecast_date": forecast_dates,
                "p10": p10,
                "p50": p50,
                "p90": p90,
            }
        )

    def save(self, drug_code: str) -> str:
        path = _artifact_path(drug_code)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(
                {
                    "method": self._method,
                    "segment": self._segment,
                    "ets_result": self._ets_result,
                    "theta_result": self._theta_result,
                    "intermittent_state": self._intermittent_state,
                    "croston_state": self._intermittent_state,  # backward-compatible key
                    "last_index": self._last_index,
                    "residual_std": self._residual_std,
                    "prediction_cap": self._prediction_cap,
                    "training_coverage": self._training_coverage,
                },
                handle,
            )
        with open(_meta_path(drug_code), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "method": self._method,
                    "segment": self._segment,
                    "prediction_cap": self._prediction_cap,
                    "training_coverage": self._training_coverage,
                },
                handle,
                indent=2,
            )
        return path

    def load(self, drug_code: str) -> None:
        path = _artifact_path(drug_code)
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        method = payload.get("method", "ets")
        # Normalize legacy method names from earlier classical prototype.
        if method == "croston":
            method = "croston_sba"
        self._method = method
        self._segment = payload.get("segment", "smooth")
        self._ets_result = payload.get("ets_result")
        self._theta_result = payload.get("theta_result")
        self._intermittent_state = payload.get("intermittent_state") or payload.get("croston_state")
        self._last_index = payload.get("last_index")
        self._residual_std = float(payload.get("residual_std", 0.0))
        self._prediction_cap = float(payload.get("prediction_cap", 50.0))
        self._training_coverage = payload.get("training_coverage") or {}

    def is_trained(self, drug_code: str) -> bool:
        return os.path.isfile(_artifact_path(drug_code))
