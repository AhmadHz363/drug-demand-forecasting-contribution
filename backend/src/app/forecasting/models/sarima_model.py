"""SARIMA forecasting model using statsmodels SARIMAX."""

from __future__ import annotations

import logging
import os
import pickle
from typing import Optional

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from app.forecasting.constants import (
    ARTIFACTS_DIR,
    MIN_HISTORY_DAYS_SARIMA,
    SARIMA_FORECAST_SHRINKAGE,
    SARIMA_MAX_ITER,
    SARIMA_ORDER,
    SARIMA_RECENT_LEVEL_DAYS,
    SARIMA_SEASONAL_ORDER,
    SARIMA_TRAIN_DAYS,
)
from app.forecasting.demand_quantity import as_consumption_demand
from app.forecasting.models.base_model import BaseForecastingModel
from app.forecasting.models.feature_columns import ensure_demand_date_index, forecast_dates_from_index

logger = logging.getLogger(__name__)


def _artifact_path(drug_code: str) -> str:
    return os.path.join(ARTIFACTS_DIR, "sarima", f"{drug_code}.pkl")


def _target_series(df: pd.DataFrame) -> pd.Series:
    if "em_corrected_quantity" in df.columns:
        return df["em_corrected_quantity"].astype(float)
    return df["total_quantity"].astype(float)


class SarimaModel(BaseForecastingModel):
    def __init__(self) -> None:
        self._result = None
        self._last_index: Optional[pd.DatetimeIndex] = None
        self._recent_level: float = 0.0

    def train(self, df: pd.DataFrame, drug_code: str) -> None:
        self.validate_min_history(df, MIN_HISTORY_DAYS_SARIMA, "SARIMA")
        working = ensure_demand_date_index(df)
        if len(working) > SARIMA_TRAIN_DAYS:
            working = working.iloc[-SARIMA_TRAIN_DAYS :]
        series = pd.Series(
            as_consumption_demand(_target_series(working)),
            index=working.index,
        )

        model = SARIMAX(
            series,
            order=SARIMA_ORDER,
            seasonal_order=SARIMA_SEASONAL_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._result = model.fit(disp=False, maxiter=SARIMA_MAX_ITER)
        self._last_index = working.index
        tail = series.iloc[-SARIMA_RECENT_LEVEL_DAYS :]
        self._recent_level = float(tail.median()) if not tail.empty else float(series.median())
        logger.info("SARIMA trained for %s on %d days", drug_code, len(working))

    def predict(self, df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
        if self._result is None:
            raise RuntimeError("SARIMA model is not trained. Call train() or load() first.")

        working = ensure_demand_date_index(df)
        index = self._last_index if self._last_index is not None else working.index
        series = pd.Series(
            as_consumption_demand(_target_series(working)),
            index=working.index,
        )
        recent_tail = series.iloc[-SARIMA_RECENT_LEVEL_DAYS :]
        recent_level = (
            float(recent_tail.median())
            if not recent_tail.empty
            else self._recent_level
        )

        forecast = self._result.get_forecast(steps=horizon_days)
        conf = forecast.conf_int(alpha=0.20)
        shrink = SARIMA_FORECAST_SHRINKAGE

        p50_raw = forecast.predicted_mean.clip(lower=0.0).astype(float)
        p50 = (1.0 - shrink) * p50_raw + shrink * recent_level
        p10_raw = conf.iloc[:, 0].clip(lower=0.0).astype(float)
        p90_raw = conf.iloc[:, 1].clip(lower=0.0).astype(float)
        p10 = (1.0 - shrink) * p10_raw + shrink * max(recent_level * 0.8, 0.0)
        p90 = (1.0 - shrink) * p90_raw + shrink * recent_level * 1.2

        dates = forecast_dates_from_index(index, horizon_days)
        return pd.DataFrame(
            {
                "forecast_date": dates,
                "p10": p10.values,
                "p50": p50.values,
                "p90": p90.values,
            }
        )

    def save(self, drug_code: str) -> str:
        if self._result is None:
            raise RuntimeError("Cannot save untrained SARIMA model")
        path = _artifact_path(drug_code)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(
                {
                    "result": self._result,
                    "last_index": self._last_index,
                    "recent_level": self._recent_level,
                },
                handle,
            )
        return path

    def load(self, drug_code: str) -> None:
        path = _artifact_path(drug_code)
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        self._result = payload["result"]
        self._last_index = payload.get("last_index")
        self._recent_level = float(payload.get("recent_level", 0.0))

    def is_trained(self, drug_code: str) -> bool:
        return os.path.isfile(_artifact_path(drug_code))
