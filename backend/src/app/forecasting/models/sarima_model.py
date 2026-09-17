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
    SARIMA_MAX_ITER,
    SARIMA_ORDER,
    SARIMA_RECENT_LEVEL_DAYS,
    SARIMA_SEASONAL_ORDER,
)
from app.forecasting.demand_quantity import as_consumption_demand
from app.forecasting.demand_segmentation import classify_demand_segment_from_frame
from app.forecasting.model_adaptation import (
    adaptive_sarima_shrinkage,
    recent_cv2,
    search_sarima_orders,
    select_sarima_train_days,
)
from app.forecasting.models.base_model import BaseForecastingModel
from app.forecasting.models.feature_columns import ensure_demand_date_index, forecast_dates_from_index

logger = logging.getLogger(__name__)


def _artifact_path(drug_code: str) -> str:
    return os.path.join(ARTIFACTS_DIR, "sarima", f"{drug_code}.pkl")


def _target_series(df: pd.DataFrame) -> pd.Series:
    if "em_corrected_quantity" in df.columns:
        return df["em_corrected_quantity"].astype(float)
    return df["total_quantity"].astype(float)


def _holiday_exog(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if "is_public_holiday" not in df.columns:
        return None
    return df[["is_public_holiday"]].astype(float)


class SarimaModel(BaseForecastingModel):
    def __init__(self) -> None:
        self._result = None
        self._last_index: Optional[pd.DatetimeIndex] = None
        self._recent_level: float = 0.0
        self._shrinkage: float = 0.4
        self._order = SARIMA_ORDER
        self._seasonal_order = SARIMA_SEASONAL_ORDER
        self._future_exog: Optional[pd.DataFrame] = None

    def train(self, df: pd.DataFrame, drug_code: str) -> None:
        self.validate_min_history(df, MIN_HISTORY_DAYS_SARIMA, "SARIMA")
        working = ensure_demand_date_index(df)
        segment = classify_demand_segment_from_frame(working)
        # as_consumption_demand returns an ndarray; wrap it back into a Series
        # so that downstream .values calls and index-aligned operations are safe.
        raw_series = pd.Series(
            as_consumption_demand(_target_series(working)),
            index=working.index,
            dtype=float,
        )
        cv2 = recent_cv2(raw_series.values)
        train_days = select_sarima_train_days(
            segment,
            len(working),
            cv2=cv2,
        )
        if len(working) > train_days:
            working = working.iloc[-train_days:]

        series = pd.Series(
            as_consumption_demand(_target_series(working)),
            index=working.index,
        )
        self._order, self._seasonal_order = search_sarima_orders(series)
        exog = _holiday_exog(working)

        model = SARIMAX(
            series,
            exog=exog,
            order=self._order,
            seasonal_order=self._seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._result = model.fit(disp=False, maxiter=SARIMA_MAX_ITER)
        self._last_index = working.index
        tail = series.iloc[-SARIMA_RECENT_LEVEL_DAYS:]
        self._recent_level = float(tail.median()) if not tail.empty else float(series.median())
        self._shrinkage = adaptive_sarima_shrinkage(cv2)
        logger.info(
            "SARIMA trained for %s on %d days (segment=%s, order=%s, shrinkage=%.3f)",
            drug_code,
            len(working),
            segment,
            self._order,
            self._shrinkage,
        )

    def predict(self, df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
        if self._result is None:
            raise RuntimeError("SARIMA model is not trained. Call train() or load() first.")

        working = ensure_demand_date_index(df)
        index = self._last_index if self._last_index is not None else working.index
        series = pd.Series(
            as_consumption_demand(_target_series(working)),
            index=working.index,
        )
        recent_tail = series.iloc[-SARIMA_RECENT_LEVEL_DAYS:]
        recent_level = (
            float(recent_tail.median())
            if not recent_tail.empty
            else self._recent_level
        )
        shrink = self._shrinkage

        forecast_dates = forecast_dates_from_index(index, horizon_days)
        future_holidays = None
        if "is_public_holiday" in working.columns:
            holiday_rows = []
            for ts in forecast_dates:
                if ts in working.index:
                    holiday_rows.append(float(working.loc[ts, "is_public_holiday"]))
                else:
                    holiday_rows.append(0.0)
            future_holidays = pd.DataFrame(
                {"is_public_holiday": holiday_rows},
                index=forecast_dates,
            )

        forecast = self._result.get_forecast(steps=horizon_days, exog=future_holidays)
        conf = forecast.conf_int(alpha=0.20)

        p50_raw = forecast.predicted_mean.clip(lower=0.0).astype(float)
        p50 = (1.0 - shrink) * p50_raw + shrink * recent_level
        p10_raw = conf.iloc[:, 0].clip(lower=0.0).astype(float)
        p90_raw = conf.iloc[:, 1].clip(lower=0.0).astype(float)
        p10 = (1.0 - shrink) * p10_raw + shrink * max(recent_level * 0.8, 0.0)
        p90 = (1.0 - shrink) * p90_raw + shrink * recent_level * 1.2

        return pd.DataFrame(
            {
                "forecast_date": forecast_dates,
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
                    "shrinkage": self._shrinkage,
                    "order": self._order,
                    "seasonal_order": self._seasonal_order,
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
        self._shrinkage = float(payload.get("shrinkage", 0.4))
        self._order = tuple(payload.get("order", SARIMA_ORDER))
        self._seasonal_order = tuple(payload.get("seasonal_order", SARIMA_SEASONAL_ORDER))

    def is_trained(self, drug_code: str) -> bool:
        return os.path.isfile(_artifact_path(drug_code))
