"""LightGBM quantile regression forecasting model."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd

from app.forecasting.constants import (
    ARTIFACTS_DIR,
    LGBM_LOOKBACK_WINDOW,
    LGBM_QUANTILES,
    LGBM_VALID_FRACTION,
    MIN_HISTORY_DAYS_LGBM,
)
from app.forecasting.demand_segmentation import classify_demand_segment_from_frame
from app.forecasting.model_adaptation import (
    adaptive_lgbm_anchor_weight,
    lgbm_training_params,
    lgbm_training_target,
    recent_cv2,
)
from app.forecasting.prediction_bounds import demand_prediction_cap, winsorize_predictions
from app.forecasting.demand_quantity import as_consumption_demand
from app.forecasting.models.base_model import BaseForecastingModel
from app.forecasting.feature_engineering.forecast_features import build_recursive_feature_row
from app.forecasting.models.feature_columns import (
    LGBM_CATEGORICAL_FEATURES,
    ensure_demand_date_index,
    lgbm_feature_columns,
)

logger = logging.getLogger(__name__)


def _import_lightgbm():
    try:
        import lightgbm as lgb
    except OSError as exc:
        raise ImportError(
            "lightgbm failed to load its native library (libomp). "
            "On macOS install OpenMP with: brew install libomp"
        ) from exc
    except ImportError as exc:
        raise ImportError(
            "lightgbm is required for gradient-boosting forecasts. "
            "Install it with: pip install 'lightgbm>=4.0.0' (see requirements.txt)."
        ) from exc
    return lgb


def _artifact_path(drug_code: str, quantile: float) -> str:
    label = int(quantile * 100)
    return os.path.join(ARTIFACTS_DIR, "lgbm", f"{drug_code}_p{label}.txt")


def _shap_path(drug_code: str) -> str:
    return os.path.join(ARTIFACTS_DIR, "lgbm", f"{drug_code}_shap.json")


class LightGBMModel(BaseForecastingModel):
    def __init__(self) -> None:
        self._boosters: dict[float, Any] = {}
        self._feature_names: list[str] = []
        self._categorical_features: list[str] = []
        self._anchor_weight: float = 0.25
        self._recent_cv2: float = 0.0

    def _prepare_xy(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        working = ensure_demand_date_index(df)
        features = lgbm_feature_columns(working)
        self._feature_names = features
        self._categorical_features = [c for c in LGBM_CATEGORICAL_FEATURES if c in working.columns]
        x = working[features].copy()
        for col in self._categorical_features:
            x[col] = x[col].astype("category")
        y = pd.Series(
            as_consumption_demand(lgbm_training_target(working)),
            index=working.index,
        )
        return x, y

    def train(self, df: pd.DataFrame, drug_code: str) -> None:
        lgb = _import_lightgbm()
        self.validate_min_history(df, MIN_HISTORY_DAYS_LGBM, "LightGBM")
        working = ensure_demand_date_index(df)
        if len(working) > LGBM_LOOKBACK_WINDOW:
            working = working.iloc[-LGBM_LOOKBACK_WINDOW :]
        segment = classify_demand_segment_from_frame(working)
        qty_col = (
            "observed_quantity"
            if "observed_quantity" in working.columns
            else "total_quantity"
        )
        self._recent_cv2 = recent_cv2(working[qty_col].astype(float).values)
        self._anchor_weight = adaptive_lgbm_anchor_weight(self._recent_cv2)
        train_params = lgbm_training_params(segment)
        x, y = self._prepare_xy(working)

        split_idx = max(int(len(x) * (1.0 - LGBM_VALID_FRACTION)), 1)
        x_train, x_valid = x.iloc[:split_idx], x.iloc[split_idx:]
        y_train, y_valid = y.iloc[:split_idx], y.iloc[split_idx:]

        train_set = lgb.Dataset(
            x_train,
            label=y_train,
            categorical_feature=self._categorical_features or "auto",
        )
        valid_set = lgb.Dataset(
            x_valid,
            label=y_valid,
            reference=train_set,
            categorical_feature=self._categorical_features or "auto",
        )

        params = {
            "objective": "quantile",
            "metric": "quantile",
            "learning_rate": train_params["learning_rate"],
            "max_depth": train_params["max_depth"],
            "num_leaves": train_params["num_leaves"],
            "verbosity": -1,
        }
        early_stopping = int(train_params["early_stopping_rounds"])
        n_estimators = int(train_params["n_estimators"])

        self._boosters = {}
        for quantile in LGBM_QUANTILES:
            if quantile == 0.50:
                q_params = {
                    **params,
                    "objective": "tweedie",
                    "tweedie_variance_power": 1.5,
                    "metric": "tweedie",
                }
            else:
                q_params = {**params, "alpha": quantile}
            booster = lgb.train(
                q_params,
                train_set,
                num_boost_round=n_estimators,
                valid_sets=[valid_set],
                callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
            )
            self._boosters[quantile] = booster

        self._save_shap_summary(x_train, drug_code)
        logger.info(
            "LightGBM trained for %s on %d days (segment=%s, anchor=%.3f)",
            drug_code,
            len(x),
            segment,
            self._anchor_weight,
        )

    def _save_shap_summary(self, x_train: pd.DataFrame, drug_code: str) -> None:
        try:
            import shap
        except ImportError as exc:
            logger.warning("shap not installed — skipping SHAP export: %s", exc)
            return

        p50_booster = self._boosters.get(0.50)
        if p50_booster is None:
            return

        explainer = shap.TreeExplainer(p50_booster)
        shap_values = explainer.shap_values(x_train)
        mean_abs = np.abs(shap_values).mean(axis=0)
        summary = {
            feature: float(value)
            for feature, value in zip(x_train.columns.tolist(), mean_abs)
        }
        path = _shap_path(drug_code)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    def _predict_feature_row(self, feature_row: dict[str, float]) -> tuple[float, float, float]:
        row = pd.DataFrame([feature_row])
        for col in self._categorical_features:
            if col in row.columns:
                row[col] = row[col].astype("category")
        x_row = row[self._feature_names]
        p10 = float(self._boosters[0.10].predict(x_row)[0])
        p50 = float(self._boosters[0.50].predict(x_row)[0])
        p90 = float(self._boosters[0.90].predict(x_row)[0])
        ordered = np.sort(np.clip([p10, p50, p90], 0.0, None))
        return float(ordered[0]), float(ordered[1]), float(ordered[2])

    def _dampen_recursive_prediction(
        self,
        predicted_p50: float,
        quantity_history: np.ndarray,
        step: int,
        median_qty: float,
    ) -> float:
        recent_tail = quantity_history[-28:] if quantity_history.size >= 7 else quantity_history
        recent_mean = float(np.mean(recent_tail)) if recent_tail.size else median_qty
        recent_std = float(np.std(recent_tail, ddof=0)) if recent_tail.size else 1.0
        if recent_std == 0.0:
            recent_std = 1.0

        damped = float(
            np.clip(
                predicted_p50,
                max(0.0, recent_mean - 2 * recent_std),
                recent_mean + 2 * recent_std,
            )
        )
        if step > 7:
            median_demand = float(np.median(quantity_history)) if quantity_history.size else median_qty
            blend_weight = min(0.5, (step - 7) / 30)
            damped = (1.0 - blend_weight) * damped + blend_weight * median_demand
        return damped

    def _apply_drift_guard(
        self,
        p10: float,
        p50: float,
        p90: float,
        quantity_history: np.ndarray,
    ) -> tuple[float, float, float]:
        recent_tail = quantity_history[-14:] if quantity_history.size >= 7 else quantity_history
        recent_mean = float(np.mean(recent_tail)) if recent_tail.size else 0.0
        if recent_mean > 0 and (p50 / recent_mean > 3.0 or p50 / recent_mean < 0.1):
            p50 = recent_mean
            p10 = p50 * 0.3
            p90 = p50 * 3.0
        return p10, p50, p90

    def predict(
        self,
        df: pd.DataFrame,
        horizon_days: int,
        *,
        future_covariates: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if not self._boosters:
            raise RuntimeError("LightGBM model is not trained. Call train() or load() first.")

        working = ensure_demand_date_index(df)
        qty = pd.Series(
            as_consumption_demand(working["total_quantity"]),
            index=working.index,
        )
        median_qty = float(qty.median())
        quantity_history = qty.to_numpy()
        last_date = pd.Timestamp(working.index[-1])

        if future_covariates is not None:
            covariates = ensure_demand_date_index(future_covariates)
        else:
            covariates = None

        forecast_dates: list[pd.Timestamp] = []
        p10_values: list[float] = []
        p50_values: list[float] = []
        p90_values: list[float] = []

        for step in range(1, horizon_days + 1):
            forecast_date = last_date + pd.Timedelta(days=step)
            cov_row = None
            if covariates is not None and step <= len(covariates):
                cov_row = covariates.iloc[step - 1]
            elif covariates is not None:
                cov_row = covariates.iloc[-1]

            feature_row = build_recursive_feature_row(
                forecast_date,
                quantity_history,
                median_qty=median_qty,
                covariate_row=cov_row,
                forecast_step=step,
            )
            p10, p50, p90 = self._predict_feature_row(feature_row)
            p10, p50, p90 = self._apply_drift_guard(p10, p50, p90, quantity_history)
            anchor_window = quantity_history[-28:] if quantity_history.size >= 7 else quantity_history
            anchor = float(np.median(anchor_window)) if anchor_window.size else median_qty
            anchor_w = self._anchor_weight
            p50 = (1.0 - anchor_w) * p50 + anchor_w * anchor
            p10 = (1.0 - anchor_w) * p10 + anchor_w * max(anchor * 0.8, 0.0)
            p90 = (1.0 - anchor_w) * p90 + anchor_w * max(anchor * 1.2, p50)
            cap = demand_prediction_cap(quantity_history)
            p50 = float(winsorize_predictions(np.array([p50]), cap)[0])
            p10 = float(winsorize_predictions(np.array([p10]), cap)[0])
            p90 = float(winsorize_predictions(np.array([p90]), cap)[0])
            p10, p50, p90 = sorted([p10, p50, p90])
            forecast_dates.append(forecast_date)
            p10_values.append(p10)
            p50_values.append(p50)
            p90_values.append(p90)
            damped = self._dampen_recursive_prediction(p50, quantity_history, step, median_qty)
            quantity_history = np.append(quantity_history, damped)

        return pd.DataFrame(
            {
                "forecast_date": forecast_dates,
                "p10": p10_values,
                "p50": p50_values,
                "p90": p90_values,
            }
        )

    def save(self, drug_code: str) -> str:
        if not self._boosters:
            raise RuntimeError("Cannot save untrained LightGBM model")
        paths = []
        for quantile in LGBM_QUANTILES:
            path = _artifact_path(drug_code, quantile)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._boosters[quantile].save_model(path)
            paths.append(path)
        meta_path = os.path.join(ARTIFACTS_DIR, "lgbm", f"{drug_code}_meta.json")
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "feature_names": self._feature_names,
                    "categorical_features": self._categorical_features,
                    "anchor_weight": self._anchor_weight,
                    "recent_cv2": self._recent_cv2,
                },
                handle,
            )
        return paths[0]

    def load(self, drug_code: str) -> None:
        lgb = _import_lightgbm()
        meta_path = os.path.join(ARTIFACTS_DIR, "lgbm", f"{drug_code}_meta.json")
        with open(meta_path, encoding="utf-8") as handle:
            meta = json.load(handle)
        self._feature_names = meta["feature_names"]
        self._categorical_features = meta.get("categorical_features", [])
        self._anchor_weight = float(meta.get("anchor_weight", 0.25))
        self._recent_cv2 = float(meta.get("recent_cv2", 0.0))
        self._boosters = {}
        for quantile in LGBM_QUANTILES:
            self._boosters[quantile] = lgb.Booster(model_file=_artifact_path(drug_code, quantile))

    def is_trained(self, drug_code: str) -> bool:
        return os.path.isfile(_artifact_path(drug_code, 0.50))

    def shap_artifact_path(self, drug_code: str) -> str:
        return _shap_path(drug_code)
