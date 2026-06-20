"""Ridge stacking meta-learner for model blending."""

from __future__ import annotations

import logging
import os
import pickle
from typing import Optional

import numpy as np
from sklearn.linear_model import Ridge

from app.forecasting.constants import (
    ARTIFACTS_DIR,
    ENSEMBLE_ALPHA,
    ENSEMBLE_MAE_OUTLIER_RATIO,
)
from app.forecasting.prediction_bounds import winsorize_predictions
from app.forecasting.schemas import ModelWeightBreakdown

logger = logging.getLogger(__name__)


def _stacking_path() -> str:
    return os.path.join(ARTIFACTS_DIR, "stacking", "stacking_meta.pkl")


class StackingMetaLearner:
    """
    Ridge regression trained on held-out validation fold.
    Input features: P50 predictions from SARIMA, LightGBM, and TFT.
    Target: actual demand.
    Produces: blended P50 forecast + per-drug model weights.

    Blending uses inverse-MAE weights from validation folds (non-negative).
    Ridge is still fitted for diagnostics and artifact compatibility.
    """

    def __init__(self) -> None:
        self._model = Ridge(alpha=ENSEMBLE_ALPHA)
        self._column_means = np.zeros(3, dtype=float)
        self._blend_weights = ModelWeightBreakdown(sarima=1 / 3, lgbm=1 / 3, tft=1 / 3)

    def _prepare_features(
        self,
        sarima_preds: np.ndarray,
        lgbm_preds: np.ndarray,
        tft_preds: np.ndarray,
        *,
        fit: bool = False,
    ) -> np.ndarray:
        sarima = np.asarray(sarima_preds, dtype=float).reshape(-1)
        lgbm = np.asarray(lgbm_preds, dtype=float).reshape(-1)
        tft = np.asarray(tft_preds, dtype=float).reshape(-1)
        features = np.column_stack([sarima, lgbm, tft])

        if fit:
            means = np.nanmean(features, axis=0)
            self._column_means = np.where(np.isnan(means), 0.0, means)

        for idx in range(features.shape[1]):
            nan_mask = np.isnan(features[:, idx])
            if nan_mask.any():
                features[nan_mask, idx] = self._column_means[idx]

        return features

    def _inverse_mae_weights(
        self,
        sarima_preds: np.ndarray,
        lgbm_preds: np.ndarray,
        tft_preds: np.ndarray,
        actuals: np.ndarray,
        *,
        tft_unavailable: bool,
    ) -> ModelWeightBreakdown:
        actuals = np.asarray(actuals, dtype=float)
        raw_weights: dict[str, float] = {}
        maes: dict[str, float] = {}
        for name, preds in (
            ("sarima", sarima_preds),
            ("lgbm", lgbm_preds),
            ("tft", tft_preds),
        ):
            if name == "tft" and tft_unavailable:
                raw_weights[name] = 0.0
                continue
            values = np.asarray(preds, dtype=float)
            valid = ~np.isnan(values)
            if not valid.any():
                raw_weights[name] = 0.0
                continue
            paired = [
                (float(act), float(pred))
                for act, pred in zip(actuals[valid], values[valid])
            ]
            mae = float(np.mean([abs(act - pred) for act, pred in paired]))
            maes[name] = mae
            raw_weights[name] = 1.0 / max(mae, 1e-6)

        if maes:
            best_mae = min(maes.values())
            for name in list(raw_weights.keys()):
                if maes.get(name, float("inf")) > best_mae * ENSEMBLE_MAE_OUTLIER_RATIO:
                    raw_weights[name] = 0.0

        total = sum(raw_weights.values())
        if total == 0.0:
            if tft_unavailable:
                return ModelWeightBreakdown(sarima=0.5, lgbm=0.5, tft=0.0)
            return ModelWeightBreakdown(sarima=1 / 3, lgbm=1 / 3, tft=1 / 3)

        return ModelWeightBreakdown(
            sarima=float(raw_weights["sarima"] / total),
            lgbm=float(raw_weights["lgbm"] / total),
            tft=float(raw_weights["tft"] / total),
        )

    def _robust_blend(
        self,
        features: np.ndarray,
        weights: ModelWeightBreakdown,
        *,
        prediction_cap: Optional[float] = None,
    ) -> np.ndarray:
        weight_arr = np.array([weights.sarima, weights.lgbm, weights.tft], dtype=float)
        if weight_arr.sum() > 0:
            weight_arr = weight_arr / weight_arr.sum()

        blended = np.zeros(features.shape[0], dtype=float)
        disagreement_threshold = (prediction_cap or 50.0) * 0.5

        for row_idx in range(features.shape[0]):
            row = features[row_idx]
            valid = ~np.isnan(row)
            if not valid.any():
                blended[row_idx] = 0.0
                continue

            values = row[valid]
            active_weights = weight_arr[valid]
            if active_weights.sum() > 0:
                active_weights = active_weights / active_weights.sum()

            spread = float(np.max(values) - np.min(values))
            if spread > disagreement_threshold:
                best_local = int(np.argmax(active_weights))
                blended[row_idx] = float(values[best_local])
            else:
                blended[row_idx] = float(np.dot(values, active_weights))

        return np.clip(blended, 0.0, None)

    def fit(
        self,
        sarima_preds: np.ndarray,
        lgbm_preds: np.ndarray,
        tft_preds: np.ndarray,
        actuals: np.ndarray,
    ) -> None:
        features = self._prepare_features(
            sarima_preds,
            lgbm_preds,
            tft_preds,
            fit=True,
        )
        actual_arr = np.asarray(actuals, dtype=float)
        self._model.fit(features, actual_arr)
        tft_arr = np.asarray(tft_preds, dtype=float)
        tft_unavailable = np.all(np.isnan(tft_arr))
        self._blend_weights = self._inverse_mae_weights(
            sarima_preds,
            lgbm_preds,
            tft_preds,
            actual_arr,
            tft_unavailable=tft_unavailable,
        )
        logger.info("Stacking meta-learner fitted on %d samples", len(actuals))

    def predict(
        self,
        sarima_preds: np.ndarray,
        lgbm_preds: np.ndarray,
        tft_preds: np.ndarray,
        *,
        prediction_cap: Optional[float] = None,
    ) -> tuple[np.ndarray, ModelWeightBreakdown]:
        sarima = np.asarray(sarima_preds, dtype=float).reshape(-1)
        lgbm = np.asarray(lgbm_preds, dtype=float).reshape(-1)
        tft = np.asarray(tft_preds, dtype=float).reshape(-1)
        if prediction_cap is not None:
            sarima = winsorize_predictions(sarima, prediction_cap)
            lgbm = winsorize_predictions(lgbm, prediction_cap)
            tft = winsorize_predictions(tft, prediction_cap)

        features = self._prepare_features(sarima, lgbm, tft)
        blended = self._robust_blend(
            features,
            self._blend_weights,
            prediction_cap=prediction_cap,
        )
        return blended, self._blend_weights

    def save(self) -> str:
        path = _stacking_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(
                {
                    "model": self._model,
                    "column_means": self._column_means,
                    "blend_weights": self._blend_weights,
                },
                handle,
            )
        return path

    def load(self) -> None:
        with open(_stacking_path(), "rb") as handle:
            payload = pickle.load(handle)
        self._model = payload["model"]
        self._column_means = payload.get(
            "column_means",
            np.array([payload.get("tft_impute_mean", 0.0)] * 3, dtype=float),
        )
        self._blend_weights = payload.get(
            "blend_weights",
            ModelWeightBreakdown(sarima=1 / 3, lgbm=1 / 3, tft=1 / 3),
        )
