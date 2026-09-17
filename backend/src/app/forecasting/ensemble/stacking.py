"""Ridge stacking meta-learner for model blending."""

from __future__ import annotations

import logging
import os
import pickle
from typing import Optional

import numpy as np
from sklearn.linear_model import Ridge

from app.forecasting.constants import (
    ENSEMBLE_ALPHA,
    ENSEMBLE_MAE_WEIGHT_POWER,
    ENSEMBLE_MIN_MODEL_WEIGHT,
    GLOBAL_DEMAND_SEGMENT,
    STACKING_DISAGREEMENT_CAP_RATIO,
)
from app.forecasting.ensemble.segment_artifacts import (
    resolve_stacking_path,
    stacking_artifact_path,
)
from app.forecasting.model_adaptation import build_stacking_meta_features
from app.forecasting.prediction_bounds import winsorize_predictions
from app.forecasting.schemas import ModelWeightBreakdown

logger = logging.getLogger(__name__)


def _stacking_path(segment: str = GLOBAL_DEMAND_SEGMENT) -> str:
    return stacking_artifact_path(segment)


class StackingMetaLearner:
    """
    Ridge regression trained on held-out validation fold.

    Base inputs: P50 predictions from SARIMA, LightGBM, and Classical.
    Meta inputs: demand segment, history length, recent CV², classical availability,
    and horizon step — so Ridge can learn conditional blending.

    When model disagreement exceeds ``STACKING_DISAGREEMENT_CAP_RATIO`` of the
    prediction cap, fall back to the highest inverse-MAE weight model.
    """

    def __init__(self, segment: str = GLOBAL_DEMAND_SEGMENT) -> None:
        self._segment = segment
        self._model = Ridge(alpha=ENSEMBLE_ALPHA)
        self._column_means = np.zeros(3, dtype=float)
        self._blend_weights = ModelWeightBreakdown(sarima=1 / 3, lgbm=1 / 3, classical=1 / 3)
        self._spread_fallback_count = 0
        self._predict_calls = 0

    @property
    def spread_fallback_rate(self) -> float:
        if self._predict_calls == 0:
            return 0.0
        return self._spread_fallback_count / self._predict_calls

    def _prepare_base_features(
        self,
        sarima_preds: np.ndarray,
        lgbm_preds: np.ndarray,
        classical_preds: np.ndarray,
        *,
        fit: bool = False,
    ) -> np.ndarray:
        sarima = np.asarray(sarima_preds, dtype=float).reshape(-1)
        lgbm = np.asarray(lgbm_preds, dtype=float).reshape(-1)
        classical = np.asarray(classical_preds, dtype=float).reshape(-1)
        features = np.column_stack([sarima, lgbm, classical])

        if fit:
            means = np.nanmean(features, axis=0)
            self._column_means = np.where(np.isnan(means), 0.0, means)

        for idx in range(features.shape[1]):
            nan_mask = np.isnan(features[:, idx])
            if nan_mask.any():
                features[nan_mask, idx] = self._column_means[idx]

        return features

    def _full_features(
        self,
        sarima_preds: np.ndarray,
        lgbm_preds: np.ndarray,
        classical_preds: np.ndarray,
        *,
        demand_segment: str,
        history_days: int | np.ndarray,
        recent_cv2: float | np.ndarray,
        classical_available: bool,
        horizon_steps: Optional[np.ndarray],
        fit: bool = False,
    ) -> np.ndarray:
        base = self._prepare_base_features(
            sarima_preds,
            lgbm_preds,
            classical_preds,
            fit=fit,
        )
        meta = build_stacking_meta_features(
            base.shape[0],
            demand_segment=demand_segment,
            history_days=history_days,
            recent_cv2=recent_cv2,
            classical_available=classical_available,
            horizon_steps=horizon_steps,
        )
        return np.hstack([base, meta])

    def _inverse_mae_weights(
        self,
        sarima_preds: np.ndarray,
        lgbm_preds: np.ndarray,
        classical_preds: np.ndarray,
        actuals: np.ndarray,
        *,
        classical_unavailable: bool,
    ) -> ModelWeightBreakdown:
        actuals = np.asarray(actuals, dtype=float)
        raw_weights: dict[str, float] = {}
        for name, preds in (
            ("sarima", sarima_preds),
            ("lgbm", lgbm_preds),
            ("classical", classical_preds),
        ):
            if name == "classical" and classical_unavailable:
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
            raw_weights[name] = 1.0 / max(mae, 1e-6) ** ENSEMBLE_MAE_WEIGHT_POWER

        active_names = [name for name, weight in raw_weights.items() if weight > 0.0]
        if not active_names:
            if classical_unavailable:
                return ModelWeightBreakdown(sarima=0.5, lgbm=0.5, classical=0.0)
            return ModelWeightBreakdown(sarima=1 / 3, lgbm=1 / 3, classical=1 / 3)

        floor = ENSEMBLE_MIN_MODEL_WEIGHT
        n_active = len(active_names)
        if n_active * floor >= 1.0:
            equal_weight = 1.0 / n_active
            return ModelWeightBreakdown(
                sarima=equal_weight if "sarima" in active_names else 0.0,
                lgbm=equal_weight if "lgbm" in active_names else 0.0,
                classical=equal_weight if "classical" in active_names else 0.0,
            )

        raw_total = sum(raw_weights[name] for name in active_names)
        remainder = 1.0 - (n_active * floor)
        return ModelWeightBreakdown(
            sarima=(
                floor + remainder * (raw_weights["sarima"] / raw_total)
                if "sarima" in active_names
                else 0.0
            ),
            lgbm=(
                floor + remainder * (raw_weights["lgbm"] / raw_total)
                if "lgbm" in active_names
                else 0.0
            ),
            classical=(
                floor + remainder * (raw_weights["classical"] / raw_total)
                if "classical" in active_names
                else 0.0
            ),
        )

    def _robust_blend(
        self,
        base_features: np.ndarray,
        ridge_predictions: np.ndarray,
        weights: ModelWeightBreakdown,
        *,
        prediction_cap: Optional[float] = None,
    ) -> np.ndarray:
        weight_arr = np.array(
            [weights.sarima, weights.lgbm, weights.classical],
            dtype=float,
        )
        if weight_arr.sum() > 0:
            weight_arr = weight_arr / weight_arr.sum()

        blended = np.zeros(base_features.shape[0], dtype=float)
        cap = prediction_cap or 50.0
        disagreement_threshold = cap * STACKING_DISAGREEMENT_CAP_RATIO

        for row_idx in range(base_features.shape[0]):
            row = base_features[row_idx, :3]
            valid = ~np.isnan(row)
            if not valid.any():
                blended[row_idx] = max(float(ridge_predictions[row_idx]), 0.0)
                continue

            values = row[valid]
            active_weights = weight_arr[valid]
            if active_weights.sum() > 0:
                active_weights = active_weights / active_weights.sum()

            spread = float(np.max(values) - np.min(values))
            if spread > disagreement_threshold:
                best_local = int(np.argmax(active_weights))
                blended[row_idx] = float(values[best_local])
                self._spread_fallback_count += 1
            else:
                blended[row_idx] = max(float(ridge_predictions[row_idx]), 0.0)

        self._predict_calls += base_features.shape[0]
        return np.clip(blended, 0.0, None)

    def fit(
        self,
        sarima_preds: np.ndarray,
        lgbm_preds: np.ndarray,
        classical_preds: np.ndarray,
        actuals: np.ndarray,
        *,
        demand_segment: str = GLOBAL_DEMAND_SEGMENT,
        history_days: int | np.ndarray = 365,
        recent_cv2: float | np.ndarray = 0.0,
        horizon_steps: Optional[np.ndarray] = None,
    ) -> None:
        classical_arr = np.asarray(classical_preds, dtype=float)
        classical_unavailable = np.all(np.isnan(classical_arr))
        classical_available = not classical_unavailable

        features = self._full_features(
            sarima_preds,
            lgbm_preds,
            classical_preds,
            demand_segment=demand_segment,
            history_days=history_days,
            recent_cv2=recent_cv2,
            classical_available=classical_available,
            horizon_steps=horizon_steps,
            fit=True,
        )
        actual_arr = np.asarray(actuals, dtype=float)
        self._model.fit(features, actual_arr)
        self._blend_weights = self._inverse_mae_weights(
            sarima_preds,
            lgbm_preds,
            classical_preds,
            actual_arr,
            classical_unavailable=classical_unavailable,
        )
        logger.info(
            "Stacking meta-learner fitted on %d samples (segment=%s, meta=%s)",
            len(actuals),
            self._segment,
            demand_segment,
        )

    def predict(
        self,
        sarima_preds: np.ndarray,
        lgbm_preds: np.ndarray,
        classical_preds: np.ndarray,
        *,
        prediction_cap: Optional[float] = None,
        demand_segment: str = GLOBAL_DEMAND_SEGMENT,
        history_days: int | np.ndarray = 365,
        recent_cv2: float | np.ndarray = 0.0,
        classical_available: Optional[bool] = None,
        horizon_steps: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, ModelWeightBreakdown]:
        sarima = np.asarray(sarima_preds, dtype=float).reshape(-1)
        lgbm = np.asarray(lgbm_preds, dtype=float).reshape(-1)
        classical = np.asarray(classical_preds, dtype=float).reshape(-1)
        if prediction_cap is not None:
            sarima = winsorize_predictions(sarima, prediction_cap)
            lgbm = winsorize_predictions(lgbm, prediction_cap)
            classical = winsorize_predictions(classical, prediction_cap)

        if classical_available is None:
            classical_available = not np.all(np.isnan(classical))

        features = self._full_features(
            sarima,
            lgbm,
            classical,
            demand_segment=demand_segment,
            history_days=history_days,
            recent_cv2=recent_cv2,
            classical_available=classical_available,
            horizon_steps=horizon_steps,
        )
        base_features = features[:, :3]
        ridge_pred = np.clip(self._model.predict(features), 0.0, None)
        blended = self._robust_blend(
            base_features,
            ridge_pred,
            self._blend_weights,
            prediction_cap=prediction_cap,
        )
        if self._predict_calls and self._spread_fallback_count:
            logger.debug(
                "Stacking spread-fallback rate (segment=%s): %.1f%%",
                self._segment,
                100.0 * self.spread_fallback_rate,
            )
        return blended, self._blend_weights

    def save(self) -> str:
        path = _stacking_path(self._segment)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(
                {
                    "model": self._model,
                    "column_means": self._column_means,
                    "blend_weights": self._blend_weights,
                    "segment": self._segment,
                    "third_model": "classical",
                },
                handle,
            )
        return path

    def load(self) -> None:
        path = resolve_stacking_path(self._segment)
        if path is None:
            raise FileNotFoundError(
                f"No stacking artifact found for segment '{self._segment}' "
                f"(checked segment, global, and legacy paths)."
            )
        with open(path, "rb") as handle:
            payload = pickle.load(handle)

        third_model = payload.get("third_model")
        if third_model != "classical":
            if third_model in (None, "tft"):
                raise ValueError(
                    "Stacking artifact uses deprecated TFT slot or lacks third_model tag. "
                    "Retrain forecasting models to regenerate stacking artifacts with classical."
                )
            raise ValueError(
                f"Unsupported third_model '{third_model}' in stacking artifact. "
                "Retrain forecasting models to regenerate stacking artifacts."
            )

        self._model = payload["model"]
        self._column_means = payload.get(
            "column_means",
            np.array(
                [
                    payload.get(
                        "classical_impute_mean",
                        payload.get("tft_impute_mean", 0.0),
                    )
                ]
                * 3,
                dtype=float,
            ),
        )
        blend_weights = payload.get(
            "blend_weights",
            ModelWeightBreakdown(sarima=1 / 3, lgbm=1 / 3, classical=1 / 3),
        )
        if hasattr(blend_weights, "tft"):
            blend_weights = ModelWeightBreakdown(
                sarima=blend_weights.sarima,
                lgbm=blend_weights.lgbm,
                classical=blend_weights.tft,
            )
        self._blend_weights = blend_weights
        self._segment = payload.get("segment", self._segment)
