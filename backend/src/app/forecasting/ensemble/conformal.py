"""MAPIE conformal calibration for ensemble forecasts."""

from __future__ import annotations

import logging
import os
import pickle
from typing import Optional

import numpy as np

from app.forecasting.constants import CONFORMAL_COVERAGE, GLOBAL_DEMAND_SEGMENT
from app.forecasting.ensemble.segment_artifacts import (
    conformal_artifact_path,
    resolve_conformal_path,
)

logger = logging.getLogger(__name__)

CONFORMAL_UPPER_SKEW_FACTOR = 1.5


def _mapie_path(segment: str = GLOBAL_DEMAND_SEGMENT) -> str:
    return conformal_artifact_path(segment)


def extend_tail_quantiles(
    p50: np.ndarray,
    p10: np.ndarray,
    p90: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extend MAPIE-calibrated P10/P90 to approximate P5/P95.

    P5/P95 are symmetric extrapolations — NOT coverage-guaranteed beyond P10/P90.
    """
    p50 = np.clip(np.asarray(p50, dtype=float), 0.0, None)
    p10 = np.clip(np.asarray(p10, dtype=float), 0.0, None)
    p90 = np.clip(np.asarray(p90, dtype=float), 0.0, None)

    p5 = np.maximum(0.0, p10 - 0.5 * (p50 - p10))
    p95 = p90 + 0.5 * (p90 - p50)
    p5 = np.minimum(p5, p10)
    p95 = np.maximum(p95, p90)
    return p5, p10, p50, p90, p95


def _import_cross_conformal_regressor():
    try:
        from mapie.regression import CrossConformalRegressor
    except ImportError as exc:
        raise ImportError(
            "mapie is required for conformal calibration. "
            "Install it with: pip install 'mapie>=0.8.0' (see requirements.txt)."
        ) from exc
    return CrossConformalRegressor


class ConformalCalibrator:
    """
    Wraps ensemble point forecasts with MAPIE cross-conformal intervals.

    Supports per-horizon scale factors so 1-day-ahead and 7-day-ahead
    residuals are calibrated separately.
    """

    def __init__(self, segment: str = GLOBAL_DEMAND_SEGMENT) -> None:
        self._segment = segment
        self._mapie = None
        self._estimator = None
        self.scale_factor = 1.0
        self._step_scale_factors: dict[int, float] = {}

    def _scale_for_step(self, step: int) -> float:
        return self._step_scale_factors.get(step, self.scale_factor)

    def calibrate_scale(
        self,
        stacked_predictions: np.ndarray,
        actuals: np.ndarray,
        p10: np.ndarray,
        p90: np.ndarray,
        *,
        target_coverage: float = CONFORMAL_COVERAGE,
        horizon_steps: Optional[np.ndarray] = None,
    ) -> None:
        """Compute coverage-correcting scale factor(s) from normalized residuals."""
        p50 = np.asarray(stacked_predictions, dtype=float).reshape(-1)
        actual_arr = np.asarray(actuals, dtype=float).reshape(-1)
        lower = np.asarray(p10, dtype=float).reshape(-1)
        upper = np.asarray(p90, dtype=float).reshape(-1)

        def _residuals(mask: np.ndarray) -> list[float]:
            out: list[float] = []
            for actual, pred, lo, hi in zip(
                actual_arr[mask],
                p50[mask],
                lower[mask],
                upper[mask],
            ):
                half_width = (hi - lo) / 2.0
                if half_width > 0:
                    out.append(abs(actual - pred) / half_width)
            return out

        all_residuals = _residuals(np.ones(len(p50), dtype=bool))
        if all_residuals:
            self.scale_factor = float(np.quantile(all_residuals, target_coverage))
        else:
            self.scale_factor = 1.0

        self._step_scale_factors = {}
        if horizon_steps is not None:
            steps = np.asarray(horizon_steps, dtype=int).reshape(-1)
            if steps.size == len(p50):
                for step in sorted(set(steps.tolist())):
                    mask = steps == step
                    step_residuals = _residuals(mask)
                    if step_residuals:
                        self._step_scale_factors[int(step)] = float(
                            np.quantile(step_residuals, target_coverage)
                        )
                    else:
                        self._step_scale_factors[int(step)] = self.scale_factor

    def adjust_intervals_asymmetric(
        self,
        p10: np.ndarray,
        p50: np.ndarray,
        p90: np.ndarray,
        *,
        skewness_factor: float = CONFORMAL_UPPER_SKEW_FACTOR,
        horizon_steps: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Expand upper interval more than lower for right-skewed demand."""
        p50_arr = np.clip(np.asarray(p50, dtype=float), 0.0, None)
        p10_arr = np.clip(np.asarray(p10, dtype=float), 0.0, None)
        p90_arr = np.clip(np.asarray(p90, dtype=float), 0.0, None)

        if horizon_steps is None:
            lower_hw = (p50_arr - p10_arr) * self.scale_factor
            upper_hw = (p90_arr - p50_arr) * self.scale_factor * skewness_factor
        else:
            steps = np.asarray(horizon_steps, dtype=int).reshape(-1)
            if steps.size != p50_arr.size:
                raise ValueError("horizon_steps length must match prediction length")
            lower_hw = np.zeros_like(p50_arr)
            upper_hw = np.zeros_like(p50_arr)
            for idx, step in enumerate(steps):
                scale = self._scale_for_step(int(step))
                lower_hw[idx] = (p50_arr[idx] - p10_arr[idx]) * scale
                upper_hw[idx] = (p90_arr[idx] - p50_arr[idx]) * scale * skewness_factor

        lower = np.clip(p50_arr - lower_hw, 0.0, None)
        upper = np.clip(p50_arr + upper_hw, 0.0, None)
        lower = np.minimum(lower, p50_arr)
        upper = np.maximum(upper, p50_arr)
        return p50_arr, lower, upper

    def fit(
        self,
        stacked_predictions: np.ndarray,
        actuals: np.ndarray,
        *,
        horizon_steps: Optional[np.ndarray] = None,
    ) -> None:
        CrossConformalRegressor = _import_cross_conformal_regressor()
        from sklearn.linear_model import Ridge

        x = np.asarray(stacked_predictions, dtype=float).reshape(-1, 1)
        y = np.asarray(actuals, dtype=float)
        n_splits = min(5, max(2, len(y) // 20))

        self._estimator = Ridge(alpha=1e-6)
        self._mapie = CrossConformalRegressor(
            estimator=self._estimator,
            confidence_level=CONFORMAL_COVERAGE,
            method="plus",
            cv=n_splits,
        )
        self._mapie.fit_conformalize(x, y)
        point, intervals = self._mapie.predict_interval(x)
        lower = np.clip(intervals[:, 0, 0], 0.0, None)
        upper = np.clip(intervals[:, 1, 0], 0.0, None)
        self.calibrate_scale(point, y, lower, upper, horizon_steps=horizon_steps)
        logger.info(
            "Conformal calibrator fitted on %d samples at %.0f%% coverage "
            "(global_scale=%.3f, per_step=%d, segment=%s)",
            len(y),
            CONFORMAL_COVERAGE * 100,
            self.scale_factor,
            len(self._step_scale_factors),
            self._segment,
        )

    def predict_interval(
        self,
        stacked_predictions: np.ndarray,
        *,
        horizon_steps: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._mapie is None:
            raise RuntimeError("ConformalCalibrator is not fitted. Call fit() or load() first.")

        x = np.asarray(stacked_predictions, dtype=float).reshape(-1, 1)
        point, intervals = self._mapie.predict_interval(x)
        lower = np.clip(intervals[:, 0, 0], 0.0, None)
        upper = np.clip(intervals[:, 1, 0], 0.0, None)
        point = np.clip(point, 0.0, None)
        return self.adjust_intervals_asymmetric(
            lower,
            point,
            upper,
            horizon_steps=horizon_steps,
        )

    def save(self) -> str:
        if self._mapie is None:
            raise RuntimeError("Cannot save unfitted ConformalCalibrator")
        path = _mapie_path(self._segment)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(
                {
                    "mapie": self._mapie,
                    "estimator": self._estimator,
                    "scale_factor": self.scale_factor,
                    "step_scale_factors": self._step_scale_factors,
                    "segment": self._segment,
                },
                handle,
            )
        return path

    def load(self) -> None:
        path = resolve_conformal_path(self._segment)
        if path is None:
            raise FileNotFoundError(
                f"No conformal artifact found for segment '{self._segment}' "
                f"(checked segment, global, and legacy paths)."
            )
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        self._mapie = payload["mapie"]
        self._estimator = payload["estimator"]
        self.scale_factor = float(payload.get("scale_factor", 1.0))
        self._step_scale_factors = {
            int(k): float(v)
            for k, v in payload.get("step_scale_factors", {}).items()
        }
        self._segment = payload.get("segment", self._segment)
