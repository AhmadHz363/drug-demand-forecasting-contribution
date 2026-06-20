"""MAPIE conformal calibration for ensemble forecasts."""

from __future__ import annotations

import logging
import os
import pickle

import numpy as np

from app.forecasting.constants import ARTIFACTS_DIR, CONFORMAL_COVERAGE

logger = logging.getLogger(__name__)

CONFORMAL_UPPER_SKEW_FACTOR = 1.5


def _mapie_path() -> str:
    return os.path.join(ARTIFACTS_DIR, "conformal", "mapie_wrapper.pkl")


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
    Uses CrossConformalRegressor with method=\"plus\" for a target 90% coverage band.
    """

    def __init__(self) -> None:
        self._mapie = None
        self._estimator = None
        self.scale_factor = 1.0

    def calibrate_scale(
        self,
        stacked_predictions: np.ndarray,
        actuals: np.ndarray,
        p10: np.ndarray,
        p90: np.ndarray,
        *,
        target_coverage: float = CONFORMAL_COVERAGE,
    ) -> None:
        """Compute coverage-correcting scale factor from normalized residuals."""
        p50 = np.asarray(stacked_predictions, dtype=float).reshape(-1)
        actual_arr = np.asarray(actuals, dtype=float).reshape(-1)
        lower = np.asarray(p10, dtype=float).reshape(-1)
        upper = np.asarray(p90, dtype=float).reshape(-1)
        residuals: list[float] = []
        for actual, pred, lo, hi in zip(actual_arr, p50, lower, upper):
            half_width = (hi - lo) / 2.0
            if half_width > 0:
                residuals.append(abs(actual - pred) / half_width)
        if residuals:
            self.scale_factor = float(np.quantile(residuals, target_coverage))
        else:
            self.scale_factor = 1.0

    def adjust_intervals_asymmetric(
        self,
        p10: np.ndarray,
        p50: np.ndarray,
        p90: np.ndarray,
        *,
        skewness_factor: float = CONFORMAL_UPPER_SKEW_FACTOR,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Expand upper interval more than lower for right-skewed demand."""
        p50_arr = np.clip(np.asarray(p50, dtype=float), 0.0, None)
        p10_arr = np.clip(np.asarray(p10, dtype=float), 0.0, None)
        p90_arr = np.clip(np.asarray(p90, dtype=float), 0.0, None)
        lower_hw = (p50_arr - p10_arr) * self.scale_factor
        upper_hw = (p90_arr - p50_arr) * self.scale_factor * skewness_factor
        lower = np.clip(p50_arr - lower_hw, 0.0, None)
        upper = np.clip(p50_arr + upper_hw, 0.0, None)
        lower = np.minimum(lower, p50_arr)
        upper = np.maximum(upper, p50_arr)
        return p50_arr, lower, upper

    def fit(self, stacked_predictions: np.ndarray, actuals: np.ndarray) -> None:
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
        self.calibrate_scale(point, y, lower, upper)
        logger.info(
            "Conformal calibrator fitted on %d samples at %.0f%% coverage (scale=%.3f)",
            len(y),
            CONFORMAL_COVERAGE * 100,
            self.scale_factor,
        )

    def predict_interval(
        self,
        stacked_predictions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._mapie is None:
            raise RuntimeError("ConformalCalibrator is not fitted. Call fit() or load() first.")

        x = np.asarray(stacked_predictions, dtype=float).reshape(-1, 1)
        point, intervals = self._mapie.predict_interval(x)
        lower = np.clip(intervals[:, 0, 0], 0.0, None)
        upper = np.clip(intervals[:, 1, 0], 0.0, None)
        point = np.clip(point, 0.0, None)
        return self.adjust_intervals_asymmetric(lower, point, upper)

    def save(self) -> str:
        if self._mapie is None:
            raise RuntimeError("Cannot save unfitted ConformalCalibrator")
        path = _mapie_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(
                {
                    "mapie": self._mapie,
                    "estimator": self._estimator,
                    "scale_factor": self.scale_factor,
                },
                handle,
            )
        return path

    def load(self) -> None:
        with open(_mapie_path(), "rb") as handle:
            payload = pickle.load(handle)
        self._mapie = payload["mapie"]
        self._estimator = payload["estimator"]
        self.scale_factor = float(payload.get("scale_factor", 1.0))
