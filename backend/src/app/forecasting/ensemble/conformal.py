"""MAPIE conformal calibration for ensemble forecasts."""

from __future__ import annotations

import logging
import os
import pickle
from typing import Optional

import numpy as np

from app.forecasting.constants import (
    CONFORMAL_COVERAGE,
    CONFORMAL_MIN_STEP_RESIDUAL_SAMPLES,
    CONFORMAL_MIN_STEP_SCALE_RATIO,
    GLOBAL_DEMAND_SEGMENT,
)
from app.forecasting.ensemble.segment_artifacts import (
    conformal_artifact_path,
    resolve_conformal_path,
)

logger = logging.getLogger(__name__)

CONFORMAL_UPPER_SKEW_FACTOR = 1.5
# When MAPIE inference lower half-width collapses, fall back to calibration offsets.
CONFORMAL_LOWER_HW_MIN_UPPER_RATIO = 0.15


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
        self._lower_hw_baseline = 0.0
        self._upper_hw_baseline = 0.0
        self._step_lower_hw: dict[int, float] = {}
        self._step_upper_hw: dict[int, float] = {}

    def _scale_for_step(self, step: int) -> float:
        step_scale = self._step_scale_factors.get(step)
        if step_scale is None:
            return self.scale_factor
        floor = self.scale_factor * CONFORMAL_MIN_STEP_SCALE_RATIO
        return max(float(step_scale), floor)

    def _finalize_step_scale_factors(self) -> None:
        """Prevent a few horizon steps from collapsing interval width at inference."""
        if not self._step_scale_factors:
            return
        floor = self.scale_factor * CONFORMAL_MIN_STEP_SCALE_RATIO
        for step in list(self._step_scale_factors):
            self._step_scale_factors[step] = max(
                float(self._step_scale_factors[step]),
                floor,
            )

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
                    if len(step_residuals) >= CONFORMAL_MIN_STEP_RESIDUAL_SAMPLES:
                        self._step_scale_factors[int(step)] = float(
                            np.quantile(step_residuals, target_coverage)
                        )
                    else:
                        self._step_scale_factors[int(step)] = self.scale_factor
                    logger.debug(
                        "Conformal step %d calibration: n=%d scale=%.4f (global=%.4f)",
                        step,
                        len(step_residuals),
                        self._step_scale_factors[int(step)],
                        self.scale_factor,
                    )
        self._finalize_step_scale_factors()

    def _record_calibration_half_widths(
        self,
        point: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        *,
        horizon_steps: Optional[np.ndarray] = None,
    ) -> None:
        """Persist asymmetric residual offsets from the calibration set."""
        point_arr = np.clip(np.asarray(point, dtype=float).reshape(-1), 0.0, None)
        lower_arr = np.clip(np.asarray(lower, dtype=float).reshape(-1), 0.0, None)
        upper_arr = np.clip(np.asarray(upper, dtype=float).reshape(-1), 0.0, None)

        lower_hw = np.maximum(point_arr - lower_arr, 0.0)
        upper_hw = np.maximum(upper_arr - point_arr, 0.0)
        if lower_hw.size:
            self._lower_hw_baseline = float(np.quantile(lower_hw, CONFORMAL_COVERAGE))
            self._upper_hw_baseline = float(np.quantile(upper_hw, CONFORMAL_COVERAGE))
        else:
            self._lower_hw_baseline = 0.0
            self._upper_hw_baseline = 0.0

        self._step_lower_hw = {}
        self._step_upper_hw = {}
        if horizon_steps is not None:
            steps = np.asarray(horizon_steps, dtype=int).reshape(-1)
            if steps.size == point_arr.size:
                for step in sorted(set(steps.tolist())):
                    mask = steps == step
                    step_lower = lower_hw[mask]
                    step_upper = upper_hw[mask]
                    if step_lower.size:
                        self._step_lower_hw[int(step)] = float(
                            np.quantile(step_lower, CONFORMAL_COVERAGE)
                        )
                    if step_upper.size:
                        self._step_upper_hw[int(step)] = float(
                            np.quantile(step_upper, CONFORMAL_COVERAGE)
                        )
        self._finalize_step_half_width_baselines()

    def _finalize_step_half_width_baselines(self) -> None:
        """Back-fill missing or near-zero per-step offsets from the global baseline."""
        min_hw = max(
            self._lower_hw_baseline,
            self._upper_hw_baseline / CONFORMAL_UPPER_SKEW_FACTOR,
            1e-6,
        )
        for step in range(1, 31):
            lower = self._step_lower_hw.get(step)
            upper = self._step_upper_hw.get(step)
            if lower is None or lower < min_hw:
                self._step_lower_hw[step] = max(self._lower_hw_baseline, min_hw)
            if upper is None or upper < min_hw * CONFORMAL_UPPER_SKEW_FACTOR:
                self._step_upper_hw[step] = max(
                    self._upper_hw_baseline,
                    min_hw * CONFORMAL_UPPER_SKEW_FACTOR,
                )

    def _baseline_lower_hw(
        self,
        size: int,
        *,
        horizon_steps: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if horizon_steps is None:
            return np.full(size, self._lower_hw_baseline, dtype=float)
        steps = np.asarray(horizon_steps, dtype=int).reshape(-1)
        return np.asarray(
            [self._step_lower_hw.get(int(step), self._lower_hw_baseline) for step in steps],
            dtype=float,
        )

    def _baseline_upper_hw(
        self,
        size: int,
        *,
        horizon_steps: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if horizon_steps is None:
            return np.full(size, self._upper_hw_baseline, dtype=float)
        steps = np.asarray(horizon_steps, dtype=int).reshape(-1)
        return np.asarray(
            [self._step_upper_hw.get(int(step), self._upper_hw_baseline) for step in steps],
            dtype=float,
        )

    def _enforce_minimum_half_widths(
        self,
        lower_hw: np.ndarray,
        upper_hw: np.ndarray,
        *,
        horizon_steps: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Floor each horizon step at calibration-derived residual offsets."""
        baseline_lower = self._baseline_lower_hw(lower_hw.size, horizon_steps=horizon_steps)
        baseline_upper = self._baseline_upper_hw(upper_hw.size, horizon_steps=horizon_steps)
        return (
            np.maximum(lower_hw, baseline_lower),
            np.maximum(upper_hw, baseline_upper),
        )

    def _repair_degenerate_lower_half_width(
        self,
        lower_hw: np.ndarray,
        upper_hw: np.ndarray,
        *,
        horizon_steps: Optional[np.ndarray] = None,
        skewness_factor: float = CONFORMAL_UPPER_SKEW_FACTOR,
    ) -> np.ndarray:
        """
        MAPIE can return a zero lower offset at inference when calibration
        residuals were one-sided. Fall back to stored calibration offsets.
        """
        repaired = lower_hw.copy()
        degenerate = (repaired <= 1e-9) | (
            repaired < CONFORMAL_LOWER_HW_MIN_UPPER_RATIO * np.maximum(upper_hw, 1e-9)
        )
        if not degenerate.any():
            return repaired

        baseline = self._baseline_lower_hw(repaired.size, horizon_steps=horizon_steps)
        symmetric = upper_hw / max(skewness_factor, 1e-6)
        fallback = np.maximum(baseline, symmetric)
        return np.where(degenerate, fallback, repaired)

    def adjust_intervals_asymmetric(
        self,
        p10: np.ndarray,
        p50: np.ndarray,
        p90: np.ndarray,
        *,
        anchor_p50: Optional[np.ndarray] = None,
        skewness_factor: float = CONFORMAL_UPPER_SKEW_FACTOR,
        horizon_steps: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Expand upper interval more than lower for right-skewed demand.

        ``p10``/``p50``/``p90`` are MAPIE lower bound, point, and upper bound.
        When ``anchor_p50`` is supplied, intervals are re-centered on the
        champion point forecast instead of MAPIE's internal point estimate.
        """
        width_center = np.clip(np.asarray(p50, dtype=float), 0.0, None)
        mapie_lower = np.clip(np.asarray(p10, dtype=float), 0.0, None)
        mapie_upper = np.clip(np.asarray(p90, dtype=float), 0.0, None)
        center = (
            np.clip(np.asarray(anchor_p50, dtype=float), 0.0, None)
            if anchor_p50 is not None
            else width_center
        )

        if horizon_steps is None:
            lower_hw = (width_center - mapie_lower) * self.scale_factor
            upper_hw = (mapie_upper - width_center) * self.scale_factor * skewness_factor
        else:
            steps = np.asarray(horizon_steps, dtype=int).reshape(-1)
            if steps.size != center.size:
                raise ValueError("horizon_steps length must match prediction length")
            lower_hw = np.zeros_like(center)
            upper_hw = np.zeros_like(center)
            for idx, step in enumerate(steps):
                scale = self._scale_for_step(int(step))
                lower_hw[idx] = (width_center[idx] - mapie_lower[idx]) * scale
                upper_hw[idx] = (mapie_upper[idx] - width_center[idx]) * scale * skewness_factor

        lower_hw = self._repair_degenerate_lower_half_width(
            lower_hw,
            upper_hw,
            horizon_steps=horizon_steps,
            skewness_factor=skewness_factor,
        )
        lower_hw, upper_hw = self._enforce_minimum_half_widths(
            lower_hw,
            upper_hw,
            horizon_steps=horizon_steps,
        )

        lower = np.clip(center - lower_hw, 0.0, None)
        upper = np.clip(center + upper_hw, 0.0, None)
        lower = np.minimum(lower, center)
        upper = np.maximum(upper, center)
        return center, lower, upper

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
        self._record_calibration_half_widths(
            point,
            lower,
            upper,
            horizon_steps=horizon_steps,
        )
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
        champion_predictions: np.ndarray,
        *,
        horizon_steps: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return champion-anchored P50 with MAPIE-calibrated interval widths.

        MAPIE supplies residual offsets only; the point forecast always comes
        from ``champion_predictions`` (stacked ensemble or champion base model).
        """
        if self._mapie is None:
            raise RuntimeError("ConformalCalibrator is not fitted. Call fit() or load() first.")

        champion = np.clip(
            np.asarray(champion_predictions, dtype=float).reshape(-1),
            0.0,
            None,
        )
        x = champion.reshape(-1, 1)
        mapie_point, intervals = self._mapie.predict_interval(x)
        mapie_lower = np.clip(intervals[:, 0, 0], 0.0, None)
        mapie_upper = np.clip(intervals[:, 1, 0], 0.0, None)
        mapie_point = np.clip(mapie_point.reshape(-1), 0.0, None)
        return self.adjust_intervals_asymmetric(
            mapie_lower,
            mapie_point,
            mapie_upper,
            anchor_p50=champion,
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
                    "lower_hw_baseline": self._lower_hw_baseline,
                    "upper_hw_baseline": self._upper_hw_baseline,
                    "step_lower_hw": self._step_lower_hw,
                    "step_upper_hw": self._step_upper_hw,
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
        self._lower_hw_baseline = float(payload.get("lower_hw_baseline", 0.0))
        self._upper_hw_baseline = float(payload.get("upper_hw_baseline", 0.0))
        self._step_lower_hw = {
            int(k): float(v) for k, v in payload.get("step_lower_hw", {}).items()
        }
        self._step_upper_hw = {
            int(k): float(v) for k, v in payload.get("step_upper_hw", {}).items()
        }
        self._segment = payload.get("segment", self._segment)
        self._finalize_step_scale_factors()
        self._finalize_step_half_width_baselines()
