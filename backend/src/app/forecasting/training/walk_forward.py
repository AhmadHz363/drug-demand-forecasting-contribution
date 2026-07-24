"""Walk-forward cross-validation for forecasting models."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from app.forecasting.constants import (
    WALK_FORWARD_N_SPLITS,
    WALK_FORWARD_TEST_HORIZON,
)
from app.forecasting.demand_quantity import as_consumption_demand
from app.forecasting.prediction_bounds import (
    demand_prediction_cap,
    sanitize_ensemble_predictions,
    winsorize_predictions,
)
from app.forecasting.models.base_model import BaseForecastingModel
from app.forecasting.models.classical_model import ClassicalModel
from app.forecasting.models.lgbm_model import LightGBMModel
from app.forecasting.models.sarima_model import SarimaModel

logger = logging.getLogger(__name__)


def _model_registry_name(model: BaseForecastingModel) -> str:
    if isinstance(model, SarimaModel):
        return "sarima"
    if isinstance(model, LightGBMModel):
        return "lgbm"
    if isinstance(model, ClassicalModel):
        return "classical"
    return "classical"


from app.forecasting.evaluation_metrics import (
    accuracy_from_smape,
    accuracy_skill_from_mase,
    finite_mase_or_none,
    mase,
    rolling_validation_metrics,
    smape,
)

__all__ = [
    "WalkForwardResult",
    "accuracy_from_smape",
    "accuracy_skill_from_mase",
    "mase",
    "smape",
    "walk_forward_full",
    "walk_forward_smape",
    "walk_forward_coverage",
    "walk_forward_mase",
    "collect_walk_forward_predictions",
    "measure_lgbm_recursive_drift",
    "demand_prediction_cap",
    "sanitize_ensemble_predictions",
    "winsorize_predictions",
]


class WalkForwardResult:
    """Aggregated metrics and out-of-fold predictions from a single walk-forward pass."""

    __slots__ = (
        "smape",
        "coverage",
        "mase",
        "smape_normal_supply",
        "mase_normal_supply",
        "smape_7day_full",
        "smape_30day_full",
        "mase_7day_full",
        "mase_30day_full",
        "smape_7day_normal",
        "smape_30day_normal",
        "mase_7day_normal",
        "mase_30day_normal",
        "oof_p50",
        "oof_actuals",
        "oof_horizon_steps",
    )

    def __init__(
        self,
        smape: float,
        coverage: float,
        mase: float | None,
        oof_p50: np.ndarray,
        oof_actuals: np.ndarray,
        oof_horizon_steps: np.ndarray,
        *,
        smape_normal_supply: float | None = None,
        mase_normal_supply: float | None = None,
        smape_7day_full: float | None = None,
        smape_30day_full: float | None = None,
        mase_7day_full: float | None = None,
        mase_30day_full: float | None = None,
        smape_7day_normal: float | None = None,
        smape_30day_normal: float | None = None,
        mase_7day_normal: float | None = None,
        mase_30day_normal: float | None = None,
    ) -> None:
        self.smape = smape
        self.coverage = coverage
        self.mase = mase
        self.smape_normal_supply = smape_normal_supply
        self.mase_normal_supply = mase_normal_supply
        self.smape_7day_full = smape_7day_full
        self.smape_30day_full = smape_30day_full
        self.mase_7day_full = mase_7day_full
        self.mase_30day_full = mase_30day_full
        self.smape_7day_normal = smape_7day_normal
        self.smape_30day_normal = smape_30day_normal
        self.mase_7day_normal = mase_7day_normal
        self.mase_30day_normal = mase_30day_normal
        self.oof_p50 = oof_p50
        self.oof_actuals = oof_actuals
        self.oof_horizon_steps = oof_horizon_steps


def walk_forward_full(
    model: BaseForecastingModel,
    df: pd.DataFrame,
    drug_code: str,
    n_splits: int = WALK_FORWARD_N_SPLITS,
    test_horizon: int = WALK_FORWARD_TEST_HORIZON,
) -> WalkForwardResult:
    """
    Single walk-forward pass that simultaneously computes sMAPE, coverage,
    MASE, and collects out-of-fold P50 predictions for ensemble fitting.

    Previously the trainer ran 4 separate walk-forward loops per model per drug
    (smape, coverage, mase, OOF collection), each training ``n_splits`` fold
    models.  This function consolidates all four into one loop, reducing total
    fold trainings from ``4 × n_splits`` to ``n_splits``.
    """
    n = len(df)
    total_test = n_splits * test_horizon
    if n <= total_test:
        raise ValueError(
            f"Need more than {total_test} rows for {n_splits} folds of "
            f"horizon {test_horizon}, got {n}"
        )

    model_class = type(model)
    model_name = _model_registry_name(model)

    smape_scores: list[float] = []
    smape_normal_scores: list[float] = []
    fold_mases: list[float] = []
    fold_mases_normal: list[float] = []
    covered = 0
    total_pts = 0
    oof_p50: list[float] = []
    oof_actuals: list[float] = []
    oof_steps: list[int] = []
    oof_stockout: list[bool] = []

    lgbm_preds_by_step: dict[int, list[float]] = {}
    lgbm_actuals_by_step: dict[int, list[float]] = {}

    for fold in range(n_splits):
        test_start = n - total_test + fold * test_horizon
        test_end = test_start + test_horizon
        train_df = df.iloc[:test_start]
        history_df = df.iloc[:test_end]

        fold_model = model_class()
        train_actuals = as_consumption_demand(
            train_df["total_quantity"].astype(float).values,
        )
        try:
            _train_fold_model(fold_model, model_class, train_df, drug_code)
        except ValueError as exc:
            logger.warning("Walk-forward fold %d skipped for %s: %s", fold, drug_code, exc)
            oof_p50.extend([float("nan")] * test_horizon)
            oof_actuals.extend([0.0] * test_horizon)
            oof_steps.extend(range(1, test_horizon + 1))
            oof_stockout.extend([False] * test_horizon)
            continue

        pred_df = _predict_fold(fold_model, model_name, train_df, history_df, test_horizon)

        fold_actuals = as_consumption_demand(
            df.iloc[test_start:test_end]["total_quantity"].astype(float).values,
        )
        test_slice = df.iloc[test_start:test_end]
        stockout_flags = (
            test_slice["is_stockout"].astype(bool).values
            if "is_stockout" in test_slice.columns
            else np.zeros(len(fold_actuals), dtype=bool)
        )
        oof_actuals.extend(fold_actuals.tolist())
        oof_steps.extend(range(1, test_horizon + 1))
        oof_stockout.extend(stockout_flags.tolist())

        if pred_df is None:
            oof_p50.extend([float("nan")] * test_horizon)
            continue

        predicted = pred_df["p50"].astype(float).values
        lower = pred_df["p10"].astype(float).values
        upper = pred_df["p90"].astype(float).values

        oof_p50.extend(predicted.tolist())

        # sMAPE
        for a, p, is_stockout in zip(fold_actuals, predicted, stockout_flags):
            smape_scores.append(smape(a, p))
            if not is_stockout:
                smape_normal_scores.append(smape(a, p))

        # Coverage (P10–P90)
        covered += int(((fold_actuals >= lower) & (fold_actuals <= upper)).sum())
        total_pts += len(fold_actuals)

        # MASE (seasonal-naive scale from training window)
        try:
            fold_mases.append(
                mase(fold_actuals, predicted, training_actuals=train_actuals)
            )
            if (~stockout_flags).any():
                fold_mases_normal.append(
                    mase(
                        fold_actuals[~stockout_flags],
                        predicted[~stockout_flags],
                        training_actuals=train_actuals,
                    )
                )
        except Exception:
            pass

        # LGBM recursive drift tracking
        if model_name == "lgbm":
            for step_idx, (pv, av) in enumerate(zip(predicted, fold_actuals), start=1):
                lgbm_preds_by_step.setdefault(step_idx, []).append(float(pv))
                lgbm_actuals_by_step.setdefault(step_idx, []).append(float(av))

    if not smape_scores:
        raise ValueError(f"No walk-forward folds completed for {drug_code}")

    if lgbm_preds_by_step:
        drift = measure_lgbm_recursive_drift(lgbm_preds_by_step, lgbm_actuals_by_step)
        if drift:
            step1 = drift.get(1)
            step_last = drift.get(max(drift))
            if step1 is not None and step_last is not None and step1 > 0:
                logger.info(
                    "LGBM recursive drift for %s: step-1 MAE=%.3f, step-%d MAE=%.3f (ratio=%.2f)",
                    drug_code,
                    step1,
                    max(drift),
                    step_last,
                    step_last / step1,
                )

    avg_smape = float(sum(smape_scores) / len(smape_scores))
    avg_smape_normal = (
        float(sum(smape_normal_scores) / len(smape_normal_scores))
        if smape_normal_scores
        else avg_smape
    )
    avg_coverage = float(covered / total_pts) if total_pts > 0 else 0.0
    avg_mase = float(sum(fold_mases) / len(fold_mases)) if fold_mases else None
    avg_mase_normal = (
        float(sum(fold_mases_normal) / len(fold_mases_normal))
        if fold_mases_normal
        else avg_mase
    )

    oof_actual_arr = np.asarray(oof_actuals, dtype=float)
    oof_pred_arr = np.asarray(oof_p50, dtype=float)
    oof_stockout_arr = np.asarray(oof_stockout, dtype=bool)
    prevalidation_end = n - total_test
    prevalidation_actuals = as_consumption_demand(
        df.iloc[:prevalidation_end]["total_quantity"].astype(float).values,
    )
    rolling = rolling_validation_metrics(
        oof_actual_arr,
        oof_pred_arr,
        stockout_flags=oof_stockout_arr,
        training_actuals=prevalidation_actuals,
    )

    return WalkForwardResult(
        smape=avg_smape,
        coverage=avg_coverage,
        mase=finite_mase_or_none(avg_mase) if avg_mase is not None else None,
        oof_p50=oof_pred_arr,
        oof_actuals=oof_actual_arr,
        oof_horizon_steps=np.asarray(oof_steps, dtype=int),
        smape_normal_supply=avg_smape_normal,
        mase_normal_supply=finite_mase_or_none(avg_mase_normal),
        smape_7day_full=rolling.get("smape_7day_full"),
        smape_30day_full=rolling.get("smape_30day_full"),
        mase_7day_full=rolling.get("mase_7day_full"),
        mase_30day_full=rolling.get("mase_30day_full"),
        smape_7day_normal=rolling.get("smape_7day_normal"),
        smape_30day_normal=rolling.get("smape_30day_normal"),
        mase_7day_normal=rolling.get("mase_7day_normal"),
        mase_30day_normal=rolling.get("mase_30day_normal"),
    )


def walk_forward_smape(
    model: BaseForecastingModel,
    df: pd.DataFrame,
    drug_code: str,
    n_splits: int = WALK_FORWARD_N_SPLITS,
    test_horizon: int = WALK_FORWARD_TEST_HORIZON,
) -> float:
    """
    Performs walk-forward validation.

    Prefer :func:`walk_forward_full` when also needing coverage, MASE, or OOF
    predictions — it computes all four in a single fold-training pass.
    """
    return walk_forward_full(model, df, drug_code, n_splits, test_horizon).smape


def walk_forward_mase(
    model: BaseForecastingModel,
    df: pd.DataFrame,
    drug_code: str,
    n_splits: int = WALK_FORWARD_N_SPLITS,
    test_horizon: int = WALK_FORWARD_TEST_HORIZON,
) -> float:
    """Average MASE across walk-forward folds (seasonal-naive scaled)."""
    result = walk_forward_full(model, df, drug_code, n_splits, test_horizon)
    if result.mase is None:
        raise ValueError(f"No MASE computed for {drug_code}")
    return result.mase


def _train_fold_model(
    fold_model: BaseForecastingModel,
    model_class: type[BaseForecastingModel],
    train_df: pd.DataFrame,
    drug_code: str,
) -> None:
    fold_model.train(train_df, drug_code)


def _predict_fold(
    fold_model: BaseForecastingModel,
    model_name: str,
    train_df: pd.DataFrame,
    history_df: pd.DataFrame,
    test_horizon: int,
) -> Optional[pd.DataFrame]:
    if isinstance(fold_model, LightGBMModel):
        future_covariates = history_df.iloc[len(train_df) : len(train_df) + test_horizon]
        return fold_model.predict(
            train_df,
            test_horizon,
            future_covariates=future_covariates,
        )
    return fold_model.predict(train_df, test_horizon)


def walk_forward_coverage(
    model: BaseForecastingModel,
    df: pd.DataFrame,
    drug_code: str,
    n_splits: int = WALK_FORWARD_N_SPLITS,
    test_horizon: int = WALK_FORWARD_TEST_HORIZON,
) -> float:
    """
    Fraction of walk-forward test points falling inside the model P10–P90 band.

    Prefer :func:`walk_forward_full` when also needing sMAPE, MASE, or OOF
    predictions — it computes all four in a single fold-training pass.
    """
    n = len(df)
    total_test = n_splits * test_horizon
    if n <= total_test:
        return 0.0
    return walk_forward_full(model, df, drug_code, n_splits, test_horizon).coverage


def measure_lgbm_recursive_drift(
    predicted_by_step: dict[int, list[float]],
    actuals_by_step: dict[int, list[float]],
) -> dict[int, float]:
    """
    Mean absolute error by forecast horizon step (1-indexed).

    Used to detect compounding error in recursive LightGBM forecasts.
    """
    drift: dict[int, float] = {}
    for step, preds in predicted_by_step.items():
        actuals = actuals_by_step.get(step, [])
        if not preds or not actuals or len(preds) != len(actuals):
            continue
        drift[step] = float(
            np.mean([abs(float(a) - float(p)) for a, p in zip(actuals, preds)])
        )
    return drift


def collect_walk_forward_predictions(
    df: pd.DataFrame,
    drug_code: str,
    model_classes: dict[str, type[BaseForecastingModel]],
    n_splits: int = WALK_FORWARD_N_SPLITS,
    test_horizon: int = WALK_FORWARD_TEST_HORIZON,
    *,
    precomputed_oof: dict[str, WalkForwardResult] | None = None,
) -> Optional[dict[str, np.ndarray]]:
    """
    Collect out-of-fold P50 predictions from walk-forward validation.

    When *precomputed_oof* is provided (a ``{model_name: WalkForwardResult}``
    dict populated by the trainer's :func:`walk_forward_full` calls), the
    function assembles the output from cached results instead of running
    additional fold-training loops.  Pass ``None`` to fall back to the
    original self-contained multi-fold behaviour (used externally or in tests).

    Returns arrays keyed by model name plus ``actuals``, or None when the
    series is too short for the configured fold layout.
    """
    n = len(df)
    total_test = n_splits * test_horizon
    if n <= total_test:
        return None

    # Fast path: assemble from already-computed WalkForwardResult objects.
    if precomputed_oof is not None and precomputed_oof:
        # All results share the same actuals / horizon_steps arrays.
        reference = next(iter(precomputed_oof.values()))
        collected: dict[str, np.ndarray] = {}
        for name in model_classes:
            if name in precomputed_oof:
                collected[name] = precomputed_oof[name].oof_p50
            else:
                collected[name] = np.full(len(reference.oof_actuals), np.nan)
        return {
            **collected,
            "actuals": reference.oof_actuals,
            "horizon_steps": reference.oof_horizon_steps,
        }

    # Slow path: run the full fold-training loop (backward-compatible).
    raw_collected: dict[str, list[float]] = {name: [] for name in model_classes}
    actuals: list[float] = []
    horizon_steps: list[int] = []
    lgbm_preds_by_step: dict[int, list[float]] = {}
    lgbm_actuals_by_step: dict[int, list[float]] = {}

    for fold in range(n_splits):
        test_start = n - total_test + fold * test_horizon
        test_end = test_start + test_horizon
        train_df = df.iloc[:test_start]
        history_df = df.iloc[:test_end]
        fold_actuals = as_consumption_demand(
            df.iloc[test_start:test_end]["total_quantity"].astype(float).values,
        )
        actuals.extend(fold_actuals.tolist())
        horizon_steps.extend(range(1, test_horizon + 1))

        for name, model_class in model_classes.items():
            fold_model = model_class()
            try:
                _train_fold_model(fold_model, model_class, train_df, drug_code)
                pred_df = _predict_fold(fold_model, name, train_df, history_df, test_horizon)
                if pred_df is None:
                    raw_collected[name].extend([float("nan")] * test_horizon)
                    continue
                p50_values = pred_df["p50"].astype(float).tolist()
                raw_collected[name].extend(p50_values)
                if name == "lgbm":
                    for step_idx, (pred_val, act_val) in enumerate(
                        zip(p50_values, fold_actuals),
                        start=1,
                    ):
                        lgbm_preds_by_step.setdefault(step_idx, []).append(float(pred_val))
                        lgbm_actuals_by_step.setdefault(step_idx, []).append(float(act_val))
            except (ValueError, RuntimeError) as exc:
                logger.warning(
                    "Walk-forward fold %d skipped for %s/%s: %s",
                    fold,
                    drug_code,
                    name,
                    exc,
                )
                raw_collected[name].extend([float("nan")] * test_horizon)

    if lgbm_preds_by_step:
        drift = measure_lgbm_recursive_drift(lgbm_preds_by_step, lgbm_actuals_by_step)
        if drift:
            step1 = drift.get(1)
            step_last = drift.get(max(drift))
            if step1 is not None and step_last is not None and step1 > 0:
                logger.info(
                    "LGBM recursive drift for %s: step-1 MAE=%.3f, step-%d MAE=%.3f (ratio=%.2f)",
                    drug_code,
                    step1,
                    max(drift),
                    step_last,
                    step_last / step1,
                )

    return {
        **{name: np.asarray(values, dtype=float) for name, values in raw_collected.items()},
        "actuals": np.asarray(actuals, dtype=float),
        "horizon_steps": np.asarray(horizon_steps, dtype=int),
    }
