"""Walk-forward cross-validation for forecasting models."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from app.forecasting.constants import (
    TFT_MAX_EPOCHS,
    TFT_WALK_FORWARD_EPOCH_RATIO,
    TFT_WALK_FORWARD_MAX_EPOCHS,
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
from app.forecasting.models.lgbm_model import LightGBMModel
from app.forecasting.models.sarima_model import SarimaModel
from app.forecasting.models.tft_model import TFTModel

logger = logging.getLogger(__name__)


def _model_registry_name(model: BaseForecastingModel) -> str:
    if isinstance(model, SarimaModel):
        return "sarima"
    if isinstance(model, LightGBMModel):
        return "lgbm"
    return "tft"


from app.forecasting.evaluation_metrics import (
    accuracy_from_smape,
    accuracy_skill_from_mase,
    mase,
    smape,
)

__all__ = [
    "accuracy_from_smape",
    "accuracy_skill_from_mase",
    "mase",
    "smape",
    "walk_forward_smape",
    "walk_forward_coverage",
    "walk_forward_mase",
    "collect_walk_forward_predictions",
    "measure_lgbm_recursive_drift",
    "demand_prediction_cap",
    "sanitize_ensemble_predictions",
    "winsorize_predictions",
]


def _walk_forward_tft_epochs() -> int:
    """Return the TFT epoch budget for walk-forward folds.

    TFT_WALK_FORWARD_MAX_EPOCHS is a hard ceiling — the ratio-derived
    value is capped so walk-forward folds never run longer than the
    documented limit (previously `max` was used, which made the constant
    act as a floor and ignored it entirely when the ratio exceeded it).
    """
    ratio_epochs = max(1, int(TFT_MAX_EPOCHS * TFT_WALK_FORWARD_EPOCH_RATIO))
    return min(TFT_WALK_FORWARD_MAX_EPOCHS, ratio_epochs)


def walk_forward_smape(
    model: BaseForecastingModel,
    df: pd.DataFrame,
    drug_code: str,
    n_splits: int = WALK_FORWARD_N_SPLITS,
    test_horizon: int = WALK_FORWARD_TEST_HORIZON,
) -> float:
    """
    Performs walk-forward validation:
    - Splits the time series into n_splits folds
    - For each fold: train on all data before the fold, predict the fold
    - Computes sMAPE between predicted P50 and actual demand
    - Returns average sMAPE across all folds
    """
    n = len(df)
    total_test = n_splits * test_horizon
    if n <= total_test:
        raise ValueError(
            f"Need more than {total_test} rows for {n_splits} folds of horizon {test_horizon}, got {n}"
        )

    model_class = type(model)
    scores: list[float] = []

    for fold in range(n_splits):
        test_start = n - total_test + fold * test_horizon
        test_end = test_start + test_horizon
        train_df = df.iloc[:test_start]
        history_df = df.iloc[:test_end]

        fold_model = model_class()
        try:
            _train_fold_model(fold_model, model_class, train_df, drug_code)
        except ValueError as exc:
            logger.warning("Walk-forward fold %d skipped for %s: %s", fold, drug_code, exc)
            continue

        pred_df = _predict_fold(
            fold_model,
            _model_registry_name(fold_model),
            train_df,
            history_df,
            test_horizon,
        )
        if pred_df is None:
            continue

        actuals = as_consumption_demand(
            df.iloc[test_start:test_end]["total_quantity"].astype(float).values,
        )
        predicted = pred_df["p50"].astype(float).values
        fold_scores = [smape(a, p) for a, p in zip(actuals, predicted)]
        scores.extend(fold_scores)

    if not scores:
        raise ValueError(f"No walk-forward folds completed for {drug_code}")

    return float(sum(scores) / len(scores))


def walk_forward_mase(
    model: BaseForecastingModel,
    df: pd.DataFrame,
    drug_code: str,
    n_splits: int = WALK_FORWARD_N_SPLITS,
    test_horizon: int = WALK_FORWARD_TEST_HORIZON,
) -> float:
    """Average MASE across walk-forward folds (seasonal-naive scaled)."""
    n = len(df)
    total_test = n_splits * test_horizon
    if n <= total_test:
        raise ValueError(
            f"Need more than {total_test} rows for {n_splits} folds of horizon {test_horizon}, got {n}"
        )

    model_class = type(model)
    fold_mases: list[float] = []

    for fold in range(n_splits):
        test_start = n - total_test + fold * test_horizon
        test_end = test_start + test_horizon
        train_df = df.iloc[:test_start]
        history_df = df.iloc[:test_end]

        fold_model = model_class()
        try:
            _train_fold_model(fold_model, model_class, train_df, drug_code)
        except ValueError as exc:
            logger.warning("Walk-forward fold %d skipped for %s: %s", fold, drug_code, exc)
            continue

        pred_df = _predict_fold(
            fold_model,
            _model_registry_name(fold_model),
            train_df,
            history_df,
            test_horizon,
        )
        if pred_df is None:
            continue

        actuals = as_consumption_demand(
            df.iloc[test_start:test_end]["total_quantity"].astype(float).values,
        )
        predicted = pred_df["p50"].astype(float).values
        fold_mases.append(mase(actuals, predicted))

    if not fold_mases:
        raise ValueError(f"No walk-forward folds completed for {drug_code}")

    return float(sum(fold_mases) / len(fold_mases))


def _train_fold_model(
    fold_model: BaseForecastingModel,
    model_class: type[BaseForecastingModel],
    train_df: pd.DataFrame,
    drug_code: str,
) -> None:
    if model_class is TFTModel:
        fold_model.train(train_df, drug_code, max_epochs=_walk_forward_tft_epochs())
        return
    fold_model.train(train_df, drug_code)


def _predict_fold(
    fold_model: BaseForecastingModel,
    model_name: str,
    train_df: pd.DataFrame,
    history_df: pd.DataFrame,
    test_horizon: int,
) -> Optional[pd.DataFrame]:
    if model_name == "tft" and getattr(fold_model, "_skipped", False):
        return None
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
    """Fraction of walk-forward test points falling inside the model P10–P90 band."""
    n = len(df)
    total_test = n_splits * test_horizon
    if n <= total_test:
        return 0.0

    model_class = type(model)
    covered = 0
    total = 0

    for fold in range(n_splits):
        test_start = n - total_test + fold * test_horizon
        test_end = test_start + test_horizon
        train_df = df.iloc[:test_start]
        history_df = df.iloc[:test_end]

        fold_model = model_class()
        try:
            _train_fold_model(fold_model, model_class, train_df, drug_code)
        except ValueError as exc:
            logger.warning("Coverage fold %d skipped for %s: %s", fold, drug_code, exc)
            continue

        model_name = model_class.__name__.replace("Model", "").lower()
        if model_name == "lightgbm":
            model_name = "lgbm"
        pred_df = _predict_fold(fold_model, model_name, train_df, history_df, test_horizon)
        if pred_df is None:
            continue

        actuals = as_consumption_demand(
            df.iloc[test_start:test_end]["total_quantity"].astype(float).values,
        )
        lower = pred_df["p10"].astype(float).values
        upper = pred_df["p90"].astype(float).values
        covered += int(((actuals >= lower) & (actuals <= upper)).sum())
        total += len(actuals)

    if total == 0:
        return 0.0
    return float(covered / total)


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
) -> Optional[dict[str, np.ndarray]]:
    """
    Collect out-of-fold P50 predictions from walk-forward validation.

    Returns arrays keyed by model name plus ``actuals``, or None when the
    series is too short for the configured fold layout.
    """
    n = len(df)
    total_test = n_splits * test_horizon
    if n <= total_test:
        return None

    collected: dict[str, list[float]] = {name: [] for name in model_classes}
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
                    collected[name].extend([float("nan")] * test_horizon)
                    continue
                p50_values = pred_df["p50"].astype(float).tolist()
                collected[name].extend(p50_values)
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
                collected[name].extend([float("nan")] * test_horizon)

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
        **{name: np.asarray(values, dtype=float) for name, values in collected.items()},
        "actuals": np.asarray(actuals, dtype=float),
        "horizon_steps": np.asarray(horizon_steps, dtype=int),
    }
