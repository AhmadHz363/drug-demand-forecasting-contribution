"""Walk-forward cross-validation for forecasting models."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from app.forecasting.constants import TFT_WALK_FORWARD_MAX_EPOCHS
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


from app.forecasting.evaluation_metrics import accuracy_from_smape, smape

__all__ = ["accuracy_from_smape", "smape", "walk_forward_smape", "walk_forward_coverage", "collect_walk_forward_predictions", "demand_prediction_cap", "sanitize_ensemble_predictions", "winsorize_predictions"]


def walk_forward_smape(
    model: BaseForecastingModel,
    df: pd.DataFrame,
    drug_code: str,
    n_splits: int = 3,
    test_horizon: int = 7,
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


def _train_fold_model(
    fold_model: BaseForecastingModel,
    model_class: type[BaseForecastingModel],
    train_df: pd.DataFrame,
    drug_code: str,
) -> None:
    if model_class is TFTModel:
        fold_model.train(train_df, drug_code, max_epochs=TFT_WALK_FORWARD_MAX_EPOCHS)
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
    n_splits: int = 3,
    test_horizon: int = 7,
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


def collect_walk_forward_predictions(
    df: pd.DataFrame,
    drug_code: str,
    model_classes: dict[str, type[BaseForecastingModel]],
    n_splits: int = 3,
    test_horizon: int = 7,
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

    for fold in range(n_splits):
        test_start = n - total_test + fold * test_horizon
        test_end = test_start + test_horizon
        train_df = df.iloc[:test_start]
        history_df = df.iloc[:test_end]
        fold_actuals = df.iloc[test_start:test_end]["total_quantity"].astype(float).values
        actuals.extend(as_consumption_demand(fold_actuals).tolist())

        for name, model_class in model_classes.items():
            fold_model = model_class()
            try:
                _train_fold_model(fold_model, model_class, train_df, drug_code)
                pred_df = _predict_fold(fold_model, name, train_df, history_df, test_horizon)
                if pred_df is None:
                    collected[name].extend([float("nan")] * test_horizon)
                    continue
                collected[name].extend(pred_df["p50"].astype(float).tolist())
            except (ValueError, RuntimeError) as exc:
                logger.warning(
                    "Walk-forward fold %d skipped for %s/%s: %s",
                    fold,
                    drug_code,
                    name,
                    exc,
                )
                collected[name].extend([float("nan")] * test_horizon)

    return {
        **{name: np.asarray(values, dtype=float) for name, values in collected.items()},
        "actuals": np.asarray(actuals, dtype=float),
    }
