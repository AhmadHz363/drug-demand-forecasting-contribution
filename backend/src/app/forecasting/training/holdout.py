"""Calendar-based hold-out validation for forecasting models."""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.forecasting.censored_demand.corrector import correct_demand
from app.forecasting.demand_quantity import as_consumption_demand
from app.forecasting.ensemble.conformal import ConformalCalibrator
from app.forecasting.ensemble.stacking import StackingMetaLearner
from app.forecasting.feature_engineering.pipeline import build_feature_matrix, filter_covered_rows
from app.forecasting.evaluation_metrics import (
    accuracy_from_smape,
    accuracy_skill_from_mase,
    build_evaluation_mask,
    mase,
    mean_pinball_loss,
    primary_validation_metric_for_segment,
    rmsse,
    smape_unreliable_for_series,
    zero_actual_fraction,
)
from app.forecasting.demand_segmentation import classify_demand_segment_from_frame
from app.forecasting.schemas import (
    HoldoutMetrics,
    HoldoutResponse,
    HoldoutSeriesPoint,
    ModelWeightBreakdown,
    PeriodRange,
)
from app.forecasting.training.trainer import MODEL_REGISTRY
from app.forecasting.prediction_bounds import (
    dampen_stacked_horizon_drift,
    demand_prediction_cap,
    sanitize_ensemble_predictions,
)
from app.forecasting.training.walk_forward import (
    _predict_fold,
    accuracy_from_smape,
    collect_walk_forward_predictions,
    smape,
)
from app.services.demand_aggregation import get_receipt_date_bounds

logger = logging.getLogger(__name__)


def _as_date_series(df: pd.DataFrame) -> pd.Series:
    if "demand_date" in df.columns:
        return pd.to_datetime(df["demand_date"]).dt.normalize()
    return pd.to_datetime(df.index).normalize()


def _compute_metrics(
    actuals: np.ndarray,
    predicted: np.ndarray,
    p10: Optional[np.ndarray] = None,
    p90: Optional[np.ndarray] = None,
    *,
    include_mask: Optional[np.ndarray] = None,
) -> HoldoutMetrics:
    actuals = np.asarray(actuals, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = ~np.isnan(predicted)
    if include_mask is not None:
        valid &= np.asarray(include_mask, dtype=bool)
    if not valid.any():
        return HoldoutMetrics(
            smape=0.0,
            mae=0.0,
            coverage_90=0.0,
            accuracy_pct=0.0,
            accuracy_skill_pct=0.0,
        )

    a = actuals[valid]
    p = predicted[valid]
    smape_val = float(np.mean([smape(act, pred) for act, pred in zip(a, p)]))
    mae_val = float(np.mean(np.abs(a - p)))
    mase_val = mase(a, p)
    rmsse_val = rmsse(a, p)

    coverage = 0.0
    pinball_p10 = 0.0
    pinball_p50 = 0.0
    pinball_p90 = 0.0
    if p10 is not None and p90 is not None:
        lower = np.asarray(p10, dtype=float)[valid]
        upper = np.asarray(p90, dtype=float)[valid]
        coverage = float(np.mean((a >= lower) & (a <= upper)))
        pinball_p10 = mean_pinball_loss(a, lower, 0.10)
        pinball_p50 = mean_pinball_loss(a, p, 0.50)
        pinball_p90 = mean_pinball_loss(a, upper, 0.90)

    return HoldoutMetrics(
        smape=smape_val,
        mae=mae_val,
        mase=mase_val,
        rmsse=rmsse_val,
        pinball_p10=pinball_p10,
        pinball_p50=pinball_p50,
        pinball_p90=pinball_p90,
        coverage_90=coverage,
        accuracy_pct=accuracy_from_smape(smape_val),
        accuracy_skill_pct=accuracy_skill_from_mase(mase_val),
    )


def _ensemble_intervals(
    p50: np.ndarray,
    sarima: np.ndarray,
    lgbm: np.ndarray,
    classical: np.ndarray,
    sarima_p10: np.ndarray,
    lgbm_p10: np.ndarray,
    classical_p10: np.ndarray,
    sarima_p90: np.ndarray,
    lgbm_p90: np.ndarray,
    classical_p90: np.ndarray,
    weights: ModelWeightBreakdown,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive ensemble P10/P90 from the best-performing base model."""
    model_stacks = (
        (weights.sarima, sarima_p10, sarima_p90, sarima),
        (weights.lgbm, lgbm_p10, lgbm_p90, lgbm),
        (weights.classical, classical_p10, classical_p90, classical),
    )
    best_weight, best_p10, best_p90, best_p50 = max(model_stacks, key=lambda item: item[0])
    if best_weight > 0:
        return (
            np.clip(best_p10.astype(float), 0.0, None),
            np.maximum(best_p90.astype(float), best_p50.astype(float)),
        )

    weight_arr = np.array([weights.sarima, weights.lgbm, weights.classical], dtype=float)
    if weight_arr.sum() == 0.0:
        weight_arr = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
    else:
        weight_arr = weight_arr / weight_arr.sum()

    def _weighted_quantile(stack: np.ndarray) -> np.ndarray:
        blended = np.zeros(stack.shape[0], dtype=float)
        for idx, weight in enumerate(weight_arr):
            if weight == 0.0:
                continue
            col = stack[:, idx]
            blended += weight * np.where(np.isnan(col), 0.0, col)
        return blended

    p10_stack = np.column_stack([sarima_p10, lgbm_p10, classical_p10])
    p90_stack = np.column_stack([sarima_p90, lgbm_p90, classical_p90])
    p10 = _weighted_quantile(p10_stack)
    p90 = _weighted_quantile(p90_stack)

    spread = np.nanstd(np.column_stack([sarima, lgbm, classical]), axis=1)
    spread = np.where(np.isnan(spread) | (spread == 0), np.maximum(p50 * 0.2, 1.0), spread)
    p10 = np.where(np.isnan(p10) | (p10 == 0.0), np.clip(p50 - spread, 0.0, None), p10)
    p90 = np.where(np.isnan(p90), p50 + spread, p90)
    p10 = np.minimum(p10, p50)
    p90 = np.maximum(p90, p50)
    return p10, p90


def run_holdout_validation(
    drug_code: str,
    db_session: Session,
    train_end: date,
    test_start: date,
    test_end: date,
    models: list[str],
    center_syn_id: Optional[str] = None,
    train_start: Optional[date] = None,
) -> HoldoutResponse:
    """
    Train on demand up to ``train_end``, forecast ``test_start``–``test_end``,
    and compare ensemble + base-model predictions to actual demand.
    """
    if test_start <= train_end:
        raise ValueError("test_start must be after train_end")
    if test_end < test_start:
        raise ValueError("test_end must be on or after test_start")

    valid_models = [name for name in models if name in MODEL_REGISTRY]
    if not valid_models:
        raise ValueError("No valid models requested. Choose from: sarima, lgbm, classical.")

    receipt_start, receipt_end = get_receipt_date_bounds(db_session, drug_code)
    if receipt_start is None or receipt_end is None:
        raise ValueError(f"No receipt history found for drug {drug_code}")

    resolved_train_start = train_start or receipt_start
    if receipt_end < test_end:
        raise ValueError(
            f"Receipt data ends {receipt_end.isoformat()} — cannot evaluate through "
            f"{test_end.isoformat()}. Upload more history or shorten the test window."
        )

    full_df = build_feature_matrix(
        drug_code,
        center_syn_id,
        db_session,
        start_date=resolved_train_start,
        end_date=test_end,
    )
    corrected_df = correct_demand(
        drug_code,
        center_syn_id,
        db_session,
        feature_df=full_df,
    )
    corrected_df = filter_covered_rows(corrected_df)
    dates = _as_date_series(corrected_df)
    train_mask = dates <= pd.Timestamp(train_end)
    test_mask = (dates >= pd.Timestamp(test_start)) & (dates <= pd.Timestamp(test_end))

    if not train_mask.any():
        raise ValueError("Training window is empty — check train_start / train_end.")
    if not test_mask.any():
        raise ValueError("Test window is empty — check test_start / test_end.")

    train_df = corrected_df.loc[train_mask].copy()
    history_df = corrected_df.copy()
    demand_segment = classify_demand_segment_from_frame(train_df)

    test_horizon = int(test_mask.sum())
    actuals = as_consumption_demand(
        corrected_df.loc[test_mask, "total_quantity"].astype(float).values,
    )
    test_dates = [pd.Timestamp(d).date() for d in corrected_df.index[test_mask]]

    model_preds: dict[str, pd.DataFrame] = {}
    model_errors: dict[str, str] = {}

    for model_name in valid_models:
        model_class = MODEL_REGISTRY[model_name]()
        try:
            model_class.train(train_df, drug_code)
            future_covariates = history_df.loc[test_mask] if model_name == "lgbm" else None
            if model_name == "lgbm":
                pred_df = model_class.predict(
                    train_df,
                    test_horizon,
                    future_covariates=future_covariates,
                )
            else:
                pred_df = _predict_fold(
                    model_class,
                    model_name,
                    train_df,
                    history_df,
                    test_horizon,
                )
            if pred_df is None:
                model_errors[model_name] = "Model skipped during prediction"
                continue
            model_preds[model_name] = pred_df
        except Exception as exc:
            logger.warning("Hold-out %s failed for %s: %s", model_name, drug_code, exc)
            model_errors[model_name] = str(exc)

    if not model_preds:
        detail = "; ".join(f"{k}: {v}" for k, v in model_errors.items())
        raise ValueError(f"No models produced hold-out predictions. {detail}")

    def _series(name: str, col: str) -> np.ndarray:
        if name not in model_preds:
            return np.full(test_horizon, np.nan)
        return model_preds[name][col].astype(float).values

    sarima_p50 = _series("sarima", "p50")
    lgbm_p50 = _series("lgbm", "p50")
    classical_p50 = _series("classical", "p50")
    prediction_cap = demand_prediction_cap(
        train_df["total_quantity"].astype(float).values,
    )
    sarima_p50, lgbm_p50, classical_p50 = sanitize_ensemble_predictions(
        sarima_p50,
        lgbm_p50,
        classical_p50,
        prediction_cap,
    )
    sarima_p10 = _series("sarima", "p10")
    lgbm_p10 = _series("lgbm", "p10")
    classical_p10 = _series("classical", "p10")
    sarima_p90 = _series("sarima", "p90")
    lgbm_p90 = _series("lgbm", "p90")
    classical_p90 = _series("classical", "p90")

    stack_models = {name: MODEL_REGISTRY[name] for name in model_preds}
    wf_preds = collect_walk_forward_predictions(train_df, drug_code, stack_models)
    weights = ModelWeightBreakdown(sarima=0.0, lgbm=0.0, classical=0.0)

    if wf_preds is not None and len(wf_preds["actuals"]) >= 5:
        wf_sarima, wf_lgbm, wf_classical = sanitize_ensemble_predictions(
            wf_preds.get("sarima", np.full(len(wf_preds["actuals"]), np.nan)),
            wf_preds.get("lgbm", np.full(len(wf_preds["actuals"]), np.nan)),
            wf_preds.get("classical", np.full(len(wf_preds["actuals"]), np.nan)),
            prediction_cap,
        )
        wf_eval_mask = build_evaluation_mask(wf_preds["actuals"])
        stacker = StackingMetaLearner(segment=demand_segment)
        if wf_eval_mask.any():
            stacker.fit(
                wf_sarima[wf_eval_mask],
                wf_lgbm[wf_eval_mask],
                wf_classical[wf_eval_mask],
                wf_preds["actuals"][wf_eval_mask],
            )
        else:
            stacker.fit(wf_sarima, wf_lgbm, wf_classical, wf_preds["actuals"])
        ensemble_p50, weights = stacker.predict(
            sarima_p50,
            lgbm_p50,
            classical_p50,
            prediction_cap=prediction_cap,
        )
        recent_tail = train_df["total_quantity"].astype(float).tail(28)
        positive_recent = recent_tail[recent_tail > 0]
        recent_level = (
            float(np.median(positive_recent))
            if not positive_recent.empty
            else float(np.median(recent_tail)) if not recent_tail.empty else 0.0
        )
        test_horizon_steps = np.arange(1, len(ensemble_p50) + 1, dtype=int)
        ensemble_p50 = dampen_stacked_horizon_drift(
            ensemble_p50,
            demand_segment=demand_segment,
            recent_level=recent_level,
            horizon_steps=test_horizon_steps,
        )
        wf_ensemble, _ = stacker.predict(
            wf_sarima,
            wf_lgbm,
            wf_classical,
            prediction_cap=prediction_cap,
        )
        wf_spread = np.nanstd(np.column_stack([wf_sarima, wf_lgbm, wf_classical]), axis=1)
        wf_spread = np.where(
            np.isnan(wf_spread) | (wf_spread == 0),
            np.maximum(wf_ensemble * 0.2, 1.0),
            wf_spread,
        )
        wf_p10 = np.clip(wf_ensemble - wf_spread, 0.0, None)
        wf_p90 = wf_ensemble + wf_spread
        calibrator = ConformalCalibrator()
        calibrator.calibrate_scale(
            wf_ensemble[wf_eval_mask],
            wf_preds["actuals"][wf_eval_mask],
            wf_p10[wf_eval_mask],
            wf_p90[wf_eval_mask],
        )
    else:
        available = np.column_stack([sarima_p50, lgbm_p50, classical_p50])
        ensemble_p50 = np.nanmean(available, axis=1)
        active = (~np.isnan(available)).sum(axis=0)
        total_active = max(int(active.sum()), 1)
        weights = ModelWeightBreakdown(
            sarima=float(active[0] / total_active) if active[0] else 0.0,
            lgbm=float(active[1] / total_active) if active[1] else 0.0,
            classical=float(active[2] / total_active) if active[2] else 0.0,
        )

    ensemble_p10, ensemble_p90 = _ensemble_intervals(
        ensemble_p50,
        sarima_p50,
        lgbm_p50,
        classical_p50,
        sarima_p10,
        lgbm_p10,
        classical_p10,
        sarima_p90,
        lgbm_p90,
        classical_p90,
        weights,
    )

    test_df = corrected_df.loc[test_mask]
    gap_flags = (
        test_df["is_coverage_gap"].astype(int).values == 1
        if "is_coverage_gap" in test_df.columns
        else None
    )
    eval_mask = build_evaluation_mask(
        actuals,
        is_stockout=test_df["is_stockout"].values if "is_stockout" in test_df.columns else None,
        is_coverage_gap=gap_flags,
    )

    if wf_preds is not None and len(wf_preds["actuals"]) >= 5:
        _, ensemble_p10, ensemble_p90 = calibrator.adjust_intervals_asymmetric(
            ensemble_p10,
            ensemble_p50,
            ensemble_p90,
        )

    per_model_metrics: dict[str, HoldoutMetrics] = {}
    for model_name in valid_models:
        if model_name not in model_preds:
            continue
        pred = model_preds[model_name]
        per_model_metrics[model_name] = _compute_metrics(
            actuals,
            pred["p50"].astype(float).values,
            pred["p10"].astype(float).values,
            pred["p90"].astype(float).values,
            include_mask=eval_mask,
        )

    ensemble_metrics = _compute_metrics(
        actuals,
        ensemble_p50,
        ensemble_p10,
        ensemble_p90,
        include_mask=eval_mask,
    )

    metrics_by_supply_regime: dict[str, dict[str, HoldoutMetrics]] = {}
    stockout_flags = (
        test_df["is_stockout"].astype(bool).values
        if "is_stockout" in test_df.columns
        else np.zeros(len(actuals), dtype=bool)
    )
    if eval_mask is not None:
        normal_mask = eval_mask & ~stockout_flags
        stockout_mask = eval_mask & stockout_flags
        if normal_mask.any():
            metrics_by_supply_regime["normal_supply"] = {
                "ensemble": _compute_metrics(
                    actuals,
                    ensemble_p50,
                    ensemble_p10,
                    ensemble_p90,
                    include_mask=normal_mask,
                )
            }
        if stockout_mask.any():
            metrics_by_supply_regime["stockout_affected"] = {
                "ensemble": _compute_metrics(
                    actuals,
                    ensemble_p50,
                    ensemble_p10,
                    ensemble_p90,
                    include_mask=stockout_mask,
                )
            }

    series: list[HoldoutSeriesPoint] = []
    for idx, demand_date in enumerate(test_dates):
        series.append(
            HoldoutSeriesPoint(
                date=demand_date,
                actual=float(actuals[idx]),
                sarima_p50=_optional_float(sarima_p50[idx]),
                lgbm_p50=_optional_float(lgbm_p50[idx]),
                classical_p50=_optional_float(classical_p50[idx]),
                ensemble_p50=_optional_float(ensemble_p50[idx]),
                ensemble_p10=_optional_float(ensemble_p10[idx]),
                ensemble_p90=_optional_float(ensemble_p90[idx]),
            )
        )

    zero_frac = zero_actual_fraction(actuals)
    smape_unreliable = smape_unreliable_for_series(
        actuals,
        demand_segment=demand_segment,
    )

    return HoldoutResponse(
        drug_code=drug_code,
        center_syn_id=center_syn_id,
        train_period=PeriodRange(start=resolved_train_start, end=train_end),
        test_period=PeriodRange(start=test_start, end=test_end),
        models_evaluated=sorted(model_preds.keys()),
        model_errors=model_errors,
        metrics={"ensemble": ensemble_metrics, **per_model_metrics},
        demand_segment=demand_segment,
        total_accuracy_pct=ensemble_metrics.accuracy_pct,
        total_accuracy_skill_pct=ensemble_metrics.accuracy_skill_pct,
        primary_accuracy_metric=primary_validation_metric_for_segment(demand_segment),
        smape_unreliable=smape_unreliable,
        validation_zero_actual_fraction=zero_frac,
        model_weights=weights,
        series=series,
        metrics_by_supply_regime=metrics_by_supply_regime,
    )


def _optional_float(value: float) -> Optional[float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return float(value)
