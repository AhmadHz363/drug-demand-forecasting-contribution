"""Notebook-aligned training evaluation for SHIELD-XR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from app.forecasting.shield_xr.metrics import accuracy_from_wape, wape


@dataclass
class ShieldXRTrainingMetrics:
    daily_ensemble_accuracy: float = 0.0
    daily_ensemble_wape: float = float("nan")
    hospital_weekly_accuracy: float = 0.0
    hospital_weekly_wape: float = float("nan")
    reconciled_weekly_per_drug_mean_accuracy: float = 0.0
    reconciled_weekly_per_drug_median_accuracy: float = 0.0
    volume_weighted_weekly_accuracy: float = 0.0
    non_lumpy_weekly_accuracy: float = 0.0
    hybrid_abc_combined_accuracy: float = 0.0
    per_drug_weekly_accuracy: dict[str, float] = field(default_factory=dict)
    per_sb_class_weekly_accuracy: dict[str, float] = field(default_factory=dict)
    n_test_drugs: int = 0
    n_test_weeks: int = 0
    n_hybrid_abc_named_drugs: int = 0


HYBRID_ABC_TOP_N = 15
HYBRID_ABC_STABILITY_THRESHOLD = 0.40


def _weekly_hospital_total_accuracy(
    test: pd.DataFrame,
    ensemble_pred: np.ndarray,
    *,
    use_raw: bool = True,
    drop_anomaly_days: bool = True,
) -> tuple[float, float]:
    frame = test.copy()
    frame["ensemble_pred"] = ensemble_pred
    if drop_anomaly_days and "is_anomaly_day" in frame.columns:
        frame = frame.loc[~frame["is_anomaly_day"]].copy()
    actual_col = "demand_raw" if use_raw and "demand_raw" in frame.columns else "demand"
    frame["week"] = frame["DATE"].dt.to_period("W").astype(str)
    weekly = frame.groupby("week", as_index=False).agg(
        y=(actual_col, "sum"),
        yhat=("ensemble_pred", "sum"),
    )
    if weekly.empty or weekly["y"].abs().sum() == 0:
        return 0.0, float("nan")
    score_wape = wape(weekly["y"].values, weekly["yhat"].values)
    return accuracy_from_wape(weekly["y"].values, weekly["yhat"].values), score_wape


def _per_drug_weekly_accuracy(
    weekly_test: pd.DataFrame,
    *,
    pred_col: str = "yhat_breakdown",
    actual_col: str = "y_raw",
    clean_weeks_only: bool = True,
) -> dict[str, float]:
    frame = weekly_test.copy()
    if clean_weeks_only and "n_anomaly" in frame.columns:
        frame = frame.loc[frame["n_anomaly"] == 0].copy()
    scores: dict[str, float] = {}
    for code, group in frame.groupby("CODE"):
        actual = group[actual_col].astype(float).values
        if np.abs(actual).sum() == 0:
            continue
        pred = group[pred_col].astype(float).values
        scores[str(code)] = accuracy_from_wape(actual, pred)
    return scores


def _pooled_weekly_accuracy(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    actual = frame["y_raw"].astype(float).values
    pred = frame["yhat_breakdown"].astype(float).values
    if np.abs(actual).sum() == 0:
        return 0.0
    return accuracy_from_wape(actual, pred)


def _select_hybrid_abc_named_drugs(
    weekly_valid: pd.DataFrame,
    weekly_train: pd.DataFrame,
    *,
    top_n: int = HYBRID_ABC_TOP_N,
    stability_threshold: float = HYBRID_ABC_STABILITY_THRESHOLD,
) -> list[str]:
    """Pick top-N stable high-volume SKUs using validation accuracy only."""
    valid = weekly_valid.copy()
    if "n_anomaly" in valid.columns:
        valid = valid.loc[valid["n_anomaly"] == 0].copy()
    if valid.empty or "yhat_breakdown" not in valid.columns:
        pred_col = "yhat_weekly_direct" if "yhat_weekly_direct" in valid.columns else None
        if pred_col is None:
            return []
        valid = valid.copy()
        valid["yhat_breakdown"] = valid[pred_col]

    valid_scores = _per_drug_weekly_accuracy(valid, clean_weeks_only=False)
    train_vol = weekly_train.groupby("CODE")["y_clean"].sum().sort_values(ascending=False)
    named: list[str] = []
    for code in train_vol.index.astype(str):
        if len(named) >= top_n:
            break
        if valid_scores.get(code, 0.0) >= stability_threshold:
            named.append(code)
    return named


def _hybrid_abc_combined_accuracy(
    weekly_test: pd.DataFrame,
    named_drugs: list[str],
) -> tuple[float, int]:
    clean = weekly_test.copy()
    if "n_anomaly" in clean.columns:
        clean = clean.loc[clean["n_anomaly"] == 0].copy()
    if clean.empty or not named_drugs:
        return 0.0, 0

    named_set = set(named_drugs)
    rows: list[dict[str, float]] = []
    for week_str, group in clean.groupby("week_str"):
        actual_named = group.loc[group["CODE"].isin(named_set), "y_raw"].astype(float).sum()
        pred_named = group.loc[group["CODE"].isin(named_set), "yhat_breakdown"].astype(float).sum()
        actual_tail = group.loc[~group["CODE"].isin(named_set), "y_raw"].astype(float).sum()
        pred_tail = group.loc[~group["CODE"].isin(named_set), "yhat_breakdown"].astype(float).sum()
        rows.append(
            {
                "y": actual_named + actual_tail,
                "yhat": pred_named + pred_tail,
            }
        )
    if not rows:
        return 0.0, len(named_drugs)
    pooled = pd.DataFrame(rows)
    return _pooled_weekly_accuracy(
        pooled.rename(columns={"y": "y_raw", "yhat": "yhat_breakdown"}),
    ), len(named_drugs)


def compute_training_metrics(
    test: pd.DataFrame,
    ensemble_pred: np.ndarray,
    weekly_test: Optional[pd.DataFrame],
    *,
    weekly_valid: Optional[pd.DataFrame] = None,
    weekly_train: Optional[pd.DataFrame] = None,
) -> ShieldXRTrainingMetrics:
    mask_ok = ~test["is_anomaly_day"].values if "is_anomaly_day" in test.columns else np.ones(len(test), dtype=bool)
    actual = test.loc[mask_ok, "demand_raw"].values if mask_ok.any() else test["demand"].values
    preds = ensemble_pred[mask_ok] if mask_ok.any() else ensemble_pred

    metrics = ShieldXRTrainingMetrics(
        daily_ensemble_accuracy=accuracy_from_wape(actual, preds),
        daily_ensemble_wape=wape(actual, preds),
    )
    metrics.hospital_weekly_accuracy, metrics.hospital_weekly_wape = _weekly_hospital_total_accuracy(
        test,
        ensemble_pred,
    )

    if weekly_test is not None and not weekly_test.empty and "yhat_breakdown" in weekly_test.columns:
        per_drug = _per_drug_weekly_accuracy(weekly_test)
        metrics.per_drug_weekly_accuracy = per_drug
        metrics.n_test_drugs = len(per_drug)
        metrics.n_test_weeks = weekly_test["week_str"].nunique() if "week_str" in weekly_test.columns else 0
        if per_drug:
            values = list(per_drug.values())
            metrics.reconciled_weekly_per_drug_mean_accuracy = float(np.mean(values))
            metrics.reconciled_weekly_per_drug_median_accuracy = float(np.median(values))

        clean = weekly_test.loc[weekly_test.get("n_anomaly", 0) == 0].copy()
        metrics.volume_weighted_weekly_accuracy = _pooled_weekly_accuracy(clean)
        non_lumpy = clean.loc[clean["sb_class"] != "lumpy"].copy()
        metrics.non_lumpy_weekly_accuracy = _pooled_weekly_accuracy(non_lumpy)

        if weekly_valid is not None and weekly_train is not None:
            named = _select_hybrid_abc_named_drugs(weekly_valid, weekly_train)
            hybrid_acc, n_named = _hybrid_abc_combined_accuracy(clean, named)
            metrics.hybrid_abc_combined_accuracy = hybrid_acc
            metrics.n_hybrid_abc_named_drugs = n_named

        if "sb_class" in weekly_test.columns:
            for sb_class, group in clean.groupby("sb_class"):
                actual = group["y_raw"].astype(float).values
                pred = group["yhat_breakdown"].astype(float).values
                if np.abs(actual).sum() == 0:
                    continue
                metrics.per_sb_class_weekly_accuracy[str(sb_class)] = accuracy_from_wape(actual, pred)

    return metrics
