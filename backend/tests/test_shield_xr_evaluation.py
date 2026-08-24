"""Tests for notebook-aligned SHIELD-XR evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.forecasting.shield_xr.evaluation import compute_training_metrics


def test_hospital_weekly_accuracy_perfect_when_preds_match():
    test = pd.DataFrame(
        {
            "DATE": pd.date_range("2024-01-01", periods=14, freq="D"),
            "demand_raw": [10.0] * 14,
            "demand": [10.0] * 14,
            "is_anomaly_day": [False] * 14,
        }
    )
    preds = np.array([10.0] * 14)
    metrics = compute_training_metrics(test, preds, None)
    assert metrics.hospital_weekly_accuracy == 1.0


def test_per_drug_weekly_accuracy_from_reconciled_panel():
    weekly_test = pd.DataFrame(
        {
            "CODE": ["A", "A", "B", "B"],
            "y_raw": [10.0, 12.0, 5.0, 0.0],
            "yhat_breakdown": [10.0, 12.0, 4.0, 0.0],
            "n_anomaly": [0, 0, 0, 0],
            "week_str": ["2024-W01", "2024-W02", "2024-W01", "2024-W02"],
            "sb_class": ["smooth", "smooth", "lumpy", "lumpy"],
        }
    )
    test = pd.DataFrame(
        {
            "DATE": pd.date_range("2024-01-01", periods=7, freq="D"),
            "demand_raw": [1.0] * 7,
            "demand": [1.0] * 7,
            "is_anomaly_day": [False] * 7,
        }
    )
    metrics = compute_training_metrics(test, np.ones(7), weekly_test)
    assert metrics.per_drug_weekly_accuracy["A"] == 1.0
    assert metrics.reconciled_weekly_per_drug_mean_accuracy > 0.8


def test_non_lumpy_and_hybrid_abc_metrics():
    weekly_valid = pd.DataFrame(
        {
            "CODE": ["A", "A", "B", "B", "C", "C"],
            "y_raw": [10.0, 12.0, 8.0, 9.0, 2.0, 3.0],
            "yhat_weekly_direct": [10.0, 12.0, 7.0, 8.0, 2.0, 3.0],
            "n_anomaly": [0, 0, 0, 0, 0, 0],
            "week_str": ["2024-W01", "2024-W02", "2024-W01", "2024-W02", "2024-W01", "2024-W02"],
            "sb_class": ["smooth", "smooth", "smooth", "smooth", "lumpy", "lumpy"],
        }
    )
    weekly_train = pd.DataFrame({"CODE": ["A", "B", "C"], "y_clean": [100.0, 80.0, 5.0]})
    weekly_test = pd.DataFrame(
        {
            "CODE": ["A", "A", "B", "B", "C", "C"],
            "y_raw": [10.0, 12.0, 8.0, 9.0, 1.0, 1.0],
            "yhat_breakdown": [10.0, 12.0, 8.0, 9.0, 1.0, 1.0],
            "n_anomaly": [0, 0, 0, 0, 0, 0],
            "week_str": ["2024-W01", "2024-W02", "2024-W01", "2024-W02", "2024-W01", "2024-W02"],
            "sb_class": ["smooth", "smooth", "smooth", "smooth", "lumpy", "lumpy"],
        }
    )
    test = pd.DataFrame(
        {
            "DATE": pd.date_range("2024-01-01", periods=7, freq="D"),
            "demand_raw": [1.0] * 7,
            "demand": [1.0] * 7,
            "is_anomaly_day": [False] * 7,
        }
    )
    metrics = compute_training_metrics(
        test,
        np.ones(7),
        weekly_test,
        weekly_valid=weekly_valid,
        weekly_train=weekly_train,
    )
    assert metrics.non_lumpy_weekly_accuracy == 1.0
    assert metrics.hybrid_abc_combined_accuracy > 0.9
