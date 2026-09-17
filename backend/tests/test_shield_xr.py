"""Unit tests for SHIELD-XR core components."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from app.forecasting.shield_xr.anomaly_guard import apply_anomaly_guard
from app.forecasting.shield_xr.features import (
    FEATURES_X,
    add_features,
    assign_sb_classes,
    feature_row_to_frame,
    syntetos_boylan,
)
from app.forecasting.shield_xr.metrics import accuracy_from_wape, wape


def _sample_panel(n_days: int = 120, n_skus: int = 3) -> pd.DataFrame:
    start = date(2024, 1, 1)
    rows = []
    for sku_idx in range(n_skus):
        code = f"P{sku_idx:03d}"
        for offset in range(n_days):
            day = start + timedelta(days=offset)
            demand = float(max(0, 5 + sku_idx + (offset % 7)))
            rows.append(
                {
                    "CODE": code,
                    "DATE": pd.Timestamp(day),
                    "demand": demand,
                    "ARTICLE": f"Drug {code}",
                    "CAT": "5060",
                    "n_unique_patients": 100.0,
                    "n_admissions": 10.0,
                    "n_unique_doctors": 5.0,
                    "n_unique_CR": 2.0,
                    "n_unique_CS": 1.0,
                    "n_transactions": 200.0,
                    "n_demand_txns": 150.0,
                    "top1_CR_share": 0.6,
                    "top2_CR_share": 0.3,
                    "top3_CR_share": 0.1,
                    "bed_occupancy_rate": 0.8,
                    "weekly_surgery_count": 20.0,
                }
            )
    return pd.DataFrame(rows)


def test_syntetos_boylan_smooth():
    series = pd.Series([10, 11, 9, 10, 12, 11, 10, 9])
    assert syntetos_boylan(series) == "smooth"


def test_anomaly_guard_produces_clean_demand():
    panel = _sample_panel()
    d_train_end = pd.Timestamp(panel["DATE"].quantile(0.7))
    result = apply_anomaly_guard(panel, d_train_end)
    frame = result.frame
    assert "demand_clean" in frame.columns
    assert "demand_raw" in frame.columns
    assert frame["demand_clean"].ge(0).all()


def test_feature_engineering_adds_lags():
    panel = _sample_panel()
    d_train_end = pd.Timestamp(panel["DATE"].quantile(0.7))
    guarded = apply_anomaly_guard(panel, d_train_end).frame
    classified = assign_sb_classes(guarded, d_train_end)
    feat = add_features(classified)
    assert "lag_1" in feat.columns
    assert "roll_mean_7" in feat.columns
    assert "sb_class_id" in feat.columns


def test_wape_and_accuracy():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 28.0])
    assert wape(y_true, y_pred) == pytest.approx(0.1, rel=1e-6)
    assert accuracy_from_wape(y_true, y_pred) == pytest.approx(0.9, rel=1e-6)


def test_feature_row_to_frame_coerces_object_dtypes():
    row = pd.Series({col: str(idx) for idx, col in enumerate(FEATURES_X)})
    row["sb_class"] = "smooth"
    frame = feature_row_to_frame(row, FEATURES_X)
    assert list(frame.columns) == FEATURES_X
    assert frame.dtypes.apply(lambda dt: dt.kind in "iufb").all()
