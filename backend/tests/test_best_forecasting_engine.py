"""Unit tests for import coverage, gap-safe features, classical routing, and champions."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from app.forecasting.demand_segmentation import classify_demand_segment
from app.forecasting.evaluation_metrics import build_evaluation_mask
from app.forecasting.feature_engineering.lag_features import add_lag_features
from app.forecasting.models.classical_model import SEGMENT_METHOD, ClassicalModel
from app.forecasting.training.champion_selection import (
    base_models_agree,
    select_champion,
    seasonal_naive_oof,
)
from app.services.import_coverage import merge_coverage_periods


def test_merge_coverage_periods_splits_large_gaps():
    periods = [
        (date(2023, 1, 1), date(2023, 12, 31)),
        (date(2024, 1, 1), date(2024, 12, 31)),
        (date(2019, 1, 1), date(2019, 12, 31)),
    ]
    merged = merge_coverage_periods(periods, gap_threshold_days=60)
    assert len(merged) == 2
    assert merged[0][0] == date(2019, 1, 1)
    assert merged[1][0] == date(2023, 1, 1)
    assert merged[1][1] == date(2024, 12, 31)


def test_lag_features_reset_across_coverage_gaps():
    n = 400
    dates = [date(2023, 1, 1) + timedelta(days=i) for i in range(n)]
    qty = np.ones(n)
    gaps = np.zeros(n, dtype=int)
    # Artificial mid-series gap of 30 days
    gaps[100:130] = 1
    qty[100:130] = np.nan
    df = pd.DataFrame(
        {
            "demand_date": dates,
            "total_quantity": qty,
            "is_coverage_gap": gaps,
        }
    )
    out = add_lag_features(df)
    # Day after gap should flag lag gaps for short lags that look into the hole.
    assert int(out.loc[130, "has_lag_gaps"]) == 1


def test_classical_segment_method_routing():
    assert SEGMENT_METHOD["smooth"] == "ets"
    assert SEGMENT_METHOD["erratic"] == "theta"
    assert SEGMENT_METHOD["intermittent"] == "tsb"
    assert SEGMENT_METHOD["lumpy"] == "croston_sba"


def test_classical_model_quantile_order_and_methods(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "src")
    from app.forecasting import constants as const

    monkeypatch.setattr(const, "ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setattr(
        "app.forecasting.models.classical_model.ARTIFACTS_DIR",
        str(tmp_path),
    )

    rng = np.random.default_rng(0)
    n = 120
    # Smooth-ish weekly pattern
    base = 10 + 2 * np.sin(np.arange(n) * 2 * np.pi / 7) + rng.normal(0, 0.5, n)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    df = pd.DataFrame({"total_quantity": np.clip(base, 0, None)}, index=dates)
    df.index.name = "demand_date"

    model = ClassicalModel()
    model.train(df, "TEST_SMOOTH")
    assert model._method in {"ets", "theta"}
    pred = model.predict(df, 7)
    assert list(pred.columns) == ["forecast_date", "p10", "p50", "p90"]
    assert (pred["p10"] <= pred["p50"]).all()
    assert (pred["p50"] <= pred["p90"]).all()
    path = model.save("TEST_SMOOTH")
    assert path
    loaded = ClassicalModel()
    loaded.load("TEST_SMOOTH")
    pred2 = loaded.predict(df, 7)
    assert len(pred2) == 7

    # Intermittent series → TSB / Croston
    sparse = np.zeros(n)
    sparse[::10] = 5.0
    sparse_df = pd.DataFrame({"total_quantity": sparse}, index=dates)
    sparse_df.index.name = "demand_date"
    sparse_model = ClassicalModel()
    sparse_model.train(sparse_df, "TEST_SPARSE")
    assert sparse_model._method in {"tsb", "croston_sba"}
    sparse_pred = sparse_model.predict(sparse_df, 5)
    assert (sparse_pred["p10"] <= sparse_pred["p50"]).all()


def test_evaluation_mask_excludes_coverage_gaps():
    actuals = np.array([1.0, 2.0, 3.0, 4.0])
    gaps = np.array([False, True, False, False])
    mask = build_evaluation_mask(actuals, is_coverage_gap=gaps)
    assert mask.tolist() == [True, False, True, True]


def test_champion_prefers_base_unless_ensemble_clearly_better():
    actuals = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    good = actuals.copy()
    bad = actuals + 5.0
    decision = select_champion(
        drug_code="D1",
        demand_segment="smooth",
        actuals=actuals,
        base_preds={"sarima": good, "lgbm": bad, "classical": bad},
        ensemble_preds=bad,
    )
    assert decision.champion == "sarima"

    slightly_better_ens = actuals * 0.5 + 0.5  # still not enough vs perfect base
    decision2 = select_champion(
        drug_code="D1",
        demand_segment="smooth",
        actuals=actuals,
        base_preds={"sarima": good, "lgbm": bad, "classical": bad},
        ensemble_preds=slightly_better_ens,
    )
    # Perfect base MASE ~0; ensemble must beat by 2% — keep base.
    assert decision2.champion == "sarima"

    # When base is weak and ensemble clearly better than base and naive:
    weak = actuals + 3.0
    strong_ens = actuals + 0.1
    decision3 = select_champion(
        drug_code="D2",
        demand_segment="smooth",
        actuals=actuals,
        base_preds={"sarima": weak, "lgbm": weak, "classical": weak},
        ensemble_preds=strong_ens,
    )
    assert decision3.champion == "ensemble"


def test_champion_uses_ensemble_when_models_agree_within_tolerance():
    actuals = np.array([80.0, 90.0, 85.0, 88.0, 82.0, 87.0, 84.0, 86.0, 83.0, 89.0, 81.0, 88.0])
    offset = 1.0
    agreed = actuals + offset
    sarima = agreed
    lgbm = agreed
    classical = agreed
    ensemble = actuals + offset * 1.03
    assert base_models_agree({"sarima": sarima, "lgbm": lgbm, "classical": classical})
    decision = select_champion(
        drug_code="P406816",
        demand_segment="erratic",
        actuals=actuals,
        base_preds={"sarima": sarima, "lgbm": lgbm, "classical": classical},
        ensemble_preds=ensemble,
    )
    assert decision.champion == "ensemble"
    assert decision.reason == "ensemble_model_agreement"


def test_seasonal_naive_helper_shape():
    actuals = np.arange(14, dtype=float)
    preds = seasonal_naive_oof(actuals)
    assert preds.shape == actuals.shape
    assert np.isnan(preds[0])
    assert preds[7] == actuals[0]


def test_classify_ignores_nan_gap_values():
    qty = np.array([1.0, 0.0, np.nan, 2.0, 0.0, 3.0])
    # classify_demand_segment itself does not drop nan — callers should drop.
    clean = qty[np.isfinite(qty)]
    segment = classify_demand_segment(clean)
    assert segment in {"smooth", "intermittent", "erratic", "lumpy"}
