"""
Phase 0 forecasting fixes — history source, imputation guard, ensemble weights,
interval sanity checks.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.forecasting.censored_demand.corrector import (
    assert_no_degenerate_stockout_imputation,
    correct_demand,
)
from app.forecasting.censored_demand.detector import detect_stockout_windows
from app.forecasting.censored_demand.tobit import apply_tobit_correction
from app.forecasting.ensemble.stacking import StackingMetaLearner
from app.forecasting.feature_engineering.rolling_features import add_rolling_features
from app.forecasting.feature_engineering.temporal_features import add_temporal_features
from app.forecasting.inference.forecaster import (
    DrugForecaster,
    check_forecast_interval_sanity,
    validate_forecast_interval_sanity,
)
from app.forecasting.schemas import DailyForecastPoint, ForecastResponse, ModelWeightBreakdown
from app.core.database import SessionLocal


def _synthetic_feature_frame(quantities: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=len(quantities), freq="D")
    base = pd.DataFrame({"demand_date": dates.date, "total_quantity": quantities})
    base = add_temporal_features(base)
    base = add_rolling_features(base)
    return base.set_index("demand_date")


class TestHistoryUsesObservedReceipts:
    def test_history_chart_uses_observed_not_imputed(self):
        quantities = [10.0] * 10 + [0.0] * 50 + [12.0] * 10
        df = _synthetic_feature_frame(quantities)
        db = SessionLocal()
        try:
            corrected = correct_demand("PHASE0-HIST", None, db, df)
        finally:
            db.rollback()
            db.close()

        stockout_mask = corrected["is_stockout"].astype(bool)
        assert stockout_mask.any()
        assert (
            corrected.loc[stockout_mask, "observed_quantity"].astype(float) == 0.0
        ).all()
        assert (
            corrected.loc[stockout_mask, "total_quantity"].astype(float) > 0.0
        ).all()

        history = DrugForecaster()._history_for_chart(corrected, max_days=len(corrected))
        history_by_date = {point.date: point.quantity for point in history}
        stockout_dates = corrected.index[stockout_mask]
        for demand_ts in stockout_dates:
            demand_date = pd.Timestamp(demand_ts).date()
            assert history_by_date[demand_date] == 0.0


class TestDegenerateImputationGuard:
    def test_assertion_flags_identical_stockout_imputations(self):
        df = pd.DataFrame(
            {
                "total_quantity": [31.043549225245666] * 5,
                "is_stockout": [True] * 5,
            }
        )
        with pytest.raises(ValueError, match="Degenerate stockout imputation"):
            assert_no_degenerate_stockout_imputation(df)

    def test_assertion_allows_varying_stockout_imputations(self):
        quantities = [10.0] * 10 + [0.0] * 50 + [12.0] * 10
        df = _synthetic_feature_frame(quantities).reset_index()
        df = detect_stockout_windows(df, "PHASE0-IMPUTE", db_session=None)
        corrected = apply_tobit_correction(df)
        assert_no_degenerate_stockout_imputation(corrected)

    def test_tobit_imputation_not_byte_identical_across_dates(self):
        quantities = [10.0, 12.0, 11.0, 9.0, 13.0] + [0.0] * 50 + [15.0] * 5
        df = _synthetic_feature_frame(quantities).reset_index()
        df = detect_stockout_windows(df, "PHASE0-TOBIT", db_session=None)
        corrected = apply_tobit_correction(df)
        stockout_values = corrected.loc[corrected["is_stockout"], "total_quantity"].astype(float)
        assert len(stockout_values) >= 3
        assert stockout_values.nunique() > 1


class TestStackingSoftWeights:
    def test_underperforming_model_keeps_nonzero_floor_weight(self):
        actuals = np.full(100, 20.0)
        sarima = actuals + 1.0
        lgbm = actuals + 50.0
        classical = actuals + 2.0

        stacker = StackingMetaLearner()
        stacker.fit(sarima, lgbm, classical, actuals)
        _, weights = stacker.predict(sarima[:5], lgbm[:5], classical[:5])

        assert weights.lgbm > 0.0
        assert weights.lgbm >= 0.05 - 1e-6
        assert weights.sarima + weights.lgbm + weights.classical == pytest.approx(1.0, abs=1e-6)


class TestForecastIntervalSanity:
    def _response(self, forecast: list[DailyForecastPoint]) -> ForecastResponse:
        return ForecastResponse(
            drug_code="PHASE0",
            center_syn_id=None,
            horizon_days=len(forecast),
            model_weights=ModelWeightBreakdown(sarima=0.5, lgbm=0.5, classical=0.0),
            forecast=forecast,
            uncertainty_note="test",
        )

    def test_detects_interval_width_cliff(self):
        forecast = [
            DailyForecastPoint(date=date(2024, 1, 1), p10=3.34, p50=8.76, p90=16.9),
            DailyForecastPoint(date=date(2024, 1, 2), p10=0.0, p50=8.0, p90=20.0),
            DailyForecastPoint(date=date(2024, 1, 3), p10=0.0, p50=7.5, p90=21.0),
        ]
        warnings = check_forecast_interval_sanity(self._response(forecast))
        assert any("P10 collapsed" in warning for warning in warnings)
        with pytest.raises(ValueError, match="P10 collapsed"):
            validate_forecast_interval_sanity(self._response(forecast))

    def test_accepts_monotonic_interval_growth(self):
        forecast = [
            DailyForecastPoint(date=date(2024, 1, idx), p10=5.0 + idx, p50=10.0 + idx, p90=15.0 + idx)
            for idx in range(1, 8)
        ]
        assert check_forecast_interval_sanity(self._response(forecast)) == []
        validate_forecast_interval_sanity(self._response(forecast))
