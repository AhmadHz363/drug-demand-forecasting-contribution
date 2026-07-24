"""
Phase 3 forecasting — data quality gating, drift detection, newsvendor
decisioning, and monitoring schemas.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.forecasting.censored_demand.corrector import assert_no_degenerate_stockout_imputation
from app.forecasting.constants import (
    DATA_QUALITY_MAX_EXTERNAL_DEFAULT_RATE,
    DRIFT_MASE_DEGRADATION_PCT,
    DRIFT_SMAPE_DEGRADATION_PCT,
    NEWSVENDOR_COST_RATIO,
)
from app.forecasting.data_quality import (
    QUALITY_FLAGGED,
    QUALITY_OK,
    QUALITY_REJECTED,
    assess_series_quality,
)
from app.forecasting.drift_detection import assess_metric_drift
from app.forecasting.inference.forecaster import (
    check_forecast_interval_sanity,
    validate_forecast_quantiles,
)
from app.forecasting.newsvendor import (
    interpolate_forecast_quantile,
    operating_quantile,
    recommended_quantities,
    resolve_ven_class,
)
from app.forecasting.schemas import (
    DailyForecastPoint,
    ForecastResponse,
    InferenceHealth,
    ModelWeightBreakdown,
)


class TestDataQualityGating:
    def test_clean_series_passes(self):
        df = pd.DataFrame(
            {
                "total_quantity": np.linspace(5, 12, 90),
                "external_features_is_default": np.zeros(90, dtype=int),
                "supplier_features_is_default": np.zeros(90, dtype=int),
            }
        )
        report = assess_series_quality(df, "DRUG-A")
        assert report.status == QUALITY_OK

    def test_high_external_defaults_flagged(self):
        df = pd.DataFrame(
            {
                "total_quantity": np.ones(90),
                "external_features_is_default": np.ones(90, dtype=int),
                "supplier_features_is_default": np.zeros(90, dtype=int),
            }
        )
        report = assess_series_quality(df, "DRUG-B")
        assert report.status == QUALITY_FLAGGED
        assert report.external_default_rate > DATA_QUALITY_MAX_EXTERNAL_DEFAULT_RATE

    def test_degenerate_imputation_rejected(self):
        df = pd.DataFrame(
            {
                "total_quantity": [10.0, 10.0, 10.0, 10.0, 5.0],
                "is_stockout": [True, True, True, True, False],
            }
        )
        with pytest.raises(ValueError):
            assert_no_degenerate_stockout_imputation(df)

        report = assess_series_quality(df, "DRUG-C")
        assert report.status == QUALITY_REJECTED


class TestDriftDetection:
    def test_smape_degradation_detected(self):
        drift = assess_metric_drift(20.0, 30.0, None, None)
        assert drift.smape_degraded
        assert drift.smape_delta_pct == pytest.approx(50.0)

    def test_stable_metrics_not_flagged(self):
        drift = assess_metric_drift(20.0, 21.0, 0.8, 0.82)
        assert not drift.has_drift

    def test_thresholds_configured(self):
        assert DRIFT_SMAPE_DEGRADATION_PCT > 0
        assert DRIFT_MASE_DEGRADATION_PCT > 0


class TestNewsvendorDecisioning:
    def test_vital_drugs_target_high_quantile(self):
        assert operating_quantile("V") > operating_quantile("N")

    def test_operating_quantile_formula(self):
        ratio = NEWSVENDOR_COST_RATIO["E"]
        assert operating_quantile("E") == pytest.approx(ratio / (1.0 + ratio))

    def test_resolve_unknown_ven_defaults_to_n(self):
        assert resolve_ven_class(None) == "N"
        assert resolve_ven_class("x") == "N"

    def test_recommended_quantities_monotonic_with_criticality(self):
        p10 = np.array([1.0, 1.0])
        p50 = np.array([5.0, 5.0])
        p90 = np.array([10.0, 10.0])
        _, vital = recommended_quantities(p10, p50, p90, "V")
        _, normal = recommended_quantities(p10, p50, p90, "N")
        assert float(vital[0]) > float(normal[0])

    def test_interpolate_between_quantiles(self):
        value = interpolate_forecast_quantile(10.0, 20.0, 30.0, 0.50)
        assert value == pytest.approx(20.0)


class TestForecastRegressionGuards:
    def _sample_response(self, *, p10_cliff: bool = False) -> ForecastResponse:
        points = []
        for day in range(1, 8):
            p10 = 0.0 if p10_cliff and day > 1 else 2.0
            points.append(
                DailyForecastPoint(
                    date=date(2026, 1, day),
                    p10=p10,
                    p50=5.0,
                    p90=8.0,
                )
            )
        return ForecastResponse(
            drug_code="E2E-TEST",
            center_syn_id=None,
            horizon_days=7,
            model_weights=ModelWeightBreakdown(sarima=0.4, lgbm=0.4, classical=0.2),
            forecast=points,
            uncertainty_note="test",
            inference_health=InferenceHealth(
                demand_segment="smooth",
                used_stacking=True,
                used_conformal=True,
                used_spread_fallback=False,
            ),
        )

    def test_quantile_ordering_validated(self):
        response = self._sample_response()
        validate_forecast_quantiles(response)

    def test_weights_sum_to_one(self):
        response = self._sample_response()
        total = (
            response.model_weights.sarima
            + response.model_weights.lgbm
            + response.model_weights.classical
        )
        assert total == pytest.approx(1.0)

    def test_interval_sanity_catches_p10_cliff(self):
        response = self._sample_response(p10_cliff=True)
        warnings = check_forecast_interval_sanity(response)
        assert any("P10 collapsed" in w for w in warnings)

    def test_smooth_intervals_no_warnings(self):
        response = self._sample_response()
        assert check_forecast_interval_sanity(response) == []
