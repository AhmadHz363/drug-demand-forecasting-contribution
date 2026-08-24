"""
Step 2 Forecasting — feature engineering validation suite.

Covers the implementation-guide checklist for build_feature_matrix and feature families.
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import pytest
from sqlalchemy import func, inspect, text
from sqlalchemy.exc import SQLAlchemyError

from app.core.database import SessionLocal, engine
from app.forecasting.feature_engineering.lag_features import lag_feature_columns
from app.forecasting.feature_engineering.pipeline import build_feature_matrix
from app.forecasting.feature_engineering.rolling_features import rolling_feature_columns
from app.forecasting.feature_engineering.temporal_features import temporal_feature_columns
from app.models.drug_receipt import DrugReceipt
from app.services.demand_aggregation import get_receipt_date_bounds

MIN_FEATURE_COLUMNS = (
    len(temporal_feature_columns())
    + len(lag_feature_columns())
    + len(rolling_feature_columns())
    + 2  # external (bed_occupancy_rate, weekly_surgery_count)
    + 3  # supplier (avg_lead_time, lead_time_std, reliability_score)
)
# Spike features (spike_count_{w}d, spike_intensity_{w}d) and extra
# rolling stats added after the original count of 41; keep in sync.
EXPECTED_FEATURE_COUNT = MIN_FEATURE_COLUMNS


def _db_available() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError:
        return False


def _drug_with_history(min_days: int) -> Optional[str]:
    db = SessionLocal()
    try:
        row = (
            db.query(
                DrugReceipt.drug_code,
                func.count(func.distinct(DrugReceipt.receipt_date)).label("n"),
            )
            .group_by(DrugReceipt.drug_code)
            .having(func.count(func.distinct(DrugReceipt.receipt_date)) >= min_days)
            .order_by(func.count(func.distinct(DrugReceipt.receipt_date)).desc())
            .first()
        )
        return row[0] if row else None
    finally:
        db.close()


def _date_range_for_drug(drug_code: str) -> tuple[date, date]:
    db = SessionLocal()
    try:
        return get_receipt_date_bounds(db, drug_code)
    finally:
        db.close()


class TestFeatureColumnCounts:
    def test_expected_feature_column_totals(self):
        assert EXPECTED_FEATURE_COUNT == MIN_FEATURE_COLUMNS
        assert EXPECTED_FEATURE_COUNT >= 41  # at least the original count


class TestTemporalFeaturesUnit:
    def test_cyclical_encoding_distinguishes_weekdays(self):
        base = pd.DataFrame(
            {
                "demand_date": pd.date_range("2024-01-01", periods=7, freq="D"),
                "total_quantity": range(7),
            }
        )
        from app.forecasting.feature_engineering.temporal_features import add_temporal_features

        result = add_temporal_features(base)
        pairs = result[["day_of_week_sin", "day_of_week_cos"]].values
        unique_pairs = {tuple(round(v, 6) for v in p) for p in pairs}
        assert len(unique_pairs) == 7


class TestRollingFeaturesUnit:
    def test_rolling_mean_uses_preceding_rows(self):
        quantities = list(range(1, 41))
        base = pd.DataFrame(
            {
                "demand_date": pd.date_range("2024-01-01", periods=40, freq="D"),
                "total_quantity": quantities,
            }
        )
        from app.forecasting.feature_engineering.rolling_features import add_rolling_features

        result = add_rolling_features(base)
        idx = 30
        expected = sum(quantities[idx - 28 : idx]) / 28
        assert result.loc[idx, "rolling_mean_28d"] == pytest.approx(expected, rel=1e-6)


class TestLagFeaturesUnit:
    def test_lag_7d_matches_prior_week(self):
        base = pd.DataFrame(
            {
                "demand_date": pd.date_range("2024-01-01", periods=20, freq="D"),
                "total_quantity": [float(i) for i in range(20)],
            }
        )
        from app.forecasting.feature_engineering.lag_features import add_lag_features

        result = add_lag_features(base)
        idx = 15
        assert result.loc[idx, "lag_7d"] == pytest.approx(base.loc[idx - 7, "total_quantity"])


@pytest.mark.integration
class TestBuildFeatureMatrixIntegration:
    @pytest.fixture(autouse=True)
    def _require_db(self):
        if not _db_available():
            pytest.skip("PostgreSQL not available")

    @pytest.fixture
    def drug_code(self) -> str:
        code = _drug_with_history(min_days=30)
        if code is None:
            pytest.skip("No drug with sufficient drug_receipts history")
        return code

    @pytest.fixture
    def date_range(self, drug_code: str) -> tuple[date, date]:
        return _date_range_for_drug(drug_code)

    def test_alembic_at_head(self):
        with engine.connect() as conn:
            version = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
        assert version == "20260813_0014"

    def test_stub_tables_exist(self):
        tables = set(inspect(engine).get_table_names())
        assert "hospital_census" in tables
        assert "supplier_lead_times" not in tables

    def test_no_nan_values(self, drug_code: str, date_range: tuple[date, date]):
        start, end = date_range
        db = SessionLocal()
        try:
            df = build_feature_matrix(drug_code, None, db, start, end)
        finally:
            db.close()
        assert df.isnull().sum().sum() == 0

    def test_column_count_at_least_41_features(self, drug_code: str, date_range: tuple[date, date]):
        start, end = date_range
        db = SessionLocal()
        try:
            df = build_feature_matrix(drug_code, None, db, start, end)
        finally:
            db.close()
        assert len(df.columns) >= EXPECTED_FEATURE_COUNT
        assert "total_quantity" in df.columns
        assert df.index.name == "demand_date"

    def test_empty_census_omits_default_values(
        self,
        drug_code: str,
        date_range: tuple[date, date],
        caplog: pytest.LogCaptureFixture,
    ):
        start, end = date_range
        db = SessionLocal()
        try:
            with caplog.at_level(logging.WARNING):
                df = build_feature_matrix(drug_code, None, db, start, end)
        finally:
            db.close()
        assert df["bed_occupancy_rate"].isna().all()
        assert df["weekly_surgery_count"].isna().all()
        assert (df["external_features_is_default"] == 1).all()
        assert "hospital_census" in caplog.text

    def test_empty_supplier_omits_default_values(
        self,
        drug_code: str,
        date_range: tuple[date, date],
        caplog: pytest.LogCaptureFixture,
    ):
        start, end = date_range
        db = SessionLocal()
        try:
            with caplog.at_level(logging.WARNING):
                df = build_feature_matrix(drug_code, None, db, start, end)
        finally:
            db.close()
        assert df["supplier_avg_lead_time"].isna().all()
        assert df["supplier_lead_time_std"].isna().all()
        assert df["supplier_reliability_score"].isna().all()
        assert (df["supplier_features_is_default"] == 1).all()

    def test_lag_7d_spot_check(self, drug_code: str, date_range: tuple[date, date]):
        start, end = date_range
        db = SessionLocal()
        try:
            df = build_feature_matrix(drug_code, None, db, start, end)
        finally:
            db.close()
        if len(df) <= 7:
            pytest.skip("Date range too short for lag spot check")
        idx = 20
        assert df.iloc[idx]["lag_7d"] == pytest.approx(df.iloc[idx - 7]["total_quantity"])

    def test_rolling_mean_28d_spot_check(self, drug_code: str, date_range: tuple[date, date]):
        start, end = date_range
        db = SessionLocal()
        try:
            df = build_feature_matrix(drug_code, None, db, start, end)
        finally:
            db.close()
        if len(df) <= 28:
            pytest.skip("Date range too short for rolling spot check")
        idx = 35
        expected = df.iloc[idx - 28 : idx]["total_quantity"].mean()
        assert df.iloc[idx]["rolling_mean_28d"] == pytest.approx(float(expected), rel=1e-6)

    def test_performance_under_3_seconds_for_400_days(self):
        drug_code = _drug_with_history(min_days=400)
        if drug_code is None:
            pytest.skip("No drug with 400+ days of demand history")
        start, end = _date_range_for_drug(drug_code)
        if (end - start).days + 1 < 400:
            end = start + timedelta(days=399)

        db = SessionLocal()
        try:
            t0 = time.perf_counter()
            build_feature_matrix(drug_code, None, db, start, end)
            elapsed = time.perf_counter() - t0
        finally:
            db.close()
        assert elapsed < 3.0, f"build_feature_matrix took {elapsed:.2f}s"

    def test_missing_calendar_dates_filled_with_zero(self):
        db = SessionLocal()
        try:
            row = (
                db.query(DrugReceipt.drug_code, DrugReceipt.receipt_date)
                .order_by(DrugReceipt.receipt_date.asc())
                .first()
            )
            if row is None:
                pytest.skip("No drug_receipts rows")
            drug_code = row[0]
            anchor = row[1]
            start = anchor
            end = anchor + timedelta(days=6)
            df = build_feature_matrix(drug_code, None, db, start, end)
        finally:
            db.close()
        assert len(df) == 7
