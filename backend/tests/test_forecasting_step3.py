"""
Step 3 Forecasting — censored demand correction validation suite.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional
from unittest.mock import patch

import pandas as pd
import pytest
from sqlalchemy import func, text
from sqlalchemy.exc import SQLAlchemyError

from app.core.database import SessionLocal, engine
from app.forecasting.censored_demand.corrector import correct_demand
from app.forecasting.censored_demand.detector import detect_stockout_windows
from app.forecasting.censored_demand.em_sarima import em_corrected_series
from app.forecasting.censored_demand.survival import apply_weibull_correction
from app.forecasting.censored_demand.tobit import apply_tobit_correction
from app.forecasting.feature_engineering.pipeline import build_feature_matrix
from app.forecasting.feature_engineering.rolling_features import add_rolling_features
from app.forecasting.feature_engineering.temporal_features import add_temporal_features
from app.models.drug_receipt import DrugReceipt
from app.models.stockout_flag import StockoutFlag
from app.services.demand_aggregation import get_receipt_date_bounds


def _db_available() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError:
        return False


def _synthetic_feature_frame(quantities: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=len(quantities), freq="D")
    base = pd.DataFrame({"demand_date": dates.date, "total_quantity": quantities})
    base = add_temporal_features(base)
    base = add_rolling_features(base)
    return base.set_index("demand_date")


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


class TestDetectStockoutWindows:
    def test_isolated_zero_day_not_flagged(self):
        quantities = [5.0] * 5 + [0.0] + [8.0] * 5 + [0.0] * 20
        df = _synthetic_feature_frame(quantities)
        df = df.reset_index()
        result = detect_stockout_windows(df, "TEST-DRUG", db_session=None)
        assert not result.loc[5, "is_stockout"]
        assert not result.loc[0, "is_stockout"]
        assert not result["is_stockout"].any()

    def test_short_zero_run_not_flagged(self):
        quantities = [10.0] * 5 + [0.0] * 20
        df = _synthetic_feature_frame(quantities)
        df = df.reset_index()
        result = detect_stockout_windows(df, "TEST-DRUG", db_session=None)
        assert not result["is_stockout"].any()

    def test_long_zero_run_flagged(self):
        quantities = [10.0] * 5 + [0.0] * 50 + [12.0] * 5
        df = _synthetic_feature_frame(quantities)
        df = df.reset_index()
        result = detect_stockout_windows(df, "TEST-DRUG", db_session=None)
        flagged = result.loc[5:54, "is_stockout"]
        assert flagged.all()
        assert not result.loc[0, "is_stockout"]
        assert not result.iloc[-1]["is_stockout"]


class TestTobitCorrection:
    def test_imputed_values_exceed_zero(self):
        quantities = [10.0] * 10 + [0.0] * 50 + [12.0] * 10
        df = _synthetic_feature_frame(quantities).reset_index()
        df = detect_stockout_windows(df, "TEST-DRUG", db_session=None)
        corrected = apply_tobit_correction(df)
        stockout_rows = corrected.loc[corrected["is_stockout"]]
        assert not stockout_rows.empty
        assert (stockout_rows["total_quantity"] > 0).all()
        assert (stockout_rows["correction_method"] == "tobit").all()


class TestWeibullCorrection:
    def test_imputed_values_positive(self):
        quantities = [10.0] * 5 + [0.0] * 50 + [12.0] * 5
        df = _synthetic_feature_frame(quantities).reset_index()
        df = detect_stockout_windows(df, "TEST-DRUG", db_session=None)
        corrected = apply_weibull_correction(df)
        stockout_rows = corrected.loc[corrected["is_stockout"]]
        assert not stockout_rows.empty
        assert (stockout_rows["total_quantity"] > 0).all()

    def test_missing_lifelines_raises_import_error(self):
        quantities = [10.0] * 5 + [0.0] * 50 + [12.0] * 5
        df = _synthetic_feature_frame(quantities).reset_index()
        df = detect_stockout_windows(df, "TEST-DRUG", db_session=None)

        import builtins

        real_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "lifelines":
                raise ImportError("blocked for test")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=blocked_import):
            with pytest.raises(ImportError, match="requirements.txt"):
                apply_weibull_correction(df)


class TestEmCorrectedSeries:
    def test_converges_within_max_iterations(self, caplog):
        quantities = [10.0] * 15 + [0.0] * 50 + [12.0] * 15
        df = _synthetic_feature_frame(quantities).reset_index()
        df = detect_stockout_windows(df, "TEST-DRUG", db_session=None)
        with caplog.at_level(logging.INFO):
            series = em_corrected_series(df, max_iterations=20)
        assert len(series) == len(df)
        assert (series.loc[df["is_stockout"]] > 0).all()
        assert any("EM-SARIMA" in record.message for record in caplog.records)


class TestCorrectDemandUnit:
    def test_zero_stockout_rate_unchanged_quantities(self):
        quantities = [float(i + 1) for i in range(30)]
        df = _synthetic_feature_frame(quantities)
        original = df["total_quantity"].copy()
        db = SessionLocal()
        try:
            result = correct_demand("NO-STOCKOUT", None, db, df)
        finally:
            db.rollback()
            db.close()
        pd.testing.assert_series_equal(
            result["total_quantity"].astype(float),
            original.astype(float),
            check_names=False,
        )


@pytest.mark.integration
class TestCorrectDemandIntegration:
    @pytest.fixture(autouse=True)
    def _require_db(self):
        if not _db_available():
            pytest.skip("PostgreSQL not available")

    @pytest.fixture
    def drug_code(self) -> str:
        code = _drug_with_history(min_days=60)
        if code is None:
            pytest.skip("No drug with sufficient demand history")
        return code

    def test_correct_demand_writes_stockout_flags(self, drug_code: str):
        db = SessionLocal()
        bounds = get_receipt_date_bounds(db, drug_code)
        start, end = bounds
        if (end - start).days > 120:
            start = end - timedelta(days=120)

        before = (
            db.query(func.count(StockoutFlag.id))
            .filter(StockoutFlag.drug_code == drug_code)
            .scalar()
        )
        try:
            features = build_feature_matrix(drug_code, None, db, start, end)
            corrected = correct_demand(drug_code, None, db, features)
            db.commit()
            after = (
                db.query(func.count(StockoutFlag.id))
                .filter(StockoutFlag.drug_code == drug_code)
                .scalar()
            )
            stockouts = int(corrected["is_stockout"].sum()) if "is_stockout" in corrected.columns else 0
            if stockouts > 0:
                assert after >= before + stockouts
                assert "em_corrected_quantity" in corrected.columns
        finally:
            db.rollback()
            db.close()

    def test_full_pipeline_no_nan_after_correction(self, drug_code: str):
        db = SessionLocal()
        try:
            start, end = get_receipt_date_bounds(db, drug_code)
            if (end - start).days > 90:
                start = end - timedelta(days=90)
            features = build_feature_matrix(drug_code, None, db, start, end)
            corrected = correct_demand(drug_code, None, db, features)
            assert corrected.isnull().sum().sum() == 0
        finally:
            db.rollback()
            db.close()
