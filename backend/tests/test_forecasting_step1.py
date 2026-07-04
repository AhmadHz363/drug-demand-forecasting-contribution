"""
Step 1 Forecasting — validation suite.

Covers the implementation-guide checklist: constants, schemas, DB tables, imports.
"""

from __future__ import annotations

import os

import pytest
from pydantic import ValidationError
from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

from app.core.database import engine
from app.forecasting import constants, schemas
from app.forecasting.constants import ARTIFACTS_DIR, FORECAST_HORIZON
from app.forecasting.schemas import ForecastRequest, TrainForecastingRequest

STOCKOUT_FLAG_COLUMNS = {
    "id",
    "drug_code",
    "center_syn_id",
    "flag_date",
    "observed_quantity",
    "estimated_true_demand",
    "correction_method",
    "created_at",
}

FORECAST_RESULT_COLUMNS = {
    "id",
    "drug_code",
    "center_syn_id",
    "generated_at",
    "forecast_date",
    "p5",
    "p10",
    "p50",
    "p90",
    "p95",
    "model_weight_sarima",
    "model_weight_lgbm",
    "model_weight_tft",
}

MODEL_PERFORMANCE_COLUMNS = {
    "id",
    "drug_code",
    "model_name",
    "smape",
    "coverage_90",
    "evaluated_at",
    "mase",
    "training_run_id",
    "demand_segment",
    "data_quality_status",
    "weight_sarima",
    "weight_lgbm",
    "weight_tft",
}

ARTIFACT_SUBDIRS = ("sarima", "lgbm", "tft", "stacking", "conformal")


def _db_available() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError:
        return False


class TestForecastingImports:
    def test_constants_and_schemas_import(self):
        assert constants.FORECAST_HORIZON == 30
        assert schemas.ForecastRequest is not None

    def test_package_init_exports(self):
        from app.forecasting import ARTIFACTS_DIR as pkg_artifacts_dir
        from app.forecasting import ForecastRequest as pkg_forecast_request

        assert pkg_artifacts_dir == ARTIFACTS_DIR
        assert pkg_forecast_request is ForecastRequest


class TestForecastingSchemas:
    def test_forecast_request_defaults(self):
        req = ForecastRequest(drug_code="P325096")
        assert req.horizon_days == 7
        assert req.center_syn_id is None
        assert req.include_shap is False

    def test_forecast_request_rejects_horizon_above_max(self):
        with pytest.raises(ValidationError):
            ForecastRequest(drug_code="P325096", horizon_days=FORECAST_HORIZON + 1)

    def test_forecast_request_accepts_max_horizon(self):
        req = ForecastRequest(drug_code="P325096", horizon_days=FORECAST_HORIZON)
        assert req.horizon_days == FORECAST_HORIZON

    def test_train_request_defaults(self):
        req = TrainForecastingRequest()
        assert req.drug_codes is None
        assert req.models == ["sarima", "lgbm", "tft"]
        assert req.force_retrain is False


class TestForecastingConstants:
    def test_artifacts_dir_resolves(self):
        assert os.path.isdir(os.path.dirname(ARTIFACTS_DIR))
        assert ARTIFACTS_DIR.endswith("forecasting/artifacts")

    def test_artifact_subdirectories_creatable(self):
        for subdir in ARTIFACT_SUBDIRS:
            path = os.path.join(ARTIFACTS_DIR, subdir)
            os.makedirs(path, exist_ok=True)
            assert os.path.isdir(path)


@pytest.mark.integration
class TestForecastingDatabaseIntegration:
    @pytest.fixture(autouse=True)
    def _require_db(self):
        if not _db_available():
            pytest.skip("PostgreSQL not available")

    def test_alembic_at_head(self):
        with engine.connect() as conn:
            version = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
        assert version == "20260506_0004"

    def test_stockout_flags_table_schema(self):
        cols = {c["name"] for c in inspect(engine).get_columns("stockout_flags")}
        assert STOCKOUT_FLAG_COLUMNS == cols

    def test_forecast_results_table_schema(self):
        cols = {c["name"] for c in inspect(engine).get_columns("forecast_results")}
        assert FORECAST_RESULT_COLUMNS == cols

    def test_model_performance_table_schema(self):
        cols = {c["name"] for c in inspect(engine).get_columns("model_performance")}
        assert MODEL_PERFORMANCE_COLUMNS == cols

    def test_stockout_flags_indexes(self):
        indexes = {idx["name"] for idx in inspect(engine).get_indexes("stockout_flags")}
        assert "ix_stockout_flags_drug_code" in indexes
        assert "ix_stockout_flags_flag_date" in indexes

    def test_forecast_results_indexes(self):
        indexes = {idx["name"] for idx in inspect(engine).get_indexes("forecast_results")}
        assert "ix_forecast_results_drug_code" in indexes
        assert "ix_forecast_results_generated_at" in indexes
        assert "ix_forecast_results_forecast_date" in indexes

    def test_forecast_results_unique_constraint(self):
        uniques = inspect(engine).get_unique_constraints("forecast_results")
        names = {u["name"] for u in uniques}
        assert "uq_forecast_results_drug_center_generated_date" in names

    def test_model_performance_indexes(self):
        indexes = {idx["name"] for idx in inspect(engine).get_indexes("model_performance")}
        assert "ix_model_performance_drug_code" in indexes
