"""
Step 1 Cold Start — full validation suite.

Covers the implementation-guide checklist plus real-data checks using
sample_drug_receipts_100.csv (first 100 rows from drug_receipts).
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

from app.cold_start.catalog_adapter import drug_catalog_to_metadata
from app.cold_start.receipt_adapter import infer_metadata_from_receipt
from app.cold_start.constants import ARTIFACTS_DIR, METADATA_INPUT_DIM
from app.cold_start.drug_metadata_encoder import encode_drug_metadata
from app.cold_start.schemas import (
    ColdStartPredictRequest,
    DrugMetadataInput,
    PharmacistEstimate,
)
from app.core.database import SessionLocal, engine
from app.models.drug_catalog import DrugCatalog
from app.models.drug_receipt import DrugReceipt

SAMPLE_CSV = Path(__file__).resolve().parent / "fixtures" / "sample_drug_receipts_100.csv"

DRUG_CATALOG_COLUMNS = {
    "id",
    "drug_code",
    "drug_name",
    "therapeutic_class",
    "atc_category",
    "pharmaceutical_form",
    "ven_class",
    "abc_class",
    "unit_price_tier",
    "requires_refrigeration",
    "is_controlled_substance",
    "average_shelf_life_days",
    "route_of_administration",
    "created_at",
    "updated_at",
}

CEFTRIAXONE_VECTOR = [
    0.0,
    0.36,
    0.1818,
    1.0,
    1.0,
    0.75,
    0.0,
    0.0,
    0.39,
    0.125,
    1.0,
    1.0,
]


def _sample_drug(**overrides) -> DrugMetadataInput:
    base = dict(
        drug_code="NEW-001",
        drug_name="Ceftriaxone 1g",
        therapeutic_class="antibiotic",
        atc_category="J01",
        pharmaceutical_form="injection",
        ven_class="V",
        abc_class="A",
        unit_price_tier=4,
        requires_refrigeration=False,
        is_controlled_substance=False,
        average_shelf_life_days=730,
        route_of_administration="iv",
    )
    base.update(overrides)
    return DrugMetadataInput(**base)


def _db_available() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError:
        return False


def infer_catalog_metadata(drug_code: str, drug_name: str, unit_price: float | None) -> DrugMetadataInput:
    """Backward-compatible alias used by Step 1 tests."""
    return infer_metadata_from_receipt(drug_code, drug_name, unit_price)


# --- Sample CSV integrity (real export) ---


class TestSampleReceiptsCsv:
    def test_fixture_exists(self):
        assert SAMPLE_CSV.is_file()

    def test_row_count_is_100(self, sample_receipts_rows: list[dict[str, str]]):
        assert len(sample_receipts_rows) == 100

    def test_required_columns_present(self, sample_receipts_rows: list[dict[str, str]]):
        required = {
            "id",
            "receipt_id",
            "receipt_date",
            "drug_code",
            "drug_name",
            "quantity",
        }
        assert required.issubset(sample_receipts_rows[0].keys())

    def test_core_fields_not_null(self, sample_receipts_rows: list[dict[str, str]]):
        for row in sample_receipts_rows:
            for col in ("receipt_id", "drug_code", "receipt_date"):
                assert row.get(col), col

    def test_unique_drug_codes_in_sample(self, unique_drugs_from_sample: list[dict[str, str]]):
        assert len(unique_drugs_from_sample) >= 10
        codes = [r["drug_code"] for r in unique_drugs_from_sample]
        assert len(codes) == len(set(codes))

    def test_first_row_matches_known_export(self, sample_receipts_rows: list[dict[str, str]]):
        row = sample_receipts_rows[0]
        assert row["id"] == "1"
        assert row["drug_code"] == "P325096"
        assert row["drug_name"] == "METAPRO 40MG"
        assert row["receipt_date"] == "2024-01-01"


# --- Encoder (Step 1 checklist) ---


class TestDrugMetadataEncoder:
    def test_returns_exactly_12_floats(self):
        vec = encode_drug_metadata(_sample_drug())
        assert len(vec) == METADATA_INPUT_DIM == 12

    def test_all_values_in_unit_interval(self):
        vec = encode_drug_metadata(_sample_drug())
        assert all(0.0 <= v <= 1.0 for v in vec)

    def test_ceftriaxone_reference_vector(self):
        vec = encode_drug_metadata(_sample_drug())
        rounded = [round(v, 4) for v in vec]
        assert rounded == CEFTRIAXONE_VECTOR

    def test_deterministic_encoding(self):
        drug = _sample_drug()
        assert encode_drug_metadata(drug) == encode_drug_metadata(drug)

    def test_unknown_therapeutic_class_falls_back_to_other(self):
        drug = _sample_drug(therapeutic_class="exotic_unknown_xyz")
        vec = encode_drug_metadata(drug)
        assert len(vec) == 12
        assert all(0.0 <= v <= 1.0 for v in vec)
        # "other" is index 12 -> normalized 12/12 = 1.0
        assert vec[0] == 1.0

    def test_unknown_pharmaceutical_form_falls_back(self):
        drug = _sample_drug(pharmaceutical_form="unknown_form_xyz")
        vec = encode_drug_metadata(drug)
        assert len(vec) == 12

    def test_unknown_route_falls_back(self):
        drug = _sample_drug(route_of_administration="unknown_route_xyz")
        vec = encode_drug_metadata(drug)
        assert len(vec) == 12

    def test_case_insensitive_ven_abc(self):
        drug = _sample_drug(ven_class="v", abc_class="a")
        vec = encode_drug_metadata(drug)
        assert vec[3] == 1.0
        assert vec[4] == 1.0

    def test_encode_all_unique_drugs_from_sample(self, unique_drugs_from_sample: list[dict[str, str]]):
        for row in unique_drugs_from_sample:
            unit_price = row.get("unit_price")
            meta = infer_catalog_metadata(
                row["drug_code"],
                row["drug_name"],
                float(unit_price) if unit_price else None,
            )
            vec = encode_drug_metadata(meta)
            assert len(vec) == 12
            assert all(0.0 <= v <= 1.0 for v in vec), row["drug_code"]


# --- Pydantic schemas ---


class TestColdStartSchemas:
    def test_predict_request_valid(self):
        req = ColdStartPredictRequest(
            drug_metadata=_sample_drug(),
            pharmacist_estimate=PharmacistEstimate(weekly_units=200, confidence=0.8),
            forecast_horizon_days=7,
        )
        assert req.forecast_horizon_days == 7

    def test_predict_request_without_pharmacist_estimate(self):
        req = ColdStartPredictRequest(drug_metadata=_sample_drug())
        assert req.pharmacist_estimate is None

    def test_invalid_unit_price_tier_rejected(self):
        with pytest.raises(ValidationError):
            DrugMetadataInput(**{**_sample_drug().model_dump(), "unit_price_tier": 0})

    def test_invalid_weekly_units_rejected(self):
        with pytest.raises(ValidationError):
            PharmacistEstimate(weekly_units=0)


# --- Constants & artifacts ---


class TestColdStartConstants:
    def test_artifacts_dir_creatable(self):
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        assert os.path.isdir(ARTIFACTS_DIR)


# --- Database integration (live DB + CSV cross-check) ---


@pytest.mark.integration
class TestDatabaseIntegration:
    @pytest.fixture(autouse=True)
    def _require_db(self):
        if not _db_available():
            pytest.skip("PostgreSQL not available")

    def test_alembic_at_head(self):
        with engine.connect() as conn:
            version = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
        assert version == "20260505_0002"

    def test_drug_catalog_table_schema(self):
        cols = {c["name"] for c in inspect(engine).get_columns("drug_catalog")}
        assert DRUG_CATALOG_COLUMNS == cols

    def test_drug_receipts_has_at_least_sample_size(self):
        with engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM drug_receipts")).scalar()
        assert count >= 100

    def test_csv_first_row_exists_in_database(self, sample_receipts_rows: list[dict[str, str]]):
        expected = sample_receipts_rows[0]
        db = SessionLocal()
        try:
            row = db.get(DrugReceipt, int(expected["id"]))
            assert row is not None
            assert row.drug_code == str(expected["drug_code"])
            assert row.receipt_id == str(expected["receipt_id"])
            assert row.drug_name == str(expected["drug_name"])
            assert row.receipt_date == date.fromisoformat(str(expected["receipt_date"]))
        finally:
            db.close()

    def test_sample_drug_codes_exist_in_receipts(self, unique_drugs_from_sample: list[dict[str, str]]):
        codes = {r["drug_code"] for r in unique_drugs_from_sample}
        db = SessionLocal()
        try:
            found = {
                r[0]
                for r in db.query(DrugReceipt.drug_code)
                .filter(DrugReceipt.drug_code.in_(codes))
                .distinct()
                .all()
            }
            assert codes.issubset(found), f"missing: {codes - found}"
        finally:
            db.close()

    def test_catalog_insert_encode_roundtrip_rollback(self, unique_drugs_from_sample: list[dict[str, str]]):
        """Insert catalog rows for sample drugs, encode via adapter, then rollback."""
        db = SessionLocal()
        encoded_count = 0
        test_codes: list[str] = []
        try:
            for row in unique_drugs_from_sample[:10]:
                unit_price = row.get("unit_price")
                meta = infer_catalog_metadata(
                    row["drug_code"],
                    row["drug_name"],
                    float(unit_price) if unit_price else None,
                )
                test_code = f"TEST-{meta.drug_code}"
                test_codes.append(test_code)
                catalog = DrugCatalog(
                    drug_code=test_code,
                    drug_name=meta.drug_name,
                    therapeutic_class=meta.therapeutic_class,
                    atc_category=meta.atc_category,
                    pharmaceutical_form=meta.pharmaceutical_form,
                    ven_class=meta.ven_class,
                    abc_class=meta.abc_class,
                    unit_price_tier=meta.unit_price_tier,
                    requires_refrigeration=meta.requires_refrigeration,
                    is_controlled_substance=meta.is_controlled_substance,
                    average_shelf_life_days=meta.average_shelf_life_days,
                    route_of_administration=meta.route_of_administration,
                )
                db.add(catalog)
                db.flush()
                vec = encode_drug_metadata(drug_catalog_to_metadata(catalog))
                assert len(vec) == 12
                encoded_count += 1
            assert encoded_count == 10
            db.rollback()
            remaining = (
                db.query(DrugCatalog).filter(DrugCatalog.drug_code.in_(test_codes)).count()
            )
            assert remaining == 0
        finally:
            db.close()

    def test_drug_catalog_separate_from_receipts(self):
        with engine.connect() as conn:
            receipts = conn.execute(text("SELECT COUNT(*) FROM drug_receipts")).scalar()
            catalog = conn.execute(text("SELECT COUNT(*) FROM drug_catalog")).scalar()
        assert receipts >= 100
        assert catalog >= 0
