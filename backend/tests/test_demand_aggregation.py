"""Tests for receipt → daily demand aggregation."""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

from app.core.database import SessionLocal, engine
from app.models.drug import Drug
from app.models.drug_receipt import DrugReceipt
from app.services.demand_aggregation import (
    aggregate_daily_demand_rows,
    drug_has_receipt_history,
    get_distinct_drug_codes_from_receipts,
    get_receipt_date_bounds,
)
from app.services.drug_registry import insert_receipt_rows

TEST_DRUG = "AGG-TEST-001"
TEST_DRUGS = (TEST_DRUG, "AGG-TEST-002")


def _db_available() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError:
        return False


pytestmark = pytest.mark.skipif(
    not _db_available() or "drug_receipts" not in set(inspect(engine).get_table_names()),
    reason="PostgreSQL with drug_receipts required",
)


@pytest.fixture
def agg_db():
    db = SessionLocal()
    db.query(DrugReceipt).filter(DrugReceipt.drug_code.in_(TEST_DRUGS)).delete(
        synchronize_session=False
    )
    db.query(Drug).filter(Drug.drug_code.in_(TEST_DRUGS)).delete(synchronize_session=False)
    db.commit()
    yield db
    db.query(DrugReceipt).filter(DrugReceipt.drug_code.in_(TEST_DRUGS)).delete(
        synchronize_session=False
    )
    db.query(Drug).filter(Drug.drug_code.in_(TEST_DRUGS)).delete(synchronize_session=False)
    db.commit()
    db.close()


class TestDemandAggregation:
    def test_aggregate_daily_demand_rows_from_receipts(self, agg_db):
        insert_receipt_rows(
            agg_db,
            [
                {
                    "receipt_id": "r1",
                    "drug_code": TEST_DRUG,
                    "receipt_date": date(2024, 1, 1),
                    "quantity": 10.0,
                },
                {
                    "receipt_id": "r2",
                    "drug_code": TEST_DRUG,
                    "receipt_date": date(2024, 1, 1),
                    "quantity": 5.0,
                },
                {
                    "receipt_id": "r3",
                    "drug_code": TEST_DRUG,
                    "receipt_date": date(2024, 1, 2),
                    "quantity": 7.0,
                },
            ],
        )
        agg_db.commit()

        rows = aggregate_daily_demand_rows(
            agg_db,
            TEST_DRUG,
            date(2024, 1, 1),
            date(2024, 1, 31),
        )
        assert len(rows) == 2
        assert rows[0][1] == pytest.approx(15.0)
        assert rows[1][1] == pytest.approx(7.0)

    def test_aggregate_matches_receipt_sum(self, agg_db):
        insert_receipt_rows(
            agg_db,
            [
                {
                    "receipt_id": "r1",
                    "drug_code": TEST_DRUG,
                    "receipt_date": date(2024, 2, 1),
                    "quantity": 3.0,
                },
                {
                    "receipt_id": "r2",
                    "drug_code": TEST_DRUG,
                    "receipt_date": date(2024, 2, 2),
                    "quantity": 4.0,
                },
            ],
        )
        agg_db.commit()

        rows = aggregate_daily_demand_rows(
            agg_db,
            TEST_DRUG,
            date(2024, 2, 1),
            date(2024, 2, 28),
        )
        assert len(rows) == 2
        assert sum(qty for _, qty in rows) == pytest.approx(7.0)

    def test_receipt_date_bounds_and_history(self, agg_db):
        insert_receipt_rows(
            agg_db,
            [
                {
                    "receipt_id": "r1",
                    "drug_code": TEST_DRUG,
                    "receipt_date": date(2024, 3, 1),
                    "quantity": 1.0,
                },
                {
                    "receipt_id": "r2",
                    "drug_code": TEST_DRUG,
                    "receipt_date": date(2024, 3, 10),
                    "quantity": 2.0,
                },
            ],
        )
        agg_db.commit()

        start, end = get_receipt_date_bounds(agg_db, TEST_DRUG)
        assert start == date(2024, 3, 1)
        assert end == date(2024, 3, 10)
        assert drug_has_receipt_history(agg_db, TEST_DRUG) is True

    def test_distinct_drug_codes_from_receipts(self, agg_db):
        insert_receipt_rows(
            agg_db,
            [
                {
                    "receipt_id": "r1",
                    "drug_code": TEST_DRUG,
                    "receipt_date": date(2024, 4, 1),
                    "quantity": 1.0,
                },
                {
                    "receipt_id": "r2",
                    "drug_code": "AGG-TEST-002",
                    "receipt_date": date(2024, 4, 1),
                    "quantity": 2.0,
                },
            ],
        )
        agg_db.commit()

        codes = get_distinct_drug_codes_from_receipts(agg_db)
        assert TEST_DRUG in codes
        assert "AGG-TEST-002" in codes
