"""Tests for receipt → daily demand aggregation."""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

from app.core.database import SessionLocal, engine
from app.models.daily_drug_demand import DailyDrugDemand
from app.models.drug import Drug
from app.models.drug_receipt import DrugReceipt
from app.services.demand_aggregation import (
    aggregate_daily_demand_rows,
    drug_has_receipt_history,
    get_distinct_drug_codes_from_receipts,
    get_receipt_date_bounds,
    sync_daily_demand_from_receipts,
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
    db.query(DailyDrugDemand).filter(DailyDrugDemand.drug_code.in_(TEST_DRUGS)).delete(
        synchronize_session=False
    )
    db.query(DrugReceipt).filter(DrugReceipt.drug_code.in_(TEST_DRUGS)).delete(
        synchronize_session=False
    )
    db.query(Drug).filter(Drug.drug_code.in_(TEST_DRUGS)).delete(synchronize_session=False)
    db.commit()
    yield db
    db.query(DailyDrugDemand).filter(DailyDrugDemand.drug_code.in_(TEST_DRUGS)).delete(
        synchronize_session=False
    )
    db.query(DrugReceipt).filter(DrugReceipt.drug_code.in_(TEST_DRUGS)).delete(
        synchronize_session=False
    )
    db.query(Drug).filter(Drug.drug_code.in_(TEST_DRUGS)).delete(synchronize_session=False)
    db.commit()
    db.close()


class TestDemandAggregation:
    def test_sync_materializes_daily_demand_from_receipts(self, agg_db):
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

        synced = sync_daily_demand_from_receipts(agg_db, drug_codes=[TEST_DRUG])
        agg_db.commit()
        assert synced == 2

        rows = (
            agg_db.query(DailyDrugDemand)
            .filter(DailyDrugDemand.drug_code == TEST_DRUG)
            .order_by(DailyDrugDemand.demand_date.asc())
            .all()
        )
        assert len(rows) == 2
        assert float(rows[0].total_quantity) == pytest.approx(15.0)
        assert float(rows[1].total_quantity) == pytest.approx(7.0)

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
            date(2024, 2, 2),
        )
        assert rows == [(date(2024, 2, 1), 3.0), (date(2024, 2, 2), 4.0)]
        start, end = get_receipt_date_bounds(agg_db, TEST_DRUG)
        assert start == date(2024, 2, 1)
        assert end == date(2024, 2, 2)

    def test_patient_movements_exclude_transfers(self, agg_db):
        insert_receipt_rows(
            agg_db,
            [
                {
                    "receipt_id": "r1",
                    "line_count": 1,
                    "movement_number": "5",
                    "drug_code": TEST_DRUG,
                    "receipt_date": date(2024, 5, 1),
                    "quantity": 10.0,  # sale
                },
                {
                    "receipt_id": "r1",
                    "line_count": 2,
                    "movement_number": "6",
                    "drug_code": TEST_DRUG,
                    "receipt_date": date(2024, 5, 1),
                    "quantity": 3.0,  # return
                },
                {
                    "receipt_id": "r2",
                    "line_count": 1,
                    "movement_number": "2",
                    "drug_code": TEST_DRUG,
                    "receipt_date": date(2024, 5, 1),
                    "quantity": 500.0,  # transfer — excluded
                },
            ],
        )
        agg_db.commit()

        rows = aggregate_daily_demand_rows(
            agg_db,
            TEST_DRUG,
            date(2024, 5, 1),
            date(2024, 5, 1),
        )
        assert rows == [(date(2024, 5, 1), 7.0)]

    def test_distinct_drug_codes_from_receipts(self, agg_db):
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
                    "drug_code": "AGG-TEST-002",
                    "receipt_date": date(2024, 3, 2),
                    "quantity": 2.0,
                },
            ],
        )
        agg_db.commit()

        codes = get_distinct_drug_codes_from_receipts(agg_db)
        assert TEST_DRUG in codes
        assert "AGG-TEST-002" in codes
        assert drug_has_receipt_history(agg_db, TEST_DRUG)
        assert not drug_has_receipt_history(agg_db, "MISSING-DRUG")
