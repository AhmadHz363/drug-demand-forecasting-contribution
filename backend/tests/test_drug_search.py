"""Unit tests for drug search lookup helpers."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.database import SessionLocal, engine
from app.models.drug import Drug
from app.models.drug_receipt import DrugReceipt
from app.models.forecast_result import ForecastResult
from app.services.demand_aggregation import search_receipt_drug_codes
from app.services.drug_registry import insert_receipt_rows
from app.services.forecast_drug_lookup import search_forecasted_drug_codes

TEST_DRUGS = ("SEARCH-001", "SEARCH-002", "SEARCH-ABC")


def _db_available() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError:
        return False


pytestmark = pytest.mark.skipif(
    not _db_available()
    or "drug_receipts" not in set(inspect(engine).get_table_names())
    or "forecast_results" not in set(inspect(engine).get_table_names()),
    reason="PostgreSQL with drug_receipts and forecast_results required",
)


@pytest.fixture()
def search_db():
    db = SessionLocal()
    db.query(ForecastResult).filter(
        ForecastResult.drug_code.in_(TEST_DRUGS)
    ).delete(synchronize_session=False)
    db.query(DrugReceipt).filter(DrugReceipt.drug_code.in_(TEST_DRUGS)).delete(
        synchronize_session=False
    )
    db.query(Drug).filter(Drug.drug_code.in_(TEST_DRUGS)).delete(synchronize_session=False)
    insert_receipt_rows(
        db,
        [
            {
                "receipt_id": "r1",
                "drug_code": "SEARCH-001",
                "drug_name": "Alpha Tablet",
                "receipt_date": date(2024, 1, 1),
                "quantity": 1.0,
            },
            {
                "receipt_id": "r2",
                "drug_code": "SEARCH-002",
                "drug_name": "Beta Injection",
                "receipt_date": date(2024, 1, 2),
                "quantity": 2.0,
            },
            {
                "receipt_id": "r3",
                "drug_code": "SEARCH-ABC",
                "drug_name": "Gamma Syrup",
                "receipt_date": date(2024, 1, 3),
                "quantity": 3.0,
            },
        ],
    )
    generated_at = datetime(2024, 6, 1, tzinfo=timezone.utc)
    db.bulk_insert_mappings(
        ForecastResult,
        [
            {
                "drug_code": "SEARCH-001",
                "center_syn_id": None,
                "generated_at": generated_at,
                "forecast_date": date(2024, 6, 2),
                "p10": 1.0,
                "p50": 2.0,
                "p90": 3.0,
                "model_weight_sarima": 0.3,
                "model_weight_lgbm": 0.7,
                "model_weight_classical": 0.0,
            },
            {
                "drug_code": "SEARCH-ABC",
                "center_syn_id": None,
                "generated_at": generated_at,
                "forecast_date": date(2024, 6, 2),
                "p10": 4.0,
                "p50": 5.0,
                "p90": 6.0,
                "model_weight_sarima": 0.2,
                "model_weight_lgbm": 0.8,
                "model_weight_classical": 0.0,
            },
        ],
    )
    db.commit()
    yield db
    db.query(ForecastResult).filter(
        ForecastResult.drug_code.in_(TEST_DRUGS)
    ).delete(synchronize_session=False)
    db.query(DrugReceipt).filter(DrugReceipt.drug_code.in_(TEST_DRUGS)).delete(
        synchronize_session=False
    )
    db.query(Drug).filter(Drug.drug_code.in_(TEST_DRUGS)).delete(synchronize_session=False)
    db.commit()
    db.close()


def test_search_receipt_drug_codes_by_prefix(search_db: Session):
    items, total = search_receipt_drug_codes(search_db, "SEA", page=1, page_size=10)
    codes = [row["drug_code"] for row in items]
    assert total == 3
    assert codes == ["SEARCH-001", "SEARCH-002", "SEARCH-ABC"]


def test_search_receipt_drug_codes_pagination(search_db: Session):
    items, total = search_receipt_drug_codes(search_db, "SEA", page=1, page_size=2)
    assert total == 3
    assert len(items) == 2

    items_page_2, _ = search_receipt_drug_codes(search_db, "SEA", page=2, page_size=2)
    assert len(items_page_2) == 1


def test_search_forecasted_drug_codes_list_all(search_db: Session):
    items, total = search_forecasted_drug_codes(search_db, None, page=1, page_size=10)
    assert total == 2
    assert items == ["SEARCH-001", "SEARCH-ABC"]


def test_search_forecasted_drug_codes_filter(search_db: Session):
    items, total = search_forecasted_drug_codes(search_db, "ABC", page=1, page_size=10)
    assert total == 1
    assert items == ["SEARCH-ABC"]
