"""Receipt date coercion for Lebanese hospital DD/MM/YY exports."""

from __future__ import annotations

from datetime import date

from app.services.receipt_ingestion import _coerce_date, _dedupe_key, _dedupe_records


class TestCoerceDate:
    def test_spaced_day_first_string(self):
        parsed, err = _coerce_date(" 2/01/ 24", "receipt_date")
        assert err is None
        assert parsed == date(2024, 1, 2)

    def test_does_not_swap_to_us_month_first(self):
        parsed, err = _coerce_date("13/02/23", "receipt_date")
        assert err is None
        assert parsed == date(2023, 2, 13)

    def test_iso_and_native(self):
        assert _coerce_date("2024-03-15", "receipt_date")[0] == date(2024, 3, 15)
        assert _coerce_date(date(2024, 4, 1), "receipt_date")[0] == date(2024, 4, 1)


class TestDedupeIncludesLine:
    def test_keeps_distinct_lines_same_doc_drug_date(self):
        rows = [
            {
                "receipt_id": "100",
                "line_count": 1,
                "movement_number": "5",
                "drug_code": "P1",
                "receipt_date": date(2024, 1, 1),
                "quantity": 1.0,
            },
            {
                "receipt_id": "100",
                "line_count": 2,
                "movement_number": "5",
                "drug_code": "P1",
                "receipt_date": date(2024, 1, 1),
                "quantity": 2.0,
            },
        ]
        unique, dups = _dedupe_records(rows)
        assert dups == 0
        assert len(unique) == 2
        assert _dedupe_key(rows[0]) != _dedupe_key(rows[1])

    def test_collapses_exact_duplicate_lines(self):
        row = {
            "receipt_id": "100",
            "line_count": 1,
            "movement_number": "5",
            "drug_code": "P1",
            "receipt_date": date(2024, 1, 1),
            "quantity": 1.0,
        }
        unique, dups = _dedupe_records([row, dict(row)])
        assert dups == 1
        assert len(unique) == 1
