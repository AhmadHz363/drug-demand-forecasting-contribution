"""Tests for hospital daily demand enrichment."""

from __future__ import annotations

from datetime import date

import pandas as pd

from app.services.hospital_demand_enrichment import (
    PATIENT_SALE_MOVEMENT_DESCRIPTION,
    build_enriched_daily_panel,
    parse_hospital_date,
)


class TestParseHospitalDate:
    def test_compact_day_first(self):
        assert parse_hospital_date(" 1/01/ 24") == date(2024, 1, 1)

    def test_placeholder_returns_none(self):
        assert parse_hospital_date("00/00/00") is None


class TestBuildEnrichedDailyPanel:
    def test_filters_non_sale_movements(self):
        df = pd.DataFrame(
            [
                {
                    "DATE": "1/01/24",
                    "Mov des": PATIENT_SALE_MOVEMENT_DESCRIPTION,
                    "CODE": "P001",
                    "ARTICLE": "Drug A",
                    "CAT": 100,
                    "QTY": -3,
                    "MRN": 1,
                    "AD DATE": "00/00/00",
                    "C.R": 10,
                    "C.S": 20,
                    "DR": "Dr A",
                    "AGE": "0000/00/00",
                },
                {
                    "DATE": "1/01/24",
                    "Mov des": "تـحـويـل مــن قـســم الـى",
                    "CODE": "P001",
                    "ARTICLE": "Drug A",
                    "CAT": 100,
                    "QTY": -99,
                    "MRN": 2,
                    "AD DATE": "00/00/00",
                    "C.R": 10,
                    "C.S": 20,
                    "DR": "Dr B",
                    "AGE": "0000/00/00",
                },
            ],
        )
        panel, filtered = build_enriched_daily_panel(df)
        assert filtered == 1
        assert len(panel) == 1
        row = panel.iloc[0]
        assert row["drug_code"] == "P001"
        assert row["demand"] == 3
        assert row["n_patients_drug"] == 1
        assert row["hospital_total_demand"] == 3

    def test_dense_panel_fills_zero_demand_days(self):
        df = pd.DataFrame(
            [
                {
                    "DATE": "1/01/24",
                    "Mov des": PATIENT_SALE_MOVEMENT_DESCRIPTION,
                    "CODE": "P001",
                    "ARTICLE": "Drug A",
                    "CAT": 100,
                    "QTY": -1,
                    "MRN": 1,
                    "AD DATE": "00/00/00",
                    "C.R": 10,
                    "C.S": 20,
                    "DR": "Dr A",
                    "AGE": "0000/00/00",
                },
                {
                    "DATE": "3/01/24",
                    "Mov des": PATIENT_SALE_MOVEMENT_DESCRIPTION,
                    "CODE": "P001",
                    "ARTICLE": "Drug A",
                    "CAT": 100,
                    "QTY": -2,
                    "MRN": 2,
                    "AD DATE": "00/00/00",
                    "C.R": 10,
                    "C.S": 20,
                    "DR": "Dr B",
                    "AGE": "0000/00/00",
                },
            ],
        )
        panel, _ = build_enriched_daily_panel(df)
        assert len(panel) == 3
        demands = dict(zip(panel["demand_date"], panel["demand"]))
        assert demands[date(2024, 1, 1)] == 1
        assert demands[date(2024, 1, 2)] == 0
        assert demands[date(2024, 1, 3)] == 2


class TestHospitalHeaderCanonicalization:
    def test_mov_des_alias(self):
        from app.services.hospital_receipt_ingestion import (
            _canonicalize_hospital_headers,
            _missing_required_columns,
        )

        df = pd.DataFrame(
            {
                "MOV DES": ["sale"],
                "DATE": ["1/01/24"],
                "CODE": ["P001"],
                "QTY": [-1],
            },
        )
        canonical = _canonicalize_hospital_headers(df)
        assert _missing_required_columns(canonical) == []

    def test_missing_mov_des_detected(self):
        from app.services.hospital_receipt_ingestion import (
            _canonicalize_hospital_headers,
            _missing_required_columns,
        )

        df = pd.DataFrame({"DATE": ["1/01/24"], "CODE": ["P001"], "QTY": [-1]})
        assert _missing_required_columns(_canonicalize_hospital_headers(df)) == ["Mov des"]


class TestSanitizeEnrichedRecord:
    def test_nan_integer_fields_become_null(self):
        from app.services.hospital_receipt_ingestion import _sanitize_enriched_record

        row = _sanitize_enriched_record(
            {
                "demand_date": date(2023, 1, 10),
                "drug_code": "KT00526",
                "article": "NUTRISON CONC.500ML ",
                "cat": 5050.0,
                "demand": 0,
                "n_patients_drug": 0,
                "n_doctors_drug": 0,
                "top_cr_drug": float("nan"),
                "top_cs_drug": float("nan"),
                "top_dr_drug": float("nan"),
                "n_unique_patients": 0,
                "n_admissions": 0,
                "n_unique_doctors": 0,
                "n_unique_cr": 0,
                "n_unique_cs": 0,
                "n_transactions": 0,
                "n_demand_txns": 0,
                "hospital_total_demand": 0,
                "age_mean": float("nan"),
                "age_median": float("nan"),
                "top1_cr": float("nan"),
                "top1_cr_share": 0.0,
                "top2_cr": float("nan"),
                "top2_cr_share": 0.0,
                "top3_cr": float("nan"),
                "top3_cr_share": 0.0,
            },
            source_file="hospital 2023.xlsx",
        )
        assert row["cat"] == 5050
        assert row["top1_cr"] is None
        assert row["top_dr_drug"] is None
        assert row["top_cr_drug"] is None
