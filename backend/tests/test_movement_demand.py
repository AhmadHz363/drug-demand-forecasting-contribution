"""Unit tests for pharmacy movement taxonomy and patient demand contributions."""

from __future__ import annotations

import pytest

from app.services.movement_demand import (
    MovementRole,
    classify_movement,
    clip_daily_demand,
    normalize_movement_number,
    patient_demand_contribution,
)


class TestNormalizeMovementNumber:
    def test_int_and_float_strings(self):
        assert normalize_movement_number(5) == 5
        assert normalize_movement_number(5.0) == 5
        assert normalize_movement_number("5.0") == 5
        assert normalize_movement_number(" 6 ") == 6

    def test_nullish(self):
        assert normalize_movement_number(None) is None
        assert normalize_movement_number("") is None
        assert normalize_movement_number("nan") is None


class TestClassifyMovement:
    def test_patient_roles(self):
        assert classify_movement(5) is MovementRole.PATIENT_SALE
        assert classify_movement("6.0") is MovementRole.PATIENT_RETURN
        assert classify_movement(8) is MovementRole.CANCEL_SALE
        assert classify_movement(2) is MovementRole.TRANSFER
        assert classify_movement(30) is MovementRole.OTHER
        assert classify_movement(None) is MovementRole.UNKNOWN


class TestPatientDemandContribution:
    def test_sale_uses_absolute_quantity(self):
        # Excel-signed sale (-2) and abs-normalized DB sale (+2) both → +2
        assert patient_demand_contribution(5, -2.0) == pytest.approx(2.0)
        assert patient_demand_contribution("5.0", 2.0) == pytest.approx(2.0)

    def test_return_and_cancel_reduce_demand(self):
        assert patient_demand_contribution(6, 3.0) == pytest.approx(-3.0)
        assert patient_demand_contribution(8, -1.0) == pytest.approx(-1.0)

    def test_transfers_excluded(self):
        assert patient_demand_contribution(2, 500.0) == pytest.approx(0.0)
        assert patient_demand_contribution(7, 40.0) == pytest.approx(0.0)

    def test_legacy_unknown_keeps_signed_quantity(self):
        assert patient_demand_contribution(None, 10.0) == pytest.approx(10.0)
        assert patient_demand_contribution("", -4.0) == pytest.approx(-4.0)

    def test_daily_clip(self):
        assert clip_daily_demand(-3.0) == pytest.approx(0.0)
        assert clip_daily_demand(12.5) == pytest.approx(12.5)
