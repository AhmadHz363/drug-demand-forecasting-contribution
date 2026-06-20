"""Consumption demand normalization tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.forecasting.demand_quantity import apply_consumption_demand, as_consumption_demand


class TestAsConsumptionDemand:
    def test_negative_becomes_positive(self):
        assert as_consumption_demand(-6.0) == pytest.approx(6.0)

    def test_positive_unchanged(self):
        assert as_consumption_demand(19.0) == pytest.approx(19.0)

    def test_zero_unchanged(self):
        assert as_consumption_demand(0.0) == pytest.approx(0.0)

    def test_array(self):
        values = np.array([-3.0, 0.0, 5.0])
        np.testing.assert_allclose(as_consumption_demand(values), [3.0, 0.0, 5.0])


class TestApplyConsumptionDemand:
    def test_transforms_column(self):
        df = pd.DataFrame({"demand_date": ["2024-01-01"], "total_quantity": [-8.0]})
        out = apply_consumption_demand(df)
        assert out.iloc[0]["total_quantity"] == pytest.approx(8.0)
