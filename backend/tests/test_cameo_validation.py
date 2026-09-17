"""Unit tests for CAMEO hold-out validation metrics."""

from __future__ import annotations

import numpy as np

from app.cold_start.cameo.validation import accuracy_from_wape, wape


def test_wape_and_accuracy_helpers():
    actual = np.array([10.0, 20.0, 30.0])
    forecast = np.array([12.0, 18.0, 33.0])
    value = wape(actual, forecast)
    assert value == np.abs(actual - forecast).sum() / actual.sum()
    assert accuracy_from_wape(value) == 1 - value
