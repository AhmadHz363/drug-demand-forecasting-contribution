"""KNN bootstrap — neighbour selection with sparse demand history."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from app.cold_start.knn_bootstrap.bootstrap import MIN_DAYS_REQUIRED, compute_baseline_forecast
from app.cold_start.schemas import PharmacistEstimate


def _quantities_for_code(drug_code: str) -> list[float]:
    """Top-5 similar codes lack data; 6th has enough days."""
    if drug_code == "HAS-DATA":
        return [10.0] * MIN_DAYS_REQUIRED
    return [1.0] * max(0, MIN_DAYS_REQUIRED - 1)


@pytest.fixture
def db_session(monkeypatch):
    session = MagicMock()

    def fake_fetch(_session, drug_code: str) -> list[float]:
        return _quantities_for_code(drug_code)

    monkeypatch.setattr(
        "app.cold_start.knn_bootstrap.bootstrap._fetch_recent_quantities",
        fake_fetch,
    )
    return session


def test_uses_neighbour_beyond_top_k_when_sparse(db_session):
    neighbours = [(f"SKIP-{i}", 1.0 - i * 0.01) for i in range(5)]
    neighbours.append(("HAS-DATA", 0.5))

    forecast, used = compute_baseline_forecast(
        neighbours,
        db_session,
        horizon_days=3,
        pharmacist_estimate=None,
    )

    assert len(forecast) == 3
    assert used == [("HAS-DATA", 0.5)]


def test_pharmacist_only_when_no_eligible_neighbours(db_session):
    neighbours = [(f"SKIP-{i}", 1.0) for i in range(3)]

    forecast, used = compute_baseline_forecast(
        neighbours,
        db_session,
        horizon_days=2,
        pharmacist_estimate=PharmacistEstimate(weekly_units=70.0, confidence=0.5),
    )

    assert used == []
    assert forecast[0].p50 == pytest.approx(10.0)
    assert isinstance(forecast[0].date, date)
