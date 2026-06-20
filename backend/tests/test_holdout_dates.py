"""Tests for hold-out date suggestion helpers."""

from __future__ import annotations

from datetime import date

from app.forecasting.training.holdout_dates import (
    compute_holdout_date_suggestions,
    compute_holdout_defaults,
)


class TestHoldoutDefaults:
    def test_defaults_use_recent_test_window(self):
        data_start = date(2022, 1, 1)
        data_end = date(2026, 3, 31)
        defaults = compute_holdout_defaults(data_start, data_end)

        assert defaults.test_end == data_end
        assert defaults.test_start > defaults.train_end
        assert defaults.test_start <= defaults.test_end
        assert (defaults.test_end - defaults.test_start).days + 1 >= 28

    def test_short_history_still_splits(self):
        data_start = date(2025, 1, 1)
        data_end = date(2025, 6, 30)
        defaults = compute_holdout_defaults(data_start, data_end)

        assert defaults.train_end >= data_start
        assert defaults.test_end == data_end
        assert defaults.test_start > defaults.train_end


class TestHoldoutDateSuggestions:
    def test_suggestions_include_month_boundaries(self):
        data_start = date(2022, 1, 15)
        data_end = date(2026, 3, 31)
        suggestions = compute_holdout_date_suggestions(data_start, data_end)

        assert suggestions.data_start == data_start
        assert suggestions.data_end == data_end
        assert suggestions.defaults.train_end in suggestions.train_end_options
        assert suggestions.defaults.test_start in suggestions.test_start_options
        assert suggestions.defaults.test_end in suggestions.test_end_options
        assert suggestions.data_end in suggestions.test_end_options
