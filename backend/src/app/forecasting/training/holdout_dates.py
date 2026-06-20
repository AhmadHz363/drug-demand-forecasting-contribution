"""Smart hold-out date suggestions from receipt history bounds."""

from __future__ import annotations

import calendar
from datetime import date, timedelta

from app.forecasting.schemas import HoldoutDateDefaults, HoldoutDateSuggestions

MIN_TRAIN_DAYS = 90
MIN_TEST_DAYS = 28
DEFAULT_TEST_DAYS = 90


def _month_end(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def _iter_month_ends(start: date, end: date) -> list[date]:
    """Month-end dates from start through end (inclusive), at most one per month."""
    if start > end:
        return []

    cursor = date(start.year, start.month, 1)
    options: list[date] = []
    while cursor <= end:
        last = _month_end(cursor.year, cursor.month)
        if start <= last <= end:
            options.append(last)
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    return options


def _latest_train_end(data_start: date, data_end: date) -> date:
    return data_end - timedelta(days=MIN_TEST_DAYS)


def _earliest_train_end(data_start: date, data_end: date) -> date:
    earliest = data_start + timedelta(days=MIN_TRAIN_DAYS - 1)
    latest = _latest_train_end(data_start, data_end)
    return min(earliest, latest)


def compute_holdout_defaults(data_start: date, data_end: date) -> HoldoutDateDefaults:
    """Pick a train/test split that fits the drug's receipt history."""
    span_days = (data_end - data_start).days + 1
    test_days = min(DEFAULT_TEST_DAYS, max(MIN_TEST_DAYS, span_days // 5))

    test_end = data_end
    test_start = test_end - timedelta(days=test_days - 1)
    train_end = test_start - timedelta(days=1)

    earliest_train = _earliest_train_end(data_start, data_end)
    if train_end < earliest_train:
        train_end = earliest_train
        test_start = train_end + timedelta(days=1)
        if test_start > test_end:
            test_start = test_end

    return HoldoutDateDefaults(
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )


def compute_holdout_date_suggestions(
    data_start: date,
    data_end: date,
) -> HoldoutDateSuggestions:
    """Build default windows and selectable month-boundary suggestions."""
    defaults = compute_holdout_defaults(data_start, data_end)

    earliest_train = _earliest_train_end(data_start, data_end)
    latest_train = _latest_train_end(data_start, data_end)
    train_end_options = _iter_month_ends(earliest_train, latest_train)
    if not train_end_options:
        train_end_options = [defaults.train_end]

    earliest_test_start = defaults.train_end + timedelta(days=1)
    latest_test_start = data_end - timedelta(days=MIN_TEST_DAYS - 1)
    if latest_test_start < earliest_test_start:
        latest_test_start = earliest_test_start

    test_start_options: list[date] = []
    seen: set[date] = set()
    for month_end in _iter_month_ends(earliest_test_start, latest_test_start):
        candidate = date(month_end.year, month_end.month, 1)
        if candidate < earliest_test_start:
            candidate = earliest_test_start
        if candidate <= latest_test_start and candidate not in seen:
            test_start_options.append(candidate)
            seen.add(candidate)
    if earliest_test_start not in seen:
        test_start_options.insert(0, earliest_test_start)
    if not test_start_options:
        test_start_options = [defaults.test_start]

    test_end_options = _iter_month_ends(defaults.test_start, data_end)
    if data_end not in test_end_options:
        test_end_options.append(data_end)
    test_end_options = sorted(set(test_end_options))

    return HoldoutDateSuggestions(
        data_start=data_start,
        data_end=data_end,
        defaults=defaults,
        train_end_options=train_end_options,
        test_start_options=test_start_options,
        test_end_options=test_end_options,
    )
