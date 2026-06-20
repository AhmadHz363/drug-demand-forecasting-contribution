"""Calendar and holiday-based temporal features."""

from __future__ import annotations

import math
from datetime import date

import holidays
import numpy as np
import pandas as pd

_LEBANON_HOLIDAYS = holidays.Lebanon(years=range(2018, 2030))
_SORTED_HOLIDAY_DATES = sorted(_LEBANON_HOLIDAYS.keys())


def _days_to_next_holiday(d: date, sorted_holidays: list[date]) -> int:
    for h in sorted_holidays:
        if h >= d:
            delta = (h - d).days
            return min(delta, 30)
    return 30


def _days_since_last_holiday(d: date, sorted_holidays: list[date]) -> int:
    for h in reversed(sorted_holidays):
        if h <= d:
            delta = (d - h).days
            return min(delta, 30)
    return 30


def _cyclical_encode(values: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    angle = 2 * math.pi * values / period
    return np.sin(angle), np.cos(angle)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append calendar features to a DataFrame with a ``demand_date`` column.

    Requires columns: demand_date
    """
    out = df.copy()
    dates = pd.to_datetime(out["demand_date"])

    day_of_week = dates.dt.dayofweek
    out["day_of_week"] = day_of_week
    out["day_of_week_sin"], out["day_of_week_cos"] = _cyclical_encode(day_of_week, 7)

    week_of_year = dates.dt.isocalendar().week.astype(int)
    out["week_of_year"] = week_of_year
    out["week_of_year_sin"], out["week_of_year_cos"] = _cyclical_encode(week_of_year, 53)

    month = dates.dt.month
    out["month"] = month
    out["month_sin"], out["month_cos"] = _cyclical_encode(month, 12)

    out["quarter"] = dates.dt.quarter
    out["is_weekend"] = day_of_week.isin([5, 6]).astype(int)

    holiday_dates = dates.dt.date
    out["is_public_holiday"] = holiday_dates.map(lambda d: int(d in _LEBANON_HOLIDAYS))

    out["days_to_next_holiday"] = holiday_dates.map(
        lambda d: _days_to_next_holiday(d, _SORTED_HOLIDAY_DATES)
    )
    out["days_since_last_holiday"] = holiday_dates.map(
        lambda d: _days_since_last_holiday(d, _SORTED_HOLIDAY_DATES)
    )

    return out


def temporal_feature_columns() -> list[str]:
    return [
        "day_of_week",
        "day_of_week_sin",
        "day_of_week_cos",
        "week_of_year",
        "week_of_year_sin",
        "week_of_year_cos",
        "month",
        "month_sin",
        "month_cos",
        "quarter",
        "is_weekend",
        "is_public_holiday",
        "days_to_next_holiday",
        "days_since_last_holiday",
    ]
