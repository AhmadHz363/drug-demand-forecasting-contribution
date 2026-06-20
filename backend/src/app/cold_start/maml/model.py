"""LSTM base learner for MAML meta-training and fast adaptation."""

from __future__ import annotations

from datetime import date, timedelta

import torch
import torch.nn as nn

from app.cold_start.constants import (
    MAML_FORECAST_HORIZON,
    MAML_HIDDEN_DIM,
    MAML_SEQUENCE_LEN,
)


def day_features(demand_date: date, quantity: float, mean: float, std: float) -> list[float]:
    """Return (normalized_demand, day_of_week, is_holiday) in roughly [0, 1] scale."""
    norm_qty = (quantity - mean) / std if std > 0 else quantity
    dow = demand_date.weekday() / 6.0
    is_holiday = 1.0 if demand_date.weekday() >= 5 else 0.0
    return [norm_qty, dow, is_holiday]


def series_window_to_tensor(
    window: list[tuple[date, float]],
    mean: float,
    std: float,
) -> torch.Tensor:
    """Convert a date/quantity window to shape [1, seq_len, 3]."""
    features = [day_features(d, q, mean, std) for d, q in window]
    return torch.tensor([features], dtype=torch.float32)


def pad_series_left(
    series: list[tuple[date, float]],
    seq_len: int,
    fill_quantity: float,
) -> list[tuple[date, float]]:
    """Left-pad a short series to seq_len using synthetic prior days."""
    if len(series) >= seq_len:
        return series[-seq_len:]

    if not series:
        anchor = date.today()
    else:
        anchor = series[0][0]

    pad_count = seq_len - len(series)
    padded: list[tuple[date, float]] = []
    for offset in range(pad_count, 0, -1):
        padded.append((anchor - timedelta(days=offset), fill_quantity))
    padded.extend(series)
    return padded


class BaseForecaster(nn.Module):
    """
    Sequence regressor: [batch, seq_len, 3] → [batch, forecast_horizon].
    Input features per timestep: (demand_t, day_of_week, is_holiday).
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = MAML_HIDDEN_DIM,
        seq_len: int = MAML_SEQUENCE_LEN,
        forecast_horizon: int = MAML_FORECAST_HORIZON,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_dim, forecast_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return self.head(hidden[-1])
