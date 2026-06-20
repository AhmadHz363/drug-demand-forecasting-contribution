"""Abstract base class for forecasting models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseForecastingModel(ABC):
    @abstractmethod
    def train(self, df: pd.DataFrame, drug_code: str) -> None:
        """Train or fit the model on the provided DataFrame."""

    @abstractmethod
    def predict(self, df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
        """
        Return a DataFrame with columns: forecast_date, p10, p50, p90.
        Rows = horizon_days.
        """

    @abstractmethod
    def save(self, drug_code: str) -> str:
        """Save model artifact to ARTIFACTS_DIR. Return the file path."""

    @abstractmethod
    def load(self, drug_code: str) -> None:
        """Load model artifact from ARTIFACTS_DIR."""

    @abstractmethod
    def is_trained(self, drug_code: str) -> bool:
        """Return True if a trained artifact exists for this drug."""

    def validate_min_history(self, df: pd.DataFrame, min_days: int, model_name: str) -> None:
        if len(df) < min_days:
            raise ValueError(
                f"{model_name} requires at least {min_days} days of history. "
                f"Got {len(df)} days for this drug."
            )
