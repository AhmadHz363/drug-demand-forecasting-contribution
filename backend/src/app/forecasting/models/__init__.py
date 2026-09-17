"""Forecasting model implementations."""

from app.forecasting.models.base_model import BaseForecastingModel
from app.forecasting.models.sarima_model import SarimaModel

__all__ = ["BaseForecastingModel", "SarimaModel", "LightGBMModel", "ClassicalModel"]


def __getattr__(name: str):
    if name == "LightGBMModel":
        from app.forecasting.models.lgbm_model import LightGBMModel

        return LightGBMModel
    if name == "ClassicalModel":
        from app.forecasting.models.classical_model import ClassicalModel

        return ClassicalModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
