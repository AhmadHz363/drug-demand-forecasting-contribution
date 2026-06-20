"""Forecasting model implementations."""

from app.forecasting.models.base_model import BaseForecastingModel
from app.forecasting.models.sarima_model import SarimaModel

__all__ = ["BaseForecastingModel", "SarimaModel", "LightGBMModel", "TFTModel"]


def __getattr__(name: str):
    if name == "LightGBMModel":
        from app.forecasting.models.lgbm_model import LightGBMModel

        return LightGBMModel
    if name == "TFTModel":
        from app.forecasting.models.tft_model import TFTModel

        return TFTModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
