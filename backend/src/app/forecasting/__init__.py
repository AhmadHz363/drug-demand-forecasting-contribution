"""Forecasting engine — demand forecasting models and ensemble pipeline."""

from app.forecasting.censored_demand.corrector import correct_demand
from app.forecasting.constants import ARTIFACTS_DIR, FORECAST_HORIZON
from app.forecasting.ensemble.conformal import ConformalCalibrator, extend_tail_quantiles
from app.forecasting.ensemble.stacking import StackingMetaLearner
from app.forecasting.feature_engineering.pipeline import build_feature_matrix
from app.forecasting.models.base_model import BaseForecastingModel
from app.forecasting.models.sarima_model import SarimaModel
from app.forecasting.schemas import (
    AttentionWeight,
    BatchForecastRequest,
    DailyForecastPoint,
    ForecastRequest,
    ForecastResponse,
    ModelWeightBreakdown,
    ShapFeature,
    TrainForecastingRequest,
    TrainStatusResponse,
)
from app.forecasting.training.walk_forward import walk_forward_smape

__all__ = [
    "ARTIFACTS_DIR",
    "FORECAST_HORIZON",
    "AttentionWeight",
    "BatchForecastRequest",
    "DailyForecastPoint",
    "ForecastRequest",
    "ForecastResponse",
    "ModelWeightBreakdown",
    "ShapFeature",
    "TrainForecastingRequest",
    "TrainStatusResponse",
    "BaseForecastingModel",
    "SarimaModel",
    "LightGBMModel",
    "TFTModel",
    "build_feature_matrix",
    "correct_demand",
    "walk_forward_smape",
    "StackingMetaLearner",
    "ConformalCalibrator",
    "extend_tail_quantiles",
]


def __getattr__(name: str):
    if name == "LightGBMModel":
        from app.forecasting.models.lgbm_model import LightGBMModel

        return LightGBMModel
    if name == "TFTModel":
        from app.forecasting.models.tft_model import TFTModel

        return TFTModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
