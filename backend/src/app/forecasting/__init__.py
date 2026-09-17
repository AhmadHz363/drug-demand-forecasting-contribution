"""Forecasting engine — demand forecasting models and ensemble pipeline."""

from app.forecasting.constants import ARTIFACTS_DIR, FORECAST_HORIZON
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
]


def __getattr__(name: str):
    if name == "correct_demand":
        from app.forecasting.censored_demand.corrector import correct_demand

        return correct_demand
    if name in {"ConformalCalibrator", "extend_tail_quantiles"}:
        from app.forecasting.ensemble import conformal

        return getattr(conformal, name)
    if name == "StackingMetaLearner":
        from app.forecasting.ensemble.stacking import StackingMetaLearner

        return StackingMetaLearner
    if name == "build_feature_matrix":
        from app.forecasting.feature_engineering.pipeline import build_feature_matrix

        return build_feature_matrix
    if name == "BaseForecastingModel":
        from app.forecasting.models.base_model import BaseForecastingModel

        return BaseForecastingModel
    if name == "SarimaModel":
        from app.forecasting.models.sarima_model import SarimaModel

        return SarimaModel
    if name == "LightGBMModel":
        from app.forecasting.models.lgbm_model import LightGBMModel

        return LightGBMModel
    if name == "ClassicalModel":
        from app.forecasting.models.classical_model import ClassicalModel

        return ClassicalModel
    if name == "walk_forward_smape":
        from app.forecasting.training.walk_forward import walk_forward_smape

        return walk_forward_smape
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
