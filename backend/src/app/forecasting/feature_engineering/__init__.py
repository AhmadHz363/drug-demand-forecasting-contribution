"""Feature engineering — transforms daily demand into model-ready matrices."""

from app.forecasting.feature_engineering.pipeline import build_feature_matrix

__all__ = ["build_feature_matrix"]
