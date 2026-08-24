"""SHIELD-XR hospital drug demand forecasting engine."""

from app.forecasting.shield_xr.trainer import ShieldXRTrainer

__all__ = ["ShieldXRTrainer", "ShieldXRForecaster"]


def __getattr__(name: str):
    if name == "ShieldXRForecaster":
        from app.forecasting.shield_xr.forecaster import ShieldXRForecaster

        return ShieldXRForecaster
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
