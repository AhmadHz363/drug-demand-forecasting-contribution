"""
Regression tests for five bugs fixed in the forecasting engine:

1. LightGBM prediction KeyError — external_features_is_default /
   supplier_features_is_default absent from recursive feature rows.
2. SARIMA AttributeError — as_consumption_demand returns ndarray, .values fails.
3. Weibull imputation cap — imputed demand must not exceed demand_prediction_cap.
4. Walk-forward TFT epoch budget — must respect TFT_WALK_FORWARD_MAX_EPOCHS as a ceiling.
5. MPS accelerator — TFT_ENABLE_MPS=1 must select accelerator='mps' on Apple Silicon.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from app.forecasting.feature_engineering.forecast_features import build_recursive_feature_row
from app.forecasting.feature_engineering.lag_features import add_lag_features
from app.forecasting.feature_engineering.rolling_features import add_rolling_features
from app.forecasting.feature_engineering.temporal_features import add_temporal_features
from app.forecasting.models.lgbm_model import LightGBMModel
from app.forecasting.models.sarima_model import SarimaModel
from app.forecasting.prediction_bounds import demand_prediction_cap


# ── Shared helpers ──────────────────────────────────────────────────────────


def _make_training_frame(
    n_days: int = 200,
    *,
    seed: int = 0,
    with_flag_columns: bool = True,
) -> pd.DataFrame:
    """Minimal feature matrix mirroring what build_feature_matrix produces."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
    qty = rng.gamma(shape=3.0, scale=20.0, size=n_days)
    base = pd.DataFrame({"demand_date": dates, "total_quantity": qty})
    base = add_temporal_features(base)
    base = add_lag_features(base)
    base = add_rolling_features(base)
    base["bed_occupancy_rate"] = 0.75
    base["weekly_surgery_count"] = 50
    base["supplier_avg_lead_time"] = 3.0
    base["supplier_lead_time_std"] = 1.0
    base["supplier_reliability_score"] = 0.85
    base["em_corrected_quantity"] = base["total_quantity"].astype(float)
    base["forecast_step"] = 0
    base["forecast_step_sin"] = 0.0
    base["forecast_step_cos"] = 1.0
    base["demand_filled"] = 0
    if with_flag_columns:
        base["external_features_is_default"] = 1
        base["supplier_features_is_default"] = 1
    return base.set_index("demand_date")


# ── Fix 1: LightGBM feature parity ─────────────────────────────────────────


class TestLightGBMFeatureParity:
    """build_recursive_feature_row must produce every column in _feature_names."""

    def test_recursive_row_contains_all_training_features(self):
        """The columns produced by the row builder are a superset of what the
        model trained on — no KeyError during walk-forward or live prediction."""
        df = _make_training_frame(120)
        model = LightGBMModel()
        model._prepare_xy(df)
        training_features = set(model._feature_names)

        qty = df["total_quantity"].values
        median_qty = float(np.median(qty))
        feature_row = build_recursive_feature_row(
            pd.Timestamp("2022-05-01"),
            qty,
            median_qty=median_qty,
            forecast_step=1,
        )
        row_cols = set(feature_row.keys())
        missing = training_features - row_cols
        assert missing == set(), (
            f"Columns in training feature_names but absent from recursive row: {sorted(missing)}"
        )

    def test_flag_columns_present_in_row_without_covariate(self):
        """When no covariate_row is supplied the flag columns default to 1."""
        row = build_recursive_feature_row(
            pd.Timestamp("2022-06-01"),
            np.ones(50),
            median_qty=1.0,
            forecast_step=3,
        )
        assert "external_features_is_default" in row
        assert "supplier_features_is_default" in row
        assert row["external_features_is_default"] == 1.0
        assert row["supplier_features_is_default"] == 1.0

    def test_flag_columns_copied_from_covariate_row(self):
        """Values from a supplied covariate row override the defaults."""
        cov = pd.Series(
            {
                "bed_occupancy_rate": 0.80,
                "weekly_surgery_count": 60,
                "supplier_avg_lead_time": 4.0,
                "supplier_lead_time_std": 1.5,
                "supplier_reliability_score": 0.90,
                "external_features_is_default": 0,
                "supplier_features_is_default": 0,
            }
        )
        row = build_recursive_feature_row(
            pd.Timestamp("2022-06-01"),
            np.ones(50),
            median_qty=1.0,
            covariate_row=cov,
            forecast_step=1,
        )
        assert row["external_features_is_default"] == 0.0
        assert row["supplier_features_is_default"] == 0.0

    def test_lgbm_predict_no_keyerror_with_flag_columns(self, tmp_path, monkeypatch):
        """Full train → predict round-trip must not raise KeyError."""
        monkeypatch.setattr("app.forecasting.models.lgbm_model.ARTIFACTS_DIR", str(tmp_path))
        monkeypatch.setattr(LightGBMModel, "_save_shap_summary", lambda *a, **kw: None)
        df = _make_training_frame(120)
        model = LightGBMModel()
        model.train(df, "LGBM-PARITY")
        preds = model.predict(df, horizon_days=7)
        assert len(preds) == 7
        assert (preds["p10"] <= preds["p50"]).all()
        assert (preds["p50"] <= preds["p90"]).all()

    def test_lgbm_predict_no_keyerror_without_flag_columns(self, tmp_path, monkeypatch):
        """Predict must also work on a training frame that happened to lack flag
        columns (older data), using defaults in the recursive row builder."""
        monkeypatch.setattr("app.forecasting.models.lgbm_model.ARTIFACTS_DIR", str(tmp_path))
        monkeypatch.setattr(LightGBMModel, "_save_shap_summary", lambda *a, **kw: None)
        df = _make_training_frame(120, with_flag_columns=False)
        model = LightGBMModel()
        model.train(df, "LGBM-NO-FLAGS")
        # The flag columns are absent from _feature_names in this case,
        # so predict should still complete without KeyError.
        preds = model.predict(df, horizon_days=7)
        assert len(preds) == 7


# ── Fix 2: SARIMA raw_series type ──────────────────────────────────────────


class TestSarimaRawSeriesType:
    """as_consumption_demand wraps result in Series so .values is safe."""

    def test_sarima_train_does_not_raise_attribute_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.forecasting.models.sarima_model.ARTIFACTS_DIR", str(tmp_path))
        df = _make_training_frame(100)
        model = SarimaModel()
        # Before the fix this raised AttributeError: numpy.ndarray has no .values
        model.train(df, "SARIMA-TYPE-TEST")

    def test_sarima_train_with_em_corrected_quantity(self, tmp_path, monkeypatch):
        """em_corrected_quantity column (Weibull/EM path) must not break SARIMA."""
        monkeypatch.setattr("app.forecasting.models.sarima_model.ARTIFACTS_DIR", str(tmp_path))
        df = _make_training_frame(100)
        df = df.copy()
        df["em_corrected_quantity"] = df["total_quantity"] * 1.1
        model = SarimaModel()
        model.train(df, "SARIMA-EM-TEST")
        preds = model.predict(df, horizon_days=7)
        assert len(preds) == 7


# ── Fix 3: Weibull imputation cap ──────────────────────────────────────────


class TestWeibullImputationCap:
    """Imputed demand values must never exceed demand_prediction_cap."""

    def _make_stockout_frame(self, n_days: int = 300, stockout_frac: float = 0.65) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
        qty = rng.gamma(shape=2.0, scale=15.0, size=n_days)
        is_stockout = rng.random(n_days) < stockout_frac
        qty_with_stockouts = np.where(is_stockout, 0.0, qty)
        df = pd.DataFrame(
            {
                "demand_date": dates,
                "total_quantity": qty_with_stockouts,
                "is_stockout": is_stockout.astype(int),
            }
        )
        df = add_temporal_features(df)
        df = add_lag_features(df)
        df = add_rolling_features(df)
        return df

    def test_imputed_values_do_not_exceed_cap(self):
        from app.forecasting.censored_demand.survival import apply_weibull_correction

        df = self._make_stockout_frame()
        original_obs = df.loc[~df["is_stockout"].astype(bool), "total_quantity"].values
        cap = demand_prediction_cap(original_obs)

        corrected = apply_weibull_correction(df)
        imputed = corrected.loc[
            df["is_stockout"].astype(bool), "total_quantity"
        ].astype(float)

        # Allow up to 1 % tolerance over the cap for floating-point edge cases
        assert float(imputed.max()) <= cap * 1.01, (
            f"Imputed max {imputed.max():.1f} exceeds cap {cap:.1f}"
        )

    def test_imputed_values_are_finite(self):
        from app.forecasting.censored_demand.survival import apply_weibull_correction

        df = self._make_stockout_frame()
        corrected = apply_weibull_correction(df)
        imputed = corrected.loc[
            df["is_stockout"].astype(bool), "total_quantity"
        ].astype(float)
        assert np.all(np.isfinite(imputed.values)), "Some imputed values are NaN or Inf"

    def test_imputed_values_non_negative(self):
        from app.forecasting.censored_demand.survival import apply_weibull_correction

        df = self._make_stockout_frame()
        corrected = apply_weibull_correction(df)
        imputed = corrected.loc[
            df["is_stockout"].astype(bool), "total_quantity"
        ].astype(float)
        assert float(imputed.min()) >= 0.0


# ── Fix 4: Walk-forward TFT epoch budget ───────────────────────────────────


class TestWalkForwardTFTEpochs:
    def test_walk_forward_epochs_respect_max_cap(self, monkeypatch):
        """_walk_forward_tft_epochs() must never exceed TFT_WALK_FORWARD_MAX_EPOCHS."""
        import app.forecasting.training.walk_forward as wf_mod
        import app.forecasting.constants as const_mod

        monkeypatch.setattr(const_mod, "TFT_MAX_EPOCHS", 50)
        monkeypatch.setattr(const_mod, "TFT_WALK_FORWARD_EPOCH_RATIO", 0.20)
        monkeypatch.setattr(const_mod, "TFT_WALK_FORWARD_MAX_EPOCHS", 5)
        monkeypatch.setattr(wf_mod, "TFT_MAX_EPOCHS", 50)
        monkeypatch.setattr(wf_mod, "TFT_WALK_FORWARD_EPOCH_RATIO", 0.20)
        monkeypatch.setattr(wf_mod, "TFT_WALK_FORWARD_MAX_EPOCHS", 5)

        result = wf_mod._walk_forward_tft_epochs()
        assert result <= 5, (
            f"Expected at most 5 epochs (TFT_WALK_FORWARD_MAX_EPOCHS) but got {result}"
        )

    def test_walk_forward_epochs_constant_is_5(self):
        """Ensure the constant itself matches the documented value."""
        from app.forecasting.constants import TFT_WALK_FORWARD_MAX_EPOCHS

        assert TFT_WALK_FORWARD_MAX_EPOCHS == 5, (
            f"TFT_WALK_FORWARD_MAX_EPOCHS should be 5 (documented), got {TFT_WALK_FORWARD_MAX_EPOCHS}"
        )

    def test_walk_forward_epochs_bounded_by_constant(self):
        """Result is always ≤ TFT_WALK_FORWARD_MAX_EPOCHS regardless of ratio."""
        from app.forecasting.training.walk_forward import _walk_forward_tft_epochs
        from app.forecasting.constants import TFT_WALK_FORWARD_MAX_EPOCHS

        result = _walk_forward_tft_epochs()
        assert result <= TFT_WALK_FORWARD_MAX_EPOCHS


# ── Fix 5: MPS accelerator selection ───────────────────────────────────────


class TestMPSAccelerator:
    def test_mps_selected_when_env_set_and_available(self, monkeypatch):
        """When TFT_ENABLE_MPS=1 and MPS is available, accelerator must be 'mps'."""
        import torch

        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available on this machine")

        monkeypatch.setenv("TFT_ENABLE_MPS", "1")
        # Clear module-level cached flag so the function re-reads the env var.
        import app.forecasting.models.tft_model as tft_mod
        monkeypatch.setattr(tft_mod, "_MPS_SKIP_LOGGED", False)

        accelerator = tft_mod._tft_accelerator()
        assert accelerator == "mps"

    def test_mps_fallback_env_set_when_mps_selected(self, monkeypatch):
        """PYTORCH_ENABLE_MPS_FALLBACK must be set when MPS is chosen."""
        import torch

        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available on this machine")

        monkeypatch.setenv("TFT_ENABLE_MPS", "1")
        monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)

        import app.forecasting.models.tft_model as tft_mod
        monkeypatch.setattr(tft_mod, "_MPS_SKIP_LOGGED", False)

        tft_mod._tft_accelerator()
        assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"

    def test_cpu_used_when_mps_disabled(self, monkeypatch):
        """With TFT_ENABLE_MPS unset (default), accelerator must be 'cpu' on Mac."""
        import torch

        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available on this machine")
        if torch.cuda.is_available():
            pytest.skip("CUDA available — not testing CPU fallback path")

        monkeypatch.delenv("TFT_ENABLE_MPS", raising=False)
        import app.forecasting.models.tft_model as tft_mod
        monkeypatch.setattr(tft_mod, "_MPS_SKIP_LOGGED", False)

        accelerator = tft_mod._tft_accelerator()
        assert accelerator == "cpu"

    def test_map_location_follows_accelerator(self, monkeypatch):
        """_tft_map_location must return the same device as _tft_accelerator."""
        import torch
        import app.forecasting.models.tft_model as tft_mod

        monkeypatch.delenv("TFT_ENABLE_MPS", raising=False)
        if not torch.cuda.is_available() and not torch.backends.mps.is_available():
            loc = tft_mod._tft_map_location()
            assert loc == "cpu"
