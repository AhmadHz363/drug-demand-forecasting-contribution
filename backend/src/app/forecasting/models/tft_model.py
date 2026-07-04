"""Temporal Fusion Transformer forecasting model."""

from __future__ import annotations

import logging
import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd

from app.forecasting.constants import (
    ARTIFACTS_DIR,
    FORECAST_HORIZON,
    LOOKBACK_WINDOW_TFT,
    LOOKBACK_WINDOW_TFT_SHORT,
    MIN_HISTORY_DAYS_TFT,
    MIN_HISTORY_DAYS_TFT_SHORT,
    TFT_ATTENTION_HEAD_SIZE,
    TFT_BATCH_SIZE,
    TFT_ENABLE_MPS,
    TFT_DROPOUT,
    TFT_GRADIENT_CLIP_VAL,
    TFT_HIDDEN_CONTINUOUS_SIZE,
    TFT_HIDDEN_SIZE,
    TFT_LEARNING_RATE,
    TFT_MAX_EPOCHS,
    TFT_MAX_TRAIN_DAYS,
    TFT_QUANTILES,
)
from app.forecasting.prediction_bounds import demand_prediction_cap, winsorize_predictions
from app.forecasting.models.base_model import BaseForecastingModel
from app.forecasting.feature_engineering.lag_features import add_lag_features
from app.forecasting.feature_engineering.rolling_features import add_rolling_features
from app.forecasting.feature_engineering.temporal_features import (
    add_temporal_features,
    temporal_feature_columns,
)
from app.forecasting.models.feature_columns import (
    ensure_demand_date_index,
    forecast_dates_from_index,
    tft_known_reals,
    tft_static_reals,
    tft_unknown_reals,
)

logger = logging.getLogger(__name__)


def _checkpoint_path(drug_code: str) -> str:
    return os.path.join(ARTIFACTS_DIR, "tft", f"{drug_code}.ckpt")


def _dataset_path(drug_code: str) -> str:
    return os.path.join(ARTIFACTS_DIR, "tft", f"{drug_code}_dataset.pkl")


_MPS_SKIP_LOGGED = False


def _tft_accelerator() -> str:
    """
    Pick a Lightning accelerator that will not crash the API process.

    CUDA (NVIDIA) is used when available.  Apple MPS is **off by default**
    because pytorch-forecasting triggers uncatchable native buffer assertions
    on MPS that terminate the process.  Set ``TFT_ENABLE_MPS=1`` to opt in.

    The env var is read at call time (not at module-import time) so the
    server process can set it after Python has already imported this module.
    PYTORCH_ENABLE_MPS_FALLBACK is also set here so that any operations not
    yet implemented on MPS silently fall back to CPU instead of crashing.
    """
    global _MPS_SKIP_LOGGED
    import torch

    # Re-read at call time so setting TFT_ENABLE_MPS=1 after import takes effect.
    enable_mps = os.environ.get("TFT_ENABLE_MPS", "").lower() in {"1", "true", "yes"}

    if torch.cuda.is_available():
        return "cuda"
    if enable_mps and torch.backends.mps.is_available():
        # Ensure ops not yet implemented on MPS fall back to CPU transparently.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps"
    if torch.backends.mps.is_available() and not _MPS_SKIP_LOGGED:
        _MPS_SKIP_LOGGED = True
        logger.info(
            "Apple MPS available — TFT uses CPU for stability. "
            "Set TFT_ENABLE_MPS=1 to attempt MPS (may crash the server)."
        )
    return "cpu"


def _tft_trainer_kwargs() -> dict:
    return {
        "accelerator": _tft_accelerator(),
        "devices": 1,
    }


def _tft_map_location() -> str:
    import torch

    enable_mps = os.environ.get("TFT_ENABLE_MPS", "").lower() in {"1", "true", "yes"}
    if torch.cuda.is_available():
        return "cuda"
    if enable_mps and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _tft_batch_size(n_rows: int) -> int:
    """Scale batch down for long histories to limit memory pressure."""
    if n_rows > 1200:
        return min(TFT_BATCH_SIZE, 8)
    if n_rows > 600:
        return min(TFT_BATCH_SIZE, 16)
    if n_rows > 300:
        return min(TFT_BATCH_SIZE, 32)
    return TFT_BATCH_SIZE


def _trim_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the recent window TFT can actually attend to."""
    working = ensure_demand_date_index(df)
    if len(working) <= TFT_MAX_TRAIN_DAYS:
        return working
    trimmed = working.iloc[-TFT_MAX_TRAIN_DAYS :].copy()
    logger.info(
        "TFT training window trimmed to last %d days (from %d total)",
        TFT_MAX_TRAIN_DAYS,
        len(working),
    )
    return trimmed


def _tft_min_history_days(series_length: int) -> int:
    """Allow shorter histories when the series is long enough for a reduced encoder."""
    if series_length >= MIN_HISTORY_DAYS_TFT:
        return MIN_HISTORY_DAYS_TFT
    if series_length >= MIN_HISTORY_DAYS_TFT_SHORT:
        return MIN_HISTORY_DAYS_TFT_SHORT
    return MIN_HISTORY_DAYS_TFT


def _tft_encoder_window(series_length: int) -> int:
    """Pick encoder length — shorter window when full year is unavailable."""
    if series_length >= MIN_HISTORY_DAYS_TFT:
        return LOOKBACK_WINDOW_TFT
    if series_length >= MIN_HISTORY_DAYS_TFT_SHORT:
        return min(LOOKBACK_WINDOW_TFT_SHORT, series_length - FORECAST_HORIZON)
    return LOOKBACK_WINDOW_TFT


class TFTModel(BaseForecastingModel):
    def __init__(self) -> None:
        self._model = None
        self._dataset_params: Optional[dict] = None
        self._skipped = False
        self._drug_code: Optional[str] = None
        self._prediction_cap: Optional[float] = None
        self._encoder_window: int = LOOKBACK_WINDOW_TFT

    def _build_dataset(self, df: pd.DataFrame, drug_code: str, predict: bool = False):
        from pytorch_forecasting import TimeSeriesDataSet
        from pytorch_forecasting.data.encoders import GroupNormalizer

        working = ensure_demand_date_index(df).reset_index()
        working["drug_code"] = drug_code
        working["time_idx"] = np.arange(len(working))

        encoder_cap = self._encoder_window or _tft_encoder_window(len(working))
        max_encoder = min(encoder_cap, max(len(working) - FORECAST_HORIZON, 1))
        max_prediction = min(FORECAST_HORIZON, max(len(working) // 4, 1))

        if self._dataset_params is not None and predict:
            return TimeSeriesDataSet.from_parameters(
                self._dataset_params,
                working,
                predict=True,
                stop_randomization=True,
            )

        return TimeSeriesDataSet(
            working,
            time_idx="time_idx",
            target="total_quantity",
            group_ids=["drug_code"],
            min_encoder_length=min(90, max_encoder),
            max_encoder_length=max_encoder,
            min_prediction_length=1,
            max_prediction_length=max_prediction,
            time_varying_known_reals=tft_known_reals(),
            time_varying_unknown_reals=tft_unknown_reals(),
            static_reals=tft_static_reals(),
            target_normalizer=GroupNormalizer(
                groups=["drug_code"],
                transformation="softplus",
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

    def train(
        self,
        df: pd.DataFrame,
        drug_code: str,
        *,
        max_epochs: Optional[int] = None,
    ) -> None:
        self._drug_code = drug_code
        self._skipped = False
        epochs = max_epochs if max_epochs is not None else TFT_MAX_EPOCHS
        min_required = _tft_min_history_days(len(df))
        if len(df) < min_required:
            logger.warning(
                "TFT skipped for %s: only %d days of history (< %d required)",
                drug_code,
                len(df),
                min_required,
            )
            self._skipped = True
            return

        self._encoder_window = _tft_encoder_window(len(df))

        try:
            import lightning.pytorch as pl
            from pytorch_forecasting import TemporalFusionTransformer
            from pytorch_forecasting.metrics import QuantileLoss
        except ImportError as exc:
            raise ImportError(
                "pytorch-forecasting and lightning/pytorch-lightning are required for TFT. "
                "Install via requirements.txt."
            ) from exc

        train_df = _trim_training_frame(df)
        self._prediction_cap = demand_prediction_cap(
            train_df["total_quantity"].astype(float).values,
        )
        training = self._build_dataset(train_df, drug_code)
        self._dataset_params = training.get_parameters()

        batch_size = _tft_batch_size(len(train_df))
        train_dataloader = training.to_dataloader(
            train=True,
            batch_size=batch_size,
            num_workers=0,
        )

        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=TFT_LEARNING_RATE,
            hidden_size=TFT_HIDDEN_SIZE,
            attention_head_size=TFT_ATTENTION_HEAD_SIZE,
            dropout=TFT_DROPOUT,
            hidden_continuous_size=TFT_HIDDEN_CONTINUOUS_SIZE,
            loss=QuantileLoss(quantiles=TFT_QUANTILES),
        )

        # pytorch-forecasting models inherit from lightning.pytorch.LightningModule;
        # Trainer must come from the same namespace (not legacy pytorch_lightning).
        accelerator = _tft_accelerator()
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=1,
            max_epochs=epochs,
            gradient_clip_val=TFT_GRADIENT_CLIP_VAL,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )
        trainer.fit(tft, train_dataloaders=train_dataloader)
        logger.info(
            "TFT training used accelerator=%s on %d days",
            accelerator,
            len(train_df),
        )

        checkpoint = _checkpoint_path(drug_code)
        os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
        trainer.save_checkpoint(checkpoint)

        with open(_dataset_path(drug_code), "wb") as handle:
            pickle.dump(self._dataset_params, handle)

        self._model = tft
        logger.info(
            "TFT trained for %s on %d days (encoder=%d, min_history=%d)",
            drug_code,
            len(train_df),
            self._encoder_window,
            min_required,
        )

    def _max_prediction_length(self) -> int:
        if self._dataset_params is not None:
            return int(self._dataset_params.get("max_prediction_length", FORECAST_HORIZON))
        return FORECAST_HORIZON

    def _extend_history(
        self,
        working: pd.DataFrame,
        chunk_predictions: np.ndarray,
        chunk_size: int,
        quantile_map: dict[float, int],
    ) -> pd.DataFrame:
        p50_idx = quantile_map[0.50]
        base = working.reset_index()
        if "demand_date" not in base.columns:
            base = base.rename(columns={"index": "demand_date"})
        base["demand_date"] = pd.to_datetime(base["demand_date"])

        last_date = pd.Timestamp(working.index[-1])
        carry_cols = [
            col
            for col in tft_known_reals() + tft_static_reals()
            if col in base.columns and col not in temporal_feature_columns()
        ]
        last_row = base.iloc[-1]
        extension_rows: list[dict[str, object]] = []
        for step in range(chunk_size):
            next_date = last_date + pd.Timedelta(days=step + 1)
            row = {"demand_date": next_date, "total_quantity": float(chunk_predictions[step, p50_idx])}
            for col in carry_cols:
                row[col] = last_row[col]
            extension_rows.append(row)

        history_qty = base[["demand_date", "total_quantity"]].copy()
        extended = pd.concat(
            [history_qty, pd.DataFrame(extension_rows)[["demand_date", "total_quantity"]]],
            ignore_index=True,
        )
        rebuilt = extended.copy()
        for col in carry_cols:
            rebuilt[col] = pd.concat(
                [base[col], pd.Series([last_row[col]] * chunk_size)],
                ignore_index=True,
            )
        rebuilt = add_temporal_features(rebuilt)
        rebuilt = add_lag_features(rebuilt)
        rebuilt = add_rolling_features(rebuilt)
        rebuilt = rebuilt.set_index("demand_date")
        rebuilt.index = pd.to_datetime(rebuilt.index)
        rebuilt.index.name = "demand_date"
        return rebuilt

    def _predict_chunk(
        self,
        df: pd.DataFrame,
        chunk_horizon: int,
    ) -> np.ndarray:
        if self._model is None:
            if self._drug_code is None:
                raise RuntimeError("TFT drug_code unknown — call train() or load() first")
            self.load(self._drug_code)

        predict_dataset = self._build_dataset(df, self._drug_code, predict=True)
        predict_dataloader = predict_dataset.to_dataloader(
            train=False,
            batch_size=1,
            num_workers=0,
        )
        raw_predictions = self._model.predict(
            predict_dataloader,
            mode="quantiles",
            return_x=False,
            trainer_kwargs=_tft_trainer_kwargs(),
        )
        if isinstance(raw_predictions, tuple):
            raw_predictions = raw_predictions[0]

        predictions = raw_predictions.detach().cpu().numpy()
        if predictions.ndim == 3:
            predictions = predictions[0, -chunk_horizon:, :]
        elif predictions.ndim == 2:
            predictions = predictions[-chunk_horizon:, :]
        return predictions

    def predict(self, df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
        if self._drug_code is None:
            raise RuntimeError("TFT drug_code unknown — call train() or load() first")
        if self._skipped or not self.is_trained(self._drug_code):
            raise RuntimeError("TFT model is not trained for this drug")

        working = ensure_demand_date_index(df).copy()
        max_chunk = max(1, self._max_prediction_length())
        quantile_map = {q: idx for idx, q in enumerate(TFT_QUANTILES)}
        chunk_predictions: list[np.ndarray] = []
        remaining = horizon_days

        while remaining > 0:
            chunk_size = min(remaining, max_chunk)
            chunk = self._predict_chunk(working, chunk_size)
            chunk_predictions.append(chunk)
            if remaining > chunk_size:
                working = self._extend_history(working, chunk, chunk_size, quantile_map)
            remaining -= chunk_size

        predictions = np.vstack(chunk_predictions)
        dates = forecast_dates_from_index(ensure_demand_date_index(df).index, horizon_days)

        result = pd.DataFrame({"forecast_date": dates})
        cap = self._prediction_cap
        if cap is None:
            cap = demand_prediction_cap(working["total_quantity"].astype(float).values)

        result["p5"] = winsorize_predictions(predictions[:, quantile_map[0.05]], cap)
        result["p10"] = winsorize_predictions(predictions[:, quantile_map[0.10]], cap)
        result["p50"] = winsorize_predictions(predictions[:, quantile_map[0.50]], cap)
        result["p90"] = winsorize_predictions(predictions[:, quantile_map[0.90]], cap)
        result["p95"] = winsorize_predictions(predictions[:, quantile_map[0.95]], cap)
        return result

    def extract_attention_weights(
        self,
        df: pd.DataFrame,
        drug_code: str,
    ) -> list[AttentionWeight]:
        from app.forecasting.schemas import AttentionWeight

        if self._drug_code is None:
            self.load(drug_code)
        if self._model is None:
            self.load(drug_code)

        predict_dataset = self._build_dataset(df, drug_code, predict=True)
        predict_dataloader = predict_dataset.to_dataloader(
            train=False,
            batch_size=1,
            num_workers=0,
        )
        output = self._model.predict(
            predict_dataloader,
            mode="raw",
            return_x=True,
            trainer_kwargs=_tft_trainer_kwargs(),
        )
        if isinstance(output, tuple):
            raw_predictions, x = output
        else:
            raw_predictions = output
            x = None

        interpretation = self._model.interpret_output(
            raw_predictions,
            reduction="sum",
        )
        attention = interpretation.get("attention")
        if attention is None:
            encoder_vars = interpretation.get("encoder_variables")
            if encoder_vars is None:
                return []
            attention = encoder_vars

        weights = attention.detach().cpu().numpy().reshape(-1)
        if weights.size == 0:
            return []

        if weights.size >= 52:
            bucket_size = max(weights.size // 52, 1)
            weekly = []
            for week in range(52):
                start = max(weights.size - (week + 1) * bucket_size, 0)
                end = max(weights.size - week * bucket_size, start + 1)
                weekly.append(float(weights[start:end].mean()))
            total = sum(weekly) or 1.0
            return [
                AttentionWeight(week_offset=week, weight=value / total)
                for week, value in enumerate(weekly)
            ]

        total = float(weights.sum()) or 1.0
        return [
            AttentionWeight(
                week_offset=idx,
                weight=float(value / total),
            )
            for idx, value in enumerate(weights[:52])
        ]

    def save(self, drug_code: str) -> str:
        path = _checkpoint_path(drug_code)
        if not os.path.isfile(path):
            raise RuntimeError("Cannot save TFT model — no checkpoint found")
        return path

    def load(self, drug_code: str) -> None:
        from pytorch_forecasting import TemporalFusionTransformer

        self._drug_code = drug_code
        with open(_dataset_path(drug_code), "rb") as handle:
            self._dataset_params = pickle.load(handle)
        # Local training artifacts embed pytorch_forecasting objects (e.g.
        # GroupNormalizer); PyTorch 2.6+ defaults weights_only=True which rejects them.
        self._model = TemporalFusionTransformer.load_from_checkpoint(
            _checkpoint_path(drug_code),
            map_location=_tft_map_location(),
            weights_only=False,
        )
        self._model.eval()
        self._skipped = False

    def is_trained(self, drug_code: str) -> bool:
        return os.path.isfile(_checkpoint_path(drug_code)) and os.path.isfile(
            _dataset_path(drug_code)
        )
