"""SHIELD-XR inference — per-drug daily forecasts from the global ensemble."""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.forecasting.constants import FORECAST_CHART_HISTORY_DAYS
from app.forecasting.schemas import (
    DailyForecastPoint,
    ForecastResponse,
    HistoryPoint,
    InferenceHealth,
    ModelWeightBreakdown,
)
from app.forecasting.shield_xr.anomaly_guard import apply_anomaly_guard
from app.forecasting.shield_xr.artifacts import is_trained, load_daily_artifacts, load_training_meta
from app.forecasting.shield_xr.daily_model import predict_ensemble_row
from app.forecasting.shield_xr.features import add_features, assign_sb_classes, coerce_feature_dtypes
from app.forecasting.shield_xr.panel_builder import build_hospital_panel, split_temporal
from app.models.model_performance import ModelPerformance
from app.services.enriched_demand import drug_has_enriched_history, get_enriched_date_bounds

logger = logging.getLogger(__name__)

UNCERTAINTY_NOTE = (
    "P10/P90 intervals are derived from SHIELD-XR validation residual spread. "
    "P5/P95 are approximate tail extrapolations."
)


class ShieldXRNotTrainedError(Exception):
    """Raised when SHIELD-XR artifacts are missing."""


class ShieldXRForecaster:
    """Generate per-drug forecasts from the hospital-wide SHIELD-XR ensemble."""

    def _assert_trained(self) -> None:
        if not is_trained():
            raise ShieldXRNotTrainedError(
                "SHIELD-XR model is not trained. Call POST /forecasting/train first."
            )

    def _model_weights(self, sb_class: str, artifacts) -> ModelWeightBreakdown:
        winner = artifacts.class_winner.get(sb_class, "SHIELD-XR")
        if winner == "Plain":
            return ModelWeightBreakdown(sarima=1.0, lgbm=0.0, classical=0.0)
        if winner == "Plain-L1":
            return ModelWeightBreakdown(sarima=0.0, lgbm=0.0, classical=1.0)
        return ModelWeightBreakdown(sarima=0.0, lgbm=1.0, classical=0.0)

    def _build_inference_frame(
        self,
        db_session: Session,
        drug_code: str,
        center_syn_id: Optional[str],
        artifacts,
    ) -> pd.DataFrame:
        start, end = get_enriched_date_bounds(db_session, drug_code)
        if start is None or end is None:
            raise ValueError(f"No enriched history for {drug_code}")
        panel = build_hospital_panel(
            db_session,
            start_date=start,
            end_date=end,
            drug_codes=[drug_code],
            center_syn_id=center_syn_id,
        )
        if panel.empty:
            raise ValueError(f"No panel rows for {drug_code}")

        d_train_end, _ = split_temporal(panel, train_frac=0.85, valid_frac=0.0)
        guarded = apply_anomaly_guard(panel, d_train_end)
        cleaned = assign_sb_classes(guarded.frame, d_train_end)
        feat = add_features(cleaned)
        meta = load_training_meta()
        code2id = artifacts.code2id
        cat2id = artifacts.cat2id
        feat["code_id"] = feat["CODE"].astype(str).map(code2id).fillna(0).astype(int)
        feat["cat_id"] = feat["CAT"].astype(str).map(cat2id).fillna(cat2id.get("UNK", 0)).astype(int)
        for col in artifacts.feature_cols:
            if col not in feat.columns:
                feat[col] = 0.0
            feat[col] = feat[col].fillna(feat[col].median() if feat[col].notna().any() else 0.0)
        feat = coerce_feature_dtypes(feat, artifacts.feature_cols)
        feat.attrs["train_end"] = meta.get("train_end")
        return feat

    def _recursive_forecast(
        self,
        history: pd.DataFrame,
        drug_code: str,
        horizon_days: int,
        artifacts,
    ) -> tuple[list[date], np.ndarray]:
        working = history[history["CODE"] == drug_code].sort_values("DATE").copy()
        if working.empty:
            raise ValueError(f"Drug {drug_code} not found in inference frame")

        last_date = pd.Timestamp(working["DATE"].max())
        predictions: list[float] = []
        forecast_dates: list[date] = []

        # Seed with historical cleaned demand for recursive lag updates.
        demand_series = working.set_index("DATE")["demand"].astype(float).copy()
        sb_class = str(working["sb_class"].iloc[-1])

        for step in range(1, horizon_days + 1):
            forecast_date = last_date + pd.Timedelta(days=step)
            forecast_dates.append(forecast_date.date())

            row = working.iloc[[-1]].copy()
            row["DATE"] = forecast_date
            row["demand"] = 0.0
            row["demand_raw"] = 0.0
            row["demand_clean"] = 0.0
            row["dow"] = forecast_date.dayofweek
            row["month"] = forecast_date.month
            row["day"] = forecast_date.day
            row["weekofyear"] = forecast_date.isocalendar().week
            row["is_weekend"] = int(forecast_date.dayofweek >= 5)
            row["is_saturday"] = int(forecast_date.dayofweek == 5)
            row["is_month_end"] = int(forecast_date.is_month_end)
            row["year"] = forecast_date.year

            recent = demand_series.tail(28)
            for lag in (1, 7, 14, 28):
                lag_date = forecast_date - pd.Timedelta(days=lag)
                row[f"lag_{lag}"] = float(demand_series.get(lag_date, recent.iloc[-1] if len(recent) else 0.0))
            shifted = demand_series
            for window in (7, 14, 28):
                window_vals = shifted.tail(window)
                row[f"roll_mean_{window}"] = float(window_vals.mean()) if len(window_vals) else 0.0
                row[f"roll_std_{window}"] = float(window_vals.std()) if len(window_vals) > 1 else 0.0
                row[f"roll_max_{window}"] = float(window_vals.max()) if len(window_vals) else 0.0
                row[f"roll_nz_{window}"] = float((window_vals > 0).mean()) if len(window_vals) else 0.0

            row["sb_class"] = sb_class
            row["sb_class_id"] = {"smooth": 0, "intermittent": 1, "erratic": 2, "lumpy": 3}.get(sb_class, 3)
            for col in artifacts.feature_cols:
                if col not in row.columns:
                    row[col] = 0.0

            p50 = predict_ensemble_row(row.iloc[0], artifacts)
            predictions.append(p50)
            demand_series.loc[forecast_date] = p50

        return forecast_dates, np.asarray(predictions, dtype=float)

    def _latest_validation(self, drug_code: str, db_session: Session) -> dict[str, float | None]:
        row = (
            db_session.query(ModelPerformance)
            .filter(
                ModelPerformance.drug_code == drug_code,
                ModelPerformance.model_name == "shield_xr_ensemble",
            )
            .order_by(ModelPerformance.evaluated_at.desc())
            .first()
        )
        if row is None:
            return {}
        return {
            "full_smape": float(row.smape) if row.smape is not None else None,
            "normal_smape": float(row.smape_normal_supply) if row.smape_normal_supply is not None else None,
            "full_mase": float(row.mase) if row.mase is not None else None,
            "normal_mase": float(row.mase_normal_supply) if row.mase_normal_supply is not None else None,
        }

    def forecast(
        self,
        drug_code: str,
        horizon_days: int,
        center_syn_id: Optional[str],
        db_session: Session,
    ) -> ForecastResponse:
        self._assert_trained()
        if not drug_has_enriched_history(db_session, drug_code):
            raise ValueError(f"Drug {drug_code} has no enriched demand history.")

        artifacts = load_daily_artifacts()
        frame = self._build_inference_frame(db_session, drug_code, center_syn_id, artifacts)
        drug_frame = frame[frame["CODE"] == drug_code]
        sb_class = str(drug_frame["sb_class"].iloc[-1]) if not drug_frame.empty else "lumpy"
        weights = self._model_weights(sb_class, artifacts)

        forecast_dates, p50 = self._recursive_forecast(frame, drug_code, horizon_days, artifacts)
        spread = max(artifacts.residual_q90, p50.mean() * 0.15 if len(p50) else 1.0)
        p10 = np.clip(p50 - spread, 0, None)
        p90 = p50 + spread
        p5 = np.clip(p50 - spread * 1.5, 0, None)
        p95 = p50 + spread * 1.5

        validation = self._latest_validation(drug_code, db_session)
        history = drug_frame.sort_values("DATE").tail(FORECAST_CHART_HISTORY_DAYS)
        qty_col = "demand_clean" if "demand_clean" in history.columns else "demand"

        generated_at = datetime.now(timezone.utc)
        del generated_at  # forecast persistence disabled until forecast_results table is restored

        return ForecastResponse(
            drug_code=drug_code,
            center_syn_id=center_syn_id,
            horizon_days=horizon_days,
            model_weights=weights,
            forecast=[
                DailyForecastPoint(
                    date=forecast_dates[idx],
                    p5=float(p5[idx]),
                    p10=float(p10[idx]),
                    p50=float(p50[idx]),
                    p90=float(p90[idx]),
                    p95=float(p95[idx]),
                    recommended_quantity=float(p50[idx]),
                )
                for idx in range(horizon_days)
            ],
            history=[
                HistoryPoint(
                    date=pd.Timestamp(row.DATE).date(),
                    quantity=float(getattr(row, qty_col)),
                )
                for row in history.itertuples(index=False)
            ],
            uncertainty_note=UNCERTAINTY_NOTE,
            smape_last_validation=validation.get("full_smape"),
            smape_validation_full_window=validation.get("full_smape"),
            smape_validation_normal_supply=validation.get("normal_smape"),
            mase_validation_full_window=validation.get("full_mase"),
            mase_validation_normal_supply=validation.get("normal_mase"),
            demand_segment=sb_class,
            primary_validation_metric="wape",
            validation_metrics_note="SHIELD-XR reports WAPE-derived accuracy from hospital-wide ensemble.",
            inference_health=InferenceHealth(
                demand_segment=sb_class,
                used_stacking=False,
                used_conformal=False,
                used_spread_fallback=True,
            ),
        )
