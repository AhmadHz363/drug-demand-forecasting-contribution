"""Single-drug forecasting inference pipeline."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.forecasting.censored_demand.corrector import correct_demand
from app.forecasting.constants import CONFORMAL_COVERAGE, LGBM_LOOKBACK_WINDOW
from app.forecasting.ensemble.conformal import (
    ConformalCalibrator,
    extend_tail_quantiles,
    _mapie_path,
)
from app.forecasting.ensemble.stacking import StackingMetaLearner, _stacking_path
from app.forecasting.feature_engineering.pipeline import (
    build_feature_matrix,
    build_future_covariates,
)
from app.forecasting.models.feature_columns import ensure_demand_date_index
from app.forecasting.models.lgbm_model import LightGBMModel
from app.forecasting.models.sarima_model import SarimaModel
from app.forecasting.models.tft_model import TFTModel
from app.forecasting.schemas import (
    AttentionWeight,
    DailyForecastPoint,
    ForecastResponse,
    ModelWeightBreakdown,
    ShapFeature,
)
from app.forecasting.training.trainer import MODEL_REGISTRY
from app.forecasting.prediction_bounds import demand_prediction_cap, sanitize_ensemble_predictions
from app.models.forecast_result import ForecastResult
from app.models.model_performance import ModelPerformance
from app.services.demand_aggregation import drug_has_receipt_history, get_receipt_date_bounds

logger = logging.getLogger(__name__)

UNCERTAINTY_NOTE = (
    f"P10/P90 intervals are conformally calibrated to ~{int(CONFORMAL_COVERAGE * 100)}% coverage. "
    "P5/P95 are approximate tail extrapolations."
)


class DrugNotInCatalogError(Exception):
    """Raised when the requested drug has no history in drug_receipts."""


class NoTrainedModelsError(Exception):
    """Raised when no forecasting artifacts exist for the drug."""


def validate_forecast_quantiles(response: ForecastResponse) -> None:
    """Assert quantile ordering and normalized model weights."""
    if response.error:
        return

    total_weight = (
        response.model_weights.sarima
        + response.model_weights.lgbm
        + response.model_weights.tft
    )
    if abs(total_weight - 1.0) > 1e-5:
        raise ValueError(
            f"Model weights must sum to 1.0, got {total_weight:.6f} for {response.drug_code}"
        )

    for point in response.forecast:
        values = [point.p10, point.p50, point.p90]
        if point.p5 is not None:
            if point.p5 > point.p10:
                raise ValueError(
                    f"Invalid quantiles for {point.date}: P5 ({point.p5}) > P10 ({point.p10})"
                )
        if point.p95 is not None and point.p90 > point.p95:
            raise ValueError(
                f"Invalid quantiles for {point.date}: P90 ({point.p90}) > P95 ({point.p95})"
            )
        if not (values[0] <= values[1] <= values[2]):
            raise ValueError(
                f"Invalid quantiles for {point.date}: "
                f"P10={point.p10}, P50={point.p50}, P90={point.p90}"
            )


def _conformal_available() -> bool:
    return os.path.isfile(_mapie_path())


class DrugForecaster:
    """Loads trained artifacts and produces calibrated ensemble forecasts."""

    def _assert_drug_has_receipt_history(
        self,
        drug_code: str,
        center_syn_id: Optional[str],
        db_session: Session,
    ) -> None:
        if not drug_has_receipt_history(db_session, drug_code, center_syn_id):
            scope = f" for center {center_syn_id}" if center_syn_id else ""
            raise DrugNotInCatalogError(
                f"No receipt history found for drug {drug_code}{scope} in drug_receipts."
            )

    def _trained_models(self, drug_code: str) -> list[str]:
        trained: list[str] = []
        for name, model_cls in MODEL_REGISTRY.items():
            if model_cls().is_trained(drug_code):
                trained.append(name)
        return trained

    def _demand_date_range(
        self,
        drug_code: str,
        center_syn_id: Optional[str],
        db_session: Session,
    ) -> tuple[date, date]:
        _, end_date = get_receipt_date_bounds(db_session, drug_code, center_syn_id)
        if end_date is None:
            end_date = date.today()
        start_date = end_date - timedelta(days=LGBM_LOOKBACK_WINDOW - 1)
        return start_date, end_date

    def _latest_smape(self, drug_code: str, db_session: Session) -> Optional[float]:
        row = (
            db_session.query(ModelPerformance)
            .filter(
                ModelPerformance.drug_code == drug_code,
                ModelPerformance.model_name == "ensemble",
            )
            .order_by(desc(ModelPerformance.evaluated_at))
            .first()
        )
        if row is not None:
            return float(row.smape)

        row = (
            db_session.query(ModelPerformance)
            .filter(ModelPerformance.drug_code == drug_code)
            .order_by(desc(ModelPerformance.evaluated_at))
            .first()
        )
        return float(row.smape) if row is not None else None

    def _model_p50_predictions(
        self,
        corrected_df: pd.DataFrame,
        drug_code: str,
        horizon_days: int,
        trained_models: list[str],
        *,
        db_session: Session,
        center_syn_id: Optional[str],
    ) -> dict[str, np.ndarray]:
        preds: dict[str, np.ndarray] = {}
        nan_array = np.full(horizon_days, np.nan, dtype=float)

        if "sarima" in trained_models:
            sarima = SarimaModel()
            sarima.load(drug_code)
            sarima_df = sarima.predict(corrected_df, horizon_days)
            preds["sarima"] = sarima_df["p50"].astype(float).values

        if "lgbm" in trained_models:
            lgbm = LightGBMModel()
            lgbm.load(drug_code)
            last_date = ensure_demand_date_index(corrected_df).index[-1].date()
            future_covariates = build_future_covariates(
                drug_code,
                center_syn_id,
                db_session,
                last_date,
                horizon_days,
            )
            lgbm_df = lgbm.predict(
                corrected_df,
                horizon_days,
                future_covariates=future_covariates,
            )
            preds["lgbm"] = lgbm_df["p50"].astype(float).values

        if "tft" in trained_models:
            tft = TFTModel()
            tft.load(drug_code)
            tft_df = tft.predict(corrected_df, horizon_days)
            preds["tft"] = tft_df["p50"].astype(float).values

        for name in MODEL_REGISTRY:
            preds.setdefault(name, nan_array.copy())
        return preds

    def _fallback_intervals(
        self,
        stacked: np.ndarray,
        sarima: np.ndarray,
        lgbm: np.ndarray,
        tft: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p50 = stacked
        spread = np.nanstd(np.column_stack([sarima, lgbm, tft]), axis=1)
        spread = np.where(np.isnan(spread) | (spread == 0), p50 * 0.2, spread)
        p10 = np.clip(p50 - spread, 0.0, None)
        p90 = p50 + spread
        return p50, p10, p90

    def _ensemble_forecast(
        self,
        model_preds: dict[str, np.ndarray],
        horizon_days: int,
        *,
        prediction_cap: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, ModelWeightBreakdown]:
        sarima = model_preds["sarima"]
        lgbm = model_preds["lgbm"]
        tft = model_preds["tft"]

        sarima, lgbm, tft = sanitize_ensemble_predictions(sarima, lgbm, tft, prediction_cap)

        if os.path.isfile(_stacking_path()):
            stacker = StackingMetaLearner()
            stacker.load()
            stacked, weights = stacker.predict(
                sarima,
                lgbm,
                tft,
                prediction_cap=prediction_cap,
            )
        else:
            available = np.column_stack([sarima, lgbm, tft])
            stacked = np.nanmean(available, axis=1)
            n_models = np.sum(~np.isnan(available), axis=1)
            stacked = np.where(n_models == 0, 0.0, stacked)
            active = (~np.isnan(available)).sum(axis=0)
            total_active = max(int(active.sum()), 1)
            weights = ModelWeightBreakdown(
                sarima=float(active[0] / total_active) if active[0] else 0.0,
                lgbm=float(active[1] / total_active) if active[1] else 0.0,
                tft=float(active[2] / total_active) if active[2] else 0.0,
            )
            if weights.tft == 0.0 and weights.sarima + weights.lgbm > 0:
                total = weights.sarima + weights.lgbm
                weights = ModelWeightBreakdown(
                    sarima=weights.sarima / total,
                    lgbm=weights.lgbm / total,
                    tft=0.0,
                )

        if _conformal_available():
            try:
                calibrator = ConformalCalibrator()
                calibrator.load()
                p50, p10, p90 = calibrator.predict_interval(stacked)
            except (ValueError, RuntimeError) as exc:
                logger.warning(
                    "Conformal interval prediction failed — using model spread fallback: %s",
                    exc,
                )
                p50, p10, p90 = self._fallback_intervals(stacked, sarima, lgbm, tft)
        else:
            p50, p10, p90 = self._fallback_intervals(stacked, sarima, lgbm, tft)

        p5, p10, p50, p90, p95 = extend_tail_quantiles(p50, p10, p90)
        return p5[:horizon_days], p10[:horizon_days], p50[:horizon_days], p90[:horizon_days], p95[:horizon_days], weights

    def _forecast_dates(
        self,
        corrected_df: pd.DataFrame,
        horizon_days: int,
    ) -> list[date]:
        index = ensure_demand_date_index(corrected_df).index
        last_date = pd.Timestamp(index[-1])
        return [
            (last_date + pd.Timedelta(days=offset)).date()
            for offset in range(1, horizon_days + 1)
        ]

    def _compute_shap_features(
        self,
        drug_code: str,
        lgbm: LightGBMModel,
        corrected_df: pd.DataFrame,
        top_n: int = 15,
    ) -> list[ShapFeature]:
        try:
            import shap
        except ImportError:
            return self._load_saved_shap_features(drug_code, corrected_df, top_n)

        x, _ = lgbm._prepare_xy(corrected_df)
        booster = lgbm._boosters.get(0.50)
        if booster is None:
            return self._load_saved_shap_features(drug_code, corrected_df, top_n)

        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(x.iloc[[-1]])
        values = np.asarray(shap_values).reshape(-1)
        features = x.columns.tolist()
        pairs = sorted(
            zip(features, values, x.iloc[-1].tolist()),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:top_n]
        return [
            ShapFeature(
                feature_name=name,
                shap_value=float(shap_val),
                feature_value=float(feat_val),
            )
            for name, shap_val, feat_val in pairs
        ]

    def _load_saved_shap_features(
        self,
        drug_code: str,
        corrected_df: pd.DataFrame,
        top_n: int,
    ) -> list[ShapFeature]:
        path = LightGBMModel().shap_artifact_path(drug_code)
        if not os.path.isfile(path):
            return []

        with open(path, encoding="utf-8") as handle:
            summary = json.load(handle)
        working = ensure_demand_date_index(corrected_df)
        last_row = working.iloc[-1]
        pairs = sorted(summary.items(), key=lambda item: item[1], reverse=True)[:top_n]
        return [
            ShapFeature(
                feature_name=name,
                shap_value=float(score),
                feature_value=float(last_row[name]) if name in last_row.index else 0.0,
            )
            for name, score in pairs
        ]

    def _extract_attention_weights(
        self,
        tft: TFTModel,
        corrected_df: pd.DataFrame,
        drug_code: str,
    ) -> Optional[list[AttentionWeight]]:
        try:
            return tft.extract_attention_weights(corrected_df, drug_code)
        except Exception as exc:
            logger.warning("TFT attention extraction failed for %s: %s", drug_code, exc)
            return None

    def _persist_forecast_results(
        self,
        db_session: Session,
        drug_code: str,
        center_syn_id: Optional[str],
        forecast_dates: list[date],
        p5: np.ndarray,
        p10: np.ndarray,
        p50: np.ndarray,
        p90: np.ndarray,
        p95: np.ndarray,
        weights: ModelWeightBreakdown,
        generated_at: datetime,
    ) -> None:
        for idx, forecast_date in enumerate(forecast_dates):
            db_session.add(
                ForecastResult(
                    drug_code=drug_code,
                    center_syn_id=center_syn_id,
                    generated_at=generated_at,
                    forecast_date=forecast_date,
                    p5=float(p5[idx]),
                    p10=float(p10[idx]),
                    p50=float(p50[idx]),
                    p90=float(p90[idx]),
                    p95=float(p95[idx]),
                    model_weight_sarima=weights.sarima,
                    model_weight_lgbm=weights.lgbm,
                    model_weight_tft=weights.tft,
                )
            )
        db_session.flush()

    def forecast(
        self,
        drug_code: str,
        horizon_days: int,
        center_syn_id: Optional[str],
        include_shap: bool,
        include_attention: bool,
        db_session: Session,
    ) -> ForecastResponse:
        started = time.perf_counter()
        self._assert_drug_has_receipt_history(drug_code, center_syn_id, db_session)

        trained_models = self._trained_models(drug_code)
        if not trained_models:
            raise NoTrainedModelsError(
                f"No trained models found for {drug_code}. Call POST /forecasting/train first."
            )

        start_date, end_date = self._demand_date_range(
            drug_code,
            center_syn_id,
            db_session,
        )
        feature_df = build_feature_matrix(
            drug_code,
            center_syn_id,
            db_session,
            start_date,
            end_date,
        )
        corrected_df = correct_demand(
            drug_code,
            center_syn_id,
            db_session,
            feature_df,
        )
        corrected_df.attrs["drug_code"] = drug_code

        model_preds = self._model_p50_predictions(
            corrected_df,
            drug_code,
            horizon_days,
            trained_models,
            db_session=db_session,
            center_syn_id=center_syn_id,
        )
        prediction_cap = demand_prediction_cap(
            corrected_df["total_quantity"].astype(float).values,
        )
        p5, p10, p50, p90, p95, weights = self._ensemble_forecast(
            model_preds,
            horizon_days,
            prediction_cap=prediction_cap,
        )
        forecast_dates = self._forecast_dates(corrected_df, horizon_days)

        shap_features: Optional[list[ShapFeature]] = None
        if include_shap and "lgbm" in trained_models:
            lgbm = LightGBMModel()
            lgbm.load(drug_code)
            shap_features = self._compute_shap_features(drug_code, lgbm, corrected_df)

        attention_weights: Optional[list[AttentionWeight]] = None
        if include_attention and "tft" in trained_models:
            tft = TFTModel()
            tft.load(drug_code)
            attention_weights = self._extract_attention_weights(
                tft,
                corrected_df,
                drug_code,
            )

        generated_at = datetime.now(timezone.utc)
        self._persist_forecast_results(
            db_session,
            drug_code,
            center_syn_id,
            forecast_dates,
            p5,
            p10,
            p50,
            p90,
            p95,
            weights,
            generated_at,
        )
        db_session.commit()

        response = ForecastResponse(
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
                )
                for idx in range(horizon_days)
            ],
            shap_features=shap_features,
            attention_weights=attention_weights,
            uncertainty_note=UNCERTAINTY_NOTE,
            smape_last_validation=self._latest_smape(drug_code, db_session),
        )

        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "Forecast for %s completed in %.1fms (horizon=%d, shap=%s, attention=%s)",
            drug_code,
            elapsed_ms,
            horizon_days,
            include_shap,
            include_attention,
        )
        return response
