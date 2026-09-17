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
from app.forecasting.constants import (
    CONFORMAL_COVERAGE,
    FORECAST_CHART_HISTORY_DAYS,
    INTERVAL_P10_CLIFF_P50_THRESHOLD,
    INTERVAL_WIDTH_SHRINK_TOLERANCE,
    LGBM_LOOKBACK_WINDOW,
    WALK_FORWARD_N_SPLITS,
    WALK_FORWARD_TEST_HORIZON,
)
from app.forecasting.ensemble.conformal import (
    ConformalCalibrator,
    extend_tail_quantiles,
)
from app.forecasting.ensemble.segment_artifacts import (
    resolve_conformal_path,
    resolve_stacking_path,
)
from app.forecasting.ensemble.stacking import StackingMetaLearner
from app.forecasting.demand_segmentation import classify_demand_segment_from_frame
from app.forecasting.evaluation_metrics import (
    accuracy_skill_from_mase,
    mase_beats_baseline,
    primary_validation_metric_for_segment,
    smape_unreliable_for_series,
    validation_metrics_note,
    zero_actual_fraction,
)
from app.forecasting.prediction_bounds import (
    dampen_stacked_horizon_drift,
    demand_prediction_cap,
    sanitize_ensemble_predictions,
)
from app.forecasting.model_adaptation import recent_cv2
from app.forecasting.feature_engineering.pipeline import (
    build_feature_matrix,
    build_future_covariates,
    filter_covered_rows,
)
from app.forecasting.models.feature_columns import ensure_demand_date_index
from app.forecasting.training.champion_selection import load_champion
from app.forecasting.models.classical_model import ClassicalModel
from app.forecasting.models.lgbm_model import LightGBMModel
from app.forecasting.models.sarima_model import SarimaModel
from app.forecasting.schemas import (
    AttentionWeight,
    DailyForecastPoint,
    ForecastResponse,
    HistoryPoint,
    InferenceHealth,
    ModelWeightBreakdown,
    ShapFeature,
)
from app.forecasting.newsvendor import recommended_quantities, resolve_ven_class
from app.forecasting.training.trainer import MODEL_REGISTRY
from app.models.forecast_result import ForecastResult
from app.models.drug_catalog import DrugCatalog
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
        + response.model_weights.classical
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


def check_forecast_interval_sanity(response: ForecastResponse) -> list[str]:
    """Return warnings for discontinuous interval growth or P10 cliffs."""
    if response.error or len(response.forecast) < 2:
        return []

    warnings: list[str] = []
    widths = [point.p90 - point.p10 for point in response.forecast]

    for idx in range(1, len(widths)):
        previous = widths[idx - 1]
        current = widths[idx]
        if previous > 0 and current < previous * INTERVAL_WIDTH_SHRINK_TOLERANCE:
            warnings.append(
                f"Interval width shrinks sharply at horizon step {idx + 1} "
                f"({previous:.4f} -> {current:.4f}) for {response.drug_code}"
            )

    consecutive_zero_p10 = 0
    for step, point in enumerate(response.forecast, start=1):
        if point.p10 == 0.0 and point.p50 > INTERVAL_P10_CLIFF_P50_THRESHOLD:
            consecutive_zero_p10 += 1
        else:
            consecutive_zero_p10 = 0
        if consecutive_zero_p10 > 1:
            warnings.append(
                f"P10 collapsed to 0 for {consecutive_zero_p10} consecutive days "
                f"while P50 > {INTERVAL_P10_CLIFF_P50_THRESHOLD} "
                f"(from step {step - consecutive_zero_p10 + 1}) for {response.drug_code}"
            )
            break

    return warnings


def validate_forecast_interval_sanity(response: ForecastResponse) -> None:
    """Raise when forecast intervals show suspicious discontinuities."""
    issues = check_forecast_interval_sanity(response)
    if issues:
        raise ValueError("; ".join(issues))


def _conformal_available(segment: str) -> bool:
    return resolve_conformal_path(segment) is not None


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

    def _latest_validation_metrics(
        self,
        drug_code: str,
        db_session: Session,
        *,
        preferred_model: Optional[str] = None,
    ) -> dict[str, float | bool | None]:
        query = db_session.query(ModelPerformance).filter(
            ModelPerformance.drug_code == drug_code,
        )
        if preferred_model in {"sarima", "lgbm", "classical", "ensemble"}:
            row = (
                query.filter(ModelPerformance.model_name == preferred_model)
                .order_by(desc(ModelPerformance.evaluated_at))
                .first()
            )
        else:
            row = None
        if row is None:
            row = (
                query.filter(ModelPerformance.model_name == "ensemble")
                .order_by(desc(ModelPerformance.evaluated_at))
                .first()
            )
        if row is None:
            row = query.order_by(desc(ModelPerformance.evaluated_at)).first()
        if row is None:
            return {}

        full_mase = float(row.mase) if row.mase is not None else None
        full_smape = float(row.smape)
        normal_mase = (
            float(row.mase_normal_supply)
            if row.mase_normal_supply is not None
            else full_mase
        )
        normal_smape = (
            float(row.smape_normal_supply)
            if row.smape_normal_supply is not None
            else full_smape
        )
        return {
            "full_smape": full_smape,
            "full_mase": full_mase,
            "normal_smape": normal_smape,
            "normal_mase": normal_mase,
            "mase_beats_baseline_full": mase_beats_baseline(full_mase),
            "mase_beats_baseline_normal": mase_beats_baseline(normal_mase),
            "mase_skill_full": (
                accuracy_skill_from_mase(full_mase) if full_mase is not None else None
            ),
            "smape_7day_full": (
                float(row.smape_7day_full) if row.smape_7day_full is not None else None
            ),
            "smape_30day_full": (
                float(row.smape_30day_full) if row.smape_30day_full is not None else None
            ),
            "mase_7day_full": (
                float(row.mase_7day_full) if row.mase_7day_full is not None else None
            ),
            "mase_30day_full": (
                float(row.mase_30day_full) if row.mase_30day_full is not None else None
            ),
            "smape_7day_normal": (
                float(row.smape_7day_normal)
                if row.smape_7day_normal is not None
                else (
                    float(row.smape_7day_full)
                    if row.smape_7day_full is not None
                    else None
                )
            ),
            "smape_30day_normal": (
                float(row.smape_30day_normal)
                if row.smape_30day_normal is not None
                else (
                    float(row.smape_30day_full)
                    if row.smape_30day_full is not None
                    else None
                )
            ),
            "mase_7day_normal": (
                float(row.mase_7day_normal)
                if row.mase_7day_normal is not None
                else (
                    float(row.mase_7day_full) if row.mase_7day_full is not None else None
                )
            ),
            "mase_30day_normal": (
                float(row.mase_30day_normal)
                if row.mase_30day_normal is not None
                else (
                    float(row.mase_30day_full)
                    if row.mase_30day_full is not None
                    else None
                )
            ),
        }

    def _latest_smape(self, drug_code: str, db_session: Session) -> Optional[float]:
        metrics = self._latest_validation_metrics(drug_code, db_session)
        full_smape = metrics.get("full_smape")
        return float(full_smape) if full_smape is not None else None

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

        if "classical" in trained_models:
            classical = ClassicalModel()
            classical.load(drug_code)
            classical_df = classical.predict(corrected_df, horizon_days)
            preds["classical"] = classical_df["p50"].astype(float).values

        for name in MODEL_REGISTRY:
            preds.setdefault(name, nan_array.copy())
        return preds

    def _fallback_intervals(
        self,
        stacked: np.ndarray,
        sarima: np.ndarray,
        lgbm: np.ndarray,
        classical: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p50 = stacked
        spread = np.nanstd(np.column_stack([sarima, lgbm, classical]), axis=1)
        spread = np.where(np.isnan(spread) | (spread == 0), p50 * 0.2, spread)
        p10 = np.clip(p50 - spread, 0.0, None)
        p90 = p50 + spread
        return p50, p10, p90

    def _lookup_ven_class(self, drug_code: str, db_session: Session) -> str:
        row = (
            db_session.query(DrugCatalog.ven_class)
            .filter(DrugCatalog.drug_code == drug_code)
            .first()
        )
        return resolve_ven_class(row[0] if row else None)

    def _ensemble_forecast(
        self,
        model_preds: dict[str, np.ndarray],
        horizon_days: int,
        *,
        prediction_cap: float,
        demand_segment: str,
        history_days: int,
        drug_cv2: float,
        classical_available: bool,
        champion: Optional[str] = None,
        recent_level: float = 0.0,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        ModelWeightBreakdown,
        InferenceHealth,
    ]:
        sarima = model_preds["sarima"]
        lgbm = model_preds["lgbm"]
        classical = model_preds["classical"]
        horizon_steps = np.arange(1, horizon_days + 1, dtype=int)

        sarima, lgbm, classical = sanitize_ensemble_predictions(sarima, lgbm, classical, prediction_cap)

        used_conformal = False
        used_spread_fallback = False
        used_stacking = False

        if champion in {"sarima", "lgbm", "classical"}:
            stacked = {
                "sarima": sarima,
                "lgbm": lgbm,
                "classical": classical,
            }[champion].copy()
            weights = ModelWeightBreakdown(
                sarima=1.0 if champion == "sarima" else 0.0,
                lgbm=1.0 if champion == "lgbm" else 0.0,
                classical=1.0 if champion == "classical" else 0.0,
            )
        elif resolve_stacking_path(demand_segment) is not None:
            used_stacking = True
            stacker = StackingMetaLearner(segment=demand_segment)
            stacker.load()
            stacked, weights = stacker.predict(
                sarima,
                lgbm,
                classical,
                prediction_cap=prediction_cap,
                demand_segment=demand_segment,
                history_days=history_days,
                recent_cv2=drug_cv2,
                classical_available=classical_available,
                horizon_steps=horizon_steps,
            )
            stacked = dampen_stacked_horizon_drift(
                stacked,
                demand_segment=demand_segment,
                recent_level=recent_level,
                horizon_steps=horizon_steps,
            )
        else:
            available = np.column_stack([sarima, lgbm, classical])
            stacked = np.nanmean(available, axis=1)
            n_models = np.sum(~np.isnan(available), axis=1)
            stacked = np.where(n_models == 0, 0.0, stacked)
            active = (~np.isnan(available)).sum(axis=0)
            total_active = max(int(active.sum()), 1)
            weights = ModelWeightBreakdown(
                sarima=float(active[0] / total_active) if active[0] else 0.0,
                lgbm=float(active[1] / total_active) if active[1] else 0.0,
                classical=float(active[2] / total_active) if active[2] else 0.0,
            )
            if weights.classical == 0.0 and weights.sarima + weights.lgbm > 0:
                total = weights.sarima + weights.lgbm
                weights = ModelWeightBreakdown(
                    sarima=weights.sarima / total,
                    lgbm=weights.lgbm / total,
                    classical=0.0,
                )

        if _conformal_available(demand_segment):
            try:
                calibrator = ConformalCalibrator(segment=demand_segment)
                calibrator.load()
                champion_point = stacked.copy()
                p50, p10, p90 = calibrator.predict_interval(
                    champion_point,
                    horizon_steps=horizon_steps,
                )
                p50 = champion_point
                used_conformal = True
            except (ValueError, RuntimeError, FileNotFoundError) as exc:
                logger.warning(
                    "Conformal interval prediction failed for segment '%s' — "
                    "using model spread fallback: %s",
                    demand_segment,
                    exc,
                )
                p50, p10, p90 = self._fallback_intervals(stacked, sarima, lgbm, classical)
                used_spread_fallback = True
        else:
            p50, p10, p90 = self._fallback_intervals(stacked, sarima, lgbm, classical)
            used_spread_fallback = True

        if used_spread_fallback:
            logger.info(
                "Forecast for segment '%s' used spread-fallback intervals "
                "(conformal_available=%s, conformal_used=%s)",
                demand_segment,
                _conformal_available(demand_segment),
                used_conformal,
            )

        p5, p10, p50, p90, p95 = extend_tail_quantiles(p50, p10, p90)
        health = InferenceHealth(
            demand_segment=demand_segment,
            used_stacking=used_stacking,
            used_conformal=used_conformal,
            used_spread_fallback=used_spread_fallback,
        )
        return (
            p5[:horizon_days],
            p10[:horizon_days],
            p50[:horizon_days],
            p90[:horizon_days],
            p95[:horizon_days],
            weights,
            health,
        )

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

    def _history_for_chart(
        self,
        corrected_df: pd.DataFrame,
        *,
        max_days: int = FORECAST_CHART_HISTORY_DAYS,
    ) -> list[HistoryPoint]:
        """Return recent raw receipt demand for chart context (not imputed values)."""
        indexed = ensure_demand_date_index(corrected_df)
        tail = indexed.tail(max_days)
        qty_col = (
            "observed_quantity"
            if "observed_quantity" in tail.columns
            else "total_quantity"
        )
        history: list[HistoryPoint] = []
        for demand_ts, row in tail.iterrows():
            demand_date = pd.Timestamp(demand_ts).date()
            history.append(
                HistoryPoint(
                    date=demand_date,
                    quantity=float(row[qty_col]),
                )
            )
        return history

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
        corrected_df: pd.DataFrame,
        drug_code: str,
    ) -> Optional[list[AttentionWeight]]:
        """Attention is deprecated — TFT no longer powers the active ensemble."""
        del corrected_df, drug_code
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
                    model_weight_classical=weights.classical,
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
        corrected_df = filter_covered_rows(corrected_df)
        corrected_df.attrs["drug_code"] = drug_code
        demand_segment = classify_demand_segment_from_frame(corrected_df)
        qty_col = (
            "observed_quantity"
            if "observed_quantity" in corrected_df.columns
            else "total_quantity"
        )
        drug_cv2 = recent_cv2(corrected_df[qty_col].astype(float).values)
        champion = load_champion(drug_code, demand_segment)
        qty_series = corrected_df[qty_col].astype(float)
        recent_tail = qty_series.tail(28)
        positive_recent = recent_tail[recent_tail > 0]
        recent_level = (
            float(np.median(positive_recent))
            if not positive_recent.empty
            else float(np.median(recent_tail)) if not recent_tail.empty else 0.0
        )
        validation_window = WALK_FORWARD_N_SPLITS * WALK_FORWARD_TEST_HORIZON
        validation_zero_fraction = zero_actual_fraction(
            qty_series.tail(validation_window).values,
        )
        smape_unreliable = smape_unreliable_for_series(
            qty_series.tail(validation_window).values,
            demand_segment=demand_segment,
        )
        primary_metric = primary_validation_metric_for_segment(demand_segment)
        metrics_note = validation_metrics_note(
            demand_segment=demand_segment,
            zero_fraction=validation_zero_fraction,
            smape_unreliable=smape_unreliable,
        )

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
        p5, p10, p50, p90, p95, weights, inference_health = self._ensemble_forecast(
            model_preds,
            horizon_days,
            prediction_cap=prediction_cap,
            demand_segment=demand_segment,
            history_days=len(corrected_df),
            drug_cv2=drug_cv2,
            classical_available="classical" in trained_models
            and not np.all(np.isnan(model_preds["classical"])),
            champion=champion,
            recent_level=recent_level,
        )
        forecast_dates = self._forecast_dates(corrected_df, horizon_days)

        ven_class = self._lookup_ven_class(drug_code, db_session)
        op_quantile, recommended = recommended_quantities(p10, p50, p90, ven_class)
        recommended_total = float(np.sum(recommended))

        shap_features: Optional[list[ShapFeature]] = None
        if include_shap and "lgbm" in trained_models:
            lgbm = LightGBMModel()
            lgbm.load(drug_code)
            shap_features = self._compute_shap_features(drug_code, lgbm, corrected_df)

        attention_weights: Optional[list[AttentionWeight]] = None
        if include_attention:
            attention_weights = self._extract_attention_weights(corrected_df, drug_code)

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

        validation = self._latest_validation_metrics(
            drug_code,
            db_session,
            preferred_model=champion if champion in {"sarima", "lgbm", "classical"} else None,
        )

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
                    recommended_quantity=float(recommended[idx]),
                )
                for idx in range(horizon_days)
            ],
            history=self._history_for_chart(corrected_df),
            shap_features=shap_features,
            attention_weights=attention_weights,
            uncertainty_note=UNCERTAINTY_NOTE,
            smape_last_validation=validation.get("full_smape"),
            smape_validation_full_window=validation.get("full_smape"),
            smape_validation_normal_supply=validation.get("normal_smape"),
            mase_validation_full_window=validation.get("full_mase"),
            mase_validation_normal_supply=validation.get("normal_mase"),
            mase_beats_baseline_full_window=validation.get("mase_beats_baseline_full"),
            mase_beats_baseline_normal_supply=validation.get("mase_beats_baseline_normal"),
            mase_skill_validation_full_window=validation.get("mase_skill_full"),
            smape_validation_7day_total=validation.get("smape_7day_full"),
            smape_validation_30day_total=validation.get("smape_30day_full"),
            mase_validation_7day_total=validation.get("mase_7day_full"),
            mase_validation_30day_total=validation.get("mase_30day_full"),
            smape_validation_7day_normal_supply=validation.get("smape_7day_normal"),
            smape_validation_30day_normal_supply=validation.get("smape_30day_normal"),
            mase_validation_7day_normal_supply=validation.get("mase_7day_normal"),
            mase_validation_30day_normal_supply=validation.get("mase_30day_normal"),
            ven_class=ven_class,
            operating_quantile=op_quantile,
            recommended_quantity_total=recommended_total,
            inference_health=inference_health,
            demand_segment=demand_segment,
            validation_zero_actual_fraction=validation_zero_fraction,
            smape_validation_unreliable=smape_unreliable,
            primary_validation_metric=primary_metric,
            validation_metrics_note=metrics_note,
        )

        for warning in check_forecast_interval_sanity(response):
            logger.warning("Forecast interval sanity: %s", warning)

        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "Forecast for %s completed in %.1fms (horizon=%d, segment=%s, shap=%s, attention=%s)",
            drug_code,
            elapsed_ms,
            horizon_days,
            demand_segment,
            include_shap,
            include_attention,
        )
        return response
