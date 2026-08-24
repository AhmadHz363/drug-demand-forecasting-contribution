"""Top-level forecasting training orchestrator."""

from __future__ import annotations

import logging
import uuid
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.forecasting.censored_demand.corrector import correct_demand
from app.forecasting.model_adaptation import clear_sarima_order_cache
from app.forecasting.constants import (
    DEMAND_SEGMENTS,
    GLOBAL_DEMAND_SEGMENT,
    MIN_DRUGS_FOR_CONFORMAL,
    MIN_HISTORY_DAYS_SARIMA,
    SEGMENT_ENSEMBLE_MIN_SAMPLES,
)
from app.forecasting.data_quality import (
    QUALITY_FLAGGED,
    QUALITY_REJECTED,
    assess_series_quality,
)
from app.forecasting.demand_segmentation import classify_demand_segment_from_frame
from app.forecasting.drift_detection import assess_metric_drift
from app.forecasting.evaluation_metrics import (
    finite_mase_or_none,
    mase as compute_mase,
    mean_smape,
    rolling_validation_metrics,
    smape as smape_single,
)
from app.forecasting.model_adaptation import recent_cv2
from app.forecasting.ensemble.conformal import ConformalCalibrator
from app.forecasting.ensemble.stacking import StackingMetaLearner
from app.forecasting.feature_engineering.pipeline import build_feature_matrix, filter_covered_rows
from app.forecasting.models.base_model import BaseForecastingModel
from app.forecasting.models.classical_model import ClassicalModel
from app.forecasting.models.lgbm_model import LightGBMModel
from app.forecasting.models.sarima_model import SarimaModel
from app.forecasting.training.champion_selection import (
    select_and_persist_champions,
)
from app.forecasting.schemas import DrugQualitySummary, TrainStatusResponse
from app.forecasting.training.walk_forward import (
    WalkForwardResult,
    collect_walk_forward_predictions,
    walk_forward_full,
    walk_forward_coverage,
    walk_forward_mase,
    walk_forward_smape,
)
from app.models.model_performance import ModelPerformance
from app.services.demand_aggregation import get_receipt_date_bounds

logger = logging.getLogger(__name__)

MODEL_REGISTRY: dict[str, type[BaseForecastingModel]] = {
    "sarima": SarimaModel,
    "lgbm": LightGBMModel,
    "classical": ClassicalModel,
}

# SHIELD-XR replaces the per-drug ensemble; legacy model names still accepted for API compat.
SHIELD_XR_MODEL_ALIASES = {
    "shield_xr",
    "plain_tweedie",
    "plain_l1",
    "sarima",
    "lgbm",
    "classical",
}


class ForecastingTrainer:
    """Runs feature engineering, model training, ensemble fitting, and persistence."""

    def _resolve_demand_range(
        self,
        db_session: Session,
        drug_code: str,
    ) -> Optional[tuple[date, date]]:
        start_date, end_date = get_receipt_date_bounds(db_session, drug_code)
        if start_date is None or end_date is None:
            logger.warning(
                "No receipt history in drug_receipts for %s — skipping",
                drug_code,
            )
            return None
        return start_date, end_date

    def _build_corrected_frame(
        self,
        drug_code: str,
        db_session: Session,
    ) -> Optional[pd.DataFrame]:
        date_range = self._resolve_demand_range(db_session, drug_code)
        if date_range is None:
            return None

        start_date, end_date = date_range
        feature_df = build_feature_matrix(
            drug_code,
            center_syn_id=None,
            db_session=db_session,
            start_date=start_date,
            end_date=end_date,
        )
        corrected = correct_demand(
            drug_code,
            center_syn_id=None,
            db_session=db_session,
            feature_df=feature_df,
        )
        # Training / segmentation / metrics use covered ledger days only.
        covered = filter_covered_rows(corrected)
        if covered.empty:
            return corrected
        # Preserve gap flag column as all zeros on the filtered frame.
        if "is_coverage_gap" not in covered.columns:
            covered = covered.copy()
            covered["is_coverage_gap"] = 0
        return covered

    def _train_single_model(
        self,
        model_name: str,
        model: BaseForecastingModel,
        corrected_df: pd.DataFrame,
        drug_code: str,
        force_retrain: bool,
    ) -> tuple[bool, Optional[str], Optional[float], Optional[float], Optional[float], Optional[WalkForwardResult]]:
        if not force_retrain and model.is_trained(drug_code):
            logger.info(
                "Skipping %s for %s — already trained",
                model_name,
                drug_code,
            )
            return False, None, None, None, None, None

        model.train(corrected_df, drug_code)
        artifact_path = model.save(drug_code)

        # Single walk-forward pass replaces the previous 4 separate loops
        # (smape, coverage, mase, OOF collection), cutting fold trainings
        # from 4 × n_splits down to n_splits per model per drug.
        try:
            wf = walk_forward_full(model, corrected_df, drug_code)
        except ValueError as exc:
            logger.warning("Walk-forward skipped for %s/%s: %s", model_name, drug_code, exc)
            return True, artifact_path, float("nan"), 0.0, None, None

        return True, artifact_path, wf.smape, wf.coverage, wf.mase, wf

    def _fit_ensemble(
        self,
        sarima_preds: np.ndarray,
        lgbm_preds: np.ndarray,
        classical_preds: np.ndarray,
        actuals: np.ndarray,
        *,
        segment: str = GLOBAL_DEMAND_SEGMENT,
        history_days: Optional[np.ndarray] = None,
        recent_cv2_values: Optional[np.ndarray] = None,
        horizon_steps: Optional[np.ndarray] = None,
        fit_conformal: bool = True,
    ) -> tuple[list[str], StackingMetaLearner, ConformalCalibrator]:
        artifacts_saved: list[str] = []
        n = len(actuals)
        if n < 10:
            logger.warning(
                "Insufficient validation samples (%d) for ensemble fitting (segment=%s)",
                n,
                segment,
            )
            return artifacts_saved, StackingMetaLearner(segment=segment), ConformalCalibrator(segment=segment)

        cal_start = max(int(n * 0.8), n - max(n // 5, 2))
        cal_start = min(cal_start, n - 2)

        fit_history = history_days[:cal_start] if history_days is not None else 365
        fit_cv2 = recent_cv2_values[:cal_start] if recent_cv2_values is not None else 0.0
        fit_steps = horizon_steps[:cal_start] if horizon_steps is not None else None

        stacker = StackingMetaLearner(segment=segment)
        stacker.fit(
            sarima_preds[:cal_start],
            lgbm_preds[:cal_start],
            classical_preds[:cal_start],
            actuals[:cal_start],
            demand_segment=segment if segment != GLOBAL_DEMAND_SEGMENT else "smooth",
            history_days=fit_history,
            recent_cv2=fit_cv2,
            horizon_steps=fit_steps,
        )
        artifacts_saved.append(stacker.save())

        cal_history = history_days[cal_start:] if history_days is not None else 365
        cal_cv2 = recent_cv2_values[cal_start:] if recent_cv2_values is not None else 0.0
        cal_steps = horizon_steps[cal_start:] if horizon_steps is not None else None
        classical_available = not np.all(np.isnan(classical_preds[cal_start:]))

        stacked_cal, _ = stacker.predict(
            sarima_preds[cal_start:],
            lgbm_preds[cal_start:],
            classical_preds[cal_start:],
            demand_segment=segment if segment != GLOBAL_DEMAND_SEGMENT else "smooth",
            history_days=cal_history,
            recent_cv2=cal_cv2,
            classical_available=classical_available,
            horizon_steps=cal_steps,
        )
        calibrator = ConformalCalibrator(segment=segment)
        if fit_conformal:
            calibrator.fit(stacked_cal, actuals[cal_start:], horizon_steps=cal_steps)
            artifacts_saved.append(calibrator.save())
        return artifacts_saved, stacker, calibrator

    def _empty_segment_bucket(self) -> dict[str, list]:
        return {
            "sarima": [],
            "lgbm": [],
            "classical": [],
            "actuals": [],
            "horizon_steps": [],
            "history_days": [],
            "recent_cv2": [],
            "drug_codes": [],
        }

    def _append_segment_predictions(
        self,
        bucket: dict[str, list],
        preds: dict[str, np.ndarray],
        *,
        history_days: int,
        drug_cv2: float,
        drug_code: str,
    ) -> None:
        actuals = np.asarray(preds["actuals"], dtype=float)
        n_pts = len(actuals)
        bucket["drug_codes"].append(drug_code)
        bucket["actuals"].extend(actuals.tolist())
        steps = np.asarray(preds.get("horizon_steps", np.arange(1, n_pts + 1)), dtype=int)
        bucket["horizon_steps"].extend(steps.tolist())
        bucket["history_days"].extend([history_days] * n_pts)
        bucket["recent_cv2"].extend([drug_cv2] * n_pts)
        for name in ("sarima", "lgbm", "classical"):
            values = np.asarray(preds.get(name, np.full(n_pts, np.nan)), dtype=float)
            bucket[name].extend(values.tolist())

    def train_all(
        self,
        drug_codes: list[str],
        models_to_train: list[str],
        db_session: Session,
        force_retrain: bool = False,
    ) -> TrainStatusResponse:
        """
        Train the hospital-wide SHIELD-XR ensemble (AnomalyGuard + hurdle stack +
        3-way class-conditional ensemble + weekly breakdown). Cleaned training
        panel rows are persisted to ``forecast_training_data``.
        """
        if not models_to_train or all(name in SHIELD_XR_MODEL_ALIASES for name in models_to_train):
            from app.forecasting.shield_xr.trainer import (
                ShieldXRTrainer,
                resolve_training_drug_codes,
            )

            codes = resolve_training_drug_codes(db_session, drug_codes or None)
            return ShieldXRTrainer().train_all(
                codes,
                models_to_train,
                db_session,
                force_retrain=force_retrain,
            )

        valid_models = [name for name in models_to_train if name in MODEL_REGISTRY]
        unknown = sorted(set(models_to_train) - set(valid_models))
        if unknown:
            logger.warning("Ignoring unknown model names: %s", ", ".join(unknown))
        if not valid_models:
            raise ValueError("No valid models requested. Choose from: sarima, lgbm, classical.")

        # Reset the per-drug SARIMA order cache so stale orders from a previous
        # training run don't bleed into the new one.
        clear_sarima_order_cache()

        training_run_id = str(uuid.uuid4())
        artifacts_saved: list[str] = []
        models_trained: set[str] = set()
        per_drug_smape: dict[str, dict[str, float]] = {}
        drugs_trained = 0
        skipped_drugs: list[DrugQualitySummary] = []
        flagged_drugs: list[DrugQualitySummary] = []
        drift_alerts: list[str] = []

        all_sarima: list[float] = []
        all_lgbm: list[float] = []
        all_classical: list[float] = []
        all_actuals: list[float] = []
        # Per-drug OOF results keyed {drug_code: {model_name: WalkForwardResult}}.
        # Used to build ensemble input without an extra walk-forward pass.
        drug_oof_results: dict[str, dict[str, WalkForwardResult]] = {}
        segment_buckets: dict[str, dict[str, list[float]]] = {
            GLOBAL_DEMAND_SEGMENT: self._empty_segment_bucket(),
            **{segment: self._empty_segment_bucket() for segment in DEMAND_SEGMENTS},
        }
        segment_by_drug: dict[str, str] = {}

        for drug_code in drug_codes:
            per_drug_smape[drug_code] = {}
            drug_had_success = False

            try:
                corrected_df = self._build_corrected_frame(drug_code, db_session)
                if corrected_df is None:
                    continue

                quality = assess_series_quality(corrected_df, drug_code)
                if quality.should_skip_training:
                    logger.warning(
                        "Skipping %s — data quality rejected: %s",
                        drug_code,
                        "; ".join(quality.reasons),
                    )
                    skipped_drugs.append(
                        DrugQualitySummary(
                            drug_code=drug_code,
                            status=QUALITY_REJECTED,
                            reasons=quality.reasons,
                        )
                    )
                    continue

                if quality.is_flagged:
                    logger.warning(
                        "Flagged %s for manual review: %s",
                        drug_code,
                        "; ".join(quality.reasons),
                    )
                    flagged_drugs.append(
                        DrugQualitySummary(
                            drug_code=drug_code,
                            status=QUALITY_FLAGGED,
                            reasons=quality.reasons,
                        )
                    )

                demand_segment = classify_demand_segment_from_frame(corrected_df)
                segment_by_drug[drug_code] = demand_segment
                logger.info(
                    "Demand segment for %s: %s",
                    drug_code,
                    demand_segment,
                )

                if len(corrected_df) < MIN_HISTORY_DAYS_SARIMA and "sarima" in valid_models:
                    logger.warning(
                        "SARIMA skipped for %s: only %d days of history (< %d required)",
                        drug_code,
                        len(corrected_df),
                        MIN_HISTORY_DAYS_SARIMA,
                    )

                drug_oof_results[drug_code] = {}

                for model_name in valid_models:
                    model = MODEL_REGISTRY[model_name]()
                    try:
                        trained, path, smape, coverage, mase, wf_result = self._train_single_model(
                            model_name,
                            model,
                            corrected_df,
                            drug_code,
                            force_retrain,
                        )
                        if not trained:
                            continue

                        drug_had_success = True
                        models_trained.add(model_name)
                        artifacts_saved.append(path)
                        per_drug_smape[drug_code][model_name] = float(smape)

                        if wf_result is not None:
                            drug_oof_results[drug_code][model_name] = wf_result

                        prev_row = (
                            db_session.query(ModelPerformance)
                            .filter(
                                ModelPerformance.drug_code == drug_code,
                                ModelPerformance.model_name == model_name,
                            )
                            .order_by(ModelPerformance.evaluated_at.desc())
                            .first()
                        )
                        stored_mase = finite_mase_or_none(mase)
                        stored_mase_normal = (
                            finite_mase_or_none(wf_result.mase_normal_supply)
                            if wf_result is not None
                            else None
                        )
                        stored_rolling: dict[str, float | None] = {}
                        if wf_result is not None:
                            for attr in (
                                "smape_7day_full",
                                "smape_30day_full",
                                "mase_7day_full",
                                "mase_30day_full",
                                "smape_7day_normal",
                                "smape_30day_normal",
                                "mase_7day_normal",
                                "mase_30day_normal",
                            ):
                                val = getattr(wf_result, attr, None)
                                if val is not None and np.isfinite(val):
                                    stored_rolling[attr] = float(val)
                                else:
                                    stored_rolling[attr] = None
                        stored_smape_normal = (
                            float(wf_result.smape_normal_supply)
                            if wf_result is not None
                            and wf_result.smape_normal_supply is not None
                            and np.isfinite(wf_result.smape_normal_supply)
                            else None
                        )
                        drift = assess_metric_drift(
                            float(prev_row.smape) if prev_row else None,
                            float(smape),
                            float(prev_row.mase) if prev_row and prev_row.mase is not None else None,
                            stored_mase,
                        )
                        if drift.has_drift:
                            if drift.smape_degraded and drift.smape_delta_pct is not None:
                                drift_alerts.append(
                                    f"{drug_code}/{model_name}: sMAPE +{drift.smape_delta_pct:.1f}%"
                                )
                            elif drift.mase_degraded and drift.mase_delta_pct is not None:
                                drift_alerts.append(
                                    f"{drug_code}/{model_name}: MASE +{drift.mase_delta_pct:.1f}%"
                                )

                        db_session.add(
                            ModelPerformance(
                                drug_code=drug_code,
                                model_name=model_name,
                                smape=float(smape),
                                smape_normal_supply=stored_smape_normal,
                                coverage_90=float(coverage or 0.0),
                                mase=stored_mase,
                                mase_normal_supply=stored_mase_normal,
                                smape_7day_full=stored_rolling.get("smape_7day_full"),
                                smape_30day_full=stored_rolling.get("smape_30day_full"),
                                mase_7day_full=stored_rolling.get("mase_7day_full"),
                                mase_30day_full=stored_rolling.get("mase_30day_full"),
                                smape_7day_normal=stored_rolling.get("smape_7day_normal"),
                                smape_30day_normal=stored_rolling.get("smape_30day_normal"),
                                mase_7day_normal=stored_rolling.get("mase_7day_normal"),
                                mase_30day_normal=stored_rolling.get("mase_30day_normal"),
                                training_run_id=training_run_id,
                                demand_segment=demand_segment,
                                data_quality_status=quality.status,
                            )
                        )

                    except ValueError as exc:
                        logger.warning(
                            "%s training skipped for %s: %s",
                            model_name,
                            drug_code,
                            exc,
                        )
                    except Exception:
                        logger.exception(
                            "%s training failed for %s",
                            model_name,
                            drug_code,
                        )

                if drug_had_success:
                    drugs_trained += 1

                # Assemble ensemble OOF predictions from already-cached WalkForwardResults
                # when available, avoiding a second walk-forward pass per model.
                stack_models: dict[str, type[BaseForecastingModel]] = {}
                for name in valid_models:
                    if name in per_drug_smape[drug_code]:
                        stack_models[name] = MODEL_REGISTRY[name]
                    elif not force_retrain and MODEL_REGISTRY[name]().is_trained(drug_code):
                        stack_models[name] = MODEL_REGISTRY[name]

                if stack_models:
                    precomputed = drug_oof_results.get(drug_code) or None
                    preds = collect_walk_forward_predictions(
                        corrected_df,
                        drug_code,
                        stack_models,
                        precomputed_oof=precomputed,
                    )
                    if preds is not None:
                        qty_col = (
                            "observed_quantity"
                            if "observed_quantity" in corrected_df.columns
                            else "total_quantity"
                        )
                        drug_cv2 = recent_cv2(
                            corrected_df[qty_col].astype(float).values,
                        )
                        history_len = len(corrected_df)
                        self._append_segment_predictions(
                            segment_buckets[GLOBAL_DEMAND_SEGMENT],
                            preds,
                            history_days=history_len,
                            drug_cv2=drug_cv2,
                            drug_code=drug_code,
                        )
                        self._append_segment_predictions(
                            segment_buckets[demand_segment],
                            preds,
                            history_days=history_len,
                            drug_cv2=drug_cv2,
                            drug_code=drug_code,
                        )
                        actuals = np.asarray(preds["actuals"], dtype=float)
                        n_pts = len(actuals)
                        all_actuals.extend(actuals.tolist())
                        all_sarima.extend(
                            np.asarray(
                                preds.get("sarima", np.full(n_pts, np.nan)),
                                dtype=float,
                            ).tolist()
                        )
                        all_lgbm.extend(
                            np.asarray(
                                preds.get("lgbm", np.full(n_pts, np.nan)),
                                dtype=float,
                            ).tolist()
                        )
                        all_classical.extend(
                            np.asarray(
                                preds.get("classical", np.full(n_pts, np.nan)),
                                dtype=float,
                            ).tolist()
                        )

            except Exception:
                logger.exception("Failed processing drug %s", drug_code)
                try:
                    db_session.rollback()
                except Exception:  # noqa: BLE001
                    pass

        ensemble_artifacts: list[str] = []
        segments_to_fit = [GLOBAL_DEMAND_SEGMENT, *DEMAND_SEGMENTS]
        for segment in segments_to_fit:
            bucket = segment_buckets[segment]
            n_drugs = len(set(bucket["drug_codes"]))
            if len(bucket["actuals"]) < SEGMENT_ENSEMBLE_MIN_SAMPLES:
                if segment != GLOBAL_DEMAND_SEGMENT:
                    logger.info(
                        "Skipping segment '%s' ensemble — only %d OOF samples "
                        "(<%d required); inference will fall back to global.",
                        segment,
                        len(bucket["actuals"]),
                        SEGMENT_ENSEMBLE_MIN_SAMPLES,
                    )
                    continue
                if not bucket["actuals"]:
                    continue

            if (
                segment != GLOBAL_DEMAND_SEGMENT
                and n_drugs < MIN_DRUGS_FOR_CONFORMAL
            ):
                logger.info(
                    "Skipping segment '%s' conformal — only %d drugs contributed "
                    "(<%d required); inference will use global conformal.",
                    segment,
                    n_drugs,
                    MIN_DRUGS_FOR_CONFORMAL,
                )
                # Still fit stacking when sample count is sufficient; conformal
                # for this segment is skipped inside _fit_ensemble below.
                try:
                    segment_artifacts, _, _ = self._fit_ensemble(
                        np.asarray(bucket["sarima"], dtype=float),
                        np.asarray(bucket["lgbm"], dtype=float),
                        np.asarray(bucket["classical"], dtype=float),
                        np.asarray(bucket["actuals"], dtype=float),
                        segment=segment,
                        history_days=np.asarray(bucket["history_days"], dtype=float),
                        recent_cv2_values=np.asarray(bucket["recent_cv2"], dtype=float),
                        horizon_steps=np.asarray(bucket["horizon_steps"], dtype=int),
                        fit_conformal=False,
                    )
                    ensemble_artifacts.extend(segment_artifacts)
                except Exception:
                    logger.exception(
                        "Ensemble fitting failed for segment '%s' — continuing",
                        segment,
                    )
                continue

            try:
                segment_artifacts, _, _ = self._fit_ensemble(
                    np.asarray(bucket["sarima"], dtype=float),
                    np.asarray(bucket["lgbm"], dtype=float),
                    np.asarray(bucket["classical"], dtype=float),
                    np.asarray(bucket["actuals"], dtype=float),
                    segment=segment,
                    history_days=np.asarray(bucket["history_days"], dtype=float),
                    recent_cv2_values=np.asarray(bucket["recent_cv2"], dtype=float),
                    horizon_steps=np.asarray(bucket["horizon_steps"], dtype=int),
                )
                ensemble_artifacts.extend(segment_artifacts)
            except Exception:
                logger.exception(
                    "Ensemble fitting failed for segment '%s' — continuing",
                    segment,
                )

        if ensemble_artifacts:
            artifacts_saved.extend(ensemble_artifacts)
        elif all_actuals:
            logger.warning(
                "Segment ensemble fitting produced no artifacts despite pooled OOF data"
            )

        # Champion selection from per-drug OOF (bases + segment stack when available).
        drug_champion_oof: dict[str, dict] = {}
        for drug_code, oof_by_model in drug_oof_results.items():
            if not oof_by_model:
                continue
            first = next(iter(oof_by_model.values()))
            actuals = np.asarray(first.oof_actuals, dtype=float)
            payload: dict = {"actuals": actuals}
            for name in ("sarima", "lgbm", "classical"):
                wf = oof_by_model.get(name)
                if wf is None:
                    continue
                preds = np.asarray(wf.oof_p50, dtype=float)
                if preds.shape != actuals.shape:
                    continue
                payload[name] = preds
            if len(payload) <= 1:
                continue
            try:
                stacker = StackingMetaLearner(segment=GLOBAL_DEMAND_SEGMENT)
                stacker.load()
                ens, _ = stacker.predict(
                    payload.get("sarima", np.full_like(actuals, np.nan)),
                    payload.get("lgbm", np.full_like(actuals, np.nan)),
                    payload.get("classical", np.full_like(actuals, np.nan)),
                    demand_segment=segment_by_drug.get(drug_code, "smooth"),
                    history_days=len(actuals),
                    recent_cv2=0.0,
                )
                payload["ensemble"] = ens
            except Exception:
                payload["ensemble"] = None
            drug_champion_oof[drug_code] = payload

        try:
            champion_paths = select_and_persist_champions(
                drug_oof=drug_champion_oof,
                segment_by_drug=segment_by_drug,
            )
            artifacts_saved.extend(champion_paths)
        except Exception:
            logger.exception("Champion selection failed — continuing with ensemble default")

        # Persist ensemble OOF metrics per drug
        for drug_code, payload in drug_champion_oof.items():
            if "ensemble" not in payload or payload["ensemble"] is None:
                continue
            try:
                actuals = np.asarray(payload["actuals"], dtype=float)
                ens_pred = np.asarray(payload["ensemble"], dtype=float)
                if actuals.size == 0 or ens_pred.size == 0:
                    continue
                
                ens_smape = float(np.mean([smape_single(float(a), float(p)) for a, p in zip(actuals, ens_pred)]))
                ens_mase = finite_mase_or_none(compute_mase(actuals, ens_pred))
                
                # Use walk-forward from first available base model for rolling metrics
                ens_wf = next((drug_oof_results.get(drug_code, {}).get(m) 
                               for m in ["sarima", "lgbm", "classical"] if m in drug_oof_results.get(drug_code, {})), None)
                
                ens_rolling = {}
                if ens_wf and ens_wf.oof_actuals.size > 0:
                    try:
                        stockout_flags = np.zeros(len(actuals), dtype=bool)  # TODO: get real flags
                        rolling_metrics_dict = rolling_validation_metrics(
                            actuals, ens_pred, stockout_flags=stockout_flags
                        )
                        for attr in ["smape_7day_full", "smape_30day_full", "mase_7day_full", "mase_30day_full",
                                     "smape_7day_normal", "smape_30day_normal", "mase_7day_normal", "mase_30day_normal"]:
                            val = rolling_metrics_dict.get(attr)
                            if val is not None and np.isfinite(val):
                                ens_rolling[attr] = float(val)
                    except Exception:
                        pass
                
                db_session.add(
                    ModelPerformance(
                        drug_code=drug_code,
                        model_name="ensemble",
                        smape=float(ens_smape),
                        smape_normal_supply=float(ens_smape),  # TODO: proper normal supply calc
                        coverage_90=0.0,  # Not computed in OOF
                        mase=ens_mase,
                        mase_normal_supply=ens_mase,
                        smape_7day_full=ens_rolling.get("smape_7day_full"),
                        smape_30day_full=ens_rolling.get("smape_30day_full"),
                        mase_7day_full=ens_rolling.get("mase_7day_full"),
                        mase_30day_full=ens_rolling.get("mase_30day_full"),
                        smape_7day_normal=ens_rolling.get("smape_7day_normal"),
                        smape_30day_normal=ens_rolling.get("smape_30day_normal"),
                        mase_7day_normal=ens_rolling.get("mase_7day_normal"),
                        mase_30day_normal=ens_rolling.get("mase_30day_normal"),
                        training_run_id=training_run_id,
                        demand_segment=segment_by_drug.get(drug_code),
                        data_quality_status="pass",
                    )
                )
                logger.info(
                    "Ensemble metrics for %s: sMAPE=%.2f%%, MASE=%s",
                    drug_code,
                    ens_smape,
                    f"{ens_mase:.3f}" if ens_mase else "N/A",
                )
            except Exception:
                logger.exception("Failed to persist ensemble metrics for %s", drug_code)

        db_session.commit()

        smape_summary: dict[str, float] = {}
        for model_name in valid_models:
            scores = [
                scores_by_model[model_name]
                for scores_by_model in per_drug_smape.values()
                if model_name in scores_by_model
            ]
            if scores:
                smape_summary[model_name] = float(sum(scores) / len(scores))

        summary_rows = []
        for drug_code, scores_by_model in sorted(per_drug_smape.items()):
            if not scores_by_model:
                continue
            row = {"drug_code": drug_code, **scores_by_model}
            summary_rows.append(row)
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            logger.info("Training summary by drug:\n%s", summary_df.to_string(index=False))

        if drift_alerts:
            logger.warning(
                "Metric drift detected for %d drug/model pairs:\n%s",
                len(drift_alerts),
                "\n".join(drift_alerts[:20]),
            )

        return TrainStatusResponse(
            status="ok",
            training_run_id=training_run_id,
            drugs_trained=drugs_trained,
            models_trained=sorted(models_trained),
            smape_summary=smape_summary,
            artifacts_saved=artifacts_saved,
            skipped_drugs=skipped_drugs,
            flagged_drugs=flagged_drugs,
            drift_alerts=drift_alerts,
        )
