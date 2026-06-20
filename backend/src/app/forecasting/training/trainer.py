"""Top-level forecasting training orchestrator."""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.forecasting.censored_demand.corrector import correct_demand
from app.forecasting.constants import MIN_HISTORY_DAYS_SARIMA
from app.forecasting.ensemble.conformal import ConformalCalibrator
from app.forecasting.ensemble.stacking import StackingMetaLearner
from app.forecasting.feature_engineering.pipeline import build_feature_matrix
from app.forecasting.models.base_model import BaseForecastingModel
from app.forecasting.models.lgbm_model import LightGBMModel
from app.forecasting.models.sarima_model import SarimaModel
from app.forecasting.models.tft_model import TFTModel
from app.forecasting.schemas import TrainStatusResponse
from app.forecasting.training.walk_forward import (
    collect_walk_forward_predictions,
    walk_forward_coverage,
    walk_forward_smape,
)
from app.models.model_performance import ModelPerformance
from app.services.demand_aggregation import get_receipt_date_bounds

logger = logging.getLogger(__name__)

MODEL_REGISTRY: dict[str, type[BaseForecastingModel]] = {
    "sarima": SarimaModel,
    "lgbm": LightGBMModel,
    "tft": TFTModel,
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
        return correct_demand(
            drug_code,
            center_syn_id=None,
            db_session=db_session,
            feature_df=feature_df,
        )

    def _train_single_model(
        self,
        model_name: str,
        model: BaseForecastingModel,
        corrected_df: pd.DataFrame,
        drug_code: str,
        force_retrain: bool,
    ) -> tuple[bool, Optional[str], Optional[float], Optional[float]]:
        if not force_retrain and model.is_trained(drug_code):
            logger.info(
                "Skipping %s for %s — already trained",
                model_name,
                drug_code,
            )
            return False, None, None, None

        model.train(corrected_df, drug_code)
        if model_name == "tft" and getattr(model, "_skipped", False):
            return False, None, None, None

        artifact_path = model.save(drug_code)
        smape = walk_forward_smape(model, corrected_df, drug_code)
        coverage = walk_forward_coverage(model, corrected_df, drug_code)
        return True, artifact_path, smape, coverage

    def _fit_ensemble(
        self,
        sarima_preds: np.ndarray,
        lgbm_preds: np.ndarray,
        tft_preds: np.ndarray,
        actuals: np.ndarray,
    ) -> tuple[list[str], StackingMetaLearner, ConformalCalibrator]:
        artifacts_saved: list[str] = []
        n = len(actuals)
        if n < 10:
            logger.warning("Insufficient validation samples (%d) for ensemble fitting", n)
            return artifacts_saved, StackingMetaLearner(), ConformalCalibrator()

        cal_start = max(int(n * 0.8), n - max(n // 5, 2))
        cal_start = min(cal_start, n - 2)

        stacker = StackingMetaLearner()
        stacker.fit(
            sarima_preds[:cal_start],
            lgbm_preds[:cal_start],
            tft_preds[:cal_start],
            actuals[:cal_start],
        )
        artifacts_saved.append(stacker.save())

        stacked_cal, _ = stacker.predict(
            sarima_preds[cal_start:],
            lgbm_preds[cal_start:],
            tft_preds[cal_start:],
        )
        calibrator = ConformalCalibrator()
        calibrator.fit(stacked_cal, actuals[cal_start:])
        artifacts_saved.append(calibrator.save())
        return artifacts_saved, stacker, calibrator

    def train_all(
        self,
        drug_codes: list[str],
        models_to_train: list[str],
        db_session: Session,
        force_retrain: bool = False,
    ) -> TrainStatusResponse:
        """
        For each drug:
        1. build_feature_matrix (feature engineering)
        2. correct_demand (censored demand)
        3. train each requested model
        4. walk_forward_smape for each model
        5. Fit StackingMetaLearner across all drugs' validation predictions
        6. Fit ConformalCalibrator on held-out set
        7. Save all artifacts
        8. Write ModelPerformance rows to DB
        """
        valid_models = [name for name in models_to_train if name in MODEL_REGISTRY]
        unknown = sorted(set(models_to_train) - set(valid_models))
        if unknown:
            logger.warning("Ignoring unknown model names: %s", ", ".join(unknown))
        if not valid_models:
            raise ValueError("No valid models requested. Choose from: sarima, lgbm, tft.")

        artifacts_saved: list[str] = []
        models_trained: set[str] = set()
        per_drug_smape: dict[str, dict[str, float]] = {}
        drugs_trained = 0

        all_sarima: list[float] = []
        all_lgbm: list[float] = []
        all_tft: list[float] = []
        all_actuals: list[float] = []

        for drug_code in drug_codes:
            per_drug_smape[drug_code] = {}
            drug_had_success = False

            try:
                corrected_df = self._build_corrected_frame(drug_code, db_session)
                if corrected_df is None:
                    continue

                if len(corrected_df) < MIN_HISTORY_DAYS_SARIMA and "sarima" in valid_models:
                    logger.warning(
                        "SARIMA skipped for %s: only %d days of history (< %d required)",
                        drug_code,
                        len(corrected_df),
                        MIN_HISTORY_DAYS_SARIMA,
                    )

                for model_name in valid_models:
                    model = MODEL_REGISTRY[model_name]()
                    try:
                        trained, path, smape, coverage = self._train_single_model(
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

                        db_session.add(
                            ModelPerformance(
                                drug_code=drug_code,
                                model_name=model_name,
                                smape=float(smape),
                                coverage_90=float(coverage or 0.0),
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

                stack_models: dict[str, type[BaseForecastingModel]] = {}
                for name in valid_models:
                    if name in per_drug_smape[drug_code]:
                        stack_models[name] = MODEL_REGISTRY[name]
                    elif not force_retrain and MODEL_REGISTRY[name]().is_trained(drug_code):
                        stack_models[name] = MODEL_REGISTRY[name]

                if stack_models:
                    preds = collect_walk_forward_predictions(
                        corrected_df,
                        drug_code,
                        stack_models,
                    )
                    if preds is not None:
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
                        all_tft.extend(
                            np.asarray(
                                preds.get("tft", np.full(n_pts, np.nan)),
                                dtype=float,
                            ).tolist()
                        )

            except Exception:
                logger.exception("Failed processing drug %s", drug_code)

        ensemble_artifacts: list[str] = []
        if all_actuals:
            try:
                sarima_arr = np.asarray(all_sarima, dtype=float)
                lgbm_arr = np.asarray(all_lgbm, dtype=float)
                tft_arr = np.asarray(all_tft, dtype=float)
                actuals_arr = np.asarray(all_actuals, dtype=float)

                ensemble_artifacts, _, _ = self._fit_ensemble(
                    sarima_arr,
                    lgbm_arr,
                    tft_arr,
                    actuals_arr,
                )
                artifacts_saved.extend(ensemble_artifacts)
            except Exception:
                logger.exception("Ensemble fitting failed — per-drug models were still saved")

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

        return TrainStatusResponse(
            status="ok",
            drugs_trained=drugs_trained,
            models_trained=sorted(models_trained),
            smape_summary=smape_summary,
            artifacts_saved=artifacts_saved,
        )
