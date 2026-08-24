"""SHIELD-XR training orchestrator."""

from __future__ import annotations

import logging
import uuid
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.forecasting.schemas import TrainStatusResponse
from app.forecasting.shield_xr.anomaly_guard import apply_anomaly_guard
from app.forecasting.shield_xr.artifacts import (
    is_trained,
    load_training_meta,
    save_daily_artifacts,
    save_training_meta,
    save_weekly_artifacts,
)
from app.forecasting.shield_xr.daily_model import (
    DailyEnsembleArtifacts,
    select_class_ensemble,
    train_plain_models,
    train_shieldxr_stack,
)
from app.forecasting.shield_xr.evaluation import compute_training_metrics
from app.forecasting.shield_xr.features import (
    FEATURES_X,
    add_features,
    add_sample_weights,
    apply_spike_labels,
    assign_sb_classes,
    encode_ids,
)
from app.forecasting.shield_xr.metrics import accuracy_from_wape, evaluate, smape_from_wape
from app.forecasting.shield_xr.panel_builder import build_hospital_panel, split_temporal
from app.forecasting.shield_xr.persistence import persist_training_panel
from app.forecasting.shield_xr.weekly_model import (
    build_weekly_panel,
    reconcile_weekly_breakdown,
    train_weekly_breakdown,
)
from app.models.model_performance import ModelPerformance
from app.services.enriched_demand import (
    enriched_data_exists,
    get_distinct_drug_codes_from_enriched,
)

logger = logging.getLogger(__name__)

SHIELD_XR_MODELS = ["shield_xr", "plain_tweedie", "plain_l1"]
LEGACY_MODEL_ALIASES = {"sarima", "lgbm", "classical", *SHIELD_XR_MODELS}


class ShieldXRTrainer:
    """Train the hospital-wide SHIELD-XR ensemble and persist cleaned panel data."""

    def _prepare_splits(
        self,
        panel: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
        if panel.empty:
            raise ValueError("No demand history available to train SHIELD-XR.")

        d_train_end, d_valid_end = split_temporal(panel)
        guarded = apply_anomaly_guard(panel, d_train_end)
        cleaned = assign_sb_classes(guarded.frame, d_train_end)
        feat = add_features(cleaned)

        train = feat[feat["DATE"] <= d_train_end].copy()
        valid = feat[(feat["DATE"] > d_train_end) & (feat["DATE"] <= d_valid_end)].copy()
        test = feat[feat["DATE"] > d_valid_end].copy()
        if train.empty or valid.empty:
            raise ValueError("Insufficient history for SHIELD-XR temporal split.")

        train_stats = train.groupby("CODE")["demand"].agg(mean="mean", std="std")
        train = apply_spike_labels(train, train_stats)
        valid = apply_spike_labels(valid, train_stats)
        test = apply_spike_labels(test, train_stats)

        all_codes = sorted(feat["CODE"].astype(str).unique())
        all_cats = sorted(feat["CAT"].astype(str).fillna("UNK").unique())
        encode_ids(train, valid, test, all_codes, all_cats)
        add_sample_weights(train, valid, test)
        return cleaned, train, valid, test, d_train_end, d_valid_end

    def train_all(
        self,
        drug_codes: list[str],
        models_to_train: list[str],
        db_session: Session,
        force_retrain: bool = False,
    ) -> TrainStatusResponse:
        del models_to_train, drug_codes  # SHIELD-XR trains on the full enriched hospital panel
        if not enriched_data_exists(db_session):
            raise ValueError(
                "No enriched training data found. Upload hospital Excel exports via Data Ingestion first."
            )
        if not force_retrain and is_trained():
            meta = load_training_meta()
            logger.info("Skipping SHIELD-XR training — artifacts present (force_retrain=False)")
            per_sb = meta.get("per_sb_class_weekly_accuracy") or {}
            test_acc = meta.get("test_accuracy")
            return TrainStatusResponse(
                status="skipped",
                training_run_id=str(meta.get("training_run_id") or ""),
                drugs_trained=int(meta.get("n_skus") or 0),
                models_trained=SHIELD_XR_MODELS,
                smape_summary={},
                artifacts_saved=[],
                accuracy_summary={
                    "daily_ensemble_accuracy_pct": round(float(test_acc) * 100, 2)
                    if test_acc is not None
                    else None,
                    "hospital_weekly_accuracy_pct": round(
                        float(meta["hospital_weekly_accuracy"]) * 100,
                        2,
                    )
                    if meta.get("hospital_weekly_accuracy") is not None
                    else None,
                    "reconciled_weekly_per_drug_mean_accuracy_pct": round(
                        float(meta["reconciled_weekly_per_drug_mean_accuracy"]) * 100,
                        2,
                    )
                    if meta.get("reconciled_weekly_per_drug_mean_accuracy") is not None
                    else None,
                    "reconciled_weekly_per_drug_median_accuracy_pct": round(
                        float(meta["reconciled_weekly_per_drug_median_accuracy"]) * 100,
                        2,
                    )
                    if meta.get("reconciled_weekly_per_drug_median_accuracy") is not None
                    else None,
                    "per_sb_class_weekly_accuracy_pct": {
                        str(key): round(float(value) * 100, 2) for key, value in per_sb.items()
                    },
                },
            )

        training_run_id = str(uuid.uuid4())
        panel = build_hospital_panel(db_session)
        if panel.empty:
            raise ValueError("Enriched panel is empty — ingest hospital Excel data before training.")

        cleaned, train, valid, test, d_train_end, d_valid_end = self._prepare_splits(panel)

        feature_cols = [c for c in FEATURES_X if c in train.columns]
        shield_xr = train_shieldxr_stack(train, valid, test, feature_cols)
        plain, plain_l1, plain_pred, plain_pred_valid, plain_l1_pred, plain_l1_pred_valid = train_plain_models(
            train, valid, test, feature_cols
        )
        class_winner, class_bias, ensemble_pred = select_class_ensemble(
            valid,
            test,
            shield_xr,
            plain_pred_valid,
            plain_pred,
            plain_l1_pred_valid,
            plain_l1_pred,
        )

        mask_ok = ~test["is_anomaly_day"].values if len(test) else np.array([], dtype=bool)
        if mask_ok.any():
            test_metrics = evaluate(test.loc[mask_ok, "demand_raw"], ensemble_pred[mask_ok], "ensemble")
        else:
            test_metrics = evaluate(test["demand"], ensemble_pred, "ensemble")

        residuals = np.abs(test["demand"].values - ensemble_pred) if len(test) else np.array([1.0])
        residual_q90 = float(np.quantile(residuals, 0.90)) if len(residuals) else 1.0

        all_codes = sorted(cleaned["CODE"].astype(str).unique())
        all_cats = sorted(cleaned["CAT"].astype(str).fillna("UNK").unique())
        code2id = {code: idx for idx, code in enumerate(all_codes)}
        cat2id = {cat: idx for idx, cat in enumerate(all_cats)}

        daily_artifacts = DailyEnsembleArtifacts(
            shield_xr=shield_xr,
            plain_tweedie=plain,
            plain_l1=plain_l1,
            plain_pred_valid=plain_pred_valid,
            plain_pred=plain_pred,
            plain_l1_pred_valid=plain_l1_pred_valid,
            plain_l1_pred=plain_l1_pred,
            class_winner=class_winner,
            class_bias=class_bias,
            feature_cols=feature_cols,
            code2id=code2id,
            cat2id=cat2id,
            residual_q90=residual_q90,
        )
        artifacts_saved = [save_daily_artifacts(daily_artifacts)]

        feat_full = add_features(cleaned)
        weekly_sku, weekly_features = build_weekly_panel(feat_full, d_train_end, d_valid_end)
        weekly_artifacts = train_weekly_breakdown(weekly_sku, weekly_features, code2id, cat2id)
        weekly_test = pd.DataFrame()
        if len(test):
            weekly_test = reconcile_weekly_breakdown(weekly_artifacts, test, ensemble_pred)
        artifacts_saved.append(save_weekly_artifacts(weekly_artifacts))

        training_metrics = compute_training_metrics(
            test,
            ensemble_pred,
            weekly_test,
            weekly_valid=weekly_artifacts.weekly_valid_panel,
            weekly_train=weekly_artifacts.weekly_train_panel,
        )

        persist_rows = cleaned[
            [
                "CODE",
                "DATE",
                "demand_raw",
                "demand_clean",
                "ARTICLE",
                "CAT",
                "sb_class",
                "is_anomaly_day",
                "is_dropout_day",
                "ceiling",
            ]
        ].copy()
        persist_training_panel(db_session, training_run_id, persist_rows)

        db_session.query(ModelPerformance).filter(
            ModelPerformance.model_name == "shield_xr_ensemble",
        ).delete(synchronize_session=False)

        accuracy = float(test_metrics.get("Accuracy", 0.0))
        for drug_code in all_codes:
            if drug_code in training_metrics.per_drug_weekly_accuracy:
                per_drug_acc = training_metrics.per_drug_weekly_accuracy[drug_code]
            else:
                drug_mask = test["CODE"].astype(str).values == drug_code
                if mask_ok.any():
                    drug_mask = drug_mask & mask_ok
                if drug_mask.any():
                    per_drug_acc = accuracy_from_wape(
                        test.loc[drug_mask, "demand_raw"].values,
                        ensemble_pred[drug_mask],
                    )
                else:
                    per_drug_acc = accuracy
            per_drug_smape = max(0.0, (1.0 - per_drug_acc) * 200.0)

            db_session.add(
                ModelPerformance(
                    drug_code=drug_code,
                    model_name="shield_xr_ensemble",
                    smape=per_drug_smape,
                    smape_normal_supply=per_drug_smape,
                    mase=max(0.0, 1.0 - per_drug_acc),
                    mase_normal_supply=max(0.0, 1.0 - per_drug_acc),
                    coverage_90=0.0,
                    demand_segment=str(
                        cleaned.loc[cleaned["CODE"] == drug_code, "sb_class"].iloc[0]
                    ),
                    data_quality_status="ok",
                    training_run_id=training_run_id,
                    weight_sarima=0.0,
                    weight_lgbm=1.0,
                    weight_classical=0.0,
                )
            )

        meta = {
            "training_run_id": training_run_id,
            "train_end": str(d_train_end.date()),
            "valid_end": str(d_valid_end.date()),
            "n_skus": len(all_codes),
            "class_winner": class_winner,
            "class_bias": class_bias,
            "gate_config": shield_xr.gate_config,
            "test_accuracy": accuracy,
            "test_wape": test_metrics.get("WAPE"),
            "hospital_weekly_accuracy": training_metrics.hospital_weekly_accuracy,
            "hospital_weekly_wape": training_metrics.hospital_weekly_wape,
            "reconciled_weekly_per_drug_mean_accuracy": (
                training_metrics.reconciled_weekly_per_drug_mean_accuracy
            ),
            "reconciled_weekly_per_drug_median_accuracy": (
                training_metrics.reconciled_weekly_per_drug_median_accuracy
            ),
            "volume_weighted_weekly_accuracy": training_metrics.volume_weighted_weekly_accuracy,
            "non_lumpy_weekly_accuracy": training_metrics.non_lumpy_weekly_accuracy,
            "hybrid_abc_combined_accuracy": training_metrics.hybrid_abc_combined_accuracy,
            "n_hybrid_abc_named_drugs": training_metrics.n_hybrid_abc_named_drugs,
            "per_sb_class_weekly_accuracy": training_metrics.per_sb_class_weekly_accuracy,
        }
        artifacts_saved.append(save_training_meta(meta))
        db_session.commit()

        logger.info(
            "SHIELD-XR training complete: run=%s skus=%d daily_acc=%.1f%% weekly_hosp_acc=%.1f%% "
            "weekly_drug_mean_acc=%.1f%%",
            training_run_id,
            len(all_codes),
            accuracy * 100,
            training_metrics.hospital_weekly_accuracy * 100,
            training_metrics.reconciled_weekly_per_drug_mean_accuracy * 100,
        )
        return TrainStatusResponse(
            status="completed",
            training_run_id=training_run_id,
            drugs_trained=len(all_codes),
            models_trained=SHIELD_XR_MODELS,
            smape_summary={"shield_xr_ensemble": smape_from_wape(
                test.loc[mask_ok, "demand_raw"].values if mask_ok.any() else test["demand"].values,
                ensemble_pred[mask_ok] if mask_ok.any() else ensemble_pred,
            )},
            artifacts_saved=artifacts_saved,
            accuracy_summary={
                "daily_ensemble_accuracy_pct": round(accuracy * 100, 2),
                "hospital_weekly_accuracy_pct": round(
                    training_metrics.hospital_weekly_accuracy * 100,
                    2,
                ),
                "reconciled_weekly_per_drug_mean_accuracy_pct": round(
                    training_metrics.reconciled_weekly_per_drug_mean_accuracy * 100,
                    2,
                ),
                "reconciled_weekly_per_drug_median_accuracy_pct": round(
                    training_metrics.reconciled_weekly_per_drug_median_accuracy * 100,
                    2,
                ),
                "volume_weighted_weekly_accuracy_pct": round(
                    training_metrics.volume_weighted_weekly_accuracy * 100,
                    2,
                ),
                "non_lumpy_weekly_accuracy_pct": round(
                    training_metrics.non_lumpy_weekly_accuracy * 100,
                    2,
                ),
                "hybrid_abc_combined_accuracy_pct": round(
                    training_metrics.hybrid_abc_combined_accuracy * 100,
                    2,
                ),
                "per_sb_class_weekly_accuracy_pct": {
                    key: round(value * 100, 2)
                    for key, value in training_metrics.per_sb_class_weekly_accuracy.items()
                },
            },
        )


def resolve_training_drug_codes(db_session: Session, drug_codes: Optional[list[str]]) -> list[str]:
    if drug_codes:
        return drug_codes
    return get_distinct_drug_codes_from_enriched(db_session)
