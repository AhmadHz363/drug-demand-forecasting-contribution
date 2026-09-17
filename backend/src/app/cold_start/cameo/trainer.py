"""Train CAMEO metric-learning model on matched-source drugs."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import Session

from app.cold_start.cameo.artifacts import CameoArtifacts, save_artifacts
from app.cold_start.cameo.features import CameoFeatureEncoder
from app.cold_start.cameo.metric_net import (
    demand_shape_features,
    train_metric_net,
    cameo_init_forecast,
)
from app.cold_start.cameo.panel import MIN_WEEKS, load_weekly_series
from app.cold_start.cameo.pipeline import conformal_intervals
from app.cold_start.cameo.validation import evaluate_cameo_holdout
from app.cold_start.constants import CAMEO_EPOCHS, CAMEO_SEED, CAMEO_TOPK
from app.cold_start.drugs_adapter import load_library_drug_metadata


class CameoTrainingError(ValueError):
    """Raised when the historical library is too small to train."""


def train_cameo(db: Session) -> tuple[CameoArtifacts, int]:
    metadata_list = load_library_drug_metadata(db)
    if len(metadata_list) < 5:
        raise CameoTrainingError(
            "Need at least 5 matched_source drugs in the drugs table before training."
        )

    drug_codes = [meta.drug_code for meta in metadata_list]
    weekly_by_code = load_weekly_series(db, drug_codes)

    hist_codes: list[str] = []
    hist_series: list[np.ndarray] = []
    hist_metadata = []
    for meta in metadata_list:
        series = weekly_by_code.get(meta.drug_code)
        if series is None or len(series) < MIN_WEEKS:
            continue
        hist_codes.append(meta.drug_code)
        hist_series.append(series)
        hist_metadata.append(meta)

    if len(hist_codes) < 5:
        raise CameoTrainingError(
            f"Need at least 5 drugs with >={MIN_WEEKS} weekly demand history; "
            f"found {len(hist_codes)}."
        )

    validation_summary = evaluate_cameo_holdout(metadata_list, weekly_by_code)

    encoder = CameoFeatureEncoder().fit(hist_metadata)
    attrs_raw = encoder.transform(hist_metadata)
    shape_targets = np.array([demand_shape_features(series) for series in hist_series])

    attr_scaler = StandardScaler().fit(attrs_raw)
    shape_scaler = StandardScaler().fit(shape_targets)
    attrs_scaled = attr_scaler.transform(attrs_raw)
    shape_scaled = shape_scaler.transform(shape_targets)

    net = train_metric_net(
        attrs_scaled,
        shape_scaled,
        epochs=CAMEO_EPOCHS,
        seed=CAMEO_SEED,
    )
    hist_embeds = net.embed(attrs_scaled)

    calib_residuals: list[float] = []
    rng = np.random.default_rng(CAMEO_SEED + 7)
    sample_size = min(25, len(hist_codes))
    sample_idx = rng.choice(len(hist_codes), size=sample_size, replace=False)
    calib_idx = sample_idx[: len(sample_idx) // 2]

    for i in calib_idx:
        embeds_without = np.delete(hist_embeds, i, axis=0)
        series_without = [series for j, series in enumerate(hist_series) if j != i]
        init = cameo_init_forecast(
            attrs_scaled[i],
            embeds_without,
            series_without,
            net,
            topk=CAMEO_TOPK,
        )
        calib_residuals.append(float(hist_series[i][:8].mean() - init.init_level))

    conformal_q = conformal_intervals(np.asarray(calib_residuals), alpha=0.1)

    artifacts = CameoArtifacts(
        metric_net=net,
        feature_encoder=encoder,
        attr_mean=attr_scaler.mean_,
        attr_scale=attr_scaler.scale_,
        shape_mean=shape_scaler.mean_,
        shape_scale=shape_scaler.scale_,
        hist_drug_codes=hist_codes,
        conformal_half_width=conformal_q,
        topk=CAMEO_TOPK,
        min_weeks=MIN_WEEKS,
        validation_summary=validation_summary,
        trained_at=datetime.now(tz=timezone.utc).isoformat(),
    )
    save_artifacts(artifacts)
    return artifacts, len(hist_codes)
