"""Persisted CAMEO training artifacts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from app.cold_start.cameo.features import CameoFeatureEncoder
from app.cold_start.cameo.metric_net import MetricNet
from app.cold_start.constants import ARTIFACTS_DIR


@dataclass
class CameoArtifacts:
    metric_net: MetricNet
    feature_encoder: CameoFeatureEncoder
    attr_mean: np.ndarray
    attr_scale: np.ndarray
    shape_mean: np.ndarray
    shape_scale: np.ndarray
    hist_drug_codes: list[str]
    conformal_half_width: float = 0.0
    topk: int = 5
    min_weeks: int = 30
    validation_summary: dict | None = None
    trained_at: str | None = None

    def scale_attrs(self, attrs_raw: np.ndarray) -> np.ndarray:
        return (attrs_raw - self.attr_mean) / self.attr_scale

    def scale_shape(self, shape_raw: np.ndarray) -> np.ndarray:
        return (shape_raw - self.shape_mean) / self.shape_scale

    def to_payload(self) -> dict:
        return {
            "metric_net": self.metric_net.to_dict(),
            "feature_encoder": {
                "column_names": self.feature_encoder.column_names,
                "strength_median": self.feature_encoder.strength_median,
            },
            "attr_mean": self.attr_mean.tolist(),
            "attr_scale": self.attr_scale.tolist(),
            "shape_mean": self.shape_mean.tolist(),
            "shape_scale": self.shape_scale.tolist(),
            "hist_drug_codes": self.hist_drug_codes,
            "conformal_half_width": self.conformal_half_width,
            "topk": self.topk,
            "min_weeks": self.min_weeks,
            "validation_summary": self.validation_summary,
            "trained_at": self.trained_at,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "CameoArtifacts":
        encoder = CameoFeatureEncoder(
            column_names=list(payload["feature_encoder"]["column_names"]),
            strength_median=float(payload["feature_encoder"]["strength_median"]),
        )
        return cls(
            metric_net=MetricNet.from_dict(payload["metric_net"]),
            feature_encoder=encoder,
            attr_mean=np.asarray(payload["attr_mean"], dtype=float),
            attr_scale=np.asarray(payload["attr_scale"], dtype=float),
            shape_mean=np.asarray(payload["shape_mean"], dtype=float),
            shape_scale=np.asarray(payload["shape_scale"], dtype=float),
            hist_drug_codes=list(payload["hist_drug_codes"]),
            conformal_half_width=float(payload.get("conformal_half_width", 0.0)),
            topk=int(payload.get("topk", 5)),
            min_weeks=int(payload.get("min_weeks", 30)),
            validation_summary=payload.get("validation_summary"),
            trained_at=payload.get("trained_at"),
        )


def artifact_path() -> str:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    return os.path.join(ARTIFACTS_DIR, "cameo_artifacts.json")


def save_artifacts(artifacts: CameoArtifacts) -> str:
    path = artifact_path()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(artifacts.to_payload(), handle, indent=2)
    return path


_artifact_cache: CameoArtifacts | None = None


def load_artifacts() -> CameoArtifacts:
    global _artifact_cache
    if _artifact_cache is not None:
        return _artifact_cache

    path = artifact_path()
    if not os.path.exists(path):
        raise RuntimeError(
            "CAMEO model not trained yet. POST /cold-start/train-cameo first."
        )

    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    _artifact_cache = CameoArtifacts.from_payload(payload)
    return _artifact_cache


def clear_artifact_cache() -> None:
    global _artifact_cache
    _artifact_cache = None


def is_trained() -> bool:
    return os.path.exists(artifact_path())


def load_validation_summary() -> dict | None:
    if not is_trained():
        return None
    try:
        with open(artifact_path(), encoding="utf-8") as handle:
            payload = json.load(handle)
        summary = payload.get("validation_summary")
        return summary if isinstance(summary, dict) else None
    except (OSError, json.JSONDecodeError):
        return None
