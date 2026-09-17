"""Persist and load SHIELD-XR model artifacts."""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import asdict, is_dataclass
from typing import Any

from app.forecasting.constants import ARTIFACTS_DIR

SHIELD_XR_DIR = os.path.join(ARTIFACTS_DIR, "shield_xr")
DAILY_ARTIFACT = "daily_ensemble.pkl"
WEEKLY_ARTIFACT = "weekly_breakdown.pkl"
META_ARTIFACT = "training_meta.json"


def _ensure_dir() -> str:
    os.makedirs(SHIELD_XR_DIR, exist_ok=True)
    return SHIELD_XR_DIR


def is_trained() -> bool:
    return os.path.isfile(os.path.join(SHIELD_XR_DIR, DAILY_ARTIFACT))


def save_daily_artifacts(artifacts: Any) -> str:
    path = os.path.join(_ensure_dir(), DAILY_ARTIFACT)
    with open(path, "wb") as handle:
        pickle.dump(artifacts, handle)
    return path


def load_daily_artifacts() -> Any:
    path = os.path.join(SHIELD_XR_DIR, DAILY_ARTIFACT)
    with open(path, "rb") as handle:
        return pickle.load(handle)


def save_weekly_artifacts(artifacts: Any) -> str:
    path = os.path.join(_ensure_dir(), WEEKLY_ARTIFACT)
    with open(path, "wb") as handle:
        pickle.dump(artifacts, handle)
    return path


def load_weekly_artifacts() -> Any | None:
    path = os.path.join(SHIELD_XR_DIR, WEEKLY_ARTIFACT)
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as handle:
        return pickle.load(handle)


def save_training_meta(meta: dict[str, Any]) -> str:
    path = os.path.join(_ensure_dir(), META_ARTIFACT)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, default=str)
    return path


def load_training_meta() -> dict[str, Any]:
    path = os.path.join(SHIELD_XR_DIR, META_ARTIFACT)
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def dataclass_to_meta(obj: Any) -> dict[str, Any]:
    if is_dataclass(obj):
        return asdict(obj)
    return dict(obj)
