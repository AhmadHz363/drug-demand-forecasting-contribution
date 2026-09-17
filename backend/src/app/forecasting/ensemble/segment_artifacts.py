"""Segment-scoped ensemble artifact path resolution."""

from __future__ import annotations

import os
from typing import Optional

from app.forecasting.constants import ARTIFACTS_DIR, GLOBAL_DEMAND_SEGMENT


def stacking_artifact_path(segment: str = GLOBAL_DEMAND_SEGMENT) -> str:
    return os.path.join(ARTIFACTS_DIR, "stacking", segment, "stacking_meta.pkl")


def conformal_artifact_path(segment: str = GLOBAL_DEMAND_SEGMENT) -> str:
    return os.path.join(ARTIFACTS_DIR, "conformal", segment, "mapie_wrapper.pkl")


def legacy_stacking_path() -> str:
    return os.path.join(ARTIFACTS_DIR, "stacking", "stacking_meta.pkl")


def legacy_conformal_path() -> str:
    return os.path.join(ARTIFACTS_DIR, "conformal", "mapie_wrapper.pkl")


def resolve_stacking_path(segment: str) -> Optional[str]:
    candidates = (
        stacking_artifact_path(segment),
        stacking_artifact_path(GLOBAL_DEMAND_SEGMENT),
        legacy_stacking_path(),
    )
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def resolve_conformal_path(segment: str) -> Optional[str]:
    candidates = (
        conformal_artifact_path(segment),
        conformal_artifact_path(GLOBAL_DEMAND_SEGMENT),
        legacy_conformal_path(),
    )
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None
