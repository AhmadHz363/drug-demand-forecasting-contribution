"""Cold Start module — CAMEO metric-learning cold-start forecasting."""

from app.cold_start.cameo import clear_artifact_cache, load_artifacts, train_cameo
from app.cold_start.cold_start_service import ColdStartService, invalidate_library_embedding_cache
from app.cold_start.constants import (
    BLEND_UNTIL,
    CAMEO_EPOCHS,
    CAMEO_TOPK,
    COLD_START_ONLY_BELOW,
)
from app.cold_start.graduation import decide_stage, get_observation_count
from app.cold_start.schemas import (
    ColdStartPredictRequest,
    ColdStartPredictResponse,
    DailyForecast,
    DrugMetadataInput,
    PharmacistEstimate,
)

__all__ = [
    "BLEND_UNTIL",
    "CAMEO_EPOCHS",
    "CAMEO_TOPK",
    "COLD_START_ONLY_BELOW",
    "ColdStartPredictRequest",
    "ColdStartPredictResponse",
    "ColdStartService",
    "DailyForecast",
    "DrugMetadataInput",
    "PharmacistEstimate",
    "clear_artifact_cache",
    "decide_stage",
    "get_observation_count",
    "invalidate_library_embedding_cache",
    "load_artifacts",
    "train_cameo",
]
