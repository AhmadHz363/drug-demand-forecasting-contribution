"""Cold Start module — drug embedding, KNN bootstrap, and MAML adaptation."""

from app.cold_start.constants import (
    ARTIFACTS_DIR,
    EMBEDDING_DIM,
    METADATA_INPUT_DIM,
)
from app.cold_start.autoencoder import get_embedding, train_autoencoder
from app.cold_start.drug_metadata_encoder import encode_drug_metadata
from app.cold_start.cold_start_service import ColdStartService, invalidate_library_embedding_cache
from app.cold_start.graduation import decide_stage, get_observation_count
from app.cold_start.knn_bootstrap import compute_baseline_forecast, find_nearest_neighbours
from app.cold_start.maml import adapt_and_forecast, train_maml
from app.cold_start.schemas import (
    ColdStartPredictRequest,
    ColdStartPredictResponse,
    DailyForecast,
    DrugMetadataInput,
    PharmacistEstimate,
)

__all__ = [
    "ARTIFACTS_DIR",
    "EMBEDDING_DIM",
    "METADATA_INPUT_DIM",
    "ColdStartPredictRequest",
    "ColdStartPredictResponse",
    "DailyForecast",
    "DrugMetadataInput",
    "PharmacistEstimate",
    "ColdStartService",
    "adapt_and_forecast",
    "compute_baseline_forecast",
    "invalidate_library_embedding_cache",
    "decide_stage",
    "encode_drug_metadata",
    "find_nearest_neighbours",
    "get_embedding",
    "get_observation_count",
    "train_autoencoder",
    "train_maml",
]
