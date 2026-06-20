"""Orchestrator — ties Stages A (embedding), B (KNN), and C (MAML) together."""

from __future__ import annotations

import hashlib
import logging
from datetime import date

from sqlalchemy.orm import Session

from app.cold_start.autoencoder.embedder import get_all_embeddings, get_embedding
from app.cold_start.constants import COLD_START_ONLY_BELOW, KNN_MAX_CANDIDATES
from app.cold_start.graduation import decide_stage, get_observation_count
from app.cold_start.knn_bootstrap.bootstrap import compute_baseline_forecast
from app.cold_start.knn_bootstrap.similarity import find_nearest_neighbours
from app.cold_start.maml.adapter import adapt_and_forecast
from app.cold_start.receipt_adapter import load_receipt_drug_metadata
from app.cold_start.schemas import ColdStartPredictRequest, ColdStartPredictResponse
from app.services.demand_aggregation import load_demand_series_from_receipts

logger = logging.getLogger(__name__)

_library_embedding_cache: dict[str, tuple[list[list[float]], list[str]]] = {}


class DrugReceiptsEmptyError(ValueError):
    """Raised when drug_receipts has no rows usable for KNN."""


DrugCatalogEmptyError = DrugReceiptsEmptyError


def invalidate_library_embedding_cache() -> None:
    """Clear cached receipt-library embeddings (call after receipt or embedder changes)."""
    _library_embedding_cache.clear()


def _library_cache_key(drug_codes: list[str]) -> str:
    joined = ",".join(sorted(drug_codes))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _load_library_embeddings(db: Session) -> tuple[list[list[float]], list[str]]:
    metadata_list = load_receipt_drug_metadata(db)
    drug_codes = [meta.drug_code for meta in metadata_list]
    if not drug_codes:
        raise DrugReceiptsEmptyError(
            "drug_receipts is empty. Load receipt history before predicting."
        )

    cache_key = _library_cache_key(drug_codes)
    cached = _library_embedding_cache.get(cache_key)
    if cached is not None:
        return cached

    embeddings = get_all_embeddings(metadata_list)
    result = (embeddings, drug_codes)
    _library_embedding_cache[cache_key] = result
    logger.debug(
        "Cached embeddings for %d receipt drugs (key=%s…)",
        len(drug_codes),
        cache_key[:12],
    )
    return result


def _fetch_real_observations(db: Session, drug_code: str) -> list[tuple[date, float]]:
    return load_demand_series_from_receipts(db, drug_code)


def _build_uncertainty_note(
    stage: str,
    observation_count: int,
    neighbour_count: int,
) -> str:
    if stage == "full_ensemble":
        return (
            "This drug has sufficient history. Route to full forecasting ensemble."
        )
    if stage == "blended":
        return (
            f"Forecast blends similarity-based baseline with a meta-learned model "
            f"using {observation_count} real observation(s). Confidence improves as "
            f"more data is collected (graduation at 12+ days)."
        )
    if neighbour_count < 5:
        return (
            f"Cold-start forecast from {neighbour_count} similar receipt drug(s); "
            "limited neighbour coverage may widen uncertainty."
        )
    return (
        "Cold-start forecast from the five most similar drugs in receipt history. "
        f"No real demand history yet for this drug ({observation_count} observation(s))."
    )


class ColdStartService:
    def __init__(self, db_session: Session) -> None:
        self.db = db_session

    def predict(self, request: ColdStartPredictRequest) -> ColdStartPredictResponse:
        """
        Full pipeline: embed → KNN → baseline → graduation → optional MAML → response.
        """
        drug = request.drug_metadata
        horizon = request.forecast_horizon_days

        embedding = get_embedding(drug)

        library_embeddings, library_codes = _load_library_embeddings(self.db)

        filtered_embeddings: list[list[float]] = []
        filtered_codes: list[str] = []
        for emb, code in zip(library_embeddings, library_codes):
            if code != drug.drug_code:
                filtered_embeddings.append(emb)
                filtered_codes.append(code)

        if not filtered_codes:
            raise ValueError(
                "No other drugs in drug_receipts to compare against. "
                "Load additional receipt history before predicting."
            )

        candidates = find_nearest_neighbours(
            embedding,
            filtered_embeddings,
            filtered_codes,
            k=min(KNN_MAX_CANDIDATES, len(filtered_codes)),
        )

        baseline, neighbours = compute_baseline_forecast(
            candidates,
            self.db,
            horizon,
            request.pharmacist_estimate,
        )

        observation_count = get_observation_count(drug.drug_code, self.db)
        stage = decide_stage(observation_count)

        if observation_count >= COLD_START_ONLY_BELOW:
            observations = _fetch_real_observations(self.db, drug.drug_code)
            forecast = adapt_and_forecast(observations, horizon, baseline)
        else:
            forecast = baseline

        return ColdStartPredictResponse(
            drug_code=drug.drug_code,
            stage_used=stage,
            observation_count=observation_count,
            embedding=embedding,
            nearest_neighbours=[code for code, _ in neighbours],
            similarity_scores=[score for _, score in neighbours],
            forecast=forecast,
            uncertainty_note=_build_uncertainty_note(
                stage,
                observation_count,
                len(neighbours),
            ),
        )
