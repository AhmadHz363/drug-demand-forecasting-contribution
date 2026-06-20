"""Inference: drug metadata → 32-dim embedding via trained autoencoder."""

from __future__ import annotations

import os

import torch

from app.cold_start.autoencoder.model import DrugAutoencoder
from app.cold_start.constants import ARTIFACTS_DIR, EMBEDDING_DIM
from app.cold_start.drug_metadata_encoder import encode_drug_metadata
from app.cold_start.schemas import DrugMetadataInput

AUTOENCODER_ARTIFACT = "autoencoder.pt"

_model_cache: DrugAutoencoder | None = None


def _artifact_path() -> str:
    return os.path.join(ARTIFACTS_DIR, AUTOENCODER_ARTIFACT)


def clear_model_cache() -> None:
    """Reset in-memory model cache (e.g. after retraining)."""
    global _model_cache
    _model_cache = None


def _load_model() -> DrugAutoencoder:
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    path = _artifact_path()
    if not os.path.isfile(path):
        raise RuntimeError("Autoencoder not trained yet. Call train_autoencoder() first.")

    model = DrugAutoencoder()
    state_dict = torch.load(path, map_location=torch.device("cpu"), weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    _model_cache = model
    return model


def get_embedding(drug: DrugMetadataInput) -> list[float]:
    """
    Encode a drug's metadata into a 32-dim embedding.
    Loads model from disk if not already cached in memory.
    Returns list of 32 floats.
    """
    model = _load_model()
    features = encode_drug_metadata(drug)
    with torch.no_grad():
        x = torch.tensor([features], dtype=torch.float32)
        embedding = model.encode(x).squeeze(0).tolist()

    if len(embedding) != EMBEDDING_DIM:
        raise RuntimeError(
            f"Expected embedding dim {EMBEDDING_DIM}, got {len(embedding)}"
        )
    return embedding


def get_all_embeddings(drugs: list[DrugMetadataInput]) -> list[list[float]]:
    """Batch version. Returns list of N embeddings."""
    if not drugs:
        return []

    model = _load_model()
    features = [encode_drug_metadata(d) for d in drugs]
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32)
        embeddings = model.encode(x).tolist()

    return embeddings
