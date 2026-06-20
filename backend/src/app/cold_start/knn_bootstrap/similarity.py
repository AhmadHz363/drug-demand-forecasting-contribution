"""Cosine similarity search over drug embedding library."""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.cold_start.constants import KNN_K, SIMILARITY_MIN_THRESHOLD


def find_nearest_neighbours(
    query_embedding: list[float],
    library_embeddings: list[list[float]],
    library_drug_codes: list[str],
    k: int = KNN_K,
) -> list[tuple[str, float]]:
    """
    Return (drug_code, cosine_similarity_score) tuples sorted by descending
    similarity, length = min(k, len(library)).
    """
    if len(library_embeddings) != len(library_drug_codes):
        raise ValueError(
            "library_embeddings and library_drug_codes must have the same length"
        )

    if not library_embeddings:
        return []

    query = np.array(query_embedding, dtype=np.float64).reshape(1, -1)
    library = np.array(library_embeddings, dtype=np.float64)
    scores = cosine_similarity(query, library)[0]

    pairs = [
        (code, float(score))
        for code, score in zip(library_drug_codes, scores)
        if score >= SIMILARITY_MIN_THRESHOLD
    ]
    pairs.sort(key=lambda item: item[1], reverse=True)

    limit = min(k, len(library_embeddings))
    return pairs[:limit]
