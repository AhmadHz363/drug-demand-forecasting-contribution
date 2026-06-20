"""Drug metadata autoencoder — Stage A of Cold Start."""

from app.cold_start.autoencoder.embedder import get_all_embeddings, get_embedding
from app.cold_start.autoencoder.model import DrugAutoencoder
from app.cold_start.autoencoder.trainer import train_autoencoder

__all__ = [
    "DrugAutoencoder",
    "get_all_embeddings",
    "get_embedding",
    "train_autoencoder",
]
