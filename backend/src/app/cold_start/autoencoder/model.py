"""PyTorch autoencoder: 12-dim metadata → 32-dim drug fingerprint."""

from __future__ import annotations

import torch
import torch.nn as nn

from app.cold_start.constants import (
    AUTOENCODER_HIDDEN_DIM,
    EMBEDDING_DIM,
    METADATA_INPUT_DIM,
)


class DrugAutoencoder(nn.Module):
    """Symmetric encoder-decoder for normalized drug metadata vectors."""

    def __init__(
        self,
        input_dim: int = METADATA_INPUT_DIM,
        hidden_dim: int = AUTOENCODER_HIDDEN_DIM,
        embedding_dim: int = EMBEDDING_DIM,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return embedding of shape [..., embedding_dim]."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Return reconstruction of shape [..., input_dim]."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (reconstruction, embedding)."""
        embedding = self.encode(x)
        reconstruction = self.decode(embedding)
        return reconstruction, embedding
