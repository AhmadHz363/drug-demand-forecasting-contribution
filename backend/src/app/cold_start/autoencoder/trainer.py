"""Training loop for the drug metadata autoencoder."""

from __future__ import annotations

import logging
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from app.cold_start.autoencoder.model import DrugAutoencoder
from app.cold_start.constants import (
    ARTIFACTS_DIR,
    AUTOENCODER_BATCH_SIZE,
    AUTOENCODER_EPOCHS,
    AUTOENCODER_LR,
    METADATA_INPUT_DIM,
)

logger = logging.getLogger(__name__)

AUTOENCODER_ARTIFACT = "autoencoder.pt"
MIN_TRAINING_DRUGS = 5


def _artifact_path() -> str:
    return os.path.join(ARTIFACTS_DIR, AUTOENCODER_ARTIFACT)


def train_autoencoder(encoded_drugs: list[list[float]]) -> DrugAutoencoder:
    """Train autoencoder on all known drugs. Returns trained model."""
    if len(encoded_drugs) < MIN_TRAINING_DRUGS:
        raise ValueError(
            f"Need at least {MIN_TRAINING_DRUGS} drugs to train the autoencoder; "
            f"got {len(encoded_drugs)}"
        )

    for i, vec in enumerate(encoded_drugs):
        if len(vec) != METADATA_INPUT_DIM:
            raise ValueError(
                f"Drug index {i}: expected {METADATA_INPUT_DIM} features, got {len(vec)}"
            )

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    device = torch.device("cpu")
    data = torch.tensor(encoded_drugs, dtype=torch.float32, device=device)
    dataset = TensorDataset(data)
    loader = DataLoader(
        dataset,
        batch_size=min(AUTOENCODER_BATCH_SIZE, len(encoded_drugs)),
        shuffle=True,
    )

    model = DrugAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=AUTOENCODER_LR)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(1, AUTOENCODER_EPOCHS + 1):
        epoch_loss = 0.0
        batch_count = 0
        for (batch,) in loader:
            optimizer.zero_grad()
            reconstruction, _ = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1

        if epoch % 50 == 0 or epoch == 1:
            avg_loss = epoch_loss / max(batch_count, 1)
            logger.info(
                "Autoencoder epoch %d/%d — loss=%.6f",
                epoch,
                AUTOENCODER_EPOCHS,
                avg_loss,
            )

    artifact_path = _artifact_path()
    torch.save(model.state_dict(), artifact_path)
    logger.info("Saved autoencoder weights to %s", artifact_path)

    from app.cold_start.autoencoder.embedder import clear_model_cache

    clear_model_cache()

    return model
