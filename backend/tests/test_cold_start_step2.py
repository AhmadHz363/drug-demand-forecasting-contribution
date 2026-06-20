"""Step 2 Cold Start — autoencoder unit tests."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
import torch

from app.cold_start.autoencoder.embedder import (
    AUTOENCODER_ARTIFACT,
    clear_model_cache,
    get_all_embeddings,
    get_embedding,
)
from app.cold_start.autoencoder.model import DrugAutoencoder
from app.cold_start.autoencoder.trainer import train_autoencoder
from app.cold_start.constants import (
    ARTIFACTS_DIR,
    AUTOENCODER_HIDDEN_DIM,
    EMBEDDING_DIM,
    METADATA_INPUT_DIM,
)
from app.cold_start.drug_metadata_encoder import encode_drug_metadata
from app.cold_start.schemas import DrugMetadataInput


def _sample_drug(**overrides) -> DrugMetadataInput:
    base = dict(
        drug_code="NEW-001",
        drug_name="Ceftriaxone 1g",
        therapeutic_class="antibiotic",
        atc_category="J01",
        pharmaceutical_form="injection",
        ven_class="V",
        abc_class="A",
        unit_price_tier=4,
        requires_refrigeration=False,
        is_controlled_substance=False,
        average_shelf_life_days=730,
        route_of_administration="iv",
    )
    base.update(overrides)
    return DrugMetadataInput(**base)


def _synthetic_catalog(n: int) -> list[list[float]]:
    drugs = [
        _sample_drug(
            drug_code=f"DRUG-{i}",
            therapeutic_class="antibiotic" if i % 2 == 0 else "analgesic",
            pharmaceutical_form="tablet" if i % 3 == 0 else "injection",
            unit_price_tier=(i % 5) + 1,
        )
        for i in range(n)
    ]
    return [encode_drug_metadata(d) for d in drugs]


@pytest.fixture(autouse=True)
def fast_training(monkeypatch):
    """Keep unit tests fast while production uses AUTOENCODER_EPOCHS from constants."""
    monkeypatch.setattr(
        "app.cold_start.autoencoder.trainer.AUTOENCODER_EPOCHS",
        20,
    )


@pytest.fixture
def artifact_backup(tmp_path: Path):
    """Isolate artifact writes; restore prior state after test."""
    artifact = Path(ARTIFACTS_DIR) / AUTOENCODER_ARTIFACT
    backup = tmp_path / "backup.pt"
    had_artifact = artifact.is_file()
    if had_artifact:
        shutil.copy2(artifact, backup)

    clear_model_cache()
    yield artifact

    clear_model_cache()
    if had_artifact:
        shutil.copy2(backup, artifact)
    elif artifact.is_file():
        artifact.unlink()


class TestDrugAutoencoderModel:
    def test_forward_shapes(self):
        model = DrugAutoencoder()
        x = torch.rand(4, METADATA_INPUT_DIM)
        recon, emb = model(x)
        assert recon.shape == (4, METADATA_INPUT_DIM)
        assert emb.shape == (4, EMBEDDING_DIM)

    def test_encode_decode(self):
        model = DrugAutoencoder()
        x = torch.rand(2, METADATA_INPUT_DIM)
        z = model.encode(x)
        out = model.decode(z)
        assert z.shape == (2, EMBEDDING_DIM)
        assert out.shape == (2, METADATA_INPUT_DIM)
        assert (out >= 0).all() and (out <= 1).all()

    def test_uses_constants(self):
        model = DrugAutoencoder()
        assert model.encoder[0].in_features == METADATA_INPUT_DIM
        assert model.encoder[0].out_features == AUTOENCODER_HIDDEN_DIM
        assert model.encoder[2].out_features == EMBEDDING_DIM


class TestAutoencoderTraining:
    def test_raises_when_fewer_than_five_drugs(self):
        with pytest.raises(ValueError, match="at least 5"):
            train_autoencoder(_synthetic_catalog(4))

    def test_saves_artifact_and_returns_model(self, artifact_backup: Path):
        encoded = _synthetic_catalog(8)
        model = train_autoencoder(encoded)
        assert isinstance(model, DrugAutoencoder)
        assert artifact_backup.is_file()

    def test_second_training_overwrites_artifact(self, artifact_backup: Path):
        train_autoencoder(_synthetic_catalog(6))
        mtime_first = artifact_backup.stat().st_mtime
        train_autoencoder(_synthetic_catalog(7))
        mtime_second = artifact_backup.stat().st_mtime
        assert mtime_second >= mtime_first
        assert artifact_backup.is_file()


class TestAutoencoderEmbedder:
    def test_runtime_error_before_training(self, artifact_backup: Path):
        if artifact_backup.is_file():
            artifact_backup.unlink()
        clear_model_cache()
        with pytest.raises(RuntimeError, match="Autoencoder not trained yet"):
            get_embedding(_sample_drug())

    def test_get_embedding_returns_32_floats(self, artifact_backup: Path):
        train_autoencoder(_synthetic_catalog(6))
        clear_model_cache()
        emb = get_embedding(_sample_drug())
        assert len(emb) == EMBEDDING_DIM
        assert all(isinstance(v, float) for v in emb)

    def test_get_all_embeddings_batch(self, artifact_backup: Path):
        train_autoencoder(_synthetic_catalog(6))
        clear_model_cache()
        drugs = [_sample_drug(drug_code=f"B-{i}") for i in range(3)]
        embs = get_all_embeddings(drugs)
        assert len(embs) == 3
        assert all(len(e) == EMBEDDING_DIM for e in embs)

    def test_model_cache_avoids_reload(self, artifact_backup: Path, monkeypatch):
        train_autoencoder(_synthetic_catalog(6))
        clear_model_cache()

        load_count = 0
        original_load = torch.load

        def counting_load(*args, **kwargs):
            nonlocal load_count
            load_count += 1
            return original_load(*args, **kwargs)

        monkeypatch.setattr(torch, "load", counting_load)
        get_embedding(_sample_drug())
        get_embedding(_sample_drug(drug_code="OTHER"))
        assert load_count == 1
