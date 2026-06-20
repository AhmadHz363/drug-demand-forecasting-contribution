"""MAML meta-training loop for the BaseForecaster."""

from __future__ import annotations

import logging
import os
import random
import statistics

import learn2learn as l2l
import torch
import torch.nn as nn
from sqlalchemy.orm import Session

from app.cold_start.constants import (
    ARTIFACTS_DIR,
    MAML_EPOCHS,
    MAML_FORECAST_HORIZON,
    MAML_INNER_LR,
    MAML_INNER_STEPS,
    MAML_META_BATCH_SIZE,
    MAML_OUTER_LR,
    MAML_QUERY_SIZE,
    MAML_SEQUENCE_LEN,
    MAML_SUPPORT_SIZE,
)
from app.cold_start.maml.model import BaseForecaster, day_features
from app.services.demand_aggregation import (
    get_distinct_drug_codes_from_receipts,
    load_demand_series_from_receipts,
)

logger = logging.getLogger(__name__)

MAML_ARTIFACT = "maml_base.pt"
MIN_DAYS_PER_DRUG = MAML_SEQUENCE_LEN + MAML_SUPPORT_SIZE + MAML_QUERY_SIZE


def _artifact_path() -> str:
    return os.path.join(ARTIFACTS_DIR, MAML_ARTIFACT)


def _load_drug_series(db_session: Session, drug_code: str) -> list[tuple]:
    return load_demand_series_from_receipts(db_session, drug_code)


def _load_eligible_drugs(db_session: Session) -> dict[str, list[tuple]]:
    codes = get_distinct_drug_codes_from_receipts(db_session)
    eligible: dict[str, list[tuple]] = {}
    for code in codes:
        series = _load_drug_series(db_session, code)
        if len(series) >= MIN_DAYS_PER_DRUG:
            eligible[code] = series
    return eligible


def _normalize_stats(series: list[tuple]) -> tuple[float, float]:
    quantities = [quantity for _, quantity in series]
    mean = statistics.mean(quantities)
    std = statistics.stdev(quantities) if len(quantities) > 1 else 1.0
    if std <= 0:
        std = 1.0
    return mean, std


def _build_examples(
    series: list[tuple],
    offsets: list[int],
    mean: float,
    std: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    inputs: list[list[list[float]]] = []
    targets: list[list[float]] = []

    for offset in offsets:
        end_input = offset + MAML_SEQUENCE_LEN
        end_target = end_input + MAML_FORECAST_HORIZON
        if end_target > len(series):
            continue

        window = series[offset:end_input]
        target_window = series[end_input:end_target]
        inputs.append([day_features(d, q, mean, std) for d, q in window])
        targets.append([((q - mean) / std) for _, q in target_window])

    if not inputs:
        return None

    return (
        torch.tensor(inputs, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
    )


def _sample_task_batch(
    series: list[tuple],
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    mean, std = _normalize_stats(series)
    max_start = len(series) - MIN_DAYS_PER_DRUG
    start = rng.randint(0, max_start) if max_start > 0 else 0

    support_offsets = list(range(start, start + MAML_SUPPORT_SIZE))
    query_offsets = list(
        range(start + MAML_SUPPORT_SIZE, start + MAML_SUPPORT_SIZE + MAML_QUERY_SIZE)
    )

    support = _build_examples(series, support_offsets, mean, std)
    query = _build_examples(series, query_offsets, mean, std)
    if support is None or query is None:
        return None

    return support[0], support[1], query[0], query[1]


def _task_loss(
    learner: l2l.algorithms.MAML,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    predictions = learner(inputs)
    return criterion(predictions, targets)


def train_maml(db_session: Session) -> tuple[l2l.algorithms.MAML, int]:
    """
    Meta-train BaseForecaster across drugs in drug_receipts.
    Saves weights to ARTIFACTS_DIR/maml_base.pt and returns the MAML wrapper
    plus the number of eligible drugs used for meta-training.
    """
    eligible = _load_eligible_drugs(db_session)
    tasks_trained_on = len(eligible)
    if not eligible:
        raise ValueError(
            f"No drugs with at least {MIN_DAYS_PER_DRUG} days of demand history. "
            "Load drug_receipts history before training MAML."
        )

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    device = torch.device("cpu")
    base_model = BaseForecaster().to(device)
    maml = l2l.algorithms.MAML(base_model, lr=MAML_INNER_LR, first_order=False)
    outer_optimizer = torch.optim.Adam(maml.parameters(), lr=MAML_OUTER_LR)
    criterion = nn.MSELoss()
    rng = random.Random(42)

    drug_codes = list(eligible.keys())

    for epoch in range(1, MAML_EPOCHS + 1):
        outer_optimizer.zero_grad()
        meta_loss = torch.tensor(0.0, device=device)
        tasks_used = 0

        batch_size = min(MAML_META_BATCH_SIZE, len(drug_codes))
        sampled_codes = rng.choices(drug_codes, k=batch_size)

        for drug_code in sampled_codes:
            task = _sample_task_batch(eligible[drug_code], rng)
            if task is None:
                continue

            support_x, support_y, query_x, query_y = task
            learner = maml.clone()

            for _ in range(MAML_INNER_STEPS):
                support_loss = _task_loss(learner, support_x, support_y, criterion)
                learner.adapt(support_loss)

            query_loss = _task_loss(learner, query_x, query_y, criterion)
            meta_loss = meta_loss + query_loss
            tasks_used += 1

        if tasks_used == 0:
            raise ValueError(
                "Could not sample any valid MAML tasks from eligible drugs."
            )

        meta_loss = meta_loss / tasks_used
        meta_loss.backward()
        outer_optimizer.step()

        if epoch % 20 == 0 or epoch == 1:
            logger.info(
                "MAML epoch %d/%d — outer_loss=%.6f tasks=%d",
                epoch,
                MAML_EPOCHS,
                meta_loss.item(),
                tasks_used,
            )

    artifact_path = _artifact_path()
    torch.save(base_model.state_dict(), artifact_path)
    logger.info(
        "Saved MAML base learner to %s (trained on %d drugs)",
        artifact_path,
        len(eligible),
    )

    from app.cold_start.maml.adapter import clear_maml_cache

    clear_maml_cache()

    return maml, tasks_trained_on


def artifact_path() -> str:
    """Public accessor for the saved MAML weights path."""
    return _artifact_path()
