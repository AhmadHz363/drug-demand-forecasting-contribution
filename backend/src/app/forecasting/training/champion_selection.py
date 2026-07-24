"""Champion selection among base models, ensemble, and naive baselines."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from app.forecasting.constants import (
    CHAMPION_AGREEMENT_MASE_TOLERANCE,
    CHAMPION_AGREEMENT_REL_TOL,
    ARTIFACTS_DIR,
    GLOBAL_DEMAND_SEGMENT,
    SEASONAL_NAIVE_PERIOD,
)
from app.forecasting.evaluation_metrics import mase, mean_pinball_loss, smape

logger = logging.getLogger(__name__)

CHAMPION_ENSEMBLE_IMPROVEMENT = 0.02  # stack must beat best base by ≥2% MASE
MIN_DRUG_OOF_POINTS = 10


@dataclass
class ChampionDecision:
    drug_code: str
    demand_segment: str
    champion: str
    reason: str
    mase_by_candidate: dict[str, float | None]
    beat_naive: bool


def _finite_mase(actuals: np.ndarray, preds: np.ndarray) -> float:
    actual = np.asarray(actuals, dtype=float)
    pred = np.asarray(preds, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(pred)
    if mask.sum() == 0:
        return float("inf")
    return float(mase(actual[mask], pred[mask]))


def seasonal_naive_oof(
    actuals: np.ndarray,
    *,
    period: int = SEASONAL_NAIVE_PERIOD,
) -> np.ndarray:
    """One-step seasonal-naive predictions aligned to ``actuals`` (NaN where undefined)."""
    values = np.asarray(actuals, dtype=float).reshape(-1)
    out = np.full_like(values, np.nan, dtype=float)
    if values.size <= period:
        return out
    out[period:] = values[:-period]
    return out


def recent_level_naive_oof(actuals: np.ndarray, *, window: int = 28) -> np.ndarray:
    values = np.asarray(actuals, dtype=float).reshape(-1)
    out = np.full_like(values, np.nan, dtype=float)
    for idx in range(len(values)):
        start = max(0, idx - window)
        hist = values[start:idx]
        hist = hist[np.isfinite(hist)]
        if hist.size:
            out[idx] = float(np.mean(hist))
    return out


def intermittent_baseline_oof(actuals: np.ndarray) -> np.ndarray:
    """Croston-like constant rate from in-sample nonzeros (causal expanding)."""
    values = np.asarray(actuals, dtype=float).reshape(-1)
    out = np.full_like(values, np.nan, dtype=float)
    nonzero_sum = 0.0
    nonzero_count = 0
    intervals: list[float] = []
    since = 0
    for idx, qty in enumerate(values):
        since += 1
        if np.isfinite(qty) and qty > 0:
            nonzero_sum += float(qty)
            nonzero_count += 1
            intervals.append(float(since))
            since = 0
        if nonzero_count > 0 and intervals:
            level = nonzero_sum / nonzero_count
            interval = float(np.mean(intervals))
            out[idx] = level / max(interval, 1.0)
        else:
            out[idx] = 0.0
    return out


def segment_naive_oof(actuals: np.ndarray, demand_segment: str) -> tuple[str, np.ndarray]:
    if demand_segment in {"intermittent", "lumpy"}:
        return "intermittent_baseline", intermittent_baseline_oof(actuals)
    if demand_segment == "erratic":
        return "recent_level_naive", recent_level_naive_oof(actuals)
    return "seasonal_naive", seasonal_naive_oof(actuals)


def zero_day_mae(actuals: np.ndarray, preds: np.ndarray) -> float:
    actual = np.asarray(actuals, dtype=float)
    pred = np.asarray(preds, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(pred) & (actual <= 0)
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs(actual[mask] - pred[mask])))


def nonzero_day_mae(actuals: np.ndarray, preds: np.ndarray) -> float:
    actual = np.asarray(actuals, dtype=float)
    pred = np.asarray(preds, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(pred) & (actual > 0)
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs(actual[mask] - pred[mask])))


def base_models_agree(
    base_preds: dict[str, np.ndarray],
    *,
    rel_tol: float = CHAMPION_AGREEMENT_REL_TOL,
) -> bool:
    """True when all base model point forecasts agree within ``rel_tol``."""
    names = [name for name in base_preds if base_preds[name] is not None]
    if len(names) < 2:
        return False

    arrs = [np.asarray(base_preds[name], dtype=float) for name in names]
    length = arrs[0].size
    if any(arr.size != length for arr in arrs):
        return False

    mask = np.ones(length, dtype=bool)
    for arr in arrs:
        mask &= np.isfinite(arr)
    if not mask.any():
        return False

    stacked = np.nanmean(np.column_stack([arr[mask] for arr in arrs]), axis=1)
    for arr in arrs:
        preds = arr[mask]
        rel_err = np.abs(preds - stacked) / np.maximum(stacked, 1e-6)
        if float(np.mean(rel_err)) > rel_tol:
            return False
    return True


def select_champion(
    *,
    drug_code: str,
    demand_segment: str,
    actuals: np.ndarray,
    base_preds: dict[str, np.ndarray],
    ensemble_preds: Optional[np.ndarray],
) -> ChampionDecision:
    """
    Choose stack only when finite OOF MASE is ≥2% better than the best base
    and better than the segment-appropriate naive baseline; else best base.
    """
    actual = np.asarray(actuals, dtype=float)
    scores: dict[str, float] = {}
    for name, preds in base_preds.items():
        scores[name] = _finite_mase(actual, preds)

    naive_name, naive_preds = segment_naive_oof(actual, demand_segment)
    scores[naive_name] = _finite_mase(actual, naive_preds)

    if ensemble_preds is not None:
        scores["ensemble"] = _finite_mase(actual, ensemble_preds)

    base_names = [n for n in base_preds if np.isfinite(scores.get(n, float("inf")))]
    if not base_names:
        return ChampionDecision(
            drug_code=drug_code,
            demand_segment=demand_segment,
            champion=naive_name,
            reason="insufficient_base_oof",
            mase_by_candidate=scores,
            beat_naive=False,
        )

    best_base = min(base_names, key=lambda n: scores[n])
    best_base_mase = scores[best_base]
    naive_mase = scores[naive_name]
    beat_naive = bool(np.isfinite(best_base_mase) and best_base_mase < naive_mase)

    champion = best_base
    reason = "best_base"

    ens_mase = scores.get("ensemble", float("inf"))
    if (
        ensemble_preds is not None
        and np.isfinite(ens_mase)
        and np.isfinite(best_base_mase)
        and best_base_mase > 0
        and ens_mase <= best_base_mase * (1.0 - CHAMPION_ENSEMBLE_IMPROVEMENT)
        and ens_mase < naive_mase
    ):
        champion = "ensemble"
        reason = "ensemble_beats_base_and_naive"
        beat_naive = True
    elif (
        ensemble_preds is not None
        and base_models_agree(base_preds)
        and np.isfinite(ens_mase)
        and np.isfinite(best_base_mase)
        and best_base_mase > 0
        and ens_mase <= best_base_mase * (1.0 + CHAMPION_AGREEMENT_MASE_TOLERANCE)
        and ens_mase < naive_mase
    ):
        champion = "ensemble"
        reason = "ensemble_model_agreement"
        beat_naive = True
    elif not beat_naive and np.isfinite(naive_mase) and naive_mase <= best_base_mase:
        # Still deploy best base for production continuity, but record failure vs naive.
        champion = best_base
        reason = "best_base_below_naive"

    return ChampionDecision(
        drug_code=drug_code,
        demand_segment=demand_segment,
        champion=champion,
        reason=reason,
        mase_by_candidate={
            k: (None if (isinstance(v, float) and not np.isfinite(v)) else float(v))
            for k, v in scores.items()
        },
        beat_naive=beat_naive,
    )


def champion_artifact_path(drug_code: str) -> str:
    return os.path.join(ARTIFACTS_DIR, "champions", f"{drug_code}.json")


def segment_champion_artifact_path(segment: str) -> str:
    return os.path.join(ARTIFACTS_DIR, "champions", f"segment_{segment}.json")


def persist_champion(decision: ChampionDecision) -> str:
    path = champion_artifact_path(decision.drug_code)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = asdict(decision)
    # Replace None MASE with null-friendly JSON
    payload["mase_by_candidate"] = {
        k: (None if v is None or (isinstance(v, float) and not np.isfinite(v)) else float(v))
        for k, v in decision.mase_by_candidate.items()
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def load_champion(drug_code: str, demand_segment: str = GLOBAL_DEMAND_SEGMENT) -> Optional[str]:
    path = champion_artifact_path(drug_code)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        return str(payload.get("champion", "ensemble"))
    seg_path = segment_champion_artifact_path(demand_segment)
    if os.path.isfile(seg_path):
        with open(seg_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        return str(payload.get("champion", "ensemble"))
    global_path = segment_champion_artifact_path(GLOBAL_DEMAND_SEGMENT)
    if os.path.isfile(global_path):
        with open(global_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        return str(payload.get("champion", "ensemble"))
    return None


def select_and_persist_champions(
    *,
    drug_oof: dict[str, dict],
    segment_by_drug: dict[str, str],
) -> list[str]:
    """
    ``drug_oof`` maps drug_code → {
        "actuals": ndarray,
        "sarima": ndarray | None,
        "lgbm": ndarray | None,
        "classical": ndarray | None,
        "ensemble": ndarray | None,
    }
    """
    artifacts: list[str] = []
    segment_votes: dict[str, list[str]] = {}
    for drug_code, payload in drug_oof.items():
        actuals = np.asarray(payload["actuals"], dtype=float)
        if actuals.size < MIN_DRUG_OOF_POINTS:
            continue
        base_preds = {
            name: np.asarray(payload[name], dtype=float)
            for name in ("sarima", "lgbm", "classical")
            if name in payload and payload[name] is not None
        }
        if not base_preds:
            continue
        segment = segment_by_drug.get(drug_code, GLOBAL_DEMAND_SEGMENT)
        decision = select_champion(
            drug_code=drug_code,
            demand_segment=segment,
            actuals=actuals,
            base_preds=base_preds,
            ensemble_preds=payload.get("ensemble"),
        )
        artifacts.append(persist_champion(decision))
        segment_votes.setdefault(segment, []).append(decision.champion)
        logger.info(
            "Champion for %s (%s): %s [%s]",
            drug_code,
            segment,
            decision.champion,
            decision.reason,
        )

    for segment, votes in segment_votes.items():
        if not votes:
            continue
        # Majority vote for segment-level fallback.
        champion = max(set(votes), key=votes.count)
        path = segment_champion_artifact_path(segment)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                {"demand_segment": segment, "champion": champion, "n_drugs": len(votes)},
                handle,
                indent=2,
            )
        artifacts.append(path)

    if segment_votes:
        all_votes = [c for votes in segment_votes.values() for c in votes]
        global_champion = max(set(all_votes), key=all_votes.count)
        path = segment_champion_artifact_path(GLOBAL_DEMAND_SEGMENT)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "demand_segment": GLOBAL_DEMAND_SEGMENT,
                    "champion": global_champion,
                    "n_drugs": len(all_votes),
                },
                handle,
                indent=2,
            )
        artifacts.append(path)

    return artifacts
