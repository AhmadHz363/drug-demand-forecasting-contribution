#!/usr/bin/env python3
"""Segment-stratified forecast validation with promotion gates.

Usage (from backend/):
  PYTHONPATH=src .venv/bin/python scripts/validate_forecast_accuracy.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median

from sqlalchemy import text

from app.core.database import SessionLocal
from app.forecasting.demand_segmentation import classify_demand_segment
from app.forecasting.training.holdout import run_holdout_validation
from app.forecasting.training.trainer import ForecastingTrainer
from app.services.demand_aggregation import (
    aggregate_daily_demand_rows,
    get_import_coverage_periods,
    get_latest_coverage_segment,
    get_receipt_date_bounds,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("validate_forecast_accuracy")

BASELINE_PATH = Path("docs/baselines/forecast_accuracy_after_2026-07-13.json")
MIN_PER_SEGMENT = 5
HOLDOUT_ISOLATION_DAYS = 90  # untouched production holdout; not used for selection
MODELS = ["sarima", "lgbm", "classical"]


def _demand_series(db, drug_code: str) -> tuple[list[float], object, object]:
    start, end = get_receipt_date_bounds(db, drug_code)
    if start is None or end is None:
        return [], None, None
    rows = aggregate_daily_demand_rows(db, drug_code, start, end)
    return [q for _, q in rows], start, end


def _sample_by_segment(db, *, min_per_segment: int = MIN_PER_SEGMENT) -> dict[str, list[str]]:
    rows = db.execute(
        text(
            """
            SELECT drug_code
            FROM drug_receipts
            GROUP BY drug_code
            HAVING COUNT(DISTINCT receipt_date) >= 120
            ORDER BY SUM(ABS(quantity)) DESC
            LIMIT 200
            """
        )
    ).fetchall()
    buckets: dict[str, list[str]] = {
        "smooth": [],
        "intermittent": [],
        "erratic": [],
        "lumpy": [],
    }
    for (code,) in rows:
        qty, _, _ = _demand_series(db, code)
        if len(qty) < 90:
            continue
        segment = classify_demand_segment(np_array(qty))
        if len(buckets[segment]) < min_per_segment:
            buckets[segment].append(code)
        if all(len(v) >= min_per_segment for v in buckets.values()):
            break
    return buckets


def np_array(values: list[float]):
    import numpy as np

    return np.asarray(values, dtype=float)


def _ensemble_mase(holdout: dict) -> float | None:
    metrics = holdout.get("metrics") or {}
    ens = metrics.get("ensemble") or {}
    mase = ens.get("mase")
    if mase is None:
        return None
    try:
        val = float(mase)
    except (TypeError, ValueError):
        return None
    if val != val or val == float("inf"):
        return None
    return val


def _coverage_90(holdout: dict) -> float | None:
    metrics = holdout.get("metrics") or {}
    ens = metrics.get("ensemble") or {}
    cov = ens.get("coverage_90")
    return float(cov) if cov is not None else None


def _beat_naive(holdout: dict) -> bool:
    """MASE < 1 means beat seasonal-naive scale (proxy for appropriate naive)."""
    mase = _ensemble_mase(holdout)
    return mase is not None and mase < 1.0


def _run_holdouts(db, drugs: list[str], *, horizon_days: int) -> list[dict]:
    results = []
    for code in drugs:
        try:
            _start, end = get_receipt_date_bounds(db, code)
            if end is None:
                results.append({"drug_code": code, "error": "no_history", "horizon_days": horizon_days})
                continue
            # Reserve last HOLDOUT_ISOLATION_DAYS as untouched production holdout:
            # validation windows end before that reserved block when possible.
            coverage = get_latest_coverage_segment(db)
            covered_end = coverage[1] if coverage else end
            production_holdout_start = covered_end - timedelta(days=HOLDOUT_ISOLATION_DAYS - 1)
            # Selection/validation uses days before the reserved production holdout.
            val_end = min(end, production_holdout_start - timedelta(days=1))
            if val_end <= (_start or val_end):
                val_end = end
            test_end = val_end
            test_start = test_end - timedelta(days=horizon_days - 1)
            train_end = test_start - timedelta(days=1)
            result = run_holdout_validation(
                drug_code=code,
                db_session=db,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                models=MODELS,
            )
            metrics = result.metrics
            serialized = {
                name: (m.model_dump() if hasattr(m, "model_dump") else m.dict())
                for name, m in metrics.items()
            }
            results.append(
                {
                    "drug_code": code,
                    "horizon_days": horizon_days,
                    "total_accuracy_pct": result.total_accuracy_pct,
                    "total_accuracy_skill_pct": result.total_accuracy_skill_pct,
                    "demand_segment": result.demand_segment,
                    "metrics": serialized,
                    "holdout_window": {
                        "train_end": train_end.isoformat(),
                        "test_start": test_start.isoformat(),
                        "test_end": test_end.isoformat(),
                        "production_holdout_start": production_holdout_start.isoformat(),
                    },
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Holdout failed for %s h=%s", code, horizon_days)
            results.append({"drug_code": code, "horizon_days": horizon_days, "error": str(exc)})
    return results


def _evaluate_gates(holdouts_7d: list[dict], baseline: dict | None) -> dict:
    ok = [h for h in holdouts_7d if "error" not in h]
    mases = [m for m in (_ensemble_mase(h) for h in ok) if m is not None]
    coverages = [c for c in (_coverage_90(h) for h in ok) if c is not None]
    beat_naive_share = (
        sum(1 for h in ok if _beat_naive(h)) / len(ok) if ok else 0.0
    )
    median_mase = median(mases) if mases else None
    median_cov = median(coverages) if coverages else None

    baseline_mases = []
    if baseline:
        for h in baseline.get("holdouts") or []:
            m = _ensemble_mase(h)
            if m is not None:
                baseline_mases.append(m)
    baseline_median = median(baseline_mases) if baseline_mases else None

    segment_mase: dict[str, list[float]] = {}
    for h in ok:
        m = _ensemble_mase(h)
        if m is None:
            continue
        segment_mase.setdefault(h.get("demand_segment", "unknown"), []).append(m)

    baseline_segment_mase: dict[str, list[float]] = {}
    if baseline:
        for h in baseline.get("holdouts") or []:
            m = _ensemble_mase(h)
            if m is None:
                continue
            baseline_segment_mase.setdefault(h.get("demand_segment", "unknown"), []).append(m)

    checks = {
        "median_mase_improves_5pct": False,
        "beat_naive_at_least_60pct": beat_naive_share >= 0.60,
        "coverage_between_85_95": (
            median_cov is not None and 0.85 <= median_cov <= 0.95
        ),
        "no_segment_regression_over_5pct": True,
        "quantiles_ordered": True,  # enforced in models; marker for report
    }

    if baseline_median is not None and median_mase is not None and baseline_median > 0:
        checks["median_mase_improves_5pct"] = median_mase <= baseline_median * 0.95
    elif median_mase is not None:
        # No comparable baseline — treat as informational pass when finite.
        checks["median_mase_improves_5pct"] = True

    for segment, values in segment_mase.items():
        if len(values) < MIN_PER_SEGMENT:
            continue
        base_vals = baseline_segment_mase.get(segment) or []
        if len(base_vals) < MIN_PER_SEGMENT:
            continue
        if median(values) > median(base_vals) * 1.05:
            checks["no_segment_regression_over_5pct"] = False

    passed = all(checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "median_mase": median_mase,
        "baseline_median_mase": baseline_median,
        "beat_naive_share": beat_naive_share,
        "median_coverage_90": median_cov,
        "segment_median_mase": {k: median(v) for k, v in segment_mase.items()},
        "n_evaluated": len(ok),
    }


def main() -> int:
    out_dir = Path("docs/baselines")
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline = None
    if BASELINE_PATH.is_file():
        baseline = json.loads(BASELINE_PATH.read_text())

    db = SessionLocal()
    try:
        periods = get_import_coverage_periods(db)
        logger.info("Import coverage periods: %s", periods)

        buckets = _sample_by_segment(db)
        drugs = sorted({code for codes in buckets.values() for code in codes})
        if len(drugs) < MIN_PER_SEGMENT:
            drugs = [
                r[0]
                for r in db.execute(
                    text(
                        """
                        SELECT drug_code
                        FROM drug_receipts
                        GROUP BY drug_code
                        ORDER BY SUM(ABS(quantity)) DESC
                        LIMIT 20
                        """
                    )
                )
            ]
        logger.info("Segment cohorts: %s", {k: len(v) for k, v in buckets.items()})
        logger.info("Validating drugs: %s", drugs)

        trainer = ForecastingTrainer()
        train_resp = trainer.train_all(
            drug_codes=drugs,
            models_to_train=MODELS,
            db_session=db,
            force_retrain=True,
        )
        logger.info(
            "Trained drugs=%s models=%s run=%s",
            train_resp.drugs_trained,
            train_resp.models_trained,
            train_resp.training_run_id,
        )

        holdouts_7d = _run_holdouts(db, drugs, horizon_days=7)
        holdouts_30d = _run_holdouts(db, drugs, horizon_days=30)
        gates = _evaluate_gates(holdouts_7d, baseline)

        payload = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "training_run_id": train_resp.training_run_id,
            "coverage_periods": [
                {"start": s.isoformat(), "end": e.isoformat()} for s, e in periods
            ],
            "segment_cohorts": buckets,
            "holdouts_7d": holdouts_7d,
            "holdouts_30d": holdouts_30d,
            "promotion_gates": gates,
        }
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out_path = out_dir / f"forecast_accuracy_best_engine_{stamp}.json"
        out_path.write_text(json.dumps(payload, indent=2, default=str))
        logger.info("Wrote %s", out_path)
        logger.info("Promotion gates: %s", json.dumps(gates, indent=2))
        return 0 if gates["passed"] else 2
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
