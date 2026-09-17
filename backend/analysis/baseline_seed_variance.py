"""
Task 3 (variance/stability) for the SARIMA/LightGBM baseline comparison.

SARIMA has no stochastic training step, and the production `LightGBMModel` (see
`app/forecasting/models/lgbm_model.py`) trains with no `bagging_fraction` /
`feature_fraction` / seed set at all, so a fixed drug + fixed split gives a
deterministic result -- there is no meaningful "random seed" to vary here without
changing production hyperparameters, which would defeat the point of testing THIS
model.

The stochastic element that actually exists in Task 1's rerun is *which drugs got
sampled* into the stratified per-class draw (`baseline_rerun.py --seed`). This script
reuses the exact same classification + training code and reruns the stratified draw +
full SARIMA/LightGBM evaluation across 5 different seeds, to report the mean +/- std
of the aggregate SMAPE that a reader would actually see if a different random sample
of drugs had been chosen -- i.e. sampling variance of the headline comparison.

Usage:
    PYTHONPATH=src .venv/bin/python3 -u analysis/baseline_seed_variance.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline_rerun import (  # noqa: E402
    SKIP_TFT,
    build_drug_frame,
    classify_all,
    evaluate_one_drug,
    load_dense_panel,
)
import baseline_rerun as baseline_rerun_mod  # noqa: E402

baseline_rerun_mod.SKIP_TFT = True

SEEDS = [12345, 1, 7, 123, 2024]
PER_CLASS = 15


def main() -> None:
    raw, full_dates = load_dense_panel()
    all_codes = sorted(raw["CODE"].unique())
    print(f"Classifying {len(all_codes)} candidate drugs into SB segments (once, seed-independent)...", flush=True)
    segments = classify_all(raw, full_dates, all_codes)
    by_class: dict[str, list[str]] = {"smooth": [], "intermittent": [], "erratic": [], "lumpy": []}
    for code, seg in segments.items():
        by_class[seg].append(code)
    for cls, codes in by_class.items():
        print(f"  {cls}: {len(codes)} candidates", flush=True)

    seed_rows = []
    per_drug_rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        sample: list[tuple[str, str]] = []
        for cls, codes in by_class.items():
            codes_arr = np.array(sorted(codes))
            k = min(PER_CLASS, len(codes_arr))
            chosen = rng.choice(codes_arr, size=k, replace=False)
            sample.extend((c, cls) for c in chosen)

        t0 = time.time()
        rows = []
        for code, cls in sample:
            df = build_drug_frame(raw, full_dates, code)
            row = evaluate_one_drug(df, code)
            row["SB_class"] = cls
            row["seed"] = seed
            rows.append(row)
            per_drug_rows.append(row)
        elapsed = time.time() - t0

        dfres = pd.DataFrame(rows)
        sarima_mean = dfres["SARIMA_smape"].mean()
        lgbm_mean = dfres["LightGBM_smape"].mean()
        seed_rows.append({
            "seed": seed,
            "n_drugs": len(dfres),
            "SARIMA_mean_smape": sarima_mean,
            "LightGBM_mean_smape": lgbm_mean,
            "elapsed_s": elapsed,
        })
        print(f"[seed={seed}] n={len(dfres)} SARIMA={sarima_mean:.2f}% LightGBM={lgbm_mean:.2f}% [{elapsed:.1f}s]", flush=True)
        pd.DataFrame(seed_rows).to_csv("analysis/task3_baseline_seed_variance.csv", index=False)
        pd.DataFrame(per_drug_rows).to_csv("analysis/task3_baseline_seed_variance_per_drug.csv", index=False)

    df = pd.DataFrame(seed_rows)
    summary = df[["SARIMA_mean_smape", "LightGBM_mean_smape"]].agg(["mean", "std"])
    print(summary, flush=True)
    summary.to_csv("analysis/task3_baseline_seed_variance_summary.csv")
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
