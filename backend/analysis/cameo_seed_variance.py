"""
Task 3 (variance/stability) for CAMEO.

Uses the real, unmodified `evaluate_cameo_holdout` from
`app.cold_start.cameo.validation` (the same leave-drugs-out harness used in
production `train_cameo`), fed with:

- Drug metadata: the real `matched_source` rows from
  `~/Downloads/drugs_training_synthetic_filled_with_codes.xlsx` (the same file
  `scripts/seed_drugs_from_excel.py` loads into the `drugs` table), rather than a
  live DB.
- Weekly demand series: built directly from `~/Downloads/hospital_daily_demand.csv`
  (2023-2024 dense panel), grouped the same way `panel.load_weekly_series` does
  (calendar-week sum), rather than querying `hospital_daily_demand_enriched`.

`evaluate_cameo_holdout(seed=...)` controls both (a) which drugs land in the
library vs. held-out split, and (b) the MetricNet's random weight
initialization — both real, meaningful sources of variance, not cosmetic seeds.

CAVEAT: this in-repo validator only compares CAMEO against the "Analogous"
nearest-neighbour baseline. ARIMA and the DDPFF-style cluster+analog baseline
that appear in Chapter 5's Table 5.3 are not implemented in this module, so this
script cannot reproduce that 4-way comparison — it can only report seed variance
for the CAMEO-vs-Analogous half of the story.

Usage:
    PYTHONPATH=src .venv/bin/python3 -u analysis/cameo_seed_variance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.cold_start.cameo.validation import evaluate_cameo_holdout  # noqa: E402
from app.cold_start.schemas import DrugMetadataInput  # noqa: E402

DOWNLOADS = Path.home() / "Downloads"
METADATA_XLSX = DOWNLOADS / "drugs_training_synthetic_filled_with_codes.xlsx"
DEMAND_CSV = DOWNLOADS / "hospital_daily_demand.csv"

SEEDS = [1, 2, 3, 4, 5]


def _clean(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def load_metadata_list() -> list[DrugMetadataInput]:
    di = pd.read_excel(METADATA_XLSX, sheet_name="drug_info_filled")
    mr = pd.read_excel(METADATA_XLSX, sheet_name="match_report")
    merged = di.merge(mr, on="drug_code", how="left", suffixes=("", "_mr"))
    matched = merged[merged["status"] == "matched_source"].drop_duplicates("drug_code")

    out = []
    for _, row in matched.iterrows():
        out.append(
            DrugMetadataInput(
                drug_code=str(row["drug_code"]),
                drug_name=_clean(row.get("input_drug_name")) or _clean(row.get("Generic Name")) or str(row["drug_code"]),
                generic_name=_clean(row.get("Generic Name")),
                drug_class=_clean(row.get("Drug Class")) or "MISSING",
                dosage_form=_clean(row.get("Dosage Form")) or "MISSING",
                strength=_clean(row.get("Strength")) or "",
                route_of_administration=_clean(row.get("Route of Administration")) or "MISSING",
                pregnancy_category=_clean(row.get("Pregnancy Category")) or "MISSING",
                availability=_clean(row.get("Availability")) or "MISSING",
                indications=_clean(row.get("Indications")) or "",
                side_effects=_clean(row.get("Side Effects")) or "",
                contraindications=_clean(row.get("Contraindications")) or "",
            )
        )
    return out


def load_weekly_by_code() -> dict[str, np.ndarray]:
    raw = pd.read_csv(DEMAND_CSV, usecols=["DATE", "CODE", "demand"])
    raw["DATE"] = pd.to_datetime(raw["DATE"])
    raw = raw.groupby(["DATE", "CODE"], as_index=False)["demand"].sum()
    raw["week"] = raw["DATE"].dt.to_period("W")
    weekly = raw.groupby(["CODE", "week"], as_index=False)["demand"].sum().sort_values(["CODE", "week"])
    out: dict[str, np.ndarray] = {}
    for code, group in weekly.groupby("CODE"):
        out[str(code)] = group["demand"].to_numpy(dtype=float)
    return out


def main() -> None:
    print("Loading drug metadata (matched_source only)...", flush=True)
    metadata_list = load_metadata_list()
    print(f"  {len(metadata_list)} matched_source drugs", flush=True)

    print("Building weekly demand series from dense panel...", flush=True)
    weekly_by_code = load_weekly_by_code()
    print(f"  {len(weekly_by_code)} drug codes with weekly series", flush=True)

    rows = []
    for seed in SEEDS:
        summary = evaluate_cameo_holdout(metadata_list, weekly_by_code, seed=seed)
        if summary.get("n_coldstart_test_drugs", 0) == 0:
            print(f"[seed={seed}] {summary}", flush=True)
            continue
        row = {
            "seed": seed,
            "n_library": summary["n_historical_library_drugs"],
            "n_test": summary["n_coldstart_test_drugs"],
            "cameo_pooled_acc_pct": summary["pooled_accuracy_pct"].get("CAMEO"),
            "analogous_pooled_acc_pct": summary["pooled_accuracy_pct"].get("Analogous"),
            "cameo_median_acc_pct": summary["median_accuracy_pct"].get("CAMEO"),
            "cameo_win_rate_pct": summary["win_rate_pct"].get("CAMEO"),
            "smooth_pooled_acc_pct": summary.get("smooth_pooled_accuracy_pct"),
            "conformal_coverage_pct": summary.get("empirical_conformal_coverage_pct"),
        }
        rows.append(row)
        print(f"[seed={seed}] {row}", flush=True)
        pd.DataFrame(rows).to_csv("analysis/task3_cameo_seed_variance.csv", index=False)

    df = pd.DataFrame(rows)
    if not df.empty:
        summary_cols = [c for c in df.columns if c not in ("seed", "n_library", "n_test")]
        summary_stats = df[summary_cols].agg(["mean", "std"])
        print(summary_stats, flush=True)
        summary_stats.to_csv("analysis/task3_cameo_seed_variance_summary.csv")
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
