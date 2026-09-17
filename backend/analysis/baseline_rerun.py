"""
Fresh, standalone SARIMA / LightGBM / TFT baseline rerun for paired significance
testing (Task 1a).

This intentionally does NOT go through the live database / ForecastingTrainer
pipeline (no drug_receipts table, no EM-SARIMA stockout correction, no hospital
census/supplier covariates are available outside the production DB). It uses
the *real* model classes (SarimaModel, LightGBMModel, TFTModel) and the *real*
feature-engineering modules (temporal/lag/rolling) straight from
`app.forecasting`, applied to the audited daily demand panel exported to
`~/Downloads/hospital_daily_demand.csv` (2023-01-01..2024-12-31, dense zero-fill).

This is a SIMPLIFIED rerun, clearly labelled as such: no EM-corrected demand,
no external hospital-driver features. It exists only to produce a genuine,
freshly-computed *paired* per-drug SMAPE table so a Wilcoxon/bootstrap test can
be run honestly, since the original per-SKU array behind Table 5.1 no longer
exists anywhere accessible. Its aggregate numbers are NOT expected to
reproduce 63.6/59.8/93.5 exactly and should be reported side by side with the
original, not as a replacement for it.

Usage:
    PYTHONPATH=src .venv/bin/python analysis/baseline_rerun.py --probe
    PYTHONPATH=src .venv/bin/python analysis/baseline_rerun.py --per-class 15 --out analysis/task1_baseline_per_drug.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# `app.forecasting.feature_engineering.__init__` eagerly imports `pipeline.py`,
# which imports `external_features.py`, which imports `app.models.hospital_census`.
# That module does not exist in this checkout (dead import in the live repo,
# unrelated to this rerun) — stub it out so we can import the pure-pandas
# feature-engineering helpers (temporal/lag/rolling) without a live DB.
_stub_models_pkg = types.ModuleType("app.models.hospital_census")


class _StubHospitalCensus:  # pragma: no cover - never queried, DB path unused here
    census_date = None
    bed_occupancy_rate = None
    weekly_surgery_count = None
    center_syn_id = None


_stub_models_pkg.HospitalCensus = _StubHospitalCensus
sys.modules.setdefault("app.models.hospital_census", _stub_models_pkg)

from app.forecasting.demand_segmentation import classify_demand_segment_from_frame  # noqa: E402
from app.forecasting.evaluation_metrics import smape as smape_single  # noqa: E402
from app.forecasting.feature_engineering.lag_features import add_lag_features  # noqa: E402
from app.forecasting.feature_engineering.rolling_features import add_rolling_features  # noqa: E402
from app.forecasting.feature_engineering.temporal_features import add_temporal_features  # noqa: E402
from app.forecasting.models.sarima_model import SarimaModel  # noqa: E402
from app.forecasting.models.lgbm_model import LightGBMModel  # noqa: E402
from app.forecasting.models.tft_model import TFTModel  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("baseline_rerun")

HORIZON = 30  # matches app.forecasting.constants.FORECAST_HORIZON
SKIP_TFT = False
DOWNLOADS = Path.home() / "Downloads"
PANEL_2YEAR = DOWNLOADS / "hospital_daily_demand.csv"
PANEL_2023 = DOWNLOADS / "hospital_daily_demand_2023.csv"


def load_dense_panel() -> pd.DataFrame:
    """Build a dense (zero-filled) 2023-2024 daily panel restricted to the
    canonical 882-drug universe (the CODE set present in the 2023-only export,
    which matches the thesis's reported drug count exactly: 882 codes)."""
    codes_882 = set(pd.read_csv(PANEL_2023, usecols=["CODE"])["CODE"].unique())
    raw = pd.read_csv(PANEL_2YEAR, usecols=["DATE", "CODE", "demand"])
    raw["DATE"] = pd.to_datetime(raw["DATE"])
    raw = raw[raw["CODE"].isin(codes_882)]
    full_dates = pd.date_range(raw["DATE"].min(), raw["DATE"].max(), freq="D")
    logger.warning(
        "Loaded %d rows, %d/%d canonical drugs present, %d calendar days (%s..%s)",
        len(raw), raw["CODE"].nunique(), len(codes_882), len(full_dates),
        full_dates[0].date(), full_dates[-1].date(),
    )
    return raw, full_dates


def build_drug_frame(raw: pd.DataFrame, full_dates: pd.DatetimeIndex, code: str) -> pd.DataFrame:
    """Dense zero-filled + feature-engineered frame for one drug code."""
    sub = raw[raw["CODE"] == code][["DATE", "demand"]].groupby("DATE", as_index=False).sum()
    sub = sub.set_index("DATE").reindex(full_dates, fill_value=0.0).rename_axis("demand_date")
    df = sub.reset_index().rename(columns={"demand": "total_quantity"})
    df["is_coverage_gap"] = 0
    df = add_temporal_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df["demand_date"] = pd.to_datetime(df["demand_date"])
    df = df.set_index("demand_date").sort_index()
    df.index.name = "demand_date"
    return df


def classify_all(raw: pd.DataFrame, full_dates: pd.DatetimeIndex, codes: list[str]) -> dict[str, str]:
    segments: dict[str, str] = {}
    for code in codes:
        df = build_drug_frame(raw, full_dates, code)
        segments[code] = classify_demand_segment_from_frame(df)
    return segments


def evaluate_one_drug(df: pd.DataFrame, code: str, *, tft_epochs: int | None = None) -> dict:
    """Train SARIMA/LightGBM/TFT on train split, forecast HORIZON days, return SMAPE per model."""
    train_df = df.iloc[:-HORIZON].copy()
    test_df = df.iloc[-HORIZON:].copy()
    actuals = test_df["total_quantity"].astype(float).values

    result = {"drug": code, "n_days": len(df)}
    timings = {}

    model_specs = [
        ("SARIMA", SarimaModel, {}),
        ("LightGBM", LightGBMModel, {}),
    ]
    if not SKIP_TFT:
        model_specs.append(("TFT", TFTModel, {"max_epochs": tft_epochs} if tft_epochs is not None else {}))

    for name, ModelCls, extra in model_specs:
        t0 = time.time()
        try:
            model = ModelCls()
            model.train(train_df, code, **extra)
            preds = model.predict(train_df, HORIZON)
            pred_vals = preds["p50"].astype(float).values[:HORIZON]
            if len(pred_vals) < HORIZON:
                pred_vals = np.pad(pred_vals, (0, HORIZON - len(pred_vals)), constant_values=pred_vals[-1] if len(pred_vals) else 0.0)
            drug_smape = float(np.mean([smape_single(float(a), float(p)) for a, p in zip(actuals, pred_vals)]))
            result[f"{name}_smape"] = drug_smape
            result[f"{name}_status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s failed for %s: %s", name, code, exc)
            result[f"{name}_smape"] = np.nan
            result[f"{name}_status"] = f"error: {exc}"[:200]
        timings[name] = time.time() - t0

    result["t_sarima_s"] = timings["SARIMA"]
    result["t_lgbm_s"] = timings["LightGBM"]
    result["t_tft_s"] = timings.get("TFT", 0.0)
    if SKIP_TFT:
        result["TFT_smape"] = np.nan
        result["TFT_status"] = "skipped"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true", help="Time a couple of drugs, print, exit")
    parser.add_argument("--per-class", type=int, default=15)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--tft-epochs", type=int, default=None, help="Override TFT max_epochs for probing speed")
    parser.add_argument("--skip-tft", action="store_true", help="Skip TFT entirely (SARIMA+LightGBM only, fast)")
    parser.add_argument("--out", type=str, default="analysis/task1_baseline_per_drug.csv")
    args = parser.parse_args()
    if args.skip_tft:
        global SKIP_TFT
        SKIP_TFT = True

    raw, full_dates = load_dense_panel()
    all_codes = sorted(raw["CODE"].unique())

    if args.probe:
        probe_codes = all_codes[:1] + all_codes[len(all_codes) // 2 : len(all_codes) // 2 + 1]
        for code in probe_codes:
            df = build_drug_frame(raw, full_dates, code)
            seg = classify_demand_segment_from_frame(df)
            print(f"--- probing {code} (segment={seg}, {len(df)} days) ---")
            t0 = time.time()
            res = evaluate_one_drug(df, code, tft_epochs=args.tft_epochs)
            print(res)
            print(f"total wall time: {time.time() - t0:.1f}s")
        return

    rng = np.random.default_rng(args.seed)
    logger.warning("Classifying %d candidate drugs into SB segments...", len(all_codes))
    segments = classify_all(raw, full_dates, all_codes)
    by_class: dict[str, list[str]] = {"smooth": [], "intermittent": [], "erratic": [], "lumpy": []}
    for code, seg in segments.items():
        by_class[seg].append(code)
    for cls, codes in by_class.items():
        logger.warning("%s: %d candidates", cls, len(codes))

    sample: list[tuple[str, str]] = []
    for cls, codes in by_class.items():
        codes_arr = np.array(sorted(codes))
        k = min(args.per_class, len(codes_arr))
        chosen = rng.choice(codes_arr, size=k, replace=False)
        sample.extend((c, cls) for c in chosen)
    logger.warning("Sampled %d drugs total: %s", len(sample), {c: sum(1 for _, cc in sample if cc == c) for c in by_class})

    rows = []
    t_start = time.time()
    for i, (code, cls) in enumerate(sample, 1):
        df = build_drug_frame(raw, full_dates, code)
        row = evaluate_one_drug(df, code, tft_epochs=args.tft_epochs)
        row["SB_class"] = cls
        rows.append(row)
        elapsed = time.time() - t_start
        logger.warning(
            "[%d/%d] %s (%s) done in %.1fs (sarima=%.1fs lgbm=%.1fs tft=%.1fs) | elapsed=%.1fmin",
            i, len(sample), code, cls,
            row["t_sarima_s"] + row["t_lgbm_s"] + row["t_tft_s"],
            row["t_sarima_s"], row["t_lgbm_s"], row["t_tft_s"],
            elapsed / 60,
        )
        pd.DataFrame(rows).to_csv(args.out, index=False)  # checkpoint after every drug

    pd.DataFrame(rows).to_csv(args.out, index=False)
    logger.warning("Done. Wrote %s", args.out)


if __name__ == "__main__":
    main()
