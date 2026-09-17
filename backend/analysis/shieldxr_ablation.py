"""
SHIELD-XR ablation study (Task 2) — real pipeline, real data, standalone (no DB).

Uses the actual production modules under
`app.forecasting.shield_xr.{anomaly_guard,features,weekly_model,evaluation,metrics}`
unmodified. Only `train_shieldxr_stack` (occurrence/magnitude hurdle + EVT + isotonic +
sample weights) and `select_class_ensemble` (class-conditional model selection) are
reimplemented here as ablatable copies of the real functions in
`app.forecasting.shield_xr.daily_model`, each with a boolean flag that reproduces the
exact original behaviour when True, so the "all flags on" run is a faithful rerun of
the production stack.

Panel data: dense CODE x DATE demand from `hospital_daily_demand.csv` (2023-2024,
zero-filled), left-joined with the real exogenous hospital-driver columns from the two
`hospital_daily_demand_enriched_*.xlsx` exports (patient/doctor/admission/CR-share
counts). Gaps in the exogenous join (the enriched export is not fully dense) are
zero-filled — consistent with how `panel_builder.build_hospital_panel` already
zero-fills `bed_occupancy_rate`/`weekly_surgery_count` for the two features that were
never collected at all.

Usage:
    PYTHONPATH=src .venv/bin/python3 analysis/shieldxr_ablation.py
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_stub = types.ModuleType("app.models.hospital_census")


class _StubHospitalCensus:
    census_date = None
    bed_occupancy_rate = None
    weekly_surgery_count = None
    center_syn_id = None


_stub.HospitalCensus = _StubHospitalCensus
sys.modules.setdefault("app.models.hospital_census", _stub)

import lightgbm as lgb  # noqa: E402
from scipy.stats import genpareto  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from app.forecasting.shield_xr.anomaly_guard import apply_anomaly_guard  # noqa: E402
from app.forecasting.shield_xr.daily_model import (  # noqa: E402
    DailyEnsembleArtifacts,
    ShieldXRStackResult,
    _xy,
    train_plain_models,
)
from app.forecasting.shield_xr.evaluation import compute_training_metrics  # noqa: E402
from app.forecasting.shield_xr.features import (  # noqa: E402
    CAT_COLS,
    add_features,
    add_sample_weights,
    apply_spike_labels,
    assign_sb_classes,
    encode_ids,
)
from app.forecasting.shield_xr.metrics import accuracy_from_wape, wape  # noqa: E402
from app.forecasting.shield_xr.panel_builder import split_temporal  # noqa: E402
from app.forecasting.shield_xr.weekly_model import (  # noqa: E402
    build_weekly_panel,
    reconcile_weekly_breakdown,
    train_weekly_breakdown,
)

DOWNLOADS = Path.home() / "Downloads"
RANDOM_STATE = 42


# --------------------------------------------------------------------------------------
# Panel construction (standalone, no DB)
# --------------------------------------------------------------------------------------

def load_panel() -> pd.DataFrame:
    dense = pd.read_csv(DOWNLOADS / "hospital_daily_demand.csv", usecols=["DATE", "CODE", "ARTICLE", "demand"])
    dense["DATE"] = pd.to_datetime(dense["DATE"])
    dense = dense.groupby(["DATE", "CODE", "ARTICLE"], as_index=False)["demand"].sum()

    exog_cols = [
        "n_unique_patients", "n_admissions", "n_unique_doctors", "n_unique_CR", "n_unique_CS",
        "n_transactions", "n_demand_txns", "top1_CR_share", "top2_CR_share", "top3_CR_share", "CAT",
    ]
    frames = []
    for fname in ("hospital_daily_demand_enriched_2023.xlsx", "hospital_daily_demand_enriched_2024.xlsx"):
        xl = pd.ExcelFile(DOWNLOADS / fname)
        part = xl.parse("daily_demand_enriched")
        part["DATE"] = pd.to_datetime(part["DATE"])
        frames.append(part[["DATE", "CODE"] + exog_cols])
    enriched = pd.concat(frames, ignore_index=True).drop_duplicates(["DATE", "CODE"], keep="last")

    panel = dense.merge(enriched, on=["DATE", "CODE"], how="left")
    numeric_exog = [c for c in exog_cols if c != "CAT"]
    for col in numeric_exog:
        panel[col] = pd.to_numeric(panel[col], errors="coerce").fillna(0.0)
    panel["CAT"] = panel["CAT"].astype(str).replace({"nan": "UNK", "None": "UNK"}).fillna("UNK")
    panel["bed_occupancy_rate"] = 0.0
    panel["weekly_surgery_count"] = 0.0
    panel = panel.sort_values(["CODE", "DATE"]).reset_index(drop=True)
    return panel


# --------------------------------------------------------------------------------------
# Ablatable copy of daily_model.train_shieldxr_stack
# --------------------------------------------------------------------------------------

def train_shieldxr_stack_ablatable(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    *,
    use_evt: bool = True,
    use_isotonic: bool = True,
    use_sample_weights: bool = True,
) -> ShieldXRStackResult:
    cats = [c for c in CAT_COLS if c in feature_cols]
    x_train, y_train = _xy(train, "occur", feature_cols)
    x_valid, y_valid = _xy(valid, "occur", feature_cols)
    x_test, _ = _xy(test, "occur", feature_cols)

    occur = lgb.LGBMClassifier(
        n_estimators=700, learning_rate=0.05, num_leaves=47, subsample=0.8,
        colsample_bytree=0.8, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    )
    occur.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], categorical_feature=cats,
              callbacks=[lgb.early_stopping(40, verbose=False)])
    p_occ_valid = occur.predict_proba(x_valid)[:, 1]
    p_occ_test = occur.predict_proba(x_test)[:, 1]

    isotonic: dict[str, IsotonicRegression] = {}
    if use_isotonic:
        p_occ_valid_cal = p_occ_valid.copy()
        p_occ_test_cal = p_occ_test.copy()
        for cls in valid["sb_class"].unique():
            mask_valid = (valid["sb_class"] == cls).values
            mask_test = (test["sb_class"] == cls).values
            if mask_valid.sum() < 30 or valid.loc[mask_valid, "occur"].nunique() < 2:
                continue
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p_occ_valid[mask_valid], valid.loc[mask_valid, "occur"].values)
            isotonic[str(cls)] = iso
            p_occ_valid_cal[mask_valid] = iso.transform(p_occ_valid[mask_valid])
            if mask_test.sum():
                p_occ_test_cal[mask_test] = iso.transform(p_occ_test[mask_test])
        p_occ_valid, p_occ_test = p_occ_valid_cal, p_occ_test_cal

    sw_train = train["_w"] if (use_sample_weights and "_w" in train.columns) else None
    sw_valid = valid["_w"] if (use_sample_weights and "_w" in valid.columns) else None

    mask_train_normal = (train["demand"] > 0) & (train["spike"] == 0)
    mask_valid_normal = (valid["demand"] > 0) & (valid["spike"] == 0)
    normal = lgb.LGBMRegressor(
        n_estimators=900, learning_rate=0.05, num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        objective="tweedie", tweedie_variance_power=1.15, random_state=RANDOM_STATE, n_jobs=-1,
    )
    normal.fit(
        *_xy(train.loc[mask_train_normal], "demand", feature_cols),
        sample_weight=sw_train.loc[mask_train_normal] if sw_train is not None else None,
        eval_set=[_xy(valid.loc[mask_valid_normal], "demand", feature_cols)] if mask_valid_normal.sum() else None,
        eval_sample_weight=[sw_valid.loc[mask_valid_normal]] if (mask_valid_normal.sum() and sw_valid is not None) else None,
        categorical_feature=cats,
        callbacks=[lgb.early_stopping(50, verbose=False)] if mask_valid_normal.sum() else None,
    )
    yhat_n_valid = np.clip(normal.predict(_xy(valid, "demand", feature_cols)[0]), 0, None)
    yhat_n_test = np.clip(normal.predict(_xy(test, "demand", feature_cols)[0]), 0, None)

    full_reg = lgb.LGBMRegressor(
        n_estimators=900, learning_rate=0.05, num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        objective="tweedie", tweedie_variance_power=1.15, random_state=RANDOM_STATE, n_jobs=-1,
    )
    full_reg.fit(
        *_xy(train, "demand", feature_cols),
        sample_weight=sw_train,
        eval_set=[_xy(valid, "demand", feature_cols)],
        eval_sample_weight=[sw_valid] if sw_valid is not None else None,
        categorical_feature=cats,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    yhat_full_valid = np.clip(full_reg.predict(_xy(valid, "demand", feature_cols)[0]), 0, None)
    yhat_full_test = np.clip(full_reg.predict(_xy(test, "demand", feature_cols)[0]), 0, None)

    spike_clf = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.05, num_leaves=47, subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    )
    spike_clf.fit(*_xy(train, "spike", feature_cols), eval_set=[_xy(valid, "spike", feature_cols)],
                  categorical_feature=cats, callbacks=[lgb.early_stopping(40, verbose=False)])
    p_spk_valid = spike_clf.predict_proba(_xy(valid, "spike", feature_cols)[0])[:, 1]
    p_spk_test = spike_clf.predict_proba(_xy(test, "spike", feature_cols)[0])[:, 1]

    mask_train_spike = train["spike"] == 1
    if mask_train_spike.sum() >= 40:
        spike_q = lgb.LGBMRegressor(
            n_estimators=700, learning_rate=0.05, num_leaves=31, objective="quantile", alpha=0.90,
            subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1,
        )
        mask_valid_spike = valid["spike"] == 1
        spike_q.fit(
            *_xy(train.loc[mask_train_spike], "demand", feature_cols),
            eval_set=[_xy(valid.loc[mask_valid_spike], "demand", feature_cols)] if mask_valid_spike.sum() else None,
            categorical_feature=cats,
            callbacks=[lgb.early_stopping(40, verbose=False)] if mask_valid_spike.sum() else None,
        )
        yhat_s_valid = np.clip(spike_q.predict(_xy(valid, "demand", feature_cols)[0]), 0, None)
        yhat_s_test = np.clip(spike_q.predict(_xy(test, "demand", feature_cols)[0]), 0, None)
    else:
        p90 = train.groupby("CODE")["demand"].quantile(0.90)
        yhat_s_valid = valid["CODE"].map(p90).fillna(train["demand"].quantile(0.90)).values
        yhat_s_test = test["CODE"].map(p90).fillna(train["demand"].quantile(0.90)).values
        spike_q = None

    evt_frac = 0.0
    if use_evt:
        resid = valid["demand"].values - yhat_n_valid
        threshold = np.quantile(resid, 0.90)
        excess = resid[resid > threshold] - threshold
        if len(excess) >= 30:
            shape, _, scale = genpareto.fit(excess, floc=0)
            evt_abs = float(scale / (1 - shape)) if shape < 1 else float(np.mean(excess))
        else:
            evt_abs = float(np.mean(excess)) if len(excess) else 0.0
        evt_abs = max(evt_abs, 0.0)
        baseline_scale = float(np.mean(yhat_n_valid[yhat_n_valid > 0])) if (yhat_n_valid > 0).any() else 1.0
        evt_frac = float(np.clip(evt_abs / max(baseline_scale, 1e-6), 0.0, 3.0))
        yhat_s_valid = np.maximum(yhat_s_valid, yhat_n_valid * (1 + evt_frac))
        yhat_s_test = np.maximum(yhat_s_test, yhat_n_test * (1 + evt_frac))

    is_smoothish_valid = valid["sb_class"].isin(["smooth", "erratic"]).values
    is_smoothish_test = test["sb_class"].isin(["smooth", "erratic"]).values
    best = {"wape": 1e9, "t_spk": 1.01, "hard_occ_t": None, "w_smooth": 0.0, "bias_c": 1.0}
    for t_spk in (1.01, 0.85, 0.70, 0.50, 0.35, 0.20):
        use_spk_valid = p_spk_valid >= t_spk
        mag_valid = np.where(use_spk_valid, yhat_s_valid, yhat_n_valid)
        for hard_occ_t in (None, 0.50, 0.40, 0.30, 0.20, 0.10):
            occ_term_valid = p_occ_valid if hard_occ_t is None else (p_occ_valid >= hard_occ_t).astype(float)
            hurdle_valid = occ_term_valid * mag_valid
            for w_smooth in np.linspace(0.0, 0.9, 10):
                pred_valid = hurdle_valid.copy()
                pred_valid[is_smoothish_valid] = (
                    (1 - w_smooth) * hurdle_valid[is_smoothish_valid] + w_smooth * yhat_full_valid[is_smoothish_valid]
                )
                s_pred = pred_valid.sum()
                bias_c = float(np.clip(valid["demand"].sum() / s_pred, 0.5, 2.0)) if s_pred > 0 else 1.0
                score = wape(valid["demand"].values, pred_valid * bias_c)
                if score < best["wape"]:
                    best = {"wape": score, "t_spk": t_spk, "hard_occ_t": hard_occ_t,
                            "w_smooth": float(w_smooth), "bias_c": bias_c}

    use_spk_test = p_spk_test >= best["t_spk"]
    mag_test = np.where(use_spk_test, yhat_s_test, yhat_n_test)
    occ_term_test = p_occ_test if best["hard_occ_t"] is None else (p_occ_test >= best["hard_occ_t"]).astype(float)
    hurdle_test = occ_term_test * mag_test
    pred_test = hurdle_test.copy()
    pred_test[is_smoothish_test] = (
        (1 - best["w_smooth"]) * hurdle_test[is_smoothish_test] + best["w_smooth"] * yhat_full_test[is_smoothish_test]
    )
    pred_test = np.clip(pred_test * best["bias_c"], 0, None)

    use_spk_valid_f = p_spk_valid >= best["t_spk"]
    mag_valid_f = np.where(use_spk_valid_f, yhat_s_valid, yhat_n_valid)
    occ_term_valid_f = p_occ_valid if best["hard_occ_t"] is None else (p_occ_valid >= best["hard_occ_t"]).astype(float)
    hurdle_valid_f = occ_term_valid_f * mag_valid_f
    pred_valid_f = hurdle_valid_f.copy()
    pred_valid_f[is_smoothish_valid] = (
        (1 - best["w_smooth"]) * hurdle_valid_f[is_smoothish_valid] + best["w_smooth"] * yhat_full_valid[is_smoothish_valid]
    )
    pred_valid_f = np.clip(pred_valid_f * best["bias_c"], 0, None)

    try:
        auc = roc_auc_score(valid["occur"], p_occ_valid)
    except ValueError:
        auc = float("nan")
    del auc

    return ShieldXRStackResult(
        pred=pred_test, pred_valid=pred_valid_f, p_occur=p_occ_test, p_spike=p_spk_test,
        gate_config=best, evt_frac=evt_frac, isotonic_calibrators=isotonic,
        models={"occur": occur, "normal": normal, "full_reg": full_reg, "spike_clf": spike_clf,
                "spike_q": spike_q, "yhat_n": normal, "yhat_full": full_reg,
                "yhat_s_fallback_p90": train.groupby("CODE")["demand"].quantile(0.90).to_dict()},
    )


def select_class_ensemble_ablatable(
    valid: pd.DataFrame,
    test: pd.DataFrame,
    shield_xr: ShieldXRStackResult,
    plain_pred_valid: np.ndarray,
    plain_pred: np.ndarray,
    plain_l1_pred_valid: np.ndarray,
    plain_l1_pred: np.ndarray,
    *,
    class_conditional: bool = True,
) -> tuple[dict[str, str], dict[str, float], np.ndarray]:
    candidates = {
        "SHIELD-XR": (shield_xr.pred_valid, shield_xr.pred),
        "Plain": (plain_pred_valid, plain_pred),
        "Plain-L1": (plain_l1_pred_valid, plain_l1_pred),
    }
    class_winner: dict[str, str] = {}
    class_bias: dict[str, float] = {}
    ensemble_pred = np.zeros(len(test))

    if class_conditional:
        groups = list(valid["sb_class"].unique())
    else:
        groups = ["__global__"]

    for cls in groups:
        mask = (valid["sb_class"] == cls).values if class_conditional else np.ones(len(valid), dtype=bool)
        if mask.sum() == 0:
            continue
        y_valid = valid.loc[mask, "demand"].values
        scores = {name: wape(y_valid, pred[mask]) for name, (pred, _) in candidates.items()}
        winner = min(scores, key=scores.get)
        va_pred = candidates[winner][0][mask]
        total = va_pred.sum()
        bias = float(np.clip(y_valid.sum() / total, 0.75, 1.35)) if total > 0 else 1.0

        if class_conditional:
            class_winner[str(cls)] = winner
            class_bias[str(cls)] = bias
            mask_test = (test["sb_class"] == cls).values
            ensemble_pred[mask_test] = candidates[winner][1][mask_test] * bias
        else:
            for real_cls in valid["sb_class"].unique():
                class_winner[str(real_cls)] = winner
                class_bias[str(real_cls)] = bias
            ensemble_pred[:] = candidates[winner][1] * bias

    ensemble_pred = np.clip(ensemble_pred, 0, None)
    return class_winner, class_bias, ensemble_pred


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------

def prepare_splits(panel: pd.DataFrame):
    d_train_end, d_valid_end = split_temporal(panel)
    guarded = apply_anomaly_guard(panel, d_train_end)
    cleaned = assign_sb_classes(guarded.frame, d_train_end)
    feat = add_features(cleaned)

    train = feat[feat["DATE"] <= d_train_end].copy()
    valid = feat[(feat["DATE"] > d_train_end) & (feat["DATE"] <= d_valid_end)].copy()
    test = feat[feat["DATE"] > d_valid_end].copy()

    train_stats = train.groupby("CODE")["demand"].agg(mean="mean", std="std")
    train = apply_spike_labels(train, train_stats)
    valid = apply_spike_labels(valid, train_stats)
    test = apply_spike_labels(test, train_stats)

    all_codes = sorted(feat["CODE"].astype(str).unique())
    all_cats = sorted(feat["CAT"].astype(str).fillna("UNK").unique())
    code2id, cat2id = encode_ids(train, valid, test, all_codes, all_cats)
    add_sample_weights(train, valid, test)
    return cleaned, train, valid, test, d_train_end, d_valid_end, code2id, cat2id


def run_variant(
    label: str,
    cleaned, train, valid, test, d_train_end, d_valid_end, code2id, cat2id,
    feature_cols: list[str],
    *,
    use_evt: bool = True,
    use_isotonic: bool = True,
    use_sample_weights: bool = True,
    class_conditional: bool = True,
) -> dict:
    t0 = time.time()
    shield_xr = train_shieldxr_stack_ablatable(
        train, valid, test, feature_cols,
        use_evt=use_evt, use_isotonic=use_isotonic, use_sample_weights=use_sample_weights,
    )
    plain, plain_l1, plain_pred, plain_pred_valid, plain_l1_pred, plain_l1_pred_valid = train_plain_models(
        train, valid, test, feature_cols
    )
    class_winner, class_bias, ensemble_pred = select_class_ensemble_ablatable(
        valid, test, shield_xr, plain_pred_valid, plain_pred, plain_l1_pred_valid, plain_l1_pred,
        class_conditional=class_conditional,
    )

    daily_artifacts = DailyEnsembleArtifacts(
        shield_xr=shield_xr, plain_tweedie=plain, plain_l1=plain_l1,
        plain_pred_valid=plain_pred_valid, plain_pred=plain_pred,
        plain_l1_pred_valid=plain_l1_pred_valid, plain_l1_pred=plain_l1_pred,
        class_winner=class_winner, class_bias=class_bias,
        feature_cols=feature_cols, code2id=code2id, cat2id=cat2id,
    )
    del daily_artifacts  # not persisted here; kept for parity with production code path

    feat_full = add_features(cleaned)
    weekly_sku, weekly_features = build_weekly_panel(feat_full, d_train_end, d_valid_end)
    weekly_artifacts = train_weekly_breakdown(weekly_sku, weekly_features, code2id, cat2id)
    weekly_test = pd.DataFrame()
    if len(test):
        weekly_test = reconcile_weekly_breakdown(weekly_artifacts, test, ensemble_pred)

    training_metrics = compute_training_metrics(
        test, ensemble_pred, weekly_test,
        weekly_valid=weekly_artifacts.weekly_valid_panel,
        weekly_train=weekly_artifacts.weekly_train_panel,
    )

    mask_ok = ~test["is_anomaly_day"].values if len(test) else np.array([], dtype=bool)
    daily_acc = accuracy_from_wape(
        test.loc[mask_ok, "demand_raw"].values if mask_ok.any() else test["demand"].values,
        ensemble_pred[mask_ok] if mask_ok.any() else ensemble_pred,
    )

    elapsed = time.time() - t0
    print(
        f"[{label}] daily={daily_acc*100:.2f}% weekly_hosp={training_metrics.hospital_weekly_accuracy*100:.2f}% "
        f"hybrid_abc={training_metrics.hybrid_abc_combined_accuracy*100:.2f}% "
        f"(n_hybrid_drugs={training_metrics.n_hybrid_abc_named_drugs}) [{elapsed:.1f}s]",
        flush=True,
    )
    return {
        "variant": label,
        "daily_sku_day_accuracy_pct": daily_acc * 100,
        "hospital_weekly_accuracy_pct": training_metrics.hospital_weekly_accuracy * 100,
        "hybrid_abc_accuracy_pct": training_metrics.hybrid_abc_combined_accuracy * 100,
        "reconciled_weekly_per_drug_mean_accuracy_pct": training_metrics.reconciled_weekly_per_drug_mean_accuracy * 100,
        "n_hybrid_abc_named_drugs": training_metrics.n_hybrid_abc_named_drugs,
        "elapsed_s": elapsed,
    }


def run_seed_variance(n_seeds: int = 5, seeds: list[int] | None = None) -> None:
    """Task 3: rerun the baseline (all-features-on) variant across several LightGBM
    random seeds to report mean +/- std for the headline accuracy numbers. SHIELD-XR's
    internal LightGBM models use subsample=0.8/colsample_bytree=0.8, so RANDOM_STATE
    genuinely changes which rows/features each tree sees -- this is real stochasticity,
    not a cosmetic seed."""
    import app.forecasting.shield_xr.daily_model as daily_model_mod
    import app.forecasting.shield_xr.weekly_model as weekly_model_mod

    global RANDOM_STATE
    seeds = seeds or [42, 1, 7, 123, 2024][:n_seeds]

    print("Loading panel...", flush=True)
    panel = load_panel()
    cleaned, train, valid, test, d_train_end, d_valid_end, code2id, cat2id = prepare_splits(panel)
    from app.forecasting.shield_xr.features import FEATURES_X
    feature_cols = [c for c in FEATURES_X if c in train.columns]

    rows = []
    for seed in seeds:
        RANDOM_STATE = seed  # used by train_shieldxr_stack_ablatable in this module
        daily_model_mod.RANDOM_STATE = seed  # used by train_plain_models (imported from daily_model)
        weekly_model_mod.RANDOM_STATE = seed  # used by train_weekly_breakdown
        row = run_variant(
            f"seed={seed}", cleaned, train, valid, test, d_train_end, d_valid_end, code2id, cat2id, feature_cols,
        )
        row["seed"] = seed
        rows.append(row)
        pd.DataFrame(rows).to_csv("analysis/task3_shieldxr_seed_variance.csv", index=False)

    df = pd.DataFrame(rows)
    summary = df[["daily_sku_day_accuracy_pct", "hospital_weekly_accuracy_pct", "hybrid_abc_accuracy_pct"]].agg(["mean", "std"])
    print(summary, flush=True)
    summary.to_csv("analysis/task3_shieldxr_seed_variance_summary.csv")
    print("Done. Wrote analysis/task3_shieldxr_seed_variance.csv and _summary.csv", flush=True)


def main() -> None:
    import sys as _sys
    if "--seed-variance" in _sys.argv:
        run_seed_variance()
        return
    print("Loading panel...", flush=True)
    panel = load_panel()
    print(f"Panel: {len(panel)} rows, {panel['CODE'].nunique()} drugs, "
          f"{panel['DATE'].min().date()}..{panel['DATE'].max().date()}", flush=True)

    print("Preparing splits (anomaly guard, SB classification, features)...", flush=True)
    cleaned, train, valid, test, d_train_end, d_valid_end, code2id, cat2id = prepare_splits(panel)
    print(f"train={len(train)} valid={len(valid)} test={len(test)} "
          f"(train_end={d_train_end.date()}, valid_end={d_valid_end.date()})", flush=True)

    from app.forecasting.shield_xr.features import FEATURES_X
    feature_cols = [c for c in FEATURES_X if c in train.columns]

    variants = [
        ("baseline (all features on)", dict(use_evt=True, use_isotonic=True, use_sample_weights=True, class_conditional=True)),
        ("ablate: EVT/GPD spike correction OFF", dict(use_evt=False, use_isotonic=True, use_sample_weights=True, class_conditional=True)),
        ("ablate: isotonic calibration OFF", dict(use_evt=True, use_isotonic=False, use_sample_weights=True, class_conditional=True)),
        ("ablate: volume-aware sample weighting OFF", dict(use_evt=True, use_isotonic=True, use_sample_weights=False, class_conditional=True)),
        ("ablate: class-conditional selection OFF (single global model)", dict(use_evt=True, use_isotonic=True, use_sample_weights=True, class_conditional=False)),
    ]

    rows = []
    for label, kwargs in variants:
        row = run_variant(label, cleaned, train, valid, test, d_train_end, d_valid_end, code2id, cat2id, feature_cols, **kwargs)
        rows.append(row)
        pd.DataFrame(rows).to_csv("analysis/task2_shieldxr_ablation.csv", index=False)

    print("Done. Wrote analysis/task2_shieldxr_ablation.csv", flush=True)


if __name__ == "__main__":
    main()
