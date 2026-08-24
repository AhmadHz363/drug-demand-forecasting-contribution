"""Weekly per-drug breakdown model with hierarchical reconciliation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from app.forecasting.shield_xr.features import HOSP_COLS
from app.forecasting.shield_xr.metrics import wape

RANDOM_STATE = 42
CAT_W = ["code_id", "cat_id", "month", "sb_class_id"]


@dataclass
class WeeklyBreakdownArtifacts:
    weekly_tweedie: Any
    weekly_l1: Any
    weekly_class_winner: dict[str, str] = field(default_factory=dict)
    weekly_class_bias: dict[str, float] = field(default_factory=dict)
    weekly_class_alpha: dict[str, float] = field(default_factory=dict)
    weekly_features: list[str] = field(default_factory=list)
    hospital_week_totals: pd.DataFrame | None = None
    weekly_valid_panel: pd.DataFrame | None = None
    weekly_train_panel: pd.DataFrame | None = None


def build_weekly_panel(feat: pd.DataFrame, d_train_end: pd.Timestamp, d_valid_end: pd.Timestamp) -> pd.DataFrame:
    hosp_cols_raw = [c for c in HOSP_COLS if c not in {"bed_occupancy_rate", "weekly_surgery_count"}]
    feat_w = feat[
        ["DATE", "CODE", "ARTICLE", "CAT", "sb_class", "demand", "demand_raw", "is_anomaly_day"] + hosp_cols_raw
    ].copy()
    feat_w["week"] = feat_w["DATE"].dt.to_period("W")
    weekly_sku = feat_w.groupby(["CODE", "week"], as_index=False).agg(
        y_clean=("demand", "sum"),
        y_raw=("demand_raw", "sum"),
        ARTICLE=("ARTICLE", "first"),
        CAT=("CAT", "first"),
        sb_class=("sb_class", "first"),
        n_anomaly=("is_anomaly_day", "sum"),
        n_days=("DATE", "size"),
    )
    weekly_sku = weekly_sku[weekly_sku["n_days"] == 7].copy()
    weekly_sku["week_start"] = weekly_sku["week"].dt.start_time

    hosp_daily = feat_w.groupby("DATE", as_index=False)[hosp_cols_raw].first()
    hosp_daily["week"] = hosp_daily["DATE"].dt.to_period("W")
    hosp_weekly = hosp_daily.groupby("week", as_index=False)[hosp_cols_raw].sum().sort_values("week")
    hosp_lag_cols = []
    for col in hosp_cols_raw:
        lag_col = f"wexog_{col}_lag1"
        hosp_weekly[lag_col] = hosp_weekly[col].shift(1)
        hosp_lag_cols.append(lag_col)
    weekly_sku = weekly_sku.merge(hosp_weekly[["week"] + hosp_lag_cols], on="week", how="left")

    weekly_sku = weekly_sku.sort_values(["CODE", "week"]).reset_index(drop=True)
    grouped = weekly_sku.groupby("CODE", group_keys=False)
    for lag in (1, 2, 4, 8):
        weekly_sku[f"wlag_{lag}"] = grouped["y_clean"].shift(lag)
    shifted = grouped["y_clean"].shift(1)
    for window in (4, 8, 12):
        weekly_sku[f"wroll_mean_{window}"] = shifted.groupby(weekly_sku["CODE"]).transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        weekly_sku[f"wroll_std_{window}"] = shifted.groupby(weekly_sku["CODE"]).transform(
            lambda x: x.rolling(window, min_periods=1).std()
        ).fillna(0)
        weekly_sku[f"wroll_max_{window}"] = shifted.groupby(weekly_sku["CODE"]).transform(
            lambda x: x.rolling(window, min_periods=1).max()
        )

    weekly_sku["month"] = weekly_sku["week_start"].dt.month
    weekly_sku["week_of_year"] = weekly_sku["week_start"].dt.isocalendar().week.astype(int)
    weekly_sku["sb_class_id"] = weekly_sku["sb_class"].map(
        {"smooth": 0, "intermittent": 1, "erratic": 2, "lumpy": 3}
    ).fillna(3).astype(int)

    weekly_features = (
        ["month", "week_of_year", "sb_class_id", "code_id", "cat_id",
         "wlag_1", "wlag_2", "wlag_4", "wlag_8",
         "wroll_mean_4", "wroll_std_4", "wroll_max_4",
         "wroll_mean_8", "wroll_std_8", "wroll_max_8",
         "wroll_mean_12", "wroll_std_12", "wroll_max_12"]
        + hosp_lag_cols
    )
    week_end = weekly_sku["week_start"] + pd.Timedelta(days=6)
    weekly_sku["split"] = "other"
    weekly_sku.loc[week_end <= d_train_end, "split"] = "train"
    weekly_sku.loc[(weekly_sku["week_start"] > d_train_end) & (week_end <= d_valid_end), "split"] = "valid"
    weekly_sku.loc[weekly_sku["week_start"] > d_valid_end, "split"] = "test"
    return weekly_sku, weekly_features


def train_weekly_breakdown(
    weekly_sku: pd.DataFrame,
    weekly_features: list[str],
    code2id: dict[str, int],
    cat2id: dict[str, int],
) -> WeeklyBreakdownArtifacts:
    weekly_sku = weekly_sku.copy()
    weekly_sku["code_id"] = weekly_sku["CODE"].astype(str).map(code2id).fillna(0).astype(int)
    weekly_sku["cat_id"] = weekly_sku["CAT"].astype(str).map(cat2id).fillna(cat2id.get("UNK", 0)).astype(int)

    weekly_train = weekly_sku[weekly_sku["split"] == "train"].dropna(subset=["wlag_1"]).copy()
    weekly_valid = weekly_sku[weekly_sku["split"] == "valid"].copy()
    weekly_test = weekly_sku[weekly_sku["split"] == "test"].copy()

    for col in weekly_features:
        median = weekly_train[col].median() if weekly_train[col].notna().any() else 0.0
        for part in (weekly_train, weekly_valid, weekly_test):
            part[col] = part[col].fillna(median)

    wsku_train_vol = weekly_train.groupby("CODE")["y_clean"].sum()
    wsku_weight_map = (np.log1p(wsku_train_vol) + 1.0).to_dict()
    wdefault = float(np.mean(list(wsku_weight_map.values()))) if wsku_weight_map else 1.0
    for part in (weekly_train, weekly_valid, weekly_test):
        part["_ww"] = part["CODE"].map(wsku_weight_map).fillna(wdefault)

    def _wxy(frame: pd.DataFrame, ycol: str) -> tuple[pd.DataFrame, np.ndarray]:
        return frame[weekly_features].copy(), frame[ycol].astype(float).values

    weekly_tw = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="tweedie",
        tweedie_variance_power=1.2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    weekly_tw.fit(
        *_wxy(weekly_train, "y_clean"),
        sample_weight=weekly_train["_ww"],
        eval_set=[_wxy(weekly_valid, "y_clean")],
        eval_sample_weight=[weekly_valid["_ww"]],
        categorical_feature=[c for c in CAT_W if c in weekly_features],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    weekly_l1 = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="regression_l1",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    weekly_l1.fit(
        *_wxy(weekly_train, "y_clean"),
        sample_weight=weekly_train["_ww"],
        eval_set=[_wxy(weekly_valid, "y_clean")],
        eval_sample_weight=[weekly_valid["_ww"]],
        categorical_feature=[c for c in CAT_W if c in weekly_features],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )

    tw_va = np.clip(weekly_tw.predict(_wxy(weekly_valid, "y_clean")[0]), 0, None)
    tw_te = np.clip(weekly_tw.predict(_wxy(weekly_test, "y_clean")[0]), 0, None)
    l1_va = np.clip(weekly_l1.predict(_wxy(weekly_valid, "y_clean")[0]), 0, None)
    l1_te = np.clip(weekly_l1.predict(_wxy(weekly_test, "y_clean")[0]), 0, None)

    candidates = {"WeeklyTweedie": (tw_va, tw_te), "WeeklyL1": (l1_va, l1_te)}
    class_winner: dict[str, str] = {}
    class_bias: dict[str, float] = {}
    selected_va = np.zeros(len(weekly_valid))
    for cls in weekly_valid["sb_class"].unique():
        mask = (weekly_valid["sb_class"] == cls).values
        if mask.sum() < 10:
            continue
        y_va = weekly_valid.loc[mask, "y_clean"].values
        scores = {name: wape(y_va, pred[mask]) for name, (pred, _) in candidates.items()}
        winner = min(scores, key=scores.get)
        class_winner[str(cls)] = winner
        va_pred = candidates[winner][0][mask]
        total = va_pred.sum()
        bias = float(np.clip(y_va.sum() / total, 0.7, 1.4)) if total > 0 else 1.0
        class_bias[str(cls)] = bias
        selected_va[mask] = va_pred * bias

    direct_raw = np.zeros(len(weekly_test))
    for cls, winner in class_winner.items():
        mask = (weekly_test["sb_class"] == cls).values
        direct_raw[mask] = candidates[winner][1][mask] * class_bias[cls]

    class_alpha: dict[str, float] = {}
    alphas = np.linspace(0.0, 1.0, 11)
    for cls in weekly_valid["sb_class"].unique():
        mask = (weekly_valid["sb_class"] == cls).values
        if mask.sum() < 10:
            class_alpha[str(cls)] = 1.0
            continue
        y_va = weekly_valid.loc[mask, "y_clean"].values
        hist_va = weekly_valid.loc[mask, "wroll_mean_8"].values
        best_a, best_w = 1.0, wape(y_va, selected_va[mask])
        for alpha in alphas:
            blend = alpha * selected_va[mask] + (1 - alpha) * hist_va
            score = wape(y_va, blend)
            if score < best_w:
                best_a, best_w = float(alpha), score
        class_alpha[str(cls)] = best_a

    direct = np.zeros(len(weekly_test))
    for cls in class_winner:
        mask = (weekly_test["sb_class"] == cls).values
        alpha = class_alpha.get(cls, 1.0)
        hist = weekly_test.loc[mask, "wroll_mean_8"].values
        direct[mask] = alpha * direct_raw[mask] + (1 - alpha) * hist

    weekly_valid = weekly_valid.copy()
    weekly_valid["yhat_weekly_direct"] = np.clip(selected_va, 0, None)
    weekly_valid["week_str"] = weekly_valid["week"].astype(str)

    weekly_test = weekly_test.copy()
    weekly_test["yhat_weekly_direct"] = np.clip(direct, 0, None)
    weekly_test["week_str"] = weekly_test["week"].astype(str)

    return WeeklyBreakdownArtifacts(
        weekly_tweedie=weekly_tw,
        weekly_l1=weekly_l1,
        weekly_class_winner=class_winner,
        weekly_class_bias=class_bias,
        weekly_class_alpha=class_alpha,
        weekly_features=weekly_features,
        hospital_week_totals=weekly_test,
        weekly_valid_panel=weekly_valid,
        weekly_train_panel=weekly_train,
    )


def reconcile_weekly_breakdown(
    weekly_artifacts: WeeklyBreakdownArtifacts,
    test_ok: pd.DataFrame,
    ensemble_pred: np.ndarray,
) -> pd.DataFrame:
    """Reconcile per-SKU weekly predictions to trusted bottom-up hospital totals."""
    test_ok = test_ok.copy()
    test_ok["ensemble_pred"] = ensemble_pred
    test_ok["week"] = test_ok["DATE"].dt.to_period("W").astype(str)
    weekly_hosp = test_ok.groupby("week").agg(y=("demand_raw", "sum"), ens=("ensemble_pred", "sum"))

    weekly_test = weekly_artifacts.hospital_week_totals
    if weekly_test is None or weekly_test.empty:
        return pd.DataFrame()

    trusted = weekly_hosp[["ens"]].rename(columns={"ens": "trusted_total"}).reset_index()
    own_totals = weekly_test.groupby("week_str", as_index=False)["yhat_weekly_direct"].sum().rename(
        columns={"week_str": "week", "yhat_weekly_direct": "own_total"}
    )
    recon = trusted.merge(own_totals, on="week", how="inner")
    recon["recon_factor"] = np.where(recon["own_total"] > 0, recon["trusted_total"] / recon["own_total"], 1.0)
    recon_map = dict(zip(recon["week"], recon["recon_factor"]))
    weekly_test["recon_factor"] = weekly_test["week_str"].map(recon_map).fillna(1.0)
    weekly_test["yhat_breakdown"] = np.clip(
        weekly_test["yhat_weekly_direct"] * weekly_test["recon_factor"], 0, None
    )
    weekly_artifacts.hospital_week_totals = weekly_test
    return weekly_test
