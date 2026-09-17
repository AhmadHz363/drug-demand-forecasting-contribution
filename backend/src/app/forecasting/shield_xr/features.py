"""Feature engineering and Syntetos–Boylan classification for SHIELD-XR."""

from __future__ import annotations

import numpy as np
import pandas as pd

SB_CLASS_MAP = {"smooth": 0, "intermittent": 1, "erratic": 2, "lumpy": 3}

BASE_FEATURES = [
    "dow",
    "month",
    "day",
    "weekofyear",
    "is_weekend",
    "is_saturday",
    "is_month_end",
    "year",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "roll_mean_7",
    "roll_std_7",
    "roll_max_7",
    "roll_nz_7",
    "roll_mean_14",
    "roll_std_14",
    "roll_max_14",
    "roll_nz_14",
    "roll_mean_28",
    "roll_std_28",
    "roll_max_28",
    "roll_nz_28",
    "days_since_nz",
    "z_vs_roll28",
    "sb_class_id",
]

EXOG_FEATURES = [
    "exog_n_unique_patients_lag1",
    "exog_n_admissions_lag1",
    "exog_n_unique_doctors_lag1",
    "exog_n_unique_CR_lag1",
    "exog_n_unique_CS_lag1",
    "exog_n_transactions_lag1",
    "exog_n_demand_txns_lag1",
    "exog_hospital_total_demand_lag1",
    "exog_top1_CR_share_lag1",
    "exog_top2_CR_share_lag1",
    "exog_top3_CR_share_lag1",
    "exog_bed_occupancy_rate_lag1",
    "exog_weekly_surgery_count_lag1",
]

ID_FEATURES = ["code_id", "cat_id"]
FEATURES_X = BASE_FEATURES + EXOG_FEATURES + ID_FEATURES
CAT_COLS = ["code_id", "cat_id", "dow", "month", "sb_class_id", "year"]

HOSP_COLS = [
    "n_unique_patients",
    "n_admissions",
    "n_unique_doctors",
    "n_unique_CR",
    "n_unique_CS",
    "n_transactions",
    "n_demand_txns",
    "top1_CR_share",
    "top2_CR_share",
    "top3_CR_share",
    "bed_occupancy_rate",
    "weekly_surgery_count",
]


def syntetos_boylan(series: pd.Series) -> str:
    values = series.astype(float).values
    n = len(values)
    nzc = int((values > 0).sum())
    adi = n / max(nzc, 1)
    cv2 = (values.std() / (values.mean() + 1e-9)) ** 2
    if adi <= 1.32 and cv2 <= 0.49:
        return "smooth"
    if adi > 1.32 and cv2 <= 0.49:
        return "intermittent"
    if adi <= 1.32 and cv2 > 0.49:
        return "erratic"
    return "lumpy"


def assign_sb_classes(df: pd.DataFrame, d_train_end: pd.Timestamp) -> pd.DataFrame:
    sb = df[df["DATE"] <= d_train_end].groupby("CODE")["demand"].apply(syntetos_boylan)
    out = df.merge(sb.rename("sb_class"), left_on="CODE", right_index=True, how="left")
    out["sb_class"] = out["sb_class"].fillna("lumpy")
    return out


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy().sort_values(["CODE", "DATE"])
    out["dow"] = out["DATE"].dt.dayofweek
    out["month"] = out["DATE"].dt.month
    out["day"] = out["DATE"].dt.day
    out["weekofyear"] = out["DATE"].dt.isocalendar().week.astype(int)
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["is_saturday"] = (out["dow"] == 5).astype(int)
    out["is_month_end"] = out["DATE"].dt.is_month_end.astype(int)
    out["year"] = out["DATE"].dt.year

    grouped = out.groupby("CODE", group_keys=False)
    for lag in (1, 7, 14, 28):
        out[f"lag_{lag}"] = grouped["demand"].shift(lag)

    shifted = grouped["demand"].shift(1)
    for window in (7, 14, 28):
        out[f"roll_mean_{window}"] = shifted.groupby(out["CODE"]).transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        out[f"roll_std_{window}"] = shifted.groupby(out["CODE"]).transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )
        out[f"roll_max_{window}"] = shifted.groupby(out["CODE"]).transform(
            lambda x: x.rolling(window, min_periods=1).max()
        )
        out[f"roll_nz_{window}"] = shifted.groupby(out["CODE"]).transform(
            lambda x: x.rolling(window, min_periods=1).apply(lambda z: (z > 0).mean(), raw=True)
        )

    def _days_since_nonzero(values: pd.Series) -> pd.Series:
        last = -999
        result: list[float] = []
        for idx, value in enumerate(values):
            result.append(idx - last if last >= 0 else np.nan)
            if value > 0:
                last = idx
        return pd.Series(result, index=values.index)

    out["_shift"] = shifted
    out["days_since_nz"] = out.groupby("CODE")["_shift"].transform(_days_since_nonzero)
    out["z_vs_roll28"] = (out["lag_1"] - out["roll_mean_28"]) / (out["roll_std_28"] + 1e-6)

    hosp = out.groupby("DATE", as_index=False)[HOSP_COLS].mean(numeric_only=True).sort_values("DATE")
    hosp_tot = out.groupby("DATE", as_index=False)["demand"].sum().rename(columns={"demand": "hospital_total_demand"})
    hosp = hosp.merge(hosp_tot, on="DATE", how="left")
    for col in HOSP_COLS + ["hospital_total_demand"]:
        hosp[f"exog_{col}_lag1"] = hosp[col].shift(1)
    lag_cols = ["DATE"] + [f"exog_{col}_lag1" for col in HOSP_COLS + ["hospital_total_demand"]]
    out = out.merge(hosp[lag_cols], on="DATE", how="left")

    out["CAT"] = out["CAT"].astype(str).fillna("UNK")
    out["sb_class_id"] = out["sb_class"].map(SB_CLASS_MAP).fillna(3).astype(int)
    out["occur"] = (out["demand"] > 0).astype(int)
    return out.drop(columns=["_shift"], errors="ignore")


def apply_spike_labels(
    frame: pd.DataFrame,
    train_stats: pd.DataFrame,
    *,
    z: float = 2.5,
) -> pd.DataFrame:
    merged = frame.merge(train_stats, left_on="CODE", right_index=True, how="left")
    threshold = merged["mean"] + z * merged["std"].fillna(0)
    merged["spike"] = (merged["demand"] > threshold).astype(int)
    return merged.drop(columns=["mean", "std"])


def encode_ids(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    all_codes: list[str],
    all_cats: list[str],
) -> tuple[dict[str, int], dict[str, int]]:
    code2id = {code: idx for idx, code in enumerate(all_codes)}
    cat2id = {cat: idx for idx, cat in enumerate(all_cats)}
    for part in (train, valid, test):
        part["code_id"] = part["CODE"].astype(str).map(code2id).astype(int)
        part["cat_id"] = part["CAT"].astype(str).map(cat2id).fillna(cat2id.get("UNK", 0)).astype(int)
        for col in EXOG_FEATURES:
            if col not in part.columns:
                part[col] = np.nan
            median = part[col].median() if part[col].notna().any() else 0.0
            part[col] = part[col].fillna(median)
    return code2id, cat2id


def add_sample_weights(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame) -> None:
    sku_train_vol = train.groupby("CODE")["demand"].sum()
    sku_weight_map = (np.log1p(sku_train_vol) + 1.0).to_dict()
    default_w = float(np.mean(list(sku_weight_map.values()))) if sku_weight_map else 1.0
    for part in (train, valid, test):
        part["_w"] = part["CODE"].map(sku_weight_map).fillna(default_w)


def coerce_feature_dtypes(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Ensure model feature columns are numeric for LightGBM inference/training."""
    out = frame.copy()
    for col in feature_cols:
        if col not in out.columns:
            out[col] = 0.0
        if col in CAT_COLS:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(np.int32)
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).astype(np.float64)
    return out


def feature_row_to_frame(row: pd.Series, feature_cols: list[str]) -> pd.DataFrame:
    """Build a single-row feature matrix with strict numeric dtypes."""
    values: dict[str, float | int] = {}
    for col in feature_cols:
        raw = row.get(col, 0)
        if col in CAT_COLS:
            values[col] = int(pd.to_numeric(raw, errors="coerce") or 0)
        else:
            values[col] = float(pd.to_numeric(raw, errors="coerce") or 0.0)
    return pd.DataFrame([values], columns=feature_cols)
