"""AnomalyGuard — non-clinical data artifact detection and cleaning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

K_MAD = 8.0
MIN_NONZERO_FOR_CEILING = 5
EXCEED_Z_THR = 4.0
ACTIVITY_Z_THR = 2.0
DROPOUT_Z_THR = 2.0


@dataclass
class AnomalyGuardResult:
    frame: pd.DataFrame
    day_stats: pd.DataFrame
    ceiling_map: dict[str, float]
    global_p995: float


def compute_sku_ceilings(train_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float], float]:
    nz = train_raw[train_raw["demand"] > 0]
    sku_stats = nz.groupby("CODE")["demand"].agg(
        med="median",
        mad=lambda s: (s - s.median()).abs().median(),
        n="count",
        p995=lambda s: s.quantile(0.995),
    )
    global_p995 = float(nz["demand"].quantile(0.995)) if len(nz) else 100.0

    def _ceiling(row: pd.Series) -> float:
        if row["n"] >= MIN_NONZERO_FOR_CEILING and row["mad"] > 0:
            return float(max(row["med"] + K_MAD * row["mad"], row["p995"]))
        return float(max(row.get("p995", global_p995), global_p995))

    sku_stats["ceiling"] = sku_stats.apply(_ceiling, axis=1)
    return sku_stats, sku_stats["ceiling"].to_dict(), global_p995


def flag_anomaly_days(df: pd.DataFrame, d_train_end: pd.Timestamp) -> pd.DataFrame:
    day_stats = df.groupby("DATE").agg(
        n_active=("demand", lambda s: s.notna().sum()),
        n_nonzero=("demand", lambda s: (s > 0).sum()),
        n_exceed=("exceeds_ceiling", "sum"),
        total_demand=("demand", "sum"),
        n_patients=("n_unique_patients", "first"),
        n_admissions=("n_admissions", "first"),
    ).reset_index()
    day_stats["exceed_frac"] = day_stats["n_exceed"] / day_stats["n_active"].clip(lower=1)
    day_stats["nz_frac"] = day_stats["n_nonzero"] / day_stats["n_active"].clip(lower=1)
    day_stats["nz_frac_smooth"] = day_stats["nz_frac"].rolling(5, center=True, min_periods=1).mean()

    train_day_stats = day_stats[day_stats["DATE"] <= d_train_end]
    frac_mu, frac_sd = train_day_stats["exceed_frac"].mean(), train_day_stats["exceed_frac"].std()
    nz_mu, nz_sd = train_day_stats["nz_frac_smooth"].mean(), train_day_stats["nz_frac_smooth"].std()
    pat_mu, pat_sd = train_day_stats["n_patients"].mean(), train_day_stats["n_patients"].std()
    adm_mu, adm_sd = train_day_stats["n_admissions"].mean(), train_day_stats["n_admissions"].std()

    day_stats["exceed_z"] = (day_stats["exceed_frac"] - frac_mu) / (frac_sd + 1e-9)
    day_stats["nz_z"] = (day_stats["nz_frac_smooth"] - nz_mu) / (nz_sd + 1e-9)
    day_stats["patients_z"] = (day_stats["n_patients"] - pat_mu) / (pat_sd + 1e-9)
    day_stats["admissions_z"] = (day_stats["n_admissions"] - adm_mu) / (adm_sd + 1e-9)

    day_stats["is_spike_anomaly"] = (
        (day_stats["exceed_z"] > EXCEED_Z_THR)
        & (day_stats["patients_z"].abs() < ACTIVITY_Z_THR)
        & (day_stats["admissions_z"].abs() < ACTIVITY_Z_THR)
    )
    day_stats["is_dropout_anomaly"] = day_stats["nz_z"] < -DROPOUT_Z_THR
    day_stats["is_anomaly_day"] = day_stats["is_spike_anomaly"] | day_stats["is_dropout_anomaly"]
    return day_stats


def apply_anomaly_guard(
    df: pd.DataFrame,
    d_train_end: pd.Timestamp,
) -> AnomalyGuardResult:
    """Detect and clean spike/dropout artifacts using train-only statistics."""
    sku_stats, ceiling_map, global_p995 = compute_sku_ceilings(df[df["DATE"] <= d_train_end])
    out = df.copy()
    out["ceiling"] = out["CODE"].map(ceiling_map).fillna(global_p995)
    out["exceeds_ceiling"] = out["demand"] > out["ceiling"]

    day_stats = flag_anomaly_days(out, d_train_end)
    anomaly_days = set(pd.to_datetime(day_stats.loc[day_stats["is_anomaly_day"], "DATE"]))
    dropout_days = set(pd.to_datetime(day_stats.loc[day_stats["is_dropout_anomaly"], "DATE"]))
    out["is_anomaly_day"] = out["DATE"].isin(anomaly_days)
    out["is_dropout_day"] = out["DATE"].isin(dropout_days)

    out["demand_raw"] = out["demand"].astype(float)
    dow_seasonal = (
        out[out["DATE"] <= d_train_end]
        .assign(dow=lambda x: x["DATE"].dt.dayofweek)
        .groupby(["CODE", "dow"])["demand"]
        .mean()
        .rename("dow_seasonal_mean")
        .reset_index()
    )
    out["dow"] = out["DATE"].dt.dayofweek
    out = out.merge(dow_seasonal, on=["CODE", "dow"], how="left")
    out["dow_seasonal_mean"] = out["dow_seasonal_mean"].fillna(0)
    out = out.drop(columns=["dow"])

    out["demand_clean"] = out["demand_raw"]
    spike_mask = out["is_anomaly_day"] & out["exceeds_ceiling"] & ~out["is_dropout_day"]
    out.loc[spike_mask, "demand_clean"] = out.loc[spike_mask, "ceiling"]
    out.loc[out["is_dropout_day"], "demand_clean"] = out.loc[out["is_dropout_day"], "dow_seasonal_mean"]
    out["demand"] = out["demand_clean"]
    return AnomalyGuardResult(
        frame=out,
        day_stats=day_stats,
        ceiling_map=ceiling_map,
        global_p995=global_p995,
    )
