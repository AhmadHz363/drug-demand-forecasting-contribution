"""Leave-drugs-out cold-start validation (notebook Section 13)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler

from app.cold_start.cameo.features import CameoFeatureEncoder
from app.cold_start.cameo.metric_net import (
    MetricNet,
    analog_launch_expected,
    cameo_init_forecast,
    demand_shape_features,
    train_metric_net,
)
from app.cold_start.cameo.panel import MIN_WEEKS, sb_class
from app.cold_start.cameo.pipeline import conformal_intervals, run_cameo
from app.cold_start.constants import CAMEO_EPOCHS, CAMEO_SEED, CAMEO_TOPK
from app.cold_start.schemas import DrugMetadataInput

CAMEO_METHOD = "CAMEO"
ANALOGOUS_METHOD = "Analogous"
VALIDATION_TRAIN_FRAC = 0.75
VALIDATION_T_WEEKS = 20
VALIDATION_SEED = 1
MIN_OPERATIONAL_WEEKLY_UNITS = 5.0


def wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    denom = np.abs(actual).sum()
    if denom == 0:
        return float("nan")
    return float(np.abs(actual - forecast).sum() / denom)


def accuracy_from_wape(value: float) -> float | None:
    if value != value:  # NaN
        return None
    return max(0.0, 1.0 - value)


def baseline_analogous(
    new_attr_raw: np.ndarray,
    hist_attrs_raw: np.ndarray,
    hist_series: list[np.ndarray],
    horizon_weeks: int,
    *,
    topk: int = CAMEO_TOPK,
) -> np.ndarray:
    distances = np.linalg.norm(hist_attrs_raw - new_attr_raw, axis=1)
    order = np.argsort(distances)[:topk]
    weights = 1.0 / (distances[order] + 1e-3)
    weights = weights / weights.sum()

    profiles: list[np.ndarray] = []
    for idx in order:
        series = np.asarray(hist_series[idx][:horizon_weeks], dtype=float)
        if len(series) < horizon_weeks:
            pad_val = float(series[series > 0].mean()) if len(series) and np.any(series > 0) else 0.0
            series = np.pad(series, (0, horizon_weeks - len(series)), constant_values=pad_val)
        profiles.append(series)
    matrix = np.stack(profiles, axis=0)
    return np.maximum((weights[:, None] * matrix).sum(axis=0), 0.0)


@dataclass
class _DrugRecord:
    code: str
    metadata: DrugMetadataInput
    series: np.ndarray
    sb: str


def evaluate_cameo_holdout(
    metadata_list: list[DrugMetadataInput],
    weekly_by_code: dict[str, np.ndarray],
    *,
    train_frac: float = VALIDATION_TRAIN_FRAC,
    horizon_weeks: int = VALIDATION_T_WEEKS,
    seed: int = VALIDATION_SEED,
) -> dict:
    records: list[_DrugRecord] = []
    for meta in metadata_list:
        series = weekly_by_code.get(meta.drug_code)
        if series is None or len(series) < MIN_WEEKS:
            continue
        records.append(
            _DrugRecord(
                code=meta.drug_code,
                metadata=meta,
                series=series,
                sb=sb_class(series),
            ),
        )

    if len(records) < 8:
        return {
            "n_historical_library_drugs": len(records),
            "n_coldstart_test_drugs": 0,
            "forecast_horizon_weeks": horizon_weeks,
            "note": "Need at least 8 library drugs with weekly history for hold-out validation.",
        }

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(records))
    n_train = max(5, int(round(train_frac * len(records))))
    if n_train >= len(records):
        n_train = len(records) - 1
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    train_records = [records[i] for i in train_idx]
    test_records = [records[i] for i in test_idx]

    encoder = CameoFeatureEncoder().fit([r.metadata for r in train_records])
    attrs_raw = encoder.transform([r.metadata for r in train_records])
    hist_series = [r.series for r in train_records]

    attr_scaler = StandardScaler().fit(attrs_raw)
    attrs_scaled = attr_scaler.transform(attrs_raw)
    shape_targets = np.array([demand_shape_features(series) for series in hist_series])
    shape_scaler = StandardScaler().fit(shape_targets)
    shape_scaled = shape_scaler.transform(shape_targets)

    net = train_metric_net(attrs_scaled, shape_scaled, epochs=CAMEO_EPOCHS, seed=seed)
    hist_embeds = net.embed(attrs_scaled)

    per_drug_rows: list[dict] = []
    weekly_rows: list[dict] = []

    for record in test_records:
        new_attr_raw = encoder.transform([record.metadata])[0]
        new_attr_scaled = attr_scaler.transform(new_attr_raw.reshape(1, -1))[0]
        horizon = min(horizon_weeks, len(record.series))
        actual = record.series[:horizon]

        analogous = baseline_analogous(
            new_attr_raw,
            attrs_raw,
            hist_series,
            horizon,
            topk=CAMEO_TOPK,
        )
        cameo_forecast, _ = run_cameo(
            new_attr_scaled,
            hist_embeds,
            hist_series,
            net,
            actual,
            topk=CAMEO_TOPK,
        )

        for method, forecast in (
            (ANALOGOUS_METHOD, analogous),
            (CAMEO_METHOD, cameo_forecast),
        ):
            drug_wape = wape(actual, forecast)
            per_drug_rows.append(
                {
                    "drug_code": record.code,
                    "sb_class": record.sb,
                    "method": method,
                    "wape": round(drug_wape, 4) if drug_wape == drug_wape else None,
                    "accuracy": round(accuracy_from_wape(drug_wape), 4)
                    if drug_wape == drug_wape
                    else None,
                },
            )
            for week_idx in range(horizon):
                weekly_rows.append(
                    {
                        "drug_code": record.code,
                        "sb_class": record.sb,
                        "method": method,
                        "week": week_idx,
                        "actual": float(actual[week_idx]),
                        "forecast": float(forecast[week_idx]),
                    },
                )

    methods = [ANALOGOUS_METHOD, CAMEO_METHOD]
    pooled_wape: dict[str, float] = {}
    pooled_accuracy: dict[str, float] = {}
    for method in methods:
        method_weeks = [row for row in weekly_rows if row["method"] == method]
        actuals = np.array([row["actual"] for row in method_weeks], dtype=float)
        forecasts = np.array([row["forecast"] for row in method_weeks], dtype=float)
        pooled = wape(actuals, forecasts)
        pooled_wape[method] = round(pooled, 4) if pooled == pooled else float("nan")
        acc = accuracy_from_wape(pooled)
        if acc is not None:
            pooled_accuracy[method] = round(acc, 4)

    median_accuracy: dict[str, float | None] = {}
    for method in methods:
        values = [
            row["accuracy"]
            for row in per_drug_rows
            if row["method"] == method and row["accuracy"] is not None
        ]
        median_accuracy[method] = round(float(np.median(values)), 4) if values else None

    pivot_mae: dict[str, dict[str, float]] = {}
    for record in test_records:
        code = record.code
        pivot_mae[code] = {}
        for method in methods:
            wape_val = next(
                row["wape"]
                for row in per_drug_rows
                if row["drug_code"] == code and row["method"] == method
            )
            if wape_val is not None:
                pivot_mae[code][method] = float(wape_val)

    winners = [
        min(row.items(), key=lambda item: item[1])[0]
        for row in pivot_mae.values()
        if len(row) == len(methods)
    ]
    win_counts = {method: winners.count(method) for method in methods}
    n_test = len(test_records)

    per_sb_pooled: dict[str, float] = {}
    for sb in sorted({row["sb_class"] for row in weekly_rows}):
        cameo_weeks = [
            row for row in weekly_rows if row["sb_class"] == sb and row["method"] == CAMEO_METHOD
        ]
        if not cameo_weeks:
            continue
        actuals = np.array([row["actual"] for row in cameo_weeks], dtype=float)
        forecasts = np.array([row["forecast"] for row in cameo_weeks], dtype=float)
        pooled = wape(actuals, forecasts)
        acc = accuracy_from_wape(pooled)
        if acc is not None:
            per_sb_pooled[sb] = round(acc * 100, 2)

    calib_residuals: list[float] = []
    sample_size = min(25, len(train_records))
    sample_idx = rng.choice(len(train_records), size=sample_size, replace=False)
    calib_idx = sample_idx[: len(sample_idx) // 2]
    test_cov_idx = sample_idx[len(sample_idx) // 2 :]

    for i in calib_idx:
        init = cameo_init_forecast(
            attrs_scaled[i],
            np.delete(hist_embeds, i, axis=0),
            [series for j, series in enumerate(hist_series) if j != i],
            net,
            topk=CAMEO_TOPK,
        )
        calib_residuals.append(float(hist_series[i][:8].mean() - init.init_level))

    conformal_q = conformal_intervals(np.asarray(calib_residuals), alpha=0.1)
    covered = 0
    for i in test_cov_idx:
        init = cameo_init_forecast(
            attrs_scaled[i],
            np.delete(hist_embeds, i, axis=0),
            [series for j, series in enumerate(hist_series) if j != i],
            net,
            topk=CAMEO_TOPK,
        )
        actual_level = hist_series[i][:8].mean()
        if abs(actual_level - init.init_level) <= conformal_q:
            covered += 1
    empirical_coverage = round(covered / max(len(test_cov_idx), 1), 3)

    cameo_pooled_acc = pooled_accuracy.get(CAMEO_METHOD)
    analogous_pooled_acc = pooled_accuracy.get(ANALOGOUS_METHOD)
    wape_improvement_pct = None
    if (
        cameo_pooled_acc is not None
        and analogous_pooled_acc is not None
        and analogous_pooled_acc < 1
    ):
        analogous_wape = 1 - analogous_pooled_acc
        cameo_wape = 1 - cameo_pooled_acc
        wape_improvement_pct = round(
            100 * (analogous_wape - cameo_wape) / max(analogous_wape, 1e-9),
            1,
        )

    per_drug_cameo = [
        {
            "drug_code": row["drug_code"],
            "sb_class": row["sb_class"],
            "accuracy_pct": round(row["accuracy"] * 100, 2) if row["accuracy"] is not None else None,
            "wape": row["wape"],
        }
        for row in per_drug_rows
        if row["method"] == CAMEO_METHOD
    ]

    smooth_weeks = [
        row
        for row in weekly_rows
        if row["sb_class"] == "smooth" and row["method"] == CAMEO_METHOD
    ]

    smooth_pooled_acc: float | None = None
    operational_pooled_acc: float | None = None
    non_lumpy_operational_pooled_acc: float | None = None
    if smooth_weeks:
        actuals = np.array([row["actual"] for row in smooth_weeks], dtype=float)
        forecasts = np.array([row["forecast"] for row in smooth_weeks], dtype=float)
        if actuals.sum() > 0:
            smooth_acc = accuracy_from_wape(wape(actuals, forecasts))
            if smooth_acc is not None:
                smooth_pooled_acc = round(smooth_acc * 100, 2)

    drug_actual_totals: dict[str, float] = {}
    for record in test_records:
        horizon = min(horizon_weeks, len(record.series))
        drug_actual_totals[record.code] = float(record.series[:horizon].sum())

    operational_weeks = [
        row
        for row in weekly_rows
        if row["method"] == CAMEO_METHOD
        and drug_actual_totals.get(row["drug_code"], 0.0) >= MIN_OPERATIONAL_WEEKLY_UNITS
    ]
    if operational_weeks:
        actuals = np.array([row["actual"] for row in operational_weeks], dtype=float)
        forecasts = np.array([row["forecast"] for row in operational_weeks], dtype=float)
        if actuals.sum() > 0:
            op_acc = accuracy_from_wape(wape(actuals, forecasts))
            if op_acc is not None:
                operational_pooled_acc = round(op_acc * 100, 2)

    non_lumpy_weeks = [
        row
        for row in weekly_rows
        if row["method"] == CAMEO_METHOD
        and row["sb_class"] in ("smooth", "erratic")
        and drug_actual_totals.get(row["drug_code"], 0.0) >= MIN_OPERATIONAL_WEEKLY_UNITS
    ]
    if non_lumpy_weeks:
        actuals = np.array([row["actual"] for row in non_lumpy_weeks], dtype=float)
        forecasts = np.array([row["forecast"] for row in non_lumpy_weeks], dtype=float)
        if actuals.sum() > 0:
            nl_acc = accuracy_from_wape(wape(actuals, forecasts))
            if nl_acc is not None:
                non_lumpy_operational_pooled_acc = round(nl_acc * 100, 2)

    zero_demand_test_drugs = sum(
        1
        for record in test_records
        if float(record.series[: min(horizon_weeks, len(record.series))].sum()) == 0.0
    )

    return {
        "n_historical_library_drugs": len(records),
        "n_coldstart_test_drugs": n_test,
        "forecast_horizon_weeks": horizon_weeks,
        "pooled_accuracy_pct": {
            method: round(value * 100, 2) for method, value in pooled_accuracy.items()
        },
        "median_accuracy_pct": {
            method: round(value * 100, 2) if value is not None else None
            for method, value in median_accuracy.items()
        },
        "pooled_wape": pooled_wape,
        "win_rate_pct": {
            method: round(100 * win_counts[method] / n_test, 1) if n_test else 0.0
            for method in methods
        },
        "win_counts": win_counts,
        "per_sb_class_pooled_accuracy_pct": per_sb_pooled,
        "empirical_conformal_coverage_pct": round(empirical_coverage * 100, 1),
        "conformal_half_width_weekly": round(float(conformal_q), 3),
        "cameo_vs_analogous_wape_improvement_pct": wape_improvement_pct,
        "smooth_pooled_accuracy_pct": smooth_pooled_acc,
        "operational_pooled_accuracy_pct": operational_pooled_acc,
        "non_lumpy_operational_pooled_accuracy_pct": non_lumpy_operational_pooled_acc,
        "zero_demand_test_drugs": zero_demand_test_drugs,
        "per_drug": per_drug_cameo,
    }
