# SHIELD-XR Data Cleaning — Integration Guide for Cursor

This document explains **how hospital drug demand is cleaned before SHIELD-XR forecasting**. Use it when integrating, modifying, or debugging the forecasting engine in this repo.

**Scope:** SHIELD-XR hospital-wide forecasting only (`backend/src/app/forecasting/shield_xr/`).  
**Out of scope:** CAMEO cold-start module, per-drug legacy SARIMA/LGBM/TFT pipeline (`feature_engineering/pipeline.py`).

Reference notebooks (research baseline, not runtime source of truth):
- `Forecasting_SOL_SHIELD_XR_WeeklyBreakdown_forecast.ipynb`
- `SHIELD_XR_BASELINE_v4.ipynb`

---

## Pipeline overview

```
drug_receipts (+ hospital_census)
        │
        ▼
build_hospital_panel()          ← Stage 1: raw panel assembly
        │
        ▼
split_temporal()                ← Stage 2: train/valid/test cut (70/15/15)
        │
        ▼
apply_anomaly_guard()           ← Stage 3: AnomalyGuard cleaning  ★ core data cleaning
        │
        ▼
assign_sb_classes()             ← Stage 4: Syntetos–Boylan (train-only)
        │
        ▼
add_features()                  ← lags/rolls on CLEAN demand
        │
        ▼
apply_spike_labels()            ← clinical spike flags (train-only stats)
        │
        ▼
SHIELD-XR / Plain models train on `demand` (= demand_clean)
```

**Orchestrator:** `ShieldXRTrainer._prepare_splits()` in `shield_xr/trainer.py`.

---

## Stage 1 — Build the hospital panel

**File:** `shield_xr/panel_builder.py` → `build_hospital_panel()`

### 1.1 Demand from receipts (movement filtering)

Only **patient-facing pharmacy movements** count as demand. Non-clinical ledger events are excluded.

**File:** `services/movement_demand.py`

| MOV# | Role | Contribution |
|------|------|--------------|
| 5 | Patient sale | `+abs(qty)` |
| 6 | Patient return | `-abs(qty)` |
| 8 | Cancel patient sale | `-abs(qty)` |
| 2, 7 | Transfer | `0` |
| Other / missing MOV# | Legacy rows | signed `qty` (backward compatibility) |

Daily demand per `(drug_code, receipt_date)` = sum of signed contributions, then:

```python
demand = clip_daily_demand(total)  # max(0, net) — negative net → 0
```

### 1.2 Calendar reindex (complete SKU × day grid)

- All distinct drug codes in the query window
- All dates in `[start_date, end_date]`, **restricted to import-coverage periods** when configured
- Missing `(CODE, DATE)` pairs filled with `demand = 0`
- `ARTICLE`, `CAT` forward-filled per SKU from first non-null row

**Import coverage:** `get_import_coverage_periods()` in `services/demand_aggregation.py` — hospital/file coverage intervals, not per-drug sparsity. Prevents training on calendar days where the source system had no reliable export.

### 1.3 Hospital exogenous drivers (merged by DATE)

Built in `_build_hospital_exog()` from `drug_receipts` + optional `hospital_census`:

| Column | Meaning |
|--------|---------|
| `n_unique_patients` | Distinct receipt IDs |
| `n_admissions` | Distinct admission dates |
| `n_unique_doctors` | Distinct doctor names |
| `n_unique_CR` / `n_unique_CS` | Distinct center receipt / syn IDs |
| `n_transactions` | Row count |
| `n_demand_txns` | Rows with positive patient demand contribution |
| `top1_CR_share`, `top2_CR_share`, `top3_CR_share` | Daily share of top cost centers |
| `bed_occupancy_rate`, `weekly_surgery_count` | From `hospital_census` when available |

Missing exog values are median-filled (or 0 if all missing).

### 1.4 Panel output schema (pre-cleaning)

Minimum columns expected by AnomalyGuard:

```
CODE, DATE, demand, ARTICLE, CAT,
n_unique_patients, n_admissions, ... (hospital exog)
```

`demand` at this point = **raw aggregated patient demand** (non-negative).

---

## Stage 2 — Temporal split first (no leakage)

**File:** `shield_xr/panel_builder.py` → `split_temporal()`

Default fractions: **70% train / 15% validation / 15% test** (by calendar day, global cut).

```python
d_train_end  = dates[int(n_days * 0.70)]
d_valid_end  = dates[int(n_days * 0.85)]
```

**Critical rule:** All cleaning statistics (ceilings, z-score baselines, DOW seasonal means, SB classes, spike thresholds) must be computed from **train period only** (`DATE <= d_train_end`). AnomalyGuard already enforces this internally; do not change the order to split-after-clean.

---

## Stage 3 — AnomalyGuard (core data cleaning)

**File:** `shield_xr/anomaly_guard.py` → `apply_anomaly_guard(df, d_train_end)`

AnomalyGuard removes **non-clinical data artifacts** — bulk stock corrections, transfer miscoding, and recording/export gaps — while preserving real clinical demand spikes when hospital activity explains them.

### 3.1 Per-SKU demand ceiling (train-only)

Computed on train rows with `demand > 0`:

```python
ceiling = max(median + K_MAD * MAD, P99.5)
```

| Constant | Value | Meaning |
|----------|-------|---------|
| `K_MAD` | 8.0 | Robust multiplier on median absolute deviation |
| `MIN_NONZERO_FOR_CEILING` | 5 | Minimum nonzero days to trust SKU-specific ceiling |
| Fallback | global P99.5 of all train nonzero demand | Used for sparse SKUs |

Outputs:
- `ceiling` — per-row ceiling from `ceiling_map[CODE]`
- `exceeds_ceiling` — `demand > ceiling`

### 3.2 Day-level anomaly detection

Aggregate per `DATE`:

- `exceed_frac` = fraction of active SKUs exceeding their ceiling
- `nz_frac` = fraction of SKUs with demand > 0
- `nz_frac_smooth` = 5-day centered rolling mean of `nz_frac` (reduces patchy dropout flags)

Z-scores use **train-day** mean/std for:
- `exceed_frac` → `exceed_z`
- `nz_frac_smooth` → `nz_z`
- `n_unique_patients` → `patients_z`
- `n_admissions` → `admissions_z`

#### Type 1 — SPIKE anomaly (bulk correction / transfer miscoded as sales)

```python
is_spike_anomaly = (
    exceed_z > EXCEED_Z_THR          # 4.0 — unusually many SKUs spike together
    and |patients_z| < ACTIVITY_Z_THR  # 2.0 — hospital activity is normal
    and |admissions_z| < ACTIVITY_Z_THR
)
```

Interpretation: many drugs spike on the same day, but patient/admission counts are normal → likely **data artifact**, not real clinical surge.

#### Type 2 — DROPOUT anomaly (recording / export gap)

```python
is_dropout_anomaly = nz_z < -DROPOUT_Z_THR   # -2.0
```

Interpretation: the share of SKUs with any recorded demand collapses far below normal → likely **missing data**, not a real hospital-wide demand drop.

Combined:
```python
is_anomaly_day = is_spike_anomaly | is_dropout_anomaly
is_dropout_day = is_dropout_anomaly
```

### 3.3 Build clean demand series

Always preserve raw values:

```python
demand_raw = demand   # untouched original
```

**DOW seasonal imputation table** (train-only):
```python
dow_seasonal_mean = mean(demand) grouped by (CODE, day_of_week)
```

**Cleaning rules:**

| Condition | Action on `demand_clean` |
|-----------|--------------------------|
| SPIKE day + row exceeds ceiling + not dropout | **Winsorize** to `ceiling` |
| DROPOUT day (any SKU on that date) | **Impute** with SKU's `dow_seasonal_mean` (0 if unknown) |
| All other rows | Keep `demand_raw` |

```python
spike_mask = is_anomaly_day & exceeds_ceiling & ~is_dropout_day
demand_clean[spike_mask] = ceiling[spike_mask]
demand_clean[is_dropout_day] = dow_seasonal_mean[is_dropout_day]
demand = demand_clean   # downstream code uses `demand` as the clean series
```

**Why dropout uses imputation (not capping):** dropout rows are **untrustworthily low**, not high. Capping does nothing. Imputing DOW seasonal mean prevents weeks of fake near-zeros from poisoning lag/rolling features after the gap ends.

### 3.4 AnomalyGuard outputs

`AnomalyGuardResult`:
- `frame` — panel with cleaning columns
- `day_stats` — per-day diagnostics (for auditing)
- `ceiling_map`, `global_p995`

Key columns added to `frame`:

| Column | Purpose |
|--------|---------|
| `demand_raw` | Original demand before cleaning |
| `demand_clean` | Cleaned demand |
| `demand` | Alias → `demand_clean` (training target) |
| `ceiling` | Per-SKU ceiling used for winsorization |
| `exceeds_ceiling` | Row exceeded ceiling on raw demand |
| `is_anomaly_day` | Day flagged (spike or dropout) |
| `is_dropout_day` | Day flagged as dropout specifically |
| `dow_seasonal_mean` | Imputation value for dropout days |

---

## Stage 4 — Post-cleaning steps (not artifact removal, but depend on clean series)

### 4.1 Syntetos–Boylan demand class

**File:** `shield_xr/features.py` → `assign_sb_classes(df, d_train_end)`

Computed on **train clean demand** per SKU:

```python
ADI = n_periods / n_nonzero_periods
CV² = (std / mean)²
```

| ADI | CV² | Class |
|-----|-----|-------|
| ≤ 1.32 | ≤ 0.49 | smooth |
| > 1.32 | ≤ 0.49 | intermittent |
| ≤ 1.32 | > 0.49 | erratic |
| > 1.32 | > 0.49 | lumpy |

Stored as `sb_class` / `sb_class_id`. Used for model blending and evaluation breakdowns.

### 4.2 Feature engineering

**File:** `shield_xr/features.py` → `add_features()`

- All lags (`lag_1`, `lag_7`, …) and rolling stats use **`demand` (= clean)**
- `hospital_total_demand` exog driver recomputed from clean demand (prevents anomaly leakage)
- Hospital exog columns lagged by 1 day (`exog_*_lag1`)

### 4.3 Clinical spike labels (for SHIELD-XR hurdle stack)

**File:** `shield_xr/features.py` → `apply_spike_labels()`

Train-only per-SKU stats:
```python
spike = 1 if demand > mean + 2.5 * std else 0
```

This labels **genuine high-demand days on the clean series** for the spike head — different from AnomalyGuard artifact detection.

---

## Evaluation convention (raw vs clean)

| View | Actuals used | When to report |
|------|--------------|----------------|
| **Clean / clinical** | `demand_raw` on non-`is_anomaly_day` test rows | Primary accuracy (~85% hospital-week target) |
| **Raw** | `demand_raw` on all test rows | Diagnostic — shows artifact contamination impact |

In `ShieldXRTrainer.train_all()`:
```python
mask_ok = ~test["is_anomaly_day"]
evaluate(test.loc[mask_ok, "demand_raw"], ensemble_pred[mask_ok], ...)
```

Metrics: `shield_xr/metrics.py` → `wape()`, `accuracy_from_wape()` where `Accuracy = max(0, 1 - WAPE)`.

---

## Persistence after training

**File:** `shield_xr/persistence.py` → `persist_training_panel()`

Cleaned panel rows saved to `forecast_training_data` table:

```
CODE, DATE, demand_raw, demand_clean, ARTICLE, CAT, sb_class,
is_anomaly_day, is_dropout_day, ceiling
```

Useful for auditing AnomalyGuard decisions without re-running training.

---

## Relationship to legacy per-drug pipeline

The older single-drug pipeline (`feature_engineering/pipeline.py`, `data_quality.py`, `censored_demand/`) has its **own** quality gates (coverage gaps, stockout imputation, external/supplier default rates). SHIELD-XR does **not** call those modules today.

| Concern | SHIELD-XR path | Legacy per-drug path |
|---------|----------------|----------------------|
| Movement filtering | `panel_builder` + `movement_demand` | `demand_aggregation` |
| Artifact cleaning | `anomaly_guard` | `censored_demand` / EM correction |
| Quality gating | implicit via AnomalyGuard + min history in trainer | `data_quality.assess_series_quality()` |
| Target column | `demand` / `demand_clean` | `total_quantity` / `em_corrected_quantity` |

When integrating new ingestion paths, ensure **both** pipelines receive consistently filtered patient demand if both remain active.

---

## Integration rules for Cursor (do not break)

1. **Never compute cleaning stats on validation/test data.** Always pass `d_train_end` into AnomalyGuard and SB/spike label functions.

2. **Keep `demand_raw` immutable** after panel build. All cleaning writes go to `demand_clean`; then set `demand = demand_clean`.

3. **Dropout imputation ≠ zero-fill.** Do not replace dropout days with 0 — use DOW seasonal mean.

4. **Spike cleaning is conditional.** Only cap rows where `is_anomaly_day & exceeds_ceiling & ~is_dropout_day`. Do not blanket-cap all ceiling exceedances on normal days (those may be real clinical spikes).

5. **Feature lags must use clean demand.** If you add new lag features, source them from `demand` after AnomalyGuard, not from raw receipts.

6. **Hospital activity cross-check is required for spike detection.** Removing `n_unique_patients` / `n_admissions` from the panel will cause false spike flags on busy clinical days.

7. **Import coverage periods matter.** `build_hospital_panel()` restricts the calendar to covered export intervals — do not bypass this when reindexing.

8. **Tests to run after changes:**
   ```bash
   cd backend && pytest tests/test_shield_xr.py -q
   ```

---

## File map

| File | Responsibility |
|------|----------------|
| `shield_xr/panel_builder.py` | DB → SKU×day panel, exog merge, temporal split |
| `shield_xr/anomaly_guard.py` | Ceilings, spike/dropout detection, `demand_clean` |
| `shield_xr/features.py` | SB class, lags/rolls, spike labels, ID encoding |
| `shield_xr/trainer.py` | Orchestrates cleaning → features → train → persist |
| `shield_xr/persistence.py` | Saves cleaned panel to DB |
| `services/movement_demand.py` | MOV# taxonomy, signed demand contribution |
| `services/demand_aggregation.py` | Import coverage periods, legacy aggregation helpers |
| `models/forecast_training_data.py` | ORM for persisted cleaned panel |
| `tests/test_shield_xr.py` | Unit tests for AnomalyGuard + features |

---

## Tuning constants (change only with re-validation)

All in `shield_xr/anomaly_guard.py`:

```python
K_MAD = 8.0
MIN_NONZERO_FOR_CEILING = 5
EXCEED_Z_THR = 4.0      # spike: co-exceedance z-score
ACTIVITY_Z_THR = 2.0    # spike: patients/admissions must be "normal"
DROPOUT_Z_THR = 2.0     # dropout: nz_frac_smooth z-score (negative)
```

Validated behavior on 2023–2024 hospital data (~731 days, ~893 SKUs):
- ~75 anomaly days flagged (~10%), mostly dropout-type (~63) vs spike-type (~12)
- Hospital-week ensemble accuracy ~84–85% on clean test weeks after full pipeline

---

## Quick debugging checklist

When forecasts look wrong, inspect in order:

1. **Raw panel:** Are MOV# filters correct? (`movement_demand.patient_demand_contribution`)
2. **Coverage gaps:** Are missing export days creating artificial zeros before AnomalyGuard?
3. **AnomalyGuard flags:** Query `forecast_training_data` for `is_anomaly_day`, `is_dropout_day`
4. **Ceiling map:** Are high-volume SKUs getting reasonable ceilings?
5. **Clean vs raw totals:** Compare `sum(demand_raw)` vs `sum(demand_clean)` on flagged days
6. **Feature leakage:** Confirm lags use post-AnomalyGuard `demand`, not pre-clean values
