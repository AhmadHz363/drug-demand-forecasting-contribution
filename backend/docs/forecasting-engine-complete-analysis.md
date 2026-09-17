# Forecasting Engine — Complete Technical Analysis

> **Purpose:** This document is a self-contained reference for understanding the backend drug demand forecasting engine — its data sources, fields, algorithms, pipelines, API contracts, and artifacts. It is written for AI assistants and engineers who need full context without reading the codebase.

**Codebase root:** `backend/src/app/forecasting/`  
**Companion doc (feature-focused):** `backend/docs/forecasting-features.md`

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Data Sources & Database Schema](#3-data-sources--database-schema)
4. [Demand Aggregation & Preprocessing](#4-demand-aggregation--preprocessing)
5. [Feature Engineering — All Fields](#5-feature-engineering--all-fields)
6. [Censored Demand Correction](#6-censored-demand-correction)
7. [Base Models (SARIMA, LightGBM, TFT)](#7-base-models-sarima-lightgbm-tft)
8. [Ensemble & Uncertainty Quantification](#8-ensemble--uncertainty-quantification)
9. [Training Pipeline](#9-training-pipeline)
10. [Inference Pipeline](#10-inference-pipeline)
11. [Hold-Out Validation & Evaluation](#11-hold-out-validation--evaluation)
12. [API Reference](#12-api-reference)
13. [Request/Response Schemas](#13-requestresponse-schemas)
14. [Configuration & Constants](#14-configuration--constants)
15. [Artifacts & Persistence](#15-artifacts--persistence)
16. [Integration with Other Systems](#16-integration-with-other-systems)
17. [Cold Start (Separate Path)](#17-cold-start-separate-path)
18. [Dependencies](#18-dependencies)
19. [Error Handling & Edge Cases](#19-error-handling--edge-cases)
20. [Source File Index](#20-source-file-index)

---

## 1. Executive Summary

The forecasting engine predicts **daily pharmacy drug consumption demand** per `drug_code`, optionally scoped to a hospital center (`center_syn_id`).

**What it predicts:** Non-negative daily quantity (units consumed/dispensed), aggregated from pharmacy receipt lines.

**How it works (high level):**

1. Aggregate receipt lines into a daily time series per drug.
2. Engineer ~47 calendar, lag, rolling, census, and supplier features.
3. Correct censored demand during stockout periods.
4. Run three base models in parallel:
   - **SARIMA** — univariate time series on EM-corrected demand
   - **LightGBM** — gradient boosting on all engineered features (recursive multi-step)
   - **TFT** — Temporal Fusion Transformer with attention over feature groups
5. Blend base model P50 predictions via **Ridge stacking** (inverse-MAE weights).
6. Calibrate **P10/P90 uncertainty bands** with MAPIE conformal prediction.
7. Extrapolate approximate **P5/P95** tail quantiles.
8. Persist results to `forecast_results` and return via REST API.

**Forecast horizon:** 1–30 days (default 7).  
**Training scope:** Always all centers (`center_syn_id=None`).  
**Inference scope:** Can filter to one center.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PostgreSQL Database                               │
│  drug_receipts │ hospital_census │ supplier_lead_times │ stockout_flags    │
│  forecast_results │ model_performance │ daily_drug_demand (materialized)   │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     demand_aggregation.py                                   │
│  aggregate_daily_demand_rows() → SUM(quantity) GROUP BY drug_code, date     │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              feature_engineering/pipeline.py                                │
│  build_feature_matrix() → temporal + lag + rolling + external + supplier    │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              censored_demand/corrector.py                                   │
│  correct_demand() → stockout detection, Tobit/Weibull, EM-SARIMA            │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
        ┌──────────┐          ┌──────────┐          ┌──────────┐
        │  SARIMA  │          │ LightGBM │          │   TFT    │
        │ univariate│         │ features │          │ attention│
        └────┬─────┘          └────┬─────┘          └────┬─────┘
             │                     │                     │
             └─────────────────────┼─────────────────────┘
                                   ▼
                    ┌──────────────────────────┐
                    │  StackingMetaLearner     │
                    │  (Ridge + inverse-MAE)   │
                    └────────────┬─────────────┘
                                 ▼
                    ┌──────────────────────────┐
                    │  ConformalCalibrator     │
                    │  (MAPIE CrossConformal)  │
                    └────────────┬─────────────┘
                                 ▼
                    ┌──────────────────────────┐
                    │  extend_tail_quantiles   │
                    │  P5, P10, P50, P90, P95  │
                    └────────────┬─────────────┘
                                 ▼
                    ┌──────────────────────────┐
                    │  forecast_results (DB)   │
                    │  ForecastResponse (API)  │
                    └──────────────────────────┘
```

### Module Registry

| Component | File | Class/Function |
|-----------|------|--------------|
| API router | `app/api/forecasting.py` | FastAPI router `prefix="/forecasting"` |
| Inference | `forecasting/inference/forecaster.py` | `DrugForecaster`, `validate_forecast_quantiles()` |
| Training | `forecasting/training/trainer.py` | `ForecastingTrainer`, `MODEL_REGISTRY` |
| Feature pipeline | `forecasting/feature_engineering/pipeline.py` | `build_feature_matrix()`, `build_future_covariates()` |
| Censored demand | `forecasting/censored_demand/corrector.py` | `correct_demand()` |
| SARIMA | `forecasting/models/sarima_model.py` | `SarimaModel` |
| LightGBM | `forecasting/models/lgbm_model.py` | `LightGBMModel` |
| TFT | `forecasting/models/tft_model.py` | `TFTModel` |
| Ensemble | `forecasting/ensemble/stacking.py` | `StackingMetaLearner` |
| Conformal | `forecasting/ensemble/conformal.py` | `ConformalCalibrator`, `extend_tail_quantiles()` |
| Demand aggregation | `app/services/demand_aggregation.py` | `aggregate_daily_demand_rows()` |
| Schemas | `forecasting/schemas.py` | All Pydantic models |
| Constants | `forecasting/constants.py` | Hyperparameters |

---

## 3. Data Sources & Database Schema

### 3.1 Primary Source: `drug_receipts`

**Table:** `drug_receipts`  
**Model:** `app/models/drug_receipt.py` → `DrugReceipt`

This is the **source of truth** for forecasting. Each row is a pharmacy receipt line.

| Column | Type | Used by Forecasting | Description |
|--------|------|---------------------|-------------|
| `drug_code` | `String(255)` | **Yes — primary key** | Drug identifier for all forecasting operations |
| `receipt_date` | `Date` | **Yes** | Date demand is attributed to (becomes `demand_date`) |
| `quantity` | `Numeric(18,4)` | **Yes** | Line quantity; summed daily per drug |
| `center_syn_id` | `String(255)` | **Yes (optional filter)** | Hospital center; scopes aggregation at inference |
| `drug_name` | `String(1024)` | Search only | Display name in drug search API |
| `drug_id` | `Integer` (FK) | No | Links to `drugs` registry table |
| `category_id` | `Integer` (FK) | No | Category reference |
| `receipt_id` | `String(255)` | No | Receipt identifier |
| `movement_type` | `String(255)` | No | Movement classification |
| `unit_price`, `total_price` | `Numeric` | No | Financial fields |
| `admission_date`, `room_number`, `bed_number`, `doctor_name` | Various | No | Clinical metadata |

**Daily demand formula:**

```sql
SELECT drug_code, receipt_date AS demand_date, COALESCE(SUM(quantity), 0) AS total_quantity
FROM drug_receipts
WHERE drug_code = :drug_code
  AND receipt_date BETWEEN :start_date AND :end_date
  [AND center_syn_id = :center_syn_id]  -- optional
GROUP BY drug_code, receipt_date
ORDER BY receipt_date
```

### 3.2 Supporting Tables

#### `hospital_census`

| Column | Type | Default if Missing |
|--------|------|------------------|
| `census_date` | `Date` | — |
| `bed_occupancy_rate` | `Numeric(5,4)` | `0.75` |
| `weekly_surgery_count` | `Integer` | `50` |
| `center_syn_id` | `String(255)` | Averaged across centers when unscoped |

#### `supplier_lead_times`

| Column | Type | Default if Missing |
|--------|------|------------------|
| `drug_code` | `String(255)` unique | — |
| `avg_lead_time_days` | `Numeric(6,2)` | `3.0` → feature `supplier_avg_lead_time` |
| `lead_time_std_days` | `Numeric(6,2)` | `1.0` → feature `supplier_lead_time_std` |
| `reliability_score` | `Numeric(4,3)` | `0.85` → feature `supplier_reliability_score` |

#### `stockout_flags` (written by correction pipeline)

| Column | Type | Description |
|--------|------|-------------|
| `drug_code` | `String(255)` | Drug |
| `center_syn_id` | `String(255)` nullable | Center scope |
| `flag_date` | `Date` | Stockout date |
| `observed_quantity` | `Numeric` | Always `0.0` |
| `estimated_true_demand` | `Numeric` | Imputed demand |
| `correction_method` | `String` | `"tobit"`, `"weibull"`, or `"none"` |

#### `forecast_results` (output persistence)

| Column | Type | Description |
|--------|------|-------------|
| `drug_code` | `String(255)` | Drug |
| `center_syn_id` | `String(255)` nullable | Center scope |
| `generated_at` | `DateTime(tz)` | When forecast was produced |
| `forecast_date` | `Date` | Target date |
| `p5`, `p10`, `p50`, `p90`, `p95` | `Numeric(18,4)` | Quantile predictions |
| `model_weight_sarima` | `Numeric(6,4)` | Ensemble weight at generation |
| `model_weight_lgbm` | `Numeric(6,4)` | Ensemble weight at generation |
| `model_weight_tft` | `Numeric(6,4)` | Ensemble weight at generation |

**Unique constraint:** `(drug_code, center_syn_id, generated_at, forecast_date)`

#### `model_performance` (training metrics)

| Column | Type | Description |
|--------|------|-------------|
| `drug_code` | `String(255)` | Drug |
| `model_name` | `String(32)` | `"sarima"`, `"lgbm"`, `"tft"`, or `"ensemble"` |
| `smape` | `Numeric(8,4)` | Symmetric MAPE from walk-forward validation |
| `coverage_90` | `Numeric(6,4)` | Fraction of actuals within P10–P90 |
| `evaluated_at` | `DateTime(tz)` | Auto timestamp |

#### `daily_drug_demand` (materialized, optional)

Pre-aggregated daily totals. Forecasting reads **`drug_receipts` directly** by default. A sync function `sync_daily_demand_from_receipts()` exists to refresh this table but is not required for the main pipeline.

---

## 4. Demand Aggregation & Preprocessing

### 4.1 Aggregation

**Function:** `aggregate_daily_demand_rows(db_session, drug_code, start_date, end_date, center_syn_id=None)`

Returns: `list[tuple[date, float]]` — one `(demand_date, total_quantity)` per day with receipt activity.

### 4.2 Calendar Reindexing

**Function:** `_load_demand_series()` in `pipeline.py`

- Builds a complete daily calendar from `start_date` to `end_date`.
- Missing days are filled with `total_quantity = 0.0`.
- Adds `demand_filled = 1` for synthetic zero-fill days, `0` for days with actual receipt data.

### 4.3 Consumption Demand Normalization

**Function:** `apply_consumption_demand()` in `demand_quantity.py`

```python
total_quantity = abs(net_daily_sum)
```

Negative daily totals (e.g., inter-department transfers) are treated as positive consumption because they still deplete stock.

### 4.4 Null Handling

After all feature engineering, any remaining NaN values are filled with `0`.

---

## 5. Feature Engineering — All Fields

**Entry point:** `build_feature_matrix(drug_code, center_syn_id, db_session, start_date, end_date)`

**Target column:** `total_quantity` — the value models learn to predict.

**Index:** `demand_date` (daily, complete calendar)

### 5.1 Column Inventory Summary

| Family | Count | Source File |
|--------|-------|-------------|
| Target | 1 | `total_quantity` |
| Temporal | 14 | `temporal_features.py` |
| Lag | 5 | `lag_features.py` |
| Rolling | 23 | `rolling_features.py` |
| External | 2 | `external_features.py` |
| Supplier | 3 | `supplier_features.py` |
| Pipeline extras | 4 | `pipeline.py` |
| Censored-demand metadata | 3 | `corrector.py` |
| **Total columns in feature matrix** | **~55** | |

### 5.2 Temporal Features (14 columns)

**Function:** `add_temporal_features()`  
**Source:** Derived from `demand_date`. Holidays from `holidays.Lebanon` (years 2018–2029).

| Column | Type | Description |
|--------|------|-------------|
| `day_of_week` | int 0–6 | Monday = 0 |
| `day_of_week_sin` | float | Cyclical encoding, period 7 |
| `day_of_week_cos` | float | Cyclical encoding, period 7 |
| `week_of_year` | int | ISO week number |
| `week_of_year_sin` | float | Cyclical encoding, period 53 |
| `week_of_year_cos` | float | Cyclical encoding, period 53 |
| `month` | int 1–12 | Calendar month |
| `month_sin` | float | Cyclical encoding, period 12 |
| `month_cos` | float | Cyclical encoding, period 12 |
| `quarter` | int 1–4 | Calendar quarter |
| `is_weekend` | int 0/1 | 1 if Saturday or Sunday |
| `is_public_holiday` | int 0/1 | 1 if Lebanon public holiday |
| `days_to_next_holiday` | int | Days until next holiday (capped at 30) |
| `days_since_last_holiday` | int | Days since last holiday (capped at 30) |

### 5.3 Lag Features (5 columns)

**Function:** `add_lag_features()`  
**Lag offsets:** `[1, 7, 28, 365]` days (from `LAG_DAYS` constant)

| Column | Type | Description |
|--------|------|-------------|
| `lag_1d` | float | Demand 1 day ago |
| `lag_7d` | float | Demand 7 days ago |
| `lag_28d` | float | Demand 28 days ago |
| `lag_365d` | float | Demand 365 days ago |
| `has_lag_gaps` | int 0/1 | **Categorical.** 1 if any lag was imputed |

**Imputation rule:** If history is shorter than a lag offset, fill with the **median** of `total_quantity` and set `has_lag_gaps = 1`.

### 5.4 Rolling Features (23 columns)

**Function:** `add_rolling_features()`  
**Windows:** `[28, 91, 182]` days (from `ROLLING_WINDOWS`)  
**Leakage prevention:** All rolling stats use `series.shift(1)` — preceding days only, never same-day demand.  
**Minimum observations:** `min_periods=7`; otherwise filled with series median or 0.

**Per window (×3 windows = 15 columns):**

| Column Pattern | Description |
|----------------|-------------|
| `rolling_mean_{w}d` | Rolling mean |
| `rolling_std_{w}d` | Rolling standard deviation |
| `rolling_min_{w}d` | Rolling minimum |
| `rolling_max_{w}d` | Rolling maximum |
| `rolling_cv_{w}d` | Coefficient of variation (std / mean) |

**Additional (2 columns):**

| Column | Description |
|--------|-------------|
| `trend_slope_28d` | Linear slope over preceding 28 days |
| `demand_zscore_28d` | Z-score of current demand vs 28-day mean/std |

**Spike detection (×3 windows = 6 columns):**

| Column Pattern | Description |
|----------------|-------------|
| `spike_count_{w}d` | Count of days exceeding mean + 2σ in window |
| `spike_intensity_{w}d` | Max positive deviation above rolling mean in window |

### 5.5 External Features (2 columns)

**Function:** `add_external_features()`  
**Source:** `hospital_census` table

| Column | Type | Default | Description |
|--------|------|---------|-------------|
| `bed_occupancy_rate` | float | 0.75 | Hospital bed occupancy rate |
| `weekly_surgery_count` | int | 50 | Weekly surgery volume |

When `center_syn_id` is set, census is filtered to that center. When `None`, values are averaged across all centers per date.

### 5.6 Supplier Features (3 columns)

**Function:** `add_supplier_features()`  
**Source:** `supplier_lead_times` table (one row per drug)

These are **constant per drug** — same value repeated for every day in the series.

| Column | DB Source | Default | Description |
|--------|-----------|---------|-------------|
| `supplier_avg_lead_time` | `avg_lead_time_days` | 3.0 | Average supplier lead time (days) |
| `supplier_lead_time_std` | `lead_time_std_days` | 1.0 | Lead time standard deviation |
| `supplier_reliability_score` | `reliability_score` | 0.85 | Supplier reliability (0–1) |

### 5.7 Pipeline Extras (4 columns)

Added by `build_feature_matrix()`:

| Column | Training Default | Inference Default | Description |
|--------|-----------------|-------------------|-------------|
| `demand_filled` | 0 or 1 | 1 for future rows | 1 = synthetic/missing day |
| `forecast_step` | 0 | 1..horizon | Step within forecast horizon |
| `forecast_step_sin` | 0.0 | cyclical | sin(2π × step / 30) |
| `forecast_step_cos` | 1.0 | cyclical | cos(2π × step / 30) |

### 5.8 Censored-Demand Metadata (3 columns)

Added by `correct_demand()`. **Not used as model input features** (except `em_corrected_quantity` as SARIMA target):

| Column | Description |
|--------|-------------|
| `em_corrected_quantity` | EM-SARIMA corrected demand — **SARIMA target only** |
| `is_stockout` | bool — day flagged as stockout |
| `correction_method` | `"tobit"`, `"weibull"`, or empty |

### 5.9 Metadata Columns (excluded from LightGBM)

Defined in `models/feature_columns.py`:

```python
METADATA_COLUMNS = {
    "total_quantity",        # target
    "em_corrected_quantity",
    "correction_method",
    "is_stockout",
    "demand_date",
}
```

**LightGBM input count:** All columns except `METADATA_COLUMNS` ≈ **51 features** (47 engineered + 4 pipeline extras).

### 5.10 Model-Specific Feature Mapping

#### SARIMA
- **Input:** Single univariate series only
- **Target:** `em_corrected_quantity` if present, else `total_quantity`

#### LightGBM
- **Features:** All non-metadata columns
- **Categorical:** `has_lag_gaps`
- **Target:** `total_quantity`

#### TFT (Temporal Fusion Transformer)

| Group | Columns | Count | Known at Forecast Time? |
|-------|---------|-------|---------------------------|
| Target | `total_quantity` | 1 | — |
| `time_varying_known_reals` | temporal + external | 16 | Yes |
| `time_varying_unknown_reals` | lags (no `has_lag_gaps`) + rolling | 27 | No — inferred |
| `static_reals` | supplier | 3 | Yes (constant per drug) |

**Note:** `has_lag_gaps` is used by LightGBM but **excluded from TFT**.

### 5.11 Future Covariates (LightGBM Recursive Inference)

**Function:** `build_future_covariates(drug_code, center_syn_id, db_session, last_date, horizon_days)`

Pre-computes known future values for recursive forecasting:
- All temporal features (calendar is known)
- External features (census — uses DB or defaults)
- Supplier features (constant)
- `forecast_step`, `forecast_step_sin`, `forecast_step_cos`
- `demand_filled = 1`

Lag and rolling features are rebuilt dynamically via `build_recursive_feature_row()` as each day's P50 is predicted and appended to history.

---

## 6. Censored Demand Correction

**Purpose:** Observed zero demand during stockouts understates true consumption. This pipeline imputes latent demand before model training/inference.

**Orchestrator:** `correct_demand()` in `censored_demand/corrector.py`

### 6.1 Pipeline Steps

```
1. detect_stockout_windows()
       ↓
2. Compute stockout_rate
       ↓
3. Choose correction method:
   rate == 0     → skip (em_corrected = total_quantity)
   rate < 0.20   → Tobit MLE correction
   rate ≥ 0.20   → Weibull AFT correction (fallback to Tobit if <3 stockouts)
       ↓
4. em_corrected_series() — EM-SARIMA iterative imputation
       ↓
5. Write stockout_flags to DB
       ↓
6. Return corrected feature matrix
```

### 6.2 Stockout Detection

**Function:** `detect_stockout_windows()` in `detector.py`

**Rule:** A day is a stockout if:
- `total_quantity == 0`, AND
- The drug had non-zero demand in the ±14-day window (29-day centered rolling sum > 0)

**Output:** `is_stockout` (bool), `stockout_rate` stored in `df.attrs['stockout_rate']`

### 6.3 Correction Methods

| Method | When Used | Library | Description |
|--------|-----------|---------|-------------|
| **Tobit MLE** | `stockout_rate < 0.20` | `scipy.optimize.minimize` | Truncated normal imputation for censored zeros |
| **Weibull AFT** | `stockout_rate ≥ 0.20` | `lifelines.WeibullAFTFitter` | Survival model for high stockout rates |
| **EM-SARIMA** | Always (if stockouts exist) | `statsmodels` SARIMAX | Iterative SARIMA fit/impute on stockout rows |

**Threshold constant:** `STOCKOUT_RATE_THRESHOLD = 0.20`  
**Weibull minimum:** `MIN_STOCKOUT_PERIODS = 3`

---

## 7. Base Models (SARIMA, LightGBM, TFT)

All models inherit from `BaseForecastingModel` and implement:
- `train(df, drug_code) → None`
- `predict(df, horizon_days, **kwargs) → DataFrame` with columns `p10`, `p50`, `p90`
- `save(drug_code) → str` (artifact path)
- `load(drug_code) → None`
- `is_trained(drug_code) → bool`

### 7.1 SARIMA (`SarimaModel`)

| Setting | Value |
|---------|-------|
| Library | `statsmodels.tsa.statespace.sarimax.SARIMAX` |
| Order | `(1, 1, 1)` |
| Seasonal order | `(1, 1, 1, 7)` — weekly seasonality |
| Target | `em_corrected_quantity` (fallback: `total_quantity`) |
| Training window | Last 182 days (`SARIMA_TRAIN_DAYS`) |
| Min history | 60 days |
| Max iterations | 200 |
| Forecast shrinkage | Blend raw forecast toward 28-day median (`SARIMA_FORECAST_SHRINKAGE = 0.4`) |
| Intervals | 80% SARIMAX confidence → mapped to p10/p90 |
| Artifact | `artifacts/sarima/{drug_code}.pkl` |

### 7.2 LightGBM (`LightGBMModel`)

| Setting | Value |
|---------|-------|
| Library | `lightgbm` |
| Quantiles | P10, P50, P90 (`LGBM_QUANTILES = [0.10, 0.50, 0.90]`) |
| P50 objective | Tweedie (variance power 1.5) |
| P10/P90 objective | Quantile regression |
| Features | All non-metadata columns (~51) |
| Categorical | `has_lag_gaps` |
| Training window | Last 365 days (`LGBM_LOOKBACK_WINDOW`) |
| Min history | 90 days |
| Estimators | 500 (`LGBM_N_ESTIMATORS`) |
| Learning rate | 0.05 |
| Max depth | 6 |
| Num leaves | 31 |
| Validation fraction | 15% |
| Early stopping | 50 rounds |
| Multi-step | **Recursive** — each P50 appended to history |
| Stabilizers | Drift guard, anchor blend (`LGBM_RECURSIVE_ANCHOR_WEIGHT = 0.25`), dampening, prediction cap |
| SHAP | Optional; saved to `{drug_code}_shap.json` |
| Artifacts | `{drug_code}_p10.txt`, `_p50.txt`, `_p90.txt`, `_meta.json` |

**Recursive forecasting flow:**
1. `build_future_covariates()` — known future calendar/census/supplier
2. `build_recursive_feature_row()` — rebuild lags/rolling from updated history
3. Predict day 1 P50 → append to history → repeat for day 2..N

### 7.3 TFT (`TFTModel`)

| Setting | Value |
|---------|-------|
| Libraries | `pytorch-forecasting`, `lightning.pytorch`, `torch` |
| Architecture | `TemporalFusionTransformer` |
| Loss | `QuantileLoss` with quantiles `[0.05, 0.10, 0.50, 0.90, 0.95]` |
| Hidden size | 64 |
| Attention heads | 4 |
| Dropout | 0.1 |
| Max epochs | 50 (5 for walk-forward, 10 for holdout) |
| Batch size | 64 |
| Learning rate | 1e-3 |
| Encoder length | Up to 365 days |
| Prediction length | Up to 30 days |
| Min history | 365 days (skipped gracefully if insufficient) |
| Long horizons | Chunked prediction with history extension |
| GPU | CUDA preferred; MPS opt-in via `TFT_ENABLE_MPS=1` env var |
| Attention | `extract_attention_weights()` → ~52 weekly buckets |
| Artifacts | `{drug_code}.ckpt`, `{drug_code}_dataset.pkl` |

---

## 8. Ensemble & Uncertainty Quantification

### 8.1 Stacking Meta-Learner

**Class:** `StackingMetaLearner` in `ensemble/stacking.py`

**Input features:** P50 predictions from SARIMA, LightGBM, TFT (3 columns)  
**Target:** Actual demand  
**Underlying model:** `sklearn.linear_model.Ridge` (alpha = `ENSEMBLE_ALPHA = 1.0`)

**Actual blending logic:**
1. Compute inverse-MAE weights from validation predictions per model.
2. Zero out models with MAE > 1.25× best model MAE (`ENSEMBLE_MAE_OUTLIER_RATIO`).
3. **Robust blend:** When model disagreement exceeds 50% of prediction cap, pick best single model for that day.
4. Apply `winsorize_predictions()` before blending.

**Output:** Blended P50 array + `ModelWeightBreakdown(sarima, lgbm, tft)` summing to 1.0.

**Fallback:** If stacking artifact missing, equal-weight nan-mean of available models.

**Artifact:** `artifacts/stacking/stacking_meta.pkl` (global, not per-drug)

### 8.2 Conformal Calibration

**Class:** `ConformalCalibrator` in `ensemble/conformal.py`

| Setting | Value |
|---------|-------|
| Library | `mapie.regression.CrossConformalRegressor` |
| Method | `"plus"` with cross-validation |
| Target coverage | 90% (`CONFORMAL_COVERAGE = 0.90`) |
| Post-processing | Asymmetric upper expansion (skew factor 1.5) |
| Scale calibration | Coverage-correcting scale factor from normalized residuals |

**Output:** Calibrated P10 and P90 around ensemble P50.

**Fallback:** If MAPIE artifact missing or fails, use std spread across base model P50s (± spread, min 20% of P50).

**Artifact:** `artifacts/conformal/mapie_wrapper.pkl` (global)

### 8.3 Tail Quantile Extrapolation

**Function:** `extend_tail_quantiles(p50, p10, p90)`

Approximates P5 and P95 from calibrated P10/P90:

```
P5  = max(0, P10 - 0.5 × (P50 - P10))
P95 = P90 + 0.5 × (P90 - P50)
```

**Important:** P5/P95 are **not coverage-guaranteed** — only P10/P90 are conformally calibrated.

### 8.4 Prediction Bounds

**Function:** `demand_prediction_cap()` in `prediction_bounds.py`

Caps extreme predictions based on recent 91-day demand history (`PREDICTION_CAP_RECENT_DAYS`). Applied before ensemble blending via `sanitize_ensemble_predictions()` and `winsorize_predictions()`.

---

## 9. Training Pipeline

**Class:** `ForecastingTrainer` in `training/trainer.py`  
**Entry:** `POST /forecasting/train`

### 9.1 Steps

```
1. Resolve drug list
   - request.drug_codes OR all distinct drug_codes from drug_receipts

2. For each drug:
   a. get_receipt_date_bounds() → full history range
   b. build_feature_matrix(drug, center=None, start, end)
   c. correct_demand() → censored correction + stockout flags
   d. For each model in MODEL_REGISTRY (sarima, lgbm, tft):
      - Skip if already trained (unless force_retrain)
      - model.train(corrected_df, drug_code)
      - TFT may set _skipped=True if <365 days history
      - model.save(drug_code)
      - walk_forward_smape() + walk_forward_coverage()
      - Insert ModelPerformance row
   e. collect_walk_forward_predictions() → OOF predictions for ensemble

3. Aggregate OOF predictions across all drugs

4. Fit global StackingMetaLearner on first 80% of pooled OOF
   Fit ConformalCalibrator on remaining 20%

5. Save ensemble artifacts; db_session.commit()

6. Return TrainStatusResponse
```

### 9.2 Walk-Forward Validation

**Functions:** `walk_forward_smape()`, `walk_forward_coverage()`, `collect_walk_forward_predictions()` in `training/walk_forward.py`

- 3 folds × 7-day horizon at series tail
- Computes per-model sMAPE and P10–P90 coverage
- Collects out-of-fold predictions for global ensemble fitting

### 9.3 Training Scope Notes

- **Always trains on all centers** (`center_syn_id=None`).
- Center-specific models are **not** trained; center filtering happens only at inference aggregation time.

---

## 10. Inference Pipeline

**Class:** `DrugForecaster.forecast()` in `inference/forecaster.py`  
**Entry:** `POST /forecasting/predict`

### 10.1 Steps

```
1. Validate drug_has_receipt_history() → 404 if missing

2. Check trained artifacts → 503 if none exist

3. History window: last 365 days (LGBM_LOOKBACK_WINDOW) through latest receipt date

4. build_feature_matrix() → correct_demand()

5. Base model P50 predictions (parallel):
   - SARIMA: sarima.predict(corrected_df, horizon_days)
   - LightGBM: lgbm.predict(corrected_df, horizon_days, future_covariates=...)
   - TFT: tft.predict(corrected_df, horizon_days)

6. demand_prediction_cap() on recent total_quantity

7. Ensemble: StackingMetaLearner.predict() or nan-mean fallback

8. Intervals: ConformalCalibrator.predict_interval() or spread fallback
   → extend_tail_quantiles() for P5/P95

9. Optional explainability:
   - include_shap=True → LightGBM SHAP top-15 features
   - include_attention=True → TFT weekly attention weights

10. Persist to forecast_results; commit

11. Return ForecastResponse with:
    - forecast: list[DailyForecastPoint]
    - history: last 90 days actual demand (FORECAST_CHART_HISTORY_DAYS)
    - model_weights, smape_last_validation, uncertainty_note
```

### 10.2 Forecast Dates

First forecast date = last demand date + 1 day. Subsequent dates increment daily.

### 10.3 Post-Inference Validation

**Function:** `validate_forecast_quantiles(response)`

- Model weights must sum to 1.0 (±1e-5)
- Quantile ordering: P5 ≤ P10 ≤ P50 ≤ P90 ≤ P95

---

## 11. Hold-Out Validation & Evaluation

**Entry:** `POST /forecasting/holdout`  
**Function:** `run_holdout_validation()` in `training/holdout.py`

### 11.1 Purpose

Backtest models on a calendar hold-out window without using future data for training.

### 11.2 Request Parameters

| Field | Default | Description |
|-------|---------|-------------|
| `drug_code` | required | Drug to evaluate |
| `train_end` | 2025-12-31 | Last day of training data |
| `test_start` | 2026-01-01 | First day of test period |
| `test_end` | 2026-03-31 | Last day of test period |
| `train_start` | receipt min | Optional training start |
| `center_syn_id` | None | Optional center filter |
| `models` | ["sarima","lgbm","tft"] | Models to evaluate |

### 11.3 Metrics

| Metric | Formula / Description |
|--------|----------------------|
| **sMAPE** | `200 × |actual - predicted| / (|actual| + |predicted|)` — 0 when both zero |
| **MAE** | Mean absolute error |
| **coverage_90** | Fraction of actuals within P10–P90 |
| **accuracy_pct** | `max(0, min(100, 100 - smape/2))` — displayed 0–100% |

**Evaluation mask:** `build_evaluation_mask()` excludes stockout days and dominant sentinel values from metrics.

### 11.4 Response

`HoldoutResponse` includes:
- `train_period`, `test_period`
- `metrics` per model + ensemble
- `total_accuracy_pct` (ensemble)
- `model_weights`
- `series`: daily actual vs each model P50 + ensemble P10/P50/P90

**Date suggestions:** `GET /forecasting/holdout/date-suggestions?drug_code=...` returns valid date ranges based on receipt bounds.

---

## 12. API Reference

**Base path:** `/forecasting`  
**Auth:** JWT required when `settings.auth_enabled=True`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/forecasting/receipt-drug-codes/search` | Search drugs in receipts (q min 3 chars) |
| `GET` | `/forecasting/forecasted-drugs/search` | Search drugs with saved forecasts |
| `POST` | `/forecasting/train` | Train models (sync; Celery TODO) |
| `POST` | `/forecasting/predict` | Single-drug forecast |
| `POST` | `/forecasting/predict-batch` | Batch forecast (1–70 drugs) |
| `GET` | `/forecasting/holdout/date-suggestions` | UI date picker helpers |
| `POST` | `/forecasting/holdout` | Hold-out backtest |

### HTTP Error Codes (Predict)

| Code | Condition |
|------|-----------|
| 404 | `DrugNotInCatalogError` — no receipt history |
| 503 | `NoTrainedModelsError` — no artifacts; train first |
| 422 | Validation errors (train, holdout, search) |

### Batch Predict Behavior

Per-drug failures set `error` field on that `ForecastResponse`; other drugs still returned.

---

## 13. Request/Response Schemas

All defined in `forecasting/schemas.py`.

### 13.1 ForecastRequest

```python
drug_code: str                          # required
horizon_days: int = 7                   # 1..30
center_syn_id: Optional[str] = None     # optional center filter
include_shap: bool = False              # LightGBM explainability
include_attention: bool = False         # TFT attention weights
```

### 13.2 ForecastResponse

```python
drug_code: str
center_syn_id: Optional[str]
horizon_days: int
model_weights: ModelWeightBreakdown     # sarima + lgbm + tft = 1.0
forecast: list[DailyForecastPoint]
history: list[HistoryPoint]             # last 90 days
shap_features: Optional[list[ShapFeature]]
attention_weights: Optional[list[AttentionWeight]]
uncertainty_note: str
smape_last_validation: Optional[float]
error: Optional[str]                    # set in batch mode on failure
```

### 13.3 DailyForecastPoint

```python
date: date
p5: Optional[float]     # approximate tail
p10: float              # conformally calibrated ~10th percentile
p50: float              # median / point forecast
p90: float              # conformally calibrated ~90th percentile
p95: Optional[float]    # approximate tail
```

### 13.4 ModelWeightBreakdown

```python
sarima: float   # 0.0–1.0
lgbm: float
tft: float
# sum = 1.0
```

### 13.5 ShapFeature

```python
feature_name: str
shap_value: float
feature_value: float
```

### 13.6 AttentionWeight

```python
week_offset: int    # weeks back from forecast origin
weight: float       # normalized attention weight
```

### 13.7 HistoryPoint

```python
date: date
quantity: float     # actual total_quantity (for chart)
```

### 13.8 TrainForecastingRequest

```python
drug_codes: Optional[list[str]] = None  # None = all receipt drugs
models: list[str] = ["sarima", "lgbm", "tft"]
force_retrain: bool = False
```

### 13.9 TrainStatusResponse

```python
status: str
drugs_trained: int
models_trained: list[str]
smape_summary: dict[str, float]   # per-model average sMAPE
artifacts_saved: list[str]
```

---

## 14. Configuration & Constants

All in `forecasting/constants.py`. **Do not hardcode elsewhere.**

### Feature Engineering

| Constant | Value | Purpose |
|----------|-------|---------|
| `LAG_DAYS` | `[1, 7, 28, 365]` | Lag offsets |
| `ROLLING_WINDOWS` | `[28, 91, 182]` | Rolling stat windows |
| `FORECAST_HORIZON` | 30 | Max API horizon (days) |
| `FORECAST_CHART_HISTORY_DAYS` | 90 | History returned in API |
| `PREDICTION_CAP_RECENT_DAYS` | 91 | Spike cap lookback |

### Minimum History

| Constant | Value | Model |
|----------|-------|-------|
| `MIN_HISTORY_DAYS_SARIMA` | 60 | SARIMA |
| `MIN_HISTORY_DAYS_LGBM` | 90 | LightGBM |
| `MIN_HISTORY_DAYS_TFT` | 365 | TFT |

### Training Windows

| Constant | Value | Model |
|----------|-------|-------|
| `SARIMA_TRAIN_DAYS` | 182 | SARIMA fit window |
| `LGBM_LOOKBACK_WINDOW` | 365 | LightGBM train window |
| `LOOKBACK_WINDOW_TFT` | 365 | TFT encoder max |

### SARIMA

| Constant | Value |
|----------|-------|
| `SARIMA_ORDER` | `(1, 1, 1)` |
| `SARIMA_SEASONAL_ORDER` | `(1, 1, 1, 7)` |
| `SARIMA_MAX_ITER` | 200 |
| `SARIMA_RECENT_LEVEL_DAYS` | 28 |
| `SARIMA_FORECAST_SHRINKAGE` | 0.4 |

### LightGBM

| Constant | Value |
|----------|-------|
| `LGBM_QUANTILES` | `[0.10, 0.50, 0.90]` |
| `LGBM_N_ESTIMATORS` | 500 |
| `LGBM_LEARNING_RATE` | 0.05 |
| `LGBM_MAX_DEPTH` | 6 |
| `LGBM_NUM_LEAVES` | 31 |
| `LGBM_EARLY_STOPPING_ROUNDS` | 50 |
| `LGBM_VALID_FRACTION` | 0.15 |
| `LGBM_RECURSIVE_ANCHOR_WEIGHT` | 0.25 |

### TFT

| Constant | Value |
|----------|-------|
| `TFT_HIDDEN_SIZE` | 64 |
| `TFT_ATTENTION_HEAD_SIZE` | 4 |
| `TFT_DROPOUT` | 0.1 |
| `TFT_MAX_EPOCHS` | 50 |
| `TFT_WALK_FORWARD_MAX_EPOCHS` | 5 |
| `TFT_HOLDOUT_MAX_EPOCHS` | 10 |
| `TFT_BATCH_SIZE` | 64 |
| `TFT_LEARNING_RATE` | 1e-3 |
| `TFT_QUANTILES` | `[0.05, 0.10, 0.50, 0.90, 0.95]` |
| `TFT_ENABLE_MPS` | env `TFT_ENABLE_MPS=1` |

### Censored Demand

| Constant | Value |
|----------|-------|
| `STOCKOUT_RATE_THRESHOLD` | 0.20 |
| `MIN_STOCKOUT_PERIODS` | 3 |

### Ensemble

| Constant | Value |
|----------|-------|
| `ENSEMBLE_ALPHA` | 1.0 |
| `ENSEMBLE_MAE_OUTLIER_RATIO` | 1.25 |
| `CONFORMAL_COVERAGE` | 0.90 |

### Artifacts

| Constant | Value |
|----------|-------|
| `ARTIFACTS_DIR` | `forecasting/artifacts/` (relative to package) |

### Unused Constant

`NEWSVENDOR_COST_RATIO = {"V": 10.0, "E": 4.0, "N": 1.5}` — defined but **not referenced** in the codebase.

---

## 15. Artifacts & Persistence

**Base directory:** `backend/src/app/forecasting/artifacts/`

```
artifacts/
├── sarima/
│   └── {drug_code}.pkl
├── lgbm/
│   ├── {drug_code}_p10.txt
│   ├── {drug_code}_p50.txt
│   ├── {drug_code}_p90.txt
│   ├── {drug_code}_meta.json
│   └── {drug_code}_shap.json          # optional
├── tft/
│   ├── {drug_code}.ckpt
│   └── {drug_code}_dataset.pkl
├── stacking/
│   └── stacking_meta.pkl              # global ensemble
└── conformal/
    └── mapie_wrapper.pkl              # global calibrator
```

**Per-drug artifacts:** SARIMA, LightGBM, TFT  
**Global artifacts:** Stacking meta-learner, conformal calibrator (shared across all drugs)

---

## 16. Integration with Other Systems

### 16.1 Receipt Ingestion

- **`POST /upload-receipts`** loads Excel/CSV into `drug_receipts`
- Forecasting reads receipts directly — no dependency on `daily_drug_demand` unless explicitly synced

### 16.2 Drugs Registry

- Forecasting keys off `drug_receipts.drug_code`
- Separate `drugs` table exists for registry but is not the forecasting primary key
- Drug search: `/forecasting/receipt-drug-codes/search`

### 16.3 Dashboard (Frontend)

- CORS allows `localhost:3000` / `127.0.0.1:3000`
- Uses `ForecastResponse.history` (90 days) + quantile bands for charts
- Uses `/forecasting/forecasted-drugs/search` for drugs with prior predictions
- JWT auth when `auth_enabled=True`

### 16.4 External APIs

**None.** All computation is local (PostgreSQL + filesystem artifacts). Holiday data comes from the `holidays` Python package (Lebanon), not a live API.

---

## 17. Cold Start (Separate Path)

**Module:** `backend/src/app/cold_start/`  
**Endpoints:** `/cold-start/train-embedder`, `/cold-start/train-maml`, `/cold-start/predict`

This is a **parallel forecasting path** for drugs with insufficient receipt history. It does **not** use the SARIMA/LGBM/TFT ensemble.

| Aspect | Main Ensemble | Cold Start |
|--------|---------------|------------|
| Min history | 60–365 days | <4 observations |
| Models | SARIMA + LGBM + TFT | KNN + autoencoder + optional MAML |
| Training | `ForecastingTrainer` | `cold_start_service.py` |
| Graduation | N/A | Blends to ensemble at 4–12 observations |

Shared utility: `demand_aggregation.load_recent_quantities_from_receipts()` for neighbour drug histories.

---

## 18. Dependencies

### Core (requirements.txt)

```
fastapi, uvicorn, sqlalchemy, psycopg, pandas, pydantic, pydantic-settings,
alembic, openpyxl, python-multipart
```

### ML / Forecasting (optional — runtime import with install hints)

| Package | Used For |
|---------|----------|
| `statsmodels` | SARIMA, EM-SARIMA |
| `lightgbm` | Gradient boosting quantile models |
| `pytorch`, `pytorch-forecasting`, `lightning` | TFT |
| `scikit-learn` | Ridge stacking, MAPIE estimator |
| `mapie` | Conformal intervals |
| `scipy` | Tobit optimization |
| `lifelines` | Weibull censored-demand correction |
| `holidays` | Lebanon public holidays |
| `shap` | LightGBM explainability (optional) |
| `numpy` | Throughout |

ML packages are **not pinned in requirements.txt**; missing imports raise `ImportError` with install instructions.

---

## 19. Error Handling & Edge Cases

| Condition | Behavior |
|-----------|----------|
| No receipt history | 404 on predict |
| No trained artifacts | 503 on predict |
| TFT < 365 days history | Skipped at train; ensemble reweights SARIMA+LGBM |
| Missing census/supplier rows | Default feature values used |
| MAPIE artifact missing | Fallback interval from model disagreement spread |
| Stacking artifact missing | Equal-weight nan-mean of available models |
| Batch predict failure | Per-item `error` string; other drugs still returned |
| Center filter at inference | Scopes receipt aggregation and census |
| Training | Always all-centers (`center_syn_id=None`) |
| Negative daily quantities | Converted to positive via `abs()` |
| Missing calendar days | Filled with 0, flagged `demand_filled=1` |
| Model disagreement > 50% of cap | Ensemble picks best single model for that day |
| Conformal calibrator failure | Falls back to spread-based intervals |

---

## 20. Source File Index

| File | Role |
|------|------|
| `app/api/forecasting.py` | REST API endpoints |
| `forecasting/inference/forecaster.py` | End-to-end inference orchestration |
| `forecasting/training/trainer.py` | Training orchestration |
| `forecasting/training/walk_forward.py` | Walk-forward validation |
| `forecasting/training/holdout.py` | Hold-out backtesting |
| `forecasting/training/holdout_dates.py` | Date suggestion helpers |
| `forecasting/feature_engineering/pipeline.py` | Feature matrix construction |
| `forecasting/feature_engineering/temporal_features.py` | Calendar/holiday features |
| `forecasting/feature_engineering/lag_features.py` | Lag demand features |
| `forecasting/feature_engineering/rolling_features.py` | Rolling stats + spikes |
| `forecasting/feature_engineering/external_features.py` | Hospital census features |
| `forecasting/feature_engineering/supplier_features.py` | Supplier lead-time features |
| `forecasting/feature_engineering/forecast_features.py` | Recursive LGBM inference rows |
| `forecasting/censored_demand/corrector.py` | Censored demand orchestrator |
| `forecasting/censored_demand/detector.py` | Stockout detection |
| `forecasting/censored_demand/tobit.py` | Tobit MLE correction |
| `forecasting/censored_demand/survival.py` | Weibull AFT correction |
| `forecasting/censored_demand/em_sarima.py` | EM-SARIMA imputation |
| `forecasting/models/base_model.py` | Abstract base class |
| `forecasting/models/sarima_model.py` | SARIMA implementation |
| `forecasting/models/lgbm_model.py` | LightGBM implementation |
| `forecasting/models/tft_model.py` | TFT implementation |
| `forecasting/models/feature_columns.py` | Column groupings per model |
| `forecasting/ensemble/stacking.py` | Ridge stacking meta-learner |
| `forecasting/ensemble/conformal.py` | MAPIE conformal calibration |
| `forecasting/prediction_bounds.py` | Prediction capping/winsorization |
| `forecasting/demand_quantity.py` | Consumption demand normalization |
| `forecasting/evaluation_metrics.py` | sMAPE, accuracy, evaluation mask |
| `forecasting/schemas.py` | Pydantic API schemas |
| `forecasting/constants.py` | All hyperparameters |
| `app/services/demand_aggregation.py` | Receipt → daily demand aggregation |
| `app/services/forecast_drug_lookup.py` | Forecasted drug search |
| `app/models/drug_receipt.py` | Receipt ORM model |
| `app/models/forecast_result.py` | Forecast persistence ORM |
| `app/models/model_performance.py` | Training metrics ORM |
| `app/models/hospital_census.py` | Census ORM |
| `app/models/supplier_lead_time.py` | Supplier ORM |
| `app/models/stockout_flag.py` | Stockout audit ORM |

### Test Files

```
tests/test_forecasting_step1.py .. step8
tests/test_forecasting_holdout.py
tests/test_forecasting_step8_integration.py
tests/test_demand_aggregation.py
tests/test_demand_quantity.py
```

### Scripts

```
scripts/seed_forecasting_e2e.py
scripts/validate_forecasting_step8.py
```

---

## Quick Reference: Feature Usage Matrix

| Feature Family | Count | SARIMA | LightGBM | TFT Known | TFT Unknown | TFT Static |
|----------------|-------|--------|----------|-----------|-------------|------------|
| Temporal | 14 | — | ✓ | ✓ | — | — |
| Lag | 5 | — | ✓ | — | 4 (no gap flag) | — |
| Rolling | 23 | — | ✓ | — | ✓ | — |
| External | 2 | — | ✓ | ✓ | — | — |
| Supplier | 3 | — | ✓ | — | — | ✓ |
| Pipeline extras | 4 | — | ✓ | — | — | — |
| Demand series | 1 | ✓ (target) | — (target) | — (target) | — | — |
| **Total inputs** | | **1** | **~51** | **16** | **27** | **3** |

---

*Document generated from codebase analysis. Reflects implementation as of the current repository state.*
