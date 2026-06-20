# Forecasting Features

This document describes the features used to train and generate drug demand forecasts in the backend forecasting engine.

The pipeline lives under `backend/src/app/forecasting/`. All models share a common feature matrix built by `build_feature_matrix()` in `feature_engineering/pipeline.py`. Each model then consumes a different subset of those columns.

---

## Overview

```
drug_receipts (daily demand)
        │
        ▼
build_feature_matrix()
  ├── temporal features   (14)
  ├── lag features        (5)
  ├── rolling features    (17)
  ├── external features   (2)
  └── supplier features   (3)
        │
        ▼
correct_demand()  ──► adds em_corrected_quantity (SARIMA target only)
        │
        ├─► SARIMA   — univariate demand series
        ├─► LightGBM — all engineered features (~37)
        ├─► TFT      — split into known / unknown / static groups
        └─► Ensemble — blends SARIMA + LGBM + TFT P50 predictions
```

**Total:** ~41 columns (target + features). The target column is `total_quantity` (daily consumption demand aggregated from receipts).

---

## Feature pipeline

### Entry point

`build_feature_matrix(drug_code, center_syn_id, db_session, start_date, end_date)`:

1. Loads daily demand from `drug_receipts` via `aggregate_daily_demand_rows()`.
2. Reindexes to a complete daily calendar (missing days filled with 0).
3. Applies consumption demand normalization (`apply_consumption_demand`).
4. Engineers all feature families in order: temporal → lag → rolling → external → supplier.
5. Fills remaining nulls with 0 and sets `demand_date` as the index.

Relevant constants (`forecasting/constants.py`):

| Constant | Value | Used by |
|---|---|---|
| `LAG_DAYS` | `[1, 7, 28, 365]` | Lag features |
| `ROLLING_WINDOWS` | `[28, 91, 182]` | Rolling features |
| `MIN_HISTORY_DAYS_SARIMA` | 60 | SARIMA minimum history |
| `MIN_HISTORY_DAYS_LGBM` | 90 | LightGBM minimum history |
| `MIN_HISTORY_DAYS_TFT` | 365 | TFT minimum history |
| `FORECAST_HORIZON` | 30 | Default forecast length (days) |
| `LOOKBACK_WINDOW_TFT` | 365 | TFT encoder window |
| `LGBM_LOOKBACK_WINDOW` | 182 | LightGBM training window |

---

## Feature families

### 1. Temporal features (14 columns)

**Source:** `feature_engineering/temporal_features.py`  
**Function:** `add_temporal_features()`

Calendar and holiday features derived from `demand_date`. Cyclical encoding uses sin/cos pairs to preserve periodicity.

| Column | Description |
|---|---|
| `day_of_week` | Integer 0–6 (Monday = 0) |
| `day_of_week_sin`, `day_of_week_cos` | Cyclical encoding (period = 7) |
| `week_of_year` | ISO week number |
| `week_of_year_sin`, `week_of_year_cos` | Cyclical encoding (period = 53) |
| `month` | Calendar month (1–12) |
| `month_sin`, `month_cos` | Cyclical encoding (period = 12) |
| `quarter` | Calendar quarter (1–4) |
| `is_weekend` | 1 if Saturday or Sunday, else 0 |
| `is_public_holiday` | 1 if the date is a Lebanon public holiday |
| `days_to_next_holiday` | Days until the next holiday (capped at 30) |
| `days_since_last_holiday` | Days since the last holiday (capped at 30) |

Holidays are loaded from the `holidays` library for Lebanon (`holidays.Lebanon`, years 2018–2029).

---

### 2. Lag features (5 columns)

**Source:** `feature_engineering/lag_features.py`  
**Function:** `add_lag_features()`

Past demand at fixed offsets. Requires `total_quantity`.

| Column | Description |
|---|---|
| `lag_1d` | Demand 1 day ago |
| `lag_7d` | Demand 7 days ago |
| `lag_28d` | Demand 28 days ago |
| `lag_365d` | Demand 365 days ago |
| `has_lag_gaps` | 1 if any lag was imputed (history too short), else 0 |

When history is shorter than a lag offset, the missing value is filled with the **median** of `total_quantity`. `has_lag_gaps` flags rows where imputation occurred.

---

### 3. Rolling features (17 columns)

**Source:** `feature_engineering/rolling_features.py`  
**Function:** `add_rolling_features()`

Rolling statistics over preceding demand only (shifted by 1 day to avoid same-day leakage). Windows: **28, 91, and 182** days.

For each window `{28, 91, 182}`:

| Column | Description |
|---|---|
| `rolling_mean_{window}d` | Rolling mean |
| `rolling_std_{window}d` | Rolling standard deviation |
| `rolling_min_{window}d` | Rolling minimum |
| `rolling_max_{window}d` | Rolling maximum |
| `rolling_cv_{window}d` | Coefficient of variation (std / mean) |

Additional columns:

| Column | Description |
|---|---|
| `trend_slope_28d` | Linear slope over the preceding 28 days |
| `demand_zscore_28d` | Z-score of current demand vs. 28-day rolling mean/std |

Rolling stats require at least 7 observations (`min_periods=7`); otherwise values are filled with the series median or 0.

---

### 4. External features (2 columns)

**Source:** `feature_engineering/external_features.py`  
**Function:** `add_external_features()`  
**Data source:** `hospital_census` table

| Column | Description | Default if missing |
|---|---|---|
| `bed_occupancy_rate` | Hospital bed occupancy rate | 0.75 |
| `weekly_surgery_count` | Weekly surgery volume | 50 |

When `center_syn_id` is provided, census data is filtered to that center. When it is `None`, values are averaged across all centers for each date.

---

### 5. Supplier features (3 columns)

**Source:** `feature_engineering/supplier_features.py`  
**Function:** `add_supplier_features()`  
**Data source:** `supplier_lead_times` table (per drug)

These are **constant per drug** — the same values are repeated for every row in the series.

| Column | Description | Default if missing |
|---|---|---|
| `supplier_avg_lead_time` | Average supplier lead time (days) | 3.0 |
| `supplier_lead_time_std` | Lead time standard deviation (days) | 1.0 |
| `supplier_reliability_score` | Supplier reliability score (0–1) | 0.85 |

---

## Censored demand correction (pre-model)

Before training/inference, `correct_demand()` may run to handle stockout periods where observed demand understates true demand.

It adds metadata columns that are **not** used as LightGBM/TFT inputs:

| Column | Purpose |
|---|---|
| `em_corrected_quantity` | EM-corrected demand — used as the **SARIMA target** when present |
| `correction_method` | Which correction method was applied |
| `is_stockout` | Whether the day was flagged as a stockout |

---

## Model-specific feature usage

### SARIMA

**File:** `models/sarima_model.py`

SARIMA is **univariate** — it does not use any engineered feature columns.

| Input | Description |
|---|---|
| Target series | `em_corrected_quantity` if available, otherwise `total_quantity` |

Model configuration:

- Order: `(1, 1, 1)`
- Seasonal order: `(1, 1, 1, 7)` — weekly seasonality
- Training window: last 182 days (`SARIMA_TRAIN_DAYS`)
- Minimum history: 60 days

---

### LightGBM

**File:** `models/lgbm_model.py`

Uses **all columns except metadata**:

```python
METADATA_COLUMNS = {
    "total_quantity",
    "em_corrected_quantity",
    "correction_method",
    "is_stockout",
    "demand_date",
}
```

That yields approximately **37 features**: 14 temporal + 5 lag + 17 rolling + 2 external + 3 supplier.

| Setting | Value |
|---|---|
| Categorical feature | `has_lag_gaps` |
| Quantiles predicted | 0.10, 0.50, 0.90 |
| Minimum history | 90 days |
| Lookback window | 182 days |

#### Multi-step forecasting

For future days, lag and rolling features cannot be read from history directly. The model uses **recursive forecasting**:

1. `build_future_covariates()` pre-computes known future values: temporal, external, and supplier features.
2. `build_recursive_feature_row()` rebuilds lag and rolling features from an updated demand history that includes previously predicted quantities.
3. Each step's P50 prediction is appended to the history before predicting the next day.

---

### TFT (Temporal Fusion Transformer)

**File:** `models/tft_model.py`  
**Feature mapping:** `models/feature_columns.py`

TFT splits features into three groups for the PyTorch Forecasting `TimeSeriesDataSet`:

| Group | Columns | Count | Known at forecast time? |
|---|---|---|---|
| `time_varying_known_reals` | Temporal + external | 16 | Yes |
| `time_varying_unknown_reals` | Lags (excluding `has_lag_gaps`) + rolling | 21 | No — must be inferred |
| `static_reals` | Supplier | 3 | Yes (constant per drug) |
| **Target** | `total_quantity` | 1 | — |

| Setting | Value |
|---|---|
| Encoder length | up to 365 days (`LOOKBACK_WINDOW_TFT`) |
| Prediction length | up to 30 days (`FORECAST_HORIZON`) |
| Minimum history | 365 days |
| Quantiles predicted | 0.10, 0.50, 0.90 |

Note: `has_lag_gaps` is used by LightGBM but **excluded** from TFT.

For horizons longer than one TFT chunk, the model extends history by appending predicted P50 values and rebuilding lag/rolling features via `add_temporal_features()`, `add_lag_features()`, and `add_rolling_features()`.

---

### Ensemble (stacking)

**File:** `ensemble/stacking.py`

The Ridge meta-learner does **not** use raw features. It blends the three base models using their **P50 predictions** as input:

| Input feature | Source |
|---|---|
| SARIMA P50 | `SarimaModel.predict()` |
| LightGBM P50 | `LightGBMModel.predict()` |
| TFT P50 | `TFTModel.predict()` |

Blending weights are derived from inverse validation MAE per model. A conformal calibrator (`ensemble/conformal.py`) adjusts prediction intervals on top of the blended forecast.

---

## Inference-time explainability

When requested via the forecast API, two optional explainability outputs are available:

| Flag | Model | Output |
|---|---|---|
| `include_shap` | LightGBM | `ShapFeature` list — mean absolute SHAP values per feature |
| `include_attention` | TFT | `AttentionWeight` list — temporal attention weights |

These surface which of the features described above drove a given forecast.

---

## Quick reference

| Feature family | Count | SARIMA | LightGBM | TFT known | TFT unknown | TFT static |
|---|---|---|---|---|---|---|
| Temporal | 14 | — | ✓ | ✓ | — | — |
| Lag | 5 | — | ✓ | — | 4 (no gap flag) | — |
| Rolling | 17 | — | ✓ | — | ✓ | — |
| External | 2 | — | ✓ | ✓ | — | — |
| Supplier | 3 | — | ✓ | — | — | ✓ |
| Demand series | 1 | ✓ (target) | — | — | — | — |
| **Total inputs** | | **1** | **~37** | **16** | **21** | **3** |

---

## Key source files

| File | Role |
|---|---|
| `feature_engineering/pipeline.py` | Orchestrates feature matrix construction |
| `feature_engineering/temporal_features.py` | Calendar and holiday features |
| `feature_engineering/lag_features.py` | Lag demand features |
| `feature_engineering/rolling_features.py` | Rolling statistics and trend |
| `feature_engineering/external_features.py` | Hospital census features |
| `feature_engineering/supplier_features.py` | Supplier lead-time features |
| `feature_engineering/forecast_features.py` | Recursive feature rows for LGBM inference |
| `models/feature_columns.py` | Shared column definitions per model |
| `models/sarima_model.py` | SARIMA model |
| `models/lgbm_model.py` | LightGBM model |
| `models/tft_model.py` | TFT model |
| `inference/forecaster.py` | End-to-end forecast orchestration |
| `constants.py` | Shared hyperparameters and window sizes |
