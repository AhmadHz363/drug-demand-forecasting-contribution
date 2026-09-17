"""Forecasting engine — shared constants.

Do not hardcode these values elsewhere; always import from here.
"""

from __future__ import annotations

import os

# ── Feature Engineering ──────────────────────────────────────────────────────
LAG_DAYS = [1, 7, 28, 365]
ROLLING_WINDOWS = [28, 91, 182]
MIN_HISTORY_DAYS_SARIMA = 60
SARIMA_TRAIN_DAYS = 182
SARIMA_TRAIN_DAYS_SHORT = 90
MIN_HISTORY_DAYS_LGBM = 90
PREDICTION_CAP_RECENT_DAYS = 91
MIN_HISTORY_DAYS_TFT = 365
MIN_HISTORY_DAYS_TFT_SHORT = 180
FORECAST_HORIZON = 30
FORECAST_CHART_HISTORY_DAYS = 90
LOOKBACK_WINDOW_TFT = 365
LOOKBACK_WINDOW_TFT_SHORT = 180
LGBM_LOOKBACK_WINDOW = 365

# ── Censored Demand ───────────────────────────────────────────────────────────
STOCKOUT_RATE_THRESHOLD = 0.20
MIN_STOCKOUT_PERIODS = 3

# ── SARIMA ────────────────────────────────────────────────────────────────────
SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 7)
SARIMA_MAX_ITER = 200
SARIMA_RECENT_LEVEL_DAYS = 28
SARIMA_FORECAST_SHRINKAGE = 0.4

# ── LightGBM ─────────────────────────────────────────────────────────────────
LGBM_QUANTILES = [0.10, 0.50, 0.90]
LGBM_N_ESTIMATORS = 500
LGBM_LEARNING_RATE = 0.05
LGBM_MAX_DEPTH = 6
LGBM_NUM_LEAVES = 31
LGBM_EARLY_STOPPING_ROUNDS = 50
LGBM_VALID_FRACTION = 0.15
LGBM_RECURSIVE_ANCHOR_WEIGHT = 0.25

# ── TFT ───────────────────────────────────────────────────────────────────────
TFT_HIDDEN_SIZE = 64
TFT_ATTENTION_HEAD_SIZE = 4
TFT_DROPOUT = 0.1
TFT_HIDDEN_CONTINUOUS_SIZE = 16
TFT_MAX_EPOCHS = 50
TFT_HOLDOUT_MAX_EPOCHS = 10
TFT_BATCH_SIZE = 64
TFT_LEARNING_RATE = 1e-3
TFT_GRADIENT_CLIP_VAL = 0.1
TFT_QUANTILES = [0.05, 0.10, 0.50, 0.90, 0.95]
TFT_WALK_FORWARD_MAX_EPOCHS = 5
TFT_WALK_FORWARD_EPOCH_RATIO = 0.20
# pytorch-forecasting + MPS can hit uncatchable native buffer assertions on Apple
# Silicon that kill the API process.  MPS is opt-in only via TFT_ENABLE_MPS=1.
TFT_ENABLE_MPS = os.environ.get("TFT_ENABLE_MPS", "").lower() in {"1", "true", "yes"}
TFT_MAX_TRAIN_DAYS = LOOKBACK_WINDOW_TFT + FORECAST_HORIZON + 14

# ── Demand segmentation (Syntetos–Boylan) ─────────────────────────────────────
ADI_THRESHOLD = 1.32
CV2_THRESHOLD = 0.49
DEMAND_SEGMENTS = ("smooth", "intermittent", "erratic", "lumpy")
GLOBAL_DEMAND_SEGMENT = "global"
SEGMENT_ENSEMBLE_MIN_SAMPLES = 20

# ── Walk-forward validation ───────────────────────────────────────────────────
WALK_FORWARD_N_SPLITS = 5
WALK_FORWARD_TEST_HORIZON = 7
SEASONAL_NAIVE_PERIOD = 7

# ── Ensemble ──────────────────────────────────────────────────────────────────
ENSEMBLE_ALPHA = 1.0
ENSEMBLE_MAE_WEIGHT_POWER = 1.0
ENSEMBLE_MIN_MODEL_WEIGHT = 0.05
STACKING_DISAGREEMENT_CAP_RATIO = 0.5
CONFORMAL_COVERAGE = 0.90

# ── Forecast sanity checks ────────────────────────────────────────────────────
INTERVAL_WIDTH_SHRINK_TOLERANCE = 0.5
INTERVAL_P10_CLIFF_P50_THRESHOLD = 1.0

# ── Data quality gating (Phase 3) ─────────────────────────────────────────────
DATA_QUALITY_MIN_HISTORY_DAYS = 60
DATA_QUALITY_MAX_EXTERNAL_DEFAULT_RATE = 0.50
DATA_QUALITY_MAX_SUPPLIER_DEFAULT_RATE = 0.50
DATA_QUALITY_REJECT_DEGENERATE_IMPUTATION = True

# ── Drift detection (Phase 3) ─────────────────────────────────────────────────
DRIFT_SMAPE_DEGRADATION_PCT = 15.0
DRIFT_MASE_DEGRADATION_PCT = 15.0

# ── Newsvendor ────────────────────────────────────────────────────────────────
NEWSVENDOR_COST_RATIO = {"V": 10.0, "E": 4.0, "N": 1.5}

# ── Artifacts ─────────────────────────────────────────────────────────────────
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
