"""Cold Start module — shared constants.

Do not hardcode these values elsewhere; always import from here.
"""

from __future__ import annotations

import os

# Autoencoder
METADATA_INPUT_DIM = 12
EMBEDDING_DIM = 32
AUTOENCODER_HIDDEN_DIM = 64
AUTOENCODER_EPOCHS = 300
AUTOENCODER_LR = 1e-3
AUTOENCODER_BATCH_SIZE = 16

# KNN Bootstrap
KNN_K = 5
# Scan beyond top-K similarity when nearest drugs lack demand history (large sparse catalogs).
KNN_MAX_CANDIDATES = 200
PHARMACIST_ESTIMATE_WEIGHT = 0.20
SIMILARITY_MIN_THRESHOLD = 0.0

# MAML
MAML_INNER_LR = 0.01
MAML_OUTER_LR = 1e-3
MAML_INNER_STEPS = 5
MAML_EPOCHS = 200
MAML_META_BATCH_SIZE = 8
MAML_SUPPORT_SIZE = 4
MAML_QUERY_SIZE = 8
MAML_HIDDEN_DIM = 32
MAML_SEQUENCE_LEN = 14
MAML_FORECAST_HORIZON = 7

# CAMEO (notebook defaults + accuracy tuning for intermittent launch demand)
CAMEO_TOPK = 5
CAMEO_EPOCHS = 800
CAMEO_SEED = 1
CAMEO_DRIFT_HAZARD = 1 / 25
CAMEO_DRIFT_THRESHOLD = 0.6
CAMEO_ONLINE_TAU = 1.5
CAMEO_CONFORMAL_ALPHA = 0.1
CAMEO_LAUNCH_WEEKS = 8
CAMEO_SHAPE_RERANK_WEIGHT = 0.5
CAMEO_ANALOG_CANDIDATE_MULT = 3

# Graduation thresholds (daily observations from enriched demand panel)
COLD_START_ONLY_BELOW = 4
BLEND_UNTIL = 12

# Metadata encoding bounds
SHELF_LIFE_MIN_DAYS = 30
SHELF_LIFE_MAX_DAYS = 1825
UNIT_PRICE_TIER_MIN = 1
UNIT_PRICE_TIER_MAX = 5

# Artifacts directory (relative to this package)
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
