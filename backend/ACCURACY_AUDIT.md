# Forecasting Accuracy Audit - July 22, 2026

## Current State

### System Overview
- **Total drugs**: 968
- **Trainable (≥90 days)**: 472 drugs
- **Actually trained**: 226 drugs
- **Models**: SARIMA, LightGBM, Classical

### Critical Findings

#### 1. **ACCURACY FAR BELOW TARGET**
Target: <30% sMAPE  
Current: **~54% average sMAPE**

| Model | Drugs | Avg sMAPE | Avg MASE | >30% Error | >50% Error |
|-------|-------|-----------|----------|------------|------------|
| SARIMA | 226 | 53.09% | 1.045 | 203 (90%) | 124 (55%) |
| LightGBM | 226 | 54.73% | 1.094 | 215 (95%) | 131 (58%) |
| Classical | 222 | 54.62% | 1.084 | 204 (92%) | 131 (59%) |

**MASE ~1.0-1.1** means models are barely beating naive weekly seasonal baseline.

#### 2. **NO ENSEMBLE METRICS PERSISTED**
- Trainer only saves individual model metrics (sarima/lgbm/classical)
- `model_name='ensemble'` records are NEVER created
- Inference uses ensemble, but we don't track its validation performance
- **BUG**: Ensemble OOF predictions are collected but never evaluated/stored

#### 3. **SPARSE SERIES DOMINATE WORST PERFORMERS**
Top 15 worst drugs are mostly **lumpy** or **intermittent**:
- P000089 (lumpy): 184% sMAPE but 0.64 MASE (sMAPE broken)
- P000154 (lumpy): 167-175% sMAPE, 0.69-0.88 MASE
- P113114 (intermittent): 144-163% sMAPE, 1.07-1.11 MASE

These drugs have high zero-day proportions (>40%) where sMAPE inflates on zero-actual/nonzero-forecast pairs.

#### 4. **ONLY 2 DRUGS IN SAMPLE OUTPUTS**
User provided P113114 and P406816 outputs - these are NOT representative:
- Most trained drugs performing much worse (54% avg vs their 49-155%)
- Need to check top performers for patterns

## Root Causes

### A. Metric Issues
1. **sMAPE unreliable on sparse series** (already partially addressed in round 5)
2. **Ensemble metrics not tracked** - can't compare ensemble vs base models
3. **No per-drug accuracy targeting** - treating all drugs the same

### B. Model Issues
1. **MASE ~1.0-1.1** = models barely better than naive
2. **Segmentation may be wrong** for many drugs
3. **Feature engineering** may not capture patterns
4. **Hyperparameters** not tuned per segment
5. **Stacking not helping** - need to verify it's being used and calibrated

### C. Data Issues
1. **High sparsity**: Top drugs by history have 0.2-8% zeros (good), but many trained drugs likely have higher
2. **Coverage gaps** not fully analyzed
3. **Training window selection** may be suboptimal
4. **Censored demand correction** may introduce error

## Action Plan

### Phase 1: Fix Critical Bugs (P0)
1. **Add ensemble metric persistence** in trainer
2. **Verify stacking is actually being used** at inference for trained drugs
3. **Fix conformal calibration** for sparse series (P113114 p10=0 issue)

### Phase 2: Improve Model Accuracy (P0)
1. **Retrain top 50 drugs** with audit fixes
2. **Tune hyperparameters per segment**:
   - Lumpy/intermittent: Stronger TSB/Croston weighting
   - Smooth: More aggressive SARIMA
   - Erratic: Better LGBM regularization
3. **Add segment-specific champion selection logic**
4. **Improve feature engineering**:
   - Better lag selection for sparse series
   - External feature quality check
   - Rolling feature window tuning

### Phase 3: Optimize Training Pipeline (P1)
1. **Expand training to all 472 trainable drugs**
2. **Implement cross-validation tuning** per drug
3. **Add early stopping** for poor performers
4. **Segment reclassification** based on recent data

### Phase 4: Validation & Monitoring (P1)
1. **Set per-segment accuracy targets**:
   - Smooth: <25% sMAPE
   - Intermittent/lumpy: <1.0 MASE (use MASE not sMAPE)
   - Erratic: <35% sMAPE
2. **Add ensemble performance dashboard**
3. **Automated retraining triggers** when accuracy degrades

## Next Steps

Immediate actions:
1. Fix ensemble metric persistence bug
2. Retrain P113114 and P406816 with proper segmentation/hyperparams
3. Expand to top 20 drugs and measure improvement
4. If improvement <10%, investigate feature engineering deeper
