# Accuracy Improvements Applied - July 22, 2026

## Summary of Changes

Target: Reduce forecast error to <30% sMAPE with minimal swinging/divergence.

### 1. Fixed Critical Bugs ✅

#### A. Ensemble Metrics Not Persisted
**Problem**: Trainer only saved individual model (SARIMA/LGBM/Classical) metrics. Ensemble predictions existed but metrics were never computed or stored.

**Fix**: 
- Added ensemble metric computation in `trainer.py` after champion selection
- Now persists `ModelPerformance` records with `model_name='ensemble'`
- Includes sMAPE, MASE, and rolling validation metrics

**Impact**: Can now track and compare ensemble performance against base models.

#### B. Import Name Conflicts
**Problem**: `smape` and `mase` function imports conflicted with variables in trainer loop.

**Fix**: 
- Renamed imports to `smape_single` and `compute_mase`
- Prevents shadowing by local variables

### 2. Hyperparameter Tuning ✅

#### A. Classical Model (TSB/Croston) - Intermittent/Lumpy Demand
**Before**: Alpha = 0.1 (very slow adaptation)
**After**: Alpha = 0.15 (faster response to demand changes)

**Rationale**: Intermittent/lumpy series benefit from slightly faster adaptation while still smoothing noise.

#### B. SARIMA Shrinkage - Smooth Demand
**Before**: 0.35 (heavy dampening)
**After**: 0.28 (more responsive to trends)

**Rationale**: Reduced shrinkage allows SARIMA to capture trends better for smooth series without over-extrapolating.

#### C. LightGBM Regularization - Sparse Series
**Before**: 
- Intermittent/lumpy: depth=4, leaves=15, n_est=300
- Erratic: depth=5, leaves=23, n_est=400

**After**:
- Intermittent/lumpy: depth=5, leaves=20, n_est=400
- Erratic: depth=5, leaves=26, n_est=450

**Rationale**: Slightly relaxed regularization allows capturing more patterns in sparse data while maintaining generalization.

### 3. Feature Engineering Validation ✅

Verified the following are properly implemented:
- **Lag features**: [1, 7, 28, 365] days with gap handling
- **Rolling windows**: [28, 91, 182] days for trend/volatility
- **Coverage gap handling**: Lags reset properly across gaps
- **Spike detection**: Multi-window spike counting and intensity

**No changes needed** - feature engineering is robust.

### 4. Round 5 Fixes (Already Applied)

These were applied in previous session:
- sMAPE reliability detection for sparse series
- MASE as primary metric for intermittent/lumpy segments
- Stacking drift dampening for long horizons

## Test Results

### P113114 (Intermittent)
| Model | sMAPE | MASE | Notes |
|-------|-------|------|-------|
| SARIMA | 154.78% | 1.073 | Sparse series, sMAPE unreliable |
| LightGBM | 163.31% | 1.106 | |
| Classical | 144.32% | 1.095 | |
| **Ensemble** | **175.06%** | **10.662** | ⚠️ MASE very high, needs investigation |

### P406816 (Intermittent) 
| Model | sMAPE | MASE | Notes |
|-------|-------|------|-------|
| SARIMA | 49.14% | 0.897 | Best base model |
| LightGBM | 82.59% | 1.522 | |
| Classical | 51.88% | 1.096 | |
| **Ensemble** | **41.69%** | **0.498** | ✅ **Ensemble improves!** |

## Current Status

### What's Working ✅
1. Ensemble metrics now tracked
2. P406816 ensemble achieves 41.69% sMAPE (below 50% threshold)
3. P406816 MASE = 0.498 (beats naive baseline significantly)
4. Training pipeline runs successfully with improved hyperparameters

### Outstanding Issues ⚠️

1. **P113114 Ensemble MASE = 10.662**
   - Base models: 1.07-1.11
   - Ensemble: 10.662 (10x worse!)
   - Likely bug in ensemble OOF prediction or metric computation
   - Need to investigate stacking weights for this drug

2. **Overall Accuracy Still > 30%**
   - System-wide average: ~54% sMAPE
   - Only 10-20% of drugs below 30% target
   - Need to expand training to more drugs with new hyperparameters

3. **Missing External Features**
   - `hospital_census` table empty (100% default-filled)
   - `supplier_lead_times` missing for both drugs
   - May limit forecast quality

## Next Steps

### Immediate (P0)
1. **Debug P113114 ensemble MASE**
   - Check stacking weights
   - Verify OOF predictions aren't corrupted
   - May need segment-specific ensemble tuning

2. **Retrain top 20 drugs** with new hyperparameters
   - Focus on drugs with >100 days history
   - Measure improvement vs baseline

3. **Fix conformal intervals** for P113114 (p10=0 issue from round 5)

### Short-term (P1)
1. Populate external features (hospital census, supplier lead times)
2. Expand training to all 472 trainable drugs
3. Set segment-specific accuracy targets:
   - Smooth: <25% sMAPE
   - Intermittent/Lumpy: <1.0 MASE (ignore sMAPE)
   - Erratic: <35% sMAPE

### Medium-term (P2)
1. Implement per-drug hyperparameter tuning
2. Add automated retraining triggers for drift
3. Build accuracy monitoring dashboard
