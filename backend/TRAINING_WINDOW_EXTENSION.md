# Training Window Extension - 2 Year Lookback

Date: July 22, 2026

## Changes Made

Extended all model training windows to use **full 2 years** of available data (2023-2024) instead of being limited to shorter periods.

### Before vs After

| Component | Before | After | Impact |
|-----------|--------|-------|--------|
| **LGBM Lookback** | 365 days (1 year) | **730 days (2 years)** | Uses all available history |
| **SARIMA Train Days** | 182 days (6 months) | **730 days (2 years)** | 4x more training data |
| **SARIMA Train Short** | 90 days (3 months) | **365 days (1 year)** | For intermittent series |
| **TFT Lookback** | 365 days | **730 days (2 years)** | If TFT is enabled |
| **Lag Features** | [1, 7, 28, 365] | **[1, 7, 28, 365, 730]** | Added 2-year lag |
| **Rolling Windows** | [28, 91, 182] | **[28, 91, 182, 365]** | Added 1-year rolling stats |

## Expected Benefits

### 1. More Training Data for Models
- **SARIMA**: Now uses up to 2 years instead of 6 months
  - Better seasonal pattern detection
  - More stable parameter estimates
  - Multi-year trends captured

- **LightGBM**: Now uses up to 2 years instead of 1 year
  - More examples for training
  - Better handling of rare events
  - Improved pattern recognition

### 2. New Features Available
- **`lag_730d`**: 2-year-ago demand (captures annual cycles)
- **`rolling_mean_365d`**: Full-year rolling average
- **`rolling_std_365d`**: Full-year volatility
- **`rolling_cv_365d`**: Annual coefficient of variation

### 3. Better for Sparse/Intermittent Drugs
- More nonzero demand events to learn from
- More reliable TSB/Croston smoothing parameters
- Better baseline estimates for MASE calculation

### 4. Improved Validation
- Walk-forward validation has more folds possible
- Champion selection more reliable
- Drift detection more accurate

## Expected Impact on Accuracy

### Drugs Most Likely to Improve

| Drug Type | Expected Improvement | Reason |
|-----------|---------------------|--------|
| **Smooth, seasonal** | 5-15% sMAPE reduction | Better seasonal pattern capture |
| **Sparse/intermittent** | 3-8% MASE reduction | More nonzero events to train on |
| **Recently added** | May now become trainable | If they now have ≥90 days |
| **High volatility** | 2-5% reduction | Better volatility estimates |

### Drugs Less Likely to Improve

- Very erratic/random demand (no patterns to extract)
- Drugs with regime changes (old data may confuse)
- Drugs with poor data quality in early years

## What to Do Next

### 1. Retrain All Models (Required)
Current trained models still use old short windows. You must retrain to see benefits:

```python
# Retrain all trainable drugs with new windows
trainer.train_all(
    db_session=session,
    drug_codes=list_of_all_trainable_drugs,  # 472 drugs
    models_to_train=["sarima", "lgbm", "classical"],
    force_retrain=True
)
```

### 2. Monitor Impact
After retraining, compare:
- Before: `backend/docs/baselines/forecast_accuracy_baseline_2026-07-13.json`
- After: New accuracy measurements

### 3. Adjust If Needed
If some drugs get **worse**:
- May indicate regime change (formulary/protocol shift)
- Consider per-drug lookback tuning
- Or trim to most recent N days for those drugs

## Potential Issues to Watch

### 1. Training Time
- **LGBM**: 2x more data = ~1.5-2x training time
- **SARIMA**: 4x more data = ~2-3x training time
- Total training for 472 drugs may take 2-4 hours (was ~1-2 hours)

### 2. Memory Usage
- Larger feature matrices
- More OOF predictions stored
- May need more RAM for batch training

### 3. Feature Quality
New features depend on data quality:
- `lag_730d` only useful if 2023 data is clean
- `rolling_mean_365d` affected by any coverage gaps
- Check coverage_gaps table for 2023

### 4. Overfitting Risk
- SARIMA with 730 days may overfit smooth series
- LightGBM regularization may need adjustment
- Monitor validation metrics closely

## Rollback Plan

If accuracy degrades after retraining, revert by:

```python
# In constants.py
LGBM_LOOKBACK_WINDOW = 365  # Back to 1 year
SARIMA_TRAIN_DAYS = 182     # Back to 6 months
LAG_DAYS = [1, 7, 28, 365]  # Remove 730
ROLLING_WINDOWS = [28, 91, 182]  # Remove 365
```

Then retrain again.

## Testing Before Full Rollout

Recommended to test on a subset first:

```python
# Test on top 5 drugs that already perform well
test_drugs = ["P591934", "P326266", "P488886", "P753273", "P304745"]

# Train with new windows
trainer.train_all(
    db_session=session,
    drug_codes=test_drugs,
    models_to_train=["sarima", "lgbm", "classical"],
    force_retrain=True
)

# Compare accuracy before/after
# If improved or same → proceed to all 472 drugs
# If worse → investigate and possibly adjust windows
```

## Code Files Modified

1. `src/app/forecasting/constants.py`:
   - Lines 11-25: Extended lookback windows
   - Added lag_730d and rolling_365d

No other code changes needed - the pipeline automatically uses these constants.
