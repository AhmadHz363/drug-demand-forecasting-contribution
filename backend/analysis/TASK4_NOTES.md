# Task 4 — TFT Transparency

## What was added

**`Chapters/Chapter_3/Chapter3.tex`** — new `\subsubsection{TFT training configuration}`
after the "Baseline families" paragraph, reporting the exact configuration pulled from
`backend/src/app/forecasting/constants.py` and `models/tft_model.py`:

- Architecture: hidden size 64, attention head size 4, hidden continuous size 16,
  dropout 0.1.
- Optimizer: Adam, learning rate $1\times10^{-3}$, gradient clipping 0.1.
- Encoder window: up to 730 days (365 for shorter series); covariates = calendar
  features + lag/rolling features as time-varying, quantile loss over
  {0.05, 0.10, 0.50, 0.90, 0.95}.
- **Training budget: fixed 50 epochs, no early stopping**, CPU-only (Apple MPS
  disabled — it crashes pytorch-forecasting's native kernels on this hardware; no
  CUDA available).
- Explicit contrast: every LightGBM model in the thesis stops on a validation metric
  with 40-50 round patience inside a 700-900 round budget, and a boosting round is far
  cheaper than a TFT epoch.
- The measured cost (~55s/epoch, ~45 min/drug for the full 50-epoch run, from Task 1's
  probes) is cited as the concrete, evidence-backed reason a systematic hyperparameter
  search or a full stratified TFT rerun was infeasible at this stage — turning "TFT
  underperforms" into a scoped, defensible statement rather than an implied general
  result about attention architectures.

**`Chapters/Chapter_5/Chapter5.tex`** — one sentence added to the RQ1 discussion
paragraph making the same tuning-budget asymmetry explicit at the point where the
SARIMA/LightGBM/TFT ranking is actually discussed, so a reader doesn't have to jump to
Chapter 3 to find the caveat.

**`Chapters/Chapter_6/Chapter6.tex`** — two edits:
1. The "methods never trained" limitations paragraph now also notes TFT's fixed
   CPU-only budget and points back to Chapter 3 for the full configuration.
2. The future-work "larger, multi-hospital panel" bullet now explicitly names "a
   properly tuned, GPU-trained TFT run on a search budget comparable to what LightGBM
   got here" as an open extension.

Recompiled clean: no errors, no overfull/underfull-hbox regressions.

## Task 4 status: CLOSED
