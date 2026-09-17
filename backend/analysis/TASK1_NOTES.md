# Task 1 — Statistical significance testing

Status: **CAMEO done. SARIMA-vs-LightGBM done with a fresh real rerun. TFT blocked by real compute cost, not missing data this time — see "TFT" section at the bottom for what I measured and the decision I need from you.**

## What I found before running anything

I searched the whole repo, `~/Downloads`, and `~/Desktop` for the actual per-SKU /
per-drug arrays behind the thesis's headline numbers, rather than trusting the
aggregates already printed in the chapters. Result:

| Comparison in the thesis | Per-item paired data found? | Source |
|---|---|---|
| CAMEO vs Analogous vs DDPFF-style vs ARIMA (Table 5.3/5.4) | **Yes** | `~/Downloads/cameo_real_per_drug_results.csv` — 53 drugs × 4 methods × MAE/MAPE/MASE |
| SARIMA vs LightGBM vs TFT (Table 5.1) | **No** | Not found anywhere accessible |

I verified the CAMEO file is the real source (not a stale draft) by recomputing its
aggregates and matching them exactly against what Chapter 5 already reports: mean MAE
89.06/100.39/103.69/104.82 for CAMEO/Analogous/ARIMA/DDPFF-style, and win rates
28/53=52.8%, 9/53=17.0%, 4/53=7.5%, 12/53=22.6%. Bit-for-bit match. Two duplicate
copies of the file in Downloads are identical, so there's no ambiguity about which run
is "the" run.

For the SARIMA/LightGBM/TFT comparison, I traced the code path: the production system
(`backend/src/app/forecasting`) writes per-drug, per-model SMAPE into a database table
(`model_name`, `smape` columns, per `docs/forecasting-engine-complete-analysis.md`), not
to a flat file. No local Postgres/SQLite file with that table exists in this workspace,
and no CSV/notebook export of it exists in Downloads or Desktop either. The
`63.6% / 59.8% / 93.5%` numbers in Table 5.1 are real aggregates from a real run (they
don't match any of the other ad-hoc test logs I found, e.g. `TWO_YEAR_TRAINING_RESULTS.md`
covers only 2 drugs and is a different experiment), but the per-SKU array behind that
mean no longer exists anywhere I can reach.

**I am not going to invent that per-SKU array or approximate it from the two unrelated
2-drug logs I did find** — that would be exactly the kind of fabrication you told me not
to do. Options, your call:

1. **Re-run the baseline comparison from scratch** against the audited panel
   (`hospital_daily_demand_enriched_2023_2024.xlsx` type files are in Downloads) using
   the actual `SarimaModel`/LightGBM/`TFTModel` code in `backend/src/app/forecasting`.
   This is real work, not a quick script — TFT alone trains one model per SKU (the repo
   already has 177 old `lightning_logs` runs from prior iterations, so a full-panel
   TFT sweep is multi-hour at minimum, likely much longer without a GPU). I'd scope
   this to a defensible subsample (e.g. a stratified sample across the four Syntetos–
   Boylan classes, sized for the significance test to have power) rather than all ~880
   SKUs, and tell you the exact subsample size and runtime estimate before starting.
2. **Caveat it in the thesis instead of re-testing it.** Add a sentence to Chapter 3/5
   stating plainly that the SARIMA/LightGBM/TFT ranking is a single-run aggregate
   comparison without paired significance testing, because the per-SKU predictions
   from that run are no longer available — which is an honest limitation, not a
   fabricated one.
3. **You locate the missing export** (maybe it's on another machine, in the partner
   hospital's environment, or in a notebook you have elsewhere) and hand it to me, and
   I run the same Wilcoxon/bootstrap procedure on it.

I'd lean toward (2) unless you think (1) is worth the compute time — but this is your
call, not mine, since it changes how much of Task 1 the defense can actually claim.

## CAMEO significance testing — done, with real numbers

Script: inline Python, `scipy.stats.wilcoxon` (paired, two-sided, `zero_method="wilcox"`)
on per-drug MAE, plus a drug-level bootstrap (10,000 resamples, seeded) for 95% CIs on
both the mean and the median paired difference (CAMEO − competitor; negative = CAMEO
better). Outputs saved to `backend/analysis/task1_cameo_significance_all_drugs.csv` and
`task1_cameo_significance_by_class.csv`.

### All 53 drugs pooled

| Comparison | n | CAMEO lower MAE on | Median Δ MAE (95% CI) | Mean Δ MAE (95% CI) | Wilcoxon p |
|---|---|---|---|---|---|
| CAMEO vs ARIMA | 53 | 43/53 drugs | **−1.91** (−6.86, −0.40) | −14.6 (−41.4, 13.9) | **0.00023** |
| CAMEO vs Analogous | 53 | 42/53 drugs | **−2.97** (−8.64, −1.55) | −11.3 (−35.3, 15.9) | **0.000036** |
| CAMEO vs DDPFF-style | 53 | 38/53 drugs | **−0.67** (−1.92, −0.27) | −15.8 (−56.4, 8.9) | 0.0332 |

**Read this carefully — it's not a clean "CAMEO wins" story:**
- On the **median**, CAMEO is significantly better than all three baselines (all raw
  p < 0.05; ARIMA and Analogous survive Bonferroni correction across all 15 tests run
  here, α = 0.0033; DDPFF-style does not survive correction).
- On the **mean**, none of the three comparisons are significant — every mean-difference
  CI straddles zero. This is because MAE is extremely right-skewed across drugs (a
  handful of lumpy/intermittent drugs have MAE in the hundreds to ~2,800, versus a
  median around 12–15), so a few outlier drugs dominate the mean and inflate its
  variance. This is not a new problem I'm introducing — it's the same
  "drug-averaged MASE blows up on the wrong drug" issue the thesis already calls out
  in Chapter 5 (P652623) — but it does mean the mean-MAE gap you might quote informally
  is not statistically defensible, only the median gap and the win-rate are.

### By Syntetos–Boylan class (small-n warning applies — see below)

| Class (n drugs) | vs ARIMA p | vs Analogous p | vs DDPFF-style p |
|---|---|---|---|
| smooth (11) | **0.00098** ✓Bonf | **0.00098** ✓Bonf | 0.102 |
| erratic (9) | 0.570 | 0.652 | 0.250 |
| intermittent (14) | 0.135 | 0.135 | 0.0295 |
| lumpy (19) | 0.123 | **0.0082** | 0.953 |

This lines up with, and now statistically backs, the thesis's existing narrative that
CAMEO's real win is concentrated in the **smooth** class — that's the only class where
CAMEO beats *both* ARIMA and Analogous at a level that survives multiple-comparison
correction. In erratic, none of the three comparisons reach significance at all — with
only 9 drugs, this could be a real null result or just underpowered; I can't tell them
apart at this n. Lumpy shows a real, correction-surviving-adjacent edge over Analogous only.

**Caveats you should know about before this goes anywhere near the thesis:**
1. **Per-class n is small** (9–19 drugs). Non-parametric tests at this n have limited
   power — a non-significant result in erratic/intermittent is not proof CAMEO doesn't
   help there, just that this held-out set can't tell us either way.
2. **Multiple comparisons.** I ran 15 Wilcoxon tests total (3 aggregate + 12 by-class).
   At uncorrected α=0.05 we'd expect ~0.75 false positives by chance. I've marked which
   results survive a conservative Bonferroni correction (α=0.0033) above — only 4 of the
   15 do. I'd report both raw and corrected p-values in the thesis table, as you asked.
3. **Single split.** This is one leave-drugs-out draw of 53 held-out drugs, not repeated
   resampling of *which* drugs get held out. So this tells us the ranking is consistent
   *within this split*, not that it would replicate under a different random 53-drug
   holdout — that's a distinct question from Task 3 (variance across seeds), which I
   have not touched yet.
4. Ties: zero exact ties in MAE across all three comparisons (53/53, 0 ties each), so
   the Wilcoxon zero-handling method doesn't materially affect the result here.

## Does this weaken any existing thesis claim?

Yes, one, worth flagging now per your instructions: Chapter 5 currently states CAMEO
"gets the lowest mean MAE" as if that's the headline comparison. It is the lowest mean,
but the mean difference is not statistically significant against any baseline given the
skew in this data — only the median difference and win rate are defensible at a
population level. I'd suggest (pending your go-ahead) softening "lowest mean MAE" to
lead with the win rate and median gap, and demoting the mean MAE to a supporting number
with the caveat that it's not statistically distinguishable from the alternatives. I
have not touched the thesis text yet — flagging only, as instructed.

## SARIMA vs LightGBM — fresh rerun, real code, real data

You told me to re-run this rather than caveat it. I did — using the actual
`SarimaModel` / `LightGBMModel` classes from `backend/src/app/forecasting`
and the actual feature-engineering modules (temporal/lag/rolling), on the
audited 2023–2024 daily panel (`~/Downloads/hospital_daily_demand.csv`,
restricted to the 882 canonical drug codes). Script: `backend/analysis/baseline_rerun.py`.

**This is a genuinely different experimental setup from whatever produced
63.6%/59.8%/93.5% in Table 5.1 — read the caveat before using these numbers
anywhere:**
- Single chronological holdout (last 30 days = FORECAST_HORIZON) vs. whatever
  walk-forward/multi-fold scheme the original run used.
- No EM-SARIMA stockout correction (needs the live `drug_receipts` DB table,
  which doesn't exist in this workspace) and no hospital census/supplier
  covariates (same reason) — the original numbers had these, mine don't.
- Stratified **equally** across the 4 demand classes (up to 20 drugs/class),
  not weighted by however drugs were actually distributed in the original
  full-panel run — lumpy/intermittent/erratic drugs forecast much worse than
  smooth ones, so oversampling them (relative to the true panel mix) pulls
  the aggregate SMAPE up a lot.

That's why my fresh aggregate SMAPE (SARIMA 107.0%, LightGBM 96.1%, both far
worse than 63.6%/59.8%) is **not a replacement number** for Table 5.1 — it's
a harder, differently-sampled test, run only so the *paired, per-drug*
comparison is real. Do not swap these aggregates into the thesis.

### Paired significance result (n=63 drugs with both models trained OK, out of 65 sampled; 2 LightGBM training failures on drugs with literally zero demand in the 30-day holdout, `Tweedie` objective error — expected on ultra-sparse SKUs, logged not hidden)

| | LightGBM lower SMAPE | SARIMA lower SMAPE | Median Δ (LGBM−SARIMA) | Mean Δ | Wilcoxon p |
|---|---|---|---|---|---|
| Pooled (n=63) | 42/63 | 21/63 | **−15.1** (95% CI −24.9, −0.9) | −9.7 (CI −26.6, 7.3) | **0.0449** |

Real, freshly computed, and directionally consistent with the thesis's existing claim
that LightGBM beats SARIMA — and this time it comes with an actual p-value: **0.045**,
significant at α=0.05 but not by a wide margin, and (same pattern as CAMEO) only the
median difference is robust; the mean-difference CI still straddles zero.

### By class — and this is the part that should change how confidently the thesis states this

| Class (n) | Median Δ | Wilcoxon p |
|---|---|---|
| smooth (11) | −16.2 | **0.014** |
| erratic (14) | −15.0 | **0.025** |
| intermittent (19) | −33.3 | 0.104 (not significant) |
| lumpy (19) | **+6.0** | 0.169 (not significant; direction flips — SARIMA slightly better here) |

**This is a real result you should know about before it goes anywhere near the defense:**
in this fresh test, "LightGBM beats SARIMA" only holds up statistically in the smooth and
erratic classes. In intermittent it's a big median gap but not significant (n=19, high
variance). In lumpy, the median difference actually flips in SARIMA's favor, though not
significantly. A blanket "LightGBM is the better baseline" claim is not fully supported by
this data — it's class-dependent, same as the CAMEO story. I have not touched the thesis
text; flagging per your instructions.

## TFT — blocked by real compute cost (measured, not assumed)

You asked me to re-run this too. I tried. Here's what I measured before stopping to ask you:

- `TFTModel.train()` on a real ~700-day series takes **~50–60 seconds per epoch** on this
  machine (CPU only — no CUDA; Apple MPS exists but is deliberately disabled in the
  production code because `pytorch-forecasting` crashes on MPS, per a comment already in
  `tft_model.py`). I confirmed this isn't a hang by attaching `faulthandler` mid-run: it's
  genuinely inside `torch/autograd` doing real backward-pass work, just slow.
- Production trains **50 epochs per drug** (`TFT_MAX_EPOCHS = 50` in `constants.py`, no
  early stopping — see Task 4 notes). At ~55s/epoch that's **~45 minutes per drug**, and
  that's the *good* case; one pathological drug (DM00058, 1 non-zero day out of 731) was
  still mid-backward-pass after 40 seconds on epoch 1 of 1 — sparse/near-dead SKUs are
  slower still, not faster.
- A stratified sample big enough to say anything with real power (my SARIMA/LightGBM run
  used up to 20 drugs/class = 65 total) would cost roughly **65 × 45 min ≈ 49 hours** of
  wall-clock CPU time for TFT alone, on this hardware, at production settings.

I'm not going to silently either (a) burn ~2 days of background compute without telling
you, or (b) quietly cut epochs/sample size to make it fast and call that "the same test" —
a crippled TFT (fewer epochs than production) would make it look *even worse* than the
already-bad 93.5% SMAPE, which is exactly the kind of unfair-to-a-baseline move you told me
not to make. So — your call, concretely:

1. **Small-n TFT, full 50 epochs, run in the background over hours** — e.g. 6 drugs (rough
   floor for a within-class read), ~4.5 hours; or 12 drugs (3/class), ~9 hours. I'd kick it
   off now, checkpoint after every drug to CSV, and report back partial results as they
   land rather than making you wait for the whole thing.
2. **Skip a fresh TFT rerun. Caveat Table 5.1 instead**, and fold this exact measurement
   (55s/epoch, 45 min/drug, no early stopping) into Task 4 (TFT transparency) as the
   concrete, evidence-backed reason TFT was dropped from the production ensemble in favor
   of SARIMA+LightGBM+Classical — which is a *stronger*, more defensible statement than an
   unsupported "TFT underperforms" claim, since it's now backed by a real measurement, not
   just an aggregate number with no context.
3. Something in between — you tell me the n and I'll give you a time estimate before
   starting.

**Decision (2026-09-14): skip a fresh TFT rerun.** Per your go-ahead, Table 5.1's
SARIMA/LightGBM/TFT comparison will be caveated rather than paired-tested, and the
measured 55s/epoch, ~45min/drug, no-early-stopping finding will be carried into Task 4
(TFT transparency) as the evidence-backed reason TFT was dropped. No thesis text has
been edited yet — this and every other finding gets applied only after the Task 8
changelog is approved, per your original instructions.

## Decision (2026-09-14): apply both pending text edits now

Per your go-ahead at the Task 8 checkpoint, both flagged edits below were applied to
`Chapters/Chapter_5/Chapter5.tex`:

1. The RQ1 baseline paragraph now reports the actual Wilcoxon result (pooled
   $p=0.045$, median $\Delta=-15.1$ SMAPE) and the per-class breakdown (significant in
   smooth $p=0.014$ and erratic $p=0.025$; not significant in intermittent $p=0.104$;
   flips, not significantly, in lumpy $p=0.169$), with the conclusion reworded from a
   blanket "LightGBM beats SARIMA" to a class-scoped statement.
2. The CAMEO paragraph now leads with the win rate and median MAE (with the paired
   Wilcoxon p-values: $p=0.00023$ vs ARIMA, $p=0.000036$ vs Analogous, both survive
   Bonferroni; $p=0.033$ vs DDPFF-style, does not survive), and explicitly demotes the
   mean-MAE comparison as not statistically defensible (skew, CI straddles zero), rather
   than leading with "lowest mean MAE" as the headline.

Item 3 (TFT framing) was checked across `Chapter1.tex`, `Chapter3.tex`, `Chapter5.tex`,
and `Chapter6.tex`: nowhere does the text imply TFT is a live/current competitor inside
the production ensemble — it is consistently framed as a baseline that was trained,
lost, and (per Task 4) was given a fixed, modest, CPU-only budget. No edit needed.

Recompiled clean (no errors, no new overfull/underfull warnings); both edited passages
visually verified on the rendered PDF.

## Task 1 status: CLOSED (analysis done, text edits applied)

- CAMEO significance: done, real data, real Wilcoxon + bootstrap, now in the thesis text.
- SARIMA vs LightGBM significance: done, real fresh rerun, real Wilcoxon + bootstrap, now in the thesis text.
- SARIMA/LightGBM vs TFT significance: not run — caveat only, backed by the compute
  measurement above (carried into Task 4's TFT transparency subsection).

## Files written this task
- `backend/analysis/task1_cameo_significance_all_drugs.csv`
- `backend/analysis/task1_cameo_significance_by_class.csv`
- `backend/analysis/cameo_mae_pivot.csv` (intermediate pivot, per-drug MAE by method)
- `backend/analysis/baseline_rerun.py` (reusable script — real model classes, real data)
- `backend/analysis/task1_baseline_sarima_lgbm.csv` (65-drug fresh per-drug SMAPE)
- `backend/analysis/task1_baseline_significance_by_class.csv`
- `backend/analysis/README.md`
- `backend/analysis/cameo_real_per_drug_results.csv` (copy of the source data, gitignored)

Also worth knowing: while tracing this I found that `backend/src/app/forecasting/training/trainer.py`'s
live `MODEL_REGISTRY` is `{sarima, lgbm, classical}` — **TFT is not part of the current
production ensemble at all**. The `TFTModel` class still exists and works (I just ran it),
but it's not wired into `ForecastingTrainer.train_all()`. That's consistent with the
thesis's own narrative (TFT underperforms and was dropped), but if Table 5.1 or the text
implies TFT is/was a live competitor inside the current system rather than a
now-retired exploratory baseline, that framing should probably be tightened. Flagging,
haven't touched the text.
