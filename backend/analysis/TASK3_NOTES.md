# Task 3 — Variance / Stability Across Seeds

Three separate reruns, each using real production code, each varying a genuine source
of stochasticity (not a cosmetic seed) 5 times. Raw data in
`analysis/task3_*_seed_variance*.csv`.

## 1. SHIELD-XR ensemble — LOW variance (good news)

Reran the full baseline SHIELD-XR stack (`shieldxr_ablation.py --seed-variance`) with
`RANDOM_STATE` in {42, 1, 7, 123, 2024}, which genuinely changes model behaviour here
because the internal LightGBM models use `subsample=0.8` / `colsample_bytree=0.8`.

| Metric | Mean | Std |
|---|---|---|
| SKU-day accuracy | 31.98% | **0.15pp** |
| Hospital-week accuracy | 83.36% | **0.40pp** |
| Hybrid-ABC accuracy | 89.85% | **0.30pp** |

**This is a genuinely positive, reportable result.** The headline SHIELD-XR numbers
are stable to well under half a percentage point across 5 different LightGBM seeds.
Whatever else is true about the class-conditional selector (Task 2), the ensemble's
output is not a lucky roll of one random seed.

## 2. SARIMA vs. LightGBM baseline — moderate variance, but a consistent direction

Reran the Task 1 stratified-sample comparison (`baseline_seed_variance.py`) with 5
different random draws of the same 15-per-class sample (55 drugs each time — 878
canonical drugs available this run, not all 882, see Task 5). SARIMA has no stochastic
training step and the production `LightGBMModel` uses no bagging/feature subsampling,
so the only real source of variance here is *which drugs got sampled*.

| Metric | Mean | Std | CV |
|---|---|---|---|
| SARIMA mean SMAPE | 109.30% | 5.43pp | 5.0% |
| LightGBM mean SMAPE | 90.61% | 13.78pp | 15.2% |

LightGBM's aggregate is noticeably more sensitive to which drugs land in the sample
(CV 15% vs. 5%) — worth knowing before treating any single 55-drug draw as definitive.
**That said, the direction is completely consistent: LightGBM beat SARIMA on mean
SMAPE in all 5 of the 5 independent samples** (112.8<113.9, 93.9<102.1, 82.6<114.3,
86.3<105.2, 77.5<111.1). This actually *strengthens* Task 1's significance finding
rather than weakening it — the LightGBM-over-SARIMA direction is not an artifact of
one lucky sample.

(These SMAPE levels, ~80-115%, are much worse than the thesis's reported 59.8%/63.6%
— expected and already caveated in `baseline_rerun.py`'s docstring: this simplified
rerun has no EM-correction and no external hospital-driver features, unlike the
production pipeline.)

## 3. CAMEO — HIGH variance in pooled/median accuracy (needs a decision)

Reran the real, unmodified `evaluate_cameo_holdout` from
`app/cold_start/cameo/validation.py` — the actual leave-drugs-out harness `train_cameo`
uses in production — fed with the real matched-source drug metadata
(`drugs_training_synthetic_filled_with_codes.xlsx`, 299 matched-source drugs) and real
weekly demand built from the dense 2023-2024 panel. Varied `seed` in {1,2,3,4,5},
which controls both the library/held-out split (213 library / 53 test drugs — the
53 matches Chapter 5's test count) and the MetricNet's random initialization.

**Caveat up front**: this in-repo validator only compares CAMEO against the
"Analogous" nearest-neighbour baseline — ARIMA and the DDPFF-style baseline in
Chapter 5's Table 5.3 aren't implemented here, so this is a 2-method check, not a
reproduction of the 4-way table.

| Metric | Mean | Std | Per-seed values |
|---|---|---|---|
| CAMEO pooled accuracy | 14.7% | **13.5pp** | 21.4, 0.0, 26.1, 25.8, 0.0 |
| CAMEO median accuracy | 5.9% | **13.2pp** | 0.0, 0.0, 29.5, 0.0, 0.0 |
| CAMEO win rate vs. Analogous | 56.2% | 5.9pp | 62.3, 49.1, 60.4, 58.5, 50.9 |
| Smooth-class pooled accuracy | 35.5% | 24.4pp | 39.3, 0.0, 23.5, 60.8, 53.6 |
| Conformal coverage (90% target) | 63.0% | 13.8pp | 53.8, 53.8, 69.2, 84.6, 53.8 |
| Analogous pooled accuracy | 0.0% | 0.0pp | 0, 0, 0, 0, 0 (WAPE ≥ 1 every seed) |

**This is a real, substantial finding, not noise I'd dismiss.** Two things stand out:

1. **CAMEO's win rate against the Analogous baseline is comparatively stable
   (56% ± 6%)** — reassuringly close to the thesis's reported 52.8% win rate (against
   a harder 4-way field, so a somewhat higher number here is expected), and it wins in
   4 of 5 seeds outright.
2. **But the pooled and median accuracy numbers swing enormously — literally between
   0% and ~26–30% depending on the random library/test split and network
   initialization.** In 3 of 5 seeds, median accuracy came out to exactly 0%. This
   means a single held-out run (like the one behind Chapter 5's Table 5.3/5.4) could
   have reported meaningfully better *or* worse pooled numbers purely by chance of
   which 53 drugs ended up in the test set and how the embedder happened to
   initialize — the current single-run numbers are a point estimate with a wide,
   previously unreported error bar around them.

This doesn't contradict a specific existing sentence the way Task 2 did (the thesis
already reports CAMEO's headline results with real caveats), but it's the kind of
uncertainty a defense committee would ask about, and I think it belongs somewhere near
Table 5.3/5.4 or the CAMEO limitations discussion.

## What I'd suggest (need your call before touching text)

- SHIELD-XR's low variance is unambiguously good news — I'd add one sentence
  reporting it as evidence of robustness.
- The baseline comparison's consistent direction across 5 samples is also good news
  for the existing significance claim — one sentence would strengthen Task 1's
  write-up.
- **CAMEO's variance is the one that needs a decision.** Options:
  (a) Add a variance/stability caveat next to Table 5.3/5.4 reporting this seed
  sensitivity honestly (recommended — it's exactly the kind of "we checked, and it's
  not as stable as a single run suggests" disclosure that's been the thesis's
  strength so far).
  (b) Note it in the Task 8 changelog only and leave text changes for a single pass at
  the end.
  (c) Treat this as out of scope since it's a different (2-method) harness than
  Table 5.3's comparison, and not add anything to the text — just keep it in the
  analysis folder as supporting material.

I have not touched any thesis text for Task 3 yet.

## Decision (2026-09-14)

Per your go-ahead: CAMEO's seed variance stays as supporting analysis only —
`evaluate_cameo_holdout` is a different (2-method) harness than Chapter 5's Table 5.3
comparison, so no thesis text is being changed for it. Data is preserved in
`analysis/task3_cameo_seed_variance*.csv` and this note in case it's useful for the
defense Q&A even though it isn't in the manuscript.

## Task 3 status: CLOSED

Follow-up decision: added the two positive findings to `Chapter5.tex`'s Discussion
section — one sentence on the RQ1 SARIMA/LightGBM resampling check (5/5 consistent
direction), one sentence on the SHIELD-XR ablation paragraph reporting the <0.5pp
seed-to-seed movement in the 30.1/84.6/86.4 headline numbers. Recompiled clean (no
errors, no overfull boxes). CAMEO's seed variance was left out of the manuscript per
your call, kept as supporting analysis only.
