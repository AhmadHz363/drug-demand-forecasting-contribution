# Task 8 — Final Changelog: Empirical Rigor Pass

One-page summary of every empirical addition, every number that changed, and every
claim that was newly caveated across Tasks 1–7. Full backing detail and raw output
files are in `TASK1_NOTES.md`–`TASK7_NOTES.md` and `analysis/*.csv` in this same
directory. Nothing below was folded into the manuscript before this changelog.

## 1. Statistical significance (new)
- **CAMEO vs ARIMA/Analogous/DDPFF-style** (real per-drug data, 53 drugs): paired
  Wilcoxon on MAE. Median gap significant vs ARIMA ($p=0.00023$) and Analogous
  ($p=0.000036$, both survive Bonferroni across 15 tests); marginal vs DDPFF-style
  ($p=0.033$, does not survive correction). **Mean-MAE difference is not significant
  against any baseline** (heavy right skew) — Chapter 5's "lowest mean MAE" framing was
  demoted; win rate (52.8%) and median MAE (11.6) are now the headline, mean is a
  secondary number.
- **LightGBM vs SARIMA** (fresh 65-drug rerun, real model code, real data, since the
  original per-drug array no longer existed anywhere accessible): pooled Wilcoxon
  $p=0.045$ (median $\Delta=-15.1$ SMAPE, LightGBM's favor). **Not uniform by class**:
  significant in smooth ($p=0.014$) and erratic ($p=0.025$); not significant in
  intermittent ($p=0.104$); flips (not significantly) toward SARIMA in lumpy
  ($p=0.169$). Chapter 5's blanket claim was reworded to state this class-dependence.
- **SARIMA/LightGBM vs TFT**: not paired-tested — the original per-SKU array behind
  Table 5.1's 93.5% TFT SMAPE no longer exists (DB-only, no flat-file export), and a
  fresh full-sample TFT rerun was measured at ~49 CPU-hours for a comparably-sized
  sample. Caveated instead (see item 4).

## 2. SHIELD-XR ablation (new)
- EVT/GPD spike correction, isotonic calibration, and volume-aware sample weighting
  each show **zero measured effect** on SKU-day/hospital-week/hybrid-ABC accuracy when
  removed individually — traced to the class-conditional selector never choosing the
  hurdle/EVT branch in any of the 4 demand classes (confirmed against a real production
  `training_meta.json` artifact independently of this rerun).
- Removing **class-conditional selection itself** (forcing one global model) cost 1.8pp
  SKU-day and 1.2pp hospital-week accuracy — this is the mechanism doing the work.
- **Reframed claims** in `Chapter1.tex` (RQ2 wording), `Chapter5.tex` (replaced "the
  hurdle-plus-EVT-plus-ensemble stack is what makes it usable" with the ablation
  numbers and the corrected conclusion), and `Chapter6.tex` (RQ2 summary bullet) to
  credit class-conditional ensembling as the load-bearing idea, and to describe the
  hurdle/EVT stack as a candidate that competes but has not won a class yet.

## 3. Variance / stability (new)
- **SHIELD-XR**, 5 LightGBM seeds: headline numbers move by **<0.5pp** — stable. One
  sentence added to `Chapter5.tex`.
- **SARIMA vs LightGBM**, 5 independent stratified samples: LightGBM's mean SMAPE beats
  SARIMA's in **5/5** draws — direction is not a lucky-draw artifact, though absolute
  SMAPE level moves a lot between samples. One sentence added to `Chapter5.tex`.
- **CAMEO**, 5 seeds via `evaluate_cameo_holdout`: high seed-to-seed variance in pooled
  and median accuracy. Kept as supporting analysis only, per your call — not added to
  the manuscript, since this harness only compares CAMEO-vs-Analogous (2 methods), not
  the full 4-method Table 5.3 comparison.

## 4. TFT transparency (new)
- New `\subsubsection{TFT training configuration}` in `Chapter3.tex`: full architecture
  (hidden size 64, 4 attention heads, dropout 0.1), optimizer, encoder window, quantile
  loss levels, and the **fixed 50-epoch, no-early-stopping, CPU-only** budget, contrasted
  explicitly with LightGBM's adaptive 40–50-round-patience budget.
- Measured cost cited as the reason a fair-budget rerun was infeasible: ~55s/epoch,
  ~45 min/drug. One sentence each added to `Chapter5.tex` (RQ1 discussion) and
  `Chapter6.tex` (limitations + future work) scoping the "TFT underperforms" verdict to
  this specific budget, not attention architectures generally.

## 5. Consistency pass (fixed)
- **SKU count 890 → 882** in `Chapter1.tex` (1 instance) and `Chapter2.tex` (2
  instances) — 882 is the audited, computed figure (matches Table 4.2); 890 was an
  unreconciled early estimate. All other chapters already said 882.
- Full numeric sweep (SMAPE, SHIELD-XR grains, CAMEO MAE/WAPE table, win-rate splits,
  panel totals, 2023 ledger breakdown) across all chapters, the abstract, and tables:
  **no other discrepancies found.**

## 6. Abstract/results alignment (fixed)
- Abstract's CAMEO sentence rewritten from one clause ("...; all-drug pooled WAPE
  remained above one") into two sentences that explicitly (a) name the smooth-demand
  class as where the win lives, and (b) state as its own sentence that all-drug pooled
  WAPE > 1 for every method (clipped accuracy = 0), and that intermittent/lumpy remain
  largely unsolved.
- Checked other abstract/Chapter 1 result claims (SHIELD-XR grains, RQ1 ranking): no
  similar unscoped generalization found; no further edits needed.

## 7. Scope/title alignment
- Title kept as-is ("AI-Driven Drug Supply Chain for Hospital Departments") — your
  explicit choice among 4 options presented.
- New `\subsection{A formal problem statement for restocking optimization (not
  implemented)}` added to `Chapter6.tex`: newsvendor-style objective (Eq. 6.1–6.2),
  asymmetric VED-tier shortage penalty, and the order-up-to special case (Eq. 6.3) tying
  directly to this thesis's own quantile/conformal outputs as $F^{-1}$. Explicitly
  marked not implemented, with a one-sentence justification.

## Net effect on the thesis's claims
No numbers were removed or hidden; several were added (p-values, CIs, ablation deltas,
seed variance) and a small number of existing claims were **narrowed** to what the
evidence actually supports: "LightGBM beats SARIMA" is now class-scoped, "CAMEO's
lowest mean MAE" is now "CAMEO's lowest median MAE and win rate," and "the
hurdle-plus-EVT stack drives SHIELD-XR" is now "class-conditional ensembling drives
SHIELD-XR, with the hurdle/EVT stack as an as-yet-unproven candidate." This is
consistent with the honesty-about-weaknesses stance the jury already flagged as a
strength — the new numbers make several existing caveats sharper rather than softer.

## Status: ALL 8 TASKS CLOSED. Full recompile clean (0 errors, 0 overfull boxes).
