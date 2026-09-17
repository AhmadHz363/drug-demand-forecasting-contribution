# Task 2 — SHIELD-XR Ablation Study

## What was actually run

A standalone, DB-free rerun of the **real** production code in
`backend/src/app/forecasting/shield_xr/` (`anomaly_guard.py`, `features.py`,
`daily_model.py`, `weekly_model.py`, `evaluation.py`, `metrics.py` — imported, not
reimplemented, except for the two functions that needed ablation flags — see below).

- **Panel**: dense CODE×DATE demand for 2023–2024 (`hospital_daily_demand.csv`, 893
  drugs, 652,783 rows) left-joined with the real exogenous hospital-driver columns
  (patient/admission/doctor/CR counts) from the two
  `hospital_daily_demand_enriched_{2023,2024}.xlsx` exports. Small gaps in that join
  (the enriched export isn't 100% dense) were zero-filled — same convention the
  production `panel_builder.py` already uses for `bed_occupancy_rate` /
  `weekly_surgery_count`, which are always zero-filled because that data was never
  collected.
- **Split**: identical logic to production (`split_temporal`: 70/15/15 chronological),
  landing at train ≤ 2024-05-26, valid ≤ 2024-09-13, test after.
- **Script**: `backend/analysis/shieldxr_ablation.py`.
- **Sanity check**: the baseline (all features on) run reproduces the thesis's
  headline numbers closely — 32.1% SKU-day / 83.4% hospital-week / 89.9% hybrid-ABC
  here, vs. 30.1% / 84.6% / 86.4% in the thesis. Close enough (different panel
  reconstruction, not a re-run of the identical DB snapshot) to trust the *deltas*
  below, even though the absolute numbers aren't identical to Chapter 5's table.

## Ablations run (one component removed at a time, everything else held fixed)

| Variant | SKU-day acc. | Hospital-week acc. | Hybrid-ABC acc. |
|---|---|---|---|
| **Baseline (all features on)** | 32.14% | 83.42% | 89.86% |
| EVT/GPD spike correction OFF | 32.14% (Δ 0.00) | 83.42% (Δ 0.00) | 89.86% (Δ 0.00) |
| Isotonic calibration OFF | 32.14% (Δ 0.00) | 83.42% (Δ 0.00) | 89.86% (Δ 0.00) |
| Volume-aware sample weighting OFF | 32.14% (Δ 0.00) | 83.42% (Δ 0.00) | 89.86% (Δ 0.00) |
| Class-conditional selection OFF (single global model) | 30.34% (**Δ −1.80**) | 82.18% (**Δ −1.24**) | 89.95% (Δ +0.09, noise) |

Raw CSV: `backend/analysis/task2_shieldxr_ablation.csv`.

## 🚩 Flagging this immediately, per your instruction

**Three of the four ablations show exactly zero measured effect — not "small," literally
bit-identical to the baseline.** I did not stop at "huh, no effect" and move on; I
traced why, because a flat zero across three unrelated components (EVT, isotonic,
sample weights) is the kind of result that's more often a wiring bug than a real
finding. Here's what I found:

**The class-conditional ensemble selector never picks the SHIELD-XR hurdle stack as
the winner, in any of the four demand classes, in this rerun.** It picks **Plain-L1**
(the plain single LightGBM regressor with L1 loss, no hurdle/EVT/isotonic machinery at
all) in every class:

| Class | SHIELD-XR valid. WAPE | Plain valid. WAPE | Plain-L1 valid. WAPE | Winner |
|---|---|---|---|---|
| smooth | 0.256 | 0.246 | **0.243** | Plain-L1 |
| erratic | 0.384 | 0.384 | **0.374** | Plain-L1 |
| intermittent | 0.642 | 0.774 | **0.563** | Plain-L1 |
| lumpy | 0.627 | 0.745 | **0.615** | Plain-L1 |

Because the selector routes every class away from the SHIELD-XR stack, whatever I do
*inside* that stack (turn EVT off, turn isotonic off, turn sample weighting off) has
**zero downstream effect on the numbers you'd report** — those code paths are simply
never read by the winning model. That's not a bug in the ablation; it's an accurate
measurement of *this rerun's* selector outcome. Two honest readings, and I don't yet
know which is closer to the truth:

1. **This is a property of the class-conditional design working as intended.** The
   whole point of "class-conditional candidate selection" is to let a simpler model
   win when it's genuinely better on held-out data, and that's exactly what's
   happening — Plain-L1 beats the more elaborate hurdle stack in every class on this
   split. If so, the one ablation that *did* move the needle (removing the selector
   itself, −1.8pp SKU-day / −1.2pp weekly) is the real story: **the ensembling/selection
   logic is what's earning SHIELD-XR its accuracy, not the EVT/isotonic/hurdle
   engineering specifically** — at least as measured on this data.
2. **This could be an artifact of my standalone reconstruction diverging from the
   exact production panel** (different exogenous-feature completeness from the
   enriched-export join, a slightly different resulting train/valid/test boundary, or
   duplicate-row handling in the dense merge) — in which case the *actual* trained
   artifacts on the real DB panel might have SHIELD-XR winning at least one class, and
   this null result wouldn't transfer. The baseline numbers being close-but-not-identical
   to Chapter 5's table (32.1 vs 30.1, 83.4 vs 84.6, 89.9 vs 86.4) is consistent with
   either explanation — the gaps are small enough to be normal panel-reconstruction
   noise, but I can't rule out that they're large enough to flip a close class-level
   model comparison.

**This is a legitimate risk to a specific thesis claim.** If Chapter 3/5 currently
implies that the EVT/GPD spike handling, isotonic calibration, and volume-aware
weighting are each pulling their weight in the *headline* accuracy numbers, that
framing needs to soften to: these components matter for the classes/weeks where the
SHIELD-XR stack *is* selected (which the current write-up doesn't break out), and the
class-conditional ensembling around them is what's empirically shown to matter here.
I have not touched any thesis text yet.

## Confirmed against the real production artifact (this is no longer just my reconstruction)

`backend/src/app/forecasting/artifacts/shield_xr/training_meta.json` is a leftover
persisted artifact from an actual production training run (not something I generated).
It trained on a different window (train through 2023-09-13, valid through 2023-11-07,
**733 SKUs** — yet another SKU count, see Task 5) and its headline numbers don't match
Chapter 5's table exactly either (test_accuracy 54.5%, hospital-week 85.7%, hybrid-ABC
84.9% — a different run than whichever one produced 30.1%/84.6%/86.4%). But its
`class_winner` field is the smoking gun:

```json
"class_winner": {
  "lumpy": "Plain-L1",
  "smooth": "Plain",
  "erratic": "Plain",
  "intermittent": "Plain-L1"
}
```

**The SHIELD-XR hurdle/EVT/isotonic stack is not selected in a single class, in a real
production run, independently of my reconstruction.** Two different real runs, two
different data windows, same code, same conclusion: the class-conditional selector
always prefers the plain global LightGBM regressor (Tweedie or L1) over the hurdle
stack. This is convergent evidence, not a reconstruction artifact.

## 🚩 This directly contradicts a specific sentence in Chapter 5

`Chapters/Chapter_5/Chapter5.tex`, in the Discussion section (RQ2/RQ3 paragraph):

> "The hurdle-plus-EVT-plus-ensemble stack is really what makes the daily building
> block usable enough to sum in the first place."

The measured evidence says the opposite of the "hurdle-plus-EVT" part specifically:
the daily building block is usable because of the **plain Tweedie/L1 regressors plus
the class-conditional selection and bias correction**, not because of the hurdle
occurrence/magnitude decomposition, isotonic calibration, or EVT spike uplift — those
are trained, evaluated, and then quietly outcompeted on validation WAPE in every
class, in both runs I have evidence for.

This doesn't break RQ2/RQ3 (the *ensemble as a whole*, including the fact that it's
allowed to select a simpler model per class, does hit ~84–86% at the weekly/hybrid
grain) — but it does mean the specific engineering credited in that sentence, and
implicitly in Chapter 1's RQ2 framing ("a class-conditional hurdle architecture, with
artifact repair and explicit spike handling") and Chapter 6 ("A class-conditional
hurdle architecture with EVT spike handling... produces a usable hospital-week
forecast"), is not what the evidence shows is doing the work. The thing that is doing
the work — "class-conditional candidate selection," where the hurdle stack is one
candidate among three and is allowed to lose — is already named in the text, just not
foregrounded as *the* load-bearing idea.

## What I'd suggest before finalizing this task (need your call)

- **(a)** Trust this result as-is and reframe the SHIELD-XR contribution honestly around
  "ensembling matters most; the hurdle/EVT/isotonic engineering pays off narrowly and I
  don't have hospital-DB access to isolate exactly where" — fastest option, most
  consistent with the thesis's existing honest tone.
- **(b)** I dig one level deeper for free (no rerun cost) and report per-class WAPE
  breakdowns from the artifacts already computed above, so we can at least say *how
  close* SHIELD-XR came in each class (e.g., "SHIELD-XR is within 1pp of Plain-L1 in
  smooth/erratic, meaningfully behind in intermittent/lumpy") rather than a flat "never
  wins."
- **(c)** If you have access to the original DB-backed training artifacts /
  `training_meta.json` from the actual thesis run (saved under whatever
  `artifacts.py` persists to), send me the `class_winner` dict from that file — that's
  a 10-second check that would tell us definitively whether this null result is
  specific to my reconstruction or also true in the original run.

I'd lean towards (b) then (a) — it's free, already computed, and turns a flat "no
effect" into an honest, specific, defensible statement — but this is your call, and I
want your read on it before I draft any wording changes.

## Decision (2026-09-14): reframe now

Per your go-ahead, three sentences were edited to credit the class-conditional
ensembling/selection mechanism as the load-bearing idea, rather than the hurdle/EVT
engineering specifically, and to report the ablation finding inline:

- `Chapters/Chapter_1/Chapter1.tex` — RQ2 reworded from "a class-conditional hurdle
  architecture, with artifact repair and explicit spike handling" to "a
  class-conditional ensemble, built on artifact repair and a per-class choice between a
  hurdle-plus-spike model and simpler LightGBM regressors."
- `Chapters/Chapter_5/Chapter5.tex` — the sentence "The hurdle-plus-EVT-plus-ensemble
  stack is really what makes the daily building block usable enough to sum in the
  first place" replaced with a short paragraph reporting the ablation numbers (EVT,
  isotonic, and sample-weighting all show zero effect because the selector never picks
  that branch; removing selection itself costs 1.8pp SKU-day / 1.2pp weekly) and the
  corrected conclusion that class-conditional ensembling, not the hurdle/EVT
  engineering, is what is empirically shown to matter.
- `Chapters/Chapter_6/Chapter6.tex` — the RQ2 summary bullet reworded the same way,
  plus one sentence added pointing to the Chapter 5 ablation.

Chapter 3's methodology section (`Chapters/Chapter_3/Chapter3.tex`, "SHIELD-XR hurdle,
spikes, and ensemble") was left untouched — it already correctly describes the three
candidates competing on validation WAPE without claiming which one wins, so no
correction was needed there.

## Task 2 status: CLOSED
