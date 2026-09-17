# Task 5 — Consistency Pass

## SKU count: resolved to 882

Found the discrepancy exactly where you flagged it, plus one more instance:

- `Chapter1.tex`: "roughly 890 stock-keeping units" → **fixed to "882"**.
- `Chapter2.tex` (×2): "close to 890 daily... hospital series" and "close to 890 SKUs"
  → **fixed to "882"** in both places.
- Everywhere else — `Chapter4.tex`, `Chapter5.tex`, `Chapter6.tex`, and
  `table4_2.tex` (the actual data table) — already consistently said **882**, and
  Table 4.2 is the computed source: "Unique drug codes & $882$".

**882 is the correct, single number to standardize on** — it's what the audited dense
panel actually contains (matches `hospital_daily_demand_2023.csv`'s canonical drug set,
which is also what `baseline_rerun.py` used in Task 1/3 as "the 882-drug universe").
The two "890" mentions look like an early, rounded guess that never got updated once
the panel was finalized; there was never a real 890-count computation to reconcile
against.

(Separately, this session's own rerun scripts encountered *other* SKU counts — 733 in a
stale `training_meta.json` artifact, 878 and 893 in different reconstructions built
from different source files — but none of those are thesis claims, they're artifacts
of which raw export each analysis script happened to join against. Nothing to fix in
the manuscript for those; noted here only so it's clear they were investigated and
ruled out as relevant.)

## General numeric consistency check

Cross-checked every headline number that appears more than once across chapters
(grep-based sweep, not a sample): SMAPE baselines (63.6/59.8/93.5), SHIELD-XR grains
(30.1/69.2/84.6/86.4/95.2), the 79.8%→86.4% stability-screening swap, CAMEO's
160-library/53-test/20-week setup, the MAE table (89.1/100.4/103.7, median 11.6), the
4-way win-rate split (52.8/22.6/17.0/7.5% — arithmetic checks out exactly against 53
drugs: 28/12/9/4), pooled WAPE by method (1.03/1.16/1.20/1.21) and by class
(0.51 smooth, 13.98 vs 20.81 intermittent), conformal coverage (69.2% vs. 90% target),
the panel totals (731 days, 644,742 cells, 75.0% zero, 1.32M/1.34M units), and the 2023
ledger breakdown (747,195 demand-bearing + 58,735 excluded = 805,930, i.e. 92.7% —
matches the figure caption exactly).

**No other discrepancies found.** Every one of these is stated identically everywhere
it recurs (Chapters 1, 2, 4, 5, 6, the abstract, and the relevant tables/figures).

Also checked the abstract specifically (`main.tex`): it already says 882 SKUs, 84.6%,
86.4%, 52.8%/53, and 0.51, all consistent with the body chapters, and it already states
"all-drug pooled WAPE remained above one" right next to the CAMEO win claim — which is
most of what Task 6 (Abstract/Results Alignment) asks for. I'll confirm the remaining
piece (whether "positive results are concentrated in the smooth demand class" is clear
enough) when we get to Task 6 rather than duplicating that work here.

## Status: CLOSED

Edits made: `Chapter1.tex` (1 fix), `Chapter2.tex` (2 fixes). Recompiled clean — no
errors, no overfull boxes.
