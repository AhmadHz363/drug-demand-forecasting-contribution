# Task 6 — Abstract / Results Alignment

## What was there already

The abstract (`main.tex`) already stated "all-drug pooled WAPE remained above one"
directly next to the CAMEO win claim — Task 5's check confirmed this was already
present. So the raw fact wasn't missing; it was under-emphasized (tacked on as a
trailing clause) and didn't explicitly name *which* class the positive result actually
belongs to.

## What changed

Old:
> "CAMEO won 52.8% of 53 held-out launches on MAE and cut smooth-launch pooled WAPE to
> 0.51; all-drug pooled WAPE remained above one."

New (two sentences, `main.tex`):
> "CAMEO won 52.8% of 53 held-out launches on MAE, and its strongest result is
> class-specific: pooled WAPE on smooth-demand launches fell to 0.51, roughly halving
> the classical-analog baseline. That result does not generalize to a single all-drug
> cold-start accuracy figure, however: pooled across every demand class, WAPE stayed
> above one for CAMEO and for every baseline tested, so clipped accuracy on the full
> pool is zero, and intermittent and lumpy launches remain largely unsolved."

This makes both of the user's required points explicit rather than implicit:
explicitly names the smooth demand class as where the positive result lives, and
explicitly states the all-drug pooled WAPE > 1 / clipped-accuracy-is-zero fact as its
own sentence rather than a trailing clause, plus adds the intermittent/lumpy
"largely unsolved" framing so a reader can't mistake win-rate (a per-drug count) for
pooled accuracy (a volume-weighted one).

## Other intro/abstract claims checked

- Chapter 1 makes no result claims at all — only the internship's 85% *target*
  (forward-looking, not a result), so no caveat was needed there.
- The abstract's SHIELD-XR sentence (84.6% hospital-week, 86.4% hybrid-ABC) already
  reports the aggregate grains that are the actual headline numbers, not a
  class-specific result being generalized, so it doesn't have the same
  "class result read as a general claim" risk that the CAMEO sentence had. Left as is.
- The abstract's LightGBM-vs-SARIMA-vs-TFT sentence is a plain ranking statement with
  no accuracy figure attached, so there's no number there to mis-scope.

## Status: CLOSED

Recompiled clean — no errors, no overfull boxes.
