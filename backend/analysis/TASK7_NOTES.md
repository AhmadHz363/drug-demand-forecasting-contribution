# Task 7 — Scope / Title Alignment

User chose "both": (a) draft the formal restocking problem statement, and (b) consider
alternative titles. Decision on (b): **keep the current title** ("AI-Driven Drug Supply
Chain for Hospital Departments") — no title change made.

## (a) Formal problem statement — added, not implemented

New `\subsection{A formal problem statement for restocking optimization (not
implemented)}` added to `Chapters/Chapter_6/Chapter6.tex`, under "Study Limitations and
Scope for Future Work," right after the existing one-line restocking-engine mention.

States precisely, with three numbered equations:

- **Decision variables**: $Q_{s,w}$, the order-up quantity per drug $s$ per week $w$.
- **State**: opening inventory $I_{s,w}$; demand $D_{s,w}$ with distribution $F_{s,w}$
  supplied directly by this thesis's own output — SHIELD-XR quantiles for warm-start
  drugs, CAMEO conformal intervals for cold-start ones — rather than assumed.
- **Cost structure** (Eq. 6.1): holding cost $c_s^h$, expiry cost $c_s^e$, and an
  asymmetric shortage penalty $c_s^p = \lambda_s c_0^p$ with $\lambda_s > 1$ for
  VED-vital drugs, tying directly into the VED/ABC criticality tiers already discussed
  in Chapters 3 and 5.
- **Objective** (Eq. 6.2): minimize total expected cost across the catalog and a
  $W$-week horizon, subject to a service-level constraint on vital drugs and an implicit
  budget/capacity constraint.
- **Special case** (Eq. 6.3): the classical newsvendor order-up-to level
  $Q^*_{s,w} = F_{s,w}^{-1}\!\left(\frac{c_s^p}{c_s^p+c_s^h}\right) - I_{s,w}$, made
  concrete by noting $F_{s,w}^{-1}$ is exactly the quantile output already produced by
  this thesis, not an assumption someone downstream still has to make.

Explicitly not implemented — the closing sentence says why (fitting the real cost/VED
data and solving at scale is "not a small addition" and depends on the forecasts being
trustworthy first, which was this thesis's actual object of study).

Recompiled and visually checked (rendered pages 56–57): all three equations render
cleanly, no overflow, no overfull/underfull-hbox regressions introduced.

## (b) Title alternatives — proposed, user chose "keep current title"

Presented four options (keep as-is; a minimal reword moving "AI-Driven" onto "Demand
Forecasting"; a mechanism-first reword naming SHIELD-XR/CAMEO's warm/cold-start split;
a "Decision Support" framing matching Chapter 6's own language). User selected **keep
the current title** — no changes made to `title.tex`, `main.tex` metadata, or any
chapter header.

Noted for the record: Chapter 6's "Summary" section (line ~20) already states
explicitly that "a restocking optimizer remains the next layer to build, not something
this stage delivers," so the body text does not overpromise regardless of the title;
the (a) addition above only reinforces that existing, correctly-scoped framing with a
concrete, citable problem statement instead of a one-line aspiration.

## Status: CLOSED
