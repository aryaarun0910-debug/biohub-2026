---
tags:
  - gotcha
---

# Gotcha: aggregate `mult` is not a node-count readout

[[metric2]]'s `aggregate()` averages per-film values with
**`weight = tp + fp + fn`** — the edge-confusion union, which depends on the
**predictions**.

So any stage that moves coordinates changes edge matching, which changes each
film's weight in the average, which drifts the aggregate multiplier by ~1e-5
**with node counts completely untouched**.

Measured: across 24 [[Linefit Smoothing]] cells, aggregate `mult` drifts up to
**8.85e-05** while `ratio` is **identical** at 0.8977.

**Consequence:** the [[Node Count Exploit]] shows up in **`ratio`**, not in a
small `mult` wobble. Assert on `ratio`.

This caused a real false alarm — [[Script 99 Linefit On s08]] initially reported
FAIL on a correct chain because the assertion was wrong.
