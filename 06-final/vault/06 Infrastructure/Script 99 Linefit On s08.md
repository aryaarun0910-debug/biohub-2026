---
tags:
  - infra
---

# `scripts/99_linefit_on_s08.py`

Sweeps [[Linefit Smoothing]] `(weight, window)` over the full [[s08]] chain —
24 cells. Produced [[Linefit Surface On s09]] and the choice behind [[s09]].

Asserts the right invariant: **`ratio`**, not `mult`. The first version asserted
on `mult` and reported FAIL; the assertion was wrong, not the chain. See
[[Aggregate Mult Re-weighting]].

Reports a **plateau** rather than an argmax, because the surface wobbles ~0.002
between neighbouring cells on only 8 films.
