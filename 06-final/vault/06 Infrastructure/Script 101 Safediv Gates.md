---
tags:
  - infra
---

# `scripts/101_safediv_gates_on_s09.py`

Re-prices all seven [[Safe Division]] gates on the [[s09]] chain — the family
[[Script 100 Reprice On s09]] missed, and the one that matters most because
divisions are the good axis ([[Axis Priors]]).

Deliberately **fine steps** near the deployed values, because the one known
gate-blocked division misses by 0.35 µm and a coarse sweep would step over it.

Screening rule: a row only counts if **divJ rises** — not TP. That is what makes
[[Failed Loosening Tau Diverge]] label itself instead of looking like a win.

**Result: nothing.** See [[Safe Div Gates At Local Optimum]].
