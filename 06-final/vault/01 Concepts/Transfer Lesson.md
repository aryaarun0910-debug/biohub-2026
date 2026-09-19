---
tags:
  - concept
  - key
---

# The Transfer Lesson

> **A price measured on one topology does not hold on another.**

Learned expensively via [[s06]], and now confirmed **three independent times**:

| knob | on one base | on another | 
|---|---|---|
| [[Linefit Smoothing]] w=0.4 | **+0.0024** raw graphs | **−0.00135** relinked ([[s06]]) |
| [[Gap Closing]] `GAP_CLOSE_UM` 5→8 | **+0.00089** raw graphs | **−0.00015** [[s09]] chain |
| [[Gap2 Recovery]] reorder | +0.00922 three-stage | **+0.00748** full chain (sign held) |

The third is the instructive one: the price *shrank* 19% but kept its sign,
because the mechanism is discrete (a contest over one orphan node) rather than
continuous (a geometric fit whose neighbours can move it).

**Operational consequence:** re-price, never inherit. [[Script 100 Reprice On s09]]
and [[Script 101 Safediv Gates]] exist entirely because of this rule, and both
came back empty — see [[Parametric Search Closed]].

The corollary is why no [[Upstream Knobs]] variant has been pushed: changing the
edge scorer moves the ILP graph, which is the topology [[s08]] and [[s09]] were
measured on.
