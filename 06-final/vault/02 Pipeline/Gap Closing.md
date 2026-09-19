---
tags:
  - stage
---

# Gap Closing

Rejoins a track end at *t* to a track start at *t+2*, Hungarian per frame.
Prefers reusing an existing unlinked node near the midpoint over synthesising one.

Deployed: `GAP_CLOSE_UM` 5.0, `GAP_CLOSE_REUSE_UM` 3.2, `GAP_CLOSE_MAX_GAP` 2.

It **eats the orphan pool** that [[Safe Division]] draws daughters from — the same
competition that makes [[Gap2 Recovery]]'s position matter. But measured, moving
gap closing after safe division is worth only ~0.00001. Not a lever.

`GAP_CLOSE_UM` 5→8 is **+0.00089 on raw graphs and −0.00015 on the [[s09]] chain** —
the third instance of the [[Transfer Lesson]].

`GAP_CLOSE_UM` is under the [[Drift Guard]].

Related: [[Parametric Search Closed]], [[Inert Variables]]
