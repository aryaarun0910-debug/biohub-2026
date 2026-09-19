---
tags:
  - stage
---

# Gap2 Recovery

Two-frame recovery: joins a track end at *t* to a start at *t+3* via two
synthetic nodes. `GAP2_MAX_TOTAL_UM` 10.2, `GAP2_MAX_STEP_UM` 4.4.

**It was in the wrong position.** See [[Gap2 In The Wrong Position]] — the finding
behind [[s08]].

The start it joins to is an *orphan* (no incoming edge), which is exactly the pool
[[Safe Division]] draws its second daughters from. Run gap2 first and it eats a
daughter: one true division becomes a false positive plus a false negative.

Its own parameters are inert — `g2_step`, `g2_frac`, `g2_abs` all return exactly
0.00000. See [[Inert Variables]].

Related: [[Ordering Bug Class]], [[Gap Closing]]
