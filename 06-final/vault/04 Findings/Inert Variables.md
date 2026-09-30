---
tags:
  - finding
---

# Inert Variables

72 post-processing variables exist; **38 have never been set by anyone** in the
fork lineage. All were priced against real ground truth and **none beats one
division event** ([[One Division Event Floor]]).

**Six are fully inert** — they return exactly 0.00000:
`g2_frac`, `g2_frame`, `g2_step`, `gc_frac`, `gc_reuse`, `sd_req_nn`

Re-confirmed on the [[s09]] chain after three topology changes, along with
`prune=False` and `st_forks=False` ([[Prune Isolated]] has nothing to remove).

**One never-touched boolean is load-bearing**, and in the dangerous direction:
`SAFE_DIV_REQUIRE_DIVERGENCE=0` costs **−0.025**, division FP 2→20.

Related: [[Parametric Search Closed]], [[Safe Division]]
