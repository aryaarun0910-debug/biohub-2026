---
tags:
  - infra
---

# `scripts/91_other_stages.py`

Ports of [[Gap Closing]], [[Gap2 Recovery]], [[Short Track Filter]],
[[Linefit Smoothing]] and [[Prune Isolated]] onto original-voxel coordinates
with the anisotropic `SCALE`.

Also supplies the `run()` / `show()` reporting harness that every later
experiment reuses, including the `<<FALSE GAIN: all multiplier` flag.

**That flag has a known false alarm**: it only inspects `J_edge` and cannot see
`divJ`, so a row whose gain is a *division* gets labelled as an exploit. This
happens on row C of the [[s08]] measurement.

Measures each stage **one at a time** against a raw + safe_div anchor, with each
stage at its deployed position — which is why its numbers are a *three-stage*
topology and needed re-measuring on the full chain. See [[Transfer Lesson]].
