---
tags:
  - stage
---

# Linefit Smoothing

Moves each node toward a line fitted over ±`window` frames of its unique
predecessor/successor chain. **Topology and node count untouched** — it can move
[[Adjusted Edge Jaccard]] only through `J`, never through the multiplier.

Deployed `OUTPUT_LINEFIT_WEIGHT` 0.8, `OUTPUT_LINEFIT_WINDOW` 2. Neither key is
set in the notebook — both sit at their `os.environ.get` default, so changing one
*adds* a line.

**The textbook case of the [[Transfer Lesson]]:** w=0.4 is −0.00135 on the relinked
pipeline ([[s06]]) and **+0.00372** on the no-relink chain. Because it fits along
unique chains, and [[Motion Relink]] changes exactly that topology.

Retuned to 0.3 in [[s09]]. See [[Linefit Surface On s09]].

It must run **after** [[Safe Division]] — smoothing first collapses divisions from
5/2/7 to 3/1/9 or 2/3/10. In the deployed notebook it already is last.
