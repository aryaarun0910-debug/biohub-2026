---
tags:
  - concept
  - key
---

# Ordering Bug Class

> **The recurring defect in this pipeline is sequence, not parameters.**

Three instances found:

1. **[[Motion Relink]] wipes the [[ILP Linker]]'s forks.** Fixed by deleting the
   stage — [[s05]], +0.0224.
2. **[[Gap2 Recovery]] before [[Safe Division]] eats the orphan it needs.**
   5/2/7 → 4/3/8. Fixed by reordering — [[s08]], +0.00748.
3. **[[Linefit Smoothing]] before [[Safe Division]]** collapses divisions to
   3/1/9 or 2/3/10. Already correct in the deployed notebook.

The unifying rule: **divisions must settle before anything perturbs geometry or
consumes orphans.**

A fourth candidate was checked and rejected on evidence: moving [[Gap Closing]]
after safe division is worth ~0.00001. Measured, not assumed.

Related: [[Safe Division]], [[Parametric Search Closed]]
