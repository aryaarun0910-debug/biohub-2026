---
tags:
  - finding
  - key
---

# Error Budget — where the remaining error actually lives

Answers one question: **is a better detector worth building?** No.

From the kernel's own validator output for [[s05]] (not our ports), 8 films:
`edge TP 5565, FP 184, FN 186`, micro edge Jaccard **0.93766**. The kernel
classifies every missed GT edge by cause. Oracle ablation — set one class to
zero, hold the rest fixed:

| oracle fix | edges | ceiling | owner |
|---|---|---|---|
| perfect DETECTION | 75 | **+0.01264** | detector — *what SSL would target* |
| zero fragmentation | 111 | +0.01870 | linker |
| zero wrong-association | 1 | +0.00017 | linker |
| **zero FALSE-POSITIVE edges** | 184 | **+0.03000** | linker |

- **detector ceiling: +0.01264**
- **linker ceiling: +0.04887 — 3.9× the detector**
- **296 of 371 errors (80%) are linker-owned**

The detector already recovers **96.75%** of GT edges. Perfect detection — not a
better model, *perfect* — is worth +0.0126, and three learned attempts already
died at chance: [[Failed Division Classifier]], [[Failed Anaphase Hypothesis]],
[[Failed Synthetic Division Data]]. With [[Frozen Weights]] and ten days, SSL is
the wrong target.

**The blind spot this exposes:** false-positive edges are the single largest
class (+0.03000) and **nothing in the pipeline targets them**. [[Gap Closing]],
[[Gap2 Recovery]] and [[Safe Division]] all *add* edges. Not one stage removes a
wrong one. Every change in the [[Submission Ledger]] has pushed in the
add-edges direction.

⚠ **Ceilings are oracles and are NOT additive.** FP and FN trade off directly —
linking more conservatively cuts FP and raises FN. The honest read is the
*asymmetry of attention*, not the arithmetic: 80% of the error is linking, and
the largest class has never been attacked.

⚠ The kernel's own breakdown sums to 187 against a reported FN of 186 — its
classifier and its metric disagree by one edge. Treat ceilings as ±1 edge.

Related: [[Evidence Tiers]], [[Upstream Knobs]], [[ILP Linker]]
