---
tags:
  - failed
  - key
---

# Edge Pruning — a real discriminator beaten by the base rate

[[Error Budget]] found false-positive edges are the largest error class
(+0.03000 oracle ceiling) and that **nothing in the pipeline removes an edge** —
[[Gap Closing]], [[Gap2 Recovery]] and [[Safe Division]] all add. The `.geff`
carries a probability per ILP-selected edge, so pruning is buildable.

**It does not work, and the reason is principled rather than empirical.**

**The discriminator is real.** Pooled **AUC 0.788** (validator) and **0.780**
(scored); FP median probability 0.83 against TP median 0.94.

**The base rate kills it.** 5,540 TP against 194 FP — **29:1**. Removing an FP
gains 1; removing a TP costs ~2, because that GT edge becomes unmatched
(`tp−1` *and* `fn+1`). So a threshold must cut FP at more than twice the TP rate,
and it never does:

| threshold | TP cut | FP cut | TP per FP | net |
|---|---|---|---|---|
| 0.55 | 60 | 16 | 3.8 | −104 |
| 0.75 | 440 | 67 | 6.6 | −813 |
| 0.83 *(FP median)* | 870 | 99 | 8.8 | **−1641** |
| 0.90 | 1840 | 147 | 12.5 | −3533 |

Empirically confirmed: every threshold from 0.50 to 0.85 loses on **both** tiers,
worsening as it rises.

**What would be needed:** a *relative* rule — among a node's candidate
successors, drop the weaker — which does not pay the base rate. That is
structurally impossible here: every `solution` flag in the `.geff` is True, so
the file stores **only ILP-selected edges** and there are no alternatives to
compare against. Same blocker as [[Learned Bonus Is Crippled]].

⚠ **A methodological catch worth keeping.** The first diagnostic required BOTH
endpoints of an edge to match a GT node, found **5** FPs instead of 184, reported
AUC **0.496**, and would have killed this direction as "no signal". The metric
(metric2.py:51-57) counts an edge as FP when **either** endpoint matches a GT
node that ought to have an edge in that direction — typically a labelled cell
linked to an *unlabelled* node. Use the metric's own definition or the diagnostic
measures nothing.

So the [[Error Budget]]'s +0.030 FP ceiling is **real but not reachable by
thresholding**. It remains the largest untouched class.
