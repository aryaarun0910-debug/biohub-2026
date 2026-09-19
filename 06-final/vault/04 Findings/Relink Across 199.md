---
tags:
  - finding
  - key
---

# Relink Removal, Across All 199 Films

The largest-sample result in the project, and the strongest support for [[s05]].

`artifacts/graphs/*.npz` stores **both arms** of exactly the s05 contrast:
`raw_edges` (greedy, forks allowed → relink OFF) and `linked_edges` (1:1
Hungarian tight-then-relaxed → relink ON).

**Calibrated first**, per [[Evidence Tiers]] — the 4 scored films are in this set
and their deployed answer is known from [[Scored Films Measurement]]. The rebuild
agrees in direction on **4/4**: +0.09382, +0.03218, +0.01333, +0.05210.

Only then the other 195:

- removing relink **helps 176/199 films (88.4%)**
- mean **+0.03858**, median +0.02456, p05 −0.00653, p95 +0.12539
- `44b6`: 52/71 (73.2%) · **`6bba`: 124/128 (96.9%)**
- weighted aggregate over 199 films: **+0.03764**

**6bba is where removal is near-universal**, and 6bba dominates the scored weight.

`scripts/105_relink_across_199.py`, 199 films × 2 arms in **1.5 s** on 16 workers.

⚠ One bug worth remembering: `gt_edges` in the npz stores **node IDs**, not
indices. Passing them straight to `score_film` gives `J = 0.00000` on every film,
**silently** — the first run said "relink helps 0/199" and that was the loader.
