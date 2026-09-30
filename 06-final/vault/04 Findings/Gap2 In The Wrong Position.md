---
tags:
  - finding
---

# Gap2 Is In The Wrong Position

The finding behind [[s07]] and [[s08]].

[[Gap2 Recovery]] joins a track end at *t* to a start at *t+3*. That start is an
**orphan** — no incoming edge — which is precisely the pool
[[Safe Division]] draws its second daughters from
(`candidate_ids = [... if node_id not in incoming]`).

Run gap2 first and it eats a daughter: one true division becomes a false positive
plus a false negative. Run safe division first and gap2 still collects its edge
gain, because the daughter it gives up is one node out of ~750.

**Measured on raw + safe_div:**
| | proxy | divJ | TP/FP/FN |
|---|---|---|---|
| anchor | 0.96964 | 0.3571 | 5/2/7 |
| + gap2 **before** | 0.96163 (−0.00802) | 0.2667 | 4/3/8 |
| + safe_div **then** gap2 | 0.97085 (+0.00121) | 0.3571 | 5/2/7 |

Proven behaviourally on the notebook's *own* stage functions: a contested orphan
goes to safe division under the new order, and uncontested gap2 pairs are
bit-identical in both orders — including synthetic node ids, because safe
division adds **edges only, never nodes**.

Related: [[Ordering Bug Class]], [[Script 96 Reorder Variant]]
