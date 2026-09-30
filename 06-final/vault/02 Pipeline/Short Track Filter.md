---
tags:
  - stage
---

# Short Track Filter

Removes connected components shorter than `OUTPUT_MIN_TRACK_LEN` (deployed 6),
optionally keeping components that contain a fork.

**This stage is the [[Node Count Exploit]] in its purest form.** Raising the
threshold makes edge Jaccard *fall* while the [[Node Count Multiplier]] rises,
and the multiplier gain is **monotone all the way to L=40**. A genuine stage
would peak somewhere. This one never does.

On the [[s09]] chain the screen flags it automatically:
- `st_len=8` → proxy **+0.00103**, dJ **−0.00157**, dratio **−0.03218**
- `st_len=9` → proxy **+0.00041**, dJ **−0.00324**, dratio **−0.04598**

Proxy up, tracking quality down. The board reading is consistent: the historical
`OUTPUT_MIN_TRACK_LEN` 6→9 submission scored 0.947, no better than base.

Related: [[Operating Rules]], [[Failed Short Track Exploit]]
