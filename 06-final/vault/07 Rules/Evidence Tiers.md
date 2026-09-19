---
tags:
  - rule
  - key
---

# Evidence Tiers — what each film set can and cannot support

Every number in this vault was measured somewhere. **Where** decides what it can
carry. Quote the tier whenever you quote a delta.

| tier | films | GT edges | GT divisions | graphs | what it supports |
|---|---|---|---|---|---|
| **SCORED** | 4 | ~2,273 | **3** | deployed ILP | the only set that *is* the target. Small and noisy, but final. |
| **VALIDATOR** | 8 | ~5,700 | 12 | deployed ILP | mechanism and divisions. **Does not predict the board** — see [[Scored Films Measurement]]. |
| **REBUILD-199** | 199 | large | 151 | *weaker rebuild* (greedy, not ILP) | **direction and consistency only.** Absolute values do not carry. Must be calibrated on the 4 scored films first. |
| **KERNEL VALIDATOR** | 8 | — | 12 | deployed, in-kernel | the instrument that rejected [[s01]]. Same 8 films as VALIDATOR. |

**The three failure modes this table exists to prevent:**
1. Quoting a VALIDATOR delta as if it were a board prediction — that is [[s09]].
2. Quoting a REBUILD-199 absolute — it has none; only its *direction* is meaningful.
3. Reporting a division difference finer than [[One Division Event Floor]], which
   is **0.00714** at a 14-division ledger and **0.0111** on the scored films.

The 199-film tier is only admissible because it is **calibrated**: it reproduced
the deployed direction on 4/4 for relink ([[Relink Across 199]]) and 3/4 for
linefit. An uncalibrated rebuild claim is not evidence.

Related: [[Both-Sets Rule]], [[Transfer Lesson]]
