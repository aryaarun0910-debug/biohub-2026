---
tags:
  - finding
  - key
---

# The 32-Film Verdict — everything holds, but 8 films inflate small effects

[[s11]] delivered 32 deployed-quality validator graphs (46 divisions, 19,332
labelled edges) against the old 8 (12 divisions, 5,751). The original 8 are a
strict **subset**, so this is a superset and every earlier number stays comparable.

**All three changes on the board survive:**

| change | on 8 films | **on 32 films** | |
|---|---|---|---|
| [[s05]] over base | +0.01794 | **+0.01945** | holds, and *grew* |
| [[s08]] over s05 | +0.00748 | **+0.00152** | holds, shrank **4.9×** |
| [[s10]] over s05 | +0.00338 | **+0.00146** | holds, shrank **2.3×** |

**The calibration this buys — 8 films inflate SMALL effects by 2–5×.** The large
effect ([[s05]]) was if anything understated; the two small ones were overstated
several-fold. [[s08]]'s shrinkage is exactly the predicted mechanism: its whole
gain was one division out of 12, and at 46 divisions that dilutes. The ledger
still moves the right way, 15/28/31 → 16/27/30.

**And the linefit surface flattens into uselessness on this tier:**

| | w=0.3 | w=0.6 | spread |
|---|---|---|---|
| 8 films | +0.00487 | +0.00338 | 0.00149 — discriminating |
| **32 films** | +0.00142 | +0.00146 | **0.00004 — indistinguishable** |
| **scored films** | **−0.00086** | **+0.00471** | **0.00557 — still discriminating** |

> **So the [[Both-Sets Rule]] did its work on the SCORED films, not the validator
> ones.** At 32 films the validator tier cannot tell [[s09]]'s 0.3 from [[s10]]'s
> 0.6 at all — it would have passed s09. Only the 4 scored films separate them,
> and they are the target. More validator films does **not** substitute for
> measuring where you are scored.

Note also that divisions get *harder* on the added films: [[s05]]'s ledger is
4/2/8 on the easy 8 and 15/28/31 across 32 — divJ 0.2857 → 0.2027.

`scripts/113_on_32_films.py`
