---
tags:
  - finding
  - key
---

# The Harness Predicted s08 Before The Run, And Was Right

The first closed prediction loop: harness number recorded in the [[Ledger]]
**before** the kernel ran, then compared.

| | kernel (8 films, full deployed chain) | harness (same films, our ports) |
|---|---|---|
| divisions | 4/1/8 → **5/1/7** | 4/2/8 → **5/2/7** |
| divJ | 0.3077 → 0.3846 | 0.2857 → 0.3571 |
| edge term | **+0.00035** | **+0.00034** |
| proxy delta | **+0.00803** | **+0.00748** |

Absolute in-kernel validator proxy: [[s05]] **0.9715**, [[s08]] **0.9795**.

**Mechanism confirmed exactly:** one division recovered, which is what
[[Division Recovery in 44b6_341df25f]] predicted.

**The entire 0.00055 gap is the denominator.** The harness carries one extra
division false positive — the un-ported [[DeepCenter]] safe-div veto, documented
*before* this run — so the division term is 0.1×(1/13) = +0.00769 for the kernel
and 0.1×(1/14) = +0.00714 for the harness. Nothing else differs.

⚠ **This validates the INSTRUMENT on the VALIDATOR tier. It does not make [[s08]]
worth anything on the board** — [[Scored Films Measurement]] puts it at +0.00000
there, because the scored films have no recoverable division. Both are true: the
harness is accurate, and the validator films are not the target. That distinction
is the whole point of [[Evidence Tiers]].

Footnote: s08's sweep kept `base` although `dcgap035` led by 0.0000147 —
`PPSWEEP_SELECT_MARGIN` is 0.001, so the guard held. See [[In-Kernel Sweep Is Inert]].
