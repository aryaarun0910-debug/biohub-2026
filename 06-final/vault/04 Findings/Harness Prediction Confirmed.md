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

Absolute in-kernel validator proxy: [[s05]] **0.9715**, [[s08]] **0.9795**,
[[s10]] **0.9736**.

**A second loop closed, with a different answer.** [[s10]] was predicted at
+0.00338 and measured **+0.00207** — sign right, magnitude over-predicted by 39%.
The mechanism was still confirmed exactly: divJ unchanged at 4/1/8 (pure edge
axis, as predicted), `edges_fragmented` 111 → 108 and `edges_recovered`
5565 → 5568, so the gain is precisely **3 recovered fragmented edges**, and
`edges_lost_to_detection` held at 75 as a coordinate-only stage requires.

**Why s08 was near-exact and s10 was not:** s08's mechanism is **discrete** — one
orphan, one division — and a missing stage cannot blur it. s10's is
**continuous**: [[Linefit Smoothing]] fits along chains, and the harness omits
[[Single Parent Repair]], the short-track rescue and the [[DeepCenter]] vetoes,
all of which change those chains. The [[Transfer Lesson]] operating one level up,
between harness and kernel.

**Rule of thumb earned here:** trust harness magnitudes for discrete mechanisms,
trust only their sign for continuous ones.

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
