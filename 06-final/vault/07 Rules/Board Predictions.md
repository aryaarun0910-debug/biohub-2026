---
tags:
  - rule
  - key
---

# Board Predictions — pre-registered, before any result

Baseline **0.947** (verbatim repro, submission 56202098). Board resolution is
**0.001**; anything finer is invisible. 7th/last prize **0.964**, leader **0.973**.

| submission | band | central | basis |
|---|---|---|---|
| **[[s05]]** | 0.960 – 0.970 | **0.969** | in-kernel +0.0224 (right chain, wrong films); scored-film +0.02707 (right films, wrong chain) |
| **[[s10]]** | s05 +0.002…+0.005 | s05 +0.003 | scored +0.00471, in-kernel +0.00207 |
| **[[s08]]** | **equal to s05** ±0.001 | = s05 | scored films +0.00000 — no recoverable division there |

**Why a real gain is expected:** the two estimates agree at ~+0.022…+0.027
*despite differing in both chain and films*. **Why to discount it:** our chain
sits 0.0564 below the board in absolute terms, and the stages accounting for that
gap — [[Single Parent Repair]] especially — repair the same damage
[[Motion Relink]] causes. Partially redundant, so the board gain should land
**at or below** the in-kernel figure, not above.

## What each outcome means

**[[s05]]** — `≥0.964` prize territory · `>0.947` lane confirmed · **`=0.947` is
the most damaging result**, worse than a small loss: it would mean +0.0224
in-kernel transferred to *nothing*, refuting the validator that rejected [[s01]]
· `<0.947` lane dead, everything today void.

**[[s10]]** — the [[Both-Sets Rule]] on trial. `>s05` validates it. `=s05` is
uninformative (below resolution), not a refutation. `<s05` refutes it: three
tiers agreed and were wrong, which would mean 12 films cannot settle an
edge-axis question and [[s11]]'s 32 films become the whole strategy.

**[[s08]] — the instrument test, and the important one.** `=s05` validates
[[Scored Films Measurement]]: it predicted +0.00000 while the validator films
said +0.00803, and was right. `≠s05` refutes it — and with it the basis for
cancelling [[s09]] and shipping [[s10]].

> **The asymmetry worth naming:** s08 is predicted to be worth *nothing* and is
> the most informative of the three. s05 carries the score; s08 carries the
> epistemics.
