---
tags:
  - finding
  - key
---

# The Board Verdict — one prediction right, two wrong, one rule broken

| submission | board | vs 0.947 | vs s05 | pre-registered | |
|---|---|---|---|---|---|
| [[s05]] remove relink | 0.945 | −0.002 | — | 0.960–0.970 | **failed** |
| **[[s08]]** + reorder | **0.946** | −0.001 | **+0.001** | = s05 ±0.001 | **CORRECT** |
| [[s10]] + linefit 0.6 | 0.944 | −0.003 | −0.001 | s05 +0.002…0.005 | **failed** |

**Nothing beat the 0.947 baseline. Standing unchanged.**

## The instrument test passed

[[s08]] was submitted *predicted to be worth nothing*, purely to test whether
[[Scored Films Measurement]] means anything. It said **+0.00000** while the
validator films shouted **+0.00803**. The board gave **+0.001** — agreement at
the resolution limit, and the validator films were wrong by an order of
magnitude. **The scored-film instrument is the one that works.**

## The rule that broke, and it was already flagged

[[s10]] was chosen by the [[Both-Sets Rule]]: positive on the scored films
(+0.00471), the validator films (+0.00338) *and* a 199-film tier (+0.00136).
Three tiers agreed. The board says **−0.001**.

But the fragility analysis had already said not to trust it: **net +4 GT edges
out of 2,127, 95% CI [−3, +12]** — a bootstrap that could not separate it from
zero. **I shipped on a positive point estimate the CI did not support.**

> **New rule, earned the expensive way: require the bootstrap CI to EXCLUDE
> ZERO, not merely a positive point estimate.** Under that rule s10 would never
> have shipped, and neither would [[s09]].

## The axis prior holds, and strengthens

[[s08]] is division-axis: **+0.001**. [[s10]] is pure edge-axis: **−0.001**.
[[Axis Priors]] was 3-for-3 on divisions and 0-for-3 on edges; edges are now
**0-for-4**. Every edge-axis change this project has ever shipped has lost.

That is a second, independent argument for [[The Division Lever]].
