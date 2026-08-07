# Cycle 3 — Lane O (OOF→public) and Lane D0 (detection preflight)

**Date:** 2026-08-03 · Correct **pre-wrapper** substrate, 199 crops · exact scorer ·
**No GPU, no submission, 0 slots.**

Artifacts: `reports/inventory/oof_public_calibration.json`,
`reports/inventory/detection_preflight.json`.
Reproduce: `scripts/oof_public_calibration.py`, `scripts/detection_preflight.py`.

---

## 1. LANE O — ~~sampling variance is FALSIFIED~~ **RETRACTED 2026-08-07**

> ### RETRACTION — this section's conclusion is wrong, and it is wrong on its own numbers.
>
> §1 tested the sampling-variance hypothesis with a **point probability where a tail
> probability is required.** It observed the flat outcome and quoted `P(flat) = 0.0322` as
> though that were a p-value. The correct one-sided test statistic is
> `P(outcome at least as extreme as flat)`:
>
> ```
> P(down) + P(flat) = 0.0683 + 0.0322 = 0.1005          <- from §1's OWN table
> at the actual public mix (0.50):  0.0737 + 0.0352 = 0.1089
> ```
>
> Admitting any between-embryo variance raises it further (independently estimated ≈0.176).
>
> **At p ≈ 0.10–0.18, four-movie sampling variance is NOT falsified at any conventional
> level.** Stated precisely: we *fail to reject* it. That is not the same as confirming it —
> hypotheses B, C and D remain live — but §1's central claim, and the "cannot carry the
> explanation on its own" wording, do not survive.
>
> **Consequence, and it is the important one: the `+0.020` submission bar was raised largely
> on the strength of this falsification.** The bar is therefore **POLICY resting on a
> statistical error**, not a derived quantity, and it must not be described as evidence-based
> until it is re-derived. Correction C7 already forbade calling it a calibrated transfer law;
> this is why.
>
> **A further finding supersedes the framing entirely.** Scaling arm B's per-crop deltas leaves
> the panel t-statistic invariant at ≈1.15 — a 4× larger arm (+0.0348) has `P(public up)`
> **0.914** against the 1× arm's **0.890**, while a *homogeneous* +0.005 has `P(up)` **1.0000**.
> **The decision variable is cross-crop homogeneity, not magnitude**, closed-form `t4 = 2/CV`.
> Arm B's per-crop CV is **1.71**; 0.95 confidence needs CV ≤ **1.22**. **Arm B never had a 95%
> chance of showing on a four-movie panel at any magnitude whatsoever.** Raising a *magnitude*
> bar could not have fixed that, and cannot now.
>
> Retained below unaltered as the record of what was believed. The per-crop pair rebuild, the
> family-mix table, and the concentration/ESS figures (top-1 4.97%, ESS 142.1/199) are
> unaffected and remain usable.

## 1. LANE O — sampling variance is FALSIFIED as the explanation *(as originally written)*

**Correction first:** the earlier `P(public tie) = 0.4148` consumed the invalid post-wrapper
`+0.000435` deltas and could not speak to this question. Rebuilt from the correct per-crop pairs.

With the true **+0.0084877** pooled effect, a 4-movie panel at 3-dp resolution:

| outcome | probability |
|---|---:|
| **public moves UP** | **0.8995** |
| public moves DOWN | 0.0683 |
| **public FLAT** | **0.0322** |

E[Δ] +0.008614, sd 0.007530, 5th pct −0.001254, 95th pct +0.022679.
P(|Δ| < 0.0005, i.e. invisible at 3 dp) = **0.0335**.

**We observed the 3.2% outcome.** Hypothesis **A (four-movie sampling variance) is falsified** —
it cannot carry the explanation on its own.

Family composition does not rescue it either. Across every mix, P(up) stays between 0.82 and 0.93:

| fraction 44b6 | E[Δ] | P(up) | P(down) | P(flat) |
|---:|---:|---:|---:|---:|
| 0.00 | +0.007924 | 0.9313 | 0.0403 | 0.0285 |
| 0.50 (the actual public mix) | +0.009023 | 0.8910 | 0.0737 | 0.0352 |
| 1.00 | +0.013416 | 0.8215 | 0.1000 | 0.0785 |

Edge-mass weighted vs crop-equal: P(up) 0.906 vs 0.882 — not the explanation.
Contribution is **not** dangerously concentrated: top-1 crop 4.97%, top-5 18.75%, top-10 31.85%,
top-20 49.60%, with an **effective sample size of 142.1 of 199** crops under edge-mass weighting.

**Remaining live hypotheses: B (family/regime shift), C (deployment analogue mismatch),
D (systematic OOF optimism).** The OOF substrate is LOEO *training* crops scored against sparse
GT; deployment is *test* movies. C is the most plausible and is not testable without test GT.
**Classified E-leaning-C: not identifiable from the points we hold.**

### Decision rule adopted (the lane's conservative policy)

- **Research promotion: pooled OOF ≥ +0.015.**
- **Public submission: ≥ +0.020 pooled**, or a measured composition of comparable expected utility.
- **OOF ranking is trustworthy; OOF magnitude is not.** Arm B ranked correctly (positive on both
  families, CI clear of zero) and still returned ~0 publicly, so treat OOF as an ordering device
  and assume substantial magnitude shrinkage in transfer.
- **Public-flat / OOF-positive candidates stay in the private portfolio** but never justify a slot
  on their own.

---

## 2. LANE D0 — detection preflight, unique missing NODES

199 crops · GT nodes **133,318** · matched **118,022** · **MISSING 15,296 (11.47%)**
· interior 13,421 (two edges each) · terminal 1,875 · **28,732 edges at stake**.

### The family split is the finding

| | GT nodes | missing | miss rate | peak within 10 µm |
|---|---:|---:|---:|---:|
| **44b6** | 20,197 | 276 | **1.37%** | **87.7%** |
| **6bba** | 113,121 | 15,020 | **13.28%** | 35.9% |

Distance from each missing GT node to the **nearest accepted detector peak** (match radius 7 µm):

| band | 44b6 | 6bba | ALL | % |
|---|---:|---:|---:|---:|
| 7–8 µm (near-miss) | 142 | 2,075 | 2,217 | 14.49% |
| 8–10 µm | 100 | 3,318 | 3,418 | 22.35% |
| 10–15 µm | 30 | 3,544 | 3,574 | 23.37% |
| **> 15 µm** | 4 | **6,083** | **6,087** | **39.79%** |

**44b6's detector is essentially healthy** — it misses 1.37% of GT nodes and 87.7% of those misses
have a peak within 10 µm, i.e. localisation-class, not absence-class. **6bba's detector is
genuinely blind**: 13.28% missing and **40.5% with nothing within 15 µm**.

### Classification against the lane's decision rule

- **Class C (candidate existed, wrapper removed it): ~0.** These are pre-wrapper peaks, so
  anything absent here was never produced by the detector. Lifecycle/pruning is not the problem.
- **Class D dominates on 6bba** — 63.9% of its misses have no peak within 10 µm.
- Class A/B (sub-threshold response vs NMS merge) **cannot be separated without the detector
  heatmap**, which needs GPU. The >15 µm bucket is strong evidence against both: NMS merges and
  scale errors do not displace a peak by 15 µm.
- The 14.49% near-miss band is exactly the population that **localisation/recentering already
  attacked and lost on** (oracle +0.009, deployed −0.0088, CLOSED). It should not be re-funded.

**Verdict: do not tune NMS/scale, and do not fix lifecycle. If detection is funded it must be a
genuinely new detector for the 6bba regime.** That is the largest single prize in the project but
also the most expensive, and it is the only route consistent with the 0.942 target.

### The economics, priced

| | Δ composite |
|---|---:|
| detection oracle, 17,444 actual FN edges → TP | **+0.103322** |
| detection oracle, all 28,732 edges at stake → TP | +0.136531 |
| **node-ratio bonus surrendered** (ratio −0.1397 → 0) | **−0.008633** |

Restoring nodes moves `total_node_ratio` from −0.1397 toward 0 and gives back the count-adjustment
bonus we currently collect. **The headwind is real but small — about 8% of the prize.** Net
detection ceiling ≈ **+0.095**. It does not change the ranking: detection remains ~3× the
association ceiling (+0.0329).

---

## 3. Lane status

| lane | status |
|---|---|
| **O** | **COMPLETE** — sampling variance falsified; decision rule adopted |
| **D0** | **COMPLETE** — class D dominates on 6bba; new detector or nothing |
| **C0-FULL** | **RUNNING** (199 pre-wrapper crops, 4 workers) |
| **C1** | blocked on C0-FULL |

Nothing submittable. P0-B remains authoritative.
