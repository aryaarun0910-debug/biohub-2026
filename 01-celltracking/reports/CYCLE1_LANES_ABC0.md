# Cycle report — Lanes A, B, C0

> # ⛔ RETRACTED 2026-08-02 — WRONG SUBSTRATE. DO NOT CITE ANY NUMBER BELOW.
>
> Every measurement in this report was computed on `artifacts/kaggle/p0strict_cache`, which
> **this repository had already examined and explicitly rejected for exactly this purpose**
> before I started — `reports/inventory/wsf_ROUTE1_VERDICT.json`:
>
> > *"Can artifacts/kaggle/p0strict_cache (199 post-wrapper OOF graphs) carry arm B? **NO. It is
> > a POST-wrapper surface and arm B edits a PRE-wrapper stage.**"*
>
> Four independent structural reasons, all measured in that verdict:
>
> 1. **No edge probabilities.** 1,801,603 pre-wrapper edges carry `edge_prob`; the cache has
>    **zero**. The relink cost is `motion + 0.05·raw − 1.0·prob`, so on this cache the learned
>    term is identically zero for *both* arms — "any delta measured there belongs to a different
>    cost function."
> 2. **Wrong node population.** 83,260 nodes (4.25%) that the relink actually sees are absent,
>    and 26,122 synthetic gap-close nodes are present that did not exist when the relink ran.
> 3. **Coordinates have moved.** 92.0% of surviving nodes are linefit-smoothed and rounded;
>    median displacement 0.575 µm against a 6.0 µm gate.
> 4. **The edges ARE the relink output** (`edges = motion_edges`), so re-gating them is a
>    *second* relink of an already-relinked graph.
>
> **Consequently the headline claim of this report — that arm B is `+0.000435` rather than
> `+0.0079822` — is WITHDRAWN. It was my error, not a defect in the record.** `+0.0079822`
> stands unrefuted. The Lane A portfolio simulation, the Lane B stage attribution and the Lane C0
> oracle ceiling are all invalid as stated, because all three were built on this substrate.
>
> This is the project's own dominant failure mode — substrate transfer — committed by me, in a
> cycle whose stated purpose was to be rigorous about it, against a written verdict already in
> `reports/inventory/`. I did not read it before building on the cache.
>
> Correct substrate: the WS-F pre-wrapper pregraphs (`f0_pregraphs` / `f1_pregraphs`, 71 + 128 =
> 199 crops, schema `source_id, target_id, edge_prob`). Re-measurement is in
> `reports/CYCLE2_SUBSTRATE_CORRECTION.md`.


**Date:** 2026-08-02 · **Substrate:** P0-strict, **all 199 cached crops**, complete wrapper,
exact scorer, `prob = 0` proxy · **No GPU. No submission. Slots spent: 0.**

Artifacts: `reports/inventory/edge_fn_census.json`, `private_portfolio.json`,
`augpath_oracle.json`. Reproduce with `scripts/edge_fn_census.py`, `scripts/fn_base_rates.py`,
`scripts/private_portfolio_sim.py`, `scripts/augpath_oracle.py`.

---

## 0. CORRECTION THAT REFRAMES THE LAST TWO CYCLES

**Arm B's exact pooled OOF delta over all 199 crops is `+0.000435`, not `+0.0079822`.**

| | recorded | measured this cycle |
|---|---:|---:|
| pooled OOF delta | +0.0079822 | **+0.000435** |
| per-crop P(Δ>0) | 1.000 | **0.2613** |

armA 0.733715 → armB 0.734150. Per-crop paired delta: mean +0.000299, sd 0.002734, **median
0.000000**. Arm B is neutral-or-worse on 74% of crops; the positive mean comes from a small
number of large wins.

**The public result corroborates the new number, not the old one.** P0-B 0.914 vs arm B 0.914 is
exactly what a +0.0004 effect looks like. A +0.008 effect should have been visible.

**This supersedes my own conclusion from last cycle.** I explained the public tie as "the panel is
too sparse to resolve the effect". The simpler and better-supported explanation is that **the
effect was never +0.008.** The sparse-annotation finding stands as a fact about the instrument
(0.655% coverage on 44b6, 8.529% on 6bba) but it is no longer needed to explain the tie, and I was
wrong to lead with it.

Unresolved: where +0.0079822 came from. My run uses the `prob = 0` proxy, which the project
measured as adequate to ≤2.26e-05 — 18× too small to explain a 0.0075 gap. The discrepancy is a
live defect in the record, not a difference of opinion.

---

## 1. LANE A — private-portfolio decision

Paired resampling of real per-crop (armA, armB) pairs. Both arms are scored on the same crops, so
movie difficulty cancels in the delta; the simulation preserves that pairing.

| panel | E[Δ] | sd | P(B>A) | P(tie@3dp) | 5th pct | 95th pct |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | +0.000454 | 0.001378 | 0.5949 | **0.4148** | −0.001748 | +0.002994 |
| 8 | +0.000455 | 0.000977 | 0.6880 | 0.3941 | −0.001099 | +0.002165 |
| 16 | +0.000442 | 0.000692 | 0.7411 | 0.4447 | −0.000667 | +0.001615 |
| 32 | +0.000440 | 0.000483 | 0.8231 | 0.4933 | −0.000349 | +0.001257 |
| 64 | +0.000438 | 0.000341 | 0.9058 | 0.5427 | −0.000105 | +0.001007 |
| 128 | +0.000432 | 0.000246 | **0.9623** | 0.5637 | **+0.000030** | +0.000835 |

Family mix (4-movie): P(B>A) falls from 0.632 at all-6bba to 0.456 at all-44b6 — **arm B is a
6bba mechanism and is coin-flip-to-negative on 44b6**. Annotation-mass weighting (γ = 0, 0.5, 1)
moves E[Δ] only between +0.00029 and +0.00045; the conclusion is not an artefact of mass weighting.

**Requested answers:**

- **P(public tie | OOF effect is real) = 0.4148** on a 4-movie panel. The tie was the single most
  likely outcome even with a genuine effect.
- **P(arm B > P0-B on private) = 0.82 / 0.91 / 0.96** at 32 / 64 / 128 movies.
- **Expected private delta ≈ +0.00043; 5th-percentile downside −0.00035 (32 movies), +0.00003
  (128 movies).**
- **Error correlation / diversity: poor.** The two artifacts share 92.9% of their edges (7.148%
  churn). This is not a diversified portfolio; it is one artifact and a near-copy.

**RECOMMENDATION: P0-B primary. Arm B as the second submission is defensible but nearly
worthless** — it is very likely (96% at 128 movies) to be microscopically better, with an expected
gain of **+0.0004**, two orders of magnitude below the +0.011 needed for 0.925. It costs nothing
to carry, so carry it, but **it is not a platform and no further work should compose onto it.**

**Suppress-all was not measured this cycle** and therefore does not enter the portfolio. Per the
standing rule it needs a measured interaction with arm B first.

---

## 2. LANE B — edge-FN first-failure census, correct substrate

199 crops. Composite 0.734150. Edge TP/FP/FN **104,331 / 14,994 / 24,552**.

| stage | 44b6 | 6bba | ALL | % | oracle Δ | owner | cardinality |
|---|---:|---:|---:|---:|---:|---|---|
| **det_never_detected** | 367 | 17,105 | **17,472** | **71.2%** | **+0.10399** | detector | no — needs a new node |
| assoc_enum_relaxed_starved | 359 | 3,180 | 3,539 | 14.4% | +0.02463 | association | **YES — pure re-aim** |
| assoc_enum_beyond_cap | 154 | 1,575 | 1,729 | 7.0% | +0.01208 | association | needs radius > 10 µm |
| assoc_bipartite_both_taken | 91 | 897 | 988 | 4.0% | +0.00691 | association | YES — 2-for-2 |
| assoc_bipartite_target_taken | 67 | 383 | 450 | 1.8% | +0.00315 | association | YES — swap |
| assoc_bipartite_source_taken | 45 | 326 | 371 | 1.5% | +0.00260 | association | YES — swap |
| assoc_synthetic_endpoint | 0 | 3 | 3 | 0.0% | +0.00002 | wrapper | no |

**Detection, not association, owns 71.2% of missed edges** — and 17,105 of the 17,472 are 6bba.

**Association-addressable (starved + bipartite): 5,348 FNs = 21.8% of all FN, oracle ceiling
+0.036860.** The 6bba share is 4,786 = **20.39% of 6bba FN**.

### Direct mechanistic confirmation of the starvation premise

Candidate surface across 199 crops: **3,886,574 tight pairs but only 2,292 relaxed.** The relaxed
10 µm pass is almost entirely starved out — by the time it runs, nearly every endpoint has been
consumed by the tight pass. Lane C's premise is not a hypothesis; it is visible in the counts.

### Required selector accuracy

| target | net TP gain d needed (SWAP) | as % of addressable |
|---:|---:|---:|
| +0.005 | 715 | 13.4% |
| +0.011 | 1,574 | 29.4% |
| **+0.015** | **2,148** | **40.2%** |
| +0.020 | 2,868 | 53.6% |

ADD regime (cardinality grows) is much harsher — at k = 4,000 selections, +0.015 needs precision
**0.736**; at k = 8,000, **0.584**; at k ≤ 2,000 it is unreachable at any precision. **The SWAP
regime is the only viable one**, because a swap removes a false edge as it adds a true one, so the
Jaccard denominator falls instead of rising.

**Correction to my own output:** the printed line
`relaxed-only surface 2292; starved FNs 3539 -> base rate 1.544066` is nonsense — a base rate
above 1. Starved FNs are *by definition* those never enumerated in the relaxed pass, so the
deployed relaxed surface is the wrong denominator. The honest figure is the union surface:
**5,348 addressable FNs against 3,888,866 candidate pairs = 0.001375, i.e. 727 candidates per
useful one.** That, not 1.54, is the number Gate C1 must beat.

---

## 3. LANE C0 — augmenting-path oracle: **BOTH GATES PASS**

40 crops, minimal-intervention alternating-path oracle: start from the deployed matching, apply
only moves that convert a GT FN into a TP, never break a true edge, add no nodes.

| | deployed | oracle |
|---|---:|---:|
| pooled composite | 0.742091 | **0.804663** |
| **oracle Δ** | | **+0.062572**  (gate ≥ +0.020 → **PASS**, 3.1×) |
| 44b6 FN | 285 | 162  (**43.16%** recovery), composite 0.884 → 0.939 |
| 6bba FN | 3,469 | 2,886  (**16.81%** recovery), composite 0.705 → 0.770 (gate ≥ 12% → **PASS**) |

Cardinality held: 803,724 → 803,618 edges (**−0.013%**). Nodes unchanged. **Intervention
efficiency 0.476 — 2.1 edges touched per GT edge gained.**

Both families improve, which is what distinguishes this from the first construction I tried.

### A construction I had to throw away

My first oracle re-solved each frame pair's assignment with a large GT bonus. It scored
+0.045657 — but **regressed 44b6 from 0.906 to 0.870**, because it churned 3,396 edges to gain 36
GT TPs and the arbitrary re-assignment among *unannotated* pairs perturbed gap-close and
short-track filtering. An oracle that hurts is measuring its own side-effects. Replaced with the
minimal-intervention version above, which touches 2.1 edges per useful gain instead of 94.3.

### The number that is NOT a base rate

**0.476 is intervention efficiency, not selector base rate.** The oracle only ever attempts
GT-correct moves. A GT-free proposer faces the Lane B denominator: **727 candidates per useful
one**. Gate C1 must compute the exchange-level denominator properly — candidates per useful
*exchange*, not per pair — before any ranker is built. That is the next step and it is not done.

---

## 4. Where this leaves the cycle

**The primary cycle target is not met and nothing is submittable.** Target was ≥+0.015 pooled OOF
with LCB ≥+0.008, 6bba ≥+0.015, 44b6 ≥−0.002. We have an oracle ceiling, not a candidate.

What the cycle established:

1. Arm B is worth **+0.0004**, not +0.008. It is not a platform.
2. **Detection owns 71.2% of missed edges** (+0.104 oracle). Association owns 21.8% (+0.037).
3. The augmenting-path primitive clears its realisability gate at **+0.0626 oracle with 16.81%
   6bba recovery at fixed cardinality** — comfortably above the +0.015 cycle target *if* a
   selector can be built.
4. The relaxed pass is starved almost to nothing (2,292 pairs vs 3,886,574 tight), confirming the
   mechanism.

**Next: Gate C1, the base-rate denominator.** 727 candidates per useful pair is the pessimistic
bound; the exchange-level rate is what matters and is unmeasured. Oracle-clears/selector-fails has
happened seven times, so C1 is mandatory before any ranker work.

**Also unresolved and material:** the +0.0079822 provenance. Until that is explained, every other
inherited OOF delta in the ledger is suspect by the same mechanism.
