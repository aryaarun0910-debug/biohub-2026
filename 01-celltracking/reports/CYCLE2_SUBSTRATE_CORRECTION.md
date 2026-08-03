# Cycle 2 — substrate correction, P0 parity, corrected Lane B

**Date:** 2026-08-03 · **Substrate:** WS-F **pre-wrapper** pregraphs (`f0_pregraphs` 71 +
`f1_pregraphs` 128 = **199 crops**), carrying `edge_prob` and the full relink node population ·
complete wrapper · exact scorer · **No GPU, no submission, 0 slots.**

Artifact: `reports/inventory/edge_fn_census_prewrapper.json`.
Reproduce: `scripts/edge_fn_census.py --substrate prewrapper --workers 6`.

---

## 1. LANE P0 — RESOLVED. The record was right; my measurement was the artifact.

| measurement | crops | pooled delta |
|---|---:|---:|
| WS-F original (`wsf_WSF_SUMMARY.json`) | 144 | **+0.0079822** |
| **This cycle, correct substrate** | **199** | **+0.0084877** |
| My retracted post-wrapper run | 199 | +0.000435 ❌ |

**Parity achieved.** The residual +0.0005 between the two valid numbers is **hypothesis A**
(crop population): WS-F scheduled cheapest-first and covered 144 of 199, explicitly noting it
"under-weights the largest crops". Adding the remaining 55 crops moves the delta *up* slightly.
No bisection was required — hypotheses **B and C** were already documented in
`reports/inventory/wsf_ROUTE1_VERDICT.json` before this cycle began.

**Arm B is a real, bilateral, statistically solid mechanism:**

| | armA | armB | delta |
|---|---:|---:|---:|
| pooled | 0.7320974 | 0.7405851 | **+0.0084877** |
| 44b6 | 0.9005342 | 0.9144462 | **+0.0139120** |
| 6bba | 0.7029468 | 0.7106376 | **+0.0076907** |

Per-crop raw-J delta mean +0.0104, median +0.0043, P(Δ>0) = 0.618.
**Bootstrap 95% CI on the mean: [+0.00745, +0.01365] — excludes zero.**
Edge counts: TP 104,094 → 104,737 (**+643**), FP 15,067 → 14,288 (**−779**), FN −643.

### What this does NOT resolve

Arm B is genuinely worth ~**+0.0085 OOF, on both families, with a CI clear of zero** — and it
moved the public leaderboard by **less than 0.001**. Both of my previous explanations are now
dead: it is not "the panel cannot resolve it" (that was cycle 1) and it is not "the effect was
never there" (that was cycle 2's retracted claim). **The OOF→public gap is the outstanding
question in this project and deserves its own lane.**

---

## 2. Second invariant bug — a live deployment defect, found by the correct substrate

The full-corpus run died at crop 185/199 on the assertion added last cycle:

```
degree invariant violated after gap close + gap2 recovery: in-degree>1 on [3863]
```

`close_single_frame_gaps` kept **two disjoint consumption sets** — `used_starts` for targets,
`used_isolated` for reused middles — and **neither admission guard consulted the other**. A node
consumed as a TARGET is recorded in `used_starts`/`incoming` but never in `used_isolated`, so it
remained eligible as a reused middle and took a **second parent**. Frames are walked in ascending
`t`, so target-then-middle is the reachable ordering.

Same defect class as the safe-division out-degree bug: **admission dedupes against one
bookkeeping record while a second path writes a different one.**

Fixed at admission against `incoming`, the authoritative "already has a parent" record. Locked by
`test_gap_close_cannot_give_a_node_two_parents`, which fails without the fix. Tests 50 → 51.

**It never fired on the post-wrapper cache**, which lacks 83,260 of the nodes the gap-closer
actually sees. The wrong substrate was masking a real defect — the substrate error cost more than
bad numbers. Deployed P0-B and arm B artifacts are unaffected (audited max in-degree 1), so the
fix should be inert on a rebuild, but that is an inference from the audit, not a re-run.

---

## 3. LANE B — corrected first-failure census, 199 crops

Composite 0.740585 · edge TP/FP/FN **104,737 / 14,288 / 24,146** ·
relink candidate surface **3,985,523** pairs (tight 3,907,924, **relaxed 77,599**).

| stage | 44b6 | 6bba | ALL | % | oracle Δ | owner | cardinality |
|---|---:|---:|---:|---:|---:|---|---|
| **det_never_detected** | 369 | 17,075 | **17,444** | **72.2%** | **+0.10332** | detector | no — needs a node |
| assoc_enum_relaxed_starved | 249 | 2,874 | 3,123 | 12.9% | +0.02184 | association | **YES — pure re-aim** |
| assoc_enum_beyond_cap | 129 | 1,557 | 1,686 | 7.0% | +0.01183 | association | needs radius > 10 µm |
| assoc_bipartite_both_taken | 68 | 686 | 754 | 3.1% | +0.00530 | association | YES — 2-for-2 |
| assoc_bipartite_target_taken | 74 | 423 | 497 | 2.1% | +0.00349 | association | YES — swap |
| assoc_bipartite_source_taken | 61 | 344 | 405 | 1.7% | +0.00285 | association | YES — swap |
| assoc_synthetic_endpoint | 34 | 179 | 213 | 0.9% | +0.00150 | wrapper | no |
| final_endpoint_only | 0 | 24 | 24 | 0.1% | +0.00017 | unattributed | unknown |

**Detection owns 72.2% of missed edges** (+0.10332 oracle), 17,075 of 17,444 in 6bba.
**Association-addressable = 4,779 FNs = 19.8%, ceiling +0.032931**; the 6bba share is 4,327 =
**18.68% of 6bba FN**.

The stage distribution proved robust to the substrate error (72.2% vs the invalid run's 71.2%),
but the **denominators did not**, and those are what C1 needs.

### The base rate that the wrong substrate destroyed

| | wrong substrate | correct substrate |
|---|---:|---:|
| relaxed surface | 2,292 | **77,599** |
| starvation base rate | 1.544 (impossible) | **0.040245** |

On the post-wrapper cache the relaxed surface was almost empty, producing a base rate above 1 —
which I flagged as nonsense at the time without drawing the right conclusion from it.

**The relaxed-starvation class has a 4.02% base rate — 1 useful in 25, and 33× better than the
0.12% (1 in 834) rate of the undifferentiated surface.** That is the single most encouraging
number for Gate C1, and it only exists on the correct substrate.

### Required selector accuracy

| target | net TP gain d (SWAP) | % of addressable |
|---:|---:|---:|
| +0.005 | 712 | 14.9% |
| +0.011 | 1,567 | 32.8% |
| **+0.015** | **2,138** | **44.7%** |
| +0.020 | 2,856 | 59.8% |

ADD regime remains far harsher (k=4,000 needs q=0.736 for +0.015; unreachable below k=2,000).
**SWAP is the only viable regime.**

---

## 4. Against the cycle target

Target: pooled ≥ +0.015, bootstrap LCB ≥ +0.008, 6bba ≥ +0.015, 44b6 ≥ −0.002.

| | pooled | LCB | 6bba | 44b6 | verdict |
|---|---:|---:|---:|---:|---|
| **arm B alone** | +0.00849 | +0.00745 | +0.00769 | +0.01391 | **FAILS** pooled and 6bba |

Arm B clears 44b6 comfortably and misses on pooled and 6bba. It is roughly **half** the required
effect. The association ceiling (+0.0329) is above target, so a selector capturing **~45%** of it
would qualify — that is precisely the C1 question, and C1 is not yet run.

---

## 5. State of the lanes

| lane | status |
|---|---|
| **P0** | **COMPLETE** — parity confirmed, record vindicated, my number retracted |
| **B** | **COMPLETE** on the correct substrate |
| **C0-FULL** | not run — instrument retargeted, ~4 h of CPU pending |
| **C1** | not run — blocked on C0-FULL |
| **D0** | not run |

**Nothing is submittable and no candidate meets the cycle target.** Detection (72.2%, +0.1033)
is now clearly the larger prize, but it is also the one that cannot hold node count fixed — and
`total_node_ratio` is negative, so added nodes surrender an adjustment bonus before earning
anything. D0 must price that explicitly.
