---
id: 07-outputs/submissions
title: Submissions
area: 07-outputs
status: active
updated: '2026-08-16'
owner: biohub
links: []
tags:
- submissions
- public-score
---

> **Provenance:** migrated verbatim from `reports/submissions/CANDIDATE_LEDGER.md` on 2026-08-16 during the research-machine restructure. Body preserved unchanged below.

# Submission candidate ledger

Every candidate must record source/config hash, output hash, structural audit, OOF or
public-source justification, graph delta from its anchor, and the causal question it asks.
No candidate may be submitted without all six.

**Submissions consumed to date: 9** (was recorded as 6 before P0-A / P0-B / P0-CR landed).
Cycle slots: **3 of 3 used** — P0-A, P0-B, P0-CR. Two Kaggle daily slots were left DELIBERATELY
UNUSED on 2026-07-31 because nothing built that day had positive expected value.

## Standing prohibitions

Never submit: identical outputs; global threshold sweeps; family-routed systems; exploit
structures; anything tuned on the four visible placeholder movies.

## Quarantined public sources (structural exploits — never adopt)

At least ten highly-upvoted public notebooks inject a hub node at `t = -1000` with coordinates
`(-10000, -10000, -10000)` plus synthetic negative-time fork chains. One (`romanrozen/biohub-best-score`)
ships `BIOHUB_AUGMENT_HUB` defaulting **ON**. Reproducing "the top public notebook" walks directly
into the exploit. Full list in the research store.

## P0-A — exact clean 0.913 reproduction

| field | value |
|---|---|
| source | `saitejabandaruin/biohub-top-notebook-0-913` |
| source sha256 | `35681256f355bc98552d9cbdc4d208a9670657f18b7e7c4ce5637e6619e5e570` |
| size / cells | 148,047 bytes / 11 cells |
| identical copies | `nikitagajbhiye30/biohub-11`, `yiliu6111/biohub-v9-0913fork` (same sha256) |
| upstream author | `indarkarhana` (markdown stripped in the copies) |
| datasets | `pilkwang/biohub-deepcenter-unet3d-center-prior-v1`, `biohub-temporal-unet3d-seed314159-v1`, `biohub-tracking-support-pack-50ep-v1`, `pilkwang-public-dataset-for-notebooks-figures`, `thtennant/taaf-kaggle-source-share-fork` |
| hardware | T4, internet OFF |
| reported public | `0.913` (self-report, grade C until we reproduce it) |
| novel delta | per-frame retention guard (revert blend to primary detector when the blend loses >10% of peaks) |

**Source structural audit — PASS (2026-07-31).**

| check | result |
|---|---|
| `AUGMENT_HUB` | 0 occurrences |
| negative-time literal (`t=-N`, `-1000`) | 0 |
| exploit coordinates (`-10000`, `-9999`) | 1 hit, **benign** |
| synthetic hub / fork construction | 0 |
| `synthetic_fork` / `fake_div` / `artificial` | 0 |
| skip edges (`t+2`, `frame_skip`) | 0 |

The single `-9999` match is `_pl.Series([-999999.0], dtype=Float64)` — a polars compiled-backend
probe, the same environment check documented in `reports/ENVIRONMENT_TRAPS.md`. Not an exploit.

**Still required before this may consume a slot:** reproduction on our account, output-CSV
structural audit (t>=0, in-degree<=1, out-degree<=2, consecutive-frame edges only, no
cross-dataset edges, coordinates in-volume), and the output sha256.

**Causal question:** does an independently reproduced clean public pipeline reach 0.913 on our
account, giving us a platform to build on instead of reasoning from another participant's score?

## P0-B — clean base + source-locked reverse-time

Source-locked mechanism: `reports/external/v19_reverse_time_block.py.txt`
(extract sha256 `eb94e2745f270649865b6b89f8927450`). Reproduction arm at the source-defined
`w=0.20`; no harmonic fusion, no blend sweep. Historical public effect `0.912 -> 0.914`.
**Causal question:** does reverse-time association reproduce its reported gain on a clean base?

## P0-C — v122 + reverse-time

Same mechanism applied to our own 0.908 baseline, detector/ILP/wrapper otherwise fixed.
**Causal question:** does reverse-time help *our* pipeline, or only theirs?

## Status

| candidate | kernel | output sha256 | audit | submitted | public |
|---|---|---|---|---|---|
| P0-A | `biohub-p0a-clean-913-repro` v1 | `8c1605b5944d25e4…` | **PASS 10/10** (re-verified independently) | **ref 55136759** | **0.913** |
| P0-B | `biohub-p0b-clean-913-reverse-time` v1 | `4c285cae0c220a11…` | **PASS** (re-verified independently) | **ref 55136908** | **0.914 — NEW BEST** |
| P0-C | `biohub-p0c-v122-revtime-run` v1 | `3370222f811fddc9…` | **FAIL** — 1 node out of volume | no | — |
| **P0-CR** | `biohub-p0cr-v122-revtime-volguard` v1 | `435bf5d19d85563c…` | **PASS 10/10** (A1–A10, outside=0) | **ref 55147215** | **0.906** |

**Slots consumed this cycle: 3 of 3.** Slot 3 spent 2026-07-31 22:23 local on P0-CR.

**RESULT: P0-A reproduced 0.913 exactly; P0-B reached 0.914, our best public score (previous 0.908).** P0-B minus P0-A is +0.001 = exactly one unit of LB resolution, so the reverse-time mechanism is positive-but-unresolved, not established. Deployment base moves to P0-B.

### P0-A result detail
237,298 rows = 120,797 nodes + 116,501 edges, 4 datasets, 314 divisions, t 0–99,
z 0–63, y 0–254, x 0–254. Kernel COMPLETE in 1522 s on T4x2. Pushed notebook asserted
byte-identical to the audited public source at build time; upstream re-pulled first and had
not drifted.

### P0-B result detail
Delta vs P0-A: nodes **+64** net (1,994 added / 1,930 removed after discounting 1,875 sub-2 µm
coordinate-smoothing shifts), edges **+103** net, divisions **−9** (314→305), **749 parent
reassignments** (0.66% of 112,993 targets with a parent in both arms). No harmonic fusion, no
blend sweep.

### P0-C — FAILS audit, withheld
`44b6_0b24845f` node 15274, t=43, **z=64** (valid 0–63). This is a latent **v122** defect, not a
property of reverse-time: v122 line-fit-smooths coordinates without clamping to the volume and
already parks 369 nodes exactly on z=63. The agent correctly did **not** clamp or drop the node,
which would have altered the pipeline mid-experiment. Delta vs v122 baseline: nodes +40, edges
+105, divisions **+10** (313→323), 850 parent reassignments.

### Causal read across P0-B and P0-C
Division counts move in **opposite directions** on the two bases — **−9** on the clean 0.913 base,
**+10** on v122 — and under 1% of parent assignments move either way. The reverse-time mechanism's
effect is **base-dependent**, so the public 0.912→0.914 claim does not transfer to our pipeline on
structural evidence alone. That is precisely what the two submitted arms are measuring.

### Licence status — UNRESOLVED
Neither the API, the SDK response, nor the `.ipynb` metadata carries a licence field. Kaggle's
notebook default is Apache 2.0; recorded as **UNVERIFIED-DEFAULT-APACHE-2.0**. The notebook
self-declares attribution to `pilkwang/biohub-cell-tracking-two-seeds-logit-blend` and flags
`metric_hack_used: false`, `public_output_used: false`. **A human should confirm on the notebook
page in a browser.**

## P0-CR — v122 + reverse-time + line-fit volume guard (slot 3, 2026-07-31)

| field | value |
|---|---|
| kernel | `aryaarun07/biohub-p0cr-v122-revtime-volguard` v1, COMPLETE |
| output sha256 | `435bf5d19d85563c84140e162b3a84f476fdf2fc2cf6ab4b9b759986b9fb9fea` |
| size | 12,343,722 bytes · 237,168 rows = 120,673 nodes + 116,495 edges · 4 datasets · 323 divisions |
| structural audit | **PASS 10/10** — A1–A10, `outside=0`, max in-degree 1, max out-degree 2, t 0–99 |
| submission | **ref 55147215**, 2026-07-31 21:23 UTC |
| expectation | **0.908–0.912** — a causal probe, explicitly **not** a score climb |
| **RESULT** | **0.906** — BELOW the v122 base it was built on. Landed 2026-08-01 after ~4.5 h PENDING. |

**Causal question:** is reverse-time's effect base-dependent? P0-B (clean 0.913 base) gave +0.001
with divisions **−9**; this arm applies the identical source-locked mechanism to v122, where the
structural evidence had divisions moving **+10**. The two arms together are the only way to
separate "reverse-time helps" from "reverse-time helps *their* base".

**Why it cleared the slot-3 bar when nothing else did:** it is the sole candidate that was a
COMPLETED kernel with a passing audit, and it answers a still-open causal question (policy
priority 4). It repairs the exact defect that failed the original P0-C — v122 line-fit-smooths
coordinates without clamping, parking node 15274 of `44b6_0b24845f` at z=64 against a valid 0–63.

**Deliberately not submitted the same day:** P1 (P0-B + node budget) was built, audited PASS 10/10,
and measured at **−0.0000103** — a slot spent to confirm noise. P2 has a bolt-on ceiling of exactly
zero. P3 does not exist. Two daily slots were left unused rather than filled.

### P0-CR result — reverse-time is BASE-DEPENDENT and harmful on v122

| base | without reverse-time | with reverse-time | Δ | division count move |
|---|---:|---:|---:|---:|
| clean 0.913 | P0-A **0.913** | P0-B **0.914** | **+0.001** | −9 |
| v122 | v122 **0.908** | P0-CR **0.906** | **−0.002** | +10 |

**The signs differ.** The same source-locked mechanism at the same `w = 0.20` helps the clean base
by one LB quantum and *hurts* v122 by two. The causal question this slot was spent on is answered:
**reverse-time is not a general mechanism — its sign depends on the base it is applied to.**

The pre-registered structural read called the direction correctly before any score existed:
divisions moved **−9** on the clean base and **+10** on v122, and the arm whose divisions moved *up*
is the one that lost score. That is a genuine predictive success for structural pre-registration and
should be reused: **measure the division-count direction before spending a slot.**

**Confound, named honestly:** P0-CR carries the line-fit volume guard as well as reverse-time, so
strictly the −0.002 is (reverse-time + guard) vs neither. The guard repaired exactly **one**
out-of-volume node (`44b6_0b24845f` node 15274, z=64) out of ~120,673, so it cannot plausibly
account for −0.002 — but the arm is not a pure single-variable contrast and must not be quoted as one.

**Deployment: unchanged.** P0-B stays the base at **0.914**. What changes is the *rule*: do not
port reverse-time onto any other substrate without re-measuring, and treat a mechanism validated on
one base as unvalidated everywhere else until shown otherwise.

**Cycle slots: 3 of 3 consumed** (P0-A, P0-B, P0-CR). Two further Kaggle daily slots were left
deliberately unused on 2026-07-31 because nothing built that day had positive expected value.

---

## 2026-08-02 — ARM B SUBMITTED (slot 10), invariant fix verified in production

**Kaggle submission `55181562` — status pending at time of writing.** One slot spent today.

### Artifacts, hashes preserved

| kernel | sha256 | bytes | rows | nodes | edges |
|---|---|---:|---:|---:|---:|
| `p2_armb_baseline` | `4c285cae0c220a11b8ffdeeea1e02de86b5c1e7795daeb2bb27c073e1ab4ecec` | 12,359,118 | 237,465 | 120,861 | 116,604 |
| `p2_armb_flowgate` (**submitted**) | `83498f9e27d4453212f1d2e75b2b4999939733d1ce4fa183b0e4d670cc6ef6dd` | 12,382,751 | 237,916 | 121,003 | 116,913 |

**GATE 1 PASSES AGAIN.** The baseline hash is byte-identical to deployed P0-B *with the
out-degree fix applied*. Since the fix is shared code, this is the proof that it is **inert on the
P0-B path** — the arm-B delta remains fully attributable to the gate change, not to the repair.

### The invariant fix, verified on the real test set

| | failed run (2026-08-01) | this run |
|---|---:|---:|
| nodes | 121,003 | 121,003 |
| edges | 116,914 | **116,913** |
| max out-degree | **3** | **2** |
| nodes at out-degree > 2 | **1** | **0** |

**Net production effect of the fix: exactly one edge removed, zero nodes changed.** The
safe-division cap was slack on `6bba_05db0fb1`, so it was a pure removal rather than the
cap-bound swap seen locally on `44b6_a2bb48bb`. Independent audit (degrees keyed on
`(dataset, node_id)` per trap 24): max in-degree 1, 0 dangling, 0 non-consecutive, 0 duplicate
nodes or edges, 0 negative times or coordinates.

### Structural delta vs P0-B — matches the pre-registered prediction

Churn **7.148%** (documented caveat said 7.2%, "B is not a superset of A") · division parents
**305 → 318** (+13; correction 8's "P0-B has 305 forks" confirmed exactly).

| dataset | −P0B | +armB | shared | churn % |
|---|---:|---:|---:|---:|
| `44b6_0113de3b` | 465 | 584 | 24,147 | 4.262 |
| `44b6_0b24845f` | 883 | 895 | 17,303 | 9.777 |
| `6bba_05b6850b` | 47 | 78 | 5,893 | 2.104 |
| `6bba_05db0fb1` | 2,618 | 2,765 | 65,248 | 7.932 |

### The "one property to watch" is now CLOSED empirically

`PRIMITIVE_MATRIX` flagged that arm B gates on the residual, so admissible `raw` is bounded by
`gate_um + |flow|` and could in principle exceed the 10 µm relaxed radius or the 14 µm
`OUTPUT_EDGE_MAX_UM`. Measured on the artifact:

| | max | p99 | p50 | >6 µm | >10 µm | >14 µm |
|---|---:|---:|---:|---:|---:|---:|
| P0-B | 7.312 µm | 4.083 | 1.724 | 24 | **0** | **0** |
| arm B | 7.846 µm | 4.143 | 1.724 | 49 | **0** | **0** |

Arm B does lengthen the tail (max +0.534 µm, >6 µm count doubles 24 → 49), but **nothing crosses
10 µm, let alone 14 µm.** The risk is real in principle and empty in practice on this substrate.

**Structural pre-registration read:** divisions moved **+13** (up). The P0-CR precedent — "the arm
whose divisions moved *up* is the one that lost score" — is the one signal pointing against arm B
here. It is a single prior observation on a different substrate, and arm B's divisions rise because
the relink leaves a different target set unlinked rather than because the division logic changed;
but it is on the record **before** the score lands, which is the point.

### RESULT — arm B `55181562` scored **0.914**. Public delta ZERO.

| submission | arm | public |
|---|---|---:|
| `55136908` | P0-B | **0.914** |
| `55181562` | arm B solo | **0.914** |

Expected ≈ +0.0045 from an OOF→public slope of ~0.56. Delivered **< 0.001**. Slots consumed: 10.

**Retain P0-B. Do not adopt arm B. Do not discard it either.**

The build was flawless — baseline byte-identical to deployed P0-B, audit clean, churn 7.148%
against a pre-registered 7.2%. This is not a deployment defect. Arm B moved 7.148% of edges and the
score did not move.

**The instrument is the likely explanation.** GT annotation covers **0.655%** of estimated cells on
44b6 and **8.529%** on 6bba — a 13× family asymmetry, median 209 and 800 annotated edges per crop,
128,883 annotated edges across all 199 training crops. Edge Jaccard is computed only over annotated
cells. The 4 public movies are therefore scored over roughly **2,000 GT edges**: one edge ≈ 0.05% of
Jaccard, and +0.008 needs ~16 net correct annotated edges. Arm B's ~8,335 churned edges touch maybe
~143 annotated ones, so the required precision edge (~56/44) is well inside the noise of which cells
happen to be annotated.

**Our OOF instrument is ~64× more sensitive than the public leaderboard.** Arm B's
P(Δ>0) = 1.000 over 144 crops may still be worth ~+0.008 privately, where there are more movies.
Public flatness is not evidence of worthlessness.

**Standing rule from this slot: never spend a submission to resolve an effect smaller than ~0.005.**
The LB is a smoke test for large effects; OOF is the instrument.

Structural pre-registration note: divisions moved **+13 (up)**, and the P0-CR precedent said the arm
whose divisions move up is the one that loses score. It did not lose — it went flat. The precedent
is not confirmed and not refuted.

---

## 2026-08-05 — P3 HARMONIC RE-BASELINE SUBMITTED (slot 11). Kaggle `55274582`, PENDING.

**Context: the field moved and we did not.** Public top is now **0.948**; 0.924 is ~15th. P0-B at
0.914 is well off the pace and the 0.920–0.925 target was obsolete.

**One semantic edit on P0-B.** Forward/reverse association fusion changes from an arithmetic mean
of logits to a weighted **harmonic mean in probability space**, renormalised and affinely rescaled
back onto the forward logit scale. λ = 0.20 unchanged. Harmonic is dominated by the **lower** of
the two directions, so a link must be supported both ways; the arithmetic mean lets one confident
direction carry a bad edge.

Built by swapping the single-quoted `_bi_new` runtime-patch literal wholesale — our 2005-char
literal for the 3518-char harmonic one, shared prefix 1702 — so the patch text is
transcription-exact. Cell 5 grew by exactly **1513 = 3518 − 2005**, confirming a clean single
swap with no collateral edit.

**Provenance.** Rule transcribed from public **CC0** notebook
`yusuketogashi/no-hack-biohub-cell-another-approch-3rd` v18 ("Biohub 145 | Bidirectional Harmonic
Probability"), via `raykkretzschmar/biohub-harmonic-bidirectional-association-v1` (id_no
129697527). Credit for the rule: Yusuke Togashi.

**Also carries both invariant fixes** (safe-division out-degree, gap-close in-degree) plus the
export assertions. Harmonic rewrites the edge logits, shifting which targets the relink leaves
unlinked — the exact mechanism that tripped the safe-division defect and blocked the first arm-B
run. The assertions passed in-kernel.

| | P0-B | P3 harmonic |
|---|---:|---:|
| sha256 | `4c285cae0c220a11…` | `3e98739f5c46f2dd…` |
| rows | 237,465 | 239,886 |
| nodes | 120,861 | **122,083** |
| edges | 116,604 | **117,803** |
| divisions | 305 | 307 |
| public | **0.914** | pending |

Structural audit **PASS 10/10**: max in-degree 1, max out-degree 2, 0 outside-volume, 0
cross-dataset, 0 non-consecutive, 0 duplicate `(dataset, node_id)`.

**Interaction warning, recorded before the score lands:** arm B's relink cost consumes `prob`,
which this edit rewrites. The two **must** be measured together and never assumed additive. This
arm is harmonic **alone**.

**Consequence for the queued work:** the C0-FULL / C1 association programme was measured against
P0-B. Harmonic changes the edge-FN population that Lane B's base rates and Lane C's ceiling are
computed from, so **the overnight run must not proceed on the old base** — that would repeat the
substrate error with a different substrate.

### RESULT — P3 harmonic `55274582` scored **0.915**. New best, +0.001 over P0-B.

| submission | arm | public |
|---|---|---:|
| `55136908` | P0-B | 0.914 |
| `55181562` | arm B solo | 0.914 |
| `55274582` | **P3 harmonic** | **0.915** |

**Harmonic mutual-support fusion is worth +0.001 on our base** — the same magnitude the arithmetic
reverse-time blend gave over clean913 (0.913 → 0.914). It is a marginal refinement of that
mechanism, not a different class of thing.

**This falsifies the re-baselining theory.** The working hypothesis was that the field's move to
0.93–0.948 was driven by this public CC0 rule, and that adopting it would land us near 0.93.
It did not. **Whatever is separating the 0.93+ teams from us is NOT harmonic fusion.**

That is a genuinely useful negative: it means the leaders hold something the public notebooks do
not, which supports treating the remaining gap as a research problem rather than an adoption
problem.

**Prediction scored.** Range given before the result: 0.913–0.928, modal band 0.916–0.921 at 55%,
low band 0.913–0.914 at 25%. Actual **0.915** — inside the full range, **below the modal band**.
Third consecutive optimistic central estimate (arm B: predicted +0.0045, got 0.000).

The reasoning that worked was **structural churn**: "arm B churned 7.148% and moved 0.000; P3
churns 4.368%, so it is hard to argue it buys +0.016." That pointed correctly at a small gain and
should have been weighted more heavily. The argument that pushed the estimate up — net-additive
+1% nodes and edges — carried no predictive signal. **Churn magnitude against a known-null
reference is the better predictor; net cardinality change is not.**

**New platform: P3 harmonic at 0.915.** Slots consumed: 11. Gap to leader (0.948): **0.033**.
