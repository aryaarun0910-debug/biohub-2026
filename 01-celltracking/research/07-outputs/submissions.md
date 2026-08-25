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
record_kind: ledger
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

**~~Standing rule from this slot: never spend a submission to resolve an effect smaller than ~0.005.
The LB is a smoke test for large effects; OOF is the instrument.~~ STRUCK 2026-08-18.**
This rule assumed OOF *was* an instrument. It is not: LB-vs-local calibration over n=6 known
anchors gives Spearman +0.500 (p=0.333), slope 0.066, and the local substrate is anti-informative
(Pearson -0.199 across the four non-armB configs). Measured MDE at 80% power: fold 0 +/-0.0043,
fold 1 +/-0.0057, pooled +/-0.0049 -- so the +0.005 bar sat BELOW fold 1's own detection limit.
The LB resolves 0.001, measures the shipped pipeline, and allows ~35 tests/week.
**The leaderboard is now the instrument.** See `internal-reports/instrument_repair_2026-08-18.md`.

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

---

## 2026-08-17 — `55585140` P3 + arm-B motion-gate — **PENDING**

Submitted 19:59 UTC (kernel `aryaarun07/biohub-p3-armb`, version 1). Still
`SubmissionStatus.PENDING` at 21:29 (1h30m). Slots consumed: 12.

**Basis.** Paired deployment-substrate LOEO on the official `tracking_cellmot` scorer, only
`BIOHUB_ARMB_FLOW_GATE` differing: 44b6 0.9037 → 0.9181 (+0.0144); 6bba 0.7051 → 0.7141
(+0.0090). Bilaterally positive, min-fold above the +0.005 bar.

**Prediction (recorded before the score).** 0.916–0.920. Stated caution, consistent with this
log's own lesson that LOEO deltas have repeatedly over-predicted LB movement (arm B solo:
predicted +0.0045, got 0.000; P3: predicted modal 0.916–0.921, got 0.915): the honest low band
is **0.915–0.916 at ~40%**, i.e. a real chance this reads as flat. A result at 0.915 would mean
a bilateral LOEO gain of +0.009/+0.014 did not transfer — which would itself be the fourth
consecutive optimistic central estimate and a substantive finding about LOEO→LB transfer.

**Score: 0.915 — IDENTICAL to P3 alone. Delta +0.000.**

**Prediction scored.** Recorded band 0.916–0.920 with an explicit low band of 0.915–0.916 at ~40%.
Actual **0.915** — in the low band, **below** the stated central estimate. That is the **fourth
consecutive optimistic central estimate** (arm B solo: predicted +0.0045, got 0.000; P3: predicted
modal 0.916–0.921, got 0.915; motion-gate: predicted 0.916–0.920, got 0.915).

**This is a methodology result, not just a lever result.** The motion-gate was measured on our
*best* instrument — a clean **paired deployment-substrate LOEO** on the official scorer, identical
crops, single toggle, bilaterally positive (+0.0144 / +0.0090, min-fold above the +0.005 bar).
That instrument was adopted specifically to fix the substrate mismatch that discredited earlier
estimates. **It still failed to transfer.** A bilateral LOEO gain of +0.009 to +0.014 produced
+0.000 on the leaderboard.

Consequences:
- **The +0.005 bilateral LOEO bar is not a sufficient promotion gate.** It has now passed a lever
  that delivered nothing. Every LOEO-only projection in the portfolio should be discounted hard.
- The open diagnostic: LOEO evaluates held-out-embryo crops from `data/train` using per-fold
  weights, while the LB scores `data/test` crops. Which part of that gap kills the signal — the
  crop population or the weights — is **not yet established** and is the single highest-value
  methodology question we have.
- Slots consumed: 12. Platform unchanged at **P3 harmonic 0.915**. Gap to leader (0.950): 0.035.

---

## 2026-08-18 — `p5_divfix` (kernel pushed, NOT yet submitted) — prediction recorded first

**What it is.** P3 harmonic + three COUPLED division changes: `BIOHUB_ILP_DIVISION_WEIGHT=0.55`
(L1), `BIOHUB_OUTPUT_SAFE_DIVISIONS=0` (patch off), `BIOHUB_RESTORE_LEARNED_DIVISIONS=1` plus the
`restore_learned_divisions()` wrapper stage (L3). Kernel `aryaarun07/biohub-p5-divfix` v1.

**Basis (measured, not projected).** Kernel `biohub-p4-preilp-loeo-f1` v2 exported the candidate
graph the ILP is handed: 176,835 fold-1 sources with out-degree ≥ 2 offered, **zero** divisions
emitted. Of 125 GT divisions, 34 mothers matched a ≥2-candidate source and **29 had BOTH true
daughters offered**, at median `edge_prob` **0.9188**. Fold-1 division-term arithmetic:
L1 alone +0.0046; **L1 + patch-OFF +0.0227**; at the 61-case ceiling +0.0483.

**Prediction, recorded BEFORE the score, applying the ledger's own low-band-as-central rule**
(four consecutive optimistic central estimates; realisation ratio ≈0.083):

| band | range | weight |
|---|---|---|
| **central (was my low band)** | **0.915–0.919** | **~45%** |
| upside | 0.920–0.930 | ~30% |
| flat/negative | 0.912–0.915 | ~25% |

**Why the caution despite a measured mechanism.** (i) The 29 offered divisions are a fold-1
count on the LEAKY arm; the hidden test may differ. (ii) `edge_prob` is a **share**, not a
calibrated probability — the loss is a bare column softmax and provably shift-invariant, so
0.9188 means "best available parent by a wide margin", not "92% likely real". (iii) The three
changes are coupled and untested TOGETHER; if L3 does not fire, this is P3-minus-safe-divisions
and should land near **0.914** (−0.0007). (iv) Divisions are only 0.117% of GT edges, so the
edge term is unchanged by construction — all movement must come from the 0.1×divJ term.

**Falsification.** ≤0.915 with `restore_div_restored > 0` in `run_stats.csv` ⇒ the division term
does not transfer to the hidden set and the whole division lane closes.
`restore_div_restored == 0` ⇒ L3 never fired; the test is void, not negative — debug and re-run.

**SUBMITTED 2026-08-18 as ref `55616685`** (kernel `aryaarun07/biohub-p5-divfix` v1).
Slots consumed: 13.

**Pre-submit verification, all passed:**
- `safe_divisions_added` = **0** — the patch is genuinely off
- `restore_div_sources_seen` = **844** — L1 worked: the ILP now EMITS divisions where it
  previously emitted none (176,835 offered / 0 taken before this change)
- `restore_div_restored` = **703** — L3 worked: the motion relink no longer deletes them
- `division_like_sources` = **703**, exactly matching — so **every** output division now
  originates in the ILP, none in the post-hoc patch
- structural audit **PASS 10/10**, max in-degree 1, max out-degree 2, artifact sha256
  `76efe6a21b8dd137...`

**OBSERVATION RECORDED AFTER THE PREDICTION, BEFORE THE SCORE** (stated separately so the
prediction above stands exactly as written): **703 divisions across 4 test crops = ~176/crop.**
Annotated GT holds 151 divisions across 199 train crops (~0.76/crop); at ~2.8% annotation that
implies a true rate near ~27/crop. **We are emitting roughly 6× the estimated true division
rate.** Division FPs are charged only on annotated cells, which damps the penalty, but this is a
material over-division risk that the recorded prediction did not account for. If the score comes
back flat or negative, over-division — not the mechanism — is the first hypothesis, and the fix
is a higher `BIOHUB_ILP_DIVISION_WEIGHT` (0.55 → 0.7–0.8), which tightens the price without
touching the architecture.

**Score: 0.915 — FLAT. Identical to P3 alone and to P3+armB.**

**Prediction scored.** Recorded central band 0.915–0.919 (~45%), upside 0.920–0.930 (~30%),
flat/negative 0.912–0.915 (~25%). Actual **0.915** — the bottom edge of the central band, i.e.
indistinguishable from flat. Promoting the low band to central (the correction adopted this
session) **worked**: this is the first prediction in five that was not optimistic. The method is
now 1/1; the previous four central estimates were all too high.

### The finding is the three-way tie, not this lever

| ref | config | divisions (4 crops) | edge churn vs P3 | public |
|---|---|---|---|---|
| `55274582` | P3 harmonic | 307 | — | **0.915** |
| `55585140` | P3 + armB | 328 | **~6%** (8,150 admitted / 9,373 excluded) | **0.915** |
| `55616685` | P3 + division fix | **703** | safe-div patch OFF | **0.915** |

**Three materially different graphs — differing in edges AND in a 2.3× change in division count —
return the identical score to three decimals.**

**This is not explainable by rounding.** Moving 0.915 → 0.916 requires +0.0005. The predicted
fold-1 division-term gain was **+0.0227**, roughly **45×** the rounding threshold. If the mechanism
had transferred at anything like its measured magnitude, the score would have moved. It did not
move at all.

**What is now established:** the division mechanism, though **proven to fire** (844 ILP division
sources, 703 restored, 0 from the patch, audit PASS 10/10), does not convert to public-LB score.
Two surviving explanations, not yet separated:
1. **Over-division** — 703 divisions over 4 crops ≈ 176/crop against an estimated true ~27/crop
   (~6×). FPs swamp `divJ = TP/(TP+FP+FN)`, leaving the 0.1×divJ term at ~0 either way.
   (Flagged in this ledger BEFORE the score.)
2. **The division term does not transfer** to the hidden set at all.

**Separating them costs one slot:** re-run at `BIOHUB_ILP_DIVISION_WEIGHT = 0.75` (tighter price,
fewer and higher-confidence divisions). If still 0.915, explanation 2 holds and the division lane
closes for good.

### A cheaper, higher-value probe now available

The three-way tie is itself an unplanned and unusually strong instrument result — stronger than
the 1-slot determinism probe `instrument_repair_2026-08-18.md` recommended, because it varies the
input instead of repeating it. It licenses a sharper question: **is the public LB responsive to
our submissions at all?** The decisive test is a deliberately DEGRADED submission (e.g. drop the
motion relink, or halve the detection threshold) which should move the score by a large,
unmistakable margin. If that also returns 0.915, the fault is in the measurement chain, not in any
lever — and every conclusion drawn from LB scores this session needs revisiting. **Run this before
spending any further slot on a lever.** One slot, unambiguous outcome either way.

Slots consumed: 13. Platform unchanged at **P3 harmonic 0.915**. Gap to leader (0.951): 0.036.

---

## 2026-08-19 — `p6_control_degraded` — INSTRUMENT CONTROL (not a lever). Prediction recorded first.

**Why.** Three materially different graphs returned **exactly 0.915**: P3 (307 divisions),
P3+armB (328 divisions, ~6% edge churn), P3+divfix (703 divisions, safe-div patch off). The
divfix prediction was **+0.0227 — ~45× the 0.0005 rounding threshold** — and nothing moved.
Before spending another slot on any lever, establish that the public LB **responds to our
submissions at all.**

**What shipped.** `BIOHUB_DET_THRESHOLD` **0.96875 → 0.999** (kernel
`aryaarun07/biohub-p6-control-degraded` v1, verified in the built notebook). This sharply cuts
detections, so node recall and edge TP must fall. Chosen over dropping the motion relink because
a relink toggle could plausibly be score-NEUTRAL, which would leave the test ambiguous; fewer
detections cannot be neutral. It is also cheaper to run, not more expensive.

**PREDICTION, recorded before the score — this one is deliberately NOT hedged.** A working
measurement chain must return **≤ 0.90, and most likely 0.80–0.89**. I would be surprised by
anything above 0.905.

**Interpretation, fixed in advance:**
- **Score drops sharply (≤ 0.90)** → the chain is SOUND. Our levers really are sub-0.0005 on the
  hidden set, the plateau is real, and the three-way tie means those three graphs genuinely score
  the same. Proceed to the abstention main line (+0.010 to +0.040).
- **Score returns 0.915 (or barely moves)** → the fault is in the MEASUREMENT CHAIN, not in any
  lever. **Every LB-based conclusion of 2026-08-18 needs revisiting**, including the arm-B kill and
  the division kill, and the LOEO→LB diagnosis reopens from the other end. First suspects would be:
  the submitted artifact not being the scored artifact, the score being dominated by a component
  our edits do not reach, or LB caching.

Note this is a stronger test than the "resubmit unchanged" determinism probe originally planned in
`instrument_repair_2026-08-18.md`, because it varies the input rather than repeating it.

**PREDICTION CORRECTED BEFORE SUBMITTING** (the original reasoning was wrong; recorded openly
rather than quietly replaced). I predicted "≤ 0.90, most likely 0.80–0.89" on the assumption that
cutting detections cuts score proportionally. **It does not.** Measured locally on the placeholder
substrate with the official scorer:

| artifact | node_recall | divJ | local SCORE |
|---|---|---|---|
| `p3_harmonic` (LB 0.915) | 0.9841 | — | 0.8907 |
| `p6_control_degraded` | **0.9147** | 0.0000 (TP=0/FP=8/FN=3) | **0.8816** |

Degradation confirmed structurally: **28.0% of nodes and 28.4% of edges removed**
(122,214 → 87,945 nodes), one crop losing 70%. Yet the score falls only **−0.0091**, because the
node-count multiplier is **unclamped above 1** — under-producing earns a bonus that partially
offsets the lost true positives. A 28% cut is worth ~9 quanta, not ~100.

**Corrected prediction: LB ≈ 0.906 if the local→LB offset is additive (0.915 − 0.0091), possibly
as low as ~0.891.** The discriminating question is therefore NOT the absolute value but simply:
**does the score move at all?** Three consecutive submissions returned exactly 0.915; any movement
of ≥ 1 quantum shows the chain responds.

- **Moves (any amount)** → chain SOUND. The three-way tie means those graphs genuinely scored the
  same, our levers are real but sub-threshold on the hidden set, and the abstention lane proceeds.
- **Returns exactly 0.915** → chain FAULT. A 28% node cut cannot be invisible. Every LB conclusion
  of 2026-08-18 reopens.

**Caveat carried from `cross_synthesis_2026-08-19.md`:** `evaluate.py:73-76` intersects
`pred_names & gt_names`, so **missing datasets are silently SKIPPED, not zeroed** — a truncated
submission would score high, not low. Worth remembering if any future artifact looks anomalously good.

**Score: 0.883. THE CHAIN IS SOUND.**

**Prediction scored.** Corrected band was "≈ 0.906 additive, possibly ~0.891", with the
discriminating question stated as *does it move at all*. Actual **0.883** — it moved **−0.032
(32 quanta)**, further than either bracket. The un-hedged pre-registration was right in direction
and conservative in magnitude.

### The public leaderboard responds to our submissions. Everything from 2026-08-18 stands.

- The three-way tie (P3 / P3+armB / P3+divfix all exactly 0.915) is **REAL**, not an artifact.
  Those three graphs genuinely score the same on the hidden set.
- **The arm-B kill and the division kill both STAND.**
- No chain-fault story is needed. The `cross_synthesis` scepticism about over-reading the tie was
  correct.

### The far more useful finding: which lever CLASSES transfer

| submission | local (placeholder) | LB | local Δ | LB Δ | transfer |
|---|---|---|---|---|---|
| P3 harmonic | 0.8907 | 0.915 | — | — | — |
| **control** (28% of nodes cut) | 0.8816 | **0.883** | **−0.0091** | **−0.0320** | **3.5× AMPLIFIED** |
| divfix (division economics) | 0.8978 | 0.915 | **+0.0071** | **+0.0000** | **0× — none** |

**A detection-surface change transfers at 3.5× its local magnitude. A division-term change does not
transfer at all.** Both measured on the same substrates with the same scorer.

**Why, and it is not a paradox.** The placeholder crops hold **3 GT divisions between them**
(2 of the 4 have zero, and two sit in the 1st–2nd percentile of annotation density). If the hidden
test resembles them — and the transfer ratios say it does for detection — then the
`0.1 x division_jaccard` term has almost no headroom to win there, while `adj_edge_jaccard`
responds strongly. That reconciles every LB result we hold:

- edge-rewiring levers (arm B): permute a fixed candidate set → **0.000**
- division levers (divfix): a term with ~no hidden-set headroom → **0.000**
- detection-surface changes (control): → **−0.032**, amplified

**Consequence — this is the campaign's operating rule from here:** *only levers that change the
detection surface or the candidate set have demonstrated LB transfer.* Which is precisely where
`error_atlas_2026-08-19` independently put the bottleneck: **17,067 detectable GT edges
(15.65%) are never nominated as candidates**, while a perfect solver over today's candidates is
worth +0.0012.

**Also now calibrated:** the placeholder substrate **understates** detection-class effects by ~3.5×.
It is usable as a directional instrument for that class, with that factor stated — a real
improvement on "we hold no validated instrument", and it costs no slots.

Slots consumed: 14. Platform unchanged at **P3 harmonic 0.915**.

---

## 2026-08-19 — `p7_cleanedge` — DECISION RULE PRE-REGISTERED BEFORE THE RESULT

Kernel `aryaarun07/biohub-p7-cleanedge` v1 (pushed, running). P3 harmonic with the edge predictor
swapped for `leevvin/biohub-movie-heldout-edge-predictor-v1` (CC0, 195-movie held-out retrain,
host-verified strict=True 136/136 into our exact `UNetNodeTransformer`). Zero training cost.

**Why this is the right CLASS.** The edge head's column softmax > 0.5 defines every candidate
edge, so replacing it is a **candidate-set** change. Measured transfer law (2026-08-19):

| lever class | local Δ | LB Δ |
|---|---|---|
| detection-surface (control, 28% node cut) | −0.0091 | **−0.0320** (3.5× amplified) |
| division term (divfix) | +0.0071 | 0.000 |
| edge permutation (arm B) | — | 0.000 |

**PRE-REGISTERED DECISION RULE (fixed before any result is seen):**
- Score `p7` locally on the four placeholder crops with the official scorer; compare against
  **P3 harmonic's 0.8907** on the identical substrate.
- **SUBMIT if local ≥ 0.8907** (not worse than P3). Slots are abundant (171 left, 5/day) and the
  binding constraint is calendar, not slots — so the value of an LB reading on a
  *candidate-set-class* lever exceeds the cost of a slot.
- **DO NOT SUBMIT if local < 0.8907.** A checkpoint that is worse on the substrate it was
  explicitly held out from is not worth a reading.
- Audit must PASS 10/10 regardless; a failing audit blocks submission unconditionally.

**Instrument note, and its limit.** leevvin held out exactly these four crops, so this checkpoint
is **uncontaminated on the scoring substrate** — the first time that has been true for us. But the
detector, DeepCenter and the `seed314159` secondary are unchanged and still contaminated, so the
**absolute** local number remains inflated. The **delta vs P3 on the identical substrate** is the
honest quantity, and per the transfer law it should be read as roughly ×3.5 to the LB.

**DECISION: DO NOT SUBMIT. Rule applied as pre-registered.**

p7 v3 completed but produced **400 nodes / 368 edges across all four crops**, against P3's
**122,214 nodes** — a **99.7% collapse**. Local score and audit both moot; the rule said
"do not submit if below 0.8907", and this is far below.

**Diagnosis (informative, not a bug).** `edge_predictor_best.pth` is the **entire model**
state_dict — 136 tensors = the `TemporalUNet3D` detector **plus** the transformer edge head. So
rebinding `--weights` replaced the **detector** as well, and leevvin's detector is not calibrated
for our inherited `BIOHUB_DET_THRESHOLD = 0.96875`: almost nothing clears it.

**This does not kill the checkpoint** — it says the swap needs its own operating point. A
`det_threshold` sweep against leevvin's weights is the correct follow-up, and it is a
detection-surface change (the class with 3.5× LB transfer). Deferred, not abandoned.

**Two prior kernel failures on the way here, both fixed and worth recording:** Kaggle's mount is
not reliably `/kaggle/input/<slug>/` (an rglob for the exact filename returned nothing), and
`REPO_DIR/weights/...` sits on a **read-only** filesystem (OSError 30 on copy). v3 searches
`/kaggle/input` for a `.pth` of exactly 8,355,927 bytes and rebinds `predict_cmd`'s `--weights`
rather than copying. Every failure was caught by an explicit guard rather than silently scoring
the old weights.

## 2026-08-22 — `p8_loosefilter` — the public 0.923 pattern, reproduced

Kernel `aryaarun07/biohub-p8-loosefilter` v1 (pushed). Source: `yunusgmsoy/kimi-notebook-v17`,
the public 0.923 frontier. Its config was diffed against ours — **31 of 41 settings identical**,
6 meaningful differences:

| setting | ours (0.915) | theirs (0.923) | role |
|---|---|---|---|
| `SAFE_DIV_MAX_UM` | 4.66 | **12.0** | loose propose |
| `SAFE_DIV_SISTER_MAX_UM` | 8.5 | **15.0** | loose propose |
| `SAFE_DIV_EXISTING_CHILD_MAX_UM` | 7.65 | **10.0** | loose propose |
| **`DEEPCENTER_SAFE_DIV_VETO`** | **0** | **1** | **the discriminator** |
| `BIDIRECTIONAL_EDGE_WEIGHT` | 0.20 | **0.30** | harmonic weight |
| DeepCenter ckpt / epoch | `checkpoint_last.pt` / 500 | **`best.pt` / 2** | veto model |

**This is the loose-propose/strict-filter design, and three independent analyses of ours
converged on it:**
1. Our division funnel measured true parent_dist median **7.42/8.87 µm against a 4.7 gate** — 135
   of 151 divisions unproposable — and concluded relaxation alone drowns at ~2,400:1 and **needs a
   discriminator**.
2. `public_code_teardown_2026-08-19` found the identical loose-propose/strict-filter shape in the
   0.917 public stack.
3. `error_atlas_2026-08-19` located the bottleneck in candidate generation.

**`BIOHUB_DEEPCENTER_SAFE_DIV_VETO` is the discriminator, it is already implemented in our
wrapper, and we had it switched OFF.** The whole change is configuration — no new code.

**Build trap caught before spending GPU:** cell 5 re-assigns
`BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT = "0.20"` *after* the env cell, so the env edit alone left it at
0.20. A second `replace` edit patches the late assignment; all six settings then verified
last-wins-correct in the built notebook.

**Prediction (recorded before the score, low-band-as-central per the standing correction):**
central **0.918–0.923**, upside 0.924–0.928 (~20%), flat 0.915 (~25%). The flat case would mean
the design does not transfer off its author's substrate.

**Score: 0.912 — WORSE than baseline by −0.003. Prediction MISSED.**

Recorded band was central 0.918–0.923, upside 0.924–0.928, flat 0.915 at ~25%. Actual **0.912**
was **below every band** — the first prediction that missed on the low side, and it missed for a
reason the host identified BEFORE the score landed but AFTER the submission: p8 ports kimi-v17's
loose gates (12/15/10) without its three precision filters
(`SAFE_DIV_REQUIRE_DIVERGENCE`, `SAFE_DIV_DIVERGE_UM`, `SAFE_DIV_REQUIRE_MUTUAL_NN`), which our
wrapper does not implement at all (diverge 0, mutual 0, orphan 0 occurrences).

**Loose propose without the strict filter is actively harmful — now measured, not inferred.**
The DeepCenter veto did fire (14,188 checked, 5,900 rejected, 41.6%), but it tests *"is there a
cell here?"* not *"are these two sisters?"*, and at ~2,400:1 most false candidates are real cells.
455 divisions emitted against 3 GT divisions on the placeholder substrate.

### This REVISES the transfer law — division changes CAN move the LB

| submission | change | LB Δ |
|---|---|---|
| divfix (703 divisions, no discriminator) | division term | **0.000** |
| **p8 (455 divisions, loose gates + veto)** | division term + bidirectional 0.20→0.30 | **−0.003** |
| control (28% node cut) | detection surface | −0.032 |

The earlier reading — "division-term changes score 0.000" — was drawn from a single sample.
**Division changes are not invisible to the LB; ours have been neutral-to-harmful.** That is a
materially different and more actionable statement.

**CONFOUND, host error:** p8 bundled the division gates AND `BIDIRECTIONAL_EDGE_WEIGHT` 0.20→0.30.
The −0.003 cannot be attributed between them. Deliberate bundling was justified for a faithful
port; it was not a faithful port, so the bundle bought a confounded result. **If the bidirectional
weight is retested, it must ship alone.**

Slots consumed: 15. Platform unchanged at **P3 harmonic 0.915** — p8 is NOT adopted.



## 2026-08-24 — `p9_coupled_division` v1 — PREDICTION RECORDED BEFORE THE SCORE

Kernel `aryaarun07/biohub-p9-coupled-division` v1, COMPLETE. Receipt-bound, verdict PASS.
Submission sha256 `7d9c24d620656c56d04d63160a64a3b7030fb62b164bf5b4c65c367fefb032a8` (12,491,452 B),
independently re-hashed from the file and matching the receipt. Built on **our** `p3_harmonic`
base (sha `70b636b6…`), so our bidirectional harmonic is retained.

**What it ships — the FULL coupled transplant, verified in the injected code**
(`scripts/kaggle_edits/coupled_division_transplant.py`), not p8's half-port:

| item | P9 | our 0.915 | source 0.926 |
|---|---|---|---|
| `SAFE_DIV_MAX_UM` | **8.0** | 4.66 | 8.0 |
| `SAFE_DIV_SISTER_MAX_UM` | **11.0** | 8.5 | 11.0 |
| `SAFE_DIV_EXISTING_CHILD_MAX_UM` | **10.0** | 7.65 | 10.0 |
| C1 mid-track parent (`:143`) | present | absent | present |
| C2 mutual-nearest-orphan (`:115`) | present | absent | present |
| C3 t+2 divergence 2.25 µm (`:205`) | present | absent | present |

**THE DECISIVE PRE-SUBMISSION CONTROL — we ran the real 0.926 notebook on the same substrate:**

| | nodes | edges | divisions |
|---|---|---|---|
| our P3 harmonic (0.915) | 122,214 | — | — |
| **P9 coupled** | **122,084** | **117,901** | **406** |
| **0.926 rocker (control)** | **120,748** | **116,536** | **384** |
| p8 loosefilter (scored **−0.003**) | — | — | 455 |
| p5 divfix (scored **0.000**) | — | — | 703 |

**P9's division count lands within 6% of the real 0.926 and clearly apart from both of our failed
division arms.** The constraint set is doing what it does in their pipeline. Node count is −130 vs
P3 (−0.1%), so the detection surface — the only class with measured 3.5× LB amplification — is
essentially untouched. That is the risk we most needed to avoid.

**PREDICTION (recorded before the score; low band promoted to central per the standing rule):**
- **central 0.919–0.925**
- upside 0.926–0.930 (~20%)
- flat/down 0.912–0.918 (~25%)

**I do NOT endorse the 0.926–0.928 expectation.** Two measured reasons to discount:
1. `div_proposal_funnel` fold 0: **condition (e), "second daughter is already parented", kills 19 of
   26 GT divisions and no gate setting touches it.** Our pipeline loses divisions upstream of every
   gate in this transplant.
2. Division-class changes have scored **0.000** and **−0.003** for us. This is the
   mechanism-complete version and the control is reassuring, but the class prior is poor.

Against that, the case for a real gain is the strongest this project has had on a division lever:
the transplant is LB-verified at 0.926 by **two independent teams**, it is the complete design rather
than p8's radii-without-filters, and our own GT says the radii are right (**GT median sister
separation at birth 10.57 µm vs our 8.5 µm gate**; only 19 of 151 divisions clear our gates at all).

**Falsifier:** a score of 0.915 ± 0.001 means the transplant does not transfer off its author's
substrate despite reproducing their division count — which would be a genuinely new fact, since it
would separate "emits the same divisions" from "scores the same".

Verification at submit time: `pytest -q` → **739 passed, 0 failures** (the 29 previously known
failures are gone); research tree 65 docs OK; claims table 53 OK.

### SCORE: **0.925** — submission `55753516`, 2026-08-24 21:13. PREDICTION HIT.

Recorded band was central **0.919–0.925**, upside 0.926–0.930 (~20%), flat/down 0.912–0.918 (~25%).
Actual **0.925** — top of the central band. **The first central estimate in seven to contain the
outcome**, and the first after the standing "promote the low band" correction was applied. The
correction worked: the naive modal estimate (0.926–0.928, held by the host) was still optimistic.

**+0.010 — the largest single gain of the campaign**, and the **first division-class change ever to
score positive** for us (prior: divfix 0.000, p8 −0.003).

Rank **207 / 2,693**, up from 449 / 2,659.

### THIS REVISES THE TRANSFER LAW AGAIN — and the revision is the useful part

| division submission | design | LB Δ |
|---|---|---|
| p5 divfix | ILP division economics, no discriminator | **0.000** |
| p8 loosefilter | loose radii 12/15/10, **filters omitted** | **−0.003** |
| **p9 coupled** | **radii 8/11/10 + C1 + C2 + C3, complete** | **+0.010** |

The prior reading — "division changes are neutral-to-harmful" — was drawn from two **incomplete**
ports. **Division changes transfer when the mechanism is complete.** The discriminating variable is
not the lever class; it is whether the precision constraints ship with the aperture. That is a
materially different and more actionable law, and it retires "post-processing is exhausted".

### THE NEW MEASUREMENT NOBODY ASKED FOR: our harmonic is now dead weight

| | base | + transplant | delta |
|---|---|---|---|
| theirs (0.913 + transplant) | 0.913 | **0.926** | +0.013 |
| ours (0.915 = 0.913 + harmonic) | 0.915 | **0.925** | **+0.010** |

Full additivity would put us at **0.928**. We measured **0.925**. So the bidirectional harmonic,
worth **+0.002** on the bare base, is worth about **−0.001** once the coupled division transplant is
present. **The two patches interfere and the harmonic is subsumed.**

Mechanism (consistent with `candidate generation` analysis): the harmonic is a mutual-support
*penalty* applied upstream of an unchanged 0.48 gate — it suppresses one-directionally-supported
edges. The transplant's C2 (mutual-nearest-orphan) is itself a mutual-support test. Both are asking
the same question; running them in series double-charges it.

**Falsifiable, one slot, no GPU: P9 with `BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT` disabled should score
≈0.926.** Note the guard at `p3_harmonic` cell 5 (`if not 0.0 < w <= 0.35: raise`) must be relaxed to
admit 0, so this is a code edit, not an env edit.

### POSITION AFTER THE GAIN — the honest read

- **202 teams at ≥0.926, 209 at ≥0.925.** The public 0.926 has been widely adopted; we are at the
  frontier, not ahead of it. Wrapping is exhausted *again*, one generation up.
- Top-3 boundary **0.953**; leader 0.962. **Gap +0.028.**
- Everything reachable by porting public code is now spent. The remaining +0.028 must come from
  something the field does not have — which returns the campaign to H1 and the zh001r asset.

### WHAT THE FUNNEL SAYS IS LEFT IN DIVISIONS

`div_proposal_funnel` fold 0, 26 GT divisions, **measured with the old gates but the killer is
gate-independent**: condition (e) — *the second daughter already has a parent* — loses **19 of 26**,
and no radius or constraint in this transplant touches it. The transplant fixed the gate half
(conditions g); condition (e) is untouched and is now the largest identified division loss.
The mechanism it implies is *contested reassignment*: allow the proposer to take a daughter away
from an existing link when division evidence is strong, rather than requiring an orphan.
**This is post-processing, needs no GPU, and is the natural successor experiment to P9.**

## 2026-08-25 — `p15_forkfree_probe` — PREDICTION RECORDED BEFORE THE SCORE

**This is a DIAGNOSTIC, not a lever. It is expected to score BELOW 0.925 and that is the point.**

Mechanism (Kaggle discussion/734192, Arul Prasad S P, VERIFIED): the official `summarise()` **drops
the division term entirely when a submission contains no divisions at all** —
`score = edge_jaccard if not has_divisions`. So a fork-free submission returns our **pure adjusted
edge Jaccard**, and the division term follows by subtraction.

Build: P9 (LB 0.925) with a single env flip, `BIOHUB_OUTPUT_SAFE_DIVISIONS = '0'`. Verified in the
built notebook: set in cell 2, read in cell 3 (`!= "0"` -> False), guard fires in cell 6
(`if not OUTPUT_SAFE_DIVISIONS ... return edges`). **Single assignment, no last-wins trap** — the
trap that nearly broke the p8 build was explicitly checked for. The bijection in
`motion_relink_edges` already forbids out-degree 2, so this flag is the sole remaining fork source.
Everything else — datasets, division transplant code, radii — is byte-identical to P9.

### The arithmetic

`0.925 = adj_edge + 0.1 * divJ`  =>  `divJ = (0.925 - probe) / 0.1`

Reference decomposition from the forum (mikelou1, VERIFIED): **0.928 = adj_edge 0.898 + 0.030**,
i.e. divJ 0.30. **So a 0.953 team carrying mikelou1's divisions needs adj_edge ~0.923.**

### PREDICTION (low band promoted to central, per the standing correction)

- **central 0.921–0.924**
- upside (probe >= 0.9245, i.e. divisions contributing almost nothing) ~25%
- downside (probe <= 0.920, i.e. divisions worth >= 0.005) ~20%

Basis: our LOEO divJ is ~0.015, which would put `0.1*divJ` at ~0.0015 and the probe at ~0.9235.
P9 emits 406 divisions; if most are false positives the division term is near-worthless and the probe
lands high.

### THE TWO BRANCHES — both are decisive, and they point OPPOSITE ways

**Branch A — probe ~0.923-0.924 (adj_edge already ~0.923).** Then our EDGE term is already at the
level a 0.953 team needs, and **our entire deficit is divisions.** This would REVERSE the strategic
read I have been building all session (that the frontier is an edge race) and would send everything
back to the division lane — where our own oracle says the bijection forfeits a +0.0232 ceiling.

**Branch B — probe ~0.915-0.920 (adj_edge ~0.915-0.920).** Then divisions are already contributing
0.005-0.010 and the remaining gap is genuinely in the edge term, confirming the forum arithmetic and
the mikelou1 datapoint. Detection-recall + reranking becomes the lane, as currently queued.

**Either way this single slot settles the question that four separate analyses have been arguing
about, at the cost of one submission and zero modelling work.**

### Falsifier
A probe score materially ABOVE 0.925 would mean the fork-free reading is wrong (divisions currently
cost us score rather than earning it), which would itself be a large finding and would immediately
retire the entire division lane including P9's +0.010.

## 2026-08-25 — DETECTION-THRESHOLD ARMS `p16_det090` / `p17_det094` — PREDICTIONS BEFORE THE SCORE

Both are single-variable arms off P9 (LB 0.925): `BIOHUB_DET_THRESHOLD` 0.96875 -> 0.90 and -> 0.94,
everything downstream byte-identical. This honours Mendrika's rule (discussion/734604): *"keep the
linker fixed while comparing detectors."*

**Build check passed the known trap.** Cell 2 of the base carries `BIOHUB_DET_THRESHOLD = "0.96875"`
and our edit appends a SECOND assignment after it; the read is in cell 3. Verified last-wins gives
0.90 / 0.94. This is the same last-wins shape that nearly broke the p8 build and that we found in the
public 0.926 and 0.927 notebooks.

### Why this arm, stated as measured facts

- **The metric charges over-prediction only `1 - 0.1 x over-prediction`** (Luka Duvanov,
  discussion/733877, VERIFIED). So +20% nodes costs ~2% if edge Jaccard does not improve at all.
- **`edge recall ~ node recall^2 x conditional linking accuracy`** (Mendrika, discussion/734604).
  Recall enters QUADRATICALLY while over-detection is taxed 0.1x linearly.
- **Cutting nodes 17% cost ~0.18 edge Jaccard** (Alan Thanickal, discussion/724917) — we are on a
  steep part of the curve, and the deployed threshold sits high on it.
- **Our own S1 sweep: detection F1 is monotone DECREASING in threshold**, optimum at or below p0.50
  (deployed-threshold recall 0.445 vs 0.563 at p0.5).
- **The near-duplicate trap does NOT apply.** Duvanov's ~9%-for-nothing penalty comes from unioning
  two models' detections without a merge pass. Lowering one detector's threshold cannot create a
  twin, because the local-max NMS already suppresses within 1.625 um — and we measured that NMS
  costs at most 0.12 pp of recall.

### PREDICTION (low band promoted to central, per the standing correction)

**p17 (0.94, the modest step):** central **0.923–0.927**, upside 0.928–0.932 (~25%),
downside 0.918–0.922 (~25%).

**p16 (0.90, the real step):** central **0.920–0.928**, upside 0.929–0.935 (~25%),
downside 0.910–0.919 (~30%). Wider band both ways — a larger move on the one lever class with a
**measured ~3.5x LB amplification**.

**The standing caution, restated honestly:** detection-surface changes amplify ~3.5x on the LB, and
our ONLY prior control in this class was a 28% node CUT that scored **-0.032**. **Adding** nodes is
the untested direction. The metric line bounds the downside at roughly `-0.1 x over-prediction` if
edge Jaccard fails to improve, which is why a big step is affordable at all.

**Read the PAIR, not either alone.** Monotone improvement 0.94 -> 0.90 says push the threshold
further and a third arm is warranted. Non-monotone (0.94 up, 0.90 down) locates an interior optimum
between them. Both down retires the whole recall-push thesis despite four independent lines of
evidence supporting it — which would itself be the most surprising result of the campaign.

### OPERATIONAL FACT DISCOVERED — record it
**Kaggle permits a maximum of 2 concurrent GPU BATCH sessions.** `p17_det094` was refused with
*"Maximum batch GPU session count of 2 reached"* while `p15_forkfree_probe` and `p16_det090` were
running. **Kernel pushes must be pipelined 2 at a time**, which is a real constraint on how many arms
can be in flight and was not previously in the ledger.
