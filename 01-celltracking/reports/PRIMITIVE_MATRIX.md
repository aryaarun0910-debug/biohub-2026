# Primitive matrix — architectural cycle 2026-08-01

**Target:** 0.925 needs +0.011 from P0-B 0.914 · 0.950 needs +0.036.
**Objective:** exact pooled composite. **6bba carries 85.06% of edge mass — a 44b6-only figure is
never a headline.** Anchors every lane must reproduce before quoting a delta — **CANONICAL, from
`inventory/pooled_objective_parity.json`, pinned by `tests/test_pooled_objective.py`**:
pooled **0.6653932886896151** · 44b6 **0.7595490689190139** · 6bba **0.6489523829566957**.
**Trap 18:** a second anchor set (pooled 0.6654043, 6bba 0.6489652) circulates in the replay
scripts and differs by exactly one edge — they rebuild edges with a bare set comprehension and
lose insertion order, which the scorer's lowest-EDGE_ID tie-breaks are sensitive to. Magnitude
1.10e-05 pooled: harmless to every gate, fatal to the meaning of "parity passed". State which
anchor you used, and prefer a per-crop check against `artifacts/kaggle/coupled_cache/scores/A__*.json`.

Basis tags: `public` · `exact-pooled-OOF` · `cross-family-LOFO` · `in-family-CV` ·
`placeholder-proxy` · `GT-oracle`.

| lane | primitive | novelty | oracle ceiling | deployable Δ | family transfer | compute | decision |
|---|---|---|---|---|---|---|---|
| 1 | shape-aware localisation | genuinely new (first coordinate-only arm) | **+0.009125** @ ≤8.5 µm `[GT-oracle]` | **−0.008778** (λ=1); +0.000333 best shrinkage | **FAILS — sign-opposite** | ~3.4 core-h, 0 GPU | **NO-GO** |
| 2 | acquisition-state inference | HIGH | **+0.020212** pooled; bilateral core **+0.003389** `[GT-oracle]` | **not measured** (needs 1 GPU pass) | core is balanced: +0.003637 / +0.003349 | 3.5 core-h, 0 GPU | **PROMOTE** |
| 3 | event-centric division state | genuinely new (t−3…t+3 states, 3 heads) | **+0.064574** pooled, complete wrapper `[GT-oracle]` | **+0.002706** — but that is suppress-all, NOT the selector | selector transfers (AUC 0.881/0.916, leak 0.512); **base rate defeats it** | ~2.2 core-h, 0 GPU | **NO-GO** |
| 4 | CAP / track-as-point | real but insufficient | 35.45% of never-detected nodes @ w8 `[GT-oracle]` | **none — not measurable** | 85.14% / 32.23% | ~2 CPU-min, 0 GPU | **CLOSED — licence, Step 1** |
| 5 | dense self-supervised 3D motion | — | — | — | — | — | **DO NOT FUND as a motion vector** (Lane 2 evidence) |
| 6 | counterfactual image critic → **WS-D** | — | — | — | — | — | staged behind WS-C's proposer |
| 7 | pseudo-lineage substrate → **WS-E** | — | — | — | — | — | staged (background) |

---

## Lane 4 — CAP / track-as-point: CLOSED at Step 1 (licence)

**Four independent blocks, any one sufficient.**

1. **The CAP repository carries no licence at all.** GitHub API returns `license: null`, `/license`
   is HTTP 404, and none of its 104 blobs is a LICENSE/COPYING/NOTICE. Default copyright applies —
   all rights reserved. **This is stricter than the OrganoidTracker GPL-2 block: GPL-2 at least
   grants use; no licence grants nothing.**
2. **It is a CoTracker derivative and imports it at runtime** (`cap/models/core/cap/blocks.py:17`
   and both loss modules). CoTracker is **CC BY-NC 4.0**. A prize-bearing competition is not
   non-commercial, and the competition Winner Licence (MIT) cannot be granted over NC-derived code.
3. **The only obtainable weights are CC BY-NC 4.0** (`facebook/cotracker` on HuggingFace).
4. **It trains on CTC data**, already licence-blocked here, and the repo redistributes CTC
   evaluation binaries (~37 MB).

**Also: the advertised checkpoints do not exist.** The abstract claims "code and model checkpoints
are available"; there is no `.pth/.ckpt/.pt/.safetensors` in the tree, zero releases, and no
HuggingFace repo. Both the tracker weights and the required anchor-UNet weights are absent. The
released code also does not execute — `build_cap` is called with 4 args against a 1-arg signature,
`metrics.py` is imported but absent, and imports mix two package roots.

**The premise correction that matters more than the licence.** CAP is **not detection-free**.
`infer.py` seeds from `get_2D_anchor_points()`, which loads a separately-trained UNet segmenter and
takes mask centroids; mid-sequence "new cells" enter only through the division slots of
already-tracked cells. So CAP is **propagate-and-divide from detector-seeded points** — "no
detection" means "no *per-frame* detection", not "no detector". It therefore does not attack the
never-detected pool the way the lane premise assumed.

### The output worth keeping — a reusable prior for ANY propagate-and-divide proposer

`[GT-oracle, exact-pooled-OOF, 199 crops, E0c substrate, host matching rule: one-to-one bipartite,
7 µm, scale 1.625/0.40625/0.40625]` — artifact `inventory/propagator_reachability.json`.

| | pooled | 44b6 | 6bba |
|---|---:|---:|---:|
| GT nodes | 133,318 | 20,197 | 113,121 |
| never detected (no prediction within 7 µm) | **14,708 (11.03%)** | 895 (4.43%) | 13,813 (12.21%) |
| forward-reachable, unbounded | 45.46% | 88.60% | 42.66% |
| **forward-reachable, window 8** | **35.45%** | 85.14% | 32.23% |
| forward-reachable, window 16 | 40.37% | 88.60% | 37.24% |
| bidirectional, window 8 | 52.47% | 96.76% | 49.60% |
| bidirectional, window 16 | 58.87% | 97.65% | 56.36% |

**Any future lane proposing to attack the never-detected 43.80% must be priced against this table
before spending compute.** A seeded-propagation proposer has a 35.45% pooled ceiling at window 8 —
2.4× the 15% recall bar, so the idea is *not* dead on reachability — but the ceiling assumes
flawless propagation through frames where the detector saw nothing, says nothing about the 40.6%
precision arm, and is **2.6× better on the 15%-mass family** (85.14% vs 32.23%). The pooled number
is the 6bba number.

**Second finding, from an independent code path:** only **66 of 14,774** unmatched GT nodes had a
candidate within 7 µm and lost the bipartite assignment. The never-detected pool is a **genuine
detection miss, not arbitration** — which closes off "fix the matching" as a route to it, and
corroborates the never-detected/discarded split without reusing the attribution code.

**Third:** CAP is strictly 2D (10 × `Conv2d`, 0 × `Conv3d`, fixed 384×512). A tri-planar projection
over a 64-deep dense embryo volume would reintroduce cross-plane association — exactly the problem
the architecture is meant to bypass. Native 3D is a rewrite whose correlation neighbourhood grows
7²→7³ and which would have to train on our two families alone, i.e. the generic retraining CLAUDE.md
blocks, against the project's own finding that family-conditional shift is the binding constraint.

**One idea free to reimplement (ideas are not copyrightable):** the `P = 3` output head, where each
tracked point emits mother-continues plus two daughter branches with independent visibility flags —
making division a property of the *trajectory head* rather than a downstream fork classifier. Given
the division track's binding constraint is the selector, that framing is worth remembering. It needs
training data we do not have, so it is a note, not a queue item.

**Do not reopen** without a licence change by the authors *and* a CoTracker-free reimplementation.


---

## Lane 1 — shape-aware localisation: NO-GO (both proceed-gates passed, final gate failed)

**Novelty is real.** Every previously closed lane edited *topology* — candidate breadth
(−0.1596/−0.1496), node budget, component retention, fork reconstruction. This is the first arm that
moves **coordinates on a frozen node and edge set**, so `N_pred` and `N_est` are invariant by
construction and it is not a metric-artifact question. Node-count invariance verified 199/199.

### Census of all 14,766 unmatched GT nodes `[exact-pooled-OOF; bucket labels GT-oracle]`

| first cause | pooled | 44b6 | 6bba |
|---|---:|---:|---:|
| (a) no predicted evidence within 14 µm | 6,972 (47.2%) | 10 (1.1%) | 6,962 (50.3%) |
| **(b) displaced free detection, 7–14 µm** | **5,826 (39.5%)** | **737 (79.8%)** | 5,089 (36.8%) |
| (e) boundary/volume | 1,784 (12.1%) | 148 (16.0%) | 1,636 (11.8%) |
| (c) merged/ambiguous component | 131 (0.9%) | 0 | 131 (0.9%) |
| (d) assignment conflict | 53 (0.4%) | 28 (3.0%) | 25 (0.2%) |

**The families have structurally different failure modes: 44b6 misses are 79.8% displaced-but-detected;
6bba misses are 50.3% nothing-within-14 µm — genuine non-detection.** Since 6bba is 85.06% of edge
mass, the corpus problem is detection, not localisation. Addressable (b+c) = 6,683 edge FN = **24.12%**
of corpus edge FN, so the ≥10% proceed-gate passed bilaterally. The census independently reproduces
the project's 56.7 / 43.3 node-vs-association split.

### Oracle ceiling — the headroom is real

| arm | pooled Δ | 44b6 | 6bba | nodes moved |
|---|---:|---:|---:|---:|
| **oracle ≤8.5 µm** | **+0.009125** | +0.011077 | +0.008699 | 2,458 |
| oracle ≤14 µm | +0.019289 | +0.014986 | +0.020018 | 6,486 |
| oracle unbounded | +0.042079 | +0.015193 | +0.047011 | 14,570 |

Crop-block bootstrap (2,000): ≤8.5 µm CI **[+0.007698, +0.010782]**, P(Δ>0) = 1.000, bilaterally
positive. Only the ≤8.5 µm arm is a *localisation* ceiling — the unbounded arm is teleportation
(median move 18.7 µm) and is a fixed-topology node-recall bound only.

### The deployable primitive fails, and the reason is measured

Background-subtracted power-2 intensity centroid, image-only, `[exact scorer, family-balanced
60-crop stratified sample]`: λ=1 gives **−0.008778** (negative on both families); λ=0.5 and λ=0.25
are **sign-opposite** across families (44b6 −0.0054 / 6bba +0.0013). Final gate fails on every arm.

**Why: the shape signal this lane was built on is absent.** Every one of six estimators has a
*worse* median miss displacement than the status quo, and each fixes a minority tail while breaking
correctly-matched nuclei — corpus-scaled net ledger: intensity_centroid **−11,192**, soft-argmax
T=0.5 −3,027, cc_centroid −75,174, LoG −74,986. Local component shape is statistically identical
between misses and correct detections (√λ_major SMD 0.122, √λ_minor 0.030, LoG σ 0.003).

### Next integration dependency — a SELECTOR, with an exact bar

Best conservative estimator (soft-argmax T=0.5) has P(fix|miss) = 0.125 and P(break|matched) = 0.032,
so **break-even purity is 20.4% against a 5.1% base rate — a 4.0× enrichment merely to reach zero**;
+0.002 needs ~50% purity over ≥11,600 nodes. **This is the same shape as component retention**, where
the selector retained 11,683 components against an oracle's 763. Reopen only if a cross-fitted
multivariate selector clears **20.4% purity leave-family-out** on the 6,496-node addressable
population — ~20 CPU-minutes, and it must pass *before* another exact-scorer replay is spent.


---

## Lane 3 — event-centric division state: NO-GO on all three gates, with a SIGN CORRECTION

### The correction — act on this before anything else in the division track

`CLAIMS.md` and the D0′ table record **suppress-all alone = −0.001728**. That is an **EDGES-ONLY
artifact on a fixed node set**. Through the **COMPLETE wrapper**:

**suppress-all = +0.002706 pooled** · 95% CI **[+0.001726, +0.003703]** · P(Δ>0) = **1.000** ·
44b6 **+0.001532** / 6bba **+0.002872** — **bilaterally positive, GT-FREE, no selector required,
and `pi_vis`-invariant** (it creates zero division FP). **A +0.0044 swing. Trap 14 in reverse: the
negative was the artifact.**

It is edge *quality*, not a count effect: d_rawJ **+0.002798** / d_mult +0.000431 → **86.9% raw edge
quality, 13.5% multiplier** (contrast node budget's 116% multiplier). The deleted content carries
d_tp −174 / d_fp −887 ⇒ **q_net 0.164**, below the project's own 0.3994 retain floor — so deleting
is correct by the independently-derived rule.

**E0c ONLY. Do not inherit it.** E0c has 20,353 forks; P0-B has **8** (TP0/FP8/FN3). The magnitude
cannot port. **Re-measuring suppress-all through the complete wrapper on P0-B is the cheapest open
item in the whole programme** — GT-free, no selector, and it closes a question the ledger currently
records backwards.

### Why mother admission fails — arithmetic, not feature engineering

92 true pairs among **4,957,806** mothers ⇒ base rate **1.856e-05**. Inverting the ROC, 12% precision
demands mother AUC **0.983 at K=100, 0.991 at K=200, 0.9992 at K=600**, and at 12% the admission
budget is **capped at K=766 by the size of the positive set itself**. Measured AUC is 0.86–0.92.
Gates: ≥12% precision **FAIL by ~250×** · ≥+0.006 **FAIL** (best +0.004823) · ≥+0.003 LCB **FAIL**
(best LCB +0.002743).

**The `pi_vis` pin is what killed it.** Raw scorer says +0.004823; pinned at 1.0 it is +0.002842 —
i.e. the temporal selector adds **+0.000136 over doing nothing but suppress-all**, and **44% of the
raw headline was annotation coverage**. The pin caught exactly what it was written to catch.

**Capacity is not the missing ingredient.** Same features, same LOFO: HistGradientBoosting scores
mother AUC **0.59–0.73** against the L2 logistic's **0.86–0.93** in all six cells. With 16/76
positives, added capacity destroys transfer.

### Kept — two hand-offs

**D2 (daughter-pair selection) is ready and waits on any lane that produces a mother set.** Against
the frozen flow-midpoint rank-0 (72/92 pairs, +0.046330), the temporal head takes **75/92, +0.049871**
— paired gain **+0.003541**, 44b6 flat (no regression) / 6bba +0.004215, CI [−0.001733, +0.008785],
P(>0) = 0.929. Not significant; it rests on 4 divisions. `pi_vis`-invariant, family AUC 0.512.

**Four new bilateral family-blind features, the best measured on this problem:**
`cont_resid_um` (the *continue*-hypothesis residual) **0.859 / 0.803** — better than appearance-4's
LOEO 0.719; `sep_growth2` **0.852 / 0.779**; `sep_growth3` 0.848 / 0.753; `sep_mono` 0.746 / 0.704.
**`n_persist` is sign-unstable (0.755 on 6bba, 0.175 on 44b6) and must not carry a positive sign.**

### Closed — do not rebuild

- **The intuitive "broken division" structural signature is EMPTY**: mother terminates & both
  daughters orphaned → **155 candidates, 0 true**; `m_terminates=1` → **474,876 candidates, 0 true**.
  E0c always links the mother forward, so a real division never presents as a clean track-end —
  and **60 of 92 true pairs require STEALING a daughter** from another track.
- **Temporal NMS is free but inert**: cuts the population 5.5× (4,957,806 → 908,922) with **zero**
  loss of true forks and **zero** precision change at every K. Keep it as a structural constraint;
  do not expect precision from it.
- **Marginal family-blindness does not compose — third independent demonstration.** Every one of 55
  features is individually blind under N1 (leak 0.41–0.52), yet the 40-feature set reconstructs
  family at **0.689** and a 46-feature "hardened" subset at 0.673. Only **cardinality** fixes it:
  n ≤ 12 passes at 0.512, n = 16 at 0.540.

### Where the 151 divisions go — new decomposition

151 GT → **113 reachable** on E0c → **92** whose true pair is inside the frozen top-3 shortlist
(proposer loses 21) → **75** survive D2 selection (D2 loses 17) → **D1 must find those 75 among
4.96M mothers.** D1 is ~100% of the remaining loss. Lane 1's finding that 50.3% of 6bba's unmatched
GT nodes have nothing within 14 µm caps the 113 from below: **no division-side method recovers an
undetected mother.**


---

## Lane 2 — acquisition-state inference: **PROMOTE** (the cycle's first)

### The premise is TRUE — and it lives entirely in 6bba

**947 byte-identical consecutive volume pairs across 114 crops — ALL 6bba, ZERO in 44b6** (all 71
44b6 crops × 100 frames checked). My seed probe found none because it sampled `44b6_d754aa59`, a
44b6 crop: **the probe was structurally incapable of finding them.** A negative result from a
one-crop sample proved nothing about a phenomenon that is family-exclusive, and I should have
sampled both families before circulating it.

Settled in **110 s** corpus-wide via trap 19 (chunk == frame ⇒ byte identity), not 160 GB of decoding.

**Near-duplicates are a DIFFERENT phenomenon and the lane did not force the framing.** The `nrmsd`
distribution is continuous with no gap, low-tail pairs carry *ordinary* GT displacement (1.28 µm
median), and 96% of their FN excess is `det_never_detected` (enrichment 8.21×). `near_duplicate` is
a **low-contrast detection state**, not a frozen-acquisition state.

### Dropped intervals — dose-response, and the GT itself confirms it

| frozen run L | effective intervals | GT displacement ratio | FN rate |
|---|---|---:|---:|
| 0 | 1 | 1.00 | 0.209 |
| 1 | 2 | **1.80** | 0.338 |
| 2 | 3 | 2.14 | 0.552 |
| 3 | 4 | **3.58** | 0.707 |

Displacement tracks **L+1**. On frozen pairs GT displacement is **exactly 0 for 8,339 / 8,339 GT
edges** — *the annotation itself treats the interval as absent*. "The embryo was merely still"
predicts neither the zero nor the dose-response. `assoc_enum_beyond_cap` × `global_translation`
enrichment is **28.68×**: the enumeration cap is the state-sensitive stage.

### Gates

Oracle ceiling **PASS** (+0.020212 vs a +0.003 bar) · edge mass **PASS** (13.77% pair-level, 74.08%
crop-level) · cross-family **SPLIT**. The full policy is positive on both families (+0.003875 /
+0.022925) so it *cannot* invert — but **85% of the ceiling comes from states with zero incidence in
44b6**, where the policy is a literal no-op. **Transfer is vacuous, not demonstrated.** That is the
mirror image of the retracted 22/26 result, not a repeat of it.

The **genuinely bilateral core** — `global_translation + deformation + intensity_discontinuity` — is
**+0.003389 pooled, +0.003637 on 44b6, +0.003349 on 6bba**: nearly balanced, and consistent with
Lane 1's structural split (44b6 misses are displaced-but-detected → global-motion states; 6bba
misses are non-detection → the low-contrast state).

### Image registration LOSES — reject it as a motion estimator

Median residual against GT continuation displacement (128,581 edges): existing kNN16 flow **1.329 µm**
· zero-motion **1.817** · block deformable registration **2.167** · global phase correlation **2.801**.
**Both image estimators are worse than assuming no motion at all**, and on the very state phase
correlation defines its residual is **19.2 µm against zero-motion's 8.83**. It is a valid *flag* and
a useless *vector*. **This substantially undercuts Lane 5's premise (dense self-supervised 3D motion)
— do not fund it as a motion estimator on this evidence.**

Free and right-signed: scaling the existing kNN flow by the observed (L+1) multiplier lifts
post-frozen reach ≤6 µm from 0.8776 to 0.9005.

### The deployable mechanism — verified in our own source

`src/biotrack/wrapper.py::motion_relink_edges` **gates on RAW distance but scores on MOTION distance**:

```
332:  raw = float(np.linalg.norm(target_pos - source_pos))
333:  if raw > gate_um:
334:      continue                                   # <- true partner excluded HERE
335:  motion = float(np.linalg.norm(target_pos - predicted))
340:  cost[i, j] = motion + 0.05 * raw - MOTION_RELINK_LEARNED_BONUS * prob
```

On a doubled interval or an abrupt global shift the true partner is discarded at line 334 *before*
the motion-aware cost at line 340 is ever evaluated. **The repair is to gate on the
flow-compensated residual at the SAME radius** — which admits no extra candidates in aggregate and
therefore does **not** repeat branch A's global-widening failure (−0.1596 / −0.1496). It re-aims the
gate rather than widening it.

**Deployable delta NOT MEASURED.** The candidate cache holds only pairs inside the 10.0 µm relaxed
gate and `edge_prob` is nonzero out to 9.88 µm, so probabilities for newly admitted pairs cannot be
assumed zero. **Next dependency: one GPU inference pass**, gating the motion relink on the
flow-compensated residual, state-conditioned on (L+1).


---

# Cycle 2 — workstreams (launched 2026-08-01)

| WS | scope | resource | status |
|---|---|---|---|
| **A** | acquisition-aware union inference cache + 5-arm exact replay | GPU + CPU replay | **LANDED — arm B PROMOTES, arms C/D/E falsified** |
| **B** | suppress-all through the complete wrapper on 199 P0-strict OOF graphs | CPU | **LANDED — PROMOTE (conditional)** |
| **C** | branch-emergence proposer (denominator compression, NOT classification) | CPU | **LANDED — CLOSE. Division route closes.** |
| D | counterfactual image evidence | — | **NOT LAUNCHED — foreclosed by WS-C arithmetic** |
| E | pseudo-lineage consensus substrate | CPU background | staged |

D and E are staged rather than launched for two reasons: D operates *per proposed mother* and so has
a genuine data dependency on C, and three concurrent heavy-CPU lanes is the measured saturation point
of this box (established across two prior cycles). They launch as capacity frees.

## Formal closures carried into cycle 2 — do not rebuild

| item | status | reason |
|---|---|---|
| Lane 1 localisation / recentering estimators | **CLOSED** | oracle +0.009125, deployed −0.008778, sign-opposite; shape does not separate misses (SMD ≤ 0.122) |
| Lane 4 CAP | **CLOSED** | licence (no licence at all + CC BY-NC upstream + CTC training data), no weights exist, not detection-free |
| Lane 5 dense registration **as a motion vector** | **DO NOT FUND** | block deformable 2.167 µm and phase correlation 2.801 are both worse than assuming ZERO motion (1.817) vs kNN flow's 1.329. Keep phase/deformable outputs **only as state flags** |
| Lane 3 flat mother classification | **CLOSED** | arithmetically hopeless: needs AUC 0.983–0.9992 at base rate 1.856e-05, measured 0.86–0.92; capacity makes it worse |
| D2 daughter-pair selector | **KEPT ON SHELF** | +0.003541 paired, 44b6 flat, `pi_vis`-invariant — waits on a viable mother proposer |

## Submission portfolio — reserved, none spent until exact replay lands

1. P0-B + **bilateral acquisition core** · 2. P0-B + **full acquisition-state policy** ·
3. P0-B + **suppress-all**, only if P0-strict OOF promotes · 4. best acquisition + suppress-all
composition **after interaction replay** · 5. **hold** for branch-emergence or a result-informed correction.

**Never spend a slot on an alternative radius, flow multiplier or threshold — those replay locally
from the common cache.** A slot must represent a distinct mechanism and pass structural audit.

## Reporting rule for WS-A specifically

Report **four separate things** and never let one stand in for another: **state-detection accuracy ·
candidate-surface recall · assignment quality · exact final pooled score.** A rise in candidate
recall is **not** success. The decision is the exact pooled graph score.


---

## WS-B — suppress-all on P0-strict OOF: **+0.0016970 pooled**, and the E0c result did NOT port

`[exact-pooled-OOF, 199 crops, complete wrapper re-run]` · parity **0/199** counter mismatches,
`refilter_identity` True on 199/199, `PooledState` vs `summarise` error exactly **0.0**, and the
fold-0 arm reproduces `laneB_h0d_p0strict_f0.json` digit-for-digit **under a different edge
insertion order** — so trap 18 has no bite on this operation.

| | pooled | 44b6 (14.74%) | 6bba (85.26%) |
|---|---:|---:|---:|
| **Δ composite** | **+0.0016970** | **−0.0006187** | **+0.0020926** |
| Δ edge term | +0.0021013 | +0.0009686 | +0.0022549 |
| Δ division term | −0.0004043 | −0.0015873 | −0.0001623 |
| bootstrap 95% CI | **[+0.000716, +0.002545]**, P(Δ>0)=0.9995 | [−0.003819, +0.001681], P=0.339 | [+0.001120, +0.002967], P=1.000 |

Edge term carries **124%** of the delta and the division term is **negative (−24%)** — the exact
inverse of node budget's 116%-multiplier profile. Deleted content `q_net` **0.2126** against the
substrate's own retain floor **0.4187** ⇒ deleting is correct by the project's independently-derived
rule. 141 crops improve / 57 regress / 1 flat.

**Prerequisite resolved properly:** the fold-1 graph cache did not exist. It was built and the
builder was **validated by running it on the fold-0 CSV and asserting frame-equality against the
shipped fold-0 cache — 71/71 crops, 0 mismatches — before being applied to fold 1.** Schema tested,
not assumed.

### Why the sign inverts on 44b6 — the mechanism, and it is not the substrate being "worse"

Suppress-all is division-free **only where divJ is already 0**. On E0c, 44b6 divisions were
TP0/FP93/FN26 ⇒ divJ = 0, so suppression cost nothing and the arm was bilaterally positive. On
P0-strict, 44b6 divisions are **TP2**/FP100/FN24 ⇒ divJ 0.015873, and suppression destroys **2 real
TPs** for −0.0015873 against an edge gain of only +0.0009686. **44b6's edge side still improves.**
What makes it expensive is the family-restricted division denominator (126); pooled, those same 3
TPs sit over a denominator of 742 and the whole division cost is −0.00040431.

**This is the fifth substrate-transfer failure of the programme** — E0c +0.002706 *bilateral* became
+0.001697 pooled with a *negative* 15%-mass family, a ~37% magnitude loss with the mechanism fully
identified.

### Zero division-exemption components — extends Lane B's fold-0 finding to fold 1

**0 components and 0 nodes** were removed because a division exemption disappeared, measured per
crop on all 199. No component on P0-strict is kept only by `has_division`, so the D0′ hazard cannot
fire. All 9,233 deleted nodes come from fork suppression *splitting* long components (585
prune-isolated + 8,648 short-track across 2,248 components); every one was in a fork-bearing
pre-edit component, so attribution is 100%.

### P0-B deployment-integrity diagnostic `[placeholder-proxy — a check, NOT a gate]`

Base reproduces 0.889224766308105 exactly. Frozen operation: **0.8892248 → 0.8899725, Δ +0.0007477**.
Divisions **TP0/FP8/FN3 → TP0/FP0/FN3** — **divJ is 0 before *and* after**, so on P0-B the operation
**cannot lose division credit at all**, and the mechanism that penalises 44b6 on P0-strict is
structurally absent there. The four movies did not select anything.

### PREMISE CORRECTION — "P0-B has 8 forks" was a UNITS ERROR, and I propagated it

The "8" was P0-B's **metric division-FP count**, not its graph fork count. Measured:

| substrate | graph forks | nodes | forks/node |
|---|---:|---:|---:|
| P0-B | **305** | 120,861 | 2.52e-3 |
| P0-strict | 10,726 | 3,752,580 | 2.86e-3 |
| E0c | 20,353 | 5,118,041 | 3.98e-3 |

On the **same four crops**, P0-strict has 307 forks against P0-B's 305. **The substrates are
comparable in fork density**, so the "magnitude cannot port because P0-B has almost no forks"
argument I circulated was wrong — the right reason to re-measure was always divJ, not fork count.

### Integration decision — a rule conflict, resolved

- WS-B clause 1 (pooled positive, bootstrap LCB ≥ 0): **MET** (+0.0016970, LCB +0.000716).
- WS-B clause 2 (≥ +0.0015 with strong mechanistic parity): **MET at the point estimate**;
  P(Δ > +0.0015) = 0.658, so the margin above that bar is itself uncertain.
- CLAUDE.md standing gate (both families improve, min-fold ≥ +0.005): **NOT MET.**

**Resolution: the pooled objective governs, and min-fold is a robustness constraint — that was
established and locked earlier in this programme** (`scripts/verify_pooled_objective.py`,
`tests/test_pooled_objective.py`), after the bilateral gate was shown to reject every better pooled
arm including v122, our own best public score. The CLAIMS entry for the old gate stands superseded,
not violated.

**Verdict: PROMOTE as a submission candidate, with its size stated honestly.** It is a **+0.0017**
mechanism, not a +0.006 one; it does not reach 0.920 alone. The 44b6 CI straddles zero
([−0.003819, +0.001681]), so 44b6 harm is *not established* either — but it is not established
absent, and that is the residual risk a slot would buy information about.


---

## WS-C — branch-emergence proposer: **CLOSE. The division route closes.**

Two of five preregistered gates fail, and they fail by a margin **a GT-oracle in-sample fit over the
same representation cannot rescue.**

| gate | budget | measured | verdict |
|---|---|---|---|
| G1 ≤ 20,000 proposed mother-events | 20,000 | 19,900 | PASS |
| **G2 retain ≥ 50% of the 92** | 0.50 | **11/92 = 0.1196** | **FAIL (4.2×)** |
| G3 same-sign retention both families | — | 44b6 18.8% · 6bba 10.5% | PASS (sign only) |
| G4 no family/crop identifier | 0.65 | family AUC **0.5601** | PASS |
| **G5 base-rate gain ≥ 100×** | 100× | **29.79×** | **FAIL (3.4×)** |

### Why it fails — three independent reasons, none of them tuning

**1. The GT oracle is 19.5× over budget.** To retain 46 of 92, `K@50%` is 445,357 for the best
GT-free ranker and **390,677 for a GT-oracle in-sample logit that reads all 92 labels with no
held-out anything**. That oracle also **flipped three of the mechanism signs**, so it had already
searched the sign space — "a sign was backwards" cannot explain the gap.

**2. The arithmetic bar, and it forecloses the image route without spending GPU.** 50% retention
inside K = 20,000 (FPR 4.034e-03) requires binormal mother AUC **0.9695**; measured is
**0.8560 / 0.7767**. A **perfectly independent** new channel would itself need AUC **0.9386** to
close the gap (d′ = √(2.6492² − 1.5025²) = 2.182). The only unspent channel is appearance, measured
on this project at **0.657 / 0.496** — and appearance scorers here are **7–232× dependent**, so even
that generous independence assumption over-prices it.

**3. The premise is defeated by our own shortlist.** The frozen top-3 shortlist is built by
minimising `flow_midpoint_residual` — which *is* the branch hypothesis's residual. The alternative
hypothesis therefore already fits near-optimally for **all 4,957,806** mothers, leaving the
likelihood ratio no discriminative alternative term. Measured directly: `branch_beats_cont`
compresses 15.7× but retains only **8 of 92** — gain **1.37×**, *worse than the null term alone*.

Pure structural compression does not exist either: over 21 GT-free predicates and conjunctions the
**best gain is 4.40×**, and that one (`no_steal`) is forbidden because it destroys 60/92.

### Ledger corrections from this lane

- **Temporal NMS is NOT free.** Lane 3 recorded W=2 NMS as losing **zero** true forks. That is a
  property of *Lane 3's score*, not of the operator: under `BE_rank` it destroys **38 of 92** true
  mothers, under `BE_phys` **48 of 92**. **Any future lane applying NMS must re-measure its own
  true-fork loss.**
- **The annotated-subpopulation inflation hypothesis is FALSE.** Per-term mother AUC on the full
  4,957,806 is essentially identical to the annotated subpopulation (`cont_resid_um` 0.8440/0.8603
  on 44b6, 0.7904/0.8017 on 6bba). Division features are **not** overstated by annotation coverage —
  they are simply not strong enough. That removes a standing explanation.
- **`cos_daughter_axis` is wrong-signed and family-unstable.** It computes axis-vs-*migration*
  (`phaseb_h1g_features.py:137`), not daughter anti-parallelism: AUC **0.3234 / 0.5528** with the
  intuitive sign. The correct feature is `daughter_angle` (`:139`) at 0.6551 / 0.6024, same sign in
  both families.
- **Real divisions here are strongly ASYMMETRIC about the mother**: only **6 of 92** true pairs have
  `min/max arm > 1/1.5`, against 27% of all candidates. The symmetric-split prior is wrong on this
  surface — consistent with 60/92 requiring a daughter steal, so the shortlisted mother sits off the
  true geometric centre.

### Integration consequence — WS-D is NOT launched

WS-D (counterfactual image evidence) exists to supply exactly the independent channel reason 2
prices. **It needs AUC 0.9386 to close the gap; appearance measures 0.657 / 0.496 and is 7–232×
dependent on the scorers already counted.** Launching it would spend GPU to confirm an arithmetic
result. **Not launched.** Reopen the division route only if some method produces mother-level AUC
**≥ 0.97 on the full 4.96M denominator** — nothing in the measured inventory is within reach.

**D2 stays on the shelf, unspent** (75/92, +0.049871 pooled, paired +0.003541), waiting on a mother
set this route cannot produce.


---

## WS-A — **arm B clears every promotion gate. First mechanism in the programme to do so.**

### The result

**Arm B = change the gate QUANTITY in `motion_relink_edges` from raw source–target distance to the
kNN16 flow-compensated residual. Same 6/10 µm radius. Same cost function. Same node population.
It routes on NOTHING** — no acquisition state, no family, no crop identity.

`[exact-pooled-OOF, 199/199 crops, complete wrapper, prob = 0 on newly admitted pairs — a LOWER
diagnostic, so the deployable value is ≥ this]` · parity gap **EXACTLY 0.0** against all three
canonical anchors, 199/199 per-crop.

| | pooled | 44b6 | 6bba |
|---|---:|---:|---:|
| **Δ composite** | **+0.0087299** | **+0.0072796** | **+0.0090040** |
| bootstrap 95% | [+0.006272, +0.011223] | [+0.002938, +0.012035] | [+0.006285, +0.011818] |
| P(Δ>0) | **1.000** | 0.9995 | 1.000 |

| gate | verdict |
|---|---|
| exact pooled ≥ +0.002 | **PASS — 4.4×** |
| both families improve | **PASS** |
| min-fold ≥ +0.005 (CLAUDE.md) | **PASS at +0.0072796** |
| regime | 141/199 crops better, 58 worse (28.8% of edge mass), **none collapses**; top crop = 6.2% of the delta |

**Not a metric artifact: 101.6% raw edge quality, −1.6% count multiplier** — the exact mirror of the
node-budget failure. Edge TP **+904**, FP **−688**, FN **−904**: precision *and* recall improve.
Nodes move +0.041%.

**Re-aim, not widen — verified numerically.** At the identical radius every arm admits FEWER pairs
than the status quo (B **−45,782**, −0.293%). Branch A's global widening is not repeated.

**The GPU pass cost zero extra model FLOPs.** `predict_unet_transformer.py:447-457` already returns
the full `(n_src, n_tgt)` logit matrix per frame pair and softmaxes over all sources; the ~9.88 µm
ceiling on exported `edge_prob` is `threshold=0.5` + parent/child caps, **not a distance gate in the
model**. With the node set unchanged the cached probabilities are bit-identical to the deployed ones —
**max-abs-diff 0.0 across 4,167,217 pairs.**

### Three falsifications — the acquisition framing itself does NOT pay

- **Arm E (full acquisition policy) is FALSIFIED — even its UPPER bound is negative**
  (lower −0.0023157, upper **−0.0022931**, bracket width 2.26e-05). The loss is in the *candidate
  surface* (net −422 GT-true pairs); a pair off the surface cannot be selected at any probability.
  **Corollary worth reusing: for a fixed gate the probability lever is worth ≤ 2.26e-05 pooled, so
  `prob = zero` is an adequate proxy for the deployable score.**
- **The (L+1) scaling is falsified as a GATE quantity.** Post-frozen reach ≤6 µm: raw 0.7280,
  unscaled **0.8325**, (L+1)-scaled 0.7353 — and at 10 µm it falls *below* raw. The flow is rebuilt
  from the raw geff edges of the same frame pair, which already span the enlarged interval, so the
  multiplier **double-counts**. Lane 2 saw the opposite sign using a flow from the *post-wrapper*
  graph — an estimator that does not exist where the gate runs.
- **Phase correlation as a vector is catastrophic** (core reach ≤10 µm 0.9185 → **0.5691**),
  independently reconfirming "valid flag, useless vector".

**So arm B is NOT an acquisition-state result.** What Lane 2 actually surfaced is that
`motion_relink_edges` gates on the **wrong quantity**; the state conditioning is not what pays.
Arms C/C2 have net-true-pairs **exactly 0** on 44b6 — transfer there is vacuous, not demonstrated.

### THE CAVEAT THAT GOVERNS DEPLOYMENT

**Arm B is measured on E0c (public 0.889), NOT on P0-B.** This programme has already recorded two
mechanisms that looked good on E0c and died on P0-B — node budget went **+0.006339 → −0.0000103** —
and five substrate-transfer failures overall. **+0.0087299 is not a deployment claim until it is
re-measured on the deployment substrate.** That re-measurement is now the single highest-value
action available.


### WS-A arm B — ACTUAL value with real probabilities (supersedes the lower diagnostic)

`replay_B_cache`, 199/199 crops, parity 199/199, complete wrapper, E0c substrate.

| | pooled | 44b6 | 6bba |
|---|---:|---:|---:|
| **Δ (ACTUAL)** | **+0.0088059** | **+0.0074359** | **+0.0090668** |
| Δ (prob=zero lower diagnostic) | +0.0087299 | +0.0072796 | +0.0090040 |
| bootstrap 95% (actual) | [+0.006163, +0.011659] | — | — |
| P(Δ>0) / P(Δ>+0.005) | 1.000 / **0.9995** | — | — |

The actual exceeds the bound by **7.6e-05**, confirming WS-A's finding that for a fixed gate the
probability lever is negligible — **`prob = zero` is an adequate proxy, so future gate arms need no
GPU inference pass.** Artifact `inventory/wsa_armB_cache_actual.json`.

### Arm E2 — measured, NOT bilateral

+0.0015418 pooled but **44b6 −0.0006104** / 6bba +0.0019071 `[exact-pooled-OOF, 199/199]`. Confirms
the pattern: every acquisition-state-conditioned arm is either negative or fails min-fold. **Arm B,
which routes on nothing, is the only one that works.**
