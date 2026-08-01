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
| 2 | acquisition-state inference | — | — | — | — | — | RUNNING |
| 3 | event-centric division state | genuinely new (t−3…t+3 states, 3 heads) | **+0.064574** pooled, complete wrapper `[GT-oracle]` | **+0.002706** — but that is suppress-all, NOT the selector | selector transfers (AUC 0.881/0.916, leak 0.512); **base rate defeats it** | ~2.2 core-h, 0 GPU | **NO-GO** |
| 4 | CAP / track-as-point | real but insufficient | 35.45% of never-detected nodes @ w8 `[GT-oracle]` | **none — not measurable** | 85.14% / 32.23% | ~2 CPU-min, 0 GPU | **CLOSED — licence, Step 1** |
| 5 | dense self-supervised 3D motion | — | — | — | — | — | staged |
| 6 | counterfactual image critic | — | — | — | — | — | staged |
| 7 | pseudo-lineage substrate | — | — | — | — | — | staged |

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
