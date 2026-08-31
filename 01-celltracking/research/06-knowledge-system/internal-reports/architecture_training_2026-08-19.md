# Architecture and training: what produces a calibrated abstention signal, and a linker that uses it — 2026-08-19

> **CORRECTION 2026-08-31 — READ BEFORE THE HOCT LICENCE CLAIMS BELOW.** This report states in three
> places (lines 90, 530, 1005) that HOCT is "code and weights MIT". **That is not accurate and must not
> be carried forward.** The MIT licence is asserted over the *repository's software*; **no document
> anywhere asserts any licence over the released checkpoints**, and the `general_v1.pt` artifact
> additionally embeds TorchScript-compiled `timm` under Apache-2.0, surfaced nowhere by the publisher.
> The use remains defensible and competition rule 2.5.a.3 explicitly covers an absent or incompatible
> grant on pretrained models. See `FACT-0452` for the current position and its attribution obligations.
> The original text below is left **unaltered on purpose**: this is a dated record of what was believed
> on 2026-08-19, and rewriting it would falsify the history rather than correct the claim. The registry
> is the source of truth.

Mandate: answer, with evidence, the question our own work converged on — **what architecture or
training change actually produces a calibrated abstention signal, and what linker change consumes
it?** Target: the 0.036 gap between our 0.915 and the leader's 0.951, on the deployed
`SCORE = adj_edge_jaccard + 0.1·division_jaccard`.

Builds on, and does not re-derive: `edge_loss_structure_2026-08-18.md` (the column softmax is
shift-invariant; `edge_prob` is a share; break-even deletion precision 59.0 % / 53.1 %),
`edge_training_frontier_2026-08-18.md` (Trackastra's loss and ablations; augmentation spec),
`retrain_recipes_2026-08-17.md` (F1/F2/F5/F6 defects), `h1_execution_spec_2026-08-18.md`
(GPU-hour anchors, patch inventory), `linker_division_capacity_2026-08-18.md` (three independent
division suppressors), `foundation_models_2026-08-17.md` (no drop-in FM helps).

Evidence labels on every claim:
**[MEASURED]** — run by me this session against real data, command and artifact named ·
**[CODE]** — read directly at the cited `file:line` ·
**[DOC]** — stated in a cited external source, with the benchmark named ·
**[INFERENCE]** — my arithmetic or reasoning, assumptions stated ·
**[UNVERIFIED]** — could not confirm, with the test that would settle it.

Licence tags are one-line factual statements. Nothing is excluded or deranked on licence.

---

## 0. HEADLINE — five findings, three of them new measurements

**0.1 The abstention signal already exists in our pipeline, and the linker deliberately destroys
it. [MEASURED]** The learned head, thresholded at 0.5 on a column softmax, declines to nominate any
parent for **8.73 % (fold 0) / 10.85 % (fold 1)** of all target nodes at `t > 0`. After
`biotrack.wrapper.motion_relink_edges`, only **3.61 % / 4.29 %** of targets remain parentless. The
deployed pipeline manufactures **15,666 (fold 0) / 15,482 (fold 1)** edges that the learned head
never nominated — 6.2 % / 9.2 % of the final edge set — and **88.5 %** of the fold-1 manufactured
edges land on targets the head declined *entirely*. The question "how do we build an abstention
signal and make the linker use it?" has a prior question: **the linker currently overrides the one
we have.**

**0.2 That override is net-positive, but only barely, and it is the single most FP-enriched pool in
the graph. [MEASURED]** A paired, node-identical A/B on the official scorer (10-crop 6bba pilot,
`node_recall = 0.7682` in every arm, so this is a pure edge lever):

| arm | edges deleted | `adj_edge_jaccard` | paired Δ | Δ per 1,000 edges deleted |
|---|---:|---:|---:|---:|
| **A — deployed** | 0 | **0.5788** | — | — |
| C25 — longest 25 % of the manufactured pool | 3,575 | 0.5754 | −0.0034 | −0.951e−3 |
| C50 — longest 50 % | 7,338 | 0.5712 | −0.0076 | −1.036e−3 |
| **B — the entire manufactured pool** | 15,482 | 0.5656 | **−0.0132** | −0.853e−3 |

Every arm loses, so the pool's false-positive share sits below the break-even bar of
`1/(1+J) = 63.3 %` at this substrate's `J`. But the pool is nonetheless **3.5–4× FP-enriched**
relative to the edge set as a whole ([INFERENCE], §2.4): ~50–56 % FP inside the pool versus 14.3 %
across all scored 6bba edges (`edge_loss_structure` §3: 14,355 FP / 100,550 scored edges).

> **This relocates the whole abstention problem. The rule does not have to find 59 % precision in a
> haystack that is 86 % true positives. It has to find 63 % precision inside a 15,482-edge pool
> whose base rate is already ~53 %. That is a 10-point lift, not a 45-point one — and it is
> reachable by a signal with far less discrimination than we have been assuming we need.**

**0.3 Raw displacement ranks in the WRONG DIRECTION inside that pool, which kills a heuristic the
record currently endorses. [MEASURED]** Deleting the **longest** 50 % of the pool costs
−1.036e−3 per 1,000 edges; deleting the remaining **shortest** 53 % costs only −0.688e−3 per 1,000
(arm B minus arm C50: 8,144 edges, Δ −0.0056). Less-negative means more FP-rich, so **the short
edges are the FP-rich half.** The mechanism is obvious in hindsight — a spurious detection is
linked to whatever nucleus is nearest, producing a *short* edge, while real nuclei genuinely move.
This directly falsifies the "downweight links whose displacement exceeds the frame's 95th
percentile" proxy proposed in `h1_execution_spec_2026-08-18.md` §5.5, at least inside the pool
where it would matter. **Geometry has no usable ranking signal here. Only a learned absolute score
can supply one.**

**0.4 The architecture cannot express "no parent" at three independent levels, and two of them are
one-line fixes. [CODE + MEASURED]**

| level | what is missing | cost to fix |
|---|---|---|
| **loss** | no background/dustbin term in `softmax(logits, dim=0)` (`train_unet_transformer.py:63`) | already written: `h1r_edge_loss_patch.py` E1 |
| **head** | `pair_mlp` ends in `nn.Linear(hidden_dim//2, 1)` — one scalar per pair, no matchability channel (`simple_node_transformer.py:96`) | **+65 parameters** [MEASURED] |
| **attention** | `SimpleNodeTransformer` runs **cross-attention only** — frame `t` attends to `t+1` and back, with **no self-attention inside either frame** (`simple_node_transformer.py:140-160`) | **+0.805 GFLOP at N=256**, ≈ 0.02 % of a training step [MEASURED] |
| **linker** | `motion_relink_edges` cost `motion + 0.05·raw − 0.75·prob` has **no rejection option**: any within-gate pair can be assigned, and unnominated pairs get `prob = 0.0`, indistinguishable from "nominated with probability 0" (`wrapper.py:283-291, 340`) | a dustbin row/column in `linear_sum_assignment`, ~10 lines |

**0.5 Joint detection+association is the wrong bet, and there is a direct head-to-head that says
so on cells. [DOC]** Cell-TRACTR (DETR + track queries for cells, PLOS Comput. Biol. 21(5):e1013071,
23 May 2025) trained **>1 week on an A100** and scores Cell-HOTA 89.87 on bacteria — **losing to
Trackastra's 91.16**, a two-stage linker trained on a single consumer GPU, *including on the
division sub-metric*. The MOT end-to-end lineage's own successors (MOTRv2 arXiv:2211.09791,
CO-MOT arXiv:2305.12724, LA-MOTR ICCV 2025) each diagnose joint training as the defect and
progressively undo it. Nothing in the MOTR family is affordable at 45 GPU-h/week (MOTR:
8×V100 × 2.5 days ≈ **480 V100-hours** for one run). **Do not spend a single Colab hour here.**

The frontier that *is* relevant is **HOCT** (Bragantini, Theodoro, Royer — the Ultrack authors —
arXiv:2607.11754, 13 Jul 2026, code+weights MIT at `royerlab/hoct`): an **edge-centric**  <!-- CORRECTED 2026-08-31: see banner; FACT-0452 -->
transformer, hand-crafted geometric features only, **no image encoder**, rank 1 on the CTC linking
benchmark overall and on Fluo-N3DH-CE, and it **reads and writes GEFF** — our own export format.
It is a two-stage linker, not a joint model. §4.3 costs a T4-sized port.

---

## 1. What our architecture can and cannot express — read off the code

### 1.1 The head is one scalar per pair, with no self-attention

`vendor/kaggle-cell-tracking/src/tracking_cellmot/models/simple_node_transformer.py` **[CODE]**:

```
:88-97   pair_mlp = Sequential(Linear(2H+3, H), GELU, Dropout, Linear(H, H//2), GELU,
                               Linear(H//2, 1))       # ONE output channel
:140-160 for block in self.blocks:
             q = block(q, kv=k, mask_t1)   # frame t attends to frame t+1
             k = block(k, kv=q, mask_t)    # frame t+1 attends to (updated) frame t
```

Measured, by instantiating the deployed configuration (`feat_dim=33, hidden=128, heads=4,
blocks=4`) **[MEASURED]**:

| quantity | value |
|---|---:|
| total edge-head parameters | **576,385** |
| — attention blocks | 529,920 |
| — `pair_mlp` | 41,601 |
| cost of a **second output channel** on `pair_mlp` | **+65 parameters** |
| cross-attention FLOPs at N = 256 tokens | 0.805 GF |
| `pair_mlp` FLOPs at N = 256 | 5.427 GF |
| **adding intra-frame self-attention** at N = 256 | **+0.805 GF (12.9 % of the edge head)** |
| the same, as a share of one training step (UNet ≈ 118 GF/volume × 32 volumes = 3,776 GF) | **≈ 0.02 %** |

**There is no self-attention inside either frame.** Cross-attention does let a child token `j`
aggregate over all candidate parents, so `j` *can* in principle form a "no good parent" feature.
What no part of the network can compute is **competition among children**: parent `i` never sees
the other targets bidding for it, and child `j` never sees the other children. SuperGlue
([arXiv:1911.11763](https://arxiv.org/abs/1911.11763), CVPR 2020) alternates self- and
cross-attention precisely for this, and ablates it: removing cross-attention drops indoor match
precision **84.4 % → 74.0 %**; removing the GNN entirely drops it to **66.0 %** **[DOC]**.

### 1.2 The 0.5 threshold on a column softmax abstains on AMBIGUITY, never on ABSENCE

`predict_unet_transformer.py:455-465` applies `softmax(raw, dim=0)` then keeps pairs with
`probs[i,j] > cfg.threshold` (default 0.5) **[CODE]**. Since each column sums to exactly 1, **at
most one row per column can exceed 0.5** — in-degree ≤ 1 is enforced structurally, not by the
`max_parents_per_node` cap. Confirmed on the real exports: pre-wrapper out-degree histograms are
`{1: 246,287}` (fold 0) and `{1: 164,673, 2: 2}` (fold 1) **[MEASURED]**, and every exported
`edge_prob` lies in `[0.5, 1.0]` (fold 0 p25/median 0.679/0.791; fold 1 0.715/0.873)
**[MEASURED]**.

The consequence is precise, and it is the diagnosis this whole report turns on:

> A column abstains **only when no single candidate parent holds more than half the mass** — i.e.
> when two or more parents are comparably plausible. It is **maximally confident** when exactly one
> nucleus is nearby, *even if the target is a spurious detection with no true parent at all*.
> The score expresses ambiguity; it cannot express absence.

And absence is our error budget: `edge_loss_structure` §3 measures that **98.5 % (44b6) / 99.4 %
(6bba) of scored false-positive edges have exactly one endpoint unmatched** — the isolated-but-
spurious case, which is the one regime where the column softmax is guaranteed to be confident.
That, not model capacity, is why the learned score has the best AUC (0.701) and the worst operating
point (49.6 %).

### 1.3 The linker has no rejection option

`src/biotrack/wrapper.py:309-345` **[CODE]**: `assign_pass` builds `cost[i,j] = motion + 0.05·raw −
0.75·prob` for every pair inside the gate (`tight = 6.0 µm`, `relaxed = 10.0 µm`) and runs
`scipy.optimize.linear_sum_assignment`. The only rejection is `cost[r,c] >= big`, which is the
distance gate — a *geometric* filter, not a confidence one. And `learned_prob` (`:283-291`) returns
**`0.0` for any pair absent from `learned_edge_probs`**, so "the head never scored this pair" and
"the head scored this pair at zero" are the same number. **The learned probability can only ever
add a bonus; it can never veto.**

This is why `edge_loss_structure` §0.2 measured that setting the bonus to zero costs one GT edge in
9,020. The bonus is not weak because the score is bad; it is weak because the cost function has no
channel through which a *low* score can act.

---

## 2. New measurements this session

All from in-repo evidence artifacts, on CPU. No GPU, no submission, no commit.

### 2.1 The head's abstentions, and their destruction

**[MEASURED]** — grouped per `dataset` (node ids are crop-local; pooling them across crops gives
wrong answers, which is worth recording as a trap).

| | fold 0 (`p3_d1_pilot_f0_v1`) | fold 1 (`p3_d1_pilot_f1_v1`) |
|---|---:|---:|
| crops in the pilot export | 9 | 10 |
| pre-wrapper targets at `t > 0` | 269,858 | 184,717 |
| — **receiving NO candidate parent** (head declines) | **23,571 (8.73 %)** | **20,040 (10.85 %)** |
| pre-wrapper out-degree histogram | `{1: 246,287}` | `{1: 164,673, 2: 2}` |
| post-wrapper targets at `t > 0` | 260,951 | 176,660 |
| — **still parentless** | **9,422 (3.61 %)** | **7,571 (4.29 %)** |
| post-wrapper out-degree histogram | `{1: 250,117, 2: 706}` | `{1: 167,975, 2: 557}` |

Artifacts: `_evidence/kaggle_runs/p3_d1_pilot_f{0,1}_v1/pregraphs_split{0,1}.parquet` and
`loeo_split{0,1}_strict.csv.gz`.

Two things worth flagging. First, **the wrapper converts roughly 60 % of the head's abstentions
into links** (fold 0: 23,571 → 9,422; fold 1: 20,040 → 7,571; node counts differ slightly pre/post
because the wrapper filters nodes, so this is approximate but the direction and magnitude are not
in doubt). Second, **all 706 / 557 post-wrapper divisions are created downstream of the learned
head**, by `add_safe_divisions_postlink` on pure geometry (`wrapper.py:831`, `"edge_prob": None` at
`:854`) — corroborating `linker_division_capacity` §0 from the opposite end.

### 2.2 The manufactured pool

Joining the final edge set against the head's nominated set on `(dataset, source_id, target_id)` —
node ids **are** preserved through the wrapper (28,759 of 28,958 post-wrapper fold-0 nodes carry an
id present pre-wrapper), while coordinates are not (the sub-voxel refine arm moved them), so the id
is the correct join key **[MEASURED]**:

| | fold 0 | fold 1 |
|---|---:|---:|
| final edges | 251,529 | 169,089 |
| present in the head's candidate set | 235,863 (93.8 %) | 153,607 (90.8 %) |
| **manufactured by the wrapper** | **15,666 (6.2 %)** | **15,482 (9.2 %)** |
| — onto targets the head declined entirely | — | **13,699 (88.5 %)** |
| — onto targets the head gave a *different* parent | — | 1,783 (11.5 %) |

Displacement, fold 1 **[MEASURED]**: manufactured edges have median **2.44 µm** (q10 1.15, q90
4.00); nominated edges median **1.82 µm** (q10 0.57, q90 3.45). The manufactured pool is
systematically longer — which is what one expects from a Hungarian mopping up leftovers — and §2.3
shows that this length does **not** make them wrong.

### 2.3 The paired A/B, on the official scorer

Method: rebuild the submission CSV with a subset of edges deleted, node rows untouched, and score
with `scripts/core/score_loeo_submission.py --gt-dir data/train` (official
`tracking_cellmot.metrics.summarise`). `node_recall = 0.7682` is **identical in all four arms**, so
the adjustment multiplier — which depends only on node counts — is constant and every Δ is exactly
paired on edges. Substrate: the 10-crop 6bba pilot **[MEASURED]**.

| arm | rule | deleted | `adj_edge_jaccard` | Δ | Δ / 1,000 deleted | division FP |
|---|---|---:|---:|---:|---:|---:|
| **A** | deployed | 0 | **0.578800** | — | — | 32 |
| C25 | longest 25 % of pool | 3,575 | 0.575400 | −0.0034 | −0.951e−3 | 22 |
| C50 | longest 50 % of pool | 7,338 | 0.571200 | −0.0076 | −1.036e−3 | 13 |
| **B** | whole pool | 15,482 | 0.565600 | **−0.0132** | −0.853e−3 | **0** |
| (B − C50) | shortest 53 % of pool | 8,144 | — | −0.0056 | **−0.688e−3** | — |

Three results, in descending order of importance.

1. **Honouring the head's abstention wholesale is a −0.0132 regression.** The wrapper's override is
   net-positive. My own pre-measurement cross-arm estimate had put the override's precision at
   ~26 %, which would have implied a large *gain* from deleting it; that estimate was wrong by
   roughly a factor of two, and the direct measurement is what settles it. **This is the report's
   own falsified hypothesis, recorded as such.**
2. **Displacement ranks backwards inside the pool.** Longest-half deletion costs −1.036e−3 per
   1,000; shortest-half deletion costs −0.688e−3 per 1,000. The short edges carry more FP mass, not
   less. §0.3 gives the mechanism. Any abstention rule built on "long jumps are errors" will
   underperform *random* deletion inside this pool.
3. **Every division false positive is manufactured.** Arm B takes division FP from 32 to 0 while
   division TP stays at 0. On this substrate that is worth nothing (`division_jaccard = 0` either
   way), but it localises the division-FP source exactly, and it is a free constraint for the
   division lane: the geometry proposer only ever fires on edges the learned head declined.

### 2.4 What the pool's FP share actually is

With `J = TP/D`, `D = GT + V − TP`, deleting `k` edges of which a fraction `p` are FP gives
`ΔJ = k·(p·J − (1−p))/D`, so **break-even requires `p > 1/(1+J)`** — 63.3 % at this substrate's
`J = 0.5788`, and 58.7 % at the deployed fold-1 `J = 0.7042`. (This is the same identity as
`edge_loss_structure` §3.2, and it is independently derived in the calibration literature review
from the published metric definition — the two agree.)

Inverting arm B's measured Δ **[INFERENCE, D not measured directly]**:
`D = 15,482·(1 − 1.5788·p)/0.0132`. For pooled `D` in the plausible range 1.5e5–3.0e5 this gives
`p` between **50 % and 56 %**. Against a whole-6bba FP share of **14.3 %** (14,355 / 100,550 scored
edges, `edge_loss_structure` §3), the pool is **3.5–4× FP-enriched**.

**Caveats, stated plainly.** `adj_edge_jaccard` is crop-weighted, not pooled, so the inversion is
approximate. The pilot substrate scores 0.5788 against the deployed fold-1 0.7042 — these ten crops
are harder than the fold average, so both the bar (63.3 % vs 58.7 %) and the pool composition may
differ on the deployed substrate. **The direction and the enrichment factor are the load-bearing
claims; the 50–56 % point estimate is not.** The test that would settle it is a scorer-exact TP/FP
labelling of the pool on the full fold, which is CPU-only and was out of this session's budget.

---

## 3. Calibrated edge scoring — what actually moves an operating point

### 3.1 The premise correction that matters most

**Temperature scaling is NOT rank-preserving on a column-softmax share.** This inverts the folklore
and it applies directly to us, because `edge_prob` *is* a column share.

- On a per-edge **sigmoid**, `σ(z/T)` is strictly increasing in `z`, so the ordering — and hence
  deletion precision at any fixed deletion *fraction* — is exactly invariant. Only the threshold
  *value* moves. Same for Platt scaling (`σ(a·s+b)`, `a>0`) and, weakly, isotonic regression.
  ([Guo et al., arXiv:1706.04599](https://arxiv.org/abs/1706.04599), ICML 2017) **[DOC]**
- On a **softmax share**, `p_r^(T) = p_r^(1/T) / Σ_l p_l^(1/T)`: the denominator depends on the
  whole column, so *"two samples with the same original confidence can yield different scaled
  confidences"* and the cross-column ranking changes. Stated and worked through explicitly in
  [Kubaty et al., arXiv:2508.21495](https://arxiv.org/abs/2508.21495) (Aug 2025), Appendix B
  **[DOC]**. Confirmed empirically at scale by [Galil et al., ICLR 2023,
  arXiv:2302.11874](https://arxiv.org/abs/2302.11874): temperature scaling *"consistently and
  greatly improves AUROC and selective performance"* and *"improves the partial order of all
  instances"* across 523 ImageNet classifiers **[DOC]**. ODIN
  ([arXiv:1706.02690](https://arxiv.org/abs/1706.02690), ICLR 2018) is the same mechanism: TS plus
  input perturbation takes FPR@95%TPR from **34.7 % → 4.3 %** on DenseNet/CIFAR-10 OOD — impossible
  for a rank-preserving map **[DOC]**.

> **Operational consequence: fitting a temperature on our existing column logits is a legitimate,
> zero-GPU lever that CAN change deletion precision at a fixed deletion fraction.** But note what
> it exploits — only the *shape of the competitor distribution within a column* (entropy, gap to
> runner-up, candidate count). It injects no absolute information, because a shift-invariant column
> softmax contains none. Expect a modest lift, not 49.6 % → 63 %.
>
> **Blocking precondition [MEASURED]:** our exports carry only the *selected* edge's probability
> (`pregraphs_*.parquet` has one `edge_prob` per emitted edge). Temperature scaling on a share
> requires the **full column of logits**. This is a one-line change to the export in
> `predict_unet_transformer.py`, and it must ship before the lever can even be tested.

A second warning from the same literature: optimising ECE is not the same as optimising the
decision, and can move it the wrong way. [Xi et al., arXiv:2402.04344](https://arxiv.org/abs/2402.04344)
measure temperature scaling *improving* ECE from 8.79 % → 3.62 % on CIFAR-100/ResNet-50 while
*enlarging* downstream conformal set size from 4.91 → 6.69 **[DOC]**.

### 3.2 Background / null / dustbin terms — they buy an absolute reference, not calibration

Four papers, all of which supervise the "no match" outcome explicitly, and **not one of which
reports a calibration measurement** **[DOC]**:

| system | mechanism | published operating point |
|---|---|---|
| **Trackastra** ([arXiv:2405.15700](https://arxiv.org/abs/2405.15700), ECCV/MICCAI 2024; BSD-3-Clause) | quiet softmax `Ã_ij = e^{Â_ij}/(1 + Σ e^{Â_i'j})` **plus** an auxiliary plain-sigmoid BCE at **λ = 1e−2** on the same logits | parental-softmax ablation, Bacteria: greedy AOGM 28.4 → **23.0** (FP edges 7.6 → 5.6); ILP 18.8 → **14.8**. ≈20 % AOGM reduction. No ECE, no reliability diagram; the 0.5 and α=0.05 thresholds are asserted |
| **DETR** ([arXiv:2005.12872](https://arxiv.org/abs/2005.12872), ECCV 2020) | trained `∅` class, log-prob down-weighted **10×** | COCO AP 42.0–44.9. **Cal-DETR** ([arXiv:2311.03570](https://arxiv.org/abs/2311.03570), NeurIPS 2023) measures Deformable-DETR at **D-ECE 12.8 % in-domain / 10.8 % out-domain**, UP-DETR at 25.5 %, and the best dedicated fix only reaches 8.4 % |
| **SuperGlue** ([arXiv:1911.11763](https://arxiv.org/abs/1911.11763), CVPR 2020) | Sinkhorn with a **single learnable scalar** dustbin `z` on an augmented row+column; unmatched terms supervised in the log-loss | indoor ScanNet match **precision 84.4 %** vs NN+mutual 50.4 %; outdoor 84.9 % vs 52.4 %. Threshold 0.2, hand-chosen. No calibration curve |
| **LightGlue** ([arXiv:2306.13643](https://arxiv.org/abs/2306.13643), ICCV 2023) | **drops the dustbin** for a per-point matchability `σ_i = Sigmoid(Linear(x_i))`, multiplied into the two softmaxes; balanced BCE on matchability | HPatches **precision 88.9 %, recall 94.3 %** vs SuperGlue 87.4 % / 94.9 % |

**The negative result is Cal-DETR's, and it is the one to internalise: a trained "no-object" class
produces badly miscalibrated scores (12.8–25.5 % D-ECE).** A dustbin buys an *absolute, thresholdable
reference*. It does not buy calibration. Anyone budgeting "add a null class and the probabilities
become meaningful" is budgeting something nobody has measured.

**LightGlue's stated reason for abandoning the dustbin is the most transferable single sentence in
this literature for us** **[DOC]**: SuperGlue's dustbin *"entangles the similarity score of all
points and thus yields suboptimal training dynamics"*; disentangling matchability from similarity
*"yields cleaner gradients."* That is an argument for our §5 E-A design — a **separate matchability
channel** — over a pure background term.

### 3.3 Focal loss — and a correction to the received view

[Lin et al., arXiv:1708.02002](https://arxiv.org/abs/1708.02002) (ICCV 2017): γ=2, α=0.25 optimal;
focal AP 36.0 vs best OHEM 32.8 vs α-balanced CE 31.1 **[DOC]**. Note the direction: at the
optimum, focal loss **down-weights positives to α=0.25** in a 1:1000 imbalance regime. The dense-
detection literature did not converge on extreme positive up-weighting.

The widely-repeated "focal loss is under-confident" claim is **not** what the canonical calibration
paper shows. [Mukhoti et al., arXiv:2002.09437](https://arxiv.org/abs/2002.09437) (NeurIPS 2020)
report optimal post-hoc temperatures of **T ≈ 1.0–1.1 for focal loss** (i.e. already near-calibrated)
against **T = 2.1–2.8 for cross-entropy** (strongly over-confident). The genuinely under-confident
rows are *label smoothing* (T = 0.7 on Tiny-ImageNet) and Brier (T = 0.9). Example ECE, CIFAR-100
ResNet-50: CE **17.52 %** pre-TS, focal-3 **5.13 %**, FLSD-53 **4.50 %** **[DOC]**. Correct summary:
**focal loss removes the need for temperature scaling; it does not overshoot into under-confidence.**

One free item worth stealing: focal loss's **prior bias initialisation**, `b = −log((1−π)/π)` with
π = 0.01, exists to keep a sigmoid head's absolute level sane under negative dominance **[DOC]**.
Our §5 E-A matchability head should ship with it.

**[UNVERIFIED]** No published focal-vs-BCE comparison of *operating point* (precision at fixed
recall / deletion precision) exists for association or matching. Every number above is
discrimination (AP) or calibration (ECE), not operating point.

### 3.4 Evidential / Dirichlet heads — do not build one

Two rigorous critiques, both NeurIPS, both structural rather than empirical:

- **Bengs, Hüllermeier & Waegeman** ([arXiv:2203.06102](https://arxiv.org/abs/2203.06102), NeurIPS
  2022). **Theorem 1**: for level-2 losses formed by averaging any convex level-1 loss, the
  empirical minimiser is **always a Dirac measure, regardless of sample size N** — the learner
  collapses to a point prediction and represents no epistemic uncertainty at all. **Theorems 2–3**:
  for KL-regularised variants, λ too large breaks consistency, λ too small restores the collapse,
  and **no constant λ satisfies both.** The reported uncertainty reflects λ-tuning, not evidence
  **[DOC]**.
- **Shen et al.** ([arXiv:2402.06160](https://arxiv.org/abs/2402.06160), NeurIPS 2024): EDL
  uncertainties are *"non-vanishing even with infinite data"*; EDL is better understood as an
  energy-based OOD detector; downstream wins occur *"despite their poor uncertainty quantification
  capabilities"* **[DOC]**.

**[UNVERIFIED]** No head-to-head showing evidential heads beating temperature scaling on a
selective-prediction/AURC benchmark was found. **Verdict: high complexity, unsupported benefit,
wrong regime.** We have an in-distribution binary decision with abundant labels, not an epistemic
OOD problem. Kill.

### 3.5 Conformal / selective prediction — a guard, not a gain

Your target — "≥59 % of what I delete must be FP" — is a **precision-of-a-selected-set** guarantee,
i.e. false-discovery-rate control. That narrows the field:

- **Split conformal** ([Angelopoulos & Bates, arXiv:2107.07511](https://arxiv.org/abs/2107.07511))
  guarantees marginal *coverage*, not selection precision. Wrong tool.
- **RCPS** ([arXiv:2101.02703](https://arxiv.org/abs/2101.02703), JACM 2021) and **Conformal Risk
  Control** ([arXiv:2208.02814](https://arxiv.org/abs/2208.02814), ICLR 2024) require a **monotone**
  risk; FDR is not monotone in the threshold in general. Awkward fit.
- **The three that do fit** **[DOC]**: **Learn then Test**
  ([arXiv:2110.01052](https://arxiv.org/abs/2110.01052)) reframes risk control as multiple
  hypothesis testing, lifting monotonicity and explicitly covering FDR; **Conformal selection**
  ([Jin & Candès, arXiv:2210.01408](https://arxiv.org/abs/2210.01408)) — literally "select a subset
  while controlling the proportion falsely selected"; **FSR control for selective classification**
  ([Zhao & Su, arXiv:2311.03811](https://arxiv.org/abs/2311.03811)), whose Theorem 2 gives
  finite-sample control of `1 − precision` on the non-abstained set under exchangeability, with an
  explicit class-prior-shift variant bounded by `C·α`.
- **Calibration-set size is not our constraint.** Zhao & Su use `n_cal = 1500`; we have 10⁵–10⁶
  labelled candidate edges per held-out embryo **[INFERENCE]**. **Exchangeability is the
  constraint**, and a held-out embryo violates it.

**Training-time abstention beats post-hoc thresholding, with numbers.** SelectiveNet
([arXiv:1901.09192](https://arxiv.org/abs/1901.09192), ICML 2019) and Deep Gamblers
([arXiv:1907.00208](https://arxiv.org/abs/1907.00208), NeurIPS 2019), CIFAR-10 error on the covered
set **[DOC]**:

| coverage | softmax-response | SelectiveNet | Deep Gamblers |
|---:|---:|---:|---:|
| 1.00 | 6.79 | 6.79 | 6.12 |
| 0.90 | 2.89 | 2.43 | **2.19** |
| 0.85 | 1.78 | 1.43 | **1.09** |
| 0.80 | 1.05 | 0.86 | **0.66** |

At 0.85 coverage that is a **39 % relative error reduction from training-time abstention alone** —
the largest documented post-hoc-vs-trained gap in this whole review. Deep Gamblers' mechanism is
literally an `(m+1)`-th "abstain" class with a payoff hyperparameter `o`; **it is the same
architectural move as a dustbin, trained with a coverage objective.**

**The blunt assessment: none of §3.5 improves discrimination.** If the score's tail precision tops
out at 49.6 %, conformal selection at the required α will correctly return the *empty* set. That is
worth having — it stops us shipping a regression — but it is a guard, not a lever.

### 3.6 Calibration under the shift we actually face

[Ovadia et al., arXiv:1906.02530](https://arxiv.org/abs/1906.02530) (NeurIPS 2019) **[DOC]**:
temperature scaling's CIFAR-10 ECE rises from ≈**0.049** i.i.d. to ≈**0.180** at high corruption;
explicit conclusion, *"calibration on the i.i.d. validation dataset does not guarantee calibration
under distributional shift"*; ranking under shift **Ensembles > Dropout > Temperature Scaling >
Vanilla**, with most ensemble gains realised at **M = 5**.

[Jaeger et al., ICLR 2023 Oral, arXiv:2211.15259](https://arxiv.org/abs/2211.15259) is the
uncomfortable one: across a large failure-detection benchmark spanning all relevant methods and
shift types, **a simple softmax-response baseline was the overall best performer**, which they say
*"underscores drastic shortcomings of current evaluation"* **[DOC]**.

Three concrete consequences for our LOEO design **[INFERENCE, grounded above]**:

1. **Threshold on a fixed deletion FRACTION, not a fixed VALUE.** Under shift the score
   *distribution* moves far more than the score *ordering*. A quantile rule transports; an absolute
   cutoff does not. (Bonus: this also makes any monotone recalibration a no-op on a per-edge
   sigmoid, which simplifies the decision surface — see §3.1.)
2. **Calibrate leave-one-embryo-out across both directions and take the worst case**, not the
   pooled average. We are already required to report both directions; use the spread as a *shift
   estimate*, not just a reporting obligation.
3. If a guarantee is wanted under shift, the tool is **non-exchangeable / weighted conformal**
   ([Barber et al., Ann. Statist. 51(2), 2023, arXiv:2202.13415](https://arxiv.org/abs/2202.13415)),
   which yields a guarantee *minus a computable departure-from-exchangeability penalty*.

### 3.7 Calibrated association scores in cell tracking specifically — a thin but on-point literature

- **"How To Make Your Cell Tracker Say 'I dunno!'"** ([arXiv:2503.09244](https://arxiv.org/abs/2503.09244),
  12 Mar 2025). The only cell-tracking work treating calibration as the object of study. Uses the
  **same column-softmax link confidence we have**, finds *"vanilla cell tracking algorithms tend to
  be overconfident"*, and reports that *"temperature scaling almost always improves ECE"*,
  especially at low temporal resolution **[DOC]**. Its mechanistic result is the one to steal: **the
  fitted temperature rises as temporal resolution falls, and correlates with observed mean squared
  displacement** — i.e. *T* is effectively a function of motion scale. **[INFERENCE]** If that holds
  here, a single global temperature is mis-specified for us, and a **conditional temperature
  `T(local density, displacement scale, candidate count)`** is the right form — and, being
  covariate-dependent, it can genuinely re-rank. Results are figures-only; numeric ECE **[UNVERIFIED]**.
- **"Cell tracking with accurate error prediction"** (Nature Methods 2025,
  [s41592-025-02845-6](https://www.nature.com/articles/s41592-025-02845-6); bioRxiv 2024.10.11.617799).
  **[UNVERIFIED — paywalled, snippet only.]** Reported approach: **microstates and partition
  functions** to compute *context-aware* error probabilities, on the intuition that a low-likelihood
  step can be high-confidence if alternatives are excluded by confident surrounding tracks; used to
  retain only high-confidence segments. **[INFERENCE]** This says edge confidence is a property of
  the *global assignment*, not of one column — which is exactly what a per-column softmax discards.
  **We already run an ILP.** The objective gap from forcing an edge out is a ready-made,
  absolutely-scaled confidence that costs no retraining. That is the cheapest untested idea in this
  entire report and it belongs in Experiment 1.
- **Biological Needs MHT** ([arXiv:2403.15011](https://arxiv.org/abs/2403.15011)) derives aleatoric
  uncertainty from test-time augmentation (spatial shifts by one cell radius) into Gaussian
  association costs; up to **5.75× Complete Tracks** and ~**6× Cell Cycle Accuracy** on
  BF-C2DL-HSC, SOTA biological metrics on 5 of 9 CTC datasets **[DOC]**.
- MOT-side (UncertaintyTrack [arXiv:2402.12303](https://arxiv.org/abs/2402.12303): **−19 % ID
  switches, +2–3 mMOTA** on BDD; MOT-CUP; UTrack) all propagate *localisation* uncertainty. **None
  produces a calibrated association score.** That gap is real.

---

## 4. Joint detection + association — the verdict is no, with one exception that is not joint

### 4.1 The direct head-to-head on cells

**Cell-TRACTR** (Ozawa et al., *PLOS Comput. Biol.* 21(5):e1013071, 23 May 2025; weights on Zenodo
14509424) is a DETR-with-track-queries model for cells, explicitly inspired by TrackFormer and
MOTR, with no post-processing, and it introduces Cell-HOTA **[DOC]**:

- Training cost: bacteria **V100, 12 epochs**; DeepCell mammalian **A100, 24 epochs**; authors state
  *"training times exceeded one week for large 2D images"* and that the cost prohibits
  hyperparameter optimisation.
- Bacteria test (n = 29 movies), Cell-HOTA: Cell-TRACTR **89.87**, EmbedTrack 89.64, DeLTA 88.56,
  **Trackastra 91.16**. On DeepCell, Caliban has the highest DetA *and* DivA.
- No 3D capability demonstrated.

> **A joint DETR model trained for over a week on an A100 loses to a two-stage transformer linker
> trained on a single consumer GPU — and loses on the division sub-metric, which is 10 % of our
> objective.** This is the closest thing to a controlled joint-vs-two-stage experiment in our
> domain, and it points away from joint.

### 4.2 The MOT lineage, costed

| model | documented training cost | dense-3D-bio evidence | absolute per-edge score? |
|---|---|---|---|
| MOTR ([2105.03247](https://arxiv.org/abs/2105.03247)) | 8×V100, 2.5 days, 200 epochs ≈ **480 V100-h** | none | no (set prediction) |
| MOTRv2 ([2211.09791](https://arxiv.org/abs/2211.09791)) | 8 GPUs, batch 1/GPU | none | no |
| MOTRv3 ([2305.14298](https://arxiv.org/abs/2305.14298)) | 80 epochs, 4 datasets | none | no |
| MeMOTR ([2307.15700](https://arxiv.org/abs/2307.15700)) | **8 GPUs ≥ 32 GB** (README) | none | no |
| CO-MOT ([2305.12724](https://arxiv.org/abs/2305.12724)) | 38 % of MOTRv2's compute | none | no |
| MOTIP ([2403.16848](https://arxiv.org/abs/2403.16848), CVPR 2025) | 8×RTX4090, 1–1.5 days ≈ **190–290 4090-h** | none | **no — softmax over an ID vocabulary**, the same share structure we are trying to escape |

**Not one is affordable at 45 GPU-h/week, and not one produces the score we need.** More
importantly, the family's own trajectory is an argument against it: MOTRv2 re-introduces a frozen
YOLOX detector because of *"the conflict between the detection and association tasks"*; CO-MOT
documents that detect queries receive almost no positive training samples because *"the majority of
the newborns come on stage at the beginning of videos"*; LA-MOTR (ICCV 2025) concedes that *"shared
decoders for simultaneous object detection and tracklet association … result in task interference"*
and separates them **[DOC]**.

**CO-MOT's newborn-suppression diagnosis maps directly onto our worst sub-metric**: newly divided
daughter nuclei and cells entering the crop are exactly "newborn objects", and joint training is
documented to suppress precisely those. **[INFERENCE]** Given `division_jaccard` is 10 % of our
score and we currently emit zero learned divisions, adopting an architecture with a documented
newborn-suppression pathology would be actively counterproductive.

### 4.3 The frontier that IS relevant: HOCT

**HOCT** — Bragantini, Theodoro & Royer, [arXiv:2607.11754](https://arxiv.org/abs/2607.11754),
13 Jul 2026; code + weights `github.com/royerlab/hoct`. **Licence: MIT (code/weights); paper text  <!-- CORRECTED 2026-08-31: see banner; FACT-0452 -->
CC BY-NC-ND 4.0.** Same lab as Ultrack, i.e. the same institutional lineage as this competition's
data and scorer.

Why it is the single most relevant external artifact found **[DOC]**:

- **Edge-centric**: candidate links are the tokens and attend to *each other*, after a node
  self-attention stage (L_n = 4, 3D RoPE with learnable per-head frequencies, spatial mask at
  τ = 300 px) and an edge attention stage (L_e = 4) with a **line-to-line distance attention bias**
  — a learnable per-head bias from the minimum Euclidean distance between two edges' 3D line
  segments, with per-head `σ_h ∈ {+1,−1}` giving attractive and repulsive heads.
- **Inputs: detections only, d = 19 hand-crafted features** (position, equivalent diameter,
  intensity statistics, 9-element inertia tensor, distance to FOV border). **No image encoder.**
- **Output**: parental softmax with an explicit `1 +` no-parent term, grouped by target node **and
  temporal gap**; focal loss γ = 3.5; **division weight 3.5×**; ILP with flow conservation.
- **Its central empirical claim is a direct critique of GNNs on graphs like ours**: candidate
  tracking graphs have **adjusted homophily ℋ_adj ≈ 0.01 ± 0.04**, so topology carries almost no
  signal for message passing. Its edge-stage ablation is the only published head-to-head of edge
  attention vs GNN message passing on cell-tracking graphs: **HOCT 0.926 ± 0.007 > Edge-Transformer
  (no geometry) 0.922 > GAT 0.916 > FAGCN 0.909** (CLB).
- **Results, official CTC hidden test set, single model**: rank **1st in CLB (0.920), LNK (0.981)
  and BIO (0.858)** by overall generalizability; **Fluo-N3DH-CE 0.950 / 0.986 / 0.913 (1st)**;
  Fluo-N3DH-SIM+ 0.997 / 1.000 / 0.988 (1st). Bacteria division benchmark **AOGM 6.36 ± 1.35, zero
  deletion errors, division F1 0.9962**, beating Trackastra's best variant with **hand-crafted
  features only**.
- **Training cost: 1× H200, ~18 h per CV split, 50k steps**, peak ~**60 GB**, crops
  256×512×512, Muon + Adam, EMA 0.98. Augmentation includes **feature dropout p = 0.2**.
- **Human-in-the-loop (Table 5)**: frozen backbone + a **logistic-regression head** with uncertainty
  sampling, 400 annotations over 20 rounds — AOGM **1418 → 587 (−58.6 %)** at ~2 min/round, versus
  Trackastra + LoRA at **−6.75 %** and 40–55 min/round.

**Feasibility verdict for us [INFERENCE]:** 18 h fits our weekly budget *in hours*, but **~60 GB
peak and 256×512×512 crops do not fit a 16 GB T4/V100**. The model itself is small (8 attention
layers, no image backbone), so a reduced port at our 64³ crop scale with small k-NN candidate degree
is plausible — the authors note k = 10 costs ~100× the memory of k = 1. It also adds a Muon
optimizer dependency. **This is a two-to-three-week project, not a 45-hour one**, and it should not
be started before the cheap levers in §6 are settled. **But its frozen-backbone + logistic-head
correction loop is directly stealable at zero training cost, and is the template for §6
Experiment 1.**

### 4.4 What the affordable two-stage linkers give us

| method | per-edge score form | training cost | dense-3D-bio evidence |
|---|---|---|---|
| **SUSHI** ([2212.03038](https://arxiv.org/abs/2212.03038), CVPR 2023) | **binary sigmoid per edge**, focal γ=1, then LP rounding | Adam 3e-4, wd 1e-4, batch 8 clips, 250 epochs; **learned module ≈ 22K params**; wall-clock **[UNVERIFIED]** | none (MOT only) |
| **MPNTrack** ([1912.07515](https://arxiv.org/abs/1912.07515), CVPR 2020; MIT) | **binary edge classification** | **6 epochs** MOT17 / 22 MOT20 | none |
| **Cell-Tracker-GNN** ([2202.04731](https://arxiv.org/abs/2202.04731), ECCV 2022) | **binary edge CE with adaptive weights** | 1× V100 DGXS 32 GB, 10-frame subsequences | **Fluo-N3DH-SIM+ TRA 0.974, rank 1/11** (3D, simulated) |
| **Trackastra** (BSD-3-Clause) | quiet softmax + auxiliary sigmoid at λ=1e−2 | single GPU; epochs/hours **[UNVERIFIED]** | ISBI 2024 CTC linking winner incl. 3D; paper tables 2D-only |
| **CAMELTrack** ([2505.01257](https://arxiv.org/abs/2505.01257)) | **InfoNCE embedding distance + Hungarian — no per-edge probability** | **< 1 GPU-hour**, 10 epochs | none (MOT/pose/BEE24) |

**The three that give a true binary sigmoid per edge — SUSHI, MPNTrack, Cell-Tracker-GNN — are
exactly the ones with the score structure we need.** None has been measured on data as dense as
ours, and none of them is a *replacement* for our stack; the transferable item is the **formulation**
(binary per-edge classification with focal γ=1, trained on real detector outputs), not the model.

**Also on the record: nobody has shown a learned linker dominating classical optimisation on the
densest 3D nuclei data.** The CTC ceiling on Fluo-N3DL-DRO is ≈0.71–0.79, held by **Ultrack** — an
ILP with **no learned linker at all** (Nature Methods, 25 Aug 2025; DRO 0.708, TRIC 0.841, CE 0.844,
all rank 1) **[DOC]**. Treat "a transformer will solve dense 3D linking" as unsupported.

---

## 5. The bundle of small fixes, with documented effect sizes

Every item below is a config or few-line change that rides the *same* retrain. Ordered by
documented evidence strength per GPU-hour.

### 5.1 AMP — the one that buys hours back rather than spending them

`F.binary_cross_entropy` under autocast is **a documented hard error, not folklore**. PyTorch's
`torch.amp` docs, verbatim: *"binary_cross_entropy and BCELoss raise an error in autocast-enabled
regions"*, because their backward *"can produce gradients that aren't representable in float16"*;
the prescribed fix is `binary_cross_entropy_with_logits` **[DOC]**. The op lists confirm the split
we want: `conv3d`, `linear`, `matmul`, `bmm` autocast to fp16; `binary_cross_entropy_with_logits`,
`softmax`, `layer_norm`, `sum`, `exp` stay fp32 automatically.

Measured 3D-U-Net speedups, NVIDIA NGC nnU-Net for PyTorch **[DOC]**:

| config | AMP | FP32/TF32 | speedup |
|---|---:|---:|---:|
| **V100 16 GB, 1 GPU, BS 2, throughput (img/s)** | **9.65** | **2.07** | **4.66×** |
| **DGX-1 V100, 3D, 1 GPU, time-to-target-accuracy** | **201 min** | **680 min** | **3.38×** (74.31 % vs 74.33 %) |
| DGX-1 V100, **2D**, 1 GPU | 60 min | 114 min | 1.90× |
| A100, 3D, 1 GPU | 104 min | 167 min (TF32) | 1.61× |

**3D gets 3.4–4.7× on Volta while 2D gets 1.9×** — the generic "AMP is 1.5–2×" rule badly
understates 3D convolutions, because fp32 3D conv has no tensor-core path. MONAI additionally
documents **8 → 20 patches of 64³ at the same VRAM (2.5× batch)** **[DOC]**, which matters on a
16 GB T4.

**[UNVERIFIED]** No published 3D-U-Net AMP measurement **on a T4**. The T4's FP16:FP32 peak ratio is
65:8.1 ≈ 8:1, the same architectural situation as V100's 125:15.7, but its 320 GB/s bandwidth (vs
900) argues for the low end. Treat "3–4× on T4" as extrapolation. Our own falsification is already
written: `h1_execution_spec` §3 requires the first Colab run to report measured s/step fp16 vs fp32
and to escalate to the pre-materialised-tensor fix if the speed-up is < 1.3×.

### 5.2 The two defects that are one bias implemented twice

This is the sharpest structural point in the whole fix bundle, and it did not appear in the prior
reports:

> **`det_neg_weight = 1e-2` (F1) and checkpoint selection on `test_acc × test_recall` with no
> precision term (F2) push in the SAME direction — more positives — and that direction is the
> OPPOSITE of a 59 %-false-positive error budget.** At 0.1 % prevalence `test_acc` is pinned near
> 0.997 regardless of threshold (measured at **0.9973 on random logits**), so the selector is
> effectively **a pure recall maximiser**. They are not two independent bugs; they are one
> systematic bias applied twice, and fixing only one may under-deliver. **[INFERENCE, from
> measured `test_acc` degeneracy + the arithmetic of the selector]**

What the literature prescribes instead **[DOC]**:

- **nnU-Net** (Nature Methods 2021; [arXiv:1809.10486](https://arxiv.org/abs/1809.10486)) — the most
  validated recipe at our foreground fraction — uses **no per-class weight at all**. It attacks
  imbalance with **sampling geometry**: 33.3 % of patches forced to contain foreground, plus
  **batch-Dice** and Dice+CE. A 1e-2 negative weight is not standard practice; it is an extreme
  outlier.
- **Unified Focal loss** ([arXiv:2102.04525](https://arxiv.org/abs/2102.04525), CMIG 2022) — same
  network, 7 losses, 5 datasets. On the most imbalanced 3D task (KiTS19 tumour): CE **0.336**,
  Dice 0.536, Combo 0.554, Unified Focal asym **0.634** DSC. But on BraTS20, **pure Dice (0.620) and
  Tversky (0.580) are both WORSE than plain CE (0.716)** — "region loss always wins under imbalance"
  is false. And the precision/recall direction is documented: Tversky buys recall at precision's
  expense (BraTS20 recall 0.740 / precision 0.525 vs CE 0.682 / 0.826).
- **Loss Odyssey** (Ma et al., *Med. Image Anal.* 71:102035, 2021; 20 losses, 4 tasks, 6 datasets):
  *"compound loss functions are the most robust losses, especially for the highly imbalanced
  segmentation tasks."* Per-dataset numerics **[UNVERIFIED]** — PDF would not extract.
- **Threshold, not weight.** [Menon et al., ICLR 2021](https://arxiv.org/abs/2007.07314): post-hoc
  logit adjustment — a pure threshold shift at **zero training cost** — captures **4.56 of the 4.83
  points** the in-loss version buys on CIFAR-10-LT (94 %), and 4.68 of 5.02 on iNaturalist (93 %).
  And their caveat cuts the other way too: *"balancing has minimal effect in separable settings"* —
  loss re-weighting scales gradients without necessarily moving the argmax, while still distorting
  calibration. Elkan's classic result (IJCAI 2001) is the formal statement that weighting and
  threshold-shifting are equivalent at the Bayes optimum **[DOC; verbatim algebra UNVERIFIED]**.

**Prescription [INFERENCE, grounded above]:** kill the 1e-2 negative weight, adopt Dice+CE with
forced-foreground sampling, and **move the operating point post-hoc on validation** — which costs
zero of our 45 hours and is separable from the retrain. Independently, this is consistent with our
own measurement that the detection-threshold lever is under water on both folds
(`detection_threshold_2026-08-18.md` §0.1).

For checkpoint selection: **[UNVERIFIED]** — no controlled study isolating "which validation metric
you select on" and measuring the final-performance delta exists; this is genuine folklore. What is
documented: nnU-Net **uses the final epoch**, not a best checkpoint; greedy model soups
([arXiv:2203.05482](https://arxiv.org/abs/2203.05482), ICML 2022) give **85.02 % vs 84.68 %** on
distribution shifts, i.e. **+0.34** over the best individual model at no inference cost; NMT
last-5 checkpoint averaging gives ~**+0.14 BLEU** "for free". And a warning:
[McDermott et al., NeurIPS 2024](https://arxiv.org/abs/2401.06091) prove that **AUPRC is not
generally superior under imbalance and can be actively harmful**, having surveyed 1.5 M papers and
found the received claim misattributed. **So: select on the task metric, or don't select — take the
last checkpoint or an EMA/SWA average, which removes the selector from the critical path and adds a
small documented free gain.** Our E5 patch (`H1R_SELECT=link_f1`) remains the right minimum.

### 5.3 Window size 2 → 3

Trackastra's ablation, verbatim: *"When using only s = 2 we find that the performance substantially
deteriorates, whereas window sizes s ∈ (3,6) all lead to good results"* **[DOC]**. The magnitude is
**[UNVERIFIED]** — Figure 4c is a plot, not a table.

Cost model at our ~300 tokens/frame **[INFERENCE]**: window 2 → 4 doubles tokens (600 → 1200),
quadruples attention entries, and multiplies projection FLOPs by 2 — expect **3–4× on the edge head
only**, and 4× vanilla attention activation memory. Trackastra itself caps at **2048 tokens/window
at s = 6**, i.e. ~340 objects/frame, so our 300/frame at s = 4 sits comfortably inside their
operating envelope **[DOC + INFERENCE]**. **s = 3 or 4 captures the documented benefit at the
lowest cost**; `edge_training_frontier` R6 deferred this on compute grounds under a 30 h budget,
and at 45 h it is affordable.

Corroboration from MOT: MeMOTR's DanceTrack ablation gives **+2.8 HOTA / +4.4 AssA** from long-term
memory + memory attention (61.1 → 63.9) **[DOC]**; SUSHI gets **77.6 vs 73.7 IDF1** from a 9-level
512-frame hierarchy **[DOC]**.

### 5.4 Graph augmentation — and the one augmentation that targets OUR error

The augmentation table in `edge_training_frontier` §3.3 is sound but is missing the item that
matters most for a 59 %-FP budget. Filling that gap:

- **PolarMOT** ([arXiv:2208.01957](https://arxiv.org/abs/2208.01957), ECCV 2022) is the only paper
  found that **explicitly injects synthetic false-positive nodes**: *"we add random bounding box
  detections at each frame before the graph construction to imitate false positive detections"*,
  alongside node dropout of **40–60 % per frame**, removal of **~20 % of all edges**, and
  class-specific Gaussian feature noise **[DOC]**. No per-augmentation ablation is published.
- **OGR3MOT** ([arXiv:2104.11747](https://arxiv.org/abs/2104.11747), RA-L 2022) supplies the
  effect size, bundled: nuScenes AMOTA **0.601 → 0.654 (CenterPoint) and 0.538 → 0.587 (MEGVII)**
  in the offline regime — **up to +5.3 AMOTA from detection-noise augmentation alone** — motivated
  explicitly by *"detector overfitting"*, where training detections are better than inference ones
  **[DOC]**. The gain is near-zero in their online regime, so treat +5.3 as regime-dependent.
- **Trackastra does NOT do node dropout, node insertion, or FP simulation** and states the
  limitation itself: *"it currently does not correct faulty detection inputs"* **[DOC]**. That is an
  open hole in the closest published system — an argument for adding it, and a warning that no
  cell-tracking-domain number exists.
- **DropEdge** ([arXiv:1907.10903](https://arxiv.org/abs/1907.10903), ICLR 2020) gives **+0.4 to
  +2.8** at shallow depth (Cora GCN 2-layer 86.10 → 86.50; Citeseer 75.90 → 78.70) and +13.5 % at
  64 layers **[DOC]**. Its headline benefit is anti-over-smoothing in *deep GCNs* — not our problem,
  and our head is a transformer. **Low EV; do not spend an ablation arm on it.**
- **GraphCL** ([arXiv:2010.13902](https://arxiv.org/abs/2010.13902), NeurIPS 2020) documents that
  **edge perturbation was HARMFUL on NCI1 and helpful on the denser PROTEINS** — augmentation choice
  is dataset-specific and must be validated **[DOC]**.

> **Only FP-node insertion attacks our dominant error mode. Node dropout attacks the opposite one.
> If exactly one augmentation is added, add FP-node insertion, at an injection rate calibrated to
> the detector's MEASURED FP rate at the deployed threshold — not an arbitrary number.**

### 5.5 Warm-start ordering — free, and the biggest documented OOD number in the bundle

**LP-FT** ([Kumar et al., ICLR 2022, arXiv:2202.10054](https://arxiv.org/abs/2202.10054)) across 10
distribution-shift datasets: full fine-tuning gets **+2 % ID but −7 % OOD** versus linear probing;
**linear-probe-then-fine-tune gets +1 % ID and +10 % OOD over full fine-tuning** **[DOC]**.
Mechanism, verbatim: *"while fine-tuning learns the head, the lower layers … change simultaneously
and distort the pretrained features."*

**[INFERENCE]** This is exactly our situation if we attach a *newly initialised* matchability head
(§6 E-A) to a warm-started detector: the head's large early gradients will flow into and distort the
public 50-epoch weights. **Freeze the UNet (or run it at ~0 LR) until the new head converges, then
unfreeze at low LR.** Zero extra GPU-hours — a schedule change.

**Surgical fine-tuning** ([Lee et al., ICLR 2023, arXiv:2210.11466](https://arxiv.org/abs/2210.11466)):
tuning a *subset* of layers matches or beats full fine-tuning across 7 tasks and 3 shift types, and
**which subset depends on the shift**: for image-corruption / low-level appearance shift, tune the
**first few layers** **[DOC]**. **[INFERENCE]** Zebrahub → competition is an imaging/appearance
shift, so tune the early encoder and keep deep layers frozen — which is also what
`edge_training_frontier` §5 argued from the transfer-risk side, now with a citation.

Catastrophic forgetting is a real risk, not a theoretical one: documented degradation ranges from
**7 % to 55 %** depending on LR, steps and shift magnitude (CLIP fine-tuning loses 16.24 %/17.12 %
zero-shot; RoentGen loses 43 % at 1k steps) **[DOC]**. The best-evidenced mitigation is **not**
sequential phases but **mixing in-domain data into every batch** — which is Arazo et al.'s
documented anti-confirmation-bias fix and is already our B1 spec.

### 5.6 The Ultrack-imitation ceiling — none of the "student beats teacher" preconditions currently hold

Noisy Student ([arXiv:1911.04252](https://arxiv.org/abs/1911.04252), CVPR 2020), Table 6 — the
load-bearing ablation **[DOC]**:

| configuration | 1.3 M unlabeled | 130 M unlabeled |
|---|---:|---:|
| EfficientNet-B5 baseline | 83.3 % | 84.0 % |
| Noisy Student Training | **83.9 %** | **85.1 %** |
| – without augmentation | 83.6 % | 84.6 % |
| – **without augmentation, stochastic depth, dropout** | **83.2 %** | 84.3 % |

**Strip the noise and the student falls BELOW the baseline (83.2 vs 83.3).** Verbatim: *"noise …
plays an important role in enabling the student model to perform better than the teacher."*
Born-Again Networks ([arXiv:1805.04770](https://arxiv.org/abs/1805.04770), ICML 2018) show a student
of identical architecture beating its teacher by **0.3–1.7 points** (CIFAR-100 DenseNet-90-60
17.69 % → 16.62 % → 16.44 %) — but with a teacher trained on *true* labels on the *same*
distribution **[DOC]**. And "Does Knowledge Distillation Really Work?"
([arXiv:2106.05945](https://arxiv.org/abs/2106.05945), NeurIPS 2021) documents that *"more closely
matching the teacher paradoxically does not always lead to better student generalization"* **[DOC]**.

Mapped onto our zh001r plan **[INFERENCE]**:

| documented precondition for student > teacher | our status |
|---|---|
| teacher trained on true labels, same distribution | ✗ Ultrack is a non-learned ILP on a different domain |
| **student is noised, teacher is not** | ✗ **we currently have no graph/node augmentation at all** |
| student capacity ≥ teacher | ill-defined against an ILP |
| soft pseudo-labels for out-of-domain data | ✗ Ultrack emits hard assignments |
| confidence filtering of pseudo-labels | ✗ not done |
| minimum labelled samples per mini-batch (Arazo) | ✗ if trained in sequential phases |

**[UNVERIFIED]** No published result shows a learned student exceeding a non-learned combinatorial
tracking teacher. **Treat "we beat Ultrack by imitating Ultrack" as unsupported unless the Noisy
Student conditions are deliberately engineered in** — which, usefully, means the §5.4 augmentation
bundle is not merely a robustness nicety; it is **the precondition that makes the H1 retrain able to
exceed its own labels at all.** That reframing raises the augmentation bundle's priority
substantially.

---

## 6. The ranked, costed plan — five experiments inside ~45 GPU-hours

GPU-hour anchors from `h1_execution_spec` §4.7 / §5.6: detector half ~0.5 GPU-h at fp16 for 20
epochs; edge half ~1.5–3 GPU-h for 20 epochs (~2 s/step, Python-loop-bound, not FLOP-bound).

Every experiment states its predicted effect **on the 59.0 % deletion-precision bar specifically**,
and a falsification that can be checked without a leaderboard slot.

---

### **Experiment 1 — Zero-GPU: find the signal inside the override pool.** 0 GPU-h, ~1 day CPU.

The measurements in §2 relocated the problem: we need **63.3 % precision inside a 15,482-edge pool
whose base rate is already ~53 %**, not 59 % across the whole graph. That is a 10-point lift.
Before booking any GPU, exhaust the signals that already exist.

Substrate: the full fold-1 (128-crop) and fold-0 (71-crop) LOEO exports, with the pool labelled
TP/FP by the scorer's own matching rule (the reusable asset is `els_bonus_ablation.py`'s verified
`motion_relink_edges` port, validated `sym_diff = 0`).

Signals to sweep, cheapest first — **none requires retraining**:

| # | signal | why it might work | provenance |
|---|---|---|---|
| 1a | **ILP objective gap from forcing the edge out** | an absolutely-scaled, *global* confidence; the practical form of the partition-function idea in Nature Methods 2025 §3.7 | **[DOC mechanism, INFERENCE transfer]** |
| 1b | **column margin** (top-1 minus top-2 share) and **column entropy** | available from the same forward pass; not currently exported | SuperGlue/LightGlue reciprocity features **[DOC]** |
| 1c | **reverse-direction (row) margin** and **mutual-best/reciprocity** | the one thing the column softmax structurally cannot see (§1.1) | **[DOC]** |
| 1d | **motion residual conditioned on local density** | our own best geometric signal (57.3 %) is unconditioned | **[MEASURED baseline]** |
| 1e | **a logistic regression on 1a–1d** | HOCT's own HITL result: frozen backbone + logistic head, **−58.6 % AOGM in 400 annotations** | **[DOC]** |
| 1f | **temperature and conditional temperature on the column** | TS is **not** rank-preserving on a share (§3.1); *T* tracks displacement scale (§3.7) | **[DOC]** |

**Blocking precondition:** 1b, 1c and 1f all need the **full column of logits**, which we do not
export. That is a one-line change in `predict_unet_transformer.py` plus one Kaggle inference kernel
per fold. Do this first.

**Falsification.** If no combination of existing signals reaches **63.3 % deletion precision inside
the pool at ≥ 20 % pool deletion** (and **58.7 %** on the deployed-substrate `J`), abstention is not
reachable without retraining — which is the case *for* Experiments 2–3, not against them. If some
combination *does* clear it, take the gain: it costs zero GPU-hours and re-baselines everything
below.
**Predicted effect on the bar:** best case, the pool's ~53 % base rate needs only modest
within-pool AUC to clear 63.3 %. **[INFERENCE]** a signal with pool-internal AUC ≥ 0.62 should do
it; our whole-graph AUC is already 0.701. This is the highest expected-value item in the report.

---

### **Experiment 2 — E-A: matchability head + background term + auxiliary sigmoid, one arm.** ~4 GPU-h.

Three changes, shipped together because they are three expressions of the same missing concept, and
because §5.5's LP-FT result says a fresh head must be introduced under a freeze schedule anyway.

- **E-A1 — background/dustbin term** in `compute_loss`: already written and self-tested
  (`h1r_edge_loss_patch.py` E1, `H1R_BG_TERM`). Trackastra's `1 + Σexp` **[DOC]**. Must ship with
  its inference companion E6 or the deployed threshold silently changes scale.
- **E-A2 — a second `pair_mlp` output channel**, trained with plain BCE-with-logits on the same
  targets: a **per-edge matchability**, absolute and candidate-count-invariant. **+65 parameters
  [MEASURED].** Initialise its bias at `−log((1−π)/π)` with π set to the measured positive rate
  **[DOC, focal-loss prior-bias trick]**. Weight it **well above Trackastra's λ = 1e-2** — that value
  makes the channel a free rider, and free riders are not calibrated. This is LightGlue's
  disentangled matchability, whose stated rationale is that dustbins *"entangle the similarity score
  of all points"* **[DOC]**.
- **E-A3 — intra-frame self-attention**: one self-attention block per layer inside each frame.
  **+0.805 GF at N = 256 = 12.9 % of the edge head, ≈ 0.02 % of a training step [MEASURED].**
  Rationale: competition among *children* for a parent is currently uncomputable at any level
  (§1.1); SuperGlue ablates its attention structure at **84.4 % → 74.0 % precision** **[DOC]**.

Schedule: **UNet frozen** while the new channels converge, then unfreeze at low LR (LP-FT, **+10 %
OOD** **[DOC]**). Selection on `H1R_SELECT=link_f1`, never `test_acc`.

**Falsification, with the number.** On a held-out embryo, E-A must (i) lift TP-vs-FP AUC above the
current **0.701**, and (ii) reach **> 63.3 % deletion precision inside the override pool at ≥ 20 %
pool deletion** and **> 58.7 %** at the deployed `J`. Miss either and the lane is dead — report and
stop. Baseline to beat: the best existing signal is geometry's 57.3 % at 1 % whole-graph deletion,
and geometry is measured (§0.3) to rank *backwards* inside the pool.
**Predicted effect on the bar [INFERENCE]:** an absolute channel is the only mechanism that
addresses the isolated-but-spurious FP class (98.5 %/99.4 % of FPs, §1.2). If it works, §3.3's
sensitivity table puts it at **+0.010 to +0.040** on 6bba edge Jaccard.

---

### **Experiment 3 — the linker that uses it: a dustbin in the assignment.** 0 GPU-h + 1 Kaggle kernel.

Without this, Experiment 2 returns approximately zero — the same conclusion `edge_loss_structure`
§5.1 reached, now with a concrete implementation.

Add `BIOHUB_MOTION_RELINK_ABSTAIN_COST` to `assign_pass` (`wrapper.py:309-345`): augment the cost
matrix with a dustbin row and column at cost `c`, so `linear_sum_assignment` can leave a node
unassigned when every within-gate partner costs more than `c`. This is the standard LAP-with-dummies
formulation and the discrete analogue of SuperGlue's learnable dustbin scalar **[DOC]**. Two further
one-line repairs in the same function:

- `learned_prob` returns **0.0** for unnominated pairs (`wrapper.py:283-291`), collapsing "never
  scored" and "scored zero" into the same value. Emit a distinct sentinel and let the cost function
  treat the two differently.
- Sweep `c` on a **fixed deletion FRACTION**, not a fixed cost value — §3.6's transportability
  argument, and it makes any monotone recalibration irrelevant.

**Falsification.** Sweep `c` through the verified CPU linker replay and report Δ `adj_edge_jaccard`
on **both folds separately** against the ±0.003 fold-0 noise floor. If the best achievable Δ is
< +0.006 pooled, the lane does not clear the floor.
**Predicted effect on the bar:** this experiment does not move the bar; it is what converts
clearing the bar into score. It is a multiplier on Experiments 1 and 2 and worth zero alone.

---

### **Experiment 4 — the bundled retrain: augmentation + plumbing fixes, two arms.** ~12–15 GPU-h.

Everything in §5 that is a config constant, plus the one ablation whose sign is genuinely unknown.

Constants (no arms spent): AMP via `binary_cross_entropy_with_logits` (§5.1, **3.38× end-to-end on
V100 3D** **[DOC]**); `det_neg_weight` off 1e-2 with forced-foreground sampling (§5.2); window
**2 → 3** (§5.3); `H1R_SELECT=link_f1` or last-checkpoint + EMA (§5.2); LP-FT freeze schedule
(§5.5); real division weight and column masking (`h1r_edge_loss_patch` E3); interleaved
competition batches in every Zebrahub batch (§5.5, §5.6).

The one arm worth paying for: **augmentation bundle ON vs OFF**, where ON = FP-node insertion at the
detector's measured FP rate + node dropout + coordinate jitter + cue dropout (§5.4). This is the
arm with the largest documented external effect (**up to +5.3 AMOTA**, OGR3MOT **[DOC]**) *and* the
one that §5.6 identifies as the **precondition for the H1 student to exceed its Ultrack teacher at
all**.

**Falsification.** Held-out-embryo edge accuracy plus LOEO in both embryo directions. Kill the
bundle if Δ is inside the seed-to-seed spread. Separately: if raising `det_neg_weight` on dense data
does not move `N_pred/N_est` downward, the structural-over-detection mechanism story is wrong and
the detector half must be re-motivated (`h1_execution_spec` §4.8 falsification, restated).
**Predicted effect on the bar [INFERENCE]:** FP-node insertion is the only intervention that trains
the head to recognise a spurious detection as spurious — i.e. the only one whose *mechanism* is the
59 %/63.3 % bar. But no cell-tracking-domain number exists for it, so budget it as a robustness
enabler for Experiment 2, not as a standalone score lever.

---

### **Experiment 5 — deep ensemble of the retrained head, if hours remain.** ~6–8 GPU-h.

The edge head is **576,385 parameters [MEASURED]** — training M = 5 seeds is cheap relative to the
detector. Ovadia et al. is the only intervention in this review with **documented robustness to
exactly the held-out-embryo shift we face** (ranking under shift: Ensembles > Dropout > Temperature
Scaling > Vanilla; *"most of the gains … with only 5 models"*) **[DOC]**.

**Falsification.** If the ensemble's TP/FP AUC and pool deletion precision on the held-out embryo do
not exceed the best single seed by more than the seed-to-seed spread, ensembling is dead weight at
our scale.
**Predicted effect on the bar:** ensembling improves the *tail* of the score distribution, which is
precisely what deletion precision at a small fraction depends on **[INFERENCE]**. It is the natural
last spend, not the first.

---

### Explicitly NOT in the plan, and why

| item | reason |
|---|---|
| **Anything in the MOTR / MOTIP / MeMOTR family** | 190–480 GPU-h per run; zero dense-3D-bio evidence; produces a share, not an absolute score; documented newborn suppression would attack our division sub-metric (§4.2) **[DOC]** |
| **A full HOCT port** | ~60 GB peak, 256×512×512 crops, Muon dependency — a 2–3 week project, not a 45-hour one. Its **frozen-backbone + logistic-head** loop is stolen into Experiment 1 instead (§4.3) |
| **Evidential / Dirichlet heads** | two NeurIPS papers show the epistemic signal is a λ-tuning artifact; no head-to-head win over temperature scaling exists (§3.4) **[DOC]** |
| **Conformal prediction as a source of gain** | it relocates the threshold; it does not create signal. Adopt it as a *guard* on the final rule once Experiment 1 or 2 clears the bar (§3.5) |
| **DropEdge** | headline benefit is anti-over-smoothing in deep GCNs; our head is a shallow transformer; documented shallow gain +0.4 to +2.8 on citation graphs (§5.4) **[DOC]** |
| **λ_div sweeps** | return is **0** on the deployed path until inference can emit forks; §2.1 measures every post-wrapper division as geometry-proposed with `edge_prob = None` |
| **Contrastive / InfoNCE edge objectives (CAMELTrack-style)** | produces an embedding distance, not a probability — the wrong object for abstention (§4.4); and identity embeddings break structurally at a division |

### Budget

| experiment | GPU-h | cumulative |
|---|---:|---:|
| 1 — zero-GPU signal search in the pool | **0** (+1 export kernel/fold, Kaggle) | 0 |
| 2 — E-A matchability + dustbin + self-attention | ~4 | 4 |
| 3 — linker dustbin, CPU replay | **0** (+1 kernel) | 4 |
| 4 — bundled retrain, augmentation on/off | ~12–15 | 16–19 |
| 5 — 5-seed edge-head ensemble | ~6–8 | 22–27 |
| reserve for the arm the noise floor makes ambiguous | ~18 | 45 |

**Sequencing is not negotiable.** Experiment 1 gates 2 (it may make it unnecessary, or may prove the
bar unreachable without it). Experiment 3 gates the *value* of 2 (without a linker that can reject,
a better score returns zero). Experiment 4 gates whether 2's head can generalise off Ultrack labels
at all. **Doing 2 alone returns approximately nothing — that is the single most important planning
consequence of this report, and it is unchanged from `edge_loss_structure` §5.1, now with the
linker-side implementation specified.**

---

## 7. What I could not verify

- **The override pool's exact FP fraction.** §2.4's 50–56 % is inverted from a crop-weighted metric
  with an assumed pooled `D`. A scorer-exact TP/FP labelling of the pool on the full fold settles
  it; CPU-only, out of this session's budget. **The enrichment factor (3.5–4×) and the sign are the
  load-bearing claims.**
- **Whether the 10-crop pilot substrate is representative.** It scores 0.5788 against the deployed
  fold-1 0.7042 — these crops are harder than average, which moves both the bar (63.3 % vs 58.7 %)
  and possibly the pool composition. Fold 0 was not A/B'd at all this session.
- **Temperature scaling on our actual scores.** Cannot be tested: the exports carry only the
  selected edge's probability, not the column. One-line export change plus one kernel.
- **Trackastra's window-size magnitudes** (Figure 4c is a plot), its optimizer/LR/epochs, and the
  ECE numbers in arXiv:2503.09244 (figures only). Carried forward unresolved from
  `edge_training_frontier` §9.
- **The Nature Methods 2025 "accurate error prediction" paper** — paywalled, snippet only. It is the
  most on-point external reference found and its partition-function formulation directly motivates
  Experiment 1a. Reading it properly is worth a session.
- **Any published focal-vs-BCE comparison of operating point for association scoring.** I believe it
  does not exist. Every cited focal number is AP (discrimination) or ECE (calibration).
- **Measured 3D-U-Net AMP speedup on a T4.** Extrapolated from V100 via the shared 8:1 peak ratio;
  the T4's 320 GB/s bandwidth argues for the low end.
- **Whether a retrained absolute score clears the bar.** This is the whole bet. Experiments 1 and 2
  are designed to settle it before any large GPU spend.

---

## 8. Licence tags (factual, one line each; nothing excluded or deranked on licence)

- **HOCT** — code and weights **MIT** (`royerlab/hoct`); paper text CC BY-NC-ND 4.0.  <!-- CORRECTED 2026-08-31: see banner; FACT-0452 -->
- **Trackastra** — **BSD-3-Clause** (verified from the repo LICENSE in a prior session; this
  *corrects* the Apache-2.0 claim still standing in `edge_training_frontier` §8's body text).
- **Ultrack** — CC BY 4.0 (paper); code at `royerlab/ultrack`.
- **MPNTrack / `mot_neural_solver`** — **MIT**, pretrained weights provided.
- **SUSHI** (dvl-tum) — licence **[UNVERIFIED]**; mechanisms and hyperparameters only.
- **Cell-Tracker-GNN** (`talbenha/cell-tracker-gnn`) — licence **[UNVERIFIED]**.
- **Cell-TRACTR** — PLOS CB (CC BY 4.0); weights on Zenodo 14509424.
- **CO-MOT** — CC BY 4.0. **SCATR** — CC BY 4.0.
- **CAMELTrack / TrackLab** — licence **[UNVERIFIED]**; mechanisms only.
- **CTC ERR_SEG data** — CTC terms of use apply if ever trained on directly **[UNVERIFIED terms]**.
- **Zebrahub tracks** — CC BY-NC, host-cleared (#734330), recorded in
  `research/04-data/data-acquisition.md`.

---

## 9. Session artifacts

Nothing was pushed, launched, submitted or committed. `vendor/` untouched. `.claude/settings.json`
and `.gitignore` untouched.

| artifact | what |
|---|---|
| this report | `research/06-knowledge-system/internal-reports/architecture_training_2026-08-19.md` |
| `armB_split1.csv.gz`, `armC25_split1.csv.gz`, `armC50_split1.csv.gz` | session scratchpad, outside Git — the three deletion arms of §2.3, rebuilt from `p3_d1_pilot_f1_v1` |
| `armA_f1.txt`, `armB_f1.txt`, `armC25_f1.txt`, `armC50_f1.txt` | raw official-scorer output for the four arms |

The reusable method, if it is wanted as an instrument: **join the final submission's edges against
`pregraphs_*.parquet` on `(dataset, source_id, target_id)` to isolate the wrapper-manufactured pool,
delete a ranked subset, and re-score.** Node ids survive the wrapper; coordinates do not. It costs
~10 CPU-minutes per arm on a 10-crop substrate and it is the cheapest paired instrument we have for
any edge-deletion hypothesis.
