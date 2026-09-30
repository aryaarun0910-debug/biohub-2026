# Edge-loss structure: what our objective optimises versus what the metric rewards — 2026-08-18

Mandate: test the hypothesis that **the edge head's LOSS STRUCTURE — not its capacity or its
data — is a first-order cause of the linking gap**, derive the disagreements between
`compute_loss` and `adj_edge_jaccard`, specify an ablatable patch set, and attribute the
measured +0.081 / +0.150 headroom.

Companion documents, cross-referenced rather than restated:
`edge_training_frontier_2026-08-18.md` (Trackastra's loss, ablations, augmentation spec),
`division_lane_2026-08-18.md` (fork proposal is the division bottleneck),
`retrain_recipes_2026-08-17.md` (F2/F5/F6 defects, AMP arithmetic),
`h1_execution_spec_2026-08-18.md` (`h1r_trainer_patch.py`, GPU-hour anchors).

Evidence labels on every claim:
**[MEASURED]** — run by me this session against real data, command and file named ·
**[CODE]** — read directly at the cited `file:line` ·
**[DOCUMENTED]** — stated in a cited external source ·
**[INFERENCE]** — my arithmetic or reasoning, assumptions stated ·
**[UNVERIFIED]** — could not confirm, with what would settle it.

---

## 0. HEADLINE — the hypothesis is half right, and the important half is falsified

**The loss structure is badly formed. It is also, right now, almost entirely disconnected from
the score.** Both halves are measured, not argued.

**0.1 The loss is malformed in exactly the way the mandate suspected [CODE + MEASURED].**
`compute_loss` (`vendor/kaggle-cell-tracking/scripts/train_unet_transformer.py:55-72`)
normalises with a bare column softmax, so **every column sums to exactly 1**: the objective
asserts that every node at t+1 has a parent among the detections at t. There is no "no parent"
outlet, and — the sharper consequence — the loss is **provably shift-invariant down a column**,
so it supplies *zero* gradient on the absolute logit level. The emitted `edge_prob` is therefore
a *share*, not a confidence. Verified numerically: shifting a whole column of logits by ±5–6
changes the vendored loss by `0.000e+00`; with Trackastra's `1 + Σexp` background term the same
shift changes it by `0.0359`
(`scripts/kaggle_edits/h1r_edge_loss_patch.py --self-test`, T2a/T2b) **[MEASURED]**.

**0.2 The learned edge head does not link anything [MEASURED].** The deployed linker is
`biotrack.wrapper.motion_relink_edges`, which **replaces the learned edge set wholesale**
(`src/biotrack/wrapper.py:1158-1163`) and consumes the learned probability only as a bonus in a
Hungarian cost, `cost = motion + 0.05·raw − 0.75·prob` (`wrapper.py:340`). I ported that function
(verified **edge-for-edge identical** to the vendored implementation on a 6,013-edge crop,
`sym_diff = 0`) and replayed it on the real pre-wrapper graph at three bonus values, over 12
6bba crops containing 9,020 metric-reachable GT edges:

| `MOTION_RELINK_LEARNED_BONUS` | reachable GT edges recovered | recall | edges changed vs bonus 0 |
|---|---:|---:|---:|
| **0.00** (learned prob OFF) | 8,263 | 0.9161 | — |
| **0.75** (deployed) | 8,264 | 0.9162 | 692 / 172,344 (0.4 %) |
| **3.00** (4× deployed) | 8,269 | 0.9167 | 1,891 / 172,344 (1.1 %) |

> **Deleting the learned edge probability entirely costs the deployed pipeline ONE reachable GT
> edge in 9,020 (0.011 %). Quadrupling its weight buys six.**

The reason is also measured: at bonus 0 the geometry linker already agrees with the head's own
nomination on **164,247 of 172,344 edges (95.3 %)**. There is nothing left for the bonus to flip.

**0.3 So a better loss cannot reach the score through RANKING. It can only reach it through
ABSTENTION.** The measured gap decomposition (§3) shows that **59 % (6bba) and 76 % (44b6) of the
linking headroom is false-positive edges, not missing ones** — and the deployed pipeline has no
mechanism to decline a link. Neither does the loss. That is the same missing concept at both
levels, and it is the one place where fixing the loss buys score.

**The abstention lane is live but unproven, and §3.3 reduces it to a single number.** Among the
edges the deployed linker actually selects, the learned `edge_prob` is the **best-discriminating
signal available** (AUC 0.701 for TP-vs-FP, ahead of the motion residual at 0.641 and raw
displacement at 0.630) — the head does know something the geometry does not. But **no signal
clears the break-even deletion precision of 59.0 % (6bba) / 53.1 % (44b6)**: the best operating
point is the *geometric* motion residual at 57.3 %, `ΔJ = −0.0002`, and the learned probability
manages only 49.6 % because its low tail is a spike of "never nominated" and the rest is
compressed into `[0.5, 1.0]` with an absolute level the loss provably does not constrain.
**Good AUC, bad operating point — the exact signature of a share masquerading as a confidence.**
Expected value if the bar is cleared: **+0.01 to +0.04** on 6bba edge Jaccard (§3.3).

**0.4 Divisions are irrelevant to `adj_edge_jaccard` and currently unreachable anyway
[MEASURED].** Only 151 of 128,883 GT edges (0.117 %) are second-daughter edges, so the entire
division population is worth ≤0.0012 of edge Jaccard. And the learned head emits **zero
divisions, ever**: across three independent pre-wrapper exports totalling 2,044,806 edges, the
out-degree histogram is `{1: N}` — every single source has exactly one child. λ_div work returns
**exactly zero** on the deployed path until inference changes. This independently corroborates
the sibling agent's inference-assignment lane from the opposite end of the pipeline (see §5.4).

---

## 1. `compute_loss`, line by line

`vendor/kaggle-cell-tracking/scripts/train_unet_transformer.py:55-72` **[CODE]**:

```
:57   active_rows = target.sum(dim=1) > 0        # source rows with >= 1 GT child
:58   active_cols = target.sum(dim=0) > 0        # target cols with a GT parent
:59   mask = active_rows.unsqueeze(1) | active_cols.unsqueeze(0)     # UNION, not intersection
:63   probs = torch.softmax(logits, dim=0)       # normalise over SOURCES, per column
:64   bce   = F.binary_cross_entropy(probs, target, reduction="none")
:65-66 p_t  = probs*target + (1-probs)*(1-target);  loss = ((1-p_t)**2) * bce
:68-70 div_rows = target.sum(1) > 1; weight = ones_like(loss); weight[div_rows] = 1.0   # NO-OP
:72   return (loss * weight)[mask].mean()
```

### 1.1 A node with no true parent

Under `:63` the columns of `probs` sum to **exactly 1** — including the column of a track birth
and the column of an unmatched false-positive detection. Verified: a 64×4 block of logits all at
−12 (the model maximally confident that nothing here is a parent) still produces column sums of
`1.000000`; with the background term the same block gives `3.9e-04`
(self-test T1a/T1b) **[MEASURED]**.

Two consequences.

- **In training.** A birth column is *inactive*, so only its cells in `active_rows` are
  supervised, all toward 0. Softmax conservation then moves that probability mass onto the
  **unsupervised** rows (unmatched detections). The objective's answer to "who is this node's
  parent?" for a genuinely parentless node is *"some unannotated nucleus"* — an arbitrary target
  the model cannot learn and is never corrected on.
- **At inference.** The same normalisation is applied
  (`predict_unet_transformer.py:455-458`) with `cfg.edge_activation = "softmax"` **[CODE]**, so
  the head is structurally incapable of saying "no parent". Measured on real output: every
  exported `edge_prob` lies in **[0.500, 1.000]** with median 0.79 (fold 0) / 0.838 (fold 1)
  and p25 0.679 — the floor is the 0.5 candidate threshold, not model confidence **[MEASURED,
  `_evidence/kaggle_runs/p3_d1_pilot_f0_v1/pregraphs_split0.parquet`,
  `_evidence/agent_runs/ws_f_armB_p0b_2026-08-01/raw/pregraphs_split1.parquet`]**.

### 1.2 A true division row

A dividing source `i` has two positives in **two different columns**, so the column softmax never
puts them in competition — the parental softmax is genuinely division-compatible, and this part
of the design is right. `:68-70` intends to upweight those rows and does not: `weight[div_rows] =
1.0` on a `torch.ones_like` tensor. `h1r_trainer_patch.py` P3 already fixes this to
`_H1R_DIV_WEIGHT` **[CODE]**.

**A trap the fix does not remove, and which the sibling report's masking recipe walks into.**
`edge_training_frontier_2026-08-18.md` §R1 prescribes "set loss weight 0 on every source row that
Zebrahub marks as dividing". **That does not withhold division supervision.** Because the softmax
is per column, a dividing row's logits keep normalising both daughter columns; zeroing the row
weight leaves a non-zero gradient there. Measured: with `weight[div_rows] = 0` the gradient on the
daughter columns is `2.1e-02`; removing the daughter **columns** from `mask` makes it exactly
`0.000e+00` (self-test T3b/T3c) **[MEASURED]**. Patch E3/`H1R_DIV_MASK_COLS` implements the
column removal.

### 1.3 The mask versus 97.2 % unannotated nuclei

Annotation density, measured from every `data/train/*.geff`'s
`attributes.geff.extra.estimated_number_of_nodes` against its actual node count **[MEASURED]**:

| embryo | crops | GT nodes | estimated true nodes | annotated |
|---|---:|---:|---:|---:|
| 44b6 | 71 | 20,197 | 2,618,970 | **0.77 %** |
| 6bba | 128 | 113,121 | 2,106,147 | **5.37 %** |
| pooled | 199 | 133,318 | 4,725,117 | **2.82 %** |

That reproduces the brief's 97.2 % figure exactly. Now the structure of the supervision, censused
over all 199 crops / 18,707 frame pairs **[MEASURED, GT-node view]**:

| | 44b6 | 6bba |
|---|---:|---:|
| frame pairs | 6,315 | 12,392 |
| GT edges | 19,826 | 109,057 |
| mean GT nodes / frame | 3.17 | 9.05 |
| mean active rows / pair | 3.135 | 8.791 |
| mean active cols / pair | 3.140 | 8.801 |
| division rows (total) | 26 | 125 |
| division rows as % of active rows | 0.131 % | 0.115 % |
| **union mask coverage of the GT×GT matrix** | **99.99 %** | **99.80 %** |
| positives as % of masked cells | 19.24 % | 8.57 % |

**First finding: on GT nodes the "sparse-annotation mask" does nothing at all** — almost every
annotated node has both an in- and an out-edge, so `active_rows | active_cols` is essentially the
whole matrix. The mask cannot be protecting us from unannotated nuclei, because unannotated
nuclei are *not nodes in the GT graph*.

The mask only becomes meaningful in the real training loop, where the tokens are **detected
peaks**, not GT nodes (`train_epoch:844-878` → `detect_and_match:620` →
`build_matched_edge_targets:747`) **[CODE]**. There, with ~302 detections per frame
(measured mean 301.6, p95 493, max 537 on the pre-wrapper export **[MEASURED]**) and 3–9 of them
annotated:

- **an ACTIVE column is fully supervised** — all ~302 rows are in the mask, and the labels there
  are clean (the annotated target's true parent is itself annotated), so there is no
  sparse-label noise inside the mask. *This part of the design is correct and must not be
  "fixed".*
- **an INACTIVE column gets 3–9 supervised cells out of ~302**, all zeros, and the softmax then
  conserves the remaining mass onto the ~295 unsupervised rows.
- **97–99 % of columns are inactive**: 3.14/301.6 = 1.0 % of columns are active for 44b6,
  8.80/301.6 = 2.9 % for 6bba **[INFERENCE from the two measured means]**.

**Second finding: essentially every edge the head emits at inference comes from a column the loss
never meaningfully touched, and for those columns the bare softmax makes abstention
representationally impossible.**

A related train/deploy mismatch, unflagged elsewhere **[CODE]**: training thresholds **raw
logits** at 0.3 (`detect_and_match` `det_threshold=0.3`, applied as `det_logits > det_threshold`
at `:681`), while inference thresholds the **sigmoid** at 0.96875–0.99
(`_detect_cells_pooled:285`, `--det-threshold` default 0.99). Logit 0.3 ≈ sigmoid 0.574, so the
training token set is far more permissive and the softmax column length differs systematically
between the two regimes. Under a shift-invariant loss (§0.1) nothing pins the probability scale
across that shift. The exact ratio of token counts is **[UNVERIFIED]** — one instrumented
training step would settle it.

### 1.4 Does the focal term help or hurt?

`loss = ((1-p_t)**2) * bce` on **softmax** probabilities. For a negative cell, `bce = −log(1−p)`
≈ `p` and the focal factor is `p²`, so the contribution is **≈ p³**. Measured at p = 0.01: plain
`1.005e-02` versus focal `1.005e-06`, a **1000× suppression** (self-test T5) **[MEASURED]**.

Verdict: under this sparsity the focal term neither helps nor hurts in the way the literature
intends — it makes the objective **almost purely positive-driven**. It does not down-weight
mislabelled negatives (there are none inside the mask, §1.3), and it does not mine hard negatives
(the softmax already suppresses them). It removes the one force that could push a whole column
down. Combined with the missing background term, **the loss contains no term whose minimisation
lowers total column mass.** γ is worth a knob (`H1R_FOCAL_GAMMA`) but is not the lever;
γ=0 is a free-rider arm, not an experiment.

---

## 2. Every disagreement between the objective and `adj_edge_jaccard`

Scorer references: `vendor/kaggle-cell-tracking/src/tracking_cellmot/metrics.py`.

| # | The objective says | The metric says | Consequence |
|---|---|---|---|
| **D1** | every target has a parent (column sums to 1, `:63`) | an omitted edge on an unlinkable target costs one FN; an emitted wrong edge costs one FN **and** one FP (`metrics.py:314-317`) | the metric strictly rewards abstention; the loss cannot express it. **Largest lever (§3).** |
| **D2** | absolute logit level is free (shift-invariant, T2a) | thresholds and cost bonuses are absolute quantities (`predict:465`, `wrapper.py:340`) | `edge_prob` is uncalibrated by construction; no threshold on it can mean "confident" |
| **D3** | all masked cells count equally, `.mean()` at `:72` | only edges with `pred_valid = out_valid \| in_valid` are scored (`metrics.py:193-198`); edges between two unannotated nodes are **free** | the loss spends its (small) negative pressure on cells the metric ignores. Benign, but it means loss deltas do not track score deltas |
| **D4** | division rows matter (`:68-70`) | they are worth 0.117 % of edge Jaccard, plus a separate `0.1 · division_jaccard` term (`metrics.py:34`) | λ_div is a *division-Jaccard* lever only, and only if inference can emit forks |
| **D5** | the selection metric is `test_acc × …` (`:1180`, and `h1r_trainer_patch` P2) | association correctness | `test_acc` thresholds a column softmax at 0.5. Measured on the patched trainer with **random** 200×200 logits and 10 GT edges: `correct/total = 3700/3710 = 0.9973` while top-1 parent accuracy is `0.0` **[MEASURED]**. The checkpoint is selected on a near-constant **[fixed by E5]** |
| **D6** | one score per edge (the softmax share) | the linker wants both a *ranking* and a *confidence* | no absolute channel exists; Trackastra's auxiliary sigmoid at λ=1e-2 supplies one **[DOCUMENTED, arXiv:2405.15700 §3.2]** **[fixed by E2]** |
| **D7** | γ=2 focal, so negatives contribute ≈p³ (T5) | FP edges on annotated nodes are the majority of the loss (§3) | nothing in the objective pushes total column mass down |
| **D8** | a zero row weight withholds supervision | — | it does not, under a column softmax (T3b) **[fixed by E3 `H1R_DIV_MASK_COLS`]** |

---

## 3. Where the +0.081 / +0.150 actually lives — measured

Method: reimplemented `metrics._evaluate_matched_graph`'s `pred_valid` rule
(`metrics.py:157-198`) and the consecutive-frame filter (`metrics.py:74-87`) over the deployed
LOEO submissions (`c:/temp/subvoxel_f{0,1}/loeo_split{0,1}_strict.csv.gz`), with node matching by
per-frame optimal 1-1 assignment at 7 µm in physical units. The recomputed *reachable* fraction
lands on the brief's independently-measured ceilings to three decimals (44b6 0.9816 vs 0.9830;
6bba 0.8533 vs 0.8537), which validates the reimplementation **[MEASURED]**.

| | 44b6 (fold 0, 71 crops) | 6bba (fold 1, 128 crops) |
|---|---:|---:|
| GT edges | 19,826 | 109,057 |
| **reachable** (both endpoints matched ≤7 µm) | 19,462 (98.16 %) | 93,055 (85.33 %) |
| TP | 18,746 | 86,195 |
| scored (`pred_valid`) predicted edges | 20,130 | 100,550 |
| **FP** | **1,384** | **14,355** |
| — of which exactly one endpoint unmatched | 1,363 (98.5 %) | 14,264 (99.4 %) |
| — of which both matched, wrong pair | 21 | 91 |
| FN total | 1,080 | 22,862 |
| — unreachable (detector) | 364 | 16,002 |
| — **reachable but unlinked** | **716** | **6,860** |
| recomputed edge Jaccard | 0.8838 | 0.6984 |
| J if every FP were deleted | **0.9455 (+0.0617)** | **0.7904 (+0.0920)** |
| J if every reachable edge were linked (best case) | 0.9497 (+0.0659) | 0.7984 (+0.1000) |
| ceiling | 0.9816 | 0.8533 |

**Split of the linking headroom: 76 % false-positive / 24 % missing-link on 44b6, and
59 % / 41 % on 6bba** (ratios of the two single-lever bounds against my own consistent
`J`) **[MEASURED]**.

### 3.1 The candidate set is not the bottleneck either

| stage | 44b6 | 6bba |
|---|---:|---:|
| learned head's raw candidate set contains … of the reachable GT edges | 92.90 % (9 crops) | 89.13 % (128 crops) |
| **deployed final edge set contains** | **96.32 %** | **92.63 %** |

**The geometry linker recovers MORE reachable GT edges than the learned head nominated.** A
better-ranking head is being asked to improve on something that already out-performs it.

### 3.2 The exchange rate for abstention [INFERENCE, from the measured counts]

With `J = TP/(GT + V − TP)`: deleting one FP changes `J` by `+J/D`; deleting one TP by `−1/D`
(`D = GT + V − TP`). Break-even therefore requires the deleted set to contain **more than
`1/J` FP per TP**:

- 6bba: `1/0.6984 = 1.43` FP per TP, i.e. deletion precision > **58.9 %**
- 44b6: `1/0.8838 = 1.13` FP per TP, i.e. deletion precision > **53.1 %**

(§3.3 works on a 24-crop 6bba subset where `J = 0.6944`, giving 1.44 / **59.0 %**. The two 6bba
numbers are the same quantity on different substrates; 59.0 % is the one to design against, being
the stricter.)

That is the number any abstention rule must beat. It is a modest bar — and today nothing in the
pipeline clears it (§3.3).

### 3.3 Can today's signals abstain? — the falsification test [MEASURED]

Replayed the deployed linker on 24 6bba crops, labelled every scorer-evaluable selected edge
TP/FP by the rule above, and swept deletion thresholds on the three quantities the pipeline
already has: the learned `edge_prob`, the motion-predicted distance, and the raw distance.

Substrate: 24 crops, 19,663 GT edges, 18,269 scorer-evaluable selected edges, TP 15,545,
FP 2,724, `J = 0.6944`, break-even ratio **1.44 FP per TP deleted** (precision **59.0 %**).
4.15 % of selected edges carry no learned probability at all (they were never nominated by the
head), which quantises `edge_prob`'s low tail.

**Global discrimination (AUC for TP vs FP among selected edges):**

| signal | AUC |
|---|---:|
| **learned `edge_prob`** | **0.701** |
| motion-predicted residual `−motion_um` | 0.641 |
| raw displacement `−raw_um` | 0.630 |

**Deletion sweeps (delete the lowest-scoring edges):**

| signal | delete | edges | TP lost | FP removed | FP:TP | **precision** | Δ edge Jaccard |
|---|---:|---:|---:|---:|---:|---:|---:|
| `edge_prob` | 1–2 % * | 758 | 382 | 376 | 0.98 | 49.6 % | **−0.0055** |
| `edge_prob` | 5 % | 914 | 505 | 409 | 0.81 | 44.8 % | −0.0101 |
| `edge_prob` | 10 % | 1,827 | 1,166 | 661 | 0.57 | 36.2 % | −0.0325 |
| `motion_um` | **1 %** | 185 | 79 | 106 | **1.34** | **57.3 %** | **−0.0002** |
| `motion_um` | 2 % | 388 | 196 | 192 | 0.98 | 49.5 % | −0.0028 |
| `raw_um` | 1 % | 227 | 108 | 119 | 1.10 | 52.4 % | −0.0011 |
| `raw_um` | 5 % | 950 | 611 | 339 | 0.55 | 35.7 % | −0.0170 |

\* identical rows: the 1 % and 2 % quantiles both land on `edge_prob = 0`.

Two things are true at once, and both matter.

1. **The learned probability is the best-discriminating signal we have** — AUC 0.701, ahead of
   both geometric quantities. The head genuinely knows something the linker's geometry does not.
2. **No signal, learned or geometric, currently clears the break-even bar.** The closest is the
   *geometric* motion residual at 1 % deletion: 57.3 % precision against 59.0 % required,
   `ΔJ = −0.0002` — exactly neutral. The learned probability cannot even be evaluated below 4.15 %
   deletion, because its low tail is a spike of "no score at all", and the rest of it is
   compressed into `[0.5, 1.0]` by the candidate threshold with its absolute level unconstrained
   by the loss (T2a).

> **This is the thesis, reduced to one falsifiable number: the abstention rule must reach
> 59.0 % deletion precision (6bba) / 53.1 % (44b6). Today's best is 57.3 %, from geometry.
> The learned score has the better AUC and the worse operating point — which is exactly the
> symptom of an uncalibrated share rather than a confidence.** E1 + E2 exist to convert the
> 0.701 AUC into a usable tail.

**Sensitivity — what a working abstention rule is worth** [INFERENCE, arithmetic on the measured
counts above]:

| removes … of the FP mass | at precision | Δ edge Jaccard |
|---|---:|---:|
| 30 % | 70 % | +0.010 |
| 50 % | 70 % | +0.017 |
| 50 % | 90 % | +0.038 |
| 80 % | 90 % | +0.063 |
| 100 % | 100 % (oracle) | +0.096 |

So the honest expected value of the abstention lane on 6bba is **+0.01 to +0.04**, with +0.09 as
an unreachable oracle bound — comfortably above the ±0.003 fold-0 noise floor, but only if the
precision bar is cleared.

---

## 4. The patch set

All patches live in **`scripts/kaggle_edits/h1r_edge_loss_patch.py`** (created this session,
inert). It **layers on top of** `scripts/kaggle_edits/h1r_trainer_patch.py` rather than
duplicating it: that file's P3 already turns `weight[div_rows] = 1.0` into `_H1R_DIV_WEIGHT`, and
its P2 already returns detection precision; both are reused. Applying both to a scratch copy of
the vendored trainer succeeds with exact counts (13 + 9 patches, compile-checked), and the
inference companion applies to a copy of `predict_unet_transformer.py` **[MEASURED]**.

Coverage check against the sibling patch file:

| item | already in `h1r_trainer_patch.py`? |
|---|---|
| (b) real division upweight | **YES** — P3, `H1R_DIV_WEIGHT` (default 3.0). Reused, not duplicated |
| precision-aware selection | **YES** — P2. Extended by E5, not replaced |
| AMP / GradScaler / cosine LR / last-checkpoint | YES — P1, P5, P6. Untouched |
| (a) background term | no |
| (c) auxiliary sigmoid | no |
| division masking that actually masks | no |
| association-aware selection metric | no |
| matching inference normalisation | no |

### E1 — background / no-parent term (patch (a))

Replaces `:63`. Trackastra's `Ã_ij = exp(Â_ij) / (1 + Σ_{i'} exp(Â_i'j))`
**[DOCUMENTED, arXiv:2405.15700 §3.2]**, implemented by concatenating a constant zero logit row
before the softmax:

```python
if _H1R_BG_TERM:
    _bg = torch.zeros(1, logits.shape[1], device=logits.device, dtype=logits.dtype)
    probs = torch.softmax(torch.cat([logits, _bg], dim=0), dim=0)[:-1]
else:
    probs = torch.softmax(logits, dim=0)
probs = probs.clamp(1e-7, 1.0 - 1e-7)     # F.binary_cross_entropy is inf at exactly 0/1
```

*Predicted direction*: FP-suppressible edges acquire a low absolute score; ranking essentially
unchanged; `edge_prob` becomes thresholdable.
*Falsification*: on a held-out Zebrahub embryo, if the AUC of the emitted score for
TP-versus-FP among linker-selected edges does not exceed the baseline's, the term bought nothing.
Self-tests T1a/T1b/T2a/T2b/T2c pin the semantics. Env `H1R_BG_TERM` (default 1).

> **Prerequisite, not optional.** E1 changes the meaning of the exported probability, so
> `apply_h1r_edge_loss_inference_patch` (`H1R_EDGE_PROB=softmax_bg`) must ship with it or the
> deployed threshold and the `−0.75·prob` bonus silently change scale.

### E2 — auxiliary sigmoid BCE at λ = 1e-2 (patch (c))

```python
if _H1R_AUX_SIGMOID > 0.0:
    loss = loss + _H1R_AUX_SIGMOID * F.binary_cross_entropy_with_logits(
        logits, target, reduction="none")
```

This is the **absolute, candidate-count-invariant** per-edge score the abstention rule needs
(self-test T4). Export it with `H1R_EDGE_PROB=sigmoid`. Env `H1R_AUX_SIGMOID` (default 0.01,
Trackastra's value).
*Predicted direction*: negligible effect on the softmax term at λ=1e-2 (that is the point — it is
a free extra channel); the win is entirely downstream.
*Falsification*: same AUC/abstention test; if the sigmoid channel's deletion precision at 2 %
deletion does not exceed **59.0 %** (6bba) / **53.1 %** (44b6), the channel is dead weight.

### E3 — real per-row weights, and masking that actually masks (patch (b))

```python
div_rows = target.sum(dim=1) > 1
weight = torch.ones_like(loss)
weight[active_rows] = _H1R_CONT_WEIGHT      # new knob, default 1.0 == today
weight[div_rows]    = _H1R_DIV_WEIGHT       # from h1r_trainer_patch P3

if _H1R_DIV_MASK_COLS and bool(div_rows.any()):
    div_cols = target[div_rows].sum(dim=0) > 0
    mask = mask & ~div_cols.unsqueeze(0)
    if not mask.any():
        return logits.sum() * 0.0
```

**λ_div, reconciled into a two-stage schedule.** The sibling report argues λ_div DOWN for
Ultrack-derived labels (division quality BC(i)≈0.47) and UP for clean competition GT. Both are
right, and both need re-calibrating against our measured division rate:

- **Stage-N (Zebrahub / Ultrack labels): `H1R_DIV_WEIGHT=0` *and* `H1R_DIV_MASK_COLS=1`.** The
  weight alone is not a mask (§1.2, T3b). Motivation is stronger than BC(i) alone: our zh001r
  sidecar carries 65,741 division-daughter edges among 1,258,182 association edges (**5.2 %**),
  versus 302 of 128,883 in competition GT (**0.23 %**) — a **22× discrepancy** (45× if the
  sidecar counts one edge per event; the convention is **[UNVERIFIED]**). Whatever the cause,
  training divisions on that distribution and deploying on ours is a domain shift, not a transfer.
- **Stage-C (competition GT): Trackastra's λ_div=10 is far weaker here than in its home
  regime.** Division rows are 151 of 128,735 annotated source rows (**0.117 %**), so at weight 11
  they carry `11×151/(11×151+128,584)` = **1.28 % of the annotated-row loss mass**. Reaching 10 %
  needs weight ≈ **95** **[INFERENCE, arithmetic from M2]**.
- **But do not spend the sweep yet.** §0.4: the head emits zero divisions and the linker is 1-to-1,
  so λ_div's return on the deployed path is **0** until a division-capable inference lands, and
  `division_lane_2026-08-18.md` shows the binding constraint there is fork *proposal geometry*
  (`wrapper.py:124`, sister gate 7.2 µm below the GT median), not ranking. Set 11 as the
  stage-C constant; sweep toward 95 only after divisions can be emitted.

### E4 — focal exponent knob

`H1R_FOCAL_GAMMA` (default 2.0 = today; 0 disables). Free-rider arm only (§1.4).

### E5 — association-aware checkpoint selection (patch (d), new)

`_evaluate_pair` additionally returns top-1 parent accuracy over supervised columns; `evaluate`
threads it through; `H1R_SELECT=link_f1` selects on `link_top1 × detection_F1`. Motivated by D5's
measured degeneracy (0.9973 accuracy on random logits). Self-tests T6a/T6b.

### E6 — inference normalisation companion (`apply_h1r_edge_loss_inference_patch`)

Patches `predict_unet_transformer.py:454-458` with an `H1R_EDGE_PROB ∈
{softmax, softmax_bg, sigmoid}` selector. **Must ship with E1.**

### E7 — the precondition nobody has done yet (specified, NOT implemented here)

The head's full n×n score matrix is discarded: `learned_edge_probs` is built only from edges the
greedy pre-selector already emitted (`wrapper.py:1146-1157`), and that pre-selector emits exactly
one child per source (§0.4). So the linker sees a learned score on **~1 candidate pair per source
out of ~300 in gate**, and 95.3 % of the time it is the pair geometry would have chosen anyway
(§0.2). **Until the linker receives a score for the pairs it is actually deciding between — or an
abstention score for the pair it selects — no loss change can move the score.** This is an
inference-side change in the sibling agent's lane; I specify it and do not implement it.

---

## 5. What each fix can plausibly recover

Headroom being divided: **+0.081 (44b6) / +0.150 (6bba)** on `adj_edge_jaccard`, of which §3
measures **76 % / 59 % to be false positives** and **24 % / 41 % missing links**.

| lever | mechanism | plausible recovery | basis |
|---|---|---|---|
| **Abstention** (E1+E2+E6+E7) | calibrated absolute score → delete links below threshold | oracle bound **+0.062 / +0.092**; **realistic +0.010 to +0.040** (6bba), conditional on lifting deletion precision from today's 57.3 % to >59.0 % | §3, §3.2, §3.3 **[MEASURED bound + INFERENCE sensitivity]** |
| Better ranking (E1..E5 without E7) | head nominates a better single parent | **≈ +0.000** | §0.2: bonus 0→0.75 moves 1 edge in 9,020; head/geometry already agree 95.3 % **[MEASURED]** |
| λ_div (E3, stage-C) | more division supervision | **+0.000 on edge Jaccard** (151/128,883 = 0.117 %); up to +0.1·Δ`division_jaccard` on SCORE, but **0 today** — no forks are emitted | §0.4 **[MEASURED]** |
| E5 selection fix | stops picking checkpoints on a near-constant | not a score lever; a **variance** lever — it makes every arm below interpretable | D5 **[MEASURED]** |
| Model capacity / data (H1 retrain) | — | not attributable from this session | **[UNVERIFIED]** |
| Detector (unreachable FN) | 364 / 16,002 GT edges have an endpoint we never detect | +0.032 / +0.055 to lift the ceiling itself | §3 **[MEASURED]**, out of scope here |

### 5.1 Honest attribution of the headroom

| bucket | 44b6 | 6bba |
|---|---:|---:|
| loss structure, **via abstention** (oracle bound / realistic) | ≤ +0.062 / ~+0.007–0.027 | ≤ +0.092 / **+0.010–0.040** |
| loss structure, via ranking | ~0 | ~0 |
| inference assignment (Hungarian 1-1, no abstention, discarded score matrix) | **this is the same bucket** — the loss supplies the signal, inference must consume it | |
| detector (unreachable endpoints) | +0.017 (ceiling 0.9816 vs 0.9497) | +0.055 (0.8533 vs 0.7984) |

> The honest statement is that **loss structure and inference assignment are not separable
> levers**: the abstention capability must be *created* in the loss (E1/E2) and *used* in the
> linker (E6/E7). Doing either alone returns approximately zero. That is the single most
> important planning consequence of this report.

### 5.2 If the sibling agent finds the linker cannot emit 1→2 at all

I have measured the pre-wrapper half of that question and the answer is stronger than expected:
the **learned head** already emits zero divisions (§0.4), *before* the Hungarian removes the
possibility a second time. So:

- **Nothing changes in my ranking** — λ_div was already ranked last for the deployed path.
- The correct sequencing becomes explicit: **(1)** make the head able to emit a second child at
  all (which requires the greedy pre-selector, `predict_unet_transformer.py:470-488`, to admit
  `max_children_per_node = 2` in practice — measured: it never does); **(2)** give the linker a
  division-capable assignment or keep `add_safe_divisions_postlink` but feed it learned scores
  (today it uses pure geometry, `wrapper.py:831`); **(3)** only then does λ_div have a route.
  `division_lane_2026-08-18.md` shows step (2)'s gates are the binding constraint.
- Nothing in the E1/E2/E5/E6 abstention lane depends on the division answer.

---

## 6. Ranking by expected value × cheapness

| rank | patch | GPU-h | expected value | why here |
|---|---|---:|---|---|
| **1** | **E5** association-aware selection | **0** (config) | high, indirect | Without it, every arm below is selected on a metric measured at 0.9973 for a random model. Fix first or the ablations are uninterpretable |
| **2** | **E1 + E6** background term + matching inference | ~1.5–3 (one stage-C fine-tune arm) | the only measured route to the +0.062/+0.092 bucket | §0.1, §3.3 |
| **3** | **E2** auxiliary sigmoid | **+0** (rides arm 2) | supplies the actual abstention channel; λ=1e-2 makes it nearly free | **[DOCUMENTED]** |
| **4** | **E7** deliver scores/abstention to the linker | **0 GPU-h**, CPU replay on existing pregraph parquets | multiplier — without it ranks 2–3 return zero | §0.2, §4/E7 |
| **5** | E3 `H1R_DIV_MASK_COLS` for stage-N | +0 (config on the Zebrahub arm) | correctness, not points | T3b |
| **6** | E4 focal γ | +0 (free-rider) | diagnostic | §1.4 |
| **7** | E3 λ_div sweep (11 → 95) | 3–6 | **0 today** | §0.4 — park until inference can emit forks |

---

## 7. Three experiments in ~10 Colab GPU-hours

GPU-hour anchors taken from `h1_execution_spec_2026-08-18.md` §5 (edge half: 60 crops × 39
windows ≈ 146 steps/epoch, 20 epochs ≈ 2,900 steps, ~1.5–3 GPU-h at ~2 s/step) **[DOCUMENTED]**.

**Experiment 0 — CPU, 0 GPU-h, already half-run; finish it before booking any GPU.**
§3.3 is the fold-1 / 24-crop instance. Extend it to both full folds, and add the composite
signals that cost nothing: `edge_prob` combined with the motion residual, and the *margin*
between the head's top-1 and top-2 parents per column (available from the same forward pass but
not currently exported). **Kill criterion:** if no combination of *existing* signals reaches
59.0 % (6bba) / 53.1 % (44b6) deletion precision at ≥2 % deletion, then abstention is not
reachable without retraining — which is the case for Experiment A, not against it. If some
existing combination *already* clears it, take that gain first: it costs zero GPU hours and
zero training, and it re-baselines everything below.

**Experiment A — background term + auxiliary sigmoid, on/off. ~4 GPU-h.**
Two arms, equal steps, identical seed, `H1R_SELECT=link_f1` in both:
`A1 = H1R_BG_TERM=0 H1R_AUX_SIGMOID=0` (today's loss, with E5 selection) and
`A2 = H1R_BG_TERM=1 H1R_AUX_SIGMOID=0.01`.
**Primary metric — deliberately not a competition score** (LOEO does not transfer, +0.0144/+0.0090
→ +0.000): **link top-1 parent accuracy and TP/FP-discrimination AUC on a held-out Zebrahub
embryo**, plus the deletion-precision curve from Experiment 0 recomputed on A2's probabilities.
**Falsification, with the number from §3.3:** A2 must lift TP/FP AUC above 0.701 *and* reach
**>59.0 % deletion precision at 2 % deletion** (A1's baseline: 49.6 % at 4.15 %, geometry's best:
57.3 % at 1 %). Miss either and E1+E2 are dead — report and stop.

**Experiment B — abstention operating point, CPU replay of A2's output. ~0 GPU-h + 1 Kaggle
inference kernel.** Sweep the deletion threshold on A2's exported score through the verified
linker replay; report `Δ` edge Jaccard on both folds separately, with the ±0.003 fold-0 noise
floor stated. **Falsification:** best achievable `Δ` < +0.006 pooled ⇒ the lane does not clear
the noise floor.

**Experiment C — stage-N division masking, correctness arm. ~3 GPU-h.**
`C1 = H1R_DIV_WEIGHT=0` alone versus `C2 = H1R_DIV_WEIGHT=0 H1R_DIV_MASK_COLS=1`, on Zebrahub.
**Falsification:** if held-out continuation link accuracy is identical between C1 and C2, the
column-masking trap does not matter in practice and E3's extra knob can be dropped. Runs only if
Experiment A survives; otherwise park with the whole edge-retrain lane.

Total: **~7 GPU-h**, leaving ~3 h of the 10 as headroom for a re-run of whichever arm the noise
floor makes ambiguous. Not in this budget, and explicitly deferred: window size 2→3 (the
sibling's R6), λ_div sweeps, and any competition-score-based gating.

---

## 8. Files created this session

All inert — nothing pushed, launched, submitted or committed.

| file | what |
|---|---|
| `scripts/kaggle_edits/h1r_edge_loss_patch.py` | E0–E6. 9 exact-string patches to a **copy** of the h1r-patched trainer + 1 to a copy of `predict_unet_transformer.py`; `--self-test` runs 12 numeric semantics checks (**12/12 pass**) |
| `research/06-knowledge-system/internal-reports/edge_loss_structure_2026-08-18.md` | this report |

Nothing under `vendor/` was modified. `.claude/settings.json` untouched.

Scratch instruments (session scratchpad, outside Git — reproduce or re-create as needed):
`gt_struct.py` (GT transition-matrix census), `els_candidate_recall.py`,
`els_final_recall.py`, `els_tpfp.py` (scorer-rule TP/FP decomposition),
`els_bonus_ablation.py` (verified `motion_relink_edges` port + learned-bonus sweep),
`els_abstention.py` (abstention sweep). **`els_bonus_ablation.py`'s port is the reusable asset**
— it replays the deployed linker on CPU with zero GPU and was validated `sym_diff = 0` against
the vendored implementation.

---

## 9. What I could not verify

- The number of detected tokens per frame **during training** (`det_threshold=0.3` on raw logits)
  versus at deployment (sigmoid > 0.96875). One instrumented training step settles it; it
  determines how large the train/deploy softmax-scale mismatch actually is.
- Whether the zh001r sidecar's 65,741 "division-daughter" edges count one or two edges per event
  — changes the Zebrahub/competition division-rate ratio between 22× and 45×.
- Whether an abstention rule built on a *retrained* score clears the break-even precision. This is
  the whole bet, and Experiment 0/A is designed to settle it before any GPU is booked.
- Trackastra's optimizer/LR/epochs (carried over unresolved from
  `edge_training_frontier_2026-08-18.md` §9).
- The exact interaction between E1's rescaled `edge_prob` and the deployed harmonic-fusion edit
  (`scripts/kaggle_specs/p3_armb.json`, which re-normalises with its own `softmax(…, dim=1)`).
  The two must be measured together; `p3_armb.json`'s own `interaction_warning` applies verbatim.
- Fold-0's reachable/TP/FP figures come from the *subvoxel* arm and fold-0's head-candidate
  recall from the *D1-pilot* arm; both are close cousins of the deployed configuration but not
  identical. The 6bba figures (128 crops, 85 % of the objective mass) are the load-bearing ones.
