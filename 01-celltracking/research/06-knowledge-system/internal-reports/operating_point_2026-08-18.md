# Operating point: grid-centre shift, break-even threshold, duplicate topology, N_est — 2026-08-18

Labels: **[MEASURED]** = run this session against the official scorer / real data.
**[CODE]** = read off source, `file:line` given. **[INFERENCE]** = derived, assumptions stated.
All measurements in this report are complete; nothing is outstanding.

Substrate for every measurement below: the two LOEO submission CSVs from the (killed) sub-voxel
arm — `c:/temp/subvoxel_f0/loeo_split0_strict.csv.gz` (71 `44b6` crops) and
`c:/temp/subvoxel_f1/loeo_split1_strict.csv.gz` (128 `6bba` crops). Scorer:
`scripts/core/score_loeo_submission.py` (official `tracking_cellmot.metrics.summarise`).
Paired record baselines: **0.9033** (fold 0) / **0.7042** (fold 1),
`research/06-knowledge-system/experimental-records.md:507-508`.

---

## 0. HEADLINE — the grid-centre shift is REFUTED at +2, and the mechanism claim is false

### 0.1 Measured deltas **[MEASURED]**

Node rows only; `y` and `x` shifted by +k level-0 integer voxels, clipped to [0,255]; `z` untouched.
Edge rows carry no coordinates, so topology is bit-identical across all arms — this isolates
localisation and nothing else.

Every baseline below is a **re-score of the unshifted CSV through the identical code path**, not
the record number, so each Δ is exactly paired. The fold-0 re-score reproduces the record's 0.9033
to **+0.000018** — the substrate and scorer are confirmed identical to the sub-voxel kill run.

| arm | fold | crops | score | **paired Δ** | adj_edge_jaccard | edge_jaccard | node_recall |
|---|---|---|---|---|---|---|---|
| **baseline (re-scored)** | 0 | 71 | **0.903318** | — (record 0.9033, +1.8e-5) | 0.901802 | 0.883828 | 0.9842 |
| **+1 y/x** | 0 | 71 | 0.904608 | **+0.001291** | 0.903058 | 0.885051 | 0.9842 |
| **+2 y/x** | 0 | 71 | 0.901691 | **−0.001626** | 0.900141 | 0.882198 | 0.9843 |
| **+3 y/x** | 0 | 71 | 0.896801 | **−0.006517** | 0.895250 | 0.877411 | 0.9838 |
| **baseline (re-scored)** | 1 | 128 | **0.704231** | — (record 0.7042, +3.1e-5) | 0.703756 | 0.698419 | 0.8656 |
| **+1 y/x** | 1 | 128 | 0.703434 | **−0.000797** | 0.702961 | 0.697615 | 0.8656 |
| **+2 y/x** | 1 | 128 | 0.704116 | **−0.000115** | 0.703638 | 0.698281 | 0.8654 |

Both baselines reproduce the record to <4e-5, so substrate and scorer are confirmed identical to
the sub-voxel kill run and every Δ above is exactly paired.

**The one arm that looked promising is refuted by the second embryo: +1 is +0.00129 on fold 0 but
−0.00080 on fold 1.** The sign flips across embryos, which is the signature of noise, not of a
systematic localisation bias — a true constant coordinate-convention error would move both folds
the same way.

Division counts are unchanged by the shift on fold 0 (TP/FP/FN = 2/106/24 at baseline, 2/103/24 at
every k>0) and near-unchanged on fold 1 — as expected, since division scoring re-matches the same
topology.

### 0.1b The deltas are inside the noise floor **[MEASURED]**

Crop-level bootstrap, 4,000 resamples over each fold's crops, recomputing the full weighted pooled
statistic each time:

| fold | arm | observed Δ(adjJ) | 95 % CI | crops up / down / unchanged | Δ edge TP | verdict |
|---|---|---|---|---|---|---|
| 0 | +1 | +0.00126 | **[−0.00158, +0.00388]** | 19 / 17 / **35** | **+10** | **NOT SIGNIFICANT** |
| 0 | +2 | −0.00166 | [−0.00535, +0.00160] | 20 / 25 / 26 | −9 | not significant |
| 0 | +3 | −0.00655 | [−0.01079, −0.00259] | 20 / 32 / 19 | −51 | **significant, negative** |
| 1 | +1 | −0.00080 | [−0.00328, +0.00165] | 60 / 64 / 4 | −35 | **NOT SIGNIFICANT** |
| 1 | +2 | −0.00012 | [−0.00404, +0.00378] | 62 / 62 / 4 | −8 | not significant |

**The apparent fold-0 +1 "gain" is noise, and fold 1 confirms it.** On fold 0 the median crop does
not move at all, the fold gains only **+10 edge TPs** across 1.77 M predicted edges, and the sign
split is 19 up / 17 down. The pooled number is carried by a handful of tiny crops:
`44b6_2f31fc2f` swings **+0.0656** in its own `adj_edge_jaccard` off a **+4 edge-TP** change,
because its `w_i` is only 119. On fold 1 the same arm *loses* 35 edge TPs with a dead-even 60/64
crop split.

**Root cause, and it generalises far beyond this lever: fold 0's entire scoring weight is
`W = Σ w_i = 21,210` edge events — versus 123,413 on fold 1, a 5.8× difference in statistical
power.** `44b6` is only 0.771 % annotated (forensics §1.4: 19,826 GT edges over 71 crops), so
per-crop `w_i` runs 82–411, and a four-edge change in one crop moves that crop's Jaccard by
several percent. Fold 0 is by far the weaker instrument, and it is the one that has been
generating our most exciting deltas.

### 0.1c Verdict

**The grid-centre shift lever is CLOSED in every arm tested.** +2 (the brief's specific claim) is
negative on both folds; +3 is significantly negative, fixing the gradient sign away from the
claimed direction; +1 is indistinguishable from zero and **flips sign between embryos**. Nothing
here is submission-ready, and the code-level refutation in §0.2 explains why no constant shift
could have paid.

**Methodological consequence worth carrying forward (VALIDATION CRISIS).** Crop-resampling noise
floors, measured, at 95 %: **fold 0 ≈ ±0.003, fold 1 ≈ ±0.004**. Any LOEO delta below ≈0.004 is
not measurable on this substrate at all, regardless of mechanism — and fold 0, with 5.8× less
scoring weight, is the *weaker* instrument despite being the fold that has produced our most
exciting numbers.

This does *not* by itself explain arm B: its +0.0144 sat above this floor and still returned
0.000 on the LB, so arm B's failure is a substrate/population mismatch, not merely crop noise.
**The two failure modes are independent, and a lever must now clear both**: it needs a paired
delta above the noise floor *and* a mechanism that changes something the LB population shares —
which, per the churn evidence (3–10 % of edges rewritten for 0.000), edge reassignment does not.

### 0.2 Why the mechanism claim was wrong **[CODE]**

The brief inherited from `retrain_recipes_2026-08-17` the claim that
`predict_unet_transformer.py:495` (`coords[:, 1:] *= ds_arr`) omits a grid-centre term, producing
a constant +0.5-downsampled-voxel (= +2 level-0 voxel = 0.8125 µm) deficit in y/x.

That argument requires the downsample to be **block-averaging**, where downsampled voxel `i`
represents the level-0 block `[4i, 4i+3]` whose centre is `4i + 1.5`. It is not. The loader is
pure striding:

```python
raw = zarr_arr[t, ::dz, ::dy, ::dx].astype(np.float32)
```
`vendor/kaggle-cell-tracking/scripts/predict_unet_transformer.py:215`, with
`downsample = [1, 4, 4]` (`:157`).

Under striding, downsampled index `i` **is** level-0 index `4i` — the intensity at that voxel is
literally the level-0 sample there, not a block aggregate. So `coords *= 4` already returns the
exact level-0 coordinate of the sampled lattice point, and no centre correction is owed. The
estimator is coarse but **unbiased**: quantisation error is symmetric about the sampled lattice,
uniform on ±2 level-0 voxels per axis, RMS `4/sqrt(12) = 1.155` voxels = 0.469 µm per axis.
`dz = 1` also explains, correctly, why z carries zero quantisation error.

This is a code-level refutation independent of the score measurement, and the two agree.

**On the fold-0 +1 residual.** With fold 1 now measured at −0.00080 for the same arm, the honest
reading is that there is **no residual effect to explain** — the sign flip plus both CIs spanning
zero (§0.1b) is what noise looks like. The candidate mechanisms below were worked through before
fold 1 returned; they are recorded because one of them is now refuted at code level and that
refutation is reusable, not because any surviving effect needs them **[CODE]**:

- ~~(a) the `F.interpolate(..., align_corners=False)` rescaling at `:217-221`, which would
  introduce a genuine half-pixel resampling offset~~ — **REFUTED. That branch never fires.** It is
  guarded by `if list(frame.shape) != target_shape`, and `target_shape` is
  `ds.image_shape[1:]` (`:328`) where `open_dataset` computes
  `ds_shape = raw_shape[:1] + tuple(-(-s // d) ...)` = `ceil(s/d)` per axis
  (`src/tracking_cellmot/io.py:60`). That is exactly the shape strided indexing yields, so the
  condition is always false and no resampling occurs.
- (b) an annotation-side centroid convention in the GT (e.g. centroids computed on a full-res
  segmentation, which sit at the *mass* centre of a cell rather than at its brightest strided
  lattice point);
- (c) a real learned bias in the detector head.

(b) and (c) would each predict a *sub-voxel* offset under one level-0 voxel — but they also
predict the **same sign on both embryos**, which is exactly what the measurement rules out. They
are not worth chasing with further CSV shifts; if anything ever revisits localisation, it needs the
detector's own probability maps (the §1.2 export), not another coordinate translation.

### 0.3 Would this transfer where arm B didn't?

The measurement makes the question moot, but the class distinction is worth recording because it
is the criterion the remaining levers are judged on. Arm B churned 3–10 % of edges and moved the
LB by 0.000 — a pure graph-edit lever reshuffling structure the metric is insensitive to. A
coordinate shift is a **different class**: it does not touch topology
at all (edge rows are identical), it moves node positions across the scorer's hard 7 µm matching
radius (`metrics.py:276`), so it changes *which nodes match*, i.e. the node/edge TP set directly.
That is the class of lever that should transfer. It simply happens to be worth ≈0 here, because
the bias it was supposed to correct does not exist.

---

## 1. Break-even detection confidence — the 0.50 claim, re-derived **[CODE] + [INFERENCE]**

### 1.1 The arithmetic, from source

Objective (`vendor/kaggle-cell-tracking/src/tracking_cellmot/metrics.py`):

| term | value | line |
|---|---|---|
| `ADJUSTMENT_ALPHA` α | 0.1 | `metrics.py:30` |
| `SCORE_DIVISION_WEIGHT` | 0.1 | `metrics.py:34` |
| `total_node_ratio = (num_pred_nodes − n_total)/n_total` | | `metrics.py:440` |
| `adj_edge_jaccard = max(0, J·(1 − α·ratio))` | | `metrics.py:447-449` |
| `edge_fp = edge_valid_pred − edge_tp` | | `metrics.py:316` |
| `edge_fn = gt_num_edges − edge_tp` | | `metrics.py:317` |
| pooled weights `w_i = TP+FP+FN` | | `metrics.py:499` |
| match radius 7.0 µm, physical | | `metrics.py:276` |

Adding one predicted node with probability `p` of being a real (annotated) cell:

- with prob `p·d_ann` it lands on annotated territory and buys **2 edges** (in + out, interior of a
  track) → gain `2·m_i/W`
- with prob `(1−p)·d_ann` it lands on annotated territory as a false detection whose links become
  FP → cost `2·adjJ/W` (each FP edge costs `adjJ/W`, `metrics.py:316` feeding `:499`)
- unconditionally it costs node budget → `0.1·TP_i/(E_i·W)` (differentiate `:447` w.r.t.
  `num_pred_nodes`)

Setting `E[Δ] = 0`:

```
p·d·2m/W  −  (1−p)·d·2·adjJ/W  −  0.1·TP_i/(E_i·W)  =  0
p* = [ adjJ + 0.05·(TP_i/E_i)/d ] / (m + adjJ)
```

**The node-budget term is negligible** (`TP_i/E_i ≈ 0.024`, `d ≈ 0.028` → `0.05·0.024/0.028 ≈
0.043`), so `p* ≈ adjJ/(m + adjJ)`. With `m ≈ 1`, `p*` is set almost entirely by `adjJ`:

| adjJ | m | p* |
|---|---|---|
| 0.915 (LB) | 1.000 | **0.500** |
| 0.9018 (fold 0 measured) | 1.020 | 0.469 |
| 0.7038 (fold 1 measured) | 1.008 | 0.412 |

**The ≈0.50 claim reproduces and is confirmed** — and it is robust, because `p* = adjJ/(1+adjJ)`
is flat in every other quantity. Deployed `BIOHUB_DET_THRESHOLD = 0.96875`
(`scripts/d1/d1f_probe.py:133`, set in the notebook env cell) is far above what the metric wants.

**[INFERENCE] The caveat that makes this not a free win.** The derivation prices a *marginal
independent* detection. It assumes the added node either buys 2 edges or costs 2 FP edges. The
duplicate census in §2 shows that is not the regime we are in: 20 % of our nodes already sit
within 7 µm of another node in the same frame, and lowering the threshold adds nodes
preferentially in exactly those contested neighbourhoods, where the added node does not buy new
edges — it competes for a GT node already matched. So `p* ≈ 0.5` is a valid *upper bound on how
conservative the budget term requires you to be*, and it correctly kills "the node budget is why
we sit at 0.97". It is not a prediction that 0.5 scores better. That is what the sweep is for.

### 1.2 Threshold-superset export — one GPU run buys the whole curve

The peak set at threshold `T` is a strict subset of the set at `T' < T`: peaks are selected by
`is_peak = (logits == pooled) & (torch.sigmoid(logits) > det_threshold)`
(`predict_unet_transformer.py:285`). The `logits == pooled` local-max test is **independent of the
threshold**; only the `sigmoid > T` mask varies, and it is monotone in `T`. So exporting every
peak at `T = 0.5` **with its sigmoid score** makes the entire threshold curve `[0.5, 1.0]` a pure
CPU replay: filter the peak table, rebuild the graph, re-link, re-score. No further GPU.

Files written this session (**inert until pushed — NOT pushed**):

| file | role |
|---|---|
| `scripts/kaggle_specs/p4_detsweep_export_f0.json` | fold 0 spec, derived from `p3_base_loeo_f0.json` (paired LOEO baseline, arm B off) |
| `scripts/kaggle_specs/p4_detsweep_export_f1.json` | fold 1 spec, same derivation |
| `scripts/kaggle_edits/detpeak_export.py` | the patch: records superset, returns pipeline subset |

**The pipeline threshold is deliberately NOT overridden.** Detecting at 0.5 would change the
graph (destroying comparability with the paired baseline) *and* blow up cost, since edge
prediction is `n_src × n_tgt` per frame pair (`predict_unet_transformer.py:449-465`) — quadratic
in peaks per frame. Instead the patch computes the local-max mask once, records everything above
`BIOHUB_DETPEAK_EXPORT_T=0.5` to the export sink, then returns only peaks above the unchanged
`BIOHUB_DET_THRESHOLD=0.96875`. The built graph is bit-identical to `p3_base`, runtime is
unchanged, and the export is purely observational.

Anchors validated against the base notebook this session: `available_gpu_count =
_torch.cuda.device_count()` occurs exactly once, at column 0, in cell 5 (where `_ps` is already
bound); the `_LOEO_KEEP` patch occurs exactly once in `loeo_export.py` and is applied as edit 19,
after that file is injected as edit 17. `detpeak_export.py` parses clean.

Export payload per peak: `t` (int16) + `zyx` (int16×3) + `logit` (float32) = 12 bytes, one
compressed `.npz` per crop, in **downsampled** grid coordinates (multiply by `[1,4,4]` for
level-0).
**Size estimate:** at `T = 0.96875` fold 0 emits 1.84 M nodes / 71 crops. The detector's sigmoid
mass between 0.5 and 0.96875 is the unknown; assuming a 3–6× superset (the local-max constraint
caps growth hard — peaks must still be strict local maxima under a 3 µm pool kernel,
`predict_unet_transformer.py:283-285`), expect **5.5–11 M peaks ≈ 80–155 MB** raw,
**≈30–60 MB gzipped** per fold. That is within the LOEO artifact budget already used
(33 MB gzipped CSVs today) and well inside the `/kaggle/working` fetch limits the
`loeo_export.py` cell was written to respect.

**Falsification test:** replay the exported table at `T ∈ {0.5, 0.7, 0.85, 0.9, 0.94, 0.96875}`
on both folds. If score is monotone decreasing as `T` falls — i.e. the peak still sits at
0.96875 with the node budget provably not binding (§1.1) — then the detector's score is
uncalibrated and the extra peaks are duplicates, not new cells; the lever dies and calibration,
not thresholding, is the only route. **Cost:** 1 GPU run per fold, then unlimited CPU replay.

---

## 2. Duplicate topology — "free duplicates" is real but the exposure is small **[MEASURED]**

Duplicate = predicted node with another predicted node in the **same frame** within the scorer's
own 7 µm physical radius, scale `(1.625, 0.40625, 0.40625)` (`metrics.py:276`, `io.py:13`).
Full census over both folds, all 199 crops.

| quantity | fold 0 (44b6, 71 crops) | fold 1 (6bba, 128 crops) |
|---|---|---|
| predicted nodes | 1,842,800 | 1,957,978 |
| predicted edges | 1,771,878 | 1,877,129 |
| duplicate **pairs** (<7 µm, same frame) | 205,618 | 219,162 |
| **nodes involved in ≥1 duplicate pair** | 370,197 (**20.1 %**) | 389,330 (**19.9 %**) |
| — both members isolated (deg 0) | **0** | **0** |
| — one member isolated | **0** | **0** |
| — **parallel disjoint** (different components) | **187,797 (91.3 %)** | **198,830 (90.7 %)** |
| — shared neighbourhood (linked / common neighbour) | 5,210 (2.5 %) | 6,211 (2.8 %) |
| — same component, distant | 12,611 (6.1 %) | 14,121 (6.4 %) |
| nodes with in-degree ≥ 2 (merges) | **0** | **0** |
| nodes with out-degree ≥ 2 (forks) | 5,279 | 6,303 |
| **pairs competing for the same GT node** | **3,839 (1.9 % of pairs)** | **12,783 (5.8 %)** |
| **FP-edge exposure from duplicates** | **5,496 – 7,474** | **19,804 – 24,520** |

Reading, against the forensics claim (`metric_forensics_2026-08-17.md` §1.3, F3):

1. **The "duplicates are free unless linked" refinement is confirmed structurally and is the
   dominant case.** 91 % of duplicate pairs are *parallel disjoint* tracks — separate weakly
   connected components. Per the measured probe table (forensics §1.3, "duplicate as a disjoint
   parallel track 0.4 µm away → 3/0/0"), those cost nothing but node budget. At −0.0067 edge-TP
   per node (forensics row 5), 370 k duplicate nodes on fold 0 amount to ≈ 2,480 edge-TP
   equivalents of budget charge.

   Two corrections to how that number should be read. First, the marginal charge is **not**
   waived by our sitting under `N_est`: `m = 1 − 0.1·(N_pred − N_est)/N_est` is linear, so
   `∂m/∂N_pred = −0.1/N_est` is constant — being under budget raises the *level* of `m` (§3) but
   never makes an extra node free. Second, 2,480 edge-TP equivalents is large relative to
   `W = 21,210` (§0.1b) — **but it is not recoverable by deletion**, because forensics §1.5 prices
   blunt pruning at a 23:1 loss-to-gain ratio: removing a node that was actually carrying two
   edges costs far more than the budget it refunds.

2. **The link-topology exposure is materially smaller than the duplicate count, but it is not
   zero.** Only 1.9 % (fold 0) / 5.8 % (fold 1) of duplicate pairs actually contest the same GT
   node, which is the only way the scorer's per-frame optimal matching (`_ctc_metrics.py:174`,
   `_match_single_frame`) can force one member to lose and turn its incident edges into FPs.
   Bounding the loser's scoreable incident edges gives **5.5–7.5 k FP edges on fold 0** and
   **19.8–24.5 k on fold 1**.

3. **Scale that against the objective.** Fold 1 measured `edge_fp` is far larger than this bound,
   so duplicates explain only a slice of FP mass; on fold 0 the exposure of ~6 k against
   `W ≈ 1.8 M` predicted edges is small. Deleting a false edge is worth `adjJ/W` (forensics row
   2), so perfectly resolving *every* contested duplicate is worth roughly
   `6,500 × 0.90 / W` on fold 0 — order **+0.003**, and that is a hard ceiling assuming the
   winner is always chosen correctly, which no filter can guarantee.

4. **F3 as originally written is already closed.** It proposed "forbid any second in-edge, and any
   second out-edge not certified as a division". We emit **zero** in-degree ≥ 2 nodes on both
   folds — 0 / 3.8 M nodes. The merge half of F3 has nothing to act on. This independently
   reconfirms the 40-crop cache result on the full 199-crop substrate.

**Verdict: the "free duplicates" refinement is materially TRUE on our outputs — it is why NMS
tightening has never paid — but the residual it exposes (≤ +0.003 on fold 0, ceiling) is a
graph-edit lever of exactly the arm-B class.** It reshuffles which of two near-coincident nodes
owns an edge without changing node recall or the detection surface. Given arm B moved 3–10 % of
edges for 0.000 on the LB, **I would not expect this to transfer**, and I do not recommend
spending a submission on it.

---

## 3. N_est operating point **[CODE] + [MEASURED]**

### 3.1 How `N_est` is delivered

`N_est` is **ground-truth metadata, not a pipeline input**: it is read from the GT `.geff` zarr
attributes as `extra["estimated_number_of_nodes"]` —
`src/biotrack/metric.py:37-47` (`estimated_nodes()`), consumed at
`scripts/core/score_loeo_submission.py:143` and inside the scorer at `metrics.py:440` via
`n_total`. Confirmed live in the data:
`data/train/44b6_0113de3b.geff/zarr.json` → `"extra": {"estimated_number_of_nodes": 25755}`.

**This is decisive for the lever's feasibility.** `n_total` comes from the *ground-truth* geff, so
at test time it is not observable — the pipeline cannot read its own crop's `N_est`. Any
"count-quantile operating point" rule must therefore *predict* the target count from the image,
not look it up.

### 3.2 Our measured `N_pred/N_est` distribution **[MEASURED]**

| fold | family | crops | min | q10 | q25 | **median** | q75 | q90 | max | pooled Σ/Σ | crops under 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 44b6 | 71 | 0.264 | 0.417 | 0.574 | **0.864** | 0.984 | 1.096 | 1.172 | **0.7036** | **55 / 71** |
| 1 | 6bba | 128 | 0.537 | 0.741 | 0.851 | **0.915** | 1.000 | 1.082 | 1.646 | **0.9296** | **96 / 128** |

**We systematically UNDER-produce nodes relative to `N_est`, on both embryos**, and dramatically so
on `44b6` (pooled 0.70; the 10th-percentile crop emits 42 % of its estimate).

### 3.3 What a count-quantile rule would change

Because clipping in `metrics.py:447` is one-sided (`max(0.0, …)` only clips from below), a ratio
below 1 yields a multiplier `m = 1 − 0.1·(ratio−1) > 1`. **We are already collecting a node-budget
bonus on both folds, and pushing `N_pred` up toward `N_est` would *reduce* it.**

The bonus must be measured `w_i`-weighted, not from the pooled ratio, because `m` is applied per
crop and then weighted by that crop's *edge* count (`metrics.py:499`) — and our heavier-weighted
crops are markedly less under-budget than the light ones **[MEASURED]**:

| fold | pooled `ΣN_pred/ΣN_est` | `w`-weighted ratio | **`w`-weighted mean `m`** |
|---|---|---|---|
| 0 (44b6) | 0.7036 | 0.7955 | **1.02045** |
| 1 (6bba) | 0.9296 | 0.9208 | **1.00792** |

Counterfactual computed directly from the per-crop rows — force every crop to `N_pred = N_est`
(so `m ≡ 1`) holding each crop's raw `edge_jaccard` fixed:

| fold | pooled adjJ now | with `m ≡ 1` | **Δ** |
|---|---|---|---|
| 0 | 0.90180 | 0.88383 | **−0.01797** |
| 1 | 0.70376 | 0.69842 | **−0.00534** |

So raising counts to hit `N_est` costs ≈ **−0.018 (fold 0) / −0.005 (fold 1)** before any
offsetting edge gain, and only pays if the added nodes actually buy edges. The lever is
**bilaterally negative on its budget term alone.**

So a count-quantile rule that targets `N_pred ≈ N_est` is **net-negative unless the added nodes
convert to edges at better than roughly 1 edge per 74 nodes** on fold 0
(`0.1·TP_i/E_i` vs `2·m·d_ann`). **[INFERENCE]** Combined with §1.1, this reframes the deployed
0.96875 threshold: it is not defended by the node budget (which would happily accept many more
nodes), it is defended by whether extra detections convert to edges — which is precisely what the
§1.2 export measures, and it is a **node-recall / detection-composition** lever, not a graph-edit
one. Of everything in this report, that is the only item I would expect to transfer where arm B
did not.

**Falsification test:** in the §1.2 CPU replay, record `N_pred/N_est` and `edge_tp` jointly at each
threshold. If `edge_tp` is flat while `N_pred` climbs, the extra peaks are duplicates (§2) and both
the threshold lever and the count-quantile lever are dead together. **Cost:** free, rides on the
same replay.

---

## 4. Ranked verdicts

| # | Lever | Class | Status | Cost | Falsification |
|---|---|---|---|---|---|
| L1 | **+2 grid-centre shift** (the brief's claim) | localisation | **DEAD [MEASURED]** −0.00163 fold 0, −0.00008 fold 1; mechanism refuted at `:215` | spent | done |
| L2 | **+1 / +3 y/x shift** | localisation | **DEAD** — +1 **flips sign across embryos** (+0.00129 f0 / −0.00080 f1), neither significant; +3 significantly negative | spent | done |
| L3 | **Threshold superset export + CPU sweep** | **node recall / detection composition** | specs + patch written, **not pushed**; anchors validated | 1 GPU run/fold, then unlimited CPU | score monotone-decreasing as `T` falls ⇒ extra peaks are duplicates ⇒ dead |
| L4 | **Duplicate relink filter** | graph edit (**arm-B class**) | ceiling ≈ +0.003 fold 0; merge half already closed (0 in-deg ≥ 2 on 3.8 M nodes) | CPU | expect no LB transfer — do not spend |
| L5 | **Count-quantile to `N_est`** | budget | **INVERTED** — we under-produce (pooled 0.70 / 0.93); targeting `N_est` *costs* ≈ **−0.018** on fold 0 before any edge gain | — | rides free on L3 replay |

**Recommended next action: L3, and only L3.** It is the single lever here whose mechanism changes
node recall and detection composition rather than edge assignment — the class arm B was not — it
is measurable entirely on CPU after one GPU export per fold, it leaves the deployed graph
bit-identical so the paired comparison is exact, and it settles L5 for free.

**Do not spend on L1/L2 (closed by measurement and by code), and do not spend on L4**: it is a
pure edge-reassignment lever with a ≈+0.003 ceiling, i.e. precisely the profile that churned
3–10 % of arm B's edges for 0.000 on the leaderboard.

**Apply the new bar to L3 before believing it.** A fold-0 LOEO delta must exceed ≈0.004 to clear
crop-resampling noise (§0.1b), *and* it must be a class of lever that survives the substrate
mismatch that killed arm B. L3 is designed to report `edge_tp` and `N_pred/N_est` jointly at every
threshold precisely so the *mechanism* (did recall actually rise?) is auditable rather than
inferred from the composite alone.

---

## Appendix — reproduction

- Shift generation: `<scratchpad>/shift_csv.py` (node rows only, `y`/`x` `+k`, clip [0,255]).
- Scoring: `.venv/Scripts/python.exe scripts/core/score_loeo_submission.py --csv <csv>
  --gt-dir data/train --json-out <json>`; raw results `<scratchpad>/f{0,1}_shift{1,2,3}.json`.
- Duplicate + `N_est` census: `<scratchpad>/dup_nest_audit.py`, results
  `<scratchpad>/dup_f{0,1}.json`. Union-find components, `cKDTree` pair query at 7 µm on
  scaled coordinates, GT territory from `data/train/*.geff` zarr props.
- `N_est` values read from `data/train/<crop>.geff/zarr.json`
  `attributes.geff.extra.estimated_number_of_nodes`.
