# Metric forensics + live public surface — 2026-08-17

Scope: (1) derive the objective term-by-term from the vendored scorer source, with
`file:line` for every claim; (2) refresh the live public surface. Companion to
`competitive_refresh_2026-08-17.md` (public-notebook/discussion sweep) — that report is
**not** repeated here; only deltas and metric-relevant items appear in Part 2.

Vendored scorer = `vendor/kaggle-cell-tracking/` = a clone of
`https://github.com/royerlab/kaggle-cell-tracking-competition.git` at merge `075fc5f`
(includes `aa65e90 updating metric to patch weakly connected component exploit`).
The matcher lives in the installed dependency `tracksdata`
(`.venv/Lib/site-packages/tracksdata/`), also cited by line.

Labels used throughout: **[CODE]** = read off the vendored source; **[MEASURED]** = run in
this session against real data/scorer; **[INFERENCE]** = derived, assumptions stated;
**[UNVERIFIED]** = could not confirm, with what would settle it.

---

## 0. THE SENSITIVITY TABLE (the deliverable)

### 0.1 The objective, reduced

From `metrics.py:498-506` and `metrics.py:519-522`, the run-level score is

```
score            = adj_edge_jaccard + 0.1 * division_jaccard
adj_edge_jaccard = Σ_i w_i · J_i · m_i / Σ_i w_i     w_i = TP_i + FP_i + FN_i
J_i              = TP_i / w_i
m_i              = max(0, 1 − 0.1 · (N_i − E_i)/E_i)
division_jaccard = ΣdTP / (ΣdTP + ΣdFP + ΣdFN)       (micro, pooled across crops)
```

Because `J_i · w_i ≡ TP_i`, this collapses exactly to

> **adj_edge_jaccard = Σ_i (TP_i · m_i) / Σ_i w_i** — a micro-averaged Jaccard whose
> **numerator only** is scaled by each crop's node-budget multiplier.

**[MEASURED]** Verified to 1e-12 against the real `summarise()` on 128 measured crop rows
(`artifacts/kaggle/m1_heldout_scores/*.json`): `summarise()` = 0.6498080428, closed form =
0.6498080428, identical. This makes every gradient below exact rather than approximate.

Notation: `W = Σ_i w_i` (total edge events), `D = dTP+dFP+dFN` (total division events),
`E_i` = crop i's `estimated_number_of_nodes`, `d_ann` = annotation density.

### 0.2 Gradients — exact, then evaluated

All Δscore values scale as `1/W`. `W` for the hidden test is unknown, so the **right-hand
column (edge-TP equivalents) is the invariant, actionable quantity**; the absolute column is
evaluated at our measured 6bba-fold operating point (`W = 128,555`, `adjJ = 0.6498`,
`m̄ = 1.0117`, `D = 746`).

| # | Controllable move | Exact ∂score | Δscore (measured op. pt.) | **In edge-TP equivalents** |
|---|---|---|---|---|
| 1 | Recover 1 missed GT edge (FN→TP) | `m_i / W` | +7.87e-6 | **1.00** (reference unit) |
| 2 | Delete 1 false edge (FP→nothing) | `adjJ / W` | +5.06e-6 | **0.64** (→ 0.92 at deployed adjJ≈0.915) |
| 3 | **Fix 1 mis-link** (FP→TP; clears an FN too) | `(m_i + adjJ) / W` | +1.29e-5 | **1.64** |
| 4 | **Recover 1 missed node** in a track interior (buys 2 edges) | `2·m_i / W` | +1.57e-5 | **2.00** |
| 5 | Emit 1 extra predicted node | `−0.1·TP_i /(E_i·W)` | −5.24e-8 (median) | **−0.0067** (≈ **150 nodes = 1 edge**) |
| 6 | **Recover 1 division (dFN→dTP)** | `0.1 / D` | +1.34e-4 | **+17.0** |
| 7 | Emit 1 spurious division fork on annotated territory | `−0.1·dTP / D²` | −3.3e-6 … −1.4e-4 | **−0.4 now → −17 later** (see note) |
| 8 | Emit any node/edge/fork wholly in **unannotated** territory | `−0.1·TP_i/(E_i·W)` only | −5.2e-8 per node | **−0.0067 per node; edges free** |
| 9 | Emit a `dt≠1` (gap-2 / backward / same-frame) edge | `0` | 0 | **0.00 — dropped, not penalised** |
| 10 | Emit a 3rd+ child from one node | `0` on the edge term | 0 | **0.00 edge-side; truncated arbitrarily** |

**Note on row 7 (the important non-linearity).** The cost of a division FP is `0.1·dTP/D²`,
which *grows* as division precision improves. At our current operating point (dTP=10, D=547)
one extra FP costs 0.4 edge-TPs — almost nothing. At dTP=60 / dFP=60 / dFN=91 (D=211) one
extra FP would cost **17 edge-TPs**. Division precision is a self-reinforcing regime: worth
little at the start, decisive once inside it. This is exactly why local search never escapes.

### 0.3 Where the points actually live (pool sizes)

| Pool | Size | Our current claim | Headroom |
|---|---|---|---|
| Edge term `adj_edge_jaccard` | 0 → ~1.0 (up to 1.1 via node bonus) | ~0.915 on LB **[INFERENCE]** | ~0.085 |
| **Division term `0.1 × divJ`** | **0 → 0.100** | **~0.000–0.002 [MEASURED]** | **~0.098** |
| Node-budget multiplier `m` | ×0 → ×1.1 | ×1.012 (6bba fold) **[MEASURED]** | ±~0.02 realistic |

**The single largest un-claimed pool is the division term, and the gap from our 0.915 to the
0.950 leader (0.035) is smaller than the division pool we currently score ~0 on.**

### 0.4 The decisive measurement

**[MEASURED]** Ran the official scorer (via `biotrack.metric`, parity-verified) over our own
cached predictions `artifacts/kaggle/oof_clean/pred_geffs_split_1/` on the 8 most
division-rich crops (31 GT divisions):

```
edge TP/FP/FN = 7109/1521/1574   micro edge J = 0.6967   adjJ = 0.6916
div  TP/FP/FN = 10/516/21        D = 547                 divJ = 0.0183
```

Division **recall is 32 % (10/31) — that is not the problem. Precision is: 516 FPs, 17× the
number of GT divisions in those crops.** Fork-set decomposition
(`division_metrics._pred_division_fork_sets`, three crops):

| Crop | total pred forks | GT div | `evaluable` (FP source) | `cross_component` | `malformed` | tp_forks |
|---|---|---|---|---|---|---|
| 6bba_debd7bfa | 1116 | 4 | 85 | 3 | 0 | 0 |
| 6bba_969618f6 | 2172 | 3 | 69 | 0 | 0 | 0 |
| 6bba_afb141ff | 558 | 4 | 75 | 0 | 0 | 3 |

**Mechanism, fully pinned: `division_FP ≈ (total forks emitted) × (annotation density)`.**
`cross_component` and `malformed` contribute ~0 for us — we emit **zero** merges (measured: 0
nodes with in-degree ≥ 2 across 40 cached crops / 586k edges). Every FP is an
`evaluable_fork`: a predicted fork that landed on an annotated GT node with GT out-degree ≥ 1
(`division_metrics.py:469-472`).

Cross-check against the ledger: `experimental-records.md:270` records P0-B (our deployment
substrate) as having **305 forks**, and `experimental-records.md:174` records its division
counts as **TP 0 / FP 8 / FN 3**. 305 × 2.8 % ≈ 8.5 ≈ 8. The mechanism predicts the deployed
substrate's FP count to within rounding. **Our deployed pipeline recovers zero divisions.**

---

## PART 1 — METRIC FORENSICS (derived from code)

### 1.1 Constants and the exact objective

| Quantity | Value | `file:line` |
|---|---|---|
| Node-penalty coefficient α | `0.1` | `metrics.py:30` |
| Division weight w | `0.1` | `metrics.py:34` |
| Match radius | `7.0` µm, physical | `metrics.py:276`, `evaluate.py:64,125` |
| Voxel scale (Z,Y,X) | `(1.625, 0.40625, 0.40625)` | `io.py:13`; per-dataset read at `io.py:129-136` |
| `N_est` source | GT geff metadata `extra["estimated_number_of_nodes"]` | `evaluate.py:51-58`, called at `evaluate.py:93` |

The node ratio and multiplier (`metrics.py:439-449`):

```python
total_node_ratio = (er.num_pred_nodes - n_total) / n_total          # metrics.py:440
adj_edge_jaccard = max(0.0, edge_jaccard * (1 - ADJUSTMENT_ALPHA * total_node_ratio))
                                                                    # metrics.py:446-449
```

`num_pred_nodes` is **every** predicted node, matched or not — `graph.num_nodes()`
(`metrics.py:330`). `n_total` is read from the **ground-truth** geff, not the submission
(`evaluate.py:93`), so it is not something a competitor can set.

**Clipping is one-sided.** `max(0.0, …)` clips only from below. **[MEASURED]** sweep through
`per_sample_metrics`:

| N_pred vs N_est | ratio | multiplier m |
|---|---|---|
| 0.5 × | −0.500 | **1.0500** |
| 1.0 × | 0.000 | 1.0000 |
| 2.0 × | +1.000 | 0.9000 |
| 11 × | +10.00 | 0.0000 (hard zero) |

So `m ∈ [0, 1.1]`; the 1.1 ceiling is reached only at `N_pred = 0`. The multiplier reaches
zero only at `N_pred = 11 × N_est` — the node budget is enormously slack, not a tight
constraint. Confirmed live: crop `6bba_05b6850b` has ratio −0.1009, `edge_jaccard` 0.7067 and
`adj_edge_jaccard` **0.7139** (`artifacts/kaggle/m1_heldout_scores/6bba_05b6850b.json`).

**Aggregation is mixed and this matters.** `adj_edge_jaccard` is a **per-sample** value
weight-averaged by `w_i = TP_i+FP_i+FN_i` (`metrics.py:498-506`), whereas `division_jaccard`
is **micro**-pooled (`metrics.py:519-521`). Documented identically at `metrics.md:131-149`.
Consequence: the node-budget multiplier is applied **per crop**, then weighted by that crop's
*edge* count — node overshoot in a crop with few annotated edges is nearly free.

Note also `evaluate_datasets()` (`metrics.py:334-383`) computes a *different*, un-adjusted
micro Jaccard. It is not what the competition uses; `summarise()` is (`evaluate.py:129`,
`metrics.md:139-141`). Do not accidentally validate against `evaluate_datasets`.

### 1.2 Matching — algorithm, tie-breaks, radius, per-frame

**Optimal, not greedy; per-frame; hard radius; maximises Σ 1/(1+d).**

- The scorer builds `DistanceMatching(max_distance=max_distance, scale=scale)` and never
  passes `optimal` (`metrics.py:249`; also `division_metrics.py:122,156`). The default is
  `optimal: bool = True` (`tracksdata/metrics/_matching.py:206`) → **optimal assignment**.
  Organiser prose agrees: `metrics.md:17-20`.
- Nodes are grouped by timepoint and matched **independently per frame**
  (`tracksdata/metrics/_ctc_metrics.py:174` groups by `T`; `_match_single_frame`,
  `_ctc_metrics.py:36-113`, handles one `t`). There is **no cross-frame consistency
  constraint** on node matching.
- Physical distance: both centroid sets are multiplied by `scale` before `cdist`
  (`_matching.py:270-279`), so the 7 µm radius is true physical Euclidean over (z,y,x).
- Candidate pairs are pre-filtered by a hard threshold `distance_matrix <= max_distance`
  (`_matching.py:282`). Beyond 7 µm a pair is not even a candidate.
- Weights are `1/(1+d)` (`_matching.py:296`) and the solver **maximises the sum**
  (`_ctc_metrics.py:84`, `maximize=True`). This is *not* minimum-total-distance — it is a
  hyperbolic objective that will trade one far pair for one very close pair.
- Solver path: `scipy.sparse.csgraph.min_weight_full_bipartite_matching`
  (`_ctc_metrics.py:84`); on `ValueError` (no full matching exists) it fills empty rows/cols
  with `-1.0` (`_ctc_metrics.py:88`; `_fill_empty` at `_ctc_metrics.py:22-33`) and retries
  (`:90`); on a second failure it densifies and uses `linear_sum_assignment` (`:97`), then
  strips the `-1` fills (`:99-103`). Real weights lie in `[0.125, 1.0]`, so fills are always
  dominated → the effective result is a maximum-weight matching over pairs within 7 µm.
- Edge TP is then an inner join of predicted edges against GT edges remapped into predicted
  node-id space (`tracksdata/graph/_base_graph.py:1311-1325`).

**Tie-breaking between exactly-equal weights is scipy-internal** — deterministic for a given
input ordering but undocumented. **[UNVERIFIED]** whether node insertion order can steer
ties; would need a controlled equal-distance experiment. Low value, not pursued.

**[MEASURED]** radius behaviour is a cliff, not a taper: a track offset 6.9 µm scores
TP/FP/FN = 3/0/0; the same track at 8.0 µm scores 0/0/3.

### 1.3 Which predicted edges are penalised, ignored, or free

Four filters run **before** any counting, in `_evaluate_matched_graph`:

1. **Duplicate (source,target) pairs deduped**, matched copy kept (`metrics.py:61-66`).
2. **`dt ≠ 1` edges dropped entirely** — kept only where `t_target − t_source == 1`
   (`metrics.py:74-87`). Backward, same-frame, self and gap-2 edges are neither TP, FP nor FN.
3. **Merge collapse**: several predicted edges mapping onto one GT edge → keep the lowest
   `EDGE_ID` (`metrics.py:115-135`).
4. **Out-degree cap**: rank by `EDGE_ID` per source, keep ≤ 2 (`metrics.py:140-153`). Excess
   children are **dropped, not penalised**, and *which* two survive is decided by edge-id
   order, not by quality.

Then validity (`metrics.py:157-198`):

```python
out_valid = gt_out_degree > 0 ; in_valid = gt_in_degree > 0    # metrics.py:165-166
… .fill_null(False)                                            # metrics.py:176-177
pred_valid = out_valid(source) | in_valid(target)              # metrics.py:193-195
```

`edge_fp = Σ pred_valid − TP` (`metrics.py:209, 315-316`); `edge_fn = gt_num_edges − TP`
(`metrics.py:317`). Documented at `metrics.md:21-34`.

**So a predicted edge is scored at all only if one endpoint matched a GT node that itself
carries a GT edge in the relevant direction.** Everything else is invisible.

**[MEASURED]** synthetic probes against the real `evaluate()` (GT = one 4-node chain unless
noted):

| Case | edge TP/FP/FN | div TP/FP/FN | Verdict |
|---|---|---|---|
| exact copy | 3/0/0 | 0/0/0 | baseline |
| + 680 nodes / 510 edges 1000 µm away | 3/0/0 | 0/0/0 | **unannotated territory is free** |
| gap-2 edge t0→t2 | 1/0/2 | 0/0/0 | **skip edge worthless, not penalised** |
| cross-link between two annotated tracks | 2/1/0 | 0/1/0 | penalised twice (edge + division) |
| extend past **last** annotated frame | 3/0/0 | 0/0/0 | **free** |
| extend before **first** annotated frame | 3/0/0 | 0/0/0 | **free** |
| fork **at** last annotated node (GT out-deg 0) | 3/0/0 | 0/0/0 | **free** (annotation end) |
| fork at **interior** annotated node | 3/1/0 | 0/1/0 | 1 edge FP + 1 division FP |
| out-degree 4 at annotated node | 3/1/0 | 0/1/0 | cap absorbs the 3rd/4th |
| duplicate node, **isolated** | 3/0/0 | 0/0/0 | **free** |
| duplicate as a **disjoint parallel track** 0.4 µm away | 3/0/0 | 0/0/0 | **free** |
| duplicate **linked into** the chain (A→dup→C) | 3/2/0 | 0/1/0 | J 1.000 → 0.600 |
| detector misses 1 / 2 / 3 of 10 frames | 7/0/2, 5/0/4, 3/0/6 | – | each miss costs **2 edges** |
| same, with an interpolated node | 9/0/0 | – | full credit restored |

The duplicate rows **refine a widely-read public claim.** `sleepymegacat` states "a duplicate
detection halves the score … NMS quality outranks everything else"
(https://www.kaggle.com/code/sleepymegacat/the-metric-decides-your-architecture-8-measured).
That holds only when the duplicate's edges touch matched nodes. A duplicate the linker leaves
isolated, or that forms its own disjoint chain, costs **nothing but node budget**.
**The penalised object is the spurious *link*, not the spurious *detection*.** That moves the
fix from "better NMS" (detector work, GPU) to "refuse edges that create a second in-edge, or a
second out-edge that is not a certified division" (post-processing, CPU).

### 1.4 Where sparse annotation creates asymmetries

**[MEASURED]** GT statistics across all 199 train crops (read from `.geff` zarr metadata):

| | crops | GT nodes | GT edges | Σ N_est | annotated |
|---|---|---|---|---|---|
| **44b6** | 71 | 20,197 | 19,826 | 2,618,970 | **0.771 %** |
| **6bba** | 128 | 113,121 | 109,057 | 2,106,147 | **5.371 %** |
| all | 199 | 133,318 | 128,883 | 4,725,117 | 2.821 % |

GT divisions = **151** total (125 in 6bba, 26 in 44b6); 112 of 199 crops contain **zero**
divisions; max 5 in one crop. (The brief said ~304; the true count is 151, matching
`sleepymegacat` fact 3. Corrected — this halves the division denominator and doubles the
value of each division TP relative to the brief's assumption.)

Asymmetries that follow, all legitimate:

- **A1 — 7× annotation-density gap between the two embryos.** Node overshoot costs
  `0.1·TP_i/(E_i·W)`, and `TP_i/E_i` is ~7× larger in 6bba than 44b6, so over-detecting in a
  sparsely-annotated crop is ~7× cheaper. **[INFERENCE]** exploitable only if crop density
  were estimable at inference; it is not (no GT at test time). This mainly *explains why the
  node budget is slack*, rather than being a lever itself.
- **A2 — ~97 % of the volume is invisible to the edge term.** Predicted edges there are free
  (row 8). The *naive* optimum maximises global linking precision; the *metric's* optimum
  maximises precision **only near annotated cells**, which is unknowable. The honest response
  is therefore **maximise recall, and spend precision effort on link topology (no merges,
  exactly one out-edge unless a real division) rather than on detection thresholds.**
- **A3 — annotation boundaries are free.** Predictions before the first / after the last
  annotated node of a track are unscored (`out_valid`/`in_valid` are False at track ends).
  Track extension and end-of-track forks cost nothing.
- **A4 — division FP is charged *only* on annotated territory** (plus cross-component and
  malformed), so the FP count is literally `forks × annotation_density`. Global fork
  suppression is the only control.
- **A5 — the node budget is a coarse organiser-supplied estimate**, not truth
  (`metrics.md:47-49`). What is rewarded is matching `estimated_number_of_nodes`, not matching
  reality. **[UNVERIFIED]** whether `N_est` is biased relative to true cell count; settling it
  needs a trusted independent count on train crops.

### 1.5 Degenerate / under-constrained regions

| Region | Code | Honest optimum ≠ naive optimum |
|---|---|---|
| **No upper clip on `m`** | `metrics.py:447` | Under-predicting nodes pays a bonus up to ×1.1. **But blunt pruning is net-negative:** removing a random node gains `0.1·TP_i/(E_i·W)` ≈ 0.0024/W and loses `2·d_ann·m/W` ≈ 0.056/W → **loss/gain ≈ 23:1** **[INFERENCE]**. Prune only nodes you are confident are false. |
| **`dt≠1` dropped silently** | `metrics.py:85-87` | Gap-closing by drawing a skip edge is worth exactly zero. Recovering a missed frame requires **interpolating the node**, then two `dt=1` edges. `metrics.md:21-30` says such an edge is counted FP; **the code disagrees and the code is what runs.** |
| **Out-degree cap keeps lowest `EDGE_ID`** | `metrics.py:140-153` | Which children survive is arbitrary w.r.t. quality. Emitting >2 children is never punished on the edge term but flips the node into fork status (division-FP risk). Never emit >2; if forced, order your best two first. |
| **Break-even detection confidence** | derived | With `d_ann`≈0.028, `m`≈1, `adjJ`≈0.915, `TP_i/E_i`≈0.0237: `E[Δ] = p·d·2m/W − 0.1·TP_i/(E_i·W) − (1−p)·d·2·adjJ/W = 0` → **p ≈ 0.50** **[INFERENCE]**. The metric's break-even is a coin flip, not 0.97. Our deployed `BIOHUB_DET_THRESHOLD = 0.96875` is far more conservative than the metric requires — implying either the detector score is badly uncalibrated, or the empirical optimum is set by *duplicate-induced mislinks* (§1.3) rather than by the budget term. These two are cheaply separable. |
| **Division FP cost is quadratic in precision** | `division_metrics.py:572-574` | At divJ≈0 extra forks are nearly free, so a gradient-following tuner sees no reason to suppress them — and stays at divJ≈0 forever. Escaping needs a discrete jump in fork precision, not incremental tuning. **This is the trap the whole plateau is in.** |

### 1.6 Division scoring — the rules that decide TP vs FP

`evaluate_divisions` (`division_metrics.py:533-574`) → `score_divisions`
(`division_metrics.py:303-390`). Organiser prose: `metrics.md:51-121`.

- GT division window = grandparent → parent → children → grandchildren (`extract_divisions`,
  `division_metrics.py:58-87`).
- The predicted graph is re-matched **independently against each GT division window**
  (`match_divisions`, `division_metrics.py:90-144`) — a fresh `pred_graph.copy()` per division.
- TP requires: a matched parent anchor; two GT daughter lineages landing on **two distinct**
  predicted child branches (`_bipartite_max_matching`, `division_metrics.py:274-300`); and
  correct directed local topology (`_is_strongly_connected_division`, `:227-271`).
- Final pairing is maximum-cardinality bipartite (`division_metrics.py:383`): one fork serves
  at most one GT division; TP = paired, FN = unpaired GT (`:572-573`).
- **FP set** (`division_metrics.py:389`):
  `fp_forks = (considered | evaluable_forks | invalid_forks) − tp_forks`, where
  - `evaluable_forks` = fork matched a GT node with `gt_out_degree ≥ 1` (`:469-472`)
    ← **our entire FP mass**
  - `cross_component_forks` = two child branches whose nearest matched evidence sits in
    different GT weakly-connected components (`:474-491`) ← what the `aa65e90` patch added;
    the pre-patch metric was exploitable by making everything connected
  - `malformed_forks` = a child (or fallback grandchild) with more than one parent (`:424`,
    `:430`) ← **fires anywhere in the graph, annotated or not**. We emit zero merges so this
    is currently 0 for us, but it is a free FP faucet for anyone whose ILP or gap-closing
    produces merges.
- A fork on a GT node with `out_degree == 0` is **excluded** — that marks the end of
  annotation (docstring `division_metrics.py:505-509`). **[MEASURED]** confirmed (§1.3 table).

**[MEASURED]** timing tolerance: a division predicted one frame late still scores div TP = 1,
but costs 1 edge FP + 2 edge FN. Division *timing* is nearly free on the division term and
costs only on the edge term.

---

## PART 2 — LIVE PUBLIC SURFACE (deltas since `competitive_refresh_2026-08-17.md`)

Pulled via authenticated Kaggle CLI this session; leaderboard CSV timestamped 2026-08-17
evening. Kaggle's discussion API returned 403 and the web UI is an SPA that WebFetch cannot
read, so **discussion coverage is not refreshed here** — the morning sweep in
`competitive_refresh_2026-08-17.md` stands as the current discussion state.

### 2.1 Leaderboard (2,471 teams, up from 2,441 this morning)

| Band | 08-17 am | 08-17 pm | Δ |
|---|---|---|---|
| ≥ 0.950 | 1 | 1 | – |
| ≥ 0.948 (top-3 boundary) | 3 | 3 | – |
| ≥ 0.945 | 7 | 7 | – |
| ≥ 0.940 | 11 | 11 | – |
| ≥ 0.935 | 16 | 17 | +1 |
| ≥ 0.930 | 28 | 30 | +2 |
| ≥ 0.925 | – | 45 | – |
| ≥ 0.920 | 74 | 77 | +3 |
| ≥ 0.918 | – | 94 | – |
| ≥ 0.916 | – | 196 | – |
| ≥ 0.915 | 462 | 484 | +22 |
| **exactly 0.915** | ~300 | **288** | ≈flat |

We are **rank 403 / 2,471 at 0.915** (was 378 this morning) — drifting down inside a static
plateau. Top-10 essentially frozen: Mark Cooper 0.950 (93 subs), TWEAK 0.949 (166), Soheil
Ayati 0.948 (29), yuto083 0.947 (48), z7777 0.945 (**7 subs**), Matt Goldfield 0.945 (124),
enddl22 0.945 (110), Amin 0.943, Tang 0.943, htnhtn 0.942.
Source: `kaggle competitions leaderboard biohub-cell-tracking-during-development -d`
(https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/leaderboard).

The 0.916–0.918 shelf (102 teams) is the `liyansen`-lineage public ceiling propagating.
**Nothing above 0.918 is public.** z7777 reaching 0.945 on **7 submissions** is the strongest
available signal that the top tier holds a *structural* edge, not a tuned one — you do not
find 0.945 by probing when 288 teams cannot find it in dozens of attempts.

### 2.2 What is visibly different about >0.94 — honest answer: nothing is visible

No team above 0.918 has published a notebook, dataset, or technique. All public authors
cross-referenced against the live LB CSV:

| Public author | LB | Rank |
|---|---|---|
| Pilkwang Kim (baseline/EDA author) | 0.921 | 67 |
| liyansen (public ceiling) | 0.918 | 83 |
| altervation | 0.917 | 115 |
| backtracking | 0.916 | 152 |
| **sleepymegacat** (best public metric analysis) | **no submission** | – |
| beicicc | 0.913 | 552 |
| abhimanyu122 | 0.913 | 604 |
| **yakizakana629** | **0.890** | **1249** |
| **kkunizaw** (Zebrahub dataset publisher) | **0.887** | **1298** |

**Calibration warning worth recording:**
`yakizakana629/biohub-metric-aware-lineage-completion` contains in-code comments claiming
`baseline = 0.950`, `V1 threshold=0.965 = 0.950`, `V3 conservative ILP = 0.949`.
**That author is rank 1249 at 0.890.** These are LLM-generated aspirational annotations, not
measurements. Treat in-notebook score claims as worthless unless cross-checked against the LB
CSV. (https://www.kaggle.com/code/yakizakana629/biohub-metric-aware-lineage-completion)

### 2.3 New public artifacts since this morning

- **`kkunizaw/biohub-zh001r`** (364 MB, uploaded 2026-08-17 15:12 — *new since the morning
  report*): `zh001r_iso.npy` (377 MB), `zh001r_tgt.npy` (377 MB), `zh001r_nodes.npz` (9 MB).
  The `_iso`/`_tgt` pairing is an **isotropically-resampled image volume plus a training
  target volume** — this is no longer raw crops, it is a **packaged detector-training set**.
  Together with `kkunizaw/biohub-zmnscrops` (3.66 GB, 08-16) this is the external-Zebrahub H1
  lane being assembled in public. **The publisher sits at 0.887**; downloads 5 and 4. The
  lever remains unclaimed. https://www.kaggle.com/datasets/kkunizaw/biohub-zh001r
- **`altervation/biohub-r48-nobudget`** (author rank 115, 0.917). README, retrieved verbatim:
  *"Biohub R48 — r35 + sparse budget OFF … Weights: spotiflow_domain_r35 @ 0.3 … 44b6: no dens
  max_pred_nodes (R35 used ×0.85); 6bba: identical to R35"*. Two findings: (a) a competitor is
  running **Spotiflow** (a dedicated sub-voxel spot detector) instead of the public UNet —
  consistent with the centroid-precision cliff being real and with our own sub-voxel refine
  lane; (b) they are **explicitly ablating the per-embryo node budget**, having used a ×0.85
  under-shoot previously. My §1.5 analysis says under-shoot is worth little and blunt pruning
  is net-negative 23:1 — their willingness to switch it off at 0.917 is weak corroboration.
  https://www.kaggle.com/datasets/altervation/biohub-r48-nobudget
- New notebooks (all by ≤0.918 authors): `abhimanyu122` v18/v20/v21 (detection-threshold sweep
  0.984→0.996, `MinTrack 9` — pure knob probing), `nishantkharga/biohub-cell-tracking-v3`,
  `backtracking/biohub-medal-v1,v2` (0.916 on 5 subs), `chukkkk/…-learned-graph-w-gap-recovery`,
  `aaaa1597/s1-06-stardist-btrack-pipeline` (StarDist + btrack — another detector swap).

### 2.4 The public 0.918 stack *suppresses* divisions rather than recovering them

`liyansen/biohub-v16-ranker-persistent-divisions` (the public ceiling, LB 0.918) sets:
`BIOHUB_SAFE_DIV_FRAME_FRAC_CAP=0.0076`, `BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP=0.00375`,
`SAFE_DIV_MAX_UM=4.66`, `SAFE_DIV_EXISTING_CHILD_MAX_UM=7.65`, plus a new
`SAFE_DIV_MIN_DIVERGENCE_UM=2.25` ("persistent diverging daughters") and
`OUTPUT_KEEP_DIVISION_COMPONENTS=1`.
(https://www.kaggle.com/code/liyansen/biohub-v16-ranker-persistent-divisions)

**These are byte-identical to our deployed values** (see the config table in
`quickwins_internal_2026-08-17.md`) — same lineage. The public state of the art on divisions
is a *cap*: hold FP down by emitting few forks and accept TP ≈ 0. Combined with §0.4:
**the entire public field, us included, scores ≈0 on a term worth 0.1, and the mechanism that
keeps everyone there is that at divJ≈0 the FP penalty is nearly zero, so no local search ever
finds the escape.**

**[INFERENCE, UNVERIFIED]** If any 0.943–0.950 team has working division recall at even
divJ ≈ 0.30, that alone accounts for +0.030 — 86 % of our gap to first place. Consistent with
(a) z7777 at 0.945 on 7 submissions and (b) no such technique existing anywhere in public. It
is **not proven**; the competing explanation is simply a better detector/edge model (the H1
retrain thesis). **The two hypotheses are cheaply separable on our own OOF** — see F1/F2.

---

## PART 3 — RANKED ACTIONABLE FINDINGS (EV × cheapness)

| # | Finding | EV | Cost | GPU? | Falsification test (one line) |
|---|---|---|---|---|---|
| **F1** | **Division term is worth 0.1 and we claim ~0.000. Not a recall problem (32 % measured) — a precision problem: 516 FP vs 31 GT divisions, and `div_FP ≈ forks × annotation_density`.** Cutting fork emission ~10–30× while keeping the best forks moves divJ 0.018→0.10–0.13 (**+0.008…+0.013**); FP≈0 at current recall gives divJ 0.32 (**+0.030**). | **Very high** | CPU-only replay on cached geffs | **No** | Rank all emitted forks by a confidence score, keep top-k, re-score LOEO: if divJ does not rise monotonically as k falls, the FP mass is not rank-separable and F1 is dead. |
| **F2** | **Deployed substrate recovers ZERO divisions** (TP 0/FP 8/FN 3, `experimental-records.md:174`) while the *weaker* e0c substrate recovers 32 %. Our safe-division caps may be suppressing exactly the forks that would score. | **High** | CPU replay | **No** | Re-score P3-harmonic OOF with `SAFE_DIV_*_FRAC_CAP` raised 4×: if division TP stays 0, the caps are not binding and the fault is in the fork *proposal* stage. |
| **F3** | **Duplicate detections are free unless linked** (§1.3). The public "NMS halves your score" claim is topology-specific. A CPU post-filter forbidding any second in-edge, and any second out-edge not certified as a division, captures the benefit without touching the detector. | High | CPU | **No** | Count nodes with in-degree ≥2 and non-division out-degree ≥2 in our deployed output; if both are ~0 (as in the 40-crop cache) the lever is already closed and F3 is dead. |
| **F4** | **Break-even detection confidence is ≈0.50, not 0.97** (§1.5). `DET_THRESHOLD=0.96875` is far more conservative than the metric requires; the node budget cannot be what holds it there (150 extra nodes = 1 edge TP). | Medium-high | CPU sweep on cached probability maps | **No** | Sweep threshold 0.90→0.97 on LOEO with F3's link filters active: if score still peaks at ~0.97 with duplicates neutralised, the detector score is uncalibrated and the gain must come from calibration, not thresholding. |
| **F5** | **Verify gap repair inserts an interpolated NODE, never a `dt=2` edge** (§1.5, `metrics.py:85-87`). A skip edge is worth exactly zero. | Medium (or zero if already correct) | Audit | **No** | Grep our output graphs for any edge with `t_target − t_source ≠ 1`; count 0 ⇒ closed. |
| **F6** | **Fixing a mis-link is worth 1.64 edge-TPs vs 1.00 for adding a missing one; recovering a missed node is worth 2.00.** Effort ranking: node recall > relink correction > FP deletion. | Medium | Re-prioritisation only | No | Implied directly by the verified closed form; falsified only if `summarise()` changes. |
| **F7** | **Node budget is nearly toothless** (150 extra nodes = 1 edge TP; zero only at 11×N_est) **and blunt pruning is net-negative 23:1.** Deprioritise all node-budget tuning. | Medium (as a *stop-doing*) | Free | No | Sweep the node budget ±20 % on LOEO; if abs(Δscore) > 0.005 the term is stronger than derived and this is wrong. |
| **F8** | Annotation boundaries and unannotated territory are free (§1.3, A2/A3). Track extension and end-of-track forks cost nothing but node budget. | Low-medium | CPU | No | Add unconditional 2-frame track extension at both ends; LOEO score must not fall. |
| **F9** | External Zebrahub now exists publicly as a **packaged detector-training set** (`zh001r_iso`/`_tgt`), published by a competitor at 0.887. Confirms the H1 lane is live in public but unconverted. | Informational | – | (H1 = yes) | – |

**Recommended next action:** F1 and F2 are the same CPU replay on cached geffs and together
settle the single largest quantified gap. Run them before any GPU work.

---

## PART 4 — OUT OF BOUNDS (noted, not pursued)

These exist in the metric's structure. We do **not** pursue them; they are recorded only so
nobody re-derives them later and mistakes them for opportunities.

1. **Leaderboard probing to infer hidden-set structure** — repeated submissions varying node
   counts to solve for per-crop `estimated_number_of_nodes`, or varying fork counts to
   localise annotated cells. The score is smooth enough in `N_pred` to make this
   arithmetically possible. **Out of bounds** (uses the evaluation server as an oracle for
   hidden labels). The top-of-board submission counts (166, 148, 124, 110) are *consistent*
   with heavy probing by others; that is their business, not a template for us.
2. **Transferring known public Zebrahub trajectories into identified hidden crops** — the
   node-id decoding published by `sleepymegacat` (`(t+offset)*BASE + label`) combined with the
   public Zebrahub source movies makes crop re-identification conceivable. **Out of bounds**
   (de-anonymising test crops / importing labels).
3. **The weakly-connected-component exploit** — already closed by the organisers in `aa65e90`
   with rescoring. Historical only.

Everything in Parts 0–3 is honest optimisation of a published objective measured on our own
data.

---

## Appendix — reproduction

- Synthetic metric probes: `<scratchpad>/probe.py`, `<scratchpad>/probe2.py`, `/tmp/probe3.py`
  (run against `vendor/kaggle-cell-tracking/src` plus the installed `tracksdata`).
- GT statistics: read directly from `data/train/*.geff/{zarr.json,nodes/ids,edges/ids}`.
- Division decomposition: `tracking_cellmot.division_metrics._pred_division_fork_sets` on
  `artifacts/kaggle/oof_clean/pred_geffs_split_1/`.
- Closed-form identity check: `tracking_cellmot.metrics.summarise` vs `Σ TP_i·m_i / Σ w_i` on
  `artifacts/kaggle/m1_heldout_scores/*.json` (128 rows, exact to 1e-12).
- Leaderboard: `kaggle competitions leaderboard … -d` (2026-08-17 evening, 2,471 teams).
