# Vendor-drift audit: every divergence from the vendor / public baseline, and whether it is justified

Date 2026-08-22. Read-only audit. No GPU, no submission, no commit.

Deployed artifact audited: `notebooks/kaggle_p3_harmonic/biohub-p3-harmonic.ipynb`
(LB 0.915, submission `55274582`), built from
`notebooks/kaggle_p0b_clean913_revtime/biohub-p0b-clean913-revtime.ipynb`
(live kernel `aryaarun07/biohub-p0b-clean-913-reverse-time`, LB 0.914) plus
`scripts/kaggle_specs/p3_harmonic.json`, with post-processing in `src/biotrack/wrapper.py`.

Baselines pulled and diffed:
`vendor/kaggle-cell-tracking/scripts/predict_unet_transformer.py`;
`yunusgmsoy/kimi-notebook-v17` (public, `c:/temp/kimi17c/`);
`pilkwang/biohub-cell-tracking-two-seeds-logit-blend`;
`kaiwalyaatulraut/biohub-cell-tracking-solution`;
`xiaoleilian/biohub-ct-mix-divaug`.

---

## 0. Scope correction that changes how the whole list reads

**Our deployed pipeline is not a divergent fork of the public baseline. It *is* the public
baseline — one generation stale.**

The env block at `notebooks/kaggle_p0b_clean913_revtime/biohub-p0b-clean913-revtime.ipynb`
cell 2 and the whole `filter_output_graph` chain are byte-shared with
`pilkwang/biohub-cell-tracking-two-seeds-logit-blend` and with `yunusgmsoy/kimi-notebook-v17`.
Verified identical: the secondary/dual-seed block (`SECONDARY_EDGE_WEIGHT=0.15`,
`SECONDARY_DETECTION_WEIGHT=0.475`, `SECONDARY_LINK_MODE=low_margin_consensus`,
`SECONDARY_LOW_MARGIN_MAX=0.35`, `DUAL_SEED_EDGE_THRESHOLD=0.48`), the 8-view D4 detection-TTA
source patch (5 `rot90` occurrences in each), and the harmonic fusion block — the public
notebook carries the *same comment string* ("Biohub 145: require mutual forward/reverse support
in probability space") as our `scripts/kaggle_specs/p3_harmonic.json` edit 1, because both were
transcribed from the same CC0 source (`yusuketogashi/no-hack-biohub-cell-another-approch-3rd`
v18; provenance recorded in `scripts/kaggle_specs/p3_harmonic.json` `provenance.mechanism_source`).

Consequence: `BIOHUB_ILP_APPEARANCE_WEIGHT = "0.0"` is **not our override**. It is in
`kimi-notebook-v17` and `pilkwang` verbatim (`pilk.txt:85`), and `kaiwalyaatulraut`
independently sets `ILP_APPEARANCE_WEIGHT = 0.0` in a separate lineage (`kaiw.txt:225`). Three public notebooks, two
independent lineages, all set it to 0.0 against vendor's 0.1. The catastrophe is real and the
mechanism is proven (section 3.1), but it is a **lineage-wide inherited wound**, not a
self-inflicted one, and a notebook carrying it scores 0.923. Reverting it is still the
best-motivated division experiment we have, but it must be pitched as "the whole public field
has division disabled at the solver and nobody noticed", not as "we broke our own pipeline".

The genuinely *ours-only* set is small: **six knob values** where we sit behind
`kimi-notebook-v17`, plus three degree-invariant guards that are ours alone and are no-ops by
construction. All six are already bundled into the built-but-unscored
`notebooks/kaggle_p8_loosefilter/` (spec `scripts/kaggle_specs/p8_loosefilter.json`,
`research/07-outputs/submissions.md:682-718`).

---

## 1. THE DELIVERABLE — UNJUSTIFIED divergences, ranked

"UNJUSTIFIED" = the value differs from the vendor default and **no measurement of that knob
exists anywhere** in `experimental-records.md` (ER), `bets.yaml` (BETS), `submissions.md` (SUB),
or `failed-experiments.md` (FE). Ranking = P(harmful) x cheapness-to-revert, filtered by the
transfer law (`research/00-system/handoff.md:52-77`): only detection-surface / candidate-set
changes have demonstrated LB transfer (3.5x amplified); division-term and edge-permutation
changes measured 0.000.

| # | Divergence | Vendor | Deployed | Where set | Recorded measurement | P(harmful) | Revert cost | Transfer class | Rank |
|---|---|---|---|---|---|---|---|---|---|
| **U1** | `ilp_disappearance_weight` | **0.1** (`vendor/kaggle-cell-tracking/scripts/predict_unet_transformer.py:79`, argparse `:631`) | **1.5** (15x) | p0b cell 2, `os.environ["BIOHUB_ILP_DISAPPEARANCE_WEIGHT"]="1.5"` | **NONE.** Never named in any record. Never varied. | **HIGH** — mechanism in 3.2 | 1 line | candidate-set (edge selection) | **1** |
| **U2** | `ilp_appearance_weight` | **0.1** (`:78`, argparse `:629`) | **0.0** | p0b cell 2 | Counts only, never a score: 176,835 offered out-deg>=2 sources produced **0 divisions** (ER:1356-1376); 29/125 GT divisions declined at median `edge_prob` 0.9188 (ER:1406-1428). The shipped division fix was `DIVISION_WEIGHT=0.55`, **not** this knob (SUB:414) | **CERTAIN** structurally; **LOW** for score | 1 line | division term — **0x transfer measured** (SUB:468) | **2** |
| **U3** | `det_tta` 4-view to 8-view D4 | **flip-xy only, 4 views** (`:69`, applied `:376-390`) | **8-view D4** (3 flips + rot90 k=1,3 + transpose + anti-transpose) | p0b `_old`/`_new` source patch, notebook text lines 834-882 | **NONE.** `det_tta` appears zero times in all five evidence files. The only TTA datum is third-party: edge-TTA "hurts (0.885)", FE:28 | **LOW-MEDIUM** — rot90 in (y,x) is label-preserving only if the detector saw rotation augmentation; but **all three ILP-based public notebooks independently use extended TTA** (`kimi` and `pilkwang` 5 `rot90`, `kaiwalyaatulraut` 16 across two lineages), which is weak evidence against harm | 1 patch block, but **needs a real kernel run** — there is no 4-view public notebook to price it against | **detection surface — 3.5x amplified** | **5** |
| **U4** | `MOTION_RELINK_LEARNED_BONUS` | n/a (no vendor relink) | **1.0** (wrapper default is 0.75, `src/biotrack/wrapper.py:47`) | p0b cell 2 | Counts only, no score: bonus 0.00 to 8,263 reachable GT edges; 0.75 to 8,264; 3.00 to 8,269, of 9,020 (ER:1112-1122) — the learned probability is worth **1 GT edge in 9,020** | **LOW** harm, but proves the learned-edge term is inert inside the relink | 1 line | edge permutation — 0x | 6 |
| **U5** | `OUTPUT_MIN_TRACK_LEN=6` + `FILTER_SHORT_TRACKS=1` | n/a (vendor writes the solver graph straight out) | on, len 6 | p0b cell 2; `src/biotrack/wrapper.py:72-74`, applied `:1237` | Knob never measured. The *function* is: deletes **5,311 GT edges the relink had correctly linked = 19.2% of all FN** (ER:212-214). Blanket retention was tested and is **worse** (adjJ 0.8821 to 0.8766, ER:177) | **MEDIUM** — known-lossy in both directions | 1 line | candidate-set (node count) | **3** |
| **U6** | `GAP_CLOSE_MAX_GAP=2`, `GAP_CLOSE_UM=5.8`, `GAP_DENSITY_*` (7 knobs) | n/a | on | p0b cell 2; `src/biotrack/wrapper.py:54-66`, applied `:1189` | **NONE.** Not one of these seven names appears in any record | MEDIUM | 7 lines, one flag | candidate-set (adds nodes) | **4** |
| **U7** | `OUTPUT_PRUNE_ISOLATED=1`, `OUTPUT_SINGLE_PARENT_REPAIR=1`, `OUTPUT_ENFORCE_NEXT_FRAME=1`, `OUTPUT_KEEP_DIVISION_COMPONENTS=1`, `OUTPUT_SINGLE_CHILD_REPAIR=0` | n/a | as shown | `src/biotrack/wrapper.py:39-43,74`; applied `:1135,1165,1176,1229` | **NONE** for any of the five names. Nearest analogue (bipartite parent repair) is **CLOSED sign-unstable**: LOFO +0.00099 / +0.00002, transformer picks the true parent 9.36% vs a 50% break-even (ER:175) | MEDIUM | flags | mixed | 7 |
| **U8** | `OUTPUT_LINEFIT_SMOOTH=1`, weight 0.8, window 2 | n/a — vendor writes solver coordinates unmodified | on | `src/biotrack/wrapper.py:76-78`, applied `:1238` | **NONE.** Cited only as *diluting* sub-voxel refinement (ER:517, BETS:163). Its one historical appearance is a defect: unclamped smoothing put node 15274 at z=64 and P0-C was **audit-FAILed and withheld** (SUB:111-115) | MEDIUM — it moves every coordinate the scorer matches on | 1 line | **detection surface (coords)** | 8 |
| **U9** | `PREFIX_DENSITY_BLEND=0.20`, `GAP_REFINE_*` (4 knobs) | n/a | on | `src/biotrack/wrapper.py:62,67-70` | **NONE** | LOW | flags | candidate-set | 9 |
| **U10** | `OUTPUT_EDGE_MAX_UM=14.0`, `OUTPUT_VOLUME_GUARD=0`, `OUTPUT_GAP2_RECOVERY=0`, `OUTPUT_DIVISION_GEOMETRY_FILTER=0` | n/a | as shown | `src/biotrack/wrapper.py:38,50,108,114` | Non-binding or confounded only: arm-B max raw edge **7.846 um**, p99 4.143, **0 edges >10 um** (SUB:230-240) — the 14 um cap is provably inert | **NIL** | — | — | do not touch |

**Divergences that are NOT unjustified**, and must not be confused with the above:
`det_threshold`, `ARMB_FLOW_GATE`, `ILP_DIVISION_WEIGHT`, `OUTPUT_SAFE_DIVISIONS`,
`RESTORE_LEARNED_DIVISIONS`, `BIDIRECTIONAL_EDGE_WEIGHT`, sub-voxel refine. All carry numbers;
see sections 3.3, 5, 6.

### 1.1 The counter-signal: two independent lineages converged on the same "wounds"

`kaiwalyaatulraut/biohub-cell-tracking-solution` is **not** a fork of the `pilkwang` notebook —
it has its own model wrapper, its own two-tier edge thresholding (`EDGE_STRONG_THRESHOLD = 0.50`
/ `EDGE_MIN_THRESHOLD = 0.25`, `EDGE_TOPK_PARENTS = 3`, `kaiw.txt:229-231`), plain Python
constants instead of the `BIOHUB_*` env convention, and no post-solver stack at all. It is a
genuinely separate implementation. Yet it independently lands on:

| knob | vendor | `pilkwang` lineage (= ours) | `kaiwalyaatulraut` (independent) |
|---|---|---|---|
| `ilp_appearance_weight` | 0.1 | **0.0** (`pilk.txt:85`) | **0.0** (`kaiw.txt:225`) |
| `ilp_disappearance_weight` | 0.1 | **1.5** (`pilk.txt:86`) | **1.4** (`kaiw.txt:226`) |
| `ilp_edge_weight` / `ilp_division_weight` | -1.0 / 1.0 | -1.0 / 1.0 | -1.0 / 1.0 (`kaiw.txt:224,227`) |
| detection threshold | 0.5 (0.99 CLI) | **0.96875** | **0.9500** (`POINT_THRESHOLD`, `kaiw.txt:220`) |
| extended rotational TTA | 4-view | **8-view** (5 `rot90`) | **broader** (16 `rot90`) |

Two independent lineages converging on appearance ~0.0, disappearance ~1.4-1.5, detection
~0.95-0.97 and extended TTA is **weak but real evidence that these were tuned by somebody**,
even though no published measurement exists for any of them and none exists in our corpus.

This does not remove them from the UNJUSTIFIED list — "unjustified" here means *we* hold no
measurement, and that remains exactly true. But it does change the expected value of reverting
them: the payoff is **primarily informational** (nobody in the field appears to have measured
these, so a clean measurement is genuinely new knowledge), not an expected score gain. Rank the
reverts as experiments, not as fixes, and do not expect U1-U3 to be free points.

---

## 2. Level (a) — configuration diff, vendor vs deployed vs public majority

Vendor authority: `PredictConfig` dataclass at
`vendor/kaggle-cell-tracking/scripts/predict_unet_transformer.py:44-93`; argparse `:599-634`;
`cfg` construction `:646-653`.

| knob | vendor dataclass | vendor argparse | **deployed (p0b/p3)** | kimi-v17 | pilkwang | kaiwalya | verdict |
|---|---|---|---|---|---|---|---|
| `use_ilp` | **False** `:76` | `--use-ilp` `:623` | **True** | True | True | True | unanimous public divergence; keep |
| `det_threshold` | **0.5** `:68` | **0.99** `:618` | **0.96875** | 0.96875 | 0.96875 | 0.95 | MEASURED (3.3) |
| `pool_kernel_um` | **3.0** `:70` | not exposed | **3.0** | 3.0 | 3.0 | 3.0 | agrees with vendor |
| `det_tta` | **True, 4-view** `:69`, `:376-390` | not exposed | **8-view D4** | 8-view D4 (5 `rot90`) | 8-view D4 (5 `rot90`) | broader still (16 `rot90`) | **U3 — unmeasured by us, but unanimous in the field** |
| `edge_activation` | **softmax** `:72` | not exposed | softmax | softmax | softmax | softmax | agrees |
| `threshold` (edge) | **0.5** `:73` | not exposed | 0.5 + `DUAL_SEED_EDGE_THRESHOLD=0.48` | same | same | 0.50/0.25 two-tier | shared |
| `ilp_edge_weight` | **-1.0** `:77` | -1.0 `:627` | **-1.0** | -1.0 | -1.0 | -1.0 | agrees |
| `ilp_appearance_weight` | **0.1** `:78` | 0.1 `:629` | **0.0** | 0.0 | 0.0 | 0.0 | **U2** (lineage-wide) |
| `ilp_disappearance_weight` | **0.1** `:79` | 0.1 `:631` | **1.5** | 1.5 | 1.5 | 1.4 | **U1** (lineage-wide) |
| `ilp_division_weight` | **1.0** `:80` | 1.0 `:633` | **1.0** | 1.0 (explicit) | 1.0 | 1.0 | agrees |
| `max_parents_per_node` | None under ILP `:89-93` | not exposed | None | None | None | 3 pre-ILP | dead code (ER:1270-1271) |
| `max_children_per_node` | None under ILP `:89-93` | not exposed | None | None | None | None | dead code (ER:1278-1279) |
| weights | single `edge_predictor_best.pth` `:660` | — | **3 models**: primary + `seed314159` secondary + DeepCenter veto | same | same | single | shared |

Vendor knobs that are **unreachable from the CLI** and can only be changed by source-string
patching — which is exactly what our notebook and `kimi`/`pilkwang` all do:
`pool_kernel_um`, `det_tta`, `edge_activation`, `threshold`, `max_parents_per_node`,
`max_children_per_node`. Argparse exposes only `det_threshold`, `use_ilp` and the four ILP
weights (`:599-634`).

**Ours-only deltas — every one of them is us sitting behind `kimi-notebook-v17`:**

| knob | ours | kimi-v17 (public 0.923) | already in p8? |
|---|---|---|---|
| `BIOHUB_SAFE_DIV_MAX_UM` | 4.66 | **12.0** | yes |
| `BIOHUB_SAFE_DIV_SISTER_MAX_UM` | 8.5 | **15.0** | yes |
| `BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM` | 7.65 | **10.0** | yes |
| `BIOHUB_DEEPCENTER_SAFE_DIV_VETO` | 0 | **1** | yes |
| `BIOHUB_DEEPCENTER_CHECKPOINT` / `EXPECTED_EPOCH` | `checkpoint_last.pt` / 500 | **`best.pt` / 2** | yes |
| `BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT` | 0.20 | **0.30** | yes |

Nothing else differs. `notebooks/kaggle_p8_loosefilter/` is built and unscored; it closes the
entire ours-vs-public gap in one artifact.

---

## 3. Mechanism, for the three that matter

### 3.1 `appearance_weight = 0.0` makes division strictly dominated (re-verified from source)

`tracksdata` minimises its objective. Flow constraints, read directly from
`.venv/Lib/site-packages/tracksdata/solvers/_ilp_solver.py:296-319`:

```
in : appear[t]    + sum(in_edges(t))  == node[t]
out: disappear[s] + sum(out_edges(s)) == node[s] + division[s]
```

Out-degree 2 on a selected source therefore requires `division[s] = 1`. Adding a second child
`c` that was previously a track start (`appear[c]: 1 -> 0`) changes the objective by

```
delta = division_weight - appearance_weight - p
```

* vendor: `1.0 - 0.1 - p = 0.9 - p` — strictly beneficial for `p > 0.9`
* deployed: `1.0 - 0.0 - p = 1.0 - p >= 0` for every `p <= 1` — **never strictly beneficial;
  ties only at `p == 1.0` exactly**

This reproduces both recorded results: the CPU toy (vendor emits at `p >= 0.90`, not at 0.88 —
ER:1244-1267) and the full fold-1 export (176,835 candidate sources, 0 divisions —
ER:1356-1376). The proof is closed. What is new here is that **three public notebooks including
the 0.923 one carry the same setting**, so "restore 0.1" is a divergence from the entire field.

### 3.2 `disappearance_weight = 1.5` — the unexamined 15x, and why it outranks U2

Adding edge `s -> t` where `s` currently terminates and `t` currently starts:

```
delta = -p - disappearance_weight - appearance_weight
      = -(p + 1.5)   deployed        vs   -(p + 0.2)   vendor
```

Both are negative, so both link greedily — the *ratio* is what changed. `edge_prob` spans at
most 1.0. A flow bias of 1.5 **exceeds the entire dynamic range of the learned edge
probability**. The keep-or-drop decision is therefore made almost entirely by the flow term,
and `p` survives only as a tie-break among competing targets (where the appear/disappear terms
cancel). The ILP degenerates from a selective solver into a near-maximum-cardinality bipartite
matching per frame pair, with `p` as tie-break.

Two independent corpus findings, previously unexplained, are exactly what that predicts:

* the geometric motion relink can **replace the ILP's edges wholesale** with essentially no
  loss (`src/biotrack/wrapper.py:1158-1161`, ER:1274-1276) — expected if the ILP was only doing
  maximal matching;
* deleting the learned probability from the relink costs **1 GT edge in 9,020** (ER:1112-1122).

At vendor 0.1 the flow bias is 0.2, one fifth of `p`'s range, and edge quality actually decides
links. This is a **candidate-set change** — the transfer-proven class — and it has zero recorded
measurement. That is why it outranks the appearance weight, whose class measured 0.000 transfer.

Caveat to state before running it: at 1.5 the objective is also **time-asymmetric** — births
are free, deaths cost 1.5 — so on a forward-time graph every node is biased toward linking
forward. Lowering it will *reduce* the edge count, and node/edge count feeds the
`adj_J = J * (1 - 0.1*(N_pred - N_est)/N_est)` multiplier. Expect a two-sided effect; do not
predict the sign.

### 3.3 `det_threshold` is measured, and defines the transfer law — do not re-open casually

0.96875 to 0.999 cost **LB -0.0320** on a local -0.0091 (SUB:527, 556-564, 579-584, 599) — the
3.5x amplification the whole ranking above depends on. Downward moves were swept offline and
are negative or inside noise everywhere honest (ER:1050-1056); a label-free check found
**0 of 28,800** sub-threshold local maxima within 7 um of GT against a uniform-null expectation
of 97 (ER:1046-1048). Classification: **MEASURED-WIN, leave alone.** Note only that no public
notebook uses either vendor value (0.5 or 0.99) — the field converged independently on
0.95-0.97.

---

## 4. Level (b) — the injected `scripts/kaggle_edits/*.py` patches

Only two things in this directory reach the deployed 0.915 artifact. The rest are
experiment-lane only and must not be mistaken for deployment surface.

**In the deployed artifact, via `scripts/kaggle_specs/p3_harmonic.json`:**

| patch | what it changes | justification | class |
|---|---|---|---|
| `p3_harmonic.json` edit 1 (inline, not a file) | arithmetic mean of forward/reverse edge logits becomes a weighted **harmonic mean in probability space**, renormalised and affinely rescaled back onto the forward logit scale; lambda = 0.20 unchanged | **MEASURED-WIN, +0.001 LB** (0.914 to 0.915, SUB:290, 337-346). Mechanism is public CC0, transcribed; the public 0.923 notebook's block is character-identical | edge permutation |
| `scripts/kaggle_edits/degree_invariants.py` (36 lines) + 4 inline guard edits in the same spec | asserts in-degree <= 1 / out-degree <= 2 at three stages; adds source-side dedupe to `add_safe_divisions_postlink` (`safe_division_skipped_outdegree`); adds `incoming` checks to `close_single_frame_gaps` | **Not a scoring change.** Fixes a real defect: one node in 121,003 reached out-degree 3 and blocked the first arm-B run (file header, lines 1-14). No-op when invariants hold. **Ours alone** — the public notebook has zero occurrences of `assert_degree_invariants` | safety |

**Not deployed — experiment / diagnostic lanes only:** `armb_flow_gate.py` (arm B; LB 0.000
twice, SUB:248-255 and ER:479-486), `armb_provenance.py` (diagnostic, runs after the CSV is
closed), `d1_inject.py` / `d1_response_audit.py` / `d1_aggregate.py` (D1 representation audit),
`detpeak_export.py` (threshold-superset export; specs written, **never pushed**, ER:765-769),
`h1r_trainer_patch.py` / `h1r_edge_loss_patch.py` / `h1r_det_train.py` (Zebrahub retrain lane;
the edge-loss patch self-tests 12/12 but is **inert, never run**, ER:1161-1168),
`loeo_export.py` / `loeo_pregraph_export.py` / `loeo_retarget.py` (LOEO harness — section 6),
`node_budget_stage.py` (**PROVISIONAL**; its own header forbids quoting +0.00157 on the P0-B
substrate), `restore_learned_divisions.py`, `subvoxel_refine.py` (**KILLED**, ER:500-510),
`swap_edge_weights.py`, `volume_guard_linefit.py`, `pre_ilp_export.py`, `pre_ilp_rollup.py`.

---

## 5. Level (c) — `src/biotrack/wrapper.py` stages with no vendor counterpart

Vendor ends at `solver.solve(graph)` then `save_graph(...)`
(`vendor/kaggle-cell-tracking/scripts/predict_unet_transformer.py:555-565`). Everything below
is post-solver invention.

**The comparison that matters:** `kaiwalyaatulraut/biohub-cell-tracking-solution` (209 votes) — a
high-vote public notebook on the same vendor architecture and the same
`pilkwang/biohub-tracking-support-pack-50ep-v1` weights — has **zero post-ILP track editing**
(`kaiw.txt:845-854`: build graph, solve, save; zero occurrences of `filter_output_graph`,
`motion_relink`, `add_safe_divisions` or `close_single_frame_gaps`). `xiaoleilian` has no solver at all. Our lineage stacks nine
stages on top.

Execution order inside `filter_output_graph` (`src/biotrack/wrapper.py:1069-1241`):

| # | stage | line | knob | status |
|---|---|---|---|---|
| 1 | non-consecutive-t edge drop | `:1135` | `OUTPUT_ENFORCE_NEXT_FRAME=1` | **U7** |
| 2 | edge length cap | `:1140` | `OUTPUT_EDGE_MAX_UM=14.0` | **inert** — max observed 7.846 um (SUB:230-240) |
| 3 | **motion relink — replaces every ILP edge** | `:1145`, call `:1158` | `OUTPUT_MOTION_RELINK=1` | never A/B'd; fires on 100% of crops, `motion_relink_fallback_raw = 0` (ER:1274-1276). Learned term worth 1 edge in 9,020 (**U4**) |
| 4 | single-parent / single-child repair | `:1165`, `:1176` | `SINGLE_PARENT_REPAIR=1`, `SINGLE_CHILD_REPAIR=0` | **U7**; analogue CLOSED sign-unstable (ER:175-176) |
| 5 | gap close, density-adaptive, synthetic-node refine | `:1189` | 12 knobs | **U6, U9** |
| 6 | strict gap-2 recovery | `:1190` | `OUTPUT_GAP2_RECOVERY=0` | off, inert |
| 7 | `add_safe_divisions_postlink` | `:1192` | `SAFE_DIV_*` | MEASURED both ways — see below |
| 8 | division geometry filter | `:1195` | `=0` | off, inert |
| 9 | prune isolated | `:1229` | `OUTPUT_PRUNE_ISOLATED=1` | **U7** |
| 10 | short-track component filter | `:1237` | `MIN_TRACK_LEN=6` | **U5** — deletes 19.2% of all FN (ER:212-214) |
| 11 | line-fit smoothing | `:1238` | weight 0.8, window 2 | **U8** — moves every scored coordinate, never measured |

Stage 7 is the only stage with real numbers, and they are contradictory by regime: turning it
OFF measured **-0.0007** (ER:915-942); once the ILP supplies divisions, OFF is worth
**+0.0227 projected** vs +0.0046 (ER:1439-1443); it shipped OFF in `p5_divfix` for **LB 0.915,
flat** (SUB:415, 449-450, 468). Its parent gate of 4.7 um sits against a true `parent_dist`
median of **7.42 um (44b6) / 8.87 um (6bba)** and admits **1 of 151** GT divisions
(ER:848-871). Widening the gates on 199 crops bought at most **+0.0004**, inside the +/-0.003
noise floor (ER:1293-1297).

---

## 6. LOEO-justified adoptions, re-listed as unverified

Per the standing finding (`research/00-system/handoff.md:24-31`, ER:479-498): the arm-B gate
cleared a bilateral paired LOEO bar of +0.0144 / +0.0090 and delivered **+0.000** on the
leaderboard (submission `55585140`). Measured LOEO-to-LB slope **0.00 (n=2)** against a
projected 0.56; realisation ratio ~0.083 across all scored predictions
(`research/00-system/handoff.md:152-157`).

| adopted or judged on LOEO | LOEO number | LB reality | status now |
|---|---|---|---|
| `BIOHUB_ARMB_FLOW_GATE` | +0.0144 (44b6) / +0.0090 (6bba); min-fold clears the +0.005 bar (ER:327-342) | **0.000** twice: solo `55181562` and on P3 `55585140` (SUB:248-255, ER:479-486) | **MEASURED-NEUTRAL. Not in the deployed 0.915 artifact — correctly not carried.** |
| sub-voxel refine | -0.0004 / -0.0009 (ER:500-510) | never submitted | **KILLED — correct** |
| grid-centre shift residual | +1: +0.00129 / -0.00080; +2: -0.00163 / -0.00012; +3: -0.00652 (ER:706-716) | never submitted | **KILLED, and the premise was refuted at code level (ER:718-732) — correct** |
| GT-free component selector | -0.008 (BETS:224-234, FE:25) | never submitted | **KILLED — correct** |
| `OUTPUT_SAFE_DIVISIONS=0` | -0.0007 / -0.00063 (ER:915-942) | shipped in `p5_divfix`, **LB flat 0.915** (SUB:468) | **MEASURED-NEUTRAL on LB; the LOEO sign was noise** |

**Nothing currently deployed in the 0.915 artifact rests on a LOEO number alone.** The LOEO
lane has functioned as a rejection instrument, and everything it accepted was subsequently
flattened by the leaderboard. It is additionally structurally invalid for the deployed
pipeline: the `seed314159` secondary reports `train_datasets: 199`, so LOEO must ablate it and
therefore measures a *different pipeline* than the one we submit (ER:634-668, 951-966,
1181-1183).

---

## 7. Concrete revert patches for the top of the list

All are single-line edits to the env block at
`notebooks/kaggle_p0b_clean913_revtime/biohub-p0b-clean913-revtime.ipynb` cell 2, or the
equivalent `"kind": "env"` entry in a new spec under `scripts/kaggle_specs/`.
**Do not apply these against the live P0-B spec** — `scripts/kaggle_specs/live_p0b.json`
`purpose` explicitly forbids building or pushing against it. Create a new spec whose
`base_notebook` is the P3 harmonic notebook.

### R1 — disappearance weight toward vendor (rank 1; candidate-set class, 3.5x transfer)

```python
-os.environ["BIOHUB_ILP_DISAPPEARANCE_WEIGHT"] = "1.5"
+os.environ["BIOHUB_ILP_DISAPPEARANCE_WEIGHT"] = "0.5"   # toward vendor 0.1 (predict_unet_transformer.py:79)
```

Use **0.5** first, not 0.1: it halves the flow bias to below `edge_prob`'s dynamic range
without inverting the field's whole design in one step.

Pre-register the falsification before running: if the ILP was already only doing maximal
matching, this must **drop the pre-relink edge count materially (>2%)** and **must not** change
the post-relink submission edge count by more than ~1%, because stage 3 rebuilds the edge list
regardless. A near-identical submission confirms the solver is decorative — a bigger result
than the score. Run it as a **pre-relink census first**, not as a submission:
`scripts/kaggle_edits/loeo_pregraph_export.py` already exports exactly that graph.

### R2 — appearance weight to vendor (rank 2; division class, 0x transfer measured)

```python
-os.environ["BIOHUB_ILP_APPEARANCE_WEIGHT"] = "0.0"
+os.environ["BIOHUB_ILP_APPEARANCE_WEIGHT"] = "0.1"   # vendor default, predict_unet_transformer.py:78
```

Restores `delta = 0.9 - p`, i.e. divisions become beneficial at `p > 0.9`. ER:1406-1428 says 29
of 125 GT divisions were declined at a median `p` of 0.9188 — those come back.
**Must be paired with `BIOHUB_RESTORE_LEARNED_DIVISIONS=1`**, or `motion_relink_edges` deletes
every ILP division wholesale (ER:1447-1449, BETS:81-83). Expected LB effect under the transfer
law: **0.000**. Ship it only bundled behind R1 or R3, never as a solo slot.

### R3 — TTA breadth (rank 5; detection surface, but no free read available)

The `_old` / `_new` string patch in the p0b notebook (text lines 834-882) is self-disabling: if
`_old not in _s` it prints `"TTA WARNING: block not found - using default 4-way"`. As a spec
edit:

```json
{"kind": "replace", "cell_match": "TTA patch applied",
 "old": "if _old in _s:",
 "new": "if False:  # AUDIT 2026-08-22: revert to vendor 4-view flip-xy TTA (predict_unet_transformer.py:376-390)",
 "expect": 1}
```

**Correction to an earlier draft of this report: there is no free natural experiment here.**
I initially recorded `pilkwang` as 4-view; on pulling and checking the notebook it carries the
same 5 `rot90` occurrences as `kimi-notebook-v17`, and `kaiwalyaatulraut` carries 16. Extended
TTA is **unanimous across every ILP-based public notebook in both lineages**, so no public LB
pair prices this knob and reverting it requires a real kernel run. That unanimity is also weak
evidence against harm, which is why U3 is ranked 5 rather than 3. Run it only after R1. If
8-view is worth nothing, we also learn that `d1_inject.py`'s A0 TTA-feature accumulator is
guarding a non-asset.

### R4 — the whole ours-vs-public gap, already built and paid for

`notebooks/kaggle_p8_loosefilter/` closes all six ours-only deltas in one artifact and is
**built but unscored** (`scripts/kaggle_specs/p8_loosefilter.json`, SUB:682-718). It should be
scored before any of R1-R3: it is the only change in this document that moves us *toward* a
notebook publicly scoring above us, and the build cost is already sunk.

---

## 8. Conclusions

1. The premise "we accumulated ~55 levers of our own" is **wrong**. Effectively the entire
   post-solver stack, the detection threshold, the dual-seed blend, the extended TTA and both
   pathological ILP weights are **inherited from the public `pilkwang` lineage** and are shared
   with a notebook scoring 0.923. Our own additions to the deployed artifact amount to the
   harmonic fusion edit (a public CC0 mechanism, +0.001 LB) and three degree-invariant
   assertions (no-ops by construction).
2. The real self-inflicted position is not a wound but a **lag**: six knob values behind
   `kimi-notebook-v17`, all six already bundled in a built, unscored artifact.
3. `ilp_disappearance_weight = 1.5` (**U1**) is the largest wholly-unexamined divergence in the
   system, sits in the transfer-proven class, plausibly explains two previously unexplained
   corpus findings, and has never been named in a single record. Temper the expectation: a
   genuinely independent public implementation chose **1.4** for the same parameter
   (`kaiw.txt:226`), so somebody probably tuned it. Run it for the knowledge, not for points.
4. The strongest structural signal in this audit is external: **a comparable public notebook
   does no post-ILP track editing at all.** Our nine-stage stack has never, in aggregate, been
   measured against not having it. That is the cheapest large experiment available and it needs
   no submission slot — `scripts/kaggle_edits/loeo_pregraph_export.py` already exports the
   pre-wrapper graph the comparison requires.


---

## 9. Verification record and one correction

Every record citation in this report was checked against the file it names, not carried from
memory. Spot-verified line-for-line: ER:175, ER:177, ER:212-214, ER:848-871 (incl. the 7.42 /
8.87 um parent_dist medians at ER:867-868), ER:1044-1048, ER:1112-1122, ER:1244-1250,
ER:1274-1276, ER:1356-1376, FE:25-28, SUB:230-240, SUB:248-255, SUB:520-545, SUB:556-600,
SUB:682-690. Vendor line numbers were read directly from
`vendor/kaggle-cell-tracking/scripts/predict_unet_transformer.py`; the ILP flow constraints in
section 3.1 were read from the installed solver at
`.venv/Lib/site-packages/tracksdata/solvers/_ilp_solver.py:296-319`, not inferred.

Public notebooks were pulled and converted to text before being cited:

| kernel | votes | local text |
|---|---|---|
| `yunusgmsoy/kimi-notebook-v17` | — | `c:/temp/nbtxt/kimi17.txt` |
| `pilkwang/biohub-cell-tracking-two-seeds-logit-blend` | 191 | `c:/temp/nbtxt/pilk.txt` |
| `kaiwalyaatulraut/biohub-cell-tracking-solution` | 209 | `c:/temp/nbtxt/kaiw.txt` |
| `xiaoleilian/biohub-ct-mix-divaug` | 234 | `c:/temp/nbtxt/xiao.txt` |

**One substantive correction was made after the first draft.** That draft recorded `pilkwang` as
using vendor 4-view TTA and built an argument on it: that `pilkwang` (4-view) versus
`kimi-notebook-v17` (8-view) formed a free natural experiment whose public LB difference would
price the TTA knob at zero cost. On pulling the notebook this is **false** — `pilk.txt` contains
the same 5 `rot90` occurrences as `kimi17.txt`, and `kaiw.txt` contains 16. Extended rotational
TTA is unanimous across every ILP-based public notebook in both lineages. Consequences applied:
U3 demoted from rank 3 to rank 5, its P(harmful) lowered to LOW-MEDIUM, its revert cost raised
to "needs a real kernel run", and the free-read advice struck from R3. The `xiaoleilian`
notebook (234 votes, the most-voted in the competition listing) was also checked and has **no
ILP solver and no rotational TTA at all**, so it does not bear on this knob either way.
