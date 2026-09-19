# Handoff — Biohub Cell Tracking

Written 2026-09-19. Competition closes **2026-09-29 23:59 UTC**.

---

## 1. Where things stand

**Board:** `aryaarun07`, standing score **0.947**, rank ~423 of 3,697. There are
**653 teams tied at exactly 0.947** (ranks 186–838) — that tie is the public
notebook plateau. The leader is **0.973**; 7th place (last prize) is **0.964**.
So the gap to money is **+0.017** and to 1st is **+0.026**.

**Submitted and awaiting a board score:** `s05` — one added line,
`BIOHUB_OUTPUT_MOTION_RELINK = 0`. Kaggle's own in-kernel validator, 8 held-out
train films:

| | proxy | adj_edge | divJ | div TP/FP/FN |
|---|---|---|---|---|
| unmodified 0.947 | 0.9491 | 0.9260 | 0.2308 | 3/1/9 |
| s01 `DIVERGE_UM=0` | 0.9369 | 0.9258 | 0.1111 | 3/15/9 |
| **s05 no relink** | **0.9715** | **0.9407** | **0.3077** | **4/1/8** |

**Not submitted:** s01 (validator rejected it), s02/s03/s04 (retired),
s06 (ran, measured **negative**, see §5).

**Built but never pushed:** `submissions/s07_reorder/` — gap2 moved after
safe_div. **Do not push it as built.** See §6.

---

## 2. The thing being worked on

Nothing here trains a model. The deployed pipeline is a **public Kaggle
notebook** (`public notebooks/biohub-0-947-lb-runnable-with-public-datasets.ipynb`,
all code in cell index 2, ~214k chars) running **frozen weights** from three
public datasets by `pilkwang` (already downloaded to `weights/`, SHA256 verified
against the values pinned in the notebook).

Its shape: temporal UNet3D detector + node-transformer edge scorer + ILP, then a
**rule-based post-processing chain**: motion relink → single-parent repair → gap
closing → gap2 recovery → safe division → prune isolated → short-track filter →
linefit smoothing. Everything we have changed is in that chain, via `BIOHUB_*`
environment variables in the config cell.

Metric: `score = adjusted_edge_jaccard + 0.1 * division_jaccard`, where
`adj = max(0, J * (1 - 0.1*(n_pred - n_est)/n_est))` — **uncapped above 1**, so
deleting nodes pays without bound. That is a trap; see §5.

---

## 3. The single most valuable asset: the local harness

**The kernel keeps its prediction `.geff` files, and 8 of them are TRAIN films**,
so we hold **deployed-quality graphs with ground truth**. Post-processing
experiments run in **seconds locally** instead of ~1.75 h on Kaggle.

```
artifacts/s01_output/tracking_repo/predictions/unknown/unet_transformer_val/split_0/*.geff
```

The harness is **validated against the board twice**: it reproduced the s01
failure direction, and it reproduces the deployed division ledger exactly
(3 TP / 1 FP / 9 FN).

**Use these, in this order:**
- `src/biohub/metric2.py` — THE metric for those graphs. `SCALE = (1.625,
  0.40625, 0.40625)`, anisotropic.
- `scripts/24_keep_ilp_edges.py` — `load_pred()`, `load_gt()`, `safe_div()`
- `scripts/91_other_stages.py` — ports of gap_close / gap2 / short_track /
  linefit / prune_isolated onto those coordinates
- `scripts/25_relink_control.py` — a `motion_relink()` that reproduces the
  deployed stage

**DO NOT use** `src/biohub/metric.py` or `src/biohub/postprocess.py` on those
graphs. They assume the **downsampled isotropic 64³ grid** and are **4× wrong in
y and x**. They fail silently. Both now carry warning comments.

Other assets: `artifacts/cache/` (199 films, detections + node-sampled features,
292 MB, downsampled grid), `artifacts/graphs/` (199 films, my own weaker
rebuild), `artifacts/films.csv`, `artifacts/loeo_split.json` (frozen).

---

## 4. What is established

**Motion relink is the one large lever.** Removing it is worth **+0.0224** on
Kaggle's validator (+0.0295 predicted locally). It is *systematic*: pooling
5,540 correctly-linked GT edges, the break rate jumps **171×** with a clean knee
exactly at the 6 µm tight gate (0.17% below, 28.6% above). The mechanism is that
the tight gate **forecloses** rather than defers — a source matched to a nearer
wrong target enters `used_i` and the 10 µm relaxed pass can never revisit it. An
**oracle** per-film adaptive skip beats deleting the stage by only **0.0014**, so
there is nothing better to build.

**The defects are structural, not parametric.** 72 post-processing variables
exist; **38 have never been set by anyone** in the fork lineage. All were priced
against real ground truth and **none beats one division event (0.0083)**. Six are
fully inert (`g2_frac`, `g2_frame`, `g2_step`, `gc_frac`, `gc_reuse`,
`sd_req_nn`). One never-touched boolean is **load-bearing**:
`SAFE_DIV_REQUIRE_DIVERGENCE=0` costs **−0.025** with division FP 2→20.

**Ordering is the recurring bug class — three instances.** Motion relink wipes
the ILP's forks; gap2 before safe_div eats the orphan safe_div needs (5/2/7 →
4/3/8). Divisions must settle before anything perturbs geometry.

**safe_div gates are at a local optimum.** Swept on clean graphs: no change in
any direction beats one division event. Loosening `tau`/`diverge` floods false
positives; tightening loses true ones. An autopsy of all 12 divisions on
deployed graphs finds **exactly one** blocked by a gate, and it misses by 0.35 µm.

**Divisions are played out at 5/12.** Of the rest: 4 have the daughter held by
another parent (3 genuinely nearer a neighbour), 3 are detection/matching, 1 is a
gate. Realistically addressable: ~2.

**Node budget.** 5,429,739 detections vs 4,725,117 estimated (+14.9%), costing
~0.0124. Split: **44b6 +1.1%, 6bba +32.1%** — the entire penalty is a 6bba
problem, and 6bba holds 125 of the 151 divisions. Raising `DET_THRESHOLD`
0.965→0.995 is ~33× cheaper per node removed than any other route.

**`MOTION_RELINK_LEARNED_BONUS` is structurally crippled.** Every `solution` flag
in the prediction `.geff` is True, so the file stores **only ILP-selected edges**
— `prob.get(pair, 0.0)` returns 0 for every alternative. The `−β·prob` term is a
flat ~0.9 µm *incumbency discount*, not a likelihood. (Moot anyway if relink is
off.)

---

## 5. Tried and failed — do not repeat

| attempt | outcome |
|---|---|
| Learned division classifier on frozen UNet features | **AUC 0.456** at the split frame — chance. Nothing clears 0.70. |
| Anaphase hypothesis (signal precedes the split) | t−1 0.510, t−2 0.484. Both chance. Tested, not supported. |
| `SAFE_DIV_DIVERGE_UM = 0` (s01) | Validator: division FP **1 → 15**, divJ 0.2308 → 0.1111. |
| Loosening `tau` / `diverge` on recall evidence | TP up, FP 5×, divJ **fell** 0.067 → 0.027. |
| Fork-before-prune ordering / contested targets | Neutral-to-worse. Contest fires on 2 of 6,063 proposals. |
| DeepCenter as a division veto | Real ranker (AUC 0.836–0.857) but **at its oracle ceiling**, worth +0.0029, and a plain **brightness threshold matches it**. Gain is one fork in one film. |
| Division detection ceiling (pool kernel merging sisters) | **Refuted.** `pool_kernel_um` quantises — 3.0/4.0/5.0 give the identical (3,3,3) kernel, true floor 3.25 µm Chebyshev. Only 1/151 divisions below it; **zero** actually lost. The instance that motivated it was a *linker* failure with all three nodes detected. |
| Short-track filter L≥6 | **Metric exploit.** J falls, multiplier rises, and the multiplier gain is **monotone to L=40**. A real stage would peak. |
| `OUTPUT_LINEFIT_WEIGHT 0.8 → 0.4` (s06) | **−0.00135.** Measured +0.0024 on *raw* graphs; sign flipped on the *relinked* pipeline. See §6. |
| Node-count / `n_est` predictor | Transfers 1-for-2 across embryos (R² 0.937 vs **0.494**, median **17% underestimate** — the dangerous direction). |
| Synthetic division data (someone else's result) | AP 0.98 on held-out synthetic, board **0.910 → 0.906**. |

---

## 6. THE LESSON, and the immediate next action

**Findings measured on raw ILP graphs do not automatically transfer to the
relinked pipeline, and vice versa.** s06 proved this the expensive way: linefit
w=0.4 is +0.0024 on raw graphs and **−0.00135** on the relinked one, because
linefit fits along unique predecessor/successor chains and **relink changes the
topology**. I asserted the two changes were independent. They were not.

**Therefore: `submissions/s07_reorder/` must be rebuilt.** It is gap2-after-
safe_div applied to the **unmodified base with relink ON**, and by the same
argument it will probably not transfer — gap2 competes with safe_div for orphan
nodes, and relink is what manufactures the orphan pool.

**Next action: rebuild s07 on top of s05** (no relink **plus** the gap2 reorder),
so it is one change relative to the configuration that actually scored well, and
is tested in the topology it was measured in. The reorder itself is already
AST-proven — see `scripts/96_reorder_variant.py`, which verifies it is a strict
permutation of statements (558 top-level statements, exactly 1 differs; within
`filter_output_graph`, `sorted(old) == sorted(new)`) and confirms safe_div adds
**edges only, never nodes**, so synthetic node ids do not drift.

After that, re-test **linefit on top of no-relink** — it was positive there
(+0.0024 to +0.0044 on raw graphs) and only failed because it was tested on the
wrong base.

---

## 7. Operating rules

`ABORT_RULES.md` is frozen and carries pre-registered gates, standing
prohibitions and a measurement-error table. The parts that matter most:

- **One change per submission.** Bundling has already cost us an
  uninterpretable sweep.
- **Report `J` and the multiplier separately, every time.** A gain that arrives
  through the multiplier while `J` is flat is the node-count exploit.
- **The node-count term is forbidden ground** for *tuning*. Measuring it is fine.
- **12 ground-truth divisions across the 8 validator films.** One division is
  **0.0083** of proxy. Do not report division differences finer than that as
  meaningful.
- **Edge-axis changes have a bad track record here**: the one published
  offline-vs-board ledger in this competition is **0 for 3 on the edge axis**
  and 3 for 3 on divisions. s05 is an edge-axis change with much better evidence
  than those three, but the base rate is real.

Ledger: `artifacts/ledger/runs.db` (SQLite) — `runs`, `scores`,
`division_events`, `submissions`. Predictions are recorded **before** results so
sign agreement stays honest.

---

## 8. Practical notes

- Kaggle allows a **maximum of 2 concurrent GPU sessions**. A third push is
  refused outright.
- A commit run is ~1.75 h, of which **~75% is the validator and its 7-candidate
  sweep** (which only ever uses train films and picks the same winner). Disabling
  both and hard-setting `MOTION_RELINK_TIGHT_UM=5.5` reproduces an identical
  submission in ~25 min. Untested but low risk.
- `scripts/make_variant.py` builds one-change notebook variants and **refuses to
  write** unless the changed-line count matches what was requested. It also warns
  when a key is covered by the notebook's `_EXPECTED_NUMERIC` drift guard —
  `SAFE_DIV_MAX_UM` is guarded, so changing it needs the guard updated in the
  same edit or the run aborts at cell 3.
- Kaggle auth: token at `~/.kaggle/access_token`, auto-detected. Submit a code
  competition with
  `kaggle competitions submit -c biohub-cell-tracking-during-development -k <owner>/<slug> -v <n> -f submission.csv -m "..."`.
- Machine: M5 Pro. **GPU batching gives literally zero gain** (50.4 ms/window at
  B=1 vs 51.5 ms at B=23) — measured, don't re-litigate. The real headroom is
  **18 CPU cores**; most scripts here are single-threaded.
- `data/` is 82 GB and gitignored. Never write to it.
