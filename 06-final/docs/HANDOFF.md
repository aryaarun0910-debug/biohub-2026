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

**Pushed 2026-09-19, kernels RUNNING, both GPU slots in use:**

| | change | on top of | local delta | kernel |
|---|---|---|---|---|
| `s08` | gap2 moved after safe_div | s05 | **+0.00748** | `biohub-s08-reorder-on-s05` |
| `s09` | `OUTPUT_LINEFIT_WEIGHT` 0.8 → 0.3 | s08 | **+0.00487** | `biohub-s09-linefit03-on-s08` |

Each is ONE change relative to its stated parent, machine-proven (§6, §9). Their
in-kernel validator readings arrive in ~1.75 h and are the real decision signal —
the board can take **8 hours**, so do not wait on it.

**Superseded, do not push:** `submissions/s07_reorder/` — the same reorder on the
unmodified base with relink **ON**. Wrong base; s08 replaces it. See §6.

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

**DONE — s08 is that rebuild.** `python scripts/96_reorder_variant.py s08`
builds it; `submissions/s08_reorder_on_s05/`. The builder proves seven things,
including two that are new: (f) the s05 env line adds **exactly one** top-level
statement and leaves `filter_output_graph` a pure permutation of the base, and
(g) the built notebook differs from the **shipped s05 notebook** by exactly the
two moved statements — so "one change relative to s05" is demonstrated, not
asserted. `python scripts/96_reorder_variant.py s07` still builds the old, wrong
variant if it is ever needed for comparison.

**And this time the transfer was checked before building**, which is the whole
point of the lesson above. `scripts/98_reorder_on_norelink.py` runs the **full**
s05 chain at deployed parameters on the 8 validator films, gap2 on either side
of safe_div, nothing else differing:

| configuration | proxy | J | mult | divJ | TP/FP/FN |
|---|---|---|---|---|---|
| A  gap2 BEFORE safe_div (= s05) | 0.96682 | 0.93483 | 1.00361 | 0.2857 | 4/2/8 |
| **B  gap2 AFTER safe_div (= s08)** | **0.97431** | 0.93516 | 1.00361 | **0.3571** | **5/2/7** |
| C  gap2 OFF entirely | 0.97297 | 0.93333 | 1.00415 | 0.3571 | 5/2/7 |

**B − A = +0.00748.** The three-stage topology predicted +0.00922; the full chain
gave 19% less but the **sign held** — unlike s06's linefit, which flipped.

Three things to keep straight about that number:

- **It is exactly one division**, not a broad gain. divJ 0.2857 → 0.3571 is one
  event at a denominator of 14, worth 0.00714; the remaining +0.00034 is edge J.
  All of it comes from **one film**, `44b6_341df25f` (0/0/1 → 1/0/0). Five of the
  eight films move by exactly 0.00000. Do not read +0.00748 as resolution it does
  not have — §7's floor rule applies, and note the floor here is **0.00714**, not
  the 0.0083 quoted for a 12-division denominator.
- **But it is invariant.** Positive in **9/9** perturbations of the stages either
  side — linefit w ∈ {0, 0.4, 0.8}, window 2/3, gap_close 3/5/8 µm, short-track
  L=6/L=9, keep_forks on/off, gap2 10.2/4.4 and 14/6 — and the spread across all
  nine is **+0.00748 to +0.00750**. That is the mechanistic signature: the reorder
  is a discrete contest over one orphan node, not a continuous geometric fit, so
  the surrounding stages cannot move it. This is precisely why it behaves unlike
  linefit.
- **Row C is the surprise.** gap2 in its *deployed* position is **worse than
  turning gap2 off** (+0.00615 for C over A). Row C trips the harness's
  `<<FALSE GAIN: all multiplier` flag, but that is a **false alarm** — the flag
  only inspects edge J and cannot see divJ, and C's gain over A is the same
  division B recovers, not node deletion. gap2 only earns its place once it
  runs after safe_div, where it beats C by +0.00133 with a real edge-J gain
  (+0.00149) and no multiplier trick. If s08 is ever rejected, `GAP2_RECOVERY=0`
  is the cheaper fallback and recovers most of the same ground.

**The gain is on the DIVISION axis** — it *is* a division — where §7's
offline-vs-board ledger is **3 for 3**, not the edge axis where it is 0 for 3.
That is a better prior than s05 had.

**FALSIFIER, and why s08 has not been pushed:** s08's base is s05, and **s05 has
no board score yet** (submitted 12:25 UTC 2026-09-19, still `PENDING`; a commit
run is ~1.75 h). If s05 does not beat 0.947, s08 is void with it — its entire
premise is that no-relink is the right base. Do not push s08 until s05 scores.

**DONE — that is s09, and the predicted sign flip is confirmed.**
`scripts/99_linefit_on_s08.py` sweeps (weight, window) over the full **s08**
chain. Deployed w=0.8/win=2 scores 0.97431; the surface:

| w \ window | 2 | 3 | 4 |
|---|---|---|---|
| 0.2 | +0.00440 | +0.00278 | +0.00278 |
| **0.3** | **+0.00487** | +0.00505 | +0.00441 |
| 0.4 | +0.00372 | +0.00570 | +0.00620 |
| 0.5 | +0.00291 | +0.00570 | +0.00166 |
| 0.8 (deployed) | 0 | +0.00032 | −0.00241 |
| 1.0 | −0.00354 | −0.00821 | −0.00981 |

s06 measured w=0.4 at **−0.00135 on the relinked pipeline**; here it is
**+0.00372**, and w=0.3 is +0.00487. Same knob, opposite sign, different base —
the §6 lesson, confirmed in the direction it predicted.

**w=0.3 was shipped, not the argmax.** The argmax is w=0.4/win=4 (+0.00620), but
w=0.3 is the most **window-stable** weight — spread across windows 2/3/4 is
**0.00064**, against 0.00248 for w=0.4 and 0.00404 for w=0.5 — and it leaves
window at its deployed 2, so only one variable moves. Taking the argmax would
have meant moving two variables onto a cell that the handoff's own ~0.002 wobble
warning says is untrustworthy.

**Caveat, and it is a real one:** the gain is high-variance. At the argmax only
**3/8 films improve**; the total is carried by 44b6_267148e4 (+0.02847),
6bba_09961292 (+0.01263) and 6bba_062c8d37 (+0.01106), while two films lose
~0.006. The division ledger is **unchanged at 5/2/7 in all 24 cells**, so unlike
s08 this is a **pure edge-axis lever** — the axis where the board ledger is
**0 for 3**. Weigh s09 accordingly: good local evidence, bad axis prior.

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
- **Kaggle slugs come from the TITLE, not the `id`.** s05 shipped with
  `"id": "aryaarun07/biohub-s05-no-relink"` and went live at
  **`biohub-s05-no-motion-relink`** = slugify("Biohub S05 no motion relink");
  s06's log is `biohub-s06-linefit-weight-0-4.log`. Querying the metadata id
  gives a misleading *"Permission 'kernels.get' was denied"*, not a 404. Both
  builders now abort unless `slug == slugify(title)`.
- **The in-kernel PP sweep is provably inert once relink is off.** In
  `artifacts/s05_output/ppsweep_results.csv`, `tight55`, `relaxed9`, `bonus125`,
  `gap2step40` and `reuse28` all return proxy `0.9715059814960677` — identical to
  base to the last digit — and `ppsweep_selected.json` picks `base` with `{}`
  overrides. Five of the eight candidates are motion-relink knobs, which cannot
  do anything with the stage switched off. It costs ~8 × 250 s ≈ **33 min** of
  every relink-off run. This is the §8 runtime cut, but now *evidenced* rather
  than "untested but low risk" — for relink-off variants only.
- **Aggregate `mult` is not a node-count readout.** `metric2.aggregate` averages
  with `weight = tp + fp + fn` (metric2.py:91), which depends on the predictions,
  so any stage that moves coordinates re-weights the average and drifts aggregate
  `mult` by ~1e-5 with node counts untouched. The node-count exploit ABORT_RULES
  warns about shows up in **`ratio` (n_pred/n_est)**, which stayed at exactly
  0.8977 across all 24 linefit cells. Do not read a 1e-5 `mult` wobble as the
  exploit, and do not assert on it — `scripts/99` asserts on `ratio`.
