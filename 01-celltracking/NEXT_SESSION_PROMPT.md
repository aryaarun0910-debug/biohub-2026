# Session context prompt — Biohub Cell Tracking, resume from 2026-08-25

Copy everything below into a fresh session.

---

You are resuming the Biohub Cell Tracking Kaggle campaign. Working dir:
`c:/Users/aryaa/Documents/Biohub-CellTracking-2026`. Python: `.venv/Scripts/python.exe` with
`PYTHONIOENCODING=utf-8`. Kaggle: `python -m kaggle` (the `.exe` is blocked by Windows App Control).

**MISSION: TOP-3 OR NOTHING.** Top-3 = **0.953**. We are at **0.925, rank 207 / 2,693**. Deadline
**2026-09-29**. Leader z7777 0.962; only **six teams** are at >=0.947.

## READ FIRST, IN THIS ORDER
1. `CLAUDE.md` — the contract. Note rule 8: no GPU launch or submission is automatic.
2. `research/00-system/handoff.md` — the LIVE section at the top supersedes everything below it.
3. `research/07-outputs/submissions.md` — every submission carries a prediction recorded BEFORE the
   score. Keep doing this.
4. `research/06-knowledge-system/experimental-records.md` — read the **2026-08-25 entries**; that day
   overturned several standing beliefs.

## THE ONE THING THAT MATTERS MOST

**The deployed pipeline's division Jaccard ceiling is structurally ZERO.** Both association stages in
the built notebook are bare `linear_sum_assignment` calls — a bijection forbids out-degree 2, so a
division is not rare, it is INEXPRESSIBLE. Measured: divJ ceiling **0.0000 at out-degree<=1 vs 0.2320
at out-degree<=2**, i.e. **+0.0232 of SCORE is structurally unreachable**, for the price of 29 edges
out of 81,055. Independently confirmed in public (Kaggle discussion/733877, "40 FN out of 40").

**But naive relaxation is HARMFUL: blanket out-degree<=2 scored -0.0081 end-to-end.** The oracle needs
a SELECTOR, not a relaxation. hengck23 (discussion/726924) names the architecture: *keep* the
bijection, add divisions as a learned STAGE 2 (a classifier on appearance change + longer-track cues).
A CC0 corpus of **165,267 labelled divisions** exists (discussion/732103, ~540x the competition's
supervision) — pretraining only, its division rate is inflated 4.07% vs 0.26% and embryo heterogeneity
is not modelled.

## MEASURED FACTS — do not re-derive these

- **Metric:** `score = adj_edge_jaccard + 0.1 * division_jaccard`;
  `adj_edge_jaccard = max(0, edge_jaccard * (1 - 0.1 * total_node_ratio))`;
  `total_node_ratio = (N_pred - N_est)/N_est`. **`node_recall` appears NOWHERE in the score.** The
  multiplier is clamped below at 0 and **UNCAPPED ABOVE 1** — we currently under-produce (pooled ratio
  **-0.0937**) and collect a ~0.94% BONUS. Adding nodes ERODES that bonus. `N_est` is GT metadata,
  **unreadable at test time**, so count-budgeting is not a lever.
- **`edge recall = P(both endpoints detected) x CLA`, NOT `node_recall^2 x CLA`** — endpoint detection
  is correlated across frames, so the square law understates. Measured f0 `0.9816 x 0.9632 = 0.9455`,
  f1 `0.8533 x 0.9263 = 0.7904`.
- **The lever ranking FLIPS by fold.** Perfecting nodes: **+0.018 (f0) / +0.136 (f1)**. Perfecting
  linking: **+0.036 (f0) / +0.063 (f1)**. Pooled, nodes ~1.55x — **essentially all headroom is in the
  fold-1 (6bba) embryo**. Always report both directions separately; pooling hid a full inversion.
- **Root cause of linking failure: node localisation noise is 84-89% of the inter-frame motion
  signal.** An oracle replacing ONLY the endpoint coordinates with GT, decoy pool held fixed, moves
  "true parent is nearest" from **3.44% -> 68.39%**. The linker loses to noisy coordinates, not to a
  crowded pool.
- **A static distance prior is BACKWARDS.** Mis-linked GT edges have LARGER true displacement (median
  2.334 vs 1.817 um; p95 8.22 vs 4.89) and the wrong parent is the NEARER one in 91.5% of ranking
  errors. The right form is a MOTION-displaced prior (Linajea centres its ball on the extrapolated
  position).
- **The nominator emits exactly ONE candidate parent per target** (in-degree histogram
  `{1: 2,162,040}`); 13.42% of targets get zero. Source out-degree `{1: 1,795,844, 2: 165,148,
  3: 10,911, 4: 715, 5: 59, 6: 2}` = **176,835** sources at out-degree >=2, all destroyed by the relink.
- **Transfer law, revised:** divfix 0.000, p8 -0.003, **P9 +0.010**. Division changes transfer when
  the mechanism is COMPLETE (radii 8/11/10 + mid-track parent + mutual-nearest-orphan + t+2 divergence).
- **We appear to LEAD rank 30 on the edge term.** mikelou1 is 0.937 with self-reported adj_edge 0.898
  (divJ ~0.39); ours is ~0.92 with divJ ~0.02. **p15 measures this directly.**

## CLOSED — do not reopen without a new mechanism
`ILP_APPEARANCE_WEIGHT` (0.0 is the tuned public consensus; code default 0.1) · HOCT (measured
under-performing a tuned ILP at ~45 min/movie) · `ILP_DIVISION_WEIGHT` (0.3/1.0/2.0/3.0 all scored
0.915 on someone's LB) · **Trackastra-as-PRUNER** (scored from cached graphs: f0 -0.187, f1 -0.084;
its 14.7/55.2/39.2% rejection rates cannot clear the 58.88% break-even deletion precision) ·
abstention/deletion generally · NMS and pool-kernel tuning (0 GT collisions) · sub-voxel refine
(quantisation is only ~8% of localisation variance) · global coordinate-bias correction (f0 net
negative) · detection/ILP parameter probing (someone's ladder: all 0.908) · Edge Top-K / feature-TTA
(measured 0.885-0.886, actively harmful).

**NOT closed:** Trackastra **OWNING** re-association with its background class live — a different
mechanism from the pruning variant that was tested.

## IN FLIGHT
- **p15 fork-free probe** (ref 55768476) — `summarise()` drops the division term when a submission has
  no forks, so its score IS our pure adj_edge; divJ follows by subtraction. **This decides the lane.**
  Caveat: it lost 578 edges / 230 nodes beyond its 406 divisions (short-track filter reacts), so the
  subtraction is approximate.
- **p16 det 0.90** (55768483) and **p17 det 0.94** (55769398) — read as a PAIR for slope. Break-even is
  **0.0023 / 0.0015** of raw edge Jaccard respectively, because adding nodes erodes the under-production
  bonus.
- **p4 detpeak export, folds 0 and 1** — buys the ENTIRE `[0.5,1.0]` detection curve offline, zero
  slots. Replay harness ready: `scripts/win_bet/detpeak_curve.py`.

## NEXT ACTIONS, ranked
1. **Temporal position smoothing** — inference-only, CPU, no slot. ~37% of localisation error is
   independent per frame (corr ~0.61); averaging along a tracklet and relinking should cut displacement
   noise rms 2.44 -> ~1.9 um. Kill it if the nearest-parent oracle does not move off 3.44%.
2. **Division stage 2** — the +0.0232 ceiling; needs a selector, not looser gates.
3. **p4's curve** picks a detection threshold without a slot per point.
4. Public code worth mining: `mahdadshakiba/biohub-code` (a learned `MitosisNet` division classifier
   plus a sibling ranker; their measured +0.0051 CPU-only, and a fork-ranking fix worth +0.0074 because
   GT daughter separation has an interior mode at **10.6 um** while the shipped score is monotone in
   distance) and `shaminkhawar/n18b-linker-train-dense` (parental softmax with an explicit null token,
   no `linear_sum_assignment`).

## DISCIPLINE THAT IS EARNING ITS KEEP
- **Verify every premise at `file:line` before building on it.** Five failed this way in one cycle,
  including two of my own computations.
- **Calibrate any derived quantity against an independently known value before trusting it.** The
  coordinate-convention bug was caught only because correctly-linked GT displacement had to come out at
  1.82 um. Atlas/pre-ILP coords are FULL-RES `(z,y,x)` and take `(1.625, 0.40625, 0.40625)`; the
  detpeak export uses the MODEL grid and is isotropic 1.625. **Two of three distance analyses started
  wrong.**
- **A negative claim built on an empty grep is not evidence.** ripgrep `-E` means `--encoding`; a
  malformed flag produced empty output that reached two reports as verified negatives. `_evidence/` and
  `artifacts/` are git-ignored and skipped silently by default.
- **Things are already on disk.** Three times this cycle we concluded we lacked something that was
  sitting there: xiaoleilian's weights (6 days), the out-degree roll-up, Trackastra on both folds.
- **Record a prediction BEFORE every score, and promote the stated low band to the central estimate.**
  P9's band (central 0.919-0.925) HIT at 0.925 — the first in seven — while the naive modal estimate
  (0.926-0.928) was still optimistic.
- **Score-titled search sees only 9.5% of the public corpus** (68 of 719 kernels). Every mechanism-
  bearing notebook we found had NO number in its title. Search by mechanism, votes, and recency.
- **No team at >=0.947 has ever posted method content.** All six team names and nine usernames were
  grepped across all 75 forum threads. Stop hunting it.

## COMPUTE AND OPERATIONS
Kaggle only — Colab is declined (host decision). Deterministically **T4x2, CC 7.5**, so fp16 +
GradScaler is correct and the DataParallel/autocast trap is guaranteed on every run (`torch.autocast`
is thread-local and does not reach replica threads). **Kaggle permits a MAXIMUM OF 2 CONCURRENT GPU
BATCH SESSIONS** — pushes must be pipelined. ~5 submissions/day. Kernel work goes through
`scripts/core/kaggle_factory.py` (build -> verify -> push -> status -> fetch -> audit -> submitcmd);
the generated submitcmd names `kaggle.exe`, so substitute `python -m kaggle`. Watch for SLUG
DIVERGENCE on push — the factory reports it and the spec must be updated or status/fetch break.

## VERIFY
```
.\.venv\Scripts\python.exe -m pytest -q                      # 769 passed, 0 failures
.\.venv\Scripts\python.exe scripts\core\validate_research_tree.py   # 65 area docs
.\.venv\Scripts\python.exe scripts\core\claims_table.py --check     # 53 claims
```
Never `git add -A`; stage explicit paths. Never touch `.claude/settings.json` (dirty before this work,
deliberately untouched). Committed through **3534386** on `master`.

## KNOWN WEAKNESSES IN THE MACHINE ITSELF
- **The defect ledger gates almost nothing** — 34 of 41 specs match zero rules, including P9.
  ~~Its `spec_name_globs` should default the substrate-independent rules to `["*"]`.~~
  **CORRECTED 2026-08-25 — that fix does not exist and would be harmful.**
  `scripts/core/kaggle_factory.py:295` **already** reads
  `globs = scope.get("spec_name_globs", ["*"])`; the default is not the problem. All 8 rules narrow
  their own scope explicitly, and 7 of them target the two training specs (`h1r_edge_s5*`,
  `h1r_det_s1*`). Widening them to `["*"]` would fail nearly every deploy spec: `DG-002` and `DG-004`
  are `require`-shaped against training-only code, and `DG-006` demands a training dataset.
  **The real gap is missing coverage, not globbing** — no rule has ever been written for the
  deploy/post-processing substrate. Writing one is a data addition, not a one-line change.
- **A whole GPU session bought nothing and looked healthy doing it.** The p4 fold-0 detpeak export
  returned an empty `detpeaks/`; its hooks were passed through the parent's `builtins` while the
  predictor runs in a `subprocess`. Both call sites were `is not None`-guarded, so the failure was
  silent in both branches. Repaired and falsified both directions (see handoff). **Second instance of
  the cross-context-state class**, after DataParallel/autocast. Treat "does state cross a process,
  thread, or replica boundary?" as a standing build-time question.
- **A live instance of an ungated defect shipped:** `p8_loosefilter` sets
  `DEEPCENTER_SAFE_DIV_VETO=1` while no spec pins `DEEPCENTER_SAFE_DIV_THRESHOLD` (silent 0.12
  default) — a **third, unrecorded confound** in p8's -0.003.
- **The claims table is 9 days stale with no 0.925 row**, so every current number sits outside the
  machine-checked path. 11 research docs still assert 0.915.
