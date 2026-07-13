# Research Journal — Biohub Cell Tracking During Development (Kaggle 2026)

A manuscript-style, continuously updated research log (Stanford-style): standard
sections for the scientific argument, plus a dated **Research Log** recording the
iterative process, decisions, and dead-ends. Figures/animations live in
`reports/figures/`. This document is committed and pushed on every change.

**Authors:** aryaarun0910-debug (PI), Claude (implementation), Codex (intelligence/red-team).
**Repo:** https://github.com/aryaarun0910-debug/Biohub-CellTracking-2026 (private)

---

## Abstract

We aim to win the Kaggle *Cell Tracking During Development* competition: detect cell
centroids in 3D+time zebrafish light-sheet (DaXi) microscopy, link them across time, and
reconstruct lineages including divisions. Scoring is
`weighted_avg(adjusted_edge_Jaccard) + 0.1·division_Jaccard`, with per-timepoint optimal
bipartite node matching within 7 µm. The training set is only **two embryos** (199 crops),
and labels are **~1–2% sparse** point annotations; the hidden test is a disjoint embryo.

Our central finding, established by an exact-metric edge-error taxonomy over **all 199 crops**
on a faithful reproduction of the public 0.842 "V3" pipeline, is that **77–80% of missed edges
are genuine non-detections** (no candidate within 7 µm), while assignment (≤2%) and linking
errors are minor. Independently, reverse-engineering the evaluator shows the field's shared structural
weakness: matching is decided *before* edges are judged, so a spatially-closer duplicate can
steal a ground-truth match from the proposal carrying the correct trajectory. These converge
on a single thesis: **over-propose internally, then select one representative per cell by
matching-aware, track-conditioned arbitration, and recover missed cells along strong tracks.**
Target: **≥0.88 on the private leaderboard** (current public leader 0.875).

## 1. Introduction

The competition (host: CZ Biohub Royer Lab) provides Zarr volumes `(T,Z,Y,X)` uint16, typical
`(100,64,256,256)`, voxel scale `(z,y,x)=(1.625,0.40625,0.40625) µm` — Z is 4× coarser
(**Fig. 6**, XZ projection). Ground truth is sparse GEFF point graphs; the count adjustment
uses a metadata `estimated_number_of_nodes`. Because only two embryo identities exist in
training and the test embryo is disjoint, the dominant risk is **domain overfitting**, not
raw model capacity.

**Fig. 1** quantifies the sparsity: the median crop annotates ~1–2% of its estimated cells.
**Fig. 6** (XY projection) makes it visceral — hundreds of visible nuclei, ~12 annotated.
[Time-lapse animation](../figures/anim_xy_timelapse.gif) shows cell motion and density over
time; [rotating 3D lineage](../figures/anim_3d_rotate.gif) shows the annotated tracks.

## 2. Literature Review (synthesis)

Full deep-dives: `reports/research/` (methods, Royer ecosystem, competitive methods).

- **Host baseline** (`royerlab/kaggle-cell-tracking-competition`): a single jointly-trained
  temporal-U-Net + cross-attention edge transformer; detection loss = BCE with single-voxel
  positives and `neg_weight=0.01`; edge loss = focal on `softmax(dim=0)` (each target picks one
  parent; a parent may spawn two → divisions). Two latent levers: a *no-op* division up-weight
  and a checkpoint metric (`acc·recall`) that is not the leaderboard metric.
- **Detection from sparse points**: Spotiflow (Nat. Methods 2025) multiscale heatmaps +
  stereographic-flow subpixel offsets — but it does *not* ignore unlabeled voxels, so the
  **Linajea soft-mask** (loss weight 1 within a nucleus radius, ~0.01–1e-6 elsewhere) is the
  key adaptation; PAC-MAP proximity-adjusted Gaussians; Hirsch–Kainmueller CPV auxiliary loss.
- **Association / global solve**: Ultrack (multi-hypothesis + ILP, CBC offline), motile (SCIP
  offline, division-aware, `fit_weights` sSVM), Trackastra (transformer linker, 3D `ctc` model).
- **Same-domain external assets** (rules-permitted): `unet-daxi.pt`/`unet-simview.pt`, ZSNS001–005
  embryos + dense track CSVs; the exact-voxel-scale 522-frame Ultrack embryo is the strongest
  provenance lead (identity unproven).

## 3. Methodology

- **Operating model.** Local machine (MX350, 2 GB) does dev, full-data EDA, CPU DoG sweeps, and
  exact scoring on all 199 crops; heavy GPU training/inference runs on Kaggle/cloud.
- **Exact metric, reimplemented.** `src/biotrack/metric_numpy.py` reproduces the organizer's
  edge term (per-timepoint optimal bipartite match maximizing `1/(1+d)` within 7 µm; directed
  matched-edge mask; adjusted Jaccard with count penalty). **Validated exactly** vs the
  tracksdata implementation on 20 perturbation cases *and* 7 hand-built adversarial
  assignment-conflict cases (`tests/`). Runs without tracksdata → usable on Kaggle. The
  division term is deliberately *not* reimplemented; division-sensitive gates use the
  authoritative tracksdata harness (`biotrack.metric`).
- **Validation design.** Leave-one-embryo-out is the honesty gate; to avoid overfitting the two
  folds, selection uses nested grouped crop/time splits *within* an embryo with bootstrap CIs,
  then evaluates the frozen config cross-embryo.
- **V3 reproduction.** `notebooks/kaggle_submit/kaggle_dog_infer.py` implements the verified public 0.842
  recipe (multiscale DoG, NMS 4 µm, XY offset, (3,9,9) refinement, two-pass velocity-aware
  Hungarian, min-track-length-4).
- **Edge-error taxonomy.** Each missed GT edge is attributed to *no-candidate* (undetected),
  *lost-assignment* (a candidate existed within 7 µm but lost the match), or *association*
  (both endpoints matched, no predicted edge).

## 4. Results

- **Data characterization.** 199 crops from 2 embryos (44b6: 71, 6bba: 128); 128,883 annotated
  edges, 151 divisions. Sparsity ~1–2% (**Fig. 1**); edge volume is embryo-imbalanced ~5:1
  (**Fig. 2**) — the 6bba fold dominates weighted scoring.
  All figures below are generated from committed CSV artifacts
  (`reports/inventory/v3_taxonomy.csv`, `norm_ablation.csv`), not hard-coded.
- **Recall drives score.** Across **all 199 crops**, adjusted edge Jaccard rises steeply with
  node recall (**Fig. 3**); crops with recall <0.65 score <0.5, crops >0.95 score >0.9.
- **Why edges are missed (the decisive result).** 3-way taxonomy over **all 199 crops**:
  no-candidate **80% (44b6) / 77% (6bba)**, lost-assignment 2% / 0%, association 18% / 23%
  (**Fig. 4**). The bottleneck is genuine under-detection (V3 count ratio <1, recall 0.82 / 0.88),
  *not* assignment stealing — validating investment in recall (learned residual detector /
  redetection), with arbitration as the guard once we over-propose. V3 reproduction adjusted J is
  **0.632 (44b6) / 0.756 (6bba)** locally; the gap to the public ~0.842 is this recall deficit.
- **Normalization has no effect (not the gap).** Per-frame (V3) vs precomputed per-volume
  quantiles gave **identical results on all 30 tested crops** (max |delta| = 0.000; **Fig. 5**) —
  because DoG uses a *relative* threshold (a fraction of the max response), which is scale-invariant
  to linear normalization. Normalization is therefore not a lever for the DoG detector.

## 5. Discussion

The evaluator couples detection and linking through its match-first design; the field optimizes
them separately, so their errors are correlated and structural. Our measured taxonomy shows the
proximate cause here is under-detection, but naïvely scaling a detector risks the assignment-
stealing failure mode once recall rises. The plan therefore pairs **high-recall internal
over-proposal** with **matching-aware, track-conditioned arbitration** and **gap redetection**,
followed by tissue-flow linking, count calibration, a learned residual detector (Spotiflow +
soft-mask + CPV, gated on the taxonomy), a global solve only if residual conflicts remain, and
sparse precision-gated divisions. Selection is private-set-first: min(fold) with bootstrap CIs,
diversified final submissions, and submissions only for locally-qualified hypotheses. External
public data is permitted for priors/pretraining; exact test-crop→public-label transfer awaits a
private organizer clarification; score-probing is forbidden. Full plan:
`reports/EXECUTION_PLAN.md`.

---

## Research Log

### 2026-07-02
- Established exact numpy edge metric; validated 20/20 vs tracksdata, then 7/7 adversarial
  (assignment stealing, crowded, anisotropic, ties, duplicates, empty). Divisions kept on the
  tracksdata harness by design.
- Reproduced public V3 (0.842 recipe) with the corrected same-cell conflict definition (7 µm is
  the evaluator gate, **not** a dedup radius — real nuclei coexist within it).
- **Corrected dataset ground truth: 199 crops, not 129** (the API manifest was incomplete);
  rebuilt inventory + leave-one-embryo-out splits from local pairs.
- **3-way edge taxonomy** (corrected from 2-way): no-candidate dominant (70–79%), lost-assignment
  ~0–1% → detection recall is the true bottleneck; Spotiflow investment justified.
- Normalization A/B: per-frame ≈ precomputed → not the V3 gap.
- Repo trustworthiness repair (Codex read-only audit): fixed `score_submission` omitted-dataset
  inflation (+tests), DAXI anisotropic NMS (axial nuclei were being collapsed; +test), locked
  deps, removed an unsafe `rm -rf data/train` permission, first intentional commit.
- Set up private GitHub repo; established figures + animations pipeline (this journal).

### 2026-07-03 (evidence hardening — Codex 2nd audit)
- Replaced hard-coded figure values with machine-readable CSV artifacts: taxonomy runner now
  parallel (7 workers) and writes `v3_taxonomy.csv`; norm ablation writes `norm_ablation.csv`.
- **Full-199 taxonomy** (was 20 crops): no-candidate 80%/77%, lost-assignment 2%/0%, assoc 18%/23%;
  V3 adjJ 0.632/0.756, recall 0.82/0.88 — the no-candidate dominance holds robustly.
- **Norm ablation** (n=30, both directions): per-frame == precomputed EXACTLY (max|delta|=0) —
  DoG relative-threshold is scale-invariant; "statistically identical" claim upgraded to
  mechanistic identity.
- Figures 3/4/5 regenerated exclusively from those CSVs.
- Fixed `fetch.py` to build the COMPLETE 199-embryo manifest (was truncated to 129); documented the
  canonical whole-competition bulk download.
- Portable dependency pin (organizer pkg via git URL, not local path) + `requirements.txt` + GitHub
  Actions CI running the synthetic-data metric tests on every push.
- Evidence hardening + Phase-1 design (see later entries).

### 2026-07-03 (Phase 1 launch)
- V3 anchor submitted to Kaggle: hit a real env gap (Kaggle image has NO zarr/numcodecs but HAS
  tensorstore) -> portable volume reader (zarr local / tensorstore Kaggle); kernel
  `aryaarun07/biohub-v3-anchor` v2 produces a valid 206,786-row submission.
- Built the Phase-1 ablation harness (`run_phase1_ablation.py`, config registry, per-fold min-adjJ
  gate) + candidate cache (`src/biotrack/cache.py`).
- **Step 1a finding (over-proposal frontier): NMS radius, not DoG threshold, is the recall lever.**
  Lowering the response threshold leaves recall/count nearly flat (missed cells produce NO DoG local
  max, not a sub-threshold one). Loosening NMS 3.2->1.0 um lifts recall (hard 44b6 crop 0.65->0.78 at
  ratio 1.33; 6bba 0.86->0.90). Finer scales HURT (add count, not recall). BUT DoG recall plateaus
  ~0.78-0.90 < the 0.95 target -> over-proposal is a partial lever; the last stretch needs
  track-conditioned redetection and/or a learned residual detector (Spotiflow). propose.py default
  set to loose NMS (1.0 um), 2 scales.
- **Next:** basin-based same-cell conflict sets + guardrail test (step 4), then brightest-rep ->
  link -> arbitrate -> relink (step 5); redetection (step 6) is now clearly needed for recall.

### 2026-07-03 (sprint to 0.90 locked; compute = cloud A100)
- Codex sprint plan adopted (reports/SPRINT_2026-07-03.md). Honest odds: 0.90 ~10-15%; robust
  0.86-0.88 ~55-65%. Primary lever = Spotiflow-3D residual fine-tune (union DoG -> arbitration).
- Fixed Codex P0/P1: propose density now in ISO isotropic space; real candidate merge audit added
  (GT too sparse for close pairs -> diameter-cap structural guarantee is the safeguard). 13 tests pass.
- Built the FIRST KILL GATE: scripts/spotiflow_zeroshot_screen.py — does pretrained synth_3d/smfish_3d
  recover >=8% of DoG-missed GT nodes zero-shot? Runs on A100. If no -> pivot to DAXI union+redetection.
- V3 full baseline (harness): min-fold adjJ 0.632. Anchor 54290725 still PENDING.

### 2026-07-03 (anchor score + local<->public calibration)
- **V3 anchor scored 0.807 public** (submission 54290725). Real public V3 = 0.842 -> we carry a
  0.035 reproduction gap (measured recall deficit; we under-detect vs real V3) = a cheap classical win.
- **Calibration (1 point):** our local both-embryo edge-weighted adjJ ~0.737 (44b6 0.632/19.8k edges +
  6bba 0.756/109k edges) -> public 0.807. The HIDDEN TEST EMBRYO IS EASIER than our hard 44b6 fold
  (0.807 > 0.632). Supports Codex's condition for 0.90 (test resembles the easier end).
- Strategic update: immediate cheapest win = close the V3 0.035 gap via recall (loose-NMS over-proposal
  + arbitration), THEN stack the learned detector. Test-easier read modestly improves the odds.
- Built the Kaggle T4 zero-shot screen notebook (notebooks/kaggle_spotiflow_screen/, self-contained:
  reads volumes + GT geffs via tensorstore, full-frame complement recall). Runs free on Kaggle T4.

### 2026-07-03 (screen error + strategic pivot to CPU arbitration)
- Kaggle Spotiflow screen ERRORED (empty log; likely pip-install/numpy-2.0 conflict). Hardened the
  notebook to surface pip + import + traceback. Ran on P100 (Kaggle default for enable_gpu); P100 is
  fine for inference screening; T4x2 (for concurrent fold TRAINING) is a UI accelerator selection.
- **Strategic pivot:** the anchor score revealed a CERTAIN free win - close the 0.807->0.842 V3 gap
  via CPU arbitration (over-proposal + same-cell dedup + link). Prioritizing step-5 arbitration over
  blind remote-debugging the marginal, out-of-domain Spotiflow zero-shot screen.

### 2026-07-03 (lateral sweep brief for Codex)
- Drafted reports/codex_lateral_sweep_2026-07-03.md: multi-agent ORTHOGONAL research sweep (8 parallel
  tracks: cross-domain tracking, metric/format exploits, provenance/leakage, newest sparse/PU detection,
  organizer footprint, Kaggle meta, domain-generalization, wildcards) + synthesis. Hard filter: skip all
  saturated competitive intel; only NEW/actionable/high-EV/cited edges. Runs parallel to execution.

### 2026-07-03 (lateral sweep results)
- Codex multi-agent lateral sweep -> reports/codex_lateral_sweep_results_2026-07-03.md. Top edges:
  (1) **motion-compensated TRACK-BEFORE-DETECT** - integrate low-threshold DoG/PSF response along
  3-7 frame motion-consistent paths BEFORE thresholding; attacks the no-candidate bottleneck; CPU,
  falsifiable in HOURS; +0.005-0.020. (2) **March-22 exact-scale embryo + zoo/Zebrafish 122MB track
  bundle (~11.85M pts)** = dense same-modality pretraining IF provenance/alignment gate passes;
  +0.010-0.040. (3) **NIS3D (3.3GB dense zebrafish, CC-BY) + nnPU** pretraining fallback; +0.005-0.030.
  Rules explicitly allow public external data for model development.
- **QUARANTINED evaluator defect**: a division-scoring loophole (distant unmatched fork in a weakly-
  connected component qualifies a GT division, dodges division-FP). Prize/DQ risk -> DO NOT SUBMIT
  without written host clearance. We do not build strategy around it. (Endorsed.)
- Immediate plan: build the track-before-detect falsification diagnostic (top bet, CPU, hours).

### 2026-07-03 (track-before-detect: NO-GO, redirect to learned detector)
- Built + ran the TBD falsification diagnostic (scripts/tbd_diagnostic.py) on 6 worst-recall crops.
  Temporal DoG-response integration recovered only 5.5% of DoG-missed GT above the null 95th pct
  (single-frame 3%) - Codex gate was >=20%. **NO-GO for naive track-before-detect** (killed in ~20min).
- Sanity check (mapping correct): missed-GT DoG response median 0.096 vs detected 0.137 vs null 0.021.
  Missed cells are SUBTHRESHOLD (real signal, above background median) but DoG has a heavy bright-
  background tail (texture/membranes) that overlaps dim cells -> temporal summing lifts both, no clean
  separation. => the recall lever is a LEARNED detector (NIS3D/Spotiflow separates dim cells from
  texture where DoG can't) + over-proposal/arbitration; NOT temporal DoG integration.
- Redirect: prioritize the learned-detector path (NIS3D dense pretraining / Spotiflow) + the free
  op_bright arbitration. March-22 provenance gate remains the conditional jackpot.

### 2026-07-03 (op_bright arbitration BEATS V3 - first real Phase-1 gain)
- op_bright (over-propose loose-NMS 1.0 -> same-cell dedup complete-linkage+diameter-cap -> V3 link,
  brightest representative) vs V3 on same 20 crops: MIN-FOLD adjJ 0.677 -> 0.720 (+0.043, ~9x the
  +0.005 gate). Hard fold 44b6 recall 0.827 -> 0.889; 6bba flat (0.782->0.783). adjJ rose so count
  penalty stayed controlled. Free CPU win; plausibly ~0.84 LB if the ~+0.07 local<->LB offset holds.
- Confirming on all 199 (background). Next: full arbitration (continuity + matching-stability scoring,
  not just brightest) = upside; then learned detector stacks on top.

### 2026-07-03 (op_bright full-199: min-fold +0.052 but edge-weighted +0.004)
- op_bright ALL 199: 44b6 0.632->0.684 (+0.052, recall 0.816->0.882), 6bba 0.756->0.751 (-0.005).
  MIN-FOLD 0.632->0.684 (+0.052) = big ROBUSTNESS win on the hard embryo. BUT edge-weighted both-fold
  (the LB proxy) only 0.737->0.741 (+0.004) because dense 6bba (5x edge volume) slightly regressed:
  crude "brightest" over-proposal adds FP/count on already-well-detected crops. LB likely ~0.81 (from 0.807).
- Runtime OK: ~25s/dense crop -> ~1.4h for hidden test (<12h). op_bright kernel pushed (aryaarun07/biohub-op-bright).
- FIX = continuity/matching-stability arbitration (not brightest) to keep recall gain without the 6bba
  regression -> should lift the weighted score. Then learned detector stacks. Submitting op_bright as a
  calibration point + robustness win (expect ~0.81, not a leap).

### 2026-07-03 (6bba regression diagnosed = over-detection, not arbitration)
- Per-crop: 6bba regressions correlate with COUNT RATIO, not recall. Worst crops: op_bright ratio
  1.06-1.36 vs v3 0.88-1.06; recall delta ~0. 6bba mean ratio 0.88->1.06 (crossed 1.0). Sparse 44b6
  (ratio<1) is helped by over-proposal; dense 6bba (already detected) is over-shot -> count penalty+FP.
- FIX = COUNT CALIBRATION (Phase 2), not continuity scoring: over-propose where recall low, restrain
  where already-detected (target ratio 0.95-1.05). Should convert +0.052 min-fold into edge-weighted gain.
- op_bright still worth submitting (robustness win on hard embryo + 2nd local<->LB calibration point).

### 2026-07-03 (Codex correction accepted; temporal smoothing = the V11 classical lever)
- Accepted Codex corrections: op_bright ~0.812 is HONEST (not misleading); count-cal is an uncalibrated
  operating point (unproven, est_n hidden at inference, ~0.815 ceiling even if perfect); CLASSICAL CEILING
  IS 0.854 (public Rule-Based V11 via temporal coordinate smoothing w=0.7), NOT 0.842. My "learned detector
  required above 0.842" was FALSE.
- Ported temporal coordinate smoothing (smooth each linked node toward its edge-neighbours' mean, w=0.7).
  20-crop ablation: V3+smooth improves BOTH folds (44b6 0.677->0.703, 6bba 0.782->0.787). op_bright+smooth
  STACKS: 44b6 0.677->0.734 (+0.057), 6bba 0.782->0.791 (+0.009). Both folds up incl. dominant 6bba ->
  real EDGE-WEIGHTED gain (~+0.016 on subset) unlike op_bright alone. Confirming on 199.
- Revised path (Codex): op_bright ~0.81 -> +count-cal ~0.815-0.825 -> +V11 smoothing ~0.84-0.855 ->
  learned detector over validated 0.85 base ~0.865-0.885. 0.88+ prob ~15-25%, rises after reproducing 0.854.
- Plan: run classical (smoothing/calibration, CPU) + learned (dense-external pretraining, GPU) CONCURRENTLY.

### 2026-07-03 (session handoff documented)
- Wrote HANDOFF.md (repo root) = single "start here for a new session" doc: current scores/baselines,
  open async threads (op_bright submission 54301967 pending; op_bright_smooth 199 running; Spotiflow
  screen broken), corrected path to 0.88, DO-NOT (quarantined evaluator defect), how-to-run commands,
  key code map, environment, immediate next steps. README points to it. User submitted op_bright.

### 2026-07-06 (learned-stack pivot; oracle; Codex + opus research swarm; reorder to ASSOCIATION)
- Pivoted from classical DoG (median 0.807) to the organizer LEARNED stack (vendored tracking_cellmot). Built+
  validated train->predict->score kernels (heavy debugging: T4x2 pin, PYTHONPATH, glob nesting, ILP OOM, model-
  specific det_threshold). Trained both embryo-held-out folds; greedy OOF fold0 0.656 / fold1 0.559 (min 0.559).
- ILP proven the lever locally: fold-1 0.559->~0.67 (+0.11), division-FP catastrophe crushed. Full-length retrain
  (45ep) OVERFIT (< 30ep on held-out) -> more training on 2 embryos is negative; bottleneck = GENERALIZATION.
- Candidate-edge ORACLE (Codex P0): max edge-J = 0.935 (44b6) / 0.885 (6bba) = min-fold detection ceiling.
  0.94 is DEAD (need 0.88 edge with margin). But +0.18 pure-linking headroom (0.67->0.885) = the win.
- Leaderboard: 1065 teams, top 0.968 (leakage outlier), pack ~0.90, everyone forked one "LB897" baseline ->
  PRIVATE (disjoint embryo) shuffle is wide open = generalization contest = our OOF.
- Codex red-team + opus 6-lane research swarm -> reordered plan (reports/WIN_PLAN.md "RESEARCH-BACKED
  EXECUTION"): 4 levers (metric-aligned Dinkelbach ILP [Nowozin 2014]; path-consistency association TTA
  [Lu CVPR24]; track-conditioned redetection to raise the 0.885 cap; two-stage fork-posterior divisions) each
  with a cheap CPU-local kill-gate. Honest private target 0.84-0.90. Cleanup: removed 82GB redundant data zip.

### 2026-07-12 (research brain swarm; verified scorer; breadth win-bet; E0b wrapper extraction)
- Built a persistent research brain (reports/research/brain/): 6-lane interdisciplinary + red-team swarm
  (01 Kaggle intel, 02 track-before-detect, 03 assoc/MOT/OT/TTA, 04 lineage/division/PU/DA, 05 metric
  decision-theory, 06 red-team) + SYNTHESIS.md + ROADMAP.md. Committed 6994935, 0cbff83.
- RED-TEAM KILLER POINT (evidence-based): both-fold-OOF gate ALREADY produced a false positive (fusion
  +0.035 OOF -> -0.024 hidden). ~2 effective samples -> run <=2 confirmatory tests/round vs the RIGHT
  baseline. KILL: from-scratch 4D Lagrangian field, naive TBD as main bet, standalone OT/FGW.
- VERIFIED scorer facts (read vendored tracking_cellmot; METRIC_SEMANTICS_VERIFIED.md): off-annotation
  edges are FREE (FP only if endpoint matches an annotated GT node with an edge); N_pred=graph.num_nodes(),
  N_est is a NODE count (hidden at inference -> analysis-only anchor, not deployable); division FP only on
  annotated CONTINUING cells; division term worth up to +0.1 (not +0.025). Corrections logged after Codex
  review (optimal 1:1 bipartite matching; added nodes still hurt via assignment-stealing).
- WIN-BET = training BREADTH (the generalization lever, not count-pruning). Acquired 4 dense Zebrahub embryo
  lineages ZSNS001/003/004/005 (~1.8GB, data/external/zebrahub/, SHA256'd) = 3x embryo diversity, dense not
  sparse. Scale-free association scorer (velocity/density-normalized, NO appearance). CROSS-EMBRYO transfer
  (train 003/004/005 -> held-out 001, different coord scale): pooled AUC 0.997; on AMBIGUOUS links (nearest
  != truth) per-source top-1 0.2935 vs NN 0.0, MRR 0.581 vs 0.430, cand recall@6 0.96. Thesis alive: model
  recovers ~29% of links pure-distance linking gets wrong. Pooled AUC is easy-negative noise; per-source
  top-1/MRR is the metric (Codex reframe).
- E0b (BLOCKER) = reproduce the pure-0.889 wrapper on fold OOF. Extracted the LB897 post-processing into
  src/biotrack/wrapper.py (config 91-140 + fns 957-1887 sliced; pure defaults = motion-relink/safe-div/
  linefit/gap-close ON, gap2/div-geom OFF, min-track 6; NO trackastra fusion = the 0.865 reject; gap-refine
  disabled, minor). Driver scripts/win_bet/e0_replay.py.
- E0b DONE (2026-07-13, 199 crops, reports/inventory/e0_wrapper_oof.txt): STRONG but deployment-INEXACT
  wrapper OOF adj-J = 44b6 0.7601 / 6bba 0.6450 (min-fold 0.6456). div-J ~0 (44b6 0 TP/91 FP; 6bba 4 TP/582
  FP/121 FN; safe-divisions net-neutral on sparse OOF). HARD-NUMBER CONFIRMATION of the red-team: this
  EXCEEDS the prior "best" Trackastra-fusion OOF 0.6948/0.6044 on BOTH folds -> the fusion +0.035/+0.036
  "gain" was an artifact of comparing to the weak greedy 0.656/0.559; vs the REAL wrapper the fusion
  REGRESSES (-0.065/-0.041), why it lost hidden (0.865 vs 0.889). Trackastra replacement conclusively dead.
- E0b NOT yet authoritative (2 mismatches vs saved 0.889 run_stats.csv): (1) min-track-len — submission used
  7 (effective 6 only on 6bba_05b6850b public-test movie); E0b used 6 for all crops. (2) image synthetic-gap
  refinement ON in submission (gap_refined 144/988/72/931); OFF in E0b. -> E0c.
- NEXT = E0c parity FIRST: min-len 7 uniform on OOF (do NOT generalize the 6bba_05b6850b=6 exception);
  enable gap-refine reading data/train/<crop>.zarr; validate wrapper diagnostics vs run_stats.csv on the 4
  test movies (nodes/edges, gap_refined, short-track removals, min_len_effective, safe_divisions) -> require
  exact/explained parity -> re-score OOF = the authoritative baseline.
- THEN Phase B: export the FULL pre-assignment candidate pool from motion_relink_edges (src+tgt, distance,
  motion residual, edge_prob, tight/relaxed membership, local density+rank, selected?), NOT just final edges
  (they lack negatives). Labels via scorer optimal pred-node->GT matching: positive only if BOTH endpoints
  match GT nodes AND the GT edge exists (nearest-GT = false supervision). Compare breadth model vs the
  wrapper's COMPOSITE decision (motion+distance+prob+constraints), not edge_prob alone.

### 2026-07-13 (E0c hybrid pipeline; deployment-exactness; several failures corrected)
- CORRECTION (review): E0b (0.7601/0.6450) was STRONG but deployment-INEXACT — used min-track-len 6 for all
  crops + disabled image gap-refine, whereas the 0.889 submission used min-len 7 (effective 6 only on the
  6bba_05b6850b public-test movie) with synthetic-gap refine ON. Not authoritative until fixed.
- E0c PARITY: extracted-wrapper diagnostics vs saved run_stats.csv on the 4 test movies = EXACT on all 15
  load-bearing fields (nodes/edges, gap_refined 144/988/72/931, short_track_min_len_effective 7/7/6/7,
  safe_divisions, motion_relink, prune). Wrapper is a bit-faithful reproduction. scripts/win_bet/e0c_parity.py.
- ARCHITECTURE (review): old e0_replay coupled slow wrapper + slow scoring in one serial loop = the real
  defect. Refactored to a hybrid: e0c_run.py (Stage 1) runs the exact wrapper ONCE (gap-refine ON, min-len 7
  uniform), caches per crop (atomic/resumable/sharded): post-wrapper graph + FULL pre-assignment candidate
  surface (Phase-B asset, same pass) + manifest (config hash, git, diagnostics, explicit status). e0c_score.py
  (Stages 2-4): numpy edge diagnostic + authoritative edge+division in a ProcessPool + numpy-vs-authoritative
  edge parity across the population. Verified: smoke gap-refine 345 refined/0 failed, 30k cands; score 2 crops
  0.8132/0.7497; Stage-4 numpy parity 0/2 mismatch, max adj-J diff 0.00e+00. Full baseline pending (cache).
- FAILURES / dead-ends this session (kept for continuity):
  * Duplicate-process incident: user "closed" the machine but it did NOT fully shut down -> pre-reboot E0c
    survived; my post-reboot relaunch made a 2nd, both tee-writing the SAME file (corrupted), and TaskStop left
    orphaned `for S in 0 1` bash loops RESPAWNING workers faster than I killed them -> 4+ python at 101% CPU,
    ~0 crops/min. Fix: kill by .venv exec-path + launch DIRECT python (no for-loop = no respawn). Lesson: shard
    E0c as independent direct processes writing distinct files, never a tee'd for-loop.
  * Kill commands self-terminated (exit 255): kill pattern 'e0_replay' appeared in the killing shell's own
    command line -> killed its own shell. Fix: exclude via Stop-Process/CimInstance guard or match exec-path.
  * gap-refine on OOF ~1-2 min/crop (reads data/train frames) -> 3-10h for 199. Nearly shipped an ablation
    (gap-refine OFF) as the baseline = WRONG (not deployment-exact). Correct answer = caching pipeline (run
    exact once, score fast). metric_numpy proposed as authoritative -> REJECTED (edge-only, no divisions).
  * Legacy reorg over-moved run_phase1_ablation/run_v3_taxonomy -> pytest FAILED (test_metric_parity imports
    them) -> reverted; remaining 12-file legacy move re-verified (50 passed). legacy/README.md.
- E0c FROZEN (2026-07-13): Stage-1 cache completed 199/199 (0 failed); e0c_score.py authoritative =
  44b6 0.7595 / 6bba 0.6484 (min-fold 0.6490). Beats E0b min-fold (0.6456) because min-len 7 helped 6bba;
  44b6 ~flat (0.7595 vs 0.7601). Division-J ~0 (0/93/26; 4/582/121). THIS is the authoritative baseline;
  gate all deltas vs it. numpy-vs-authoritative EDGE parity across 199: 194 exact, 5 MISMATCH (max adj-J
  0.055) -> metric_numpy confirmed fast-diagnostic-ONLY, not authoritative (1-crop parity did NOT hold on
  the population, as reviewer predicted). Phase-B candidate surface cached same pass: 6.15M rows (4.82M
  selected) in artifacts/kaggle/e0c_cache/candidates/.
- NEXT: Phase-B competition-transfer gate on the cached candidate surface — label candidates via scorer
  pred->GT matching, compare breadth model vs wrapper COMPOSITE (cost/selected), gate on exact graph-level
  gain over E0c (0.7595/0.6484) both folds, not AUC.

### 2026-07-13 (Phase-B labeling fix + controlled breadth gate — breadth helps but fails both-fold)
- LABELING BUG FOUND+FIXED (d4f3737): v1 treated sparse unannotated candidates as negatives -> reported
  candidate recall 0.78% / wrapper "precision" 0.6% (MEANINGLESS, diluted by 99% unannotated sources).
  v2 = three-state supervision (positive / reliable_negative / unlabeled), ranking negatives ONLY within
  answerable source groups. Corrected 1-crop: candidate coverage 0.78% -> 96%, wrapper top-1 ~98%.
  label_candidates.py -> candidates_labeled_v2/.
- Competition-only ranker control (d68c8c9, LGBMRanker w/ edge_prob+cost features): 44b6 +0.0039, 6bba
  -0.0035 vs wrapper -> FAILS both-fold -> two-embryo training insufficient (6bba regression). Not submitted.
- CONTROLLED BREADTH GATE (phaseb_gate_breadth.py, geometry-only scale-free features raw_rel/motion_rel/
  min-ratios/rank_frac/log_n_cand, NO edge_prob/cost -> transferable; answerable-group top-1/MRR):
  * Zebrahub candidates generated (gen_zebrahub_candidates.py): 737,262 answerable groups from ZSNS003/4/5.
  * 44b6: wrapper 0.9635 | comp-geom 0.9645 (+0.0010) | zebrahub+comp 0.9645 (+0.0010, +0.0000 vs comp).
  * 6bba: wrapper 0.9319 | comp-geom 0.9301 (-0.0017) | zebrahub+comp 0.9313 (-0.0005, +0.0012 vs comp).
- FINDINGS: (a) external breadth MEASURABLY helps the weak fold (6bba comp-only -0.0017 -> breadth -0.0005;
  +0.0012 over comp-only) -> reverse-fold-asymmetry thesis DIRECTIONALLY CONFIRMED. (b) BUT geometry-only
  (even +breadth) does NOT beat wrapper on BOTH folds (6bba still -0.0005) -> FAILS gate, NOT submittable as
  a wholesale ranker replacement. (c) edge_prob is strong but NON-transferable (comp-only w/ edge_prob 6bba
  -0.0035 < geometry-only -0.0017 -> overfits comp detector distribution). (d) wrapper composite already
  ~93-96% top-1 -> candidate-RERANKING headroom nearly exhausted.
- IMPLICATION: blanket geometry replacement won't win. Reranking is not the lever. NEXT (per plan):
  selective uncertainty-gated repair + EXACT graph-level scoring vs E0c (top-1 parity does not preclude
  targeted graph gains, but expect marginal); AND weigh bigger levers — divisions (+0.1 metric ceiling,
  currently ~0 on OOF), endpoint recovery. Leaderboard: #1 0.970 / #2 0.968 (extreme outliers, investigate
  but do NOT distort private validation to imitate) / #3 0.941 / #4 0.910.

### 2026-07-13 (Commander barbell plan; Track-B DIVISION ORACLE GATE PASSES)
- Commander two-track plan: Track A = fast exact-graph experiments -> ONE calibration submission (selective
  repair; no-fork ablation; existing-fork oracle; division-proposal oracle). Track B = breadth division
  reconstruction, GATED on division-proposal oracle showing >=+0.01 exact composite on BOTH folds; else kill
  division as main bet and redirect to candidate-gen/redetection. Kaggle = sparse calibration instrument only
  (one submission per system family; never promote on ranking AUC/top-1; exact graph gains first).
- DIVISION-PROPOSAL ORACLE (phaseb_division_oracle.py, read-only, from labeled_v2 source/target_gt_id +
  GT divisions): reachable = GT div whose matched pred parent has >=2 of the parent's GT children as
  candidate targets.
  * 44b6: 26 GT div, 7 reachable (26.9%), cur div-J 0.0000 -> oracle 0.2692 => optimistic d_composite +0.0269
  * 6bba: 125 GT div, 26 reachable (20.8%), cur div-J 0.0057 -> oracle 0.2080 => optimistic d_composite +0.0202
  GATE PASSES both folds (>=+0.01). Track B = breadth division posterior is GO. CAVEAT: optimistic ceiling
  (assumes 0 FP, ignores single-parent assignment-stealing); exact < this. Reachability only ~20-27% -> a
  higher-recall fork proposal generator can raise the ceiling further. NEXT: exact fork oracle (construct+
  score oracle-fork graphs) to confirm; then no-fork ablation + selective-repair exact tests (Track A);
  then build the Zebrahub-trained PU division posterior (Track B).

### 2026-07-13 (Kaggle T4 GPU lane ACTIVATED; division model transfers cross-embryo)
- Two-machine split live: LOCAL = exact evidence/gates; KAGGLE T4 = representation training; LB = sparse calib.
- Data-prep LOCAL (divevents_extract.py): scale-free mother-centric (3,5,3) division fork-events from Zebrahub
  (positive=real division; negative=genuine non-dividing continuation + nearby fake 2nd daughter). Balanced,
  57,690 events across 4 embryos. Uploaded as Kaggle dataset aryaarun07/biohub-divevents-zebrahub (SHA256'd).
  NOTE Kaggle CLI Windows path bug on upload -> must run kaggle datasets/kernels from PowerShell (native paths).
- GPU Job A (notebooks/kaggle_divmodel, aryaarun07/biohub-divmodel-t4): compact conv-temporal fork classifier,
  LOEO, T4-PINNED (machine_shape NvidiaTeslaT4 -> confirmed ran on Tesla T4), internet off, deterministic,
  checkpoint/resume, runtime-guarded, exports metrics/config/log/env. v1/v2 ran empty (pushed before dataset
  ready / hardcoded mount path) -> fixed load() to glob /kaggle/input recursively -> v3 trained.
- RESULT (LOEO cross-embryo): PR-AUC ZSNS001 0.886 / ZSNS003 0.814 / ZSNS004 0.795 / ZSNS005 0.908
  (mean 0.851); recall@P0.9 0.640 / 0.013 / 0.112 / 0.717 (mean 0.370). => divisions ARE learnable+transferable
  from scale-free trajectory geometry alone (no appearance/embryo-id) -> Track-B thesis validated. BUT
  high-precision recall is UNEVEN across embryos (0.01-0.72) -> cross-embryo calibration (the critical metric)
  not yet there. Strong v1 seed, NOT yet a deployable posterior.
- NEXT: improve high-P recall/calibration (Job B self-supervised motion pretraining; per-embryo calibration;
  richer features/temporal model; 3 seeds after transfer gate); PU fine-tune on competition; joint fork
  selection accounting for edge-J/count/lineage/assignment-stealing; then exact graph-level score vs E0c.
  Track A still owed: selective-repair + no-fork + existing-fork exact tests; one calibration submission.

### 2026-07-13 (Job-A leak fixed; Track-A no-fork ablation = forks net-neutral)
- COMMANDER CORRECTIONS accepted: (1) GPU lane is IDLE (status COMPLETE != running) -- only claim "running"
  when kaggle kernels status == RUNNING. (2) Job A LEAKED -- run_fold saved the best HELD-OUT PR-AUC epoch =
  model selection on the held-out embryo -> 0.85 LOEO is OPTIMISTIC, infra-smoke only, NOT validated. FIXED:
  fixed-epoch (25), held-out evaluated exactly once at final epoch, no held-out selection (biohub_divmodel.py).
  Full Job-B upgrade still owed: inner-validation-embryo selection, 3 fixed seeds mean+/-std, natural-prevalence
  eval (recall@P0.90/0.95/0.98 + calib), genuine self-supervised motion corpus (masked recon/next-step/f-b
  consistency/motion contrast on millions of ordinary trajectories), deployment-matched HARD negatives.
  (3) balanced dataset != deployment prevalence (divisions rare) -> balanced PR-AUC misleading.
- TRACK-A NO-FORK ABLATION (phaseb_ablation.py, exact authoritative vs E0c): 44b6 0.7595 (-0.0000),
  6bba 0.6482 (~-0.0008 vs composite 0.6490). => wrapper forks are NET-NEUTRAL (FP forks don't hurt the
  composite; the ~4 TP barely help). No free gain from suppressing existing forks -> existing-fork oracle ~0;
  ALL division value is in ADDING new correct forks (proposal-oracle +0.02 headroom = the Track-B posterior).
- HONEST SCORE ESTIMATE: current 0.889 (~#358). Divisions landing (~30-50% of +0.02 oracle) + repair ->
  ~0.895-0.910 public (into the ~0.90 pack, near #4=0.910). Everything+breadth -> ~0.91-0.92. #3=0.941 needs
  a lever we lack; #1/#2=0.968/0.970 likely public-split leakage. Private: top-20..50 if divisions transfer,
  NOT clearly #1 -> more work needed. OOF->LB calibration submission will measure actual transfer.
- STILL OWED Track A: selective-repair exact test; exact division-proposal oracle (graph construction+score);
  then one low-risk calibration submission.

### 2026-07-13 (Codex verdict adopted; bracketed-miss oracle DEFLATES the completion lever)
- CODEX corrections accepted: central estimate LOWERED (next submission ~0.890-0.900, division system
  ~0.895-0.910, 0.91+ a stretch, 0.94+ from divisions/reranking VERY UNLIKELY); REMOVE private-rank estimate
  (unknowable); division oracle is REACHABILITY not SCORE; 0.970/0.968 outliers "possibly" not "almost
  certainly" public-overfit. Codex's proposed winning bet = track-conditioned hypothesis COMPLETION + joint
  lineage selection (generate missing nodes/forks from raw evidence, select jointly) -- not a better classifier.
- BRACKETED-MISS ENDPOINT ORACLE (phaseb_bracketed_oracle.py, read-only, E0c->GT match):
  * 44b6: node recall 0.9543, 923 missed -> bracketed 147 (16%) / continuation 367 (40%) / isolated 409 (44%);
    edges restorable by bracketed completion <=295 (<=1.49% of GT edges).
  * 6bba: node recall 0.8776, 13843 missed -> bracketed 402 (3%) / continuation 2258 (16%) / isolated 11183
    (81%); restorable <=804 (<=0.74% of GT edges).
- FINDING: the SAFE bracketed gap-completion mode is SMALL (<=1.5%/0.7% edge-recall ceiling). The missed-node
  mass is mostly ISOLATED (44%/81%) = needs de-novo detection (near-ceiling, hard). So hypothesis-completion's
  safe form does NOT open a large score source; reinforces Codex ~0.91 honest ceiling. Divisions (<=+0.02) +
  bracketed (~1%) are not a credible path to 0.94+. Isolated-miss detection is the true (hard) frontier.
- EXECUTION (Codex order, in progress): [done] no-fork ablation (neutral), reachability div oracle (+0.02),
  bracketed-miss oracle (small). [next] exact topology-aware division oracle (Oracle B constrained reachability
  + Oracle C exact score with conflict resolution/parent-stealing -> the number that decides GPU division
  quota); selective-repair exact test; corrected self-supervised Job-B pretraining on T4 (nested LOEO, window/
  lineage grouping, daughter-symmetry, deployment-matched hard negatives, natural prevalence) in parallel.

### 2026-07-13 (DIVISION ORACLE C — exact gate = AMBER; value is in joint conflict resolution)
- Oracle C (phaseb_oracle_c.py): edges-only edits (no nodes -> count penalty unchanged), scorer-consistent
  E0c->GT match, global constraints (<=1 parent/daughter, <=2 children/mother, distinct daughters, daughter-
  competition), authoritative scoring. GT-informed OFFLINE upper bound, written to GT_ORACLE_do_not_submit/
  (never packaged). Baseline sanity PASSES: reproduces E0c exactly (44b6 0.7595, 6bba 0.6490).
- RESULTS vs E0c composite:
  * add_only (conflict-free): 44b6 +0.0045 / 6bba +0.0037 (div-J 0.042/0.041; only 5/29 forks added, 21/... skipped)
  * add_replace (conflict-aware, TRUE upper bound): 44b6 +0.0182 / 6bba +0.0138 (div-J 0.168/0.131; 20/93 forks
    forced; 15 steals on 44b6; edge-J ALSO +0.0014/+0.0013 -> forced daughter edges are TPs).
- GATE = AMBER (44b6 GREEN >=+0.015; 6bba +0.0138 in AMBER band; not GREEN both folds). Per commander: one
  SHORT leak-free GPU seed; proceed only if the learned posterior captures a substantial fraction at high precision.
- KEY STRUCTURAL INSIGHT: add_replace (+0.018) is ~4x add_only (+0.004) -> the division value lives in CONFLICT
  RESOLUTION (removing daughters' wrong parents / daughter competition), not division classification alone. A
  standalone fork classifier caps near +0.004 (RED); the deployable win REQUIRES a JOINT fork+edge selector with
  assignment-stealing resolution (confirms Codex). Also the exact upper bound (+0.018/+0.014) is ~6x lower than
  the optimistic reachability estimate (+0.027/+0.020) -> reachability oracle massively overstated.
- DECISION: AMBER -> prepare corrected self-supervised Job-B data/code (no major GPU spend); one short leak-free
  division seed; the real build is the joint selector. Isolated-miss signal-separability gate still owed (moonshot).

### 2026-07-13 (AMBER execution: division seed FAILS high-precision gate; isolated-miss Stage-1 pending)
- Corrected division seed v4 (leak-free nested inner-val, daughter-swap invariant [verified max|diff|=0.0],
  single seed, T4, confirmed RUNNING then COMPLETE): LOEO PR-AUC 0.675/0.739/0.671/0.777 (mean 0.715, down
  from leaked v3 0.851); recall@P0.9 0.0005/0.0018/0.000/0.176 (mean 0.045, down from leaked 0.37). The leak
  was doing heavy lifting. HONEST cross-embryo high-precision recall is ~0 on 3/4 embryos -> model can RANK
  divisions but CANNOT identify at high precision. On EASY nearby-fake negatives (deployment hard negatives
  worse). => FAILS the AMBER continuation gate (high-precision fork operation). Oracle-C's +0.018 needs
  near-perfect fork ID + conflict resolution; a posterior with ~0 high-P recall cannot deliver it. Per
  commander decision tree: STOP the division GPU program after this seed unless isolated-detection rescues.
- Isolated-miss Stage-1 reachability (isolated_stage1.py, DAXI low-thresh proposals vs isolated misses)
  RUNNING locally -> result pending; decides the de-novo detection route.
