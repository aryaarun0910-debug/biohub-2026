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
The historical time-lapse and rotating 3D lineage animations show cell motion, density,
and annotated tracks; they were removed from the lean tree and remain available from Git
tag `pre-lean-2026-07-30`.

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

- ISOLATED-MISS STAGE-1 REACHABILITY (isolated_stage1.py, DAXI unet low-thresh 0.20 proposals, NO GT coords):
  6bba_05db0fb1 27/27 reached, 44b6_0113de3b 3/3 -> 30/30 = 100%. Isolated misses ARE visible to a
  low-threshold detector (de-novo detection NOT dead at proposal gen). CAVEAT: 100% is driven by MASSIVE
  over-proposal ~41k (6bba) / 87k (44b6) peaks/frame (~40x real cell count) -> the difficulty is entirely
  Stages 2-4 (temporal tracklet filtering + PU discrimination at high precision + exact graph test protecting
  count/edge-J). Precision bar brutal (40k candidates -> must add ~0 FP). Division model's ~0 high-P recall is
  a warning. Isolated lever ALIVE at Stage 1, UNDECIDED at Stages 2-4 (the real gate).
- AMBER OUTCOME (both tracks executed): DIVISION seed FAILS high-precision gate -> division learned-posterior
  program essentially dead (Oracle-C +0.018 unreachable without near-perfect fork ID). ISOLATED detection
  passes Stage-1 reachability but faces a severe Stages-2-4 discrimination gate. Per commander decision tree:
  not "both pass"; isolated is the only credible-but-hard route left. NEXT: build isolated Stages 2-4 (de-novo
  temporal tracklet construction + natural-prevalence PU discrimination + exact graph insertion/scoring);
  gate = >=20% isolated recovery + min-fold composite >=+0.005, no count/regime collapse. If it fails, current
  architecture's winning-scale routes are falsified -> ship best disciplined ~0.90-0.91 calibration system and
  begin a genuinely different architecture search.

### 2026-07-14 (ISOLATED GATE FAILS -> both winning routes falsified; disciplined-floor pivot)
- Isolated Stages 2-4 gate (isolated_gate.py; DAXI cache 600k peaks/crop over 100 frames; tracklets len>=3;
  oracle+conf selection; exact insertion). (NOTE: reported dvs-E0c is a subset-vs-full-fold mis-comparison =
  INVALID; the recovery + node-cost signals below are valid and decisive.)
  * ORACLE (perfect selection = ceiling): 44b6 recovered 0/3 (0%), 0 nodes added; 6bba 5/57 (8.8%) but 3280
    nodes added to recover 5 misses. CONF (realistic): +1.4M/599k nodes added -> composite collapse (0.07/0.31),
    ~0% recovery.
- CONCLUSION (robust): isolated misses are reachable PER-FRAME (Stage-1 100%) but do NOT form coherent
  multi-frame tracklets (oracle recovery 0-9% << 20% gate) -- they are barely-detectable transient cells; a
  single-frame peak exists but no consistent 3-frame track, and a lone recovered node restores NO edge (its GT
  neighbours are also missed) while still paying the count penalty. Count penalty is catastrophic (low-thresh
  proposals add thousands-to-millions of nodes). Isolated de-novo detection FAILS the gate on both recovery and
  node-cost. => Both winning-scale routes (division learned-posterior + isolated detection) are FALSIFIED.
- DECISION-TREE OUTCOME: neither passes -> the detect->link->repair architecture family is exhausted at
  ~0.90-0.91. Disciplined floor = the frozen E0c wrapper (public 0.889, already deployed). No new deployable
  improvement has cleared the both-fold exact gate. NEXT: (1) rigorous final synthesis of every lever's exact
  measured ceiling; (2) evidence-based directions for a GENUINELY DIFFERENT architecture (the only remaining
  route to >0.91): learned motion-compensated temporal EVIDENCE INTEGRATION (proper track-before-detect that
  ACCUMULATES weak sub-threshold response along motion-compensated candidate tubes BEFORE declaring nodes --
  the simple DAXI-single-frame+tracklet version failed precisely because it does not integrate weak temporal
  evidence), trained on breadth; else accept ~0.90 and optimize private generalization.

### 2026-07-14 (moonshot falsification pilot: oracle-motion DAXI accumulation -> NEGATIVE lean)
- Ran the cheap falsification `FINAL_SYNTHESIS_2026-07-14.md` prescribed before committing to the temporal-
  integration moonshot: does motion-compensated accumulation of raw DAXI response along an oracle (GT-lineage)
  tube reveal signal at isolated-miss locations that single-frame thresholding misses? (`daxi_accumulation_v2.py`,
  oracle-motion upper bound; paired hard controls = cached low-thresh DAXI peaks given the same displacement
  tube; no new model.)
- RESULTS: 44b6_0113de3b pairs=3 (underpowered, no fold conclusion); 6bba_05db0fb1 pairs=55 (adequately
  sampled) single_AUC=0.0283 static_AUC=0.2225 GT_motion_AUC=0.1921 -> motion accumulation LOWER than static
  by 0.0304 AUC on the only well-powered crop. Confound noted: control matching used cached full-volume
  response while evaluation used patch inference, so absolute single-frame AUCs are not clean (this blocks a
  fully confident kill, not just a soft one).
- DECISION (per the doc's own go/no-go rule): the pilot leans NO -- oracle GT-motion accumulation did not
  show the hoped-for advantage over static accumulation on the powered crop. Per commander: do not open the
  temporal-integration GPU program on this evidence. Before any further moonshot spend, a cleaner rerun would
  need patch-response-matched controls and more 44b6 crops; absent that, treat the moonshot as UNSUPPORTED
  (not proven impossible, but no measured mechanism). Combined with the division and isolated falsifications,
  ALL THREE candidate winning-scale levers are now negative or falsified. Disciplined floor stands: frozen
  E0c wrapper, public 0.889, min-fold OOF 0.6490. No further architecture work is scheduled unless a cleaner
  falsification or a new idea clears its own cheap gate first.
### 2026-07-19 (official metric patch integrated; E0c rescore running)
- Organizer patch `075fc5f` pinned in both requirements files and checked out in the local editable vendor.
  The scoring epoch changes: nonconsecutive edges are dropped, duplicate predictions mapping to one GT edge
  are collapsed, out-degree is capped at two, and division credit now requires local directed fork topology.
- Added an off-volume weak-component hub regression; updated the fast numpy edge gate to mirror the patched
  edge canonicalization. Verification: project focused metric tests 11/11 pass; official patched division
  suites 58/58 pass.
- Full 199-crop E0c patched rescore launched with four CPU workers. Historical division-dependent results are
  quarantined until rescored. NEXT: record patched E0c aggregate, then rescore Oracle C/no-fork; in parallel,
  port the clean public 0.903 wrapper for one bilateral OOF gate and prepare the corrected temporal-signal job.

### 2026-07-19 (scoring epoch closed; CPU + Kaggle attrition campaign active)
- Patched E0c authoritative full-OOF result is stable: 44b6 adj-J/composite 0.7595/0.7595,
  division TP/FP/FN 0/93/26; 6bba adj-J/composite 0.6484/0.6490, division 4/582/121.
  The official patch does not move the disciplined floor.
- Patched no-fork: 44b6 0.7595 (neutral), 6bba 0.6482 (-0.0002). Indiscriminate fork
  suppression is rejected.
- Patched Oracle C reproduced the prior ceiling exactly: add-only +0.0045/+0.0037;
  add-replace +0.0182/+0.0138. Therefore topology-aware conflict resolution is necessary,
  while the already-failed high-precision division classifier remains the realizability blocker.
- Ported the clean-public-0.903 density-adaptive gap rule behind a default-OFF flag, added
  a regression test, a separate cache, and generic `e0c_score.py --cache`. One-crop smoke
  passed; full 199-crop OOF launched as four resumable CPU shards. This is a wrapper-only
  0.990-detection gate; exact 0.970 detection regeneration is GPU-gated on bilateral gain.
- Created private Kaggle dataset `aryaarun07/biohub-daxi-weights-diagnostic` from the
  previously verified DAXI TorchScript weight. Launched T4 kernel
  `aryaarun07/biohub-daxi-cache-20` v2, internet OFF, on 10 fixed count-stratified crops per
  embryo. V1 failed immediately because the base image lacked zarr; v2 uses the existing
  offline support wheels and is running. Output feeds the corrected, patch-response-matched
  temporal signal gate; it is diagnostic and not automatically submission-eligible.

### 2026-07-19 (Kaggle DAXI cache v2 failure fixed; v3 running)
- V2 successfully installed offline zarr wheels, mounted the private DAXI weight and
  competition train data, and selected a Tesla T4. It then failed on the first tile:
  NumPy percentile arithmetic promoted the normalized volume to float64, conflicting
  with float32 TorchScript weights. No experiment result was produced.
- Fixed normalization by explicitly casting percentile scalars, the clipped frame, and
  each contiguous tile to float32; added a runtime tensor-dtype assertion. Local preflight
  confirms dtype=float32, contiguous=True, finite=True.
- Pushed Kaggle kernel `aryaarun07/biohub-daxi-cache-20` version 3 and verified status
  `RUNNING`. The fixed 20-crop design, T4, private weight dataset, and internet-OFF policy
  are unchanged.

### 2026-07-20 (clean-wrapper gate fails; DAXI cache completes)
- Clean-public-0.903 wrapper-only OOF completed 199/199 manifests under config
  `6e2f4ca93fe73098`, zero failures. Patched authoritative score: 44b6 0.7614
  (+0.0019 vs E0c 0.7595); 6bba 0.6457 (-0.0033 vs E0c 0.6490). Division-J was
  0.0000/0.0044. Verdict: FAIL bilateral and >=+0.005 min-fold gates; kill this
  branch and do not spend GPU on deployment-exact 0.970 detection regeneration.
- Kaggle DAXI cache v3 status COMPLETE: 20/20 crop outputs plus manifest downloaded;
  T4 compute 1667.7 seconds. Cache contains 11,639,313 low-threshold peaks; 18/20 crops
  reach the 600,000 peak cap, reinforcing the high-precision selection risk. This is
  not a temporal-signal result. NEXT: corrected patch-response-matched AUC/effect analysis
  and crop bootstrap, followed by exact graph insertion only if bilateral signal exists.

### 2026-07-20 (temporal v3 built; power audit forces full 44b6 expansion)
- Implemented `scripts/win_bet/daxi_accumulation_v3.py` with a preregistered five-frame
  primary endpoint: `(spatial-flow - static)_positive - (spatial-flow - static)_control`.
  Controls are same-crop/frame DAXI peaks matched on depth, predicted density, and final
  centre response through the identical float32 patch-inference path. Cached responses
  do not determine final matching or scores.
- Deployable motion is a spatial kNN median field from unambiguous frozen-E0c edges,
  falling back to per-timepoint then global robust median. Frame-only flow and 3/7-frame
  windows are diagnostics. GT motion is an oracle ceiling. Thresholds cross-fit between
  embryo families; bootstrap unit is crop. Unit gates: 3/3 pass (spatial locality,
  crop-weighted bootstrap, frozen cross-fit threshold).
- The staged 20-crop audit is NOT adequately powered on 44b6: 59 isolated misses, only
  44 eligible five-frame events across 6 informative crops (vs 835 eligible/10 crops on
  6bba). Per preregistration, inference was not launched on this imbalance.
- Full frozen-OOF 44b6 audit: 71 crops, 41 informative crops, 409 isolated misses,
  311 eligible five-frame events (291 seven-frame), 303 selected under the per-crop cap.
  This clears the >=200-event power target. NEXT: Kaggle cache all 71 44b6 crops, then
  run v3 against that cache plus the existing 10-crop 6bba cache.

### 2026-07-20 (powered temporal accumulation gate -> KILL)
- Kaggle kernel `aryaarun07/biohub-daxi-cache-44-all` completed all 71/71 44b6 crops
  without errors in ~83.7 minutes T4 time. Downloaded 71 NPZ outputs plus manifest;
  the full output tree is 41,921,783 bytes.
- Ran `daxi_accumulation_v3.py` locally for 6613.1 seconds using 8 Torch threads and
  10,000 crop-bootstrap draws, combining the 71-crop 44b6 cache with the existing
  10-crop 6bba cache. Result artifact SHA256:
  `2F943E1B1A487056B063AC06B0CC3817F59BA297D825F9E017267D9680620D51`.
- Powered coverage: 44b6 = 311 eligible five-frame events/41 informative crops,
  303 selected, 268 paired across 40 crops; 6bba = 835 eligible/10 crops, 264 selected,
  202 paired. The pre-run power targets were met.
- Preregistered primary spatial-flow contrast: 44b6 mean +0.0000609, crop-bootstrap
  95% CI [-0.0001288,+0.0003302]; 6bba mean -0.0168708,
  CI [-0.0456935,-0.0001467]. Spatial motion is null on 44b6 and significantly worse
  than static on 6bba.
- Oracle GT-motion ceiling also fails: 44b6 +0.0003829,
  CI [-0.0000701,+0.0011551]; 6bba -0.0090423,
  CI [-0.0406812,+0.0113874]. Neither family has a positive lower bound.
- Frozen cross-fit thresholds fail both directions: train-44b6/test-6bba gives
  TPR 0.401 at control FPR 0.129; train-6bba/test-44b6 gives TPR 0.000 at FPR 0.0037.
- DECISION: `signal_gate_pass=false`; KILL temporal accumulation and do not unlock the
  compact affinity GPU model. This closes the v2 control-path confound at adequate power
  and exhausts the last identified winning-scale mechanism. Frozen E0c remains the floor.

### 2026-07-20 (live-board reset; clean-frontier attack launched)
- Authenticated Kaggle refresh: public leader 0.982; twenty teams score >=0.964; our
  submitted E0c remains 0.889. Public notebooks now openly add negative-time,
  out-of-volume hub/fork graphs to exploit division scoring. This branch stays
  quarantined because it conflicts with the patched organizer metric and prize audit.
- Pulled the latest public clean kernel. The clean lineage has a confirmed 0.909 v120
  result; v122 is a clean 0.910+ candidate coupling detector threshold 0.9690 with ILP
  appearance/disappearance 0.0/1.5, min-track 6, and density-adaptive gap closing.
  This was not covered by our failed wrapper-only test on frozen 0.990 detections.
- Forked the exact v122 public artifact into private internet-off T4 kernel
  `aryaarun07/biohub-clean-v122-reproduction` v1; push succeeded and status RUNNING.
  On completion: audit and submit immediately if structurally clean.
- Replaced the exhausted roadmap with `reports/ATTACK_REGIME_2026-07-20.md`: coupled
  C0/C1 OOF regeneration, full-data model training/ensembling, external NIS3D/Zebrahub
  image pretraining, and legal gradient-free target robustness. E0c remains fallback.

### 2026-07-20 (clean v122 audited and submitted)
- `aryaarun07/biohub-clean-v122-reproduction` v1 completed successfully. Downloaded
  `submission.csv`: 237,023 rows = 120,633 nodes + 116,390 edges across four placeholder
  crops; SHA256 `4E36B4797C0F07FF7B3C55C8FD6C73C4E9EC5AF264C4B4F21828B1F97CA9D616`.
- Independent local audit: node t 0..99; z 0..63; y/x 0..254; zero missing edge
  endpoints; zero nonconsecutive edges; max out-degree 2; max in-degree 1. No exploit
  structure is present. The completed artifact therefore passed the attack regime's
  submission gate.
- Raw-file API submission returned HTTP 400 because this is now enforced as a Kaggle
  code competition. Submitting kernel `aryaarun07/biohub-clean-v122-reproduction`
  version 1 succeeded: submission ID `54854143`, status PENDING.
- Provenance correction: `run_stats.csv` says experiment tag
  `120_clean_ilp_disappearance_150`, but runtime configuration and end-of-run output
  verify detector threshold 0.9690/v122. Corrected the retained notebook source tag to
  `122_clean_precision_det09690_disappearance150`; this does not alter the submitted
  v1 artifact. NEXT: append public score, then run C0/C1 full OOF regeneration.

### 2026-07-28 (instrument audit: OOF<->public calibration is NOT POSSIBLE with existing artifacts)
- Ran the pre-registered OOF<->public rank audit over all 6 scored submissions
  (`scripts/win_bet/instrument_audit.py`; full write-up + limitations in
  `reports/INSTRUMENT_AUDIT_2026-07-28.md`). Availability tiers were ENFORCED, never estimated.
- PRIMARY RESULT: only **n=1** submission qualifies (post-patch scorer 075fc5f AND proven
  deployment parity) = E0c `54534923` (0.889, OOF 0.7595/0.6490). Rank correlation on
  qualifying data is NOT COMPUTABLE; the audit cannot be completed as specified. Disqualified:
  Trackastra-direct `54601594` (OOF pre-patch, division term changed), classical `54290725`/
  `54301967` (kernel never parity-checked vs the local 199-crop config), Trackastra-hint
  `54588144` (no OOF exists), v122 `54854143` (no OOF yet).
- VERIFIED COMPARABILITY before computing anything: `run_phase1_ablation.py::fold_summary`
  weights by `edge_tp+edge_fp+edge_fn`, identical to `tracking_cellmot.metrics.summarise`,
  so classical-era and E0c-era fold numbers ARE the same statistic.
- PUBLIC SCORES THEMSELVES MOVED: JOURNAL recorded `54290725` = 0.807 (2026-07-03); the Kaggle
  API now returns 0.815. The patch rescore DID shift a non-exploit submission, contrary to the
  host note. Any historically recorded public score must be refreshed from the API before use.
- DEGRADED DIAGNOSTIC (tier A+B, n=4, explicitly not a calibration): min_fold Spearman -0.400 /
  Kendall -0.333; edge_weighted -0.800/-0.667; arith and harmonic means -0.400; 6bba_only -0.600;
  44b6_only **+0.800/+0.667** (the only positive). Leave-one-out on min_fold swings from +0.500
  (drop op_bright) to -1.000 (drop E0c). Child-vs-parent: op_bright vs V3 d_public -0.074 vs
  d_minfold +0.052 = **INVERTED**; Trackastra-direct vs E0c anchor = AGREE.
- INTERPRETATION (this does NOT convict min-fold): the sign is driven by a classical-vs-learned
  family confound (classical = high OOF/low public, learned = the reverse); the result is
  controlled by one observation; and op_bright -- which drives every discordant pair -- has
  UNPROVEN deployment parity (predicted ~0.81, scored 0.741), so instrument failure and config
  drift are indistinguishable. `44b6_only=+0.800` is treated as noise at n=4.
- DECISION: keep LOEO min-fold as the private-safety gate; do NOT retune it to six public
  observations; do NOT promote any deployment-ranking statistic (including 44b6_only) on this
  evidence. The instrument is uncalibrated and no existing artifact can fix it -- the only route
  to a second qualifying observation is deployment-parity post-patch OOF at the coupled operating
  point. NEXT: C0/C1 coupled experiment (now doubly load-bearing: primary attack AND the sole
  source of calibration evidence).

### 2026-07-28 (coupled C0/C1 harness: two spec corrections, Arm A parity EXACT)
- Built the coupled decomposition harness (`scripts/win_bet/coupled_arms.py`,
  `coupled_replay.py`, `tests/test_coupled_arms.py`). Retained the exact v122 source at
  `notebooks/kaggle_clean_v122/` (SHA256 703A05E2...C7C2FBC, kernel
  `aryaarun07/biohub-clean-v122-reproduction v1`) -- closes the open Phase-0 retention item.
- CORRECTION 1 -- **E0c has no ILP stage.** `oof_clean/pred_geffs_split_{0,1}` was produced by
  `kaggle_predict_score.py` with `USE_ILP = False` at det 0.99, i.e. greedy selection
  (max_parents=1, max_children=2) inside `predict_video`. Arm A is therefore
  detector + greedy + wrapper. The preregistered B->C step was consequently split into
  B->B' (introducing ILP at default 0.1/0.1) and B'->C (survival costs 0.0/1.5) so the
  structural change is not reported as a cost-tuning delta.
- CORRECTION 2 -- **the clean-903 port is not the v122 wrapper.** Diffing the retained source
  against `clean903_wrapper_run.py::set_clean903_wrapper_config` found three drifted
  constants: PREFIX_DENSITY_BLEND (v122 0.0 vs port 0.20), SAFE_DIV_GLOBAL_FRAC_CAP
  (0.00375 vs 0.00385), MOTION_RELINK_LEARNED_BONUS (1.0 vs 0.75). The first is structural:
  v122's `frame_local_spacing` returns pure local kNN spacing and the notebook contains no
  PREFIX_DENSITY concept at all -- the blended transductive per-embryo prior is our own
  addition. Both promotion candidates (D, C0) now use the faithful retained-v122 wrapper.
  The earlier clean-903 kill stands for what it tested (frozen 0.990 detections, different
  upstream graph) but does NOT transfer to the true v122 promotion pipeline.
- Added diagnostic-only seventh arm **D-port** (identical cached 0.9690 detections and
  identical C1 ILP output as D; differs ONLY in those three constants) to quantify the
  bundled wrapper drift. Not a promotion candidate; must not delay D/C0.
- CACHE DESIGN (approved): edge logits CANNOT be shared across detector thresholds -- the
  edge head cross-attends over the whole node set (`SimpleNodeTransformer`,
  `nn.MultiheadAttention`) and `predict_video` applies `softmax(dim=0)` over the source
  axis, so both the attention context and the normaliser change with the node population.
  Caching UNet features instead is infeasible (~33 MB/frame fp16 -> ~670 GB for 199 crops
  vs 361 GB free). Correct boundary is the pre-graph `(coords, edges)` output: one exact
  GPU pass per threshold, all ILP/wrapper variants replayed on CPU. B/B'/C/D/D-port share
  one 0.9690 pass; C0 needs 0.96875; A reuses the existing artifact. ~200 MB total.
- PREFLIGHT: config hashes distinct for all 7 arms (no collisions); 11 regression
  assertions pass, including a SHA256 lock on the retained notebook so the three constants
  cannot silently drift back. **Arm A parity EXACT on both preflight crops**: node/edge
  diagnostics reproduce `e0c_run_smoke.txt` (28119->27338, 6847->6214) and the composite is
  bit-identical to the cached E0c artifact (|delta| < 1e-9) at 0.8132 (44b6_0113de3b) and
  0.7497 (6bba_05b6850b). ILP confirmed available locally (pyscipopt 6.2.1, ilpy 0.6.0).
- NEXT: Kaggle inference-cache kernel; verify 0.9690 peaks are a superset of 0.990 before
  edge inference; then the two full passes and the CPU replays. Phase 2 training stays
  blocked until C0/C1 is complete and recorded.

### 2026-07-28 (coupled preflight COMPLETE: caches produced, all 7 arms replayed)
- Kernel `aryaarun07/biohub-coupled-cache-preflight` v7 COMPLETE after six environment
  failures (v1 zarr; v2 tracksdata; v3 blanket wheel install clobbered image numpy/scipy;
  v4 only-if-missing kept an old polars; v5 package named `biohub_tracking` not
  `tracking_cellmot`; v6 reached full inference but died serialising via polars, whose
  compiled runtime is not loadable on the image). v7 serialises with numpy. No GPU was
  spent on real inference before v6; no submissions consumed.
- **DETECTOR REPRODUCTION IS EXACT.** Our det-0.990 + stock 4-view TTA yields 28,119 coords
  (44b6_0113de3b) and 6,847 (6bba_05b6850b) -- byte-equal to oof_clean/E0c's node counts.
  This validates the inference path end-to-end AND empirically settles that oof_clean used
  4-view TTA, not D4 (D4 gives 27,751 / 6,855).
- PREFLIGHT GATES ALL PASS: 12/12 caches hash-verified vs the kernel manifest; peak
  superset holds in every case (0.990 subset of 0.9690, and 0.9690 subset of 0.96875,
  0 missing); Arm A bit-identical to the E0c artifact (0.8132 / 0.7497); 7/7 config
  hashes distinct; D and C0 both use the faithful retained-v122 wrapper.
- PER-ARM (2 crops only -- NOT a verdict):
  A 0.8132/0.7497 | B 0.7967/0.7088 | B' 0.7334/0.7594 | C 0.6380/0.8226 |
  D 0.6374/0.8131 | D-port 0.6374/0.8131 | C0 0.6373/0.8131
- **D vs C0 is a NULL RESULT**: d_score +/-0.0000 on both crops (9 node difference on 44b6,
  0 on 6bba). The 0.96875 -> 0.9690 "coupled precision push" separating v120 from v122 is
  essentially a no-op at the detector level on this evidence.
- **D vs D-port is a NULL RESULT**: d_score +/-0.0000 (1-3 nodes). The clean-903 port's
  three drifted wrapper constants are immaterial, so the earlier clean-903 kill is NOT
  overturned by that drift. Wrapper-drift question closed; do not split the constants.
- DOMINANT LEVER is the ILP survival cost, not the detector or wrapper: B'->C moves
  -0.0954 (44b6) / +0.0633 (6bba); C->D (whole wrapper stage) moves only -0.0006 / -0.0096.
- **THE FOLDS DISAGREE VIOLENTLY**: A->D = -0.1758 (44b6) vs +0.0633 (6bba); 44b6 node
  recall collapses 0.9423 -> 0.7115 under disappearance 1.5. CAVEAT THAT DOMINATES
  EVERYTHING: 44b6_0113de3b carries only **50 GT edges** (52 GT nodes) against
  6bba_05b6850b's **845**. The 44b6 preflight crop is near-powerless; these deltas are
  pipeline validation, NOT a promotion signal. No gate is evaluated on 2 crops.
- MEASURED COSTS: GPU 21.1 min setup (one-time) + 2.7 min (44b6, 33.5k nodes) + 1.8 min
  (6bba, 7.3k) covering BOTH TTA schemes and all 4 edge passes. Cache 1.6 MB / 12 files.
  CPU replay 60-84 s per ILP arm at 28k nodes (peak RSS ~870 MB), 9-14 s at 6k.
  199-crop projection: median 20.0k nodes, mean 25.7k, p90 59.1k, max 84.2k.
- NEXT (awaiting review, NOT started): full 199-crop passes at 0.9690 and 0.96875.

### 2026-07-28 (full-population launch: C0, D-port and D4 TTA dropped on preflight nulls)
- RESOURCE DECISION recorded BEFORE launch. Three branches killed from full-population
  expansion on preflight evidence:
  * **C0 / det 0.96875** -- detector-level no-op vs 0.9690 (9 nodes on 44b6, 0 on 6bba;
    d_score +/-0.0000 both crops). The "coupled precision push" separating public v120 from
    v122 does not exist at the detector level.
  * **D-port** -- clean-903 wrapper drift immaterial (1-3 nodes, d_score +/-0.0000). The
    wrapper-drift question is closed and the earlier clean-903 kill is NOT overturned by it.
    The three constants will NOT be split into further arms.
  * **D4 TTA** -- not part of the parity-proven baseline and would confound the gate.
    Stock 4-view is retained precisely because it reproduces oof_clean node counts EXACTLY
    (28,119 / 6,847), so A and B differ only by detector threshold.
- LAUNCHED: 199-crop cache at det 0.9690, stock 4-view TTA, LOEO routing (44b6->split_0,
  6bba->split_1), two private internet-off T4 shards. Shards built by greedy balancing on
  E0c node counts, not crop count: shard0 = 100 crops / 2,560,379 nodes (38x44b6, 62x6bba),
  shard1 = 99 crops / 2,557,662 nodes (33x44b6, 66x6bba) -> **0.11% load imbalance**, both
  families in both shards, max crop 84,233 / 82,537 nodes.
- Kernel adds atomic per-crop .npz writes, a resumable status manifest (skip-completed),
  per-crop peak GPU memory, elapsed telemetry, and per-crop hashes. No scoring or GT access
  in the GPU path; the support pack remains inference-only.
- CPU replay will REUSE WORK rather than run six pipelines: build the pre-selection graph
  once per crop, then B (greedy), B' (default ILP), C (C1 ILP), and D reuses C's solved
  graph with only the v122 wrapper swapped -- the C1 ILP is never solved twice. Arm A needs
  no inference or solve. Max 3 local ILP workers, isolated subprocesses, large crops
  scheduled apart.

### 2026-07-29 (full-population coupled cache COMPLETE; CPU replay running)
- **GPU: 199/199 crops cached, ZERO failures.** Two private internet-off T4 shards,
  det 0.9690, stock 4-view TTA, LOEO routing. shard0 100/100 in 3.51h, shard1 99/99 in
  3.53h (0.6% wall imbalance vs 0.11% predicted by node-count balancing). Peak GPU
  601/599 MB. All 199 outputs hash-verified against the kernel manifests (199 ok, 0 bad),
  46 MB, covering exactly 71x44b6 + 128x6bba.
- VERIFICATION: production kernel reproduced the v7 preflight cache BIT-FOR-BIT on the
  shared crop (44b6_0113de3b: 34,539 coords / 29,000 edges), confirming the atomic-write
  fix and production hardening did not perturb numbers.
- GUARDS WORKED: startup self-test passed at 502s; canary stayed silent; both added after
  an earlier launch burned ~26 min of T4 per shard because np.savez_compressed appends
  ".npz" to a temp name, so os.replace failed AFTER each crop's inference. Root cause was
  process, not the bug: a validated preflight script was MODIFIED (atomic writes, sharding,
  resumability) and the modified script was scaled to 199 crops without re-validation.
- CPU replay (shared-solve: C1 ILP solved once, reused by D) at 184/199, 0 failures.
  ILP cost ~8ms/node; 21 crops >=68k nodes run serially because the largest (102,959 nodes)
  peaks at ~2.2GB RSS -- three concurrently would exceed available RAM. Light crops run
  3-wide. Heavy phase ~10-14 min/crop; light phase reached ~87 crops/hr.
- Scoring/decomposition script built AND validated ahead of the data; validation caught two
  bugs pre-emptively: displacement read a non-existent edge_dist column, and -- more
  serious -- aggregates were computed over per-arm crop sets, so arm A (all 199 from the
  frozen E0c cache) was being compared against D-on-N, silently fabricating a delta. All
  aggregates are now restricted to the per-fold intersection with loud partial-coverage
  warnings.
- NEXT (resume point): `.\.venv\Scripts\python.exe scripts\win_bet\coupled_replay_shared.py --workers 3`
  (resumable, skips completed) then `.\.venv\Scripts\python.exe scripts\win_bet\coupled_score.py --workers 4`.
  Promotion gate: D improves both folds, min-fold D-A >= +0.005, no regime collapse.

### 2026-07-29 (COUPLED C0/C1 RESULT: D FAILS the bilateral gate; ILP splits the folds)
- Full 199-crop, 5-arm result under the patched authoritative scorer. Complete matched
  coverage (A/B/B'/C/D all 199), zero failures anywhere in GPU or CPU.
- **PROMOTION GATE: FAIL.** 44b6 A=0.7595 D=0.6962 (**-0.0633**); 6bba A=0.6490 D=0.6997
  (**+0.0507**). Both folds improve = False; min-fold -0.0633 vs gate >= +0.005.
- NOT NOISE: 44b6 loses on 57/71 crops, 6bba gains on 102/128; crop-bootstrap CIs are tight
  and non-overlapping (44b6 [-0.0953,-0.0492], 6bba [+0.0440,+0.0623]).
- SEQUENTIAL DECOMPOSITION (order-dependent, contains interactions), 44b6 / 6bba:
  B-A detector population  -0.0112 / +0.0036
  B'-B introduce ILP 0.1/0.1  -0.0211 / +0.0328
  C-B' survival costs 0.0/1.5  **-0.0359 / +0.0125**
  D-C wrapper -> v122  +0.0049 / +0.0018
  D-A TOTAL  -0.0633 / +0.0507
- **The ILP is the entire effect and it is fold-splitting.** Introducing it plus raising the
  disappearance cost costs 44b6 -0.057 and gains 6bba +0.045. The detector threshold is
  near-inert (-0.011/+0.004) and the WHOLE v122 wrapper is worth only +0.005/+0.002 --
  independently confirming the preflight decision to drop C0 and D-port was correct.
- MECHANISM = node-recall collapse on the sparse fold: 44b6 recall 0.9482 -> 0.7854 under the
  C1 ILP; 6bba only 0.8731 -> 0.8527. The count multiplier IMPROVES for both
  (0.9895->1.0136, 0.9974->1.0110), so D buys count-penalty margin while destroying edges on
  44b6 -- it prunes tracks the sparse fold cannot afford.
- REGIME SLICES (D-A by tercile) -- **no deployment-observable feature separates the sign**:
  every 44b6 tercile negative (-0.031..-0.100), every 6bba tercile positive (+0.006..+0.090),
  across density, count ratio and displacement. Only gradient is 6bba density (+0.090 low ->
  +0.006 high): the benefit fades with density but never inverts.
- DECISION: per the preregistered fallback this is the "helps one regime, harms another" case,
  so the ILP branch is NOT killed. BUT the slices warn that the discriminating variable may be
  embryo identity itself, which is forbidden as a routing feature. Before committing to the
  cross-fit A-vs-D selector round, test whether ANY observable feature separates the sign
  WITHIN a family; if none does, the selector cannot generalise and the branch should close.
- E0c (A) remains the honest OOF baseline: 0.7595 / 0.6490. No promotion. Phase 2 GPU training
  stays blocked pending the selector-feasibility check.

### 2026-07-29 (SELECTOR FEASIBILITY AUDIT -> KILL; detector/ILP/wrapper repair family closed)
- Ran the preregistered falsification audit on the A-vs-D selector. Deployment-legal
  features only (no GT, N_est, family/crop ID). Primary target = continuous D-A weighted
  by each crop's edge denominator.
- **STEP 1 mixed-sign power (both families DO have mixed mass, so not trivially dead):**
  44b6 D>A 14 crops / 11.6% edge mass (+0.0056), D<A 57 crops / 88.4% (-0.0763);
  6bba D>A 102 crops / 78.3% (+0.0566), D<A 26 crops / 21.7% (-0.0058).
- **ORACLE CEILING (decisive):** per-crop max(A,D) using the TRUE sign -- unreachable upper
  bound for any selector -- gives 44b6 +0.0056 and 6bba +0.0566. **Min-fold +0.0056 vs a
  +0.005 gate: a PERFECT oracle passes by 0.0006.** On 44b6 upside is +0.0056 against
  downside -0.0763 (13.6x adverse), so a deployable rule must capture 90% of the upside
  while leaking <=0.8% of the downside. No margin exists.
- STEP 2 univariate: six features DO show same-sign association in both families with 44b6
  bootstrap CIs excluding zero (ep_p10/p50/p90/mean, node_retention, edge_retention;
  44b6 rho ~0.27-0.33, 6bba rho ~0.44-0.69). Correlation is real -- which is precisely why
  the falsification protocol was necessary.
- **STEP 4 leave-family-out transfer: ALL SIX FAIL.** Trained on 6bba -> +0.052 in-sample but
  **-0.004 to -0.035 on held-out 44b6** (actively harmful). Trained on 44b6 -> best in-sample
  gain is **-0.0001**, i.e. the optimum on 44b6 IS "never pick D", and the two retention
  features degenerate to picking D on 0/128 6bba crops. No feature passes; min-fold is
  negative in every case.
- **VERDICT: SELECTOR BRANCH CLOSED.** Gate not loosened, family identity not added, no
  nonlinear search attempted. Stronger than transfer failure: a single-feature rule cannot
  extract 44b6's oracle gain even WITH its own labels.
- CONSEQUENCE per the decision tree: all current detector/ILP/wrapper repair levers are
  closed (C0, D-port, v122 wrapper, fixed-ILP D, and now the A/D selector). E0c remains
  authoritative at 0.7595 / 0.6490. Phase 2 rebases directly on E0c with the new core
  image-model round (M1).

### 2026-07-29 (M1 pre-launch: resume made bit-exact; 100x loss-weight bug caught; sampling audit PASS)
- **BASELINE-FIDELITY BUG (caught by commander request, would have invalidated M1).**
  `train_epoch()`'s own defaults are det_loss_weight=0.1 / det_neg_weight=0.1, but `train()`
  -- the baseline path the OOF model was trained through -- passes 1e1 / 1e-2. The M1 driver
  called `train_epoch` without them, so the first smoke ran a **100x smaller detection
  weight and 10x larger negative weight**. M1 would not have been comparable to the baseline.
  Now passed explicitly and locked by a test that reads train()'s defaults from the vendored
  source via AST. Enforced: det_loss_weight=10.0, det_neg_weight=0.01, pool_kernel_um=5.0.
- **RESUME NOW BIT-EXACT.** The earlier 0.0157 divergence was NOT cuDNN nondeterminism (my
  original attribution, now retracted): the checkpoint restored training state but not the
  data-stream position, so resume replayed samples 1-20 instead of continuing to 21-40.
  `ResumableSampler` derives the epoch permutation from (seed, epoch), exposes a permutation
  hash, resumes at the next unconsumed index, and aborts on hash mismatch. Controlled test:
  Control A (1-40), Control A2 (identical rerun = nondeterminism floor), Resume B
  (1-20 -> serialize/reload -> 21-40). Result: steps 21-40 identical sample ids AND identical
  augmented images (n=20); max loss deviation resume-vs-control **0.0**; control-vs-control
  floor **0.0**; final weight hashes match both ways. There is no measurable GPU
  nondeterminism on this workload.
- **SAMPLING AUDIT (zero GPU) PASS.** The smoke's `unique_crops_sampled=1` was a reporting
  artifact of a broken window->crop attribution with a "?" fallback, not a sampler defect.
  Rebuilt attribution from canonical `VideoMeta.zarr_path`. Epoch-1 first 800 positions over
  5,145 windows: **800 unique windows, 59/59 crops touched, 11/11 strata covered**, steps per
  crop min=4 median=13 max=22, zero crops with no draws, zero validation crops, zero 6bba,
  zero unresolved ids. Largest exposure deviation -10.4/+6.6 against ~15 expected = ordinary
  multinomial spread.
- SMOKE TELEMETRY: 800 steps/epoch measured 946.4s; step seconds median 1.174 p90 1.242
  max 3.437; data wait 0.6% (GPU-bound 99.4%); peak GPU 3,000 MB, peak CPU RSS 2,443 MB;
  augmentation counts over 40 samples gamma 22 / noise 22 / brightness 20 / contrast 16 /
  psf_blur 15 / drift 14 (all near configured probabilities); edge-positive targets 40/40
  windows, mean 2.6 per window; grad norm median 85.2 p90 148.7 max 232.3, 0 non-finite.
- LARGE CLIPPED GRADIENTS (median 85 vs clip threshold 1.0): **THIS CLAIM WAS WRONG AND IS
  RETRACTED.** It was measured in the pre-fix smoke that ran with train_epoch's own 0.1/0.1
  loss weights. Under the CORRECT baseline weights (10.0/0.01), session 1 measured grad-norm
  median 2.2 falling to 0.2, with clipped-step frequency declining 57% -> 35% across epochs
  1-15. The median-85 figure was an artifact of the loss-weight bug, not inherited baseline
  behaviour. Clipping and loss weights remain unaltered during M1, and clipped-step
  frequency, component losses and grad norms are instrumented per epoch.
- `max_nodes=10` computed over TRAIN crops only (baseline uses train+test, but test there is
  the held-out family M1 must not touch). It sets padding width and is masked -> semantically
  neutral.
- HASHES: trainer source c4f6317736bb3bb1..., config fc7e4644ea37a90a, augmentation
  368908ecc44c0214, manifest ea5fe9b2eb9bd0fe.
- PROJECTION ACCEPTED: 7.89 h train-only for 24,000 steps -> two sessions, epochs 1-15 and
  16-30, ~3.94 h each. LAUNCHING M1 fold 1 (train 44b6, hold out 6bba).

### 2026-07-29 (M1 fold-1 training COMPLETE: 30 epochs / 24,000 steps, zero mechanical failures)
- Session 1 epochs 1-15: 3.93 h; session 2 epochs 16-30: 3.85 h (resumed from a hash-verified
  checkpoint at step 12,000). Total 7.78 h vs the 7.89 h measured projection. **Zero
  non-finite values across all 24,000 steps**; no mechanical guard fired.
- Edge loss 0.0060 -> 0.0017 (session 1) -> 0.0006 (session 2, plateauing). Detection loss
  stayed in a noisy 0.045-0.086 band with no clear trend. Grad-norm median fell 2.2 -> 0.15
  and clipped-step frequency 57% -> 27%.
- All five candidate checkpoints retained and verified: epochs 10/15/20/25/30 with
  global_step == epoch*800, seed 20260729, config fc7e4644ea37a90a, augmentation
  368908ecc44c0214, manifest ea5fe9b2...0324f0. SHAs 1353be9c / e4c2cb53 / db7bb932 /
  244740da / 28c3aab3.
- The detection/edge trade-off (edge loss falling while detection loss is flat/noisy) is
  exactly what the 10/15/20/25/30 sweep exists to resolve; it is NOT treated as a signal in
  itself and no proxy metric was used during training.
- NEXT: guarded selection stage -- inference-parity canary, all-5-checkpoint verification,
  full 5x12=60 grid, then local E0c-wrapper + patched scoring and a single frozen-rule
  selection on the 12-crop 44b6 inner validation. Held-out 6bba still untouched.

### 2026-07-29 (M1 checkpoint selection: EPOCH 10 wins; later checkpoints degrade sharply)
- Selection stage passed every guard: inference-parity canary reproduced the parity-proven
  split-0 graph EXACTLY (28,119 coords / 25,139 greedy edges, both canonical SHAs);
  provenance verified (manifest ea5fe9b2...0324f0, trainer c4f63177...d35dc9ea, predictor
  c44e771b...31c234b9); complete 5x12=60 grid in 1.2 h; local scoring path parity-tested
  against E0c's recorded 0.8132 before any candidate was scored.
- INNER-VALIDATION COMPOSITE (12-crop 44b6, edge-volume weighted, E0c downstream):
  **epoch 10 = 0.7963** | epoch 15 = 0.7955 | epoch 25 = 0.7501 | epoch 30 = 0.6975 |
  epoch 20 = 0.6820. Selected epoch 10 by the frozen rule (max composite; ties -> earlier).
- **THE EARLIEST CHECKPOINT IS BEST AND LATER ONES DEGRADE** -- graph-level composite falls
  ~0.10 from epoch 10 to epoch 20/30 while the training edge loss kept falling 0.0060 ->
  0.0006. That is textbook proxy-metric divergence and vindicates both the decision to select
  on exact patched composite rather than acc*recall, and the decision to retain early
  candidates instead of taking the final model. A 45-epoch run would have been worse still.
- LIKE-FOR-LIKE vs E0c on the SAME 12 crops: E0c 0.7889, M1 epoch-10 0.7963 -> **+0.0074**.
  Above the +0.005 bar, but this is the TRAINING family (held-out crops, same embryo), NOT
  the promotion gate. The gate is cross-family on 6bba.
- NEXT: single one-shot evaluation of epoch 10 on the complete 128-crop held-out 6bba family;
  gate >= +0.005 over E0c's 0.6490 with no major regime collapse.

### 2026-07-30 (M1 FOLD-1 HELD-OUT: FAIL +0.0010 vs +0.005 gate; M1 stopped after one seed)
- One-shot evaluation of the selected epoch-10 checkpoint on the complete 128-crop held-out
  6bba family. Run integrity: inference-parity canary reproduced the reference graph EXACTLY
  (28,119 coords / 25,139 edges, both SHAs), 128/128 grid, single checkpoint, 2.24 h GPU.
  Scoring reproduces **E0c 0.6490 exactly**, so the comparison is population-matched and sound.
- **RESULT: E0c 0.6490 -> M1 0.6499 = +0.0010** against a +0.005 gate. Crops improved 64/128
  (a coin flip). Crop-bootstrap mean +0.0010, CI [-0.0102, +0.0132] -- **spans zero**.
  **VERDICT: FAIL.** Per the decision tree, M1 is stopped after this one seed; fold 0 is NOT
  run and no constants are swept.
- MECHANISM -- M1 did not collapse, it traded and netted nothing:
  raw edge J 0.6499 -> **0.6421 (worse linking)**; nodes 2,253,622 -> **1,797,802 (-20%)**;
  adjusted J 0.6484 -> 0.6498 because the COUNT MULTIPLIER rescues the loss. Node recall
  0.8731 -> 0.8664; division-J 0.0057 -> 0.0013. The +0.0010 is essentially the count-penalty
  term rewarding a smaller graph, not better tracking -- the same mechanism that made arm D
  look good on 6bba and catastrophic on 44b6.
- **THE DECISIVE PATTERN: inner validation (44b6, same family) +0.0074 vs held-out (6bba,
  cross-family) +0.0010.** M1 generalises within an embryo and evaporates at the family
  boundary. Identical signature to breadth reranking, learned divisions, isolated detection,
  temporal accumulation and the coupled ILP: six independent methods, one shared wall.
- WHAT THE ROUND DID EARN:
  * The checkpoint sweep was worth its cost. Inner-validation composite: epoch 10 = 0.7963,
    15 = 0.7955, 25 = 0.7501, 30 = 0.6975, 20 = 0.6820 -- a 0.11 spread -- while training edge
    loss fell monotonically 0.0060 -> 0.0006. Selecting the final model, or selecting on
    acc*recall, would have shipped a worse checkpoint and a MORE negative held-out result.
    Exact graph-level selection on early candidates is a keeper.
  * Domain randomisation (PSF/noise/gamma/contrast/drift) did NOT buy cross-embryo
    robustness; it bought within-embryo robustness. That is evidence about the augmentation
    set itself, not merely about this seed.
- E0c remains the authoritative baseline: public 0.889, OOF 0.7595 / 0.6490.

### 2026-07-30 (LEAN RESET + FAMILY-BOUNDARY PIVOT)
- Consolidated the active research surface after seven exact-gate failures. The tracked tree
  before cleanup contained 246 files, including 61 Markdown files and 125 Python files.
  The lean working tree contains 51 project files: 10 Markdown files and 26 Python files.
- Preserved full recovery in annotated Git tag `pre-lean-2026-07-30` at commit `7897511`.
  Pre-existing dirty/untracked local files were copied to
  `C:\Users\aryaa\Documents\Biohub-CellTracking-2026-prelean-local-2026-07-30` before
  pruning; historical raw logs were then removed from the active repository.
- Replaced stale and contradictory command documents with four authoritative surfaces:
  `HANDOFF.md`, `reports/NEXT_DECISION.md`, `reports/EXPERIMENT_LEDGER.md`, and
  `reports/METRIC_SEMANTICS_VERIFIED.md`. Negative evidence remains in this journal, the
  compact ledger, canonical result files, and the recovery tag.
- Removed retired M1/coupled/DAXI/division/Trackastra/solver implementations, their tests,
  duplicate Kaggle kernels, old transfer briefs, generated figures, and redundant inventory
  outputs from the active tree. Retained the E0c baseline, exact scoring utilities, public
  v122 hedge, OOF train/predict kernels, metric/wrapper tests, and six canonical results.
- LIVE LANDSCAPE CHECK: pulled two current public notebooks advertising roughly 0.95. Both
  append negative-time, out-of-volume hub/fork structures to the scored submission; those
  scores are exploit-contaminated and quarantined. The clean pre-exploit code exposes one
  potentially unmeasured mechanism: retain top-two transformer parent candidates down to
  probability 0.25 before a global ILP.
- DECISION: no generic retraining, extra M1 seeds, full-data fit, or ensemble. Active round
  has two cheap branches only: (A) candidate-breadth clean extraction/oracle gate on the two
  highest-edge-mass crops, and (B) CPU-only family-boundary decomposition. GPU is allowed
  only after a bilateral cheap gate. If neither exposes a bilateral oracle ceiling >=0.01,
  stop research compute and retain E0c/v122 as the final private/public hedge.

### 2026-07-30 (session hook; concurrent-write collision, corrected)

- No compute run. Task was to produce a cold-start hook for a new session.
- COLLISION: I rewrote `HANDOFF.md` in place; the parallel agent replaced the same file
  with its own lean rewrite between my edit and my `git add`. Commit `51d3c95` therefore
  contains the parallel agent's 81-line version, NOT the content its own message
  describes. History is left intact (the other agent may have branched from it); this
  entry is the correction. Working tree is the intended state.
- The landed `HANDOFF.md` is the better doc: it carries v122 deployment-probe OOF
  (0.6962 / 0.6997 vs public 0.908) that my draft lacked, and points at two new docs --
  `reports/EXPERIMENT_LEDGER.md` (indexed figures) and `reports/NEXT_DECISION.md` (the
  active preregistered plan). Tag `pre-lean-2026-07-30` preserves the pre-clean tree.
- One thing the lean rewrite dropped was worth keeping, so I wrote it to
  `reports/ENVIRONMENT_TRAPS.md` and linked it from HANDOFF: the six environment defects
  that each cost real time -- `train_epoch()` loss-weight defaults (0.1/0.1) diverging
  from the baseline `train()` path (1e1/1e-2); the Kaggle image's polars whose compiled
  backend does not load while `import polars` succeeds; `biohub_tracking` vs
  `tracking_cellmot` naming; the support pack predating scorer patch `075fc5f`;
  `np.savez_compressed` appending `.npz` to a `.npz.tmp` temp name and failing the atomic
  replace AFTER the GPU work; and DataLoader position needing to be checkpointed to
  resume. Plus three process rules (never scale a script modified after preflight;
  cross-check node counts after any replay; score only over the arm intersection).
- State unchanged: E0c authoritative (public 0.889, OOF 0.7595 / 0.6490), nothing
  promoted, no compute running. Next work is `reports/NEXT_DECISION.md`: branch A (clean
  extraction of expanded pre-ILP top-two/0.25 candidates, gated on a two-crop audit) and
  branch B (CPU-only family-boundary decomposition). Both are cheap falsification gates;
  at most one graduates to GPU.
- LESSON for a shared worktree: stage a file immediately after editing it, and verify the
  staged blob is what you wrote before committing. A commit message asserting content is
  worthless if the content was replaced underneath it.

### 2026-07-30 (BRANCH A GATE — candidate breadth falsified on CPU, zero GPU spent)

- Executed `reports/NEXT_DECISION.md` branch A end to end without a Kaggle session.
  Reproduction: `scripts/branchA_gate.py --stage all`. Preregistered crops only
  (`44b6_d29c9ab2` split 0, `6bba_bb9f20c3` split 1). No threshold was tuned on them.
- GATE A0.2 — the cache does NOT contain the mechanism. The organizer OOF GEFFs are
  hard-pruned upstream at `p >= 0.5` (observed minimum `edge_prob` `0.500008` / `0.500015`,
  nothing below) and the in-degree histogram is `{1: 37405}` / `{1: 20213}` — every target
  has exactly ONE parent, zero targets have two. Top-two-parents-down-to-`0.25` is therefore
  genuinely absent and cannot be replayed from cached artifacts.
  `inventory/branchA_surface_audit.json`.
- GATE A1.1 — oracle ceiling PASSES. Of E0c's matched-edge false negatives (60 on 44b6,
  215 on 6bba): recoverable by any within-10um mechanism `44 / 107` = `73.3% / 49.8%`;
  of those, not already a transformer edge `36 / 64` = `60.0% / 29.8%`; of those, requiring
  enumeration E0c never performs at all `22 / 39` = `36.7% / 18.1%`. Every level is above
  the 10% bar. `inventory/branchA_oracle_ceiling.json`.
- MECHANISM — 100% of those never-enumerated candidates are beyond E0c's 6um tight gate. E0c
  enumerates a tight 6um pass over all nodes and then a relaxed 10um pass restricted to
  tight-pass leftovers, so a true pair beyond 6um whose endpoints were consumed by the tight
  pass is never enumerated. Median separation `8.29um / 7.62um`. No transformer probability
  is needed to ENUMERATE them, so the breadth half is testable on CPU.
- GATE A1.2 — FAILS decisively. One global 10um enumeration (single preregistered
  configuration), exact patched scorer:
  44b6 adjJ `0.8821 -> 0.7225` (`-0.1596`), edge_tp `1268 -> 1177`, edge_fp `99 -> 288`;
  6bba adjJ `0.8068 -> 0.6572` (`-0.1496`), edge_tp `1664 -> 1532`, edge_fp `191 -> 459`.
  Re-running the wrapper at the E0c gate reproduced the cached graphs exactly
  (tp/fp/node counts identical), so the harness is sound.
  `inventory/branchA_widen10um_twocrop.json`.
- CEILING UNDER A PERFECT PROBABILITY — still fails. Granting the widened surface an ORACLE
  edge probability (p=1 on true GT pairs, never submitted) gives
  44b6 `-0.1611` at the shipped learned bonus and `-0.1319` with the probability dominating
  geometry (bonus 40); 6bba `-0.1500` and `-0.1281`. edge_tp stays BELOW baseline in every
  variant (`1198 < 1268`, `1555 < 1664`). `inventory/branchA_oracle_prob_twocrop.json`.
- CONTROL — not a downstream artifact. With the short-track filter disabled the widened
  oracle configuration is still `-0.1281 / -0.1125`. `inventory/branchA_control_nofilter.json`.
- WHY IT FAILS: the per-frame assignment is one-to-one. Widening the gate to 10um adds
  ~1550 (44b6) / ~940 (6bba) extra relink edges per crop, and each false assignment can also
  displace a true one, so true positives fall even when every true pair is scored perfectly.
  Candidate breadth is not merely neutral here — it is actively destructive.
- DECISION: Gate A1 requires ALL of its bullets; bullet 2 fails on both crops under the
  most favourable possible edge scoring. Branch A is CLOSED. No two-crop T4 cache, no GPU.
- RESIDUAL, stated honestly: this falsifies expanded candidates inside E0c's per-frame
  Hungarian assignment, not inside the public notebooks' global ILP. That combination
  remains formally untested. It rests on two independently negative components — this
  result, and closed lever 5 (v122 coupled ILP, 44b6 `-0.0633`). Reopening it needs a new
  argument for why the interaction beats both parts, not just the observation that it is
  untested.
- E0c unchanged as authoritative: public `0.889`, OOF `0.7595 / 0.6490`. Nothing promoted.
  Next active work is branch B (CPU-only family-boundary decomposition); if it also fails
  its bilateral `+0.01` gate, the stop rule fires and the final hedge is E0c + clean v122.

### 2026-07-30 (D0' — GREEN: composed fork suppression + reconstruction is +0.0783 / +0.0737)

- Executed D0' per the amended division priority. Reproduction:
  `scripts/win_bet/phaseb_oracle_d0prime.py --workers 6 --materialize` (five parity-controlled
  arms, 199 crops, authoritative patched scorer, edges-only edits so `N_pred` and node recall
  are identical in every arm). GT-informed graphs written under `GT_ORACLE_do_not_submit/`.
- MOTIVATION: Oracle C only ever ADDED or REPLACED forks; it never REMOVED one, so its
  division denominator retained all 93 / 584 false forks. `divJ = 20/(20+93+6) = 0.168` is
  78% false-fork denominator. Suppression alone is worthless (`0/(0+0+26) = 0`) and was
  rejected in isolation on 2026-07-19. The two operations were never composed.
- ARMS (composite, delta vs E0c 0.7595 / 0.6490):
  * baseline                            0.7595 / 0.6490  (+0.0000 / -0.0000) PARITY OK
  * suppress_all                        0.7630 / 0.6517  (+0.0035 / +0.0027)
  * suppress_all_then_add_replace       0.8413 / 0.7273  (+0.0818 / +0.0783)
  * selective_suppress_then_add_replace 0.8413 / 0.7273  (+0.0818 / +0.0783)
  * add_replace_then_selective_suppress 0.8413 / 0.7273  (+0.0818 / +0.0783)
  All three add arms are identical: order does not matter and the operations do not interfere
  (`readded_after_suppression` = 0 / 2). Division goes `TP0/FP93/FN26 -> TP20/FP0/FN6`
  (divJ 0.769) and `TP4/FP582/FN121 -> TP93/FP0/FN32` (divJ 0.744).
- GT-FREE CONTROL (`--fallback-only`, deterministic lowest-id child retention, no GT-consistent
  child assist): suppress_all 0.7595 / 0.6470; suppress_all_then_add_replace 0.8378 / 0.7227,
  i.e. **+0.0783 / +0.0737**. The GT-assisted child retention was worth only +0.0035 / +0.0046.
  GT-free suppress_all reproduces the historical patched no-fork anchor on 44b6 exactly
  (0.7595, diff -0.0000); 6bba differs by -0.0012 (0.6470 vs 0.6482) because the historical
  ablation used a different retained-child rule. `inventory/phaseb_oracle_d0prime.json` and
  `phaseb_oracle_d0prime_gtfree_control.json`.
- DECISION = GREEN on both preregistered conditions. Composed min-fold ceiling +0.0737 vs the
  +0.03 bar. False-fork suppression contributes composed minus Oracle-C-alone =
  `+0.0783-0.0182 = +0.0601` and `+0.0737-0.0138 = +0.0599`, vs the +0.01 bilateral bar.
  The composition is strongly super-additive: 0.0035 + 0.0182 = 0.0217 in isolation versus
  0.0783 composed. The extra ~0.06 is purely the false-fork denominator collapsing.
- STRUCTURE FOR D1: E0c emits 11,441 forks on 44b6 and 9,012 on 6bba, of which **0 and 2** sit
  on a true GT divider. The fork layer is essentially pure noise; only 93 / 584 of those forks
  are metric-evaluable, the rest fall in unannotated regions and are invisible to the scorer.
  Reachable true forks are 20/26 and 93/125.
- REALIZABILITY (why the precision-0.9 kill was the wrong instrument): after suppression the
  division count starts at `TP0/FP0/FN26`, so adding `k` true and `m` false forks gives
  `J = k/(26+m)`. FN is fixed; each FP costs only one denominator slot. At 50% precision with
  10 true recovered, `J = 10/36 = 0.278 -> +0.028`; at 30% precision, `J = 10/49 = 0.204 ->
  +0.020`. Both clear the `+0.005` promotion gate by a wide margin. The killed v4 posterior had
  LOEO PR-AUC 0.715, which is plausibly sufficient at this operating point even though its
  recall@P0.9 was ~0.045. This is a genuinely different gate, not a rerun of the closed method.
- HONEST LIMITS: this is a GT-informed ORACLE ceiling. Fork SELECTION remains oracle in every
  arm; the `--fallback-only` control only removes the secondary oracle assist in choosing a
  retained child. Nothing here is deployable and none of it may be submitted. The deployable
  question is unchanged and is now D0: can a real posterior, thresholded for exact composite
  rather than precision 0.9, capture a useful fraction of +0.0783 / +0.0737.
- NEXT: D0 Jaccard-optimal operating-point reanalysis. NOTE — no saved v4 posterior predictions
  exist anywhere under `artifacts/`; `artifacts/kaggle/divevents/*.npz` are the balanced
  training events only (ZSNS001 38262 / ZSNS003 3376 / ZSNS004 5728 / ZSNS005 10324). D0
  therefore requires the inference-only re-run already authorised. E0c unchanged as
  authoritative; nothing promoted.

### 2026-07-30 (D0P — GT-free proposer audit: frozen deployable surfaces RED; the cap, not the approach, is the limiter)

- Executed D0P per directive. Reproduction: `scripts/win_bet/phaseb_d0p_proposer.py --workers 6`.
  Candidate generation never consults GT: every E0c node at `t` may be a mother, daughters are
  distinct E0c nodes at `t+1`, daughters that already have a parent stay eligible (stealing is
  required), incumbent+alternative and two-alternative pairs both included, daughter order
  canonicalised and deduplicated. GT is used only afterwards for auditing and labels. The
  reported oracle is "proposable-only": `suppress_all` (GT-free lowest-id retention) then
  add-replace restricted to forks the proposer actually generated.
- RESULTS (composite, delta vs E0c 0.7595 / 0.6490):
  * `geometric_core` (10.5 um parent->daughter, 8.5 um sister; hash 8d588cd8446f)
    44b6 0.7790 (+0.0195), coverage 5/20 reachable, 5/26 all;
    6bba 0.6632 (+0.0142), coverage 20/93 reachable, 20/125 all.
    Candidates 4,197,640 / 3,213,390; reliable-negative 24,677 / 81,594; per-mother pairs
    median 2.0/1.0, p99 6.0/5.0, max 21/32.
  * `native` (pairs from the pre-assignment candidate-edge cache; hash de1097294192)
    44b6 0.7711 (+0.0116), coverage 3/20; 6bba 0.6526 (+0.0036), coverage 7/93.
    Candidates 297,313 / 269,302. CAVEAT: E0c applies `OUTPUT_LINEFIT_SMOOTH` (weight 0.8), so
    raw candidate coordinates are displaced from cached graph coordinates; endpoints were mapped
    by nearest neighbour within 2.0 um at 80.1% fidelity. Even at perfect fidelity this surface
    is far below the others.
  * `outer_diag` (fixed 15.0 / 15.0 um; hash e68b4ee31e50)
    44b6 0.8378 (+0.0783), coverage **20/20** reachable; 6bba 0.7136 (+0.0646), coverage 82/93.
    Candidates 64,699,626 / 44,145,348; reliable-negative 405,212 / 965,478; per-mother pairs
    median 18.0/7.0, max 159/197.
- DECISION per the preregistered rule = **RED**. GREEN required bilateral `>=+0.03`; AMBER
  `+0.015-0.03`; RED if either family is below `+0.015`. `geometric_core` gives 44b6 `+0.0195`
  (AMBER band) but 6bba `+0.0142`, below the `+0.015` floor. `native` is far below on both.
  No frozen deployable surface passes.
- BUT THE DIAGNOSTIC DID ITS JOB, and this is the load-bearing finding: `outer_diag` recovers
  **20/20** reachable divisions on 44b6 and reproduces the GT-free D0' ceiling **exactly**
  (+0.0783). So the prize IS present in a GT-free surface; the `10.5 um` parent->daughter cap,
  not the proposer concept, is what discards it. Coverage goes 5/20 -> 20/20 on 44b6 and
  20/93 -> 82/93 on 6bba purely by widening the caps.
- MECHANISTIC READING: the 10.5 um cap was calibrated on MIGRATION (it is E0c's motion-relink
  gate). At division the daughters separate, so parent-to-daughter displacement is
  systematically larger than ordinary frame-to-frame motion. A migration-calibrated cap is
  therefore the wrong prior for a division proposer. This is consistent with the Branch A
  measurement that recoverable missed edges sit at median 8.29 / 7.62 um with p90 9.61 / 8.78 --
  ordinary continuations live just under the cap, while divisions live above it.
- COST: the metric-visible precision problem is far smaller than the raw candidate count,
  because division FP accrues only at annotated mothers. For `outer_diag` the effective
  denominator is 405,212 / 965,478 reliable-negative candidates, not 64.7M / 44.1M. Recovering
  k=20 true forks against m false gives `J = k/(26+m)`, so m <= ~50 keeps J >= 0.29 -- roughly a
  4-order-of-magnitude discrimination task on metric-visible candidates, with a further burden
  of not firing on ~64M unlabeled candidates that can still steal assignments.
- NOT TUNING: caps were frozen before execution and are NOT being re-selected after seeing
  results. No family-specific cap was chosen. Any new surface must be preregistered on
  principle (e.g. division-specific displacement priors), not fitted to these numbers.
- E0c unchanged as authoritative. Nothing promoted. D0R is NOT started: it was conditional on
  D0P passing, and D0P failed its frozen gate.

### 2026-07-30 (division_flow_pair / H0 — RED; the SISTER cap is the sole binding constraint)

- Executed the frozen `division_flow_pair` surface (hash `35b6abef7f13`) and its frame-median
  control (`f81cba9039de`) unchanged. Constants inherited, not swept: parent<=15.0 um
  (outer_diag), sister<=8.5 um (geometric_core), pair-midpoint<=6.0 um from
  `mother + local_flow` (E0c `MOTION_RELINK_TIGHT_UM`), knn_k=16 / knn_min=4 frozen into the
  hash. Local flow from high-confidence one-parent/one-child E0c continuations only.
- RESULT (both surfaces identical to four decimals):
  * 44b6 composite 0.7790 (`+0.0195`), coverage **5/20** reachable (25.0%), 5/26 all;
  * 6bba composite 0.6632 (`+0.0142`), coverage **20/93** reachable (21.5%), 20/125 all;
  * candidates 2,942,847 / 2,213,972 = 5.16M total (budget 15M -> PASS);
  * reliable-negative 17,784 / 63,403; per-mother pairs median 1.0, p99 5.0/4.0, max 16/20;
  * flow fallback: kNN supplied 100% of estimates (14 frame-median fallbacks on 6bba, 0
    global). The frame-median control scores identically, so flow estimator choice is
    irrelevant here and no family-specific selection was needed.
- GATE = **RED on two independent criteria**: 6bba `+0.0142` is below the `+0.015` floor, and
  reachable recall 25.0% / 21.5% is below the 40% RED threshold (GREEN required 60%). Candidate
  budget passed comfortably. Per the one-shot rule, reconstruction closes and D0R/H1 do not start.
- DECISIVE MECHANISTIC FINDING, isolated precisely because all three surfaces were frozen
  rather than swept:
  * geometric_core  parent 10.5, sister 8.5            -> 5/20 and 20/93
  * division_flow_pair parent 15.0, sister 8.5, mid 6.0 -> 5/20 and 20/93  (IDENTICAL)
  * outer_diag      parent 15.0, sister 15.0           -> 20/20 and 82/93
  Raising the parent cap 10.5 -> 15.0 changed coverage by exactly ZERO. Raising the sister cap
  8.5 -> 15.0 recovered everything. **The sister-separation cap is the sole binding constraint;
  the parent-daughter cap and the flow-midpoint gate are not binding at all.**
- This INVERTS the 2026-07-30 D0P hypothesis, which is recorded there and is now corrected: I
  reasoned that divisions displace the mother further than migration and proposed a tighter
  sister constraint to control cost. The data says the opposite -- at the annotated split frame
  the daughters are already separated by MORE than 8.5 um, and it is the sister cap, not the
  parent cap, that discards 75-79% of reachable divisions. The flow-midpoint gate did cut cost
  (4.20M -> 2.94M on 44b6) without losing a single division, so it is a free constraint, but it
  is not the limiter.
- NOT DESIGNING ANOTHER SURFACE. Per directive, RED closes reconstruction without another
  geometry round. Recorded as a factual consequence only: a surface holding the flow-midpoint
  gate while relaxing the sister cap would sit far below outer_diag's 64.7M/44.1M triplets,
  because the midpoint gate demonstrably removes cost without removing divisions. That is an
  observation for the commander, not an executed experiment.
- The original D0P RED remains valid for `native` and `geometric_core`; this was a new frozen
  hypothesis prompted by the outer diagnostic's preregistered purpose, not a retrospective
  threshold adjustment. E0c unchanged as authoritative; nothing promoted; no GPU spent.

### 2026-07-30 (public V18 audit — BLOCKED at retrieval)

- Priority-1 target was exact No-Hack V18 (public 0.914) as a deployment hedge. The Kaggle CLI
  documents `<owner>/<kernel>/<version>` for `kernels pull`, so the stated Version-History
  blocker looked avoidable. It is not: version-pinned pulls of another user's kernel return
  `403 Client Error: Forbidden for url: .../GetKernel`. Tried v18 and v19 on both
  `yusuketogashi/no-hack-biohub-cell-another-approch-3rd` and
  `yusuketogashi/clean-approach-lightweight-local-cv-no-hack`; all four 403.
- Unversioned pull succeeds and returns the LATEST version only:
  `no-hack-biohub-cell-another-approch-3rd.ipynb`, sha256 `8fe651af1132cfc946102c7b5779aedd...`,
  166,627 bytes, 17 cells. It contains `harmonic` (11 occurrences) and `fusion` (10), so this is
  **V19 — the rejected harmonic-fusion version**, not the 0.914 hedge. It must not be used as
  the deployment hedge.
- Kernel metadata confirms the family: competition source `biohub-cell-tracking-during-development`,
  dataset sources `pilkwang/biohub-temporal-unet3d-seed314159-v1` and
  `pilkwang/biohub-tracking-support-pack-50ep-v1` (all-training-data artifacts, so public-only
  evidence), T4, internet off.
- Reverse-time association IS present in the retrieved V19 source (`reverse` 28, `logit` 74),
  so the D1 feature extraction is not blocked by the V18 retrieval failure even though the
  reproduction hedge is. Held outside the tracked repo per directive.
- V18 exact source therefore requires a manual Version-History download by the operator; it
  cannot be obtained headlessly with these credentials.

### 2026-07-30 (H0b — flow-gated wide-sister surface PASSES scientific viability)

- Preregistration amendment executed. H0b (`division_flow_pair_wide`, hash `34f91ada4626`)
  keeps every division_flow_pair constant identical and relaxes ONLY the falsified 8.5 um
  sister prior: parent 15.0, sister 15.0, midpoint 6.0, same kNN flow estimator (k=16, min=4),
  same suppress-all/add-replace resolver, no family routing, no sweep.
- RESULT:
  * 44b6 composite 0.8299 (`+0.0704`), coverage **18/20 reachable = 90.0%**, 18/26 all GT;
  * 6bba composite 0.7103 (`+0.0613`), coverage **78/93 reachable = 83.9%**, 78/125 all GT;
  * candidates 14,365,276 / 10,154,368 = **24.52M total**;
  * metric-visible (non-unlabeled) candidates 91,560 / 253,526 -- the real precision denominator;
  * per-mother pairs median 4.0/2.0, p90 8.0/5.0, p99 13.0/9.0, max 43/54;
  * steals 14,022,639 / 9,889,209; node recall unchanged (0.9482 / 0.8731), edges-only edits.
- SCIENTIFIC VIABILITY GATE: reachable recall >=60% bilaterally -> PASS (90.0% / 83.9%);
  oracle exact delta >=+0.03 bilaterally -> PASS (`+0.0704` / `+0.0613`). Confirms the H0
  diagnosis exactly: the 8.5 um sister prior, not the proposer concept, was the limiter.
  Relative to the GT-informed D0' ceiling (`+0.0783` / `+0.0737`), H0b's GT-free surface
  retains 90% and 83% of the available upside.
- ENGINEERING CLASSIFICATION: 24.52M exceeds the 15M raw budget, which per the amendment
  triggers the rank-compression audit rather than an automatic kill.
- RANK-COMPRESSION AUDIT (analysis B, `scripts/win_bet/phaseb_h0b_rankcompress.py`):
  IMPLEMENTED AND SMOKE-TESTED BUT THE FULL 199-CROP RUN DID NOT COMPLETE -- the session ended
  while it was in flight. NOTHING from the full run is recorded. Re-run to finish:
      .\.venv\Scripts\python.exe scripts\win_bet\phaseb_h0b_rankcompress.py --workers 6
  Smoke evidence on `6bba_48816121` (258,566 candidates, 4 reachable divisions, 3 of which are
  present in the 15/15 surface) -- rank of the true pair among that mother's candidates:
      midpoint_residual   0, 0, 2   (of 24, 20, 7)
      parent_midpoint     0, 0, 2
      fwd_support         1, 1, 5
      sister_separation   3, 10, 4
      persistence         6, 8, 5
  Shortlist sizes on that crop: K=1 25,197; K=3 73,142; K=10 195,647; full 258,566.
  READ THIS AS PROVISIONAL: one crop, three divisions. It is suggestive that the flow-midpoint
  residual is a strong label-free ranker and that sister separation is a WEAK one -- consistent
  with it having been the wrong veto -- but it is not the audit result.
- Rankings use only deployment-observable quantities, continuous, never as a veto, never
  selected using a candidate's own label. Reverse-time association, secondary detection and
  DeepCenter are recorded as UNAVAILABLE locally rather than silently omitted; reverse-time
  needs the V18 source that is still blocked at retrieval.
- E0c unchanged as authoritative. Nothing promoted. No GPU spent. H1/H2/H3 not started.

### 2026-07-30 (CORRECTION + result: H0b rank-compression audit DID complete — PASS)

- CORRECTION to the preceding entry and to commit `1fb13cf`, which recorded this audit as
  unfinished with no output. The run in fact completed; the notification arrived after the
  save. The earlier "INCOMPLETE / none of its output is recorded" statements are superseded by
  the numbers below. The single-crop smoke previously flagged as provisional is likewise
  superseded by the full 199-crop result.
- RESULT on the complete fixed 15/15 outer surface (64,699,626 / 44,145,348 candidates;
  reachable divisions present in the surface n=20 of 20 on 44b6, n=82 of 93 on 6bba).
  Rank of the true pair among its own mother's candidates, label-free single-feature rankings:

  44b6                rank med   p90   | retention K1    K3    K10
    midpoint_residual     0.0    4.2   |           0.70  0.80  1.00
    parent_midpoint       0.0    5.1   |           0.55  0.85  1.00
    fwd_support           5.0   23.3   |           0.15  0.45  0.70
    sister_separation     8.0   16.9   |           0.05  0.30  0.60
    persistence          10.5   23.3   |           0.10  0.20  0.40

  6bba                rank med   p90   | retention K1    K3    K10
    midpoint_residual     0.0    2.0   |           0.71  0.93  0.99
    parent_midpoint       0.0    3.0   |           0.55  0.87  0.98
    fwd_support           2.0   13.0   |           0.34  0.63  0.85
    sister_separation     2.0   11.0   |           0.33  0.57  0.83
    persistence           4.0   16.0   |           0.21  0.45  0.72

  Shortlist sizes (both families summed): K1 4,957,806; K3 14,371,002; K10 42,736,809.
- GATE: "can a fixed top-K-per-mother budget retain >=60% of reachable divisions while keeping
  the shortlist manageable" -> **YES, bilaterally, with margin.**
  * K=3 on flow-midpoint residual: end-to-end reachable retention 16/20 = 80.0% (44b6) and
    0.93*82 = 76/93 = 81.7% (6bba); shortlist **14.37M**, inside the 15M budget.
  * K=1: 14/20 = 70.0% and 58/93 = 62.4%; shortlist **4.96M**, far inside budget.
  Both K values clear 60% on both families. K=3 is the better operating point; K=1 is the
  cheap option and 6bba at 62.4% is close to the floor.
- MECHANISM CONFIRMED THREE WAYS: the flow-midpoint residual puts the true pair at MEDIAN RANK
  ZERO in both families, while sister separation ranks it 8.0 / 2.0. Sister separation is a
  weak discriminator, which is exactly why using it as a hard 8.5 um veto destroyed 75-79% of
  reachable divisions in H0. The quantity that works is the flow-predicted centre-of-mass
  residual -- the pair midpoint, not either daughter individually.
- No threshold was selected from any candidate's label; all five rankings are fixed single
  deployment-observable features. A learned cross-fitted combination (H1) has not been tried
  and would be expected to improve on these.
- Reverse-time association, secondary detection and DeepCenter remain UNAVAILABLE locally and
  are recorded as such, not silently omitted. `fwd_support` uses the raw OOF GEFF transformer
  probability where the pair exists as an edge (0.0 otherwise).
- CONSEQUENCE: H0c cascade compression is viable — broad 15/15 geometry generation, cheap
  flow-midpoint rank pruning to top-K per mother, expensive critic on the shortlist only, graph
  conflict resolution last. E0c unchanged; nothing promoted; no GPU spent; H1/H2/H3 unstarted.

### 2026-07-31 (H0c exact replay — PASSES all four gates; +0.0625 / +0.0597 on a bounded shortlist)

- Ran the frozen cascade unchanged (config hash `04eeac97500d`): 15/15 outer generation ->
  flow-midpoint top-3 per mother -> suppress-all (GT-free lowest-id retention) -> oracle
  selection for ceiling only -> identical add-replace resolver -> exact patched scorer.
  K and the ranking were inherited from H0b and NOT retuned. Reproduction:
  `scripts/win_bet/phaseb_h0c_replay.py --workers 6`.
- RESULT (per-crop baseline scored alongside, so nothing is assumed):
  * 44b6 baseline 0.7595 -> H0c **0.8221**, delta **+0.0625** (vs E0c anchor +0.0626);
    divJ 0.0000 -> 0.6154, TP0/FP93/FN26 -> **TP16/FP0/FN10**; adjEdgeJ +0.0010.
  * 6bba baseline 0.6490 -> H0c **0.7086**, delta **+0.0597** (vs anchor +0.0596);
    divJ 0.0057 -> 0.6080, TP4/FP582/FN121 -> **TP76/FP0/FN49**; adjEdgeJ **-0.0006**.
  * retention 16/20 = 80.0% and 76/93 = 81.7%.
  * shortlist 8,284,112 + 6,086,890 = **14,371,002** candidates, from 64.7M/44.1M pre-topK.
    This reproduces the H0b projection exactly.
  * steals only 11 / 50; mothers 2,816,276 / 2,141,530.
  * **node invariance verified, not assumed: N_pred identical on every crop and node_recall
    identical on every crop, both families.** Edits are edges-only, so the count multiplier is
    untouched.
- GATE: all four PASS.
  1. exact oracle delta >=+0.03 bilaterally -> +0.0625 / +0.0597;
  2. reachable retention >=60% bilaterally -> 80.0% / 81.7%;
  3. <=15M candidates -> 14.37M;
  4. no unexplained edge or node damage -> node metrics exactly invariant; edge effect is
     +0.0010 on 44b6 and -0.0006 on 6bba, both explained below.
- EDGE-SIDE HONESTY, the one result that is not clean. adjEdgeJ falls slightly on 6bba
  (`-0.0006`) and 22/71 (44b6) and 79/128 (6bba) individual crops regress on adjEdgeJ, worst
  cases `-0.0117` and `-0.0113`. Cause is mechanical and expected: suppress-all deletes fork
  children and add-replace performs parent steals, so a small amount of edge quality is traded
  for division credit. At the oracle operating point the trade is overwhelmingly favourable
  (division contributes +0.0615 / +0.0603 against an edge cost of at most 0.0006), but it is a
  real cost and it does not disappear when the classifier is imperfect.
- THE RISK THIS CREATES FOR H1, stated now so it is not discovered late: the edge cost is
  incurred by SUPPRESSION, which is unconditional, while the division gain is conditional on
  correctly selecting forks. A real classifier reduces the gain but not the suppression cost.
  After suppress-all, `J = k/(N_gt + m)` with `N_gt` = 26 / 125 fixed, so the break-even
  condition is roughly `0.1 * k/(N_gt+m) > edge_cost`. With `edge_cost ~ 0.0006` on 6bba the
  bar is low, but H1 must report the composite, not the division term alone.
- NOT MEASURED YET: the metric-visible (annotated-mother) subset of the 14.37M top-3 shortlist.
  H0b measured it for the un-ranked wide surface (91,560 / 253,526); the top-3 figure will be
  smaller and is the true precision denominator for H1. H1 must emit it.
- CONSEQUENCE: H1 opens. Cheap deployment-observable features first (midpoint and
  parent-midpoint residuals, forward association support, parent/daughter persistence, local
  density and track history, physically scaled covariance/eigenstructure, fluorescence mass
  conservation, peak splitting), one cross-fitted L2 logistic or compact MLP before any 3D CNN.
  Expensive image features must be encoded once per node/event and fused from cache, never run
  independently over 14M pairs. Reverse-time, secondary-model and DeepCenter evidence enter as
  later ablations when available; V18 is still blocked at retrieval and H1 does not wait for it.
- E0c unchanged as authoritative. Nothing promoted. No GPU spent. H1/H2/H3 not started.

### 2026-07-31 (H1a canonical candidate census — parity OK; the label surface is very small)

- Built the partitioned, provenance-preserving Parquet dataset over the frozen H0c top-3
  shortlist (cfg hash `04eeac97500d`; surface immutable, nothing retuned). One file per
  fold/crop under `artifacts/kaggle/e0c_cache/fork_candidates/h0c_top3/`. Reproduction:
  `scripts/win_bet/phaseb_h1a_census.py --workers 6`.
- PARITY, both checks exact:
  * total rows **14,371,002 = expected 14,371,002** (OK);
  * total positives **92 = H0c retained 92** (16 + 76) (OK) -- every H0c oracle-positive
    candidate is reproduced in the table.
- CENSUS:
  * 44b6: rows 8,284,112 (pre-topK 64,699,626), mothers 2,816,276, 71 crops;
    positives **16**, reliable negatives 54,031, unlabeled 8,230,065 (**99.35%**);
    metric-visible **54,059 = 0.653%** of shortlist; positive base rate among visible
    **0.000296** (~1 in 3,379); steals required 8,056,328; per-mother mean 2.94, max 3.
  * 6bba: rows 6,086,890 (pre-topK 44,145,348), mothers 2,141,530, 128 crops;
    positives **76**, reliable negatives 199,319, unlabeled 5,887,495 (**96.72%**);
    metric-visible **199,448 = 3.277%** of shortlist; positive base rate among visible
    **0.000381** (~1 in 2,624); steals required 5,898,212; per-mother mean 2.84, max 3.
- THE METRIC-VISIBLE NUMBER OWED FROM H0c IS NOW MEASURED: 54,059 / 199,448 = 253,507 total,
  i.e. only 1.76% of the 14.37M shortlist can ever contribute a division FP. The effective
  discrimination problem is ~1 in 3,000 among metric-visible candidates, not 1 in 156,000
  against the raw shortlist. That is the denominator H1 must be judged against.
- THE BINDING CONSTRAINT IS NOW LABELS, NOT CANDIDATES. The entire supervised surface is
  **92 positives and 253,350 reliable negatives**. In the leave-family-out direction that
  trains on 44b6 there are only **16 positives**. The two directions are therefore severely
  asymmetric (16 -> 6bba versus 76 -> 44b6) and the 44b6-trained direction should be expected
  to be the weaker and noisier of the two. Any bilateral claim must survive the 16-positive
  direction, which is the honest bottleneck for H1c.
- This strongly favours the structured per-mother formulation already directed for H1b: a
  choice among three ranked pairs plus explicit abstention needs far fewer parameters than a
  flat 14.37M binary classifier, and the abstention class is where almost all the label mass
  sits. It also argues for heavy regularisation and for reporting the ranking stage and the
  gate stage separately, since the ranker can be trained on relative order within a mother's
  choice set while the gate carries the scarce positive signal.
- Sparse annotation confirmed: 96.7-99.4% of candidates are unlabeled. Training and primary
  calibration use scorer-reliable labels only; unlabeled deployment behaviour is reported
  separately, never folded in as negatives.
- E0c unchanged as authoritative. Nothing promoted. No GPU spent. H1b/H1c not started.

### 2026-07-31 (H1 amendment — reverse-time source-locked; cost model established; G separable from F/R)

- CORRECTION ACCEPTED from the commander, and the H1a entry above is wrong on one point: I
  described the choose-among-three formulation as giving the ranker "plentiful" labelled
  signal. It does not. Three alternatives per mother improves the FORMULATION but not the
  labelled sample size, which remains **16 positive mothers on 44b6 and 76 on 6bba**. Model
  capacity must stay correspondingly tiny (L2 logistic / Firth-style; no MLP unless the linear
  model shows genuine bilateral ranking signal).
- SOURCE-LOCKED the reverse-time implementation to `vendor/public_v19_reverse/reverse_time_block.py`
  (verbatim cell-11 extract, 5,500 chars, extract sha256 `eb94e2745f270649865b6b89f8927450`),
  from kernel `yusuketogashi/no-hack-biohub-cell-another-approch-3rd`, notebook sha256
  `8fe651af1132cfc946102c7b5779aedd5aa173f29a85afc62a72c43e1147a1e4`, retrieved 2026-07-30.
  That pull is the LATEST version = **V19, the rejected harmonic-fusion build**; version-pinned
  pulls 403 and V18 (public 0.914) is still not obtained. V19 is never used as a hedge. The
  public `0.20` bidirectional blend weight is NOT adopted -- reverse scores enter H1 as
  FEATURES only.
- MECHANISM, now precisely understood:
      reverse_logits_native = model.predict_edges(unet_feat_tgt, unet_feat_src,
                                                  coords_tgt, coords_src,
                                                  pos_tgt, pos_src, mask_tgt, mask_src)
      reverse_logits_pair   = reverse_logits_native.transpose(1, 2)
  then per-column alignment to the forward scale: subtract reverse centre, multiply by
  `(forward_scale / reverse_scale).clamp(0.5, 2.0)`, add forward centre.
  It swaps the source/target argument groups of the SAME model on the SAME U-Net features.
- COST MODEL, the key feasibility finding: because the reverse view reuses the already-computed
  U-Net features of both frames, it costs roughly ONE EXTRA `predict_edges` call per frame pair
  -- not a second encode. The expensive part is the full inference pass itself, which we must
  run regardless because the cached OOF GEFF is pruned to one parent per target at `p >= 0.5`
  and therefore carries no pair-level logits for the candidates we care about.
- LOCAL FEASIBILITY CONFIRMED: model code
  `artifacts/kaggle/lb897_calibration/tracking_repo/src/biohub_tracking/models/{temporal_unet,
  simple_node_transformer}.py`; fold-specific checkpoints
  `weights_dataset/edge_predictor_best_split_{0,1}.pth` (136 tensors, `unet.*` + heads);
  configs (unet_out_channels 32, layers [32,64,128], downsample [1,4,4], window_size 2,
  pool_kernel_um 5.0); image volumes `data/train/*.zarr`. Nothing external is required.
- ABLATION DECOMPOSES CLEANLY, and this matters for sequencing:
  * **G (cheap geometry) needs NO model inference at all** -- it is derivable from the census
    table plus E0c graph topology, and can run immediately on CPU;
  * **F and R share ONE inference pass** -- forward pair logits for arbitrary candidate pairs
    are not cached anywhere, so F is not free either; once that pass is running, R is
    approximately one extra matmul per frame pair.
  Therefore G should be measured first, for free, to establish the geometry-only baseline that
  F and R must beat. This also protects the ablation's meaning: if G alone already promotes,
  the association-consistency hypothesis is not what is doing the work.
- NOT STARTED: the two-crop preflight (ID alignment both directions, double-reversal identity,
  deterministic cache hashes, low-margin coverage of positives/reliable negatives, proof that
  no graph is mutated during extraction). It requires reconstructing the inference pipeline
  (image loading, normalisation, tiling, node coordinates, window construction) from the
  notebook; the source-locked block is the association step only, not the whole path.
- E0c unchanged as authoritative. Nothing promoted. No GPU spent. G/F/R all unstarted.

### 2026-07-31 (H1-G result — CONTINUE, not PROMOTE: geometry closes ~half the log-gap, not enough)

- G feature extraction complete with exact parity: 14,371,002 rows, 92 positives, 23
  geometry/topology features, no model inference. `scripts/win_bet/phaseb_h1g_features.py`.
- BILATERAL AUC on the metric-visible subset (44b6: 16 pos / 54,031 reliable neg;
  6bba: 76 pos / 199,319). AUC < 0.5 means the feature is inversely related; sign consistency
  across families is the transfer diagnostic. `inventory/h1g_feature_auc.json`.
  Sign-CONSISTENT, strongest first (44b6 / 6bba):
    competing_parents      0.0271 / 0.1130   <- strongest bilateral separator by a wide margin
    resid_ratio            0.1740 / 0.2587
    flow_midpoint_residual 0.1926 / 0.2685   <- the frozen ranker, behaves as designed
    rank                   0.0853 / 0.2748
    daughter_angle         0.2303 / 0.3587   <- anti-parallelism CONFIRMED with correct sign
    cos_split_vs_flow      0.6878 / 0.6037
    vel_consistency        0.5824 / 0.6581
    pd_ratio               0.6331 / 0.5760
  Sign-INCONSISTENT (these ARE the family-boundary diagnosis, per the B17 protocol):
    n_persist              0.1747 / 0.7549   <- COMPLETE REVERSAL
    cos_daughter_axis      0.6951 / 0.4873
    mother_speed_um        0.4802 / 0.5651
    local_density_t        0.5198 / 0.4673
- TWO FINDINGS THAT MATTER MORE THAN THE RANKING:
  1. **Daughter anti-parallelism is confirmed bilaterally with the correct sign.** Lower
     `cos(daughter angle)` (more anti-parallel) predicts a true division in BOTH families. The
     three-source convergence (our smoke, btrack `P_branch`, mitosis literature) holds up.
  2. **`n_persist` REVERSES SIGN between families** (0.1747 vs 0.7549). Persistence into t+2 is
     the core of the public "persistence-confirmed divisions" heuristic (agent A, finding A-08).
     It does NOT transfer across our family boundary. Anything built on it is family-specific.
- THE DIMENSIONLESS HYPOTHESIS IS NOT SUPPORTED AS STATED. Dimensionless features do occupy the
  top of the sign-consistent ranking (competing_parents, resid_ratio, rank), but sign
  inconsistency is split evenly: 2 dimensionless (`n_persist`, `cos_daughter_axis`) and 2
  dimensional (`mother_speed_um`, `local_density_t`) flip. Being dimensionless is not sufficient
  for transfer. The agent-C hypothesis is downgraded from "explains the family boundary" to
  "correlates with the top of the ranking".
- STRATIFICATION AGAINST BREAK-EVEN (`inventory/h1g_steal_stratification.json`). Required
  precision-among-metric-visible for `+0.005` is **4.07% / 6.38%** (agent D, D-07).
    frozen ranker alone            0.107% / 0.101%
    competing_parents == 0         0.321% / 0.325%   (5/16 and 27/76 positives retained)
    competing_parents==0 & rank==0 **0.690% / 0.498%** (5/16 and 24/76 positives retained)
  Best label-free geometric stratum improves precision **6.4x / 4.9x** over the frozen ranker
  but remains **5.9x / 12.8x short** of break-even, and costs ~68% of recall.
- VERDICT = **CONTINUE**, not PROMOTE, by the directive's own classification: the mechanism
  measurably improves separability bilaterally, but calibration is nowhere near sufficient.
  Geometry alone cannot fund the division track. No submission is justified from G.
- CONSEQUENCE, and this now rests on a measured shortfall rather than an assumption: the
  remaining ~6-13x must come from evidence geometry does not contain -- image appearance
  (mass conservation, saddle depth, condensation, um-scaled second moments) or association
  scores. That is the first properly grounded justification for the expensive arms.
- Also note D-12 is partly answered: positives are NOT uniformly spread over the stealing
  population. `competing_parents<=1` retains 16/16 and 74/76 positives, so excluding
  full-stealing candidates does not collapse retention -- the branch-A failure mode is less
  threatening here than feared, though `competing_parents==0` alone keeps only ~31%.
- E0c unchanged as authoritative. Nothing promoted. No GPU spent. No submission consumed.

### 2026-07-31 (STRUCTURAL CORRECTION — the min-fold gate is not the competition objective)

Agent 5 challenged a premise I had supplied (that E0c and v122 rank oppositely public vs OOF).
I verified independently by re-aggregating the 995 cached per-crop rows in
`artifacts/kaggle/coupled_cache/scores/` through `tracking_cellmot.metrics.summarise`.
**The agent is right and I was wrong.**

| arm | pooled OOF | 44b6 | 6bba | min-fold | public |
|---|---:|---:|---:|---:|---:|
| A = E0c | 0.66539 | 0.75955 | 0.64895 | 0.64895 | 0.889 |
| B | 0.66677 | 0.74835 | 0.65256 | 0.65256 | - |
| Bp | 0.69162 | 0.72728 | 0.68538 | 0.68538 | - |
| C | 0.69692 | 0.69139 | 0.69788 | 0.69139 | - |
| D = v122 | **0.69909** | 0.69625 | 0.69966 | 0.69625 | **0.908** |

**44b6 edge mass = 21,578; 6bba = 122,815 -> 44b6 is 14.94% of total.**

- Pooled OOF ranks v122 above E0c by `+0.0337`. Public ranks v122 above E0c by `+0.019`.
  **Same direction. There is no public/OOF rank reversal.**
- The apparent reversal is an artifact of our own min-fold promotion gate, which weights the two
  families 50/50 while the actual scorer weights them ~15/85 by edge volume
  (`summarise` weights `adj_edge_jaccard` by `w = TP+FP+FN` per sample).
- CONSEQUENCE, and it is serious: **the min-fold bilateral gate is not the competition's
  objective function.** Arms C and Bp are `+0.0315` and `+0.0262` on pooled OOF versus E0c, yet
  both FAIL the min-fold gate because they lose on 44b6 -- a family carrying 15% of the score
  mass. v122, our best public score, is also our best pooled arm and would likewise have failed a
  min-fold gate against E0c. We have been judging candidates by a criterion the leaderboard does
  not use.
- This does NOT make the min-fold gate worthless. It is a legitimate PRIVATE-ROBUSTNESS criterion
  protecting against a private set with a different family mix. The error is that the project has
  treated it as the PRIMARY promotion criterion rather than as a risk constraint reported
  alongside the pooled score. Both numbers must be reported from now on.
- Caveat that cuts the other way: E0c scores 0.889 public but 0.66539 pooled OOF, a +0.22 gap.
  The OOF crop pool is therefore NOT representative of the test movies, which argues for weighting
  wide-composition scenarios over the OOF-like one when choosing finalists.

**Composition risk (20,000+ draws, exact scorer semantics):**
P(public winner is also private winner) = 0.9365 exchangeable -> 0.671 under independent family
composition -> 0.425 under independent family x density composition. Public rank is a weak private
signal once composition is not held fixed.

**Lower-tail:** v122 has the best mean AND best p5 in every scenario (uniform p5 0.68591;
wide-composition p5 0.66731; adversarial min 0.66500).

**Error covariance:** no low-correlation pair exists in the current candidate set -- all five arms
share one detector and one transformer. Mass-weighted per-crop error correlation with v122:
C 0.9995, Bp 0.987, E0c 0.959. At the SCORE level under composition uncertainty E0c is the only
decorrelated challenger: corr 0.233 (family-uncertain), 0.224 (joint wide), **-0.154**
(density-uncertain). C_survival is strictly dominated by v122 in all 20,000 draws and must never
occupy a slot.

**Recommended portfolio: v122 anchor + E0c challenger.** In the 37.1% of wide-joint draws where
v122 is not the private winner, E0c recovers mean +0.0075, p95 +0.0396, max +0.0702. The flip axis
is density, only partly the family label: density-low tercile v122 beats E0c by +0.081;
density-high tercile E0c beats v122 by +0.029. (Not a router -- routing on family/crop identity
remains forbidden.)

**Spurious-pass probability, independently recomputed and WORSE than the earlier estimate:**
SD of the 44b6 division-driven composite delta = **0.00952** (three methods agree: closed form,
200k fixed-crop MC, crop-clustered bootstrap), i.e. **1.91x** the `+0.005` bar, not 1.72x.
P(bilateral spurious pass) = 3.79% per trial; **17.6% at 5 trials** (not 12.7%), 32.1% at 10.
Independence is optimistic -- at rho 0.25/0.50/0.75 the 5-trial figure rises to 25.6/33.8/42.6%.
A `+0.005` bilateral gate on 44b6 is a **1.3-mother test on a 26-mother population**; it fires on
two lucky mothers. For division-term candidates the bar should be `>=0.015` on 44b6 (~4 mothers,
1.6 sigma) or a per-family division-count CI must be reported alongside. H0c itself sits at
6.6 sigma / 13.7 sigma above the null, so H0c's risk is the zero-FP oracle conditioning, not
sampling noise.

**H0c is orthogonal to the edge-arm choice.** It leaves the edge term essentially untouched
(delta adj `+0.00099 / -0.00055`) and should be applied ON TOP of whichever arm is selected rather
than consuming a submission slot.

New tools: `scripts/private_split_simulator.py`, `scripts/private_portfolio_risk.py`.

### 2026-07-31 (H1-I appearance result + CORRECTION: my AUC had a tie-handling bug)

**CORRECTION 3 — my H1-G AUC table was wrong for tied features.** Agent 3 could not reproduce
my `competing_parents` number, which exposed the cause: my AUC used ordinal ranks
(`np.argsort(np.argsort(a))`) with NO midrank tie correction. For continuous features this is
harmless -- `daughter_angle` and `flow_midpoint_residual` reproduce to 4 dp. For 2-3 valued
integer features it is badly wrong.

| feature | uniq | reported (wrong) | CORRECT (midrank) |
|---|---|---|---|
| competing_parents | 3 | 0.0271 / 0.1130 | **0.3164 / 0.3141** |
| n_persist | 3 | 0.1747 / 0.7549 | **0.5282 / 0.5019** |
| rank | 3 | 0.0853 / 0.2748 | 0.2137 / 0.3060 |
| persist_d1 | 2 | 0.0930 / 0.7705 | 0.5161 / 0.4962 |
| persist_d2 | 2 | 0.1529 / 0.7717 | 0.5158 / 0.5100 |
| daughter_angle | 53733 | 0.2303 / 0.3587 | 0.2303 / 0.3587 (unchanged) |
| flow_midpoint_residual | 53741 | 0.1926 / 0.2685 | unchanged |

Agent 3's independent value (0.3164 / 0.3157) matches the corrected figure, confirming the fix.

Two headline claims I made are therefore withdrawn:
1. **"`competing_parents` is the strongest bilateral separator by a wide margin"** -- FALSE. At
   0.3164/0.3141 it is a real but ordinary separator, weaker than `flow_midpoint_residual`
   (0.1926/0.2685) and `resid_ratio`. The strongest geometric separators are the continuous ones,
   which were never affected by the bug.
2. **"`n_persist` REVERSES SIGN between families (0.1747 vs 0.7549)"** -- FALSE, and the
   correction changes the conclusion rather than softening it. True value 0.5282 / 0.5019:
   persistence into `t+2` carries **essentially no discriminative signal in either family**. The
   public "persistence-confirmed divisions" heuristic is therefore not family-specific as I
   reported; on our metric-visible subset it is simply not discriminative. Same practical
   verdict, different and correct reason.

**H1-I RESULT — appearance beats geometry, sign-consistently.**
Coverage: 97/199 crops but **all 92 positives**, plus 65% / 47% of reliable negatives
(128,031 of 253,442 labelled metric-visible rows). Pipeline calibrated against our frozen
geometry (`daughter_angle` 0.2315/0.3594 vs 0.2303/0.3587).

| feature | src | 44b6 | 6bba | min_sep |
|---|---|---|---|---|
| **m_massratio_p1** | IMAGE | 0.1758 | 0.1978 | **0.3022** |
| G_resid_ratio | GEOM | 0.1732 | 0.2537 | 0.2463 |
| G_flow_midpoint_residual | GEOM | 0.1904 | 0.2577 | 0.2423 |
| m_saddle_p1 | IMAGE | 0.2004 | 0.2610 | 0.2390 |
| m_massratio_p2 | IMAGE | 0.1637 | 0.2660 | 0.2340 |

`m_massratio_p1` -- the mother's background-normalised core mass at `t+1` over that at `t` -- is
the strongest single separator measured so far. All 16 image features reaching the top 24 are
sign-consistent. The winners tell one coherent biological story measured four ways: at `t+1` the
mother's site loses mass, becomes a saddle, spreads, and its intensity centroid displaces. They
are correlated, not four independent bits.

**The dimensionless hypothesis is REINSTATED for intensity features specifically.** I withdrew it
on geometry evidence (where sign flips split evenly between dimensionless and dimensional). For
appearance the split is clean: every flipped feature is raw-scale (`m_massn` 0.4536/0.5983,
`d_mass_mean` 0.3741/0.5103), every ratio feature transfers. Correct statement: self-normalisation
is load-bearing for INTENSITY features; it is not a general law about geometry.

**Precision vs break-even (leave-one-family-out, capacity-compliant 1-2 features):**

| arm | 44b6 | 6bba |
|---|---|---|
| geometry alone | 0.00-0.61% | 0.50% |
| image alone | 0.61% | 1.64% |
| geometry + image | **1.82%** | **1.64%** |
| break-even | 4.07% | 6.38% |

Shortfall narrows from `5.9x / 12.8x` to **`~2.2x / 3.9x`**. Still a bilateral FAIL.

**Agent 3 flagged a temptation and I am honouring it:** at 6-10 features the point estimates DO
cross break-even on both families (n_feats=10: 44b6 42.5%, 6bba 13.1%). That must NOT be funded --
it rests on 3-4 events, is erratic and non-monotone across `n_feats`, and violates the
10-events-per-variable limit outright. It is threshold-and-hyperparameter mining on a handful of
events.

Caveat: 6bba's negatives come only from positive-bearing crops; on 44b6 (where both crop types are
sampled) the FP rate differs 0.55x-2.3x by crop type, so 6bba precision could be off ~2x either
way. 44b6's negative sample is sound.

**Runtime:** 1.31 ms/node measured uncontended; ~2.2 h single-core for all 5.1M nodes, ~45-70 min
on 2-4 workers. Optimisation from a naive 567 s/crop (31 h) to 68 s/crop via a radius-sorted
sphere kernel with edge-padded frames and moments to order 4 from a single BLAS gemm. Parallelism
scales poorly -- zarr blosc decompression dominates.

**Verdict: CONTINUE.** Appearance is the right residual and the mechanism is biologically legible
rather than a fitted artifact, but on a 92-positive label budget it does not reach deployability.
The binding constraint remains labels, not features -- which is precisely what H1-T (external
pretraining) exists to attack.

### 2026-07-31 (H1-T RED — and it REFUTES my "labels are the binding constraint" claim)

**CORRECTION 4, and this one redirects the programme.** I have repeatedly written that the binding
constraint on the division track is LABELS, not features -- most recently in the H1-I entry above.
Agent 4 tested that claim directly by obtaining **28x more positives** and it did not help. The
claim is refuted. **The missing information is MODALITY, not label count.**

**External data compliance.** Zebrahub (Lange et al., Cell 2024, doi:10.1016/j.cell.2024.09.047)
is **CC BY 4.0**, anonymous public HTTPS, no EULA, attribution required. Journal 2026-07-03 records
that competition rules permit public external data. HONEST CAVEAT: the live rules page is
JS-rendered and the stored API key 401s, so the clause could not be re-read programmatically --
**a human must re-check before any submission that depends on external data.**
**Leakage check performed and NEGATIVE:** competition voxel size (1.625, 0.40625, 0.40625) um vs
Zebrahub (1.24, 0.439, 0.439) um differs on every axis, so competition movies are not re-crops of
ZSNS001-005.

**Event set.** 1,484,482 candidates / **2,587 positives** / 1,422,906 reliable negatives across
ZSNS003/004/005, provenance-preserving (embryo, track ids, frame, voxel and um coordinates).
Candidates generated by importing `phaseb_h0c_replay.shortlist` VERBATIM (cfg `04eeac97500d`) onto
a GT-free pseudo-prediction graph, so parent-stealing rate matches deployment (0.96 vs 0.97).
Feature block proven byte-identical to deployed H1-G (76,767 rows, max abs diff < 1e-12).
ZSNS001 was EXCLUDED before any model was fitted: it claims 635,041 divisions (3.2%/frame vs
0.44-0.70% elsewhere) with mean track length 13.5 frames -- fragmentation re-linked as forks.

**Model.** 24->48->24->1 MLP, 2,401 parameters. Kaggle kernel
`aryaarun07/biohub-h1t-external-fork-critic` v1 COMPLETE in 70 s on CPU (GPU not justified).
External LOEO ROC-AUC **0.9935**, PR-AUC **0.427** at natural prevalence 1.4-3.0e-3.

**Cross-family transfer onto our 92 positives vs 253,350 reliable negatives:**

| scorer | 44b6 ROC / best divJ | 6bba ROC / best divJ |
|---|---|---|
| frozen residual (no learning) | 0.807 / 0.0018 | 0.732 / 0.0027 |
| external linear | 0.872 / 0.0048 | 0.752 / 0.0027 |
| external MLP | 0.762 / 0.0044 | 0.688 / 0.0032 |
| external GBDT | 0.799 / 0.0063 | 0.668 / 0.0027 |
| E0c baseline / H0c oracle | 0.0000 / **0.6154** | 0.0057 / **0.6080** |

Sign consistency PASSES (ROC > 0.5 both families, every arm). Deployability FAILS: every
cross-fitted arm is at or below E0c bilaterally, and under the min-fold correction (`dee0fc5`,
~15/85 edge mass) it is worse, because 6bba is where every arm loses.

**What killed it, precisely.** The critic answers *which pair* (top-1 0.812/0.724) but not
*which mother*: **of the top 100 mothers by score, ZERO are true dividers in either family.**
Domain shift is mild (0.30/0.35 SD, 17/24 features sign-consistent) -- the same features are just
2-3x less discriminative on E0c's messier graph. Composite bounded at `+0.0006 / -0.0003`, so the
199-crop exact scoring was correctly skipped; `h1t_h0c_learned.py` is validated (104 s/crop, node
counts invariant) if the exact number is ever wanted.

**Independent confirmation of H1-I.** On the hardest pool (92 positives vs the 20 highest-geometry
reliable negatives per crop, where geometry sits at AUC 0.21/0.04 by construction), appearance
reaches **cross-family AUC 0.693 / 0.750**, fit on one family and tested on the other. Nuclear
intensity SD 0.60/0.76; daughter/mother intensity 0.75/0.72 inverted; compactness 0.55/0.65 --
directions physically correct for mitosis (condensed, bright, compact mother, brighter than its own
daughters). Two agents reached this independently from different data.

**Zebrahub imagery IS obtainable** (verified by decoding real chunks): OME-Zarr v0.4, 3 levels,
0.8-2.3 MB/s; a 100-frame ZSNS003 level-1 pilot costs ~1.2 GB / under 30 min. But the only
patch-friendly embryo is the excluded low-quality ZSNS001. Do NOT download external imagery or
spend GPU before the CPU-only appearance work is exhausted.

**PROCESS FAILURE (mine).** My `git add -A scripts/ reports/` in commits `dee0fc5` and `9f250df`
swept agent-authored scripts into the tree that I had not read. Committing unreviewed code is
exactly the discipline this project otherwise enforces. The files are inert (nothing imports them
from the deployed path) but they must be reviewed or removed before anything depends on them.
Stop using `git add -A` with agents running; stage explicit paths only.

### 2026-07-31 (H0d live-filter + portability -- the node-invariance error runs the FAVOURABLE way)

**TASK 1 -- corrected H0c under the real wrapper. My flagged risk was wrong in sign.**
`scripts/win_bet/phaseb_h0d_livefilter.py` replays the frozen cascade then re-runs the real
`wrapper.filter_short_track_components` and the `OUTPUT_PRUNE_ISOLATED` block. `shortlist`/`CFG`
imported verbatim from `phaseb_h0c_replay` (hash `04eeac97500d`); nothing retuned.

| arm (E0c) | 44b6 | delta | 6bba | delta |
|---|---:|---:|---:|---:|
| baseline | 0.7595 | - | 0.6490 | - |
| h0c, no re-filter (published) | 0.8221 | +0.0625 | 0.7086 | +0.0597 |
| **h0c, real live filter** | **0.8237** | **+0.0641** | **0.7135** | **+0.0646** |
| h0c + retention guard | 0.8237 | +0.0641 | 0.7134 | +0.0645 |
| suppress-only, no re-filter | 0.7595 | -0.0001 | 0.6470 | -0.0020 |
| **suppress-only, live filter** | 0.7611 | **+0.0015** | 0.6518 | **+0.0029** |

- Node deletions 14,395 / 12,844; suppression alone 14,378 / 12,780 -- exactly my estimate.
  Count multiplier +0.00056 / +0.00068, also as estimated.
- **CORRECTION TO MY OWN RISK CLAIM.** I wrote that suppression's edge cost is UNCONDITIONAL
  while the division gain is CONDITIONAL, so "a weak classifier keeps the cost and loses the
  gain". Falsified: under the real filter suppress-only is **+0.0015 / +0.0029**, not negative.
  The published 6bba adjEdgeJ regression disappears entirely (rawJ -0.00053 -> **+0.00404**)
  because the deleted nodes were carrying metric-counted FP edges. The unconditional term is a
  small unconditional GAIN. H1's downside risk is materially lower than I reported.
- The `has_division` exemption I identified is a MINOR cause: 134 comps / 652 nodes (44b6) and
  122 / 586 (6bba) kept solely by it = 4.5% / 4.6% of deletions. ~95% comes from suppression
  fragmenting components already above min_len and orphaning children.
- **The retention guard is unnecessary and slightly harmful** (-0.00003 / -0.00009).
- TRAP: E0c built with `OUTPUT_MIN_TRACK_LEN = 7` (`e0c_run.py:53`), not the wrapper default 6;
  v122/clean903 use 6. Re-filtering at 6 makes the pass a no-op and hides 100% of deletions.
  Now read per crop from each surface's status manifest and asserted.
- Validity control: re-filtering the UNEDITED cached graph is the exact identity on all 199 crops
  x 4 surfaces. The no-refilter arm reproduces published H0c exactly.
- Un-modelled residual: `OUTPUT_LINEFIT_SMOOTH` deliberately not re-run (coordinate blend, not
  idempotent). Exposed set 17,383 / 13,210 nodes.

**TASK 2 -- portability. The proposer transfers; the CEILING does not.**

| surface | fam | N_pred | forks | reach/GT | retain | ret% | shortlist | del(supp) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| e0c | 44b6 | 2,864,419 | 11,441 | **20**/26 | 16 | 80.0 | 8,284,112 | 14,378 |
| e0c | 6bba | 2,253,622 | 9,012 | **93**/125 | 76 | 81.7 | 6,086,890 | 12,780 |
| v122 | 44b6 | 2,123,909 | 4,667 | **15**/26 | 11 | 73.3 | 5,865,670 | 4,092 |
| v122 | 6bba | 1,846,035 | 5,508 | **68**/125 | 54 | 79.4 | 4,699,347 | 5,863 |
| clean903 | 44b6 | 2,903,338 | 11,054 | **20**/26 | 17 | 85.0 | 8,402,579 | 10,731 |
| clean903 | 6bba | 2,294,949 | 8,696 | **96**/125 | 79 | 82.3 | 6,212,738 | 9,730 |
| ilp_c | 44b6 | 2,088,794 | 4,162 | **15**/26 | 12 | 80.0 | 5,756,033 | 5,147 |
| ilp_c | 6bba | 1,813,436 | 5,264 | **67**/125 | 53 | 79.1 | 4,598,893 | 7,451 |

Scored ceilings under the live filter: **E0c +0.0641 / +0.0646** (divJ 0.6154 / 0.6080) vs
**v122 +0.0419 / +0.0461** (divJ 0.4231 / 0.4320). v122 base composites reproduce arm-D
0.6962 / 0.6997 exactly.

- **One frozen proposer DOES operate unchanged on every surface** -- identical function object,
  identical schema, no retune, no failure, zero GT-map collisions across all 8 blocks, shortlist
  inside the 15M budget everywhere (4.60M-8.40M).
- **But the ceiling is substrate-dependent.** v122 loses 25% / 27% of REACHABLE divisions purely
  because its node set drops mothers or daughters. Retention GIVEN reachability is nearly
  preserved (73.3%/79.4% vs 80.0%/81.7%): **the ranker transfers, the node population is the
  limiter.**
- `ilp_c` (v122's ILP + E0c's wrapper) also shows 15/26 and 67/125, so **the reachability loss
  originates in the ILP/detector stage, not the v122 wrapper.**
- **clean903 (greedy detector population) is the best substrate measured: 20/26 and 96/125** --
  better than E0c on 6bba. The division layer should be deployed on a greedy-detector base, NOT
  on v122's ILP-pruned population.
- `has_division` retention is essentially INERT on v122/ilp_c (0-1 exempt comps vs 122-134 on
  E0c), and the multiplier direction FLIPS (E0c over-predicts, v122 under-predicts), so deletion
  economics differ 3-4x by surface.

Mother-collision assertion added in `build_gt_maps` (raises by default). 0 collisions measured.
Unguarded `gt_to_sub[m] = s` overwrites remain at `phaseb_d0p_proposer.py:287`,
`phaseb_h0c_replay.py:147`, `phaseb_h0b_rankcompress.py:138`, `phaseb_h1a_census.py:87`,
`phaseb_oracle_d0prime.py:215`.

Caveat: agent 1's clean-0.913 base has no local OOF graphs yet, so `clean903` is the closest
available proxy, labelled as the superseded port rather than the 0.913 base itself.

### 2026-07-31 (POOLED-OBJECTIVE HARD LOCK — proven, tested, and my framing corrected)

`scripts/verify_pooled_objective.py` proves the objective against the authoritative patched
scorer over the 995 cached per-crop rows. `tests/test_pooled_objective.py` locks it (4 tests;
suite now 18 passed).

**PROOF 1+2 — parity and hand reconstruction.** One combined `summarise()` over all 199 crops
equals a by-hand rebuild from raw global totals to machine precision (max |diff| `2.2e-16`
across all five arms). The weighting is therefore explicit and auditable, not trusted:
`adj_edge_jaccard` is an edge-VOLUME-weighted mean of per-sample adjusted Jaccards
(`w = TP+FP+FN`), and `division_jaccard` is micro-pooled. Published per-family anchors
reproduce (E0c 0.7595/0.6490, v122 0.6962/0.6997).

**PROOF 3 — family averaging is NOT the objective.**

| arm | pooled | mean(fam) | gap | min-fold | gap |
|---|---:|---:|---:|---:|---:|
| E0c | 0.66539 | 0.70425 | **+0.03886** | 0.64895 | -0.01644 |
| B_detpop | 0.66677 | 0.70046 | +0.03369 | 0.65256 | -0.01420 |
| Bp_ilp | 0.69162 | 0.70633 | +0.01471 | 0.68538 | -0.00624 |
| C_survival | 0.69692 | 0.69463 | -0.00228 | 0.69139 | -0.00553 |
| v122 | 0.69909 | 0.69795 | -0.00114 | 0.69625 | -0.00285 |

44b6 edge-mass share is 14.81-14.97% across arms.

**CORRECTION TO MY OWN FRAMING.** I wrote that "the min-fold gate is not the competition
objective" and implied min-fold RANKING misordered candidates. My first version of proof 4
tested exactly that and **FAILED**: min-fold ranking and pooled ranking give the identical order
here (v122 > C_survival > Bp_ilp > B_detpop > E0c). The criterion that actually closed arms was
not min-fold ranking but the **promotion gate**: delta versus E0c must be positive on BOTH
families, min-fold delta >= +0.005. That is the thing that diverges from the objective.

**PROOF 4 (corrected) — the bilateral-delta gate rejects arms the objective prefers.**

| arm | pooled d(E0c) | 44b6 d | 6bba d | bilateral gate | pooled says |
|---|---:|---:|---:|---|---|
| v122 | **+0.03370** | -0.06330 | +0.05071 | REJECT | BETTER |
| C_survival | **+0.03153** | -0.06816 | +0.04893 | REJECT | BETTER |
| Bp_ilp | **+0.02623** | -0.03227 | +0.03643 | REJECT | BETTER |
| B_detpop | +0.00137 | -0.01120 | +0.00361 | REJECT | BETTER |

**All four non-E0c arms are pooled-better and all four were rejected.** v122 -- our best public
score at 0.908 -- is among them. The gate rejected the winner.

**SELECTION DOCTRINE CHANGED.**
- PRIMARY: official exact pooled OOF composite.
- MANDATORY DIAGNOSTICS alongside every claim: both family scores, per-family delta,
  block-bootstrap uncertainty, pseudo-private rank-reversal risk, worst-regime behaviour.
- Min-fold is a ROBUSTNESS CONSTRAINT, not the optimisation target.

The regression test fails if anyone restores 50/50 family weighting or a bilateral-delta gate as
the primary criterion, and asserts on the real cache that 44b6's edge-mass share stays in
(0.10, 0.20).

**Re-ranked historical arms under the corrected objective** (`inventory/pooled_objective_parity.json`):

| rank | arm | pooled OOF | 44b6 | 6bba | public | old status | corrected status |
|---|---|---:|---:|---:|---:|---|---|
| 1 | v122 | **0.69909** | 0.69625 | 0.69966 | 0.908 | hedge only | **best pooled arm** |
| 2 | C_survival | 0.69692 | 0.69139 | 0.69788 | - | CLOSED | reopened, pooled +0.0315 |
| 3 | Bp_ilp | 0.69162 | 0.72728 | 0.68538 | - | CLOSED | reopened, pooled +0.0262 |
| 4 | B_detpop | 0.66677 | 0.74835 | 0.65256 | - | CLOSED | marginal, pooled +0.0014 |
| 5 | E0c | 0.66539 | 0.75955 | 0.64895 | 0.889 | authoritative anchor | decorrelated hedge |

Caveat carried forward from the simulator: `C_survival` is strictly dominated by v122 in
20,000/20,000 composition draws (per-crop error correlation 0.9995) and must never occupy a
submission slot despite its pooled rank.

### 2026-07-31 (Lane D — H1-T CLOSED PERMANENTLY on a structural argument, not a threshold)

H1-T was already RED as a mother-level detector. This lane asked the only remaining question:
does it survive as the SECOND factor, `P(pair | mother divides)`? Answer: no, in either family.
`scripts/win_bet/h1t_conditional_ranker.py`; results in `agent_runs/laneD/laneD_conditional_ranks.json`.

Conditioning set: the 92 mothers that genuinely divide AND whose true pair survived into the
frozen top-3 shortlist (16 in 44b6, 76 in 6bba; 259 candidate pairs). Every candidate belonging
to a conditioning mother is `metric_visible` and labelled, so the comparison is like-for-like
with no label-driven set truncation. Calibration-free ranks within each mother's choice set; no
global threshold anywhere.

| family | ranker | rank@1 | 95% CI | MRR |
|---|---|---|---|---|
| 44b6 (16) | **frozen geometry** | **14/16 = 0.875** | [0.640, 0.965] | **0.9375** |
| 44b6 | H1-T MLP | 13/16 = 0.812 | [0.570, 0.934] | 0.9062 |
| 44b6 | H1-T linear | 13/16 = 0.812 | [0.570, 0.934] | 0.9062 |
| 44b6 | H1-T GBDT | 14/16 = 0.875 | [0.640, 0.965] | 0.9271 |
| 6bba (76) | **frozen geometry** | **58/76 = 0.763** | [0.656, 0.845] | **0.8684** |
| 6bba | H1-T MLP | 55/76 = 0.724 | [0.614, 0.812] | 0.8487 |
| 6bba | H1-T linear | 55/76 = 0.724 | [0.614, 0.812] | 0.8509 |
| 6bba | H1-T GBDT | 54/76 = 0.711 | [0.600, 0.800] | 0.8377 |

Paired McNemar on identical mothers: all p >= 0.39. Correctly read as "H1-T fails to beat
geometry", NOT "H1-T is significantly worse". Restricting to mothers with a genuine choice
(n>=2 candidates; 15 and 72) changes nothing: 13/15 vs 12/15 and 54/72 vs 51/72, both favouring
geometry.

**THE STRUCTURAL FINDING THAT MAKES THIS PERMANENT: in 44b6, b = 0.** There is not one mother
where H1-T ranks the true pair first and geometry does not. H1-T's correct set is a STRICT
SUBSET of geometry's. So the union oracle -- the best any blend, gate, ensemble or cascade of
the two scorers could achieve with a perfect selector -- is **14/16 = 0.875 in 44b6, identical
to geometry alone**. Under the both-families rule, no weighting can pass. The arm closes
structurally, not at one operating point.

Two reporting notes worth keeping:
- **rank@3 is degenerate here and must never be quoted as success.** The shortlist is top-3 per
  mother, so every ranker scores 100% by construction. Only rank@1 (marginally rank@2)
  discriminates.
- Machinery validated: a `residual_as_score = -flow_midpoint_residual` control reproduced the
  frozen `rank` column's ordering event-for-event across all 92 mothers, and the H1-T top-1
  figures reproduce the prior 0.812 / 0.724 exactly from the frozen weight file -- no
  retraining, no drift. Zero score ties, zero residual ties.

Why this was the expected outcome: the conditioning set is "the true pair is among the
residual's own top 3", so geometry has already demonstrated competence on exactly these events.
The residual is not a weak incumbent -- it is the third-largest effect in H1-T's own 24-feature
block and it is the feature that BUILT the shortlist. H1-T is a noisier re-weighting of a set
the incumbent already ordered well.

There is also no headroom worth chasing: a PERFECT conditional pair ranker adds at most 2
divisions in 44b6 and 18 in 6bba, and only on mothers the first factor admits. The first factor
is where H1-T is dead (0/100 true dividers in the top-100 mothers, both families). A perfect
second factor multiplied by a zero first factor is still zero.

**VERDICT: close the H1-T external trajectory critic permanently in both roles.** The frozen
flow-midpoint residual remains the incumbent pair ranker and is not displaced. This does NOT
touch the H1-T appearance probe (different feature space, different lane), and the reusable
infrastructure (`h1t_zebrahub_events.py`, `h1t_h0c_learned.py`, the external event set) stays
valid -- it is the trajectory-feature critic that is closed.

Consequence for the factorised architecture: `P(mother divides)` remains entirely unsupplied,
and it is now the ONLY missing factor. Lane C (H1-M) is the sole live attempt at it.

### 2026-07-31 (Lane C — H1-M: the FIRST bilaterally positive mechanism; Zebrahub falsified for divisions)

Mother-level gate `P(divides | mother-centred evidence)`, one row per `(crop, mother, t)`, no
pair duplication. Scripts `scripts/h1m_{features,mother_events,zebrahub_appearance,gate,audit}.py`;
full write-up `agent_runs/laneC/H1M_FINDINGS.md`.

**Competition dataset** (`comp_mother_events.parquet`, 199 crops):

| family | mother events | dividers | realisable | hard negatives |
|---|---:|---:|---:|---:|
| 44b6 | 18,553 | 24 | 16 | 6,740 |
| 6bba | 79,039 | 113 | 76 | 28,196 |

Matched negatives 20/divider, same crop, nearest in standardised (t, density, core mass, track
age, speed); balanced to |SMD| <= 0.081.

**RESULT — in-family crop-grouped CV, cross-fitted admission budget, no threshold chosen on the
test family: `+0.00396 / +0.00201` exact composite.** For comparison H1-T was `+0.0006 / -0.0003`.
Best single features are mother-only and self-normalising: `R_massn_p1` AUC 0.242/0.209,
`D_rg_p1` 0.771/0.707. A confirmatory GBDT arm was WORSE at the tail
(`+0.00058 / -0.00261`) -- capacity fits idiosyncrasy, exactly as the EPV ceiling predicted.

**This is the first mechanism in the campaign that is bilaterally positive on exact composite
under an honest cross-fitted rule.** It is below the old per-family `+0.005` bar, but that bar is
no longer the primary objective -- the pooled response is (Lane B is measuring it).

**ZEBRAHUB IS FALSIFIED AS A DIVISION CORPUS, with a mechanism.** External LOEO AUC 0.692
(0.681/0.704/0.692) -- but transferred to competition data it is **AUC 0.494 / 0.532, i.e.
CHANCE**, cross-fitted `+0.00034 / -0.00244`. All the usual suspects were audited and PASS:
coordinates verified on both domains (Zebrahub true-vs-null AUC 0.941, core mass 130,417 vs 0),
domain shift tiny (mean |SMD| 0.210, ZERO of 77 features above 1 SD), disjointness/sampler/
determinism clean.

The decisive measurement -- forward one-step core-mass collapse, re-anchored to allow annotation
offset:

| dataset | anchor 0 | anchor -1 | anchor -2 | rg growth |
|---|---:|---:|---:|---:|
| competition 44b6 | **-0.981** | +0.049 | -0.175 | **+0.926** |
| competition 6bba | **-0.723** | -0.031 | -0.009 | **+0.649** |
| ZSNS003/4/5 (best of 4 windows) | -0.195 | +0.206 | +0.135 | +0.168 |

**Zebrahub has no mitotic signature at any anchor** -- 4.75x amplitude shortfall, and its only
cross-embryo-consistent effect has the WRONG SIGN. Two measured causes: (1) it is temporally
over-sampled, true daughter separation at t+1 is 5.7 um vs the competition's 10.3 um, so both
nascent daughters are still inside the 3 um core; (2) its fork frames come from an AUTOMATED
tracker, so `t0` marks when detections resolved, not when the cell divided -- the competition GT
is human-curated. **More imagery cannot fix a label-timing defect.** This also retrospectively
justifies excluding ZSNS001 (635,041 divisions against a +15,790 net cell gain implies ~619k
terminations; 3.2%/frame vs 0.44-0.70%) -- though the retained embryos fail for a reason ZSNS001
would share anyway.

Imagery cost was 10x cheaper than the prior estimate: 7.52 GB of level-0 OME-Zarr streamed and
decoded in 828 s (9.1 MB/s). No padding; out-of-slab events dropped and counted.

CAVEAT CARRIED FORWARD: `realisable` assumes H0c's ORACLE pair selection among the top 3. The
frozen top-1 ranker retains only 70% / 62%, so the deployable number is BELOW the figures above.
Suppression's unconditional edge cost is already charged throughout.

NEXT CORPUS: needs frame-accurate, human-curated division times. Cell Tracking Challenge
Fluo-N3DH / Fluo-N3DL are worth a cheap provenance check before any further external spend.

### 2026-07-31 (Lane E — correctness closure; and a DISPUTED count in our own baseline)

Tests 18 -> **30 passed**. Modified `src/biotrack/wrapper.py` and the five proposer scripts;
added `scripts/{repair_linefit_volume,verify_gt_collision_parity,pooled_bootstrap}.py`,
`scripts/win_bet/gt_collision.py`, `tests/test_{gt_collision,linefit_volume_guard}.py`.

**P0-C repaired by RESTORE, not clamp -- with a uniqueness proof.** Root cause:
`linefit_smooth_output_graph` blends toward a line fit over the +/-2 neighbourhood; node 15274 is
a track ENDPOINT (in-degree 0) so its window is one-sided `[0,+1,+2]` and the fit EXTRAPOLATES,
coefficients `0.8667*o0 + 0.2667*o1 - 0.1333*o2` -- the negative term pushes past the face.
Originals were never persisted, so they were recovered: the smoother is affine with
topology-determined coefficients over integer voxel inputs, and a DP over the 15-node unique-degree
chain enumerated every integer preimage. For z the feasible set is exactly **{63}** -- proven, not
guessed. (y gave {183,184} and x {247,248,249}, so uniqueness is a real certificate, not
automatic.) Repaired output PASSES all 10 gates; 1 node re-keyed, 0 net nodes/edges/divisions.

**Pile-up audit inverts the intuition: smoothing RELIEVES boundaries.** Face occupancy falls in
every crop and axis (44b6_0b24845f z=63: 224 -> 212). So the 369-nodes-on-z=63 pile-up is a
DETECTOR artifact, and a clamp repair would have pushed in exactly the wrong direction.

**DISPUTED -- out-of-volume coordinates in the E0c baseline.** Lane E reports 7,349 nodes
(0.144%), **all non-integral**, concluding smoothing is 100% responsible. My own independent
recount over the same cache disagrees:

| source | nodes OOV | z | y | x | integral |
|---|---:|---:|---:|---:|---:|
| Lane E | 7,349 | 102 | 3,132 | 4,115 | **0** |
| my recount | **14,319** (0.2798%) | 5,250 | 4,023 | 5,168 | **1,658** |

Same total node count (5,118,041) and same z range (-0.800 .. 63.667), so this is an aggregation
or bound difference, not a different dataset. The disagreement matters because 1,658 INTEGRAL
violations would mean smoothing is NOT the sole cause and some detector coordinates are
themselves out of volume. **Unresolved -- do not cite either figure as settled until reconciled.**

What both agree on, and what matters: **E0c's published 0.7595 / 0.6490 were measured with
thousands of out-of-volume coordinates present.** They stayed invisible because the kernel writer
emits `max(0, int(round(v)))`, silently clamping the low side; only a high-side leak is auditable,
which is exactly how P0-C's z=64 escaped.

**Volume guard defaults OFF** (`BIOHUB_OUTPUT_VOLUME_GUARD=0`). `e0c_run.py:163` calls
`filter_output_graph`, so defaulting it on would move thousands of coordinates and silently shift
the baseline the whole promotion gate is defined against. **That is a decision to re-measure E0c,
not a bug fix, and it needs a human call.** Set `=1` for any artifact that must pass the
structural audit.

**Five `gt_to_sub` sites guarded, outputs PROVEN identical.** The guard raises rather than
resolving, so a completed run proves it never fired. `verify_gt_collision_parity.py` monkeypatches
the exact pre-fix loop and compares full result dicts: both folds, 10 runs, all 5 sites IDENTICAL.
(First pass flagged one as different; the sole difference was the wall-clock `runtime_s` field.)

**Script classification: nothing DEAD.** AUTHORITATIVE: `audit_submission_structure.py`,
`private_split_simulator.py`, `private_portfolio_risk.py`, `phaseb_h0d_livefilter.py`. The other
ten are EXPERIMENTAL. Zero writes into `artifacts/`. One hazard flagged:
`private_portfolio_risk.py` hardcodes its output dir and would clobber another lane's file.

**COMPETITION RULE FOUND (verbatim), settling the P0-A licence question.**
`https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/rules`, accessed
2026-07-31. Winner License: **MIT**. Data: **CC0**. Section 3.6.b, Public Code Sharing:

> "b. Public Code Sharing. You are permitted to publicly share Competition Code, provided that
> such public sharing does not violate the intellectual property rights of any third party. If you
> do choose to share Competition Code or other such code, you are required to share it on
> Kaggle.com on the discussion forum or notebooks associated specifically with the Competition for
> the benefit of all competitors. By so sharing, you are deemed to have licensed the shared code
> under an Open Source Initiative-approved license (see www.opensource.org) that in no event
> limits commercial use of such Competition Code or model containing or depending on such
> Competition Code."

So `saitejabandaruin/biohub-top-notebook-0-913` is deemed OSI-licensed **by operation of the
rules**; the absent API metadata field is irrelevant. The P0-A licence question is CLOSED.

**Bootstrap (20,000 crop-block draws).** All five marginal CIs OVERLAP; only paired deltas
separate: v122 0.69909 [0.67355, 0.72283]; C_survival 0.69692; Bp_ilp 0.69162 [0.66720, 0.71438];
B_detpop 0.66677; E0c 0.66539 [0.64087, 0.68875]. P(v122 rank 1) = **99.5%**.
Bp vs C_survival (CI [-0.01121, +0.00081]) and E0c vs B_detpop ([-0.00388, +0.00100]) are NOT
separable.

**Portfolio: v122+Bp_ilp beats v122+E0c, but NOT significantly.** Bp is far MORE correlated with
v122 (0.61-0.88) than E0c (-0.17 to 0.60), yet its +0.026 level advantage outweighs the
correlation penalty: Bp wins p5 in 6/6 scenarios and E[max] in 5/6. Notably in density-only-wide
E0c is ANTI-correlated with v122 (-0.166) and still loses -- decorrelation does not cover a 0.026
deficit. BUT the margins (+0.0001 to +0.0011 E[max]) are **an order of magnitude below the
paired-delta crop-resampling SE (~0.003)**. The preference is conditional on this OOF cache, not
significant. C_survival confirmed strictly dominated (reversal 0.01%, corr 0.998, gain +0.00000).

### 2026-07-31 (Lane B — pooled break-even measured, not approximated)

`scripts/win_bet/phaseb_pooled_breakeven.py`. Edits applied to real graphs, scored by one
combined `summarise()` over all 199 crops. No 15/85 re-weighting of precomputed composites.

**DEFECT IN MY OWN SCRIPT'S DOCSTRING:** it claims "all of the above with the real live
wrapper/filter re-run after the edits". The implementation does NOT re-run the wrapper filter --
it scores the edited edge set against the cached node rows. These are therefore the NO-REFILTER
numbers. H0d showed the live filter is strictly more favourable (per-family `+0.0641/+0.0646` vs
`+0.0625/+0.0597`, and it flips suppress-only from negative to positive), so the pooled figures
below are a LOWER BOUND. The docstring is wrong and is corrected in the file.

pooled BASE = **0.66540** (cfg `04eeac97500d`)

| arm | pooled | delta | 44b6 | 6bba | divTP | divFP |
|---|---:|---:|---:|---:|---:|---:|
| base | 0.66540 | +0.00000 | 0.75955 | 0.64897 | 4 | 675 |
| supp | 0.66368 | **-0.00173** | 0.75946 | 0.64696 | 0 | 0 |
| k0.25 | 0.66770 | +0.00230 | 0.75946 | 0.65182 | 6 | 0 |
| k0.5 | 0.68180 | +0.01639 | 0.76728 | 0.66722 | 27 | 0 |
| k0.75 | 0.72147 | +0.05607 | 0.82208 | 0.70377 | 86 | 0 |
| **k1.0** | 0.72552 | **+0.06012** | 0.82208 | 0.70865 | 92 | 0 |
| fp1 | 0.70158 | +0.03618 | 0.79539 | 0.68527 | 91 | 92 |
| fp2 | 0.69087 | +0.02546 | 0.78477 | 0.67452 | 91 | 184 |
| fp5 | 0.67667 | +0.01127 | 0.76997 | 0.66041 | 88 | 460 |
| fp10 | 0.66853 | +0.00313 | 0.76120 | 0.65235 | 88 | 912 |
| fp25 | 0.65764 | -0.00776 | 0.74858 | 0.64172 | 87 | 2233 |

**REQUIRED PRECISION-AMONG-VISIBLE, pooled objective** (interpolated on measured points, not
derived from `J = k/(N+m)` algebra which was itself written under the old per-family bar):

| pooled target | max FP:TP ratio | implied precision |
|---|---:|---:|
| +0.002 | 11.55 | **7.97%** |
| +0.005 | 8.85 | **10.15%** |
| +0.010 | 5.78 | 14.75% |
| +0.015 | 4.21 | 19.19% |

**I EXPECTED THE POOLED BAR TO BE EASIER. IT IS HARDER.** The old per-family bar needed
4.07% / 6.38% precision for `+0.005` in a given family; pooled `+0.005` needs **10.15%**. That is
not a contradiction -- it is the point. `+0.005` on 44b6 alone was a ~1.3-mother test on a
26-mother population (agent 5), i.e. two lucky mothers. The pooled bar demands a real aggregate
gain, so it is both harder and far more meaningful. My speculation that the correction might make
H1-I trivially promotable was wrong.

**PER-DIVISION MARGINAL VALUE -- the asymmetry inverts under the correct objective.**

| family | divisions | family delta | per division (family) | per division (pooled) |
|---|---:|---:|---:|---:|
| 44b6 | 16 | +0.06253 | **+0.003908** | ~+0.000587 |
| 6bba | 76 | +0.05968 | +0.000785 | ~+0.000667 |

In FAMILY terms a 44b6 division looks **5.0x** more valuable. In POOLED terms they are nearly
equal, and 6bba divisions are marginally MORE valuable. The 15/85 mass split almost exactly
cancels the 26-vs-125 division-count split. This kills any temptation to over-weight 44b6 work,
and it means division effort should follow whichever family is easier to detect in, not whichever
shows the larger family-delta.

**Suppress-only is `-0.00173` pooled** in this no-refilter variant. H0d showed the live filter
turns it positive per-family, so the deployable suppression step DEPENDS on re-running the real
wrapper filter. That dependency is now load-bearing rather than cosmetic.

**H1-M ASSESSED AGAINST THIS BAR.** Lane C's cross-fitted gate gave `+0.00396 / +0.00201`
per-family, which is approximately **`+0.0023` pooled**. That sits just above the `+0.002` line
and far below `+0.005`. Against agent 5's crop-resampling paired-delta SE of ~0.003, **+0.0023 is
inside the noise band**. H1-M is pooled-positive but NOT statistically distinguishable from zero
at this scale.

**DECISION: slot 3 stays HELD.** No candidate clears a credible pooled margin. H1-M is the best
mechanism the campaign has produced and it is still not a submission.

### 2026-07-31 (Node budget + H1-M forensics + H1-M2 close — three results, one crosses the bar)

**NODE-BUDGET SWEEP (mine, `scripts/win_bet/phaseb_node_budget.py`).** RT surfaced the unclipped
count multiplier; this tests whether selective pruning pays. Weakest-component-first
(short first, division-containing components protected), deployment-observable ranking only.
12-crop smoke on E0c:

| keep | pooled | delta | 44b6 | 6bba | ratio | recall |
|---|---:|---:|---:|---:|---:|---:|
| 1.000 | 0.73328 | +0.00000 | 0.83553 | 0.71046 | +0.1171 | 0.9499 |
| **0.975** | **0.74150** | **+0.00822** | 0.83335 | 0.72090 | +0.0890 | 0.9397 |
| 0.950 | 0.74115 | +0.00787 | 0.83108 | 0.72091 | +0.0610 | 0.9330 |
| 0.900 | 0.72705 | -0.00623 | 0.80275 | 0.70991 | +0.0051 | 0.8934 |
| 0.800 | 0.70438 | -0.02890 | 0.73297 | 0.69774 | -0.1067 | 0.8273 |

Pooled optimum at **0.975 = +0.00822**; per-family optima 44b6 1.0, 6bba 0.95. **KILL RULE NOT
TRIGGERED** (optimum >= 0.99 on BOTH families would have closed it). This is a larger single
number than anything the entire H1 programme has produced, on 12 crops. Full 199-crop run
launched. Caveat: E0c over-predicts (+0.0832 ratio) so it has the most to gain; v122 already sits
at -0.1598 and may have no headroom. The full run and an arm-D repeat decide it.

**H1-M FORENSICS (agent 7).** Preregistered before outcomes (`PREREG_HASH 6bbff7f83541a917`,
amended pre-outcome for a coverage defect, with signal equivalence VERIFIED not assumed:
`R_massn_p1` vs `m_massratio_p1` Spearman 1.000000, residual max abs diff 0.0). Reproduction
bit-identical before any forensics ran.

The decision set is **218 mothers (16 TP / 202 FP), precision 7.34%**, not the 27 I quoted.
Error taxonomy of the 202 FPs: **CALIBRATION 131 (64.9%)**, RANKING 44, TEMPORAL_PHASE 26,
SUBSTRATE 1. And **118 of the 131 calibration errors sit in crops containing ZERO realisable
divisions** -- one crop absorbed 51 admissions with 0 TP. The dominant failure is not
discrimination at all.

**MY +0.005 TARGET WAS WRONG.** The 10.15% precision figure was derived AT FULL RECALL. Corrected
frontier: +0.005 needs >=10% precision held out to **~37 of 92 divisions**. H1-M's precision
collapses with recall (13.75% at TP=11, 11.38% at 14, 5.29% at 20, **2.25% at 37**). So the gap is
a **4.5x shortfall at the recall that matters**, not 1.4x at the recall we operate at. Composition
C reaches 18.52% precision (both families above 10.15%) and still only makes +0.00109.

Overlap matrix (Jaccard of removed-FP sets): H1M-H1I **0.380** (most redundant -- `R_massn_p1` IS
an H1-M feature), H1M-GEOM 0.271 (most complementary). Preregistered winner is **A (H1-M alone,
+0.00248)**; B (H1-I gated) LOSES to doing nothing, exactly as the matrix predicted. Agent 7 also
falsified the matrix as a PREDICTIVE instrument: it gets the sign at the extremes but not the
ordering, because complementarity must be weighted by standalone power (RTD_A has AUC 0.492 on
44b6 -- chance -- so its low overlap buys nothing).

**THE CHANNEL THAT CROSSES THE BAR — TEMPORAL SNAP.** 65 of A's 202 FPs (**32%**) are duplicate
admissions of the SAME predicted track within +/-2 frames.

| arm | TP | FP | precision | pooled delta |
|---|--:|--:|--:|--:|
| no snap (= A) | 16 | 202 | 0.0734 | +0.00248 |
| **GT-free per-track dedup (W=2)** | 16 | **137** | **0.1046** | **+0.00369** |
| ORACLE frame choice (ceiling) | 22 | 131 | 0.1438 | **+0.00599** |

Dedup uses NOTHING but the frozen predicted graph, is deployable today, buys **+0.00121 free**,
and crosses 10.15%. Robust across the window (+0.00306/+0.00369/+0.00316 at W=1/2/3; W=2 was
preregistered). The ORACLE frame choice reaches **+0.00599, above the bar** -- and none of
H1-I / RTD_A / RTD_G / GEOM can select the frame. **A within-track frame-selection signal is the
only measured route to +0.005.**

`scripts/h1n_exact_replay.py` built and smoked (4 crops: division TP 1->2, FP 11->8, composite
0.74120->0.74979); 147.5 s/crop, 199 crops on 6 workers ~82 min. Compute-gated.

**H1-M2 CLOSED, and it corrects lane C's diagnosis.** Lane C blamed temporal over-sampling. That
is **WRONG**: the clocks are the same (Zebrahub 1.32-1.75 vs competition 1.82 um/frame) and
nuclear size is the same (L_rg 4.47-4.53 vs 4.47 um). The anchor was repaired anyway
(34.8% of 92,172 forks realigned to a competition-like configuration, rule chosen from track
geometry alone so the A/B is not circular) and the mitotic signature **did not recover**:
R_massn effect stays +0.300 / -0.172 / +0.026 against competition's **-1.393 / -1.551**, wrong
sign on two of three embryos. The killer control is an ORACLE over all 30 (anchor, stride) pairs
-- an upper bound no rule can beat -- giving **+0.330 / +0.003 / +0.224**. There is no anchor to
move to.

Real cause: **LINEAGE FRAGMENTATION.** 5.7-11.7 track TERMINATIONS PER DIVISION, so a Zebrahub
"fork" is predominantly a fragment-to-fragment assignment. Confirmed by what actually predicts a
Zebrahub fork: `massn`/`peakn` HIGHER and `coff` LOWER (AUC 0.59-0.62) -- bright, well-centred
detections are the ones the tracker splits. That is detector confidence, not mitosis.
Realignment did move transfer from anti-correlated to chance (44b6 AUC 0.308 -> 0.576) and cut
domain distance (dividers SMD 0.409 -> 0.328), so the repair worked; the corpus is still wrong.

**$0 PRE-FLIGHT FILTER for any future external corpus** (the reusable output of this lane):
require `term/div <~ 1` AND `sep(+1) >= 0.8 x 10.57 um` BEFORE downloading a single voxel.
No GPU was spent; 7.5 GB streamed on Kaggle CPU kernels in 373 s.

### 2026-07-31 (H2a hybrid census — 199 crops, GT-free candidate generation, 33 aux-only divisions recovered on 6bba)

Ran the agent-prepared census (`phaseb_h2a_hybrid_oracle.py --arms ""`, no scorer calls).
Wall 1126 s, per-crop mean 5.7 s, 4047 worker-seconds at 4 workers.

**6bba (128 crops), v122 main + clean903 aux:**
- GT divisions 125; **reach_main 68**, reach_aux 96, **RECOVERY (aux-only) = 33**, **union 101**.
- So the hybrid lifts reachable divisions from 68/125 to **101/125** -- a 48% increase over v122
  alone, and the union again EXCEEDS max(main, aux) = 96, confirming the not-nested finding at
  corpus scale.
- **27 of 33** recovered pairs already sit in the mother's frozen top-3 (cfg `04eeac97500d`), so
  the GT-free proposer would find them once the aux nodes exist.
- Auxiliary nodes added: **41** (7 mother / 34 daughter); triples needing 1/2/3 aux = 26/6/1.
- Duplicate collisions: 11 aux within 7.0 um of a same-frame main node (6 of those GT-matched),
  3 within 3.5 um, min nn distance 0.747 um -- close enough to need watching, but aux survival
  after the REAL wrapper filter is 41/41.
- Count multiplier moves the RIGHT way: `dmult +0.000288` on `hybrid_full`, because the wrapper
  deletes 5,922 nodes while we add 41.
- Local reassignment: steals main 49 / hybrid_full 56; suppressed edges 5,509 of which 5,508 are
  pre-existing forks. GT collisions 0 on both surfaces. Crops with >=1 recovery triple: 24/128.

**44b6:** aux added 9 nodes (N_pred 2,119,813 -> 2,119,822 between `main_oracle` and
`hybrid_full`), `dmult +0.000159`.

**Total auxiliary nodes across 199 crops: ~50.** The count-multiplier objection to this design is
now dead twice over -- at smoke scale and at corpus scale.

ORDER-OF-MAGNITUDE VALUE: at the measured pooled per-division value on 6bba (~+0.000667),
33 recovered divisions is worth roughly **+0.022 pooled** if perfectly selected. That is an ORACLE
figure -- the deployable fraction depends on the FP cost of running the proposer over the enlarged
surface, which remains unmeasured and is the thing that will actually decide it.

Full scored oracle (4 arms x 199 crops, est. 45-90 min wall) is the next command, still gated.

### 2026-07-31 (Deployment lane — P0-A CONTAINS AN ALL-TRAIN MODEL; P0-C repaired in-kernel and submittable)

**LEAKAGE FACT, recovered from P0-A's own kernel outputs.** Its secondary temporal model is
`unet_transformer_alltrain_seed314159_v1` with `"train_datasets": 199` -- it has seen EVERY
training crop -- and P0-A blends it at **0.475 detection weight**. The support pack ships only
`split_0` (trained on 6bba, 44b6 held out). Consequences:

- **P0-A is a PUBLIC DEPLOYMENT platform, not an OOF-valid one.** The ledger already said this;
  the exact mechanism and weight are now known.
- Only **fold 0 + arm `strict`** (pack primary only, secondary and DeepCenter OFF) yields a number
  comparable with the E0c / clean903 / v122 substrates. That is what is running.
- **Fold 1 is not comparable** without our own `split_1`; even with it, the two folds would run
  different-vintage models. The builder refuses fold 1 unless `--weights-glob` is passed.
- Leak size, measured: P0-A's ACTUAL submission scored against TRAIN GT gives composite **0.8904**,
  node recall **0.9835** on the placeholder crops. Arm `asis` is an upper bracket only.

**LOEO analogue built and running.** `notebooks/kaggle_loeo_f0_strict/` (slug
`aryaarun07/biohub-loeo-f0-strict`, config hash `9325239999ba`), generated by
`scripts/build_loeo_analogue.py`. It retargets the identical P0-A pipeline at fold 0's 71 held-out
44b6 crops and mounts ONLY the `.zarr` images -- GT `.geff` files are deliberately not linked, so
the kernel is incapable of reading a label. Kaggle 3-crop smoke PASSED (slug
`biohub-loeo-f0-strict-smoke3`, 4.97 predict-minutes, all coordinates in-volume, secondary and
DeepCenter confirmed off in the emitted manifest). Local scoring of the smoke: adjJ
0.9058 / 1.0375 / 0.9753, node recall 0.9808. `adjJ > 1` is LEGITIMATE -- the count multiplier
going above 1 when fewer nodes are emitted, the same unclipped mechanism the node-budget sweep is
now exploiting. **The first three fold-0 crops contain zero GT divisions, so reach was 0/0: the
smoke proves plumbing only.** Full fold started 18:51, ~2.0 h predict.

**P0-C REPAIRED IN-KERNEL AND IS NOW SUBMITTABLE.** Cause of its ERROR was pinned exactly: its own
in-kernel audit at t=874 s caught `44b6_0b24845f coordinate_violations 1` (the `z=64` node). Since
a locally repaired CSV can NEVER be submitted (notebook-only rule), the repair was moved into the
kernel -- per-axis restoration of the original detector coordinate, never a clamp.
`aryaarun07/biohub-p0cr-v122-revtime-volguard` v1 COMPLETE, audit **PASS 10/10**, output sha256
`435bf5d19d85563c84140e162b3a84f476fdf2fc2cf6ab4b9b759986b9fb9fea` -- **byte-identical to the
offline `repair_linefit_volume.py` result.** That independently confirms pipeline determinism
across kernel runs AND guard/reconstruction equivalence. Slot 3 remains HELD.

**Submission factory:** `scripts/kaggle_factory.py`, spec-driven, stages
build -> verify -> push -> status -> log -> fetch -> audit -> submitcmd. Every stage hard-fails on
a documented trap: edits assert exact match counts (a silently-skipped patch is an ERROR, not a
warning), `PYTHONUTF8=1` forced, slug read back from the push response, typed status enum only,
outputs fetched by URL, audit exit code propagated, and a failing artifact gets NO submit command.
It never submits -- `submitcmd` prints the command. Self-test passed end-to-end on the live P0-A
kernel. Injected code lives in real `.py` files under `scripts/kaggle_edits/` so a graph edit can
be unit-tested before it touches a kernel.

**P0 scoring still PENDING at ~17 h** -- both kernels executed cleanly, so this is trap 13 (slow
public scoring), not an execution failure. Nothing to diagnose. When they land: the LB reports 3
dp so resolution is 0.001, and the arms differ by nodes +64 / edges +103 / divisions -9 out of
120,797 nodes -- a ~0.05% perturbation. **A delta below +/-0.001 must be read as UNRESOLVED, not
as null.**

**NEXT COMPOSITION (the important one):** if fold-0 strict shows P0-A reach >= clean903's 20/26
with node recall holding, then the H0c division cascade -- `+0.0625 / +0.0597` on E0c -- should be
replayed on **P0-A's** substrate rather than E0c's. That composition has never been evaluated and
it is the one path where the 0.913 plateau and the division track MULTIPLY instead of competing:
P0-A supplies the substrate, H0c supplies the layer that E0c's fork noise (11,441 forks, 0 on a
true divider) squanders. Early signal: on the only placeholder crop with divisions P0-A reaches
**3/3**, and its own division layer converts them to TP 0 / FP 6 / FN 3.

### 2026-07-31 (Agent 5 — the pooled objective is CLOSED-FORM, detector diversity is empty, and 63.5% of missed nodes were found then discarded)

Three new scripts: `scripts/agent5_ledger.py` (`d3a946eda79bcdeb`), `agent5_utility.py`
(`46439e21c676547f`), `agent5_detector_preflight.py` (`1b0921ff2337d13c`).

**THE ENABLING RESULT — the pooled objective has a closed form.** Because `w_i = tp+fp+fn` is
exactly the denominator the Jaccard divides by, `w_i * adjJ_i == tp_i * (1 - 0.1*r_i)`, so

    pooled = SUM_i tp_i*(1 - 0.1*r_i) / SUM_i (tp_i+fp_i+fn_i)  +  0.1*DTP/(DTP+DFP+DFN)

with `r_i` invariant under edges-only edits and `tp_i + fn_i = gt_edges_i` constant. **Proved
against the authoritative `summarise()` to 5.684e-14**, with `tp+fn == gt_edges` True and
independently recomputed edge TP matching the scorer on every smoke crop. This converts
"cumulative pooled gain over 14.37M candidates" from 14.4M scorer calls into VECTOR ARITHMETIC,
and yields the four utility weights directly with no fitting. Corollary now algebraic rather than
empirical: an edge with NEITHER endpoint annotated changes the score by exactly zero.

**SUBTRACK A — do NOT build a diverse detector yet. Cheap detector diversity is EMPTY.**

| variant vs `tta-4view__det-0.969` | genuinely new nodes | node Jaccard |
|---|---:|---:|
| `det-0.96875` | 37 / 1 | 0.9989 / 0.9999 |
| `det-0.99` | **0 / 0** | 0.8239 / 0.9539 |
| `tta-d4` | 166 / 92 (0.50% / 1.27%) | 0.9694 / 0.9701 |

Threshold diversity is **strictly nested** -- zero new nodes, only removals -- so it cannot
decorrelate anything. And the gate: reference detector recall is ALREADY **100.0% / 99.54%**;
unioning all five other variants gains **+0 and +1** GT nodes.

**THE INVERSION, and it is the most valuable unmeasured target in the programme.** Raw detector
recall **97.64%** vs the recall the scorer actually sees on the E0c graph **93.55%**. The pipeline
DISCARDS 4.09 pp / 73 GT nodes it had already found. Of the GT nodes the scorer misses, only
**36.5% were never detected -- 63.5% were detected and lost downstream** to one-to-one bipartite
competition, wrapper filters and linefit displacement. (Upper bound: the recall test is
many-to-one where the scorer is one-to-one. Direction unambiguous.) **Zero GPU, upstream of both
the division track and the portfolio-diversity problem.**

**Detection break-even, exact:** one recovered true edge is worth `+w/DEN`, one visible false edge
costs `-adj/DEN`, so a new detector must clear **precision >= adj/(w+adj) = 40.6%** among visible
new edges -- versus **10.15%** for a division action at +0.005. Detection actions are ~4x LESS
forgiving. The node-count penalty is the SMALL term (order 1e-3 of one edge); DISPLACEMENT is the
real cost, consistent with branch A's -0.1596/-0.1496.

**SUBTRACK B — the mechanism utility ordering exploits is now MEASURED, not assumed.**
Smoke (3 crops, 29,800 candidates, 7 positives), exact counter deltas of the isolated action on
the suppress-all graph:

| label | n | mean d_tp | mean d_fp | destroys a true edge |
|---|---:|---:|---:|---:|
| positive | 7 | **+1.143** | **-0.714** | **0.0%** |
| reliable_negative (visible) | 2,236 | -0.067 | +0.948 | **15.8%** |
| unlabeled | 27,557 | -0.069 | +0.055 | - |

True divisions are DOUBLY good -- they add division TP AND edge TP, and never steal a true edge.
The cost of a wrong action concentrates in the 15.8% of visible negatives that destroy a true
edge. That heterogeneity is precisely what utility ordering can exploit and probability ordering
cannot.

**Structural flaw found and fixed:** at the suppress-all anchor `DTP = 0`, so the LINEARISED
division-FP cost is identically zero and a derivative-based utility would admit unlimited false
forks. Utility is now a FINITE DIFFERENCE at an anticipated operating point, plus a
confidence-adaptive re-ranking mode that re-scores at the current `(k, m)` every 25 admissions.

**Does utility beat probability? NOT YET ANSWERABLE.** On 3 crops one fold has a single training
positive so the model degenerates to the prior and every curve is noise (peaks at n=1, +/-0.0008).
The agent explicitly declined to name a winner from that. Full corpus required. Ledger launched
(~5.7 h CPU, ~57 min wall at 6 workers).

**Two bugs caught by self-checks the agent added:** (1) numpy `bool + bool` is logical OR, not
addition -- the first vectorisation silently capped two-true-edge candidates at 1 (positives'
`e_add_tp` read 1.000 instead of 1.143); now guarded by a per-crop parity assertion re-deriving
2,000 candidates with the reference loop, 6,000 checked, 0 mismatches. (2) The naive
`iter_rows(named=True)` loop would have taken ~5 CPU-hours on 14.37M rows.

**Caveats:** all deltas are NO-REFILTER (matching `phaseb_pooled_breakeven.py`), so they are a
LOWER BOUND -- H0d showed the live filter is strictly more favourable, and any promoted operating
point must be confirmed through `phaseb_h0d_livefilter.py`. `pi_vis` differs **14x** across
families on the smoke (0.1047 vs 0.0073): annotation coverage does NOT transfer across the family
boundary, and the agent deliberately does not learn visibility from features (that would be a
metric exploit), so it enters only as a training-fold constant -- the weakest link in the
deployable utility.

### 2026-07-31 (CORRECTION — full-corpus FN attribution: 43.3% recoverable, not 63.5%)

Full-corpus utility ledger complete (199 crops, wall 852 s at 4 workers -- 4x faster than the
57-min estimate).

**CORRECTION TO THE NUMBER I REPORTED LAST TURN.** Agent 5's 3-crop smoke gave
"63.5% of missed GT nodes were detected then discarded", and I promoted that to the
top-ranked target. The full corpus says otherwise:

| quantity | value | share of all FN |
|---|---:|---:|
| GT edges | 128,883 | - |
| edge TP | 101,178 | - |
| **FN_detection** (never detected) | **15,703** (12.184% of GT edges) | **56.68%** |
| **FN_association** (detected, lost downstream) | **12,002** (9.312% of GT edges) | **43.32%** |

So the recoverable-without-a-new-detector share is **43.3%, not 63.5%** -- the smoke overstated it
by ~20 points on 3 unrepresentative crops. The zero-GPU association-recovery target is real but
materially smaller than I said.

The flip side is equally decision-relevant: **56.68% of missed edges are genuine detection
failures**, so that IS the ceiling on what any better detector could add. Agent 5's own preflight
shows that ceiling is hard to reach -- cheap detector diversity is strictly nested (0 new nodes at
det-0.99, +0/+1 GT nodes from unioning all five variants) and a new detector must clear **40.6%**
precision among visible new edges. So the larger half of the FN mass sits behind the more
expensive and more forgiving-free door.

**All identity checks passed at corpus scale:**
- `max|w*adjJ - tp*(1-0.1r)| = 2.274e-13`
- `tp + fn == gt_edges` True
- independently recomputed edge TP == scorer TP True
- vector parity: **398,000 candidates re-derived by the reference loop, 0 mismatches**

The closed-form pooled objective therefore holds at full scale, which is what makes the utility
work tractable at all.

### 2026-07-31 (Agent 4 — H1-M's "+0.0023 pooled" WAS AN IN-FAMILY CEILING PROBE; appearance-stacking closed with a mechanism)

**CORRECTION, and it is the most consequential of the cycle.** I reported H1-M at
`+0.00396 / +0.00201` per-family and derived "~+0.0023 pooled", then called it the first
bilaterally positive mechanism of the campaign. Agent 4 traced that figure: it comes from H1-M's
**IN-FAMILY CV CEILING PROBE**, not its cross-family arm. On honest CROSS-FAMILY transfer H1-M
pools to **+0.00007** (h0c edge convention) or **-0.00131** (conservative).

Exact pooled constants recovered from `coupled_cache/scores/A__*.json`: edge-mass shares
**0.148633 / 0.851367** (151,615 total), `G_pooled = 151`, `divJ_base = 0.00484262`. Pooled
break-even needs `divJ > 0.00846` (h0c) or `> 0.02227` (conservative); **+0.005 needs
`divJ > 0.0585 / 0.0723`.**

**SEPARABILITY (199 crops, 95,511 metric-visible mother events, 92 realisable, LOFO cross-fitted):**

| scorer | AUC 44b6 | AUC 6bba | LOFO k/m | divJ | dC h0c | dC cons |
|---|---:|---:|---:|---:|---:|---:|
| `ssl_split` (label-free) | 0.773 | 0.696 | 5/411 | 0.00890 | +0.00004 | -0.00134 |
| `ssl_mag` (ZERO labels, zero direction) | 0.737 | **0.841** | 5/436 | 0.00852 | +0.00001 | -0.00138 |
| `h1i_massratio` (hand-built) | 0.817 | 0.815 | 11/1088 | 0.00888 | +0.00004 | -0.00134 |
| `h1m_cross` (77-feat, labels) | 0.842 | **0.630** | 5/392 | 0.00921 | **+0.00007** | -0.00131 |
| **`ssl_split & geom_resid`** | - | - | **8/203** | **0.02260** | **+0.00141** | +0.00003 |

Note `ssl_mag` -- pure unsupervised novelty, no labels and no assumed direction -- is bilaterally
strong (0.737/0.841) exactly where the label-fitted H1-M COLLAPSES on 6bba (0.630).

**APPEARANCE x APPEARANCE STACKING IS NOW CLOSED WITH A MECHANISM, not a null.** FP-set Jaccard
vs independence: `ssl|h1m` 0.148/0.034 (74x/17x chance), `ssl|h1i` 0.533/0.408 (267x/203x),
`ssl|geom` **0.0000/0.0000 (0x)**. But complementarity only converts if LIFT (TP co-admission over
FP co-admission, both relative to independence) exceeds 1. Measured lift: `ssl x h1m` **0.00**,
`ssl x h1i` 0.04-0.09, `h1m x h1i` **0.00**. **All appearance-AND rules fail because appearance
scorers concentrate their false positives on the SAME mothers (7-232x independence) while almost
never agreeing on true ones.** That mechanistically explains every failed feature-stack in this
campaign.

Only **appearance x geometry** has anti-correlated FPs, and the working mechanism is NOT
co-ranking -- it is a **broad veto**: geometry keeps the best ~44% of mothers (7,992 / 33,641),
which contain almost no true dividers in the discarded half, letting the appearance threshold go
deeper at the same FP budget. `ssl_split` alone k=5/m=411 -> with veto **k=8/m=203**.

**FRAGILITY (crop bootstrap B=20,000):** divJ 95% CI [0.00843, 0.04050]; dC_h0c
[-0.00000, +0.00320]. P(above break-even) 0.974 (h0c) but **0.498 (conservative)**.
**P(dC > +0.005) = 0.0004 / 0.0000.** The best arm reaches 2.3% of the H0c pooled oracle
(+0.0601) against a gate of +0.005 or 30% of upside (+0.0180).

**VARIANT B (synthetic split) IS DEAD, killed by its own cheap pre-test as instructed.** 40 crops,
96 real dividers, 845 synthetic each; synthesis used the measured 10.3 um separation and exact
mass conservation, and correctly needed no PSF model (translating an already-convolved field is
exact for a shift-invariant PSF). T1 kill test real_normal vs synth_normal: multivariate CV AUC
**0.9888** -- trivially separable, FAIL. T3 decisive: train pure synthetic -> evaluate pure real
gives AUC **0.664**, versus a real-trained reference of 0.867 and the FREE hand-built feature at
0.771.

**METHODOLOGICAL FINDING WORTH PROPAGATING BEYOND THIS LANE:** T1's mean |SMD| is 0.302 with only
1 of 34 features over 1 SD -- i.e. it PASSES the SMD-based domain-shift audit that H1-M itself
used (audit E) while being **98.9% multivariately separable**. **SMD audits drastically understate
multivariate domain separability.** Any prior "domain shift is tiny" conclusion in this campaign
that rested on per-feature SMD is now suspect.

Salvage: on the 13 features informative in real data, synthetic and real signature signs agree
**12/13** (all 4 disagreements have |AUC-0.5| < 0.034). Synthesis is usable as a SIGN PRIOR, never
as a training corpus.

All local CPU, no GPU, no Kaggle, no submission. Peak ~440 MB.

**NEXT (agent's, and I agree):** the binding constraint is not AUC (gates sit at 0.63-0.86) but
TAIL PRECISION -- ~2-4% at the honest operating point against ~6% needed at full recall. The only
measured multiplicative axis left is geometry-as-broad-veto (~2x FP reduction), and it is already
spent. The cheap unmeasured question is whether a SECOND independent veto exists -- forward
association support or daughter persistence, both label-free and plausibly FP-disjoint from
appearance. A second 0.44x veto is worth more than any further appearance modelling.

### 2026-07-31 (CORRECTION — node budget on the full corpus is +0.00157, not the smoke's +0.00822)

Full 199-crop sweep, arm A (E0c), weakest-component-first, divisions protected:

| keep | pooled | delta | 44b6 | 6bba | ratio | recall |
|---|---:|---:|---:|---:|---:|---:|
| 1.000 | 0.66539 | +0.00000 | 0.75955 | 0.64895 | +0.0832 | 0.8999 |
| **0.975** | **0.66696** | **+0.00157** | 0.75740 | 0.65111 | +0.0559 | 0.8907 |
| 0.950 | 0.66462 | -0.00077 | 0.75155 | 0.64933 | +0.0288 | 0.8769 |
| 0.900 | 0.65750 | -0.00789 | 0.74200 | 0.64251 | -0.0254 | 0.8499 |
| 0.800 | 0.63002 | -0.03537 | 0.69995 | 0.61745 | -0.1338 | 0.7851 |

Pooled optimum still 0.975 and the KILL RULE is still NOT triggered (per-family optima: 44b6 1.0,
6bba 0.975), so selective pruning genuinely pays -- **but it pays +0.00157, not +0.00822.** The
12-crop smoke overstated it **5.2x**.

**THIRD SMOKE-TO-CORPUS SHRINKAGE THIS CYCLE**, and the pattern is now systematic enough to be a
rule rather than three coincidences:

| quantity | smoke / probe | corpus / honest | ratio |
|---|---:|---:|---:|
| node-budget delta | +0.00822 (12 crops) | **+0.00157** | 5.2x |
| FN association share | 63.5% (3 crops) | **43.3%** | 1.5x |
| H1-M pooled | ~+0.0023 (in-family CV) | **+0.00007** | ~30x |

In each case I reported the optimistic figure before the corpus number existed. **Standing rule
from here: no smoke or in-family probe gets quoted as a headline. Corpus numbers only, and the
basis (in-family CV vs cross-family LOFO vs oracle) named explicitly every time.**

**WHERE THE DIVISION PROGRAMME ACTUALLY STANDS, honestly:**

| mechanism | honest pooled delta | basis |
|---|---:|---|
| H0c cascade | +0.06012 | GT ORACLE, E0c substrate |
| hybrid substrate | 6bba reach 68->101 | GT ORACLE, ~50 aux nodes |
| node budget (keep 0.975) | **+0.00157** | corpus, deployable |
| ssl_split & geom_resid | +0.00141 | corpus LOFO, P(>+0.005)=0.0004 |
| temporal snap (GT-free) | +0.00369 | in-family basis -- NEEDS RE-DERIVING cross-family |
| H1-M cross-family | +0.00007 / -0.00131 | corpus LOFO |

Only two things are both deployable and corpus-verified: node budget at +0.00157 and the
ssl x geometry veto at +0.00141. Neither approaches +0.005. Every large number in this programme
is still an oracle.

**The temporal-snap figure (+0.00369) was computed on the same in-family basis that inflated
H1-M ~30x and must be re-derived cross-family before it is quoted again.**

### 2026-07-31 (SCORES LANDED — P0-A 0.913 exact, P0-B 0.914, new best public)

| ref | candidate | out sha256 | public |
|---|---|---|---:|
| 55136908 | P0-B clean base + source-locked reverse-time w=0.20 | `4c285cae0c220a11` | **0.914** |
| 55136759 | P0-A exact clean 0.913 reproduction | `8c1605b5944d25e4` | **0.913** |
| 54854143 | v122 (previous best) | - | 0.908 |

**MILESTONE MET: the 0.913-0.914 platform is confirmed on our own account**, +0.006 over our
previous best. P0-A reproduces the public notebook's advertised 0.913 EXACTLY, which validates the
whole reproduction chain end to end -- source audit, byte-identical push, kernel execution,
structural audit, and the notebook-only submission path.

**P0-B minus P0-A = +0.001, which is EXACTLY ONE UNIT OF LEADERBOARD RESOLUTION.** Agent 1
pre-registered the read before the scores existed: the LB reports 3 dp, and the two arms differ by
nodes +64 / edges +103 / divisions -9 out of 120,797 nodes -- a ~0.05% perturbation. A delta of
one LSB is **positive but at the resolution floor**: a single measurement cannot distinguish a
real `+0.001` from rounding across the 3-dp boundary.

Per the standing rule ("if P0-B improves, reverse-time joins the base; if not, the global blend
closes and conditional disagreement survives"), this outcome is genuinely ambiguous and must not
be over-read in either direction. What IS established:
- reverse-time at w=0.20 is **not harmful** on a clean base (the structural evidence had it moving
  divisions -9 on this base and +10 on v122, so harm was a live possibility);
- it is **not established as beneficial** either -- +0.001 is one quantum.
The honest position: adopt P0-B as the deployment base because it is >= P0-A on the only
measurement we have and carries no measured downside, while recording that the mechanism's value
is unresolved at this resolution.

**DEPLOYMENT STATE UPDATED.** New platform: **P0-B at 0.914**. Distances to the milestones --
0.920 exit-the-plateau needs +0.006; 0.925 next-cycle target needs +0.011; 0.935 podium needs
+0.021; 0.942 winning needs +0.028.

Sobering context from this cycle's corrections: the only two DEPLOYABLE, CORPUS-VERIFIED
mechanisms we hold are node budget at **+0.00157** and the ssl x geometry veto at **+0.00141**.
Together, if independent and if they transfer to this substrate (neither is established), they are
~+0.003 -- half of what 0.920 alone requires. Everything larger in the programme is still a GT
oracle. Slot 3 remains held.

### 2026-07-31 (LOEO FOLD-0 RECOVERED WITHOUT GPU — SUBSTRATE DECISION IS GREEN, 22/26)

**The recorded cause of death was wrong on both cause and extent.** HANDOFF and NEXT_DECISION
stated `biohub-loeo-f0-strict` "died after 16 of 71 crops, almost certainly on the
`/kaggle/working` size limit". The kernel log says otherwise:

- `Found 71 test videos` / `Found 71 prediction graphs` — **all 71 crops predicted successfully**;
- **no** disk error, **no** OOM anywhere in the log;
- it failed at `In [8]`, the injected LOEO export/audit cell, on
  `RuntimeError: 44b6_a2bb48bb: out-degree > 2` — my own assertion at `loeo_export.py:68`.

Cause: **2 nodes out of 1,900,633 (0.00011%)** carry out-degree 3 — one in `44b6_a2bb48bb`,
one in `44b6_abf82518`. A 2.7-hour GPU run (t=9727s) was discarded by a hard assertion over two
nodes, after all the expensive work had completed. The gzip/cleanup lines run *after* the
assertion, so no `.csv.gz` was written — but `/kaggle/working` survives on a FAILED kernel.

**Recovery: zero GPU.** `submission.csv` (198,211,344 bytes, sha256 `6880f2fa04969f4007908738`),
`loeo_manifest.json` and `run_stats.csv` pulled by URL from the failed session.
No shards were built and none were needed.

Manifest audit **PASS**: fold 0, arm strict, `secondary_enabled false`, `deepcenter_enabled false`,
weights `split_0/edge_predictor_best.pth`, det_threshold 0.96875, 71 crops unique and all 44b6,
and the crop list is **set-equal to the spec's declared stems**. No leakage path.

**Exact result (71 crops, corpus for fold 0 / 44b6 only):**

| quantity | value |
|---|---:|
| edge_jaccard (raw) | 0.88224 |
| adj_edge_jaccard | 0.89859 |
| division_jaccard | 0.01587 (TP 2 / FP 100 / FN 24) |
| node_recall | **0.98457** |
| composite | 0.90018 |
| **reachable GT divisions (44b6)** | **22 / 26** |

**SUBSTRATE DECISION: GREEN.** 22/26 beats E0c 20/26, clean903 20/26 and v122 15/26, and node
recall holds at 0.9846. The 0.914 deployment platform and the division track **multiply rather
than compete** — this substrate has the most reachable divisions we have ever measured.

The division layer on this substrate is meanwhile near-worthless: divJ 0.0159 on TP2/FP100/FN24.
That is the H0c operation's ideal starting condition (suppress-all costs almost nothing here
because there is almost nothing true to lose), but the deployable selector remains unbuilt and
every large division number in this programme is still a GT oracle.

**Node-ratio finding.** adj (0.89859) EXCEEDS raw (0.88224), so the multiplier is 1.01853 > 1 and
the implied mean node ratio is **−0.1853** — this arm UNDER-predicts nodes, more so than v122
(−0.1598); E0c arm A OVER-predicts (+0.0832). Under-prediction is rewarded (the multiplier is not
clipped above 1), so node budgeting has no obvious count headroom here.
**CAVEAT, stated explicitly: this is the `strict` arm with the secondary model and DeepCenter
DISABLED, so it emits fewer nodes than real P0-B. This ratio is NOT P0-B's ratio and must not be
transferred to it.** Arm-D measurement pending.

**Do NOT compare 0.89859 against E0c's published 44b6 figure of 0.7595.** The two may not share a
weighting convention and I have not reconciled them. No claim rests on that comparison.

Artifacts: `reports/inventory/loeo_f0_strict.json` (sha256 `b5f98a6499fab3c14da0e396`),
`reports/inventory/loeo_f0_strict_manifest.json`.

**Submitted P0-CR** (ref 55147215, PENDING): v122 base + reverse-time w=0.20 + line-fit volume
guard, out sha256 `435bf5d19d85563c`, structural audit **PASS 10/10** (A1–A10, outside=0 — the
volume guard fixed the z=64 defect that failed the original P0-C). Causal question: is
reverse-time's effect base-dependent? Expected ~0.908–0.912; this is a causal probe, **not** a
score-climb candidate.

#### Addendum — division headroom on the 22/26 substrate (GT ORACLE + selector sensitivity)

Arithmetic on the measured fold-0 numbers only (composite = adjEdgeJ + 0.1·divJ, adjEdgeJ held
fixed at 0.8985894; H0c's measured edge effect on E0c was +0.0010 / −0.0006, i.e. second-order).

| arm | divJ | composite | Δ vs measured |
|---|---:|---:|---:|
| measured (TP2/FP100/FN24) | 0.01587 | 0.9001767 | — |
| suppress-all only | 0.00000 | 0.8985894 | **−0.00159** |
| **H0c oracle** (reconstruct all 22 reachable, FP 0) | 0.84615 | 0.9832048 | **+0.08303** |

Suppress-all alone is now slightly *negative* (it deletes 2 real TPs), which differs from E0c where
it was edge-neutral. The gain is entirely in reconstruction, as before.

**Selector sensitivity — the precision bar is low.** k = true forks recovered, m = false admitted:

| k \ m | 0 | 10 | 30 | 60 |
|---|---:|---:|---:|---:|
| 22 | +0.0830 | +0.0595 | +0.0377 | +0.0240 |
| 15 | +0.0561 | +0.0401 | +0.0252 | +0.0159 |
| 10 | +0.0369 | +0.0262 | +0.0163 | +0.0100 |

Recovering only **10 of 26** divisions while admitting **60** false forks — ~14% precision — still
returns **+0.010** on this fold. That corroborates the documented 10.15% break-even from an
independent direction.

**THE LIMIT OF THIS RESULT, STATED PLAINLY.** Every number above is 44b6 / fold 0 only, and 44b6
carries just **14.94% of edge mass**. The pooled objective is dominated by 6bba, which is
**unmeasured on this substrate** — fold 1 is LEAKY with the pack's `split_0` weights (trained on
6bba) and needs our own `split_1`. `scripts/kaggle_specs/loeo_f1_strict.json` exists.
**Do not convert any figure above into a pooled or public-score claim.** The honest status is:
44b6 headroom is large and the precision bar is low; the 85%-mass family is unknown.

### 2026-07-31 (Lane 3 — NODE BUDGET DOES NOT TRANSFER TO P0-B; the ssl×geometry veto is not a deployable stage)

**Execution.** Fetched the live P0-B artifact (`aryaarun07/biohub-p0b-clean-913-reverse-time`,
sha256 `4c285cae0c220a11…`, 120,861 nodes / 116,604 edges / 305 divisions, structural audit
**PASS 10/10**, max out-degree 2 — the reported out-degree>2 defect is **not** present in P0-B's
test output). Ported `scripts/win_bet/phaseb_node_budget.py` verbatim to a deployable in-notebook
stage, `scripts/kaggle_edits/node_budget_stage.py`. At `keep_frac 1.0` the replay reproduces the
live artifact **byte-identically**, so the harness is exact.

**Like-for-like, same four movies, same frozen `keep_frac 0.975`, exact patched scorer:**

| substrate | pooled node ratio | composite | + node budget | Δ |
|---|---:|---:|---:|---:|
| E0c arm A | **+0.1269** | 0.749642 | 0.755981 | **+0.006339** |
| **P0-B (live 0.914)** | **−0.1028** | 0.889225 | 0.889214 | **−0.0000103** |

**P0-B's node ratio is −0.1028** — it UNDER-predicts, answering the open gate. (Caveat: measured
against the placeholder crops' `estimated_number_of_nodes`, not the hidden movies. It is also not
the strict arm's −0.1853, which the LOEO fold-0 entry correctly refused to transfer.)

**Mechanism.** The `+0.00157` corpus gain is **~116% count multiplier**: first-order,
multiplier-only Δ = `+0.0018258` against a measured `+0.0015699`, implying an edge-quality cost of
`−0.0002559`. It is a count effect, not a precision effect. The multiplier term is
`0.1·f·(1+r₀)·J`, which SHRINKS but does not change sign under under-prediction — so the sign of
r₀ is not what kills it. What kills it is the substrate: **P0-B already runs
`filter_short_track_components`**, so its weakest surviving components average 6.8 nodes and are
correct tracks. On the highest-edge-weight crop (`6bba_05db0fb1`) the stage destroys **5 counted
edge TPs and removes 0 counted edge FPs** (adjJ 0.83534 → 0.83364). On E0c the same operation
*gains* TPs (748→751, 992→994) because the junk it deletes was stealing bipartite matches.

**Proxy bias, stated explicitly.** These four crops score 0.889225 locally for the artifact whose
public score is **0.914**, so this is not the leaderboard. Their annotation is sparse (~50 counted
edges per 44b6 crop), which UNDER-samples the edge cost while the multiplier is fully realised —
the proxy is biased **in the node budget's favour**, and it still lands at −0.00001.

**P1 graph delta vs P0-B:** nodes −3,027 (0 added), edges −2,580 (0 added), divisions 305→305
(fork protection verified), **0 parent reassignments**, structural audit **PASS 10/10**, simulated
artifact sha256 `00e7fed996b6b783…`. Built as `scripts/kaggle_specs/p1_p0b_nodebudget.json`
(built sha256 `898a4113e0e5625a…`, config hash `e02bae465eaa`, blast radius = cell 6 only).
**NOT pushed. NO-GO.**

**P2 (ssl × geometry veto) is not a bolt-on veto and was not built.** Reading
`scripts/h4_ssl_fuse.py` + `scripts/win_bet/h4_ssl_gate_replay.py`: the `+0.00141` is the
**admission gate of the H0c division-reconstruction cascade** — which mothers get their rank-0
shortlist pair reconstructed — not a filter over an existing graph. Deploying it needs 15/15
candidate generation, flow-midpoint top-3 ranking, the 17-descriptor 5-frame appearance cache, a
ridge normal-continuation fit, suppress-all + add-replace on the test movies, and an admission
*rate* whose LOFO value is derived per-family from train labels. Moreover `+0.00141` is an
arithmetic projection through **constant E0c edge costs**; `h4_ssl_gate_replay.py`'s own docstring
warns against exactly that, and it **has never been run**.

The one reading that *is* a bolt-on — vetoing P0-B's existing forks — has a **GT-oracle** corridor
of only `+0.00042 … +0.00216` pooled on E0c (remove all 675 division FPs, keep the 4 TPs:
divJ `4/826 → 4/151`, minus the measured suppress-all edge cost `−0.00174`). On P0-B's own four
crops it is **exactly 0**: division TP0 / FP8 / FN3, so divJ is 0 before and after a perfect veto.

**Decision.** P1 NO-GO, P2 not buildable, **P3 does not exist**. Neither of the two "corpus-verified
deployable mechanisms" is deployable on the 0.914 base. Integrity note for whoever revisits P1:
its gain is ~116% count multiplier, while P0-B's own in-kernel report cell declares
`"metric_hack_used": false`.

Artifact: `reports/inventory/p1_node_budget_p0b_substrate.json`.

### 2026-07-31 (ASSOCIATION FN ATTRIBUTION — complete, exact, and NEGATIVE for deployment)

**Basis: E0c substrate, 199 crops, exact pooled composite. NOT the P0-B base that scores 0.914.**
Baseline pooled 0.6654043 (edge 0.6649200 + div 0.0004843); GT edges 128,883; TP 101,178 /
FP 22,731 / **FN 27,705**. Parity against the agent5 ledger: **zero** mismatches on
`edge_tp`/`edge_fn`/`gt_nodes_matched`/`num_pred_nodes`/`r_i`; closed form verified against the
authoritative patched scorer on 10 crops (dNUM error 0.000e+00).

| first-loss stage | count | share | pooled ceiling Δ | replay-only |
|---|---:|---:|---:|---|
| detector never found an endpoint | 12,135 | 43.80% | +0.00000 | no — needs a detector |
| enumeration, relaxed-pass starved (6–10 µm) | 3,871 | 13.97% | **+0.04052** | yes |
| short-component filter, GT match taken over | 2,922 | 10.55% | **+0.03210** | yes (edges-only) |
| short-component filter, endpoint lost after a *correct* link | 2,389 | 8.62% | +0.00000 | no |
| bipartite: both endpoints consumed | 1,400 | 5.05% | +0.01540 | yes |
| bipartite: target took another parent | 1,293 | 4.67% | +0.01428 | yes |
| short-component filter, no prior link | 1,179 | 4.26% | +0.00000 | no |
| bipartite: source took another child (orphan) | 1,108 | 4.00% | +0.00899 | yes |
| enumeration, beyond the 10 µm cap | 1,091 | 3.94% | +0.01114 | yes |
| endpoint exists only as a gap-inserted node | 317 | 1.14% | +0.00327 | yes |
| **total** | **27,705** | | | |

All-recoverable 12,002 (**43.32%**, reproducing the known figure exactly) → **+0.13288 GT ORACLE**.

**UNIT ECONOMICS (exact): one net-correct repair = 1.095e-05 pooled ⇒ +0.002 needs 183 net-correct
repairs, +0.003 needs 274.** Use this instead of arguing about shares.

**Three mechanisms, three verdicts.**
- **Enumeration (4,962 edges, +0.052 ceiling) — CLOSED.** True pairs sit at 6.08–9.88 µm or beyond
  10 µm; the only mechanism is widening, already falsified corpus-wide by branch A.
- **Short-component filter (5,311 edges = 19.2% of ALL FN) — NEW, mechanism falsified.** After the
  relink the only edge-removing stage is `filter_short_track_components` (corpus
  `dropped_multi_parent_edges` = 0). It deletes 5,311 GT edges the relink had **already linked
  correctly**. Count-multiplier cost of full retention is only −0.00605, so on paper retention nets
  +0.0288 — **but branch A's exact control falsifies it**: with the filter off, edge TP *falls* and
  adjJ drops 0.8821→0.8766 / 0.8068→0.7924, because re-added nodes steal bipartite matches. And the
  edge-level signal is dead: deleted true edges are statistically **identical** to the average
  selected relink edge (prob median 0.785 vs 0.786; raw_um median 2.30 vs 2.30).
- **Bipartite competition (3,801 edges, +0.039 ceiling) — the only unfalsified lane, fails
  cross-family.** Contested targets: relink is correct **87.51%** of the time. Inside `target_taken`
  the transformer prefers the true parent in only **9.36%** of cases (raw separation 7.89%), far
  below the ~50% break-even. LOFO: 44b6→6bba **+0.00099**, 6bba→44b6 **+0.00002** — a 50×
  disagreement, and the 44b6 fit's in-sample net was +4 targets, i.e. noise. Orphan swap: blind
  swapping is −0.01667; best in-sample rule +0.00059; LOFO −0.00019 / +0.00006. **Falsified.**

**DECISION: integrate nothing; do not spend a slot on this.** Best cross-family value for any
repair is +0.00006…+0.00099 against a +0.002 target. The two LOFO directions disagreeing by 50× is
the exact signature that closed seven prior methods.

**Next (cheap, gated):** component-level selective retention — the 5,311-edge bucket is the largest
pipeline-caused loss whose *component*-level signal (length, node count, mean cost, degree profile,
frame density) is still unmeasured, even though the edge-level one is dead. Gate hard: require a
GT-free component score with leave-family-out sign stability **before** any replay, since blanket
retention is already −0.0055/−0.0144. Second: re-run the attribution against
`artifacts/kaggle/clean903_wrapper_oof_cache` (199 crops, same schema) to learn whether this loss
profile transfers to the 0.913/0.914 substrate at all.

**Could not be delivered:** reciprocal forward/reverse support — no reverse-time cache exists
locally. Self-corrected defect: `det_never_detected` was tested before `final_both`, misfiling 317
edges; corrected from stored per-row flags and the split reproduces 15,703 / 12,002 to the edge.

Artifacts preserved outside the worktree at
`..._RESEARCH/agent_runs/lane4_fn_attribution_2026-07-31/` (25 MB);
ceilings copied to `reports/inventory/fn_attribution_ceilings.json`.

### 2026-07-31 (NODE BUDGET ON ARM D — CORPUS, 199 crops: −0.00088. Channel closed.)

**Basis: CORPUS. 199/199 cached OOF crops (71×44b6 + 128×6bba), exact patched scorer, pooled
objective. Not a smoke, not in-family.** Parity: keep_frac 1.0 reproduces the published v122
anchor exactly — pooled **0.69909203** vs published 0.69909, node ratio **−0.1598** vs published
−0.1598, **0 mismatches across 199 crops** against the cached `coupled_cache/scores/D__*` counters.

| keep_frac | pooled | Δ | 44b6 | 6bba | node ratio | node recall |
|---|---:|---:|---:|---:|---:|---:|
| **1.00** | **0.69909** | **0.00000** | 0.69625 | 0.69966 | −0.1598 | 0.8286 |
| 0.975 | 0.69821 | **−0.00088** | 0.68984 | 0.69976 | −0.1810 | 0.8189 |
| 0.95 | 0.69504 | −0.00405 | 0.68389 | 0.69709 | −0.2020 | 0.8084 |
| 0.90 | 0.68617 | −0.01292 | 0.66852 | 0.68938 | −0.2440 | 0.7826 |

Crop-block bootstrap (10,000 resamples; closed-form vs `summarise()` max abs diff **2.22e-16**):
at 0.975 **P(Δ>0) = 0.1088**, at 0.95 P = 0.0008, at 0.90 P = 0.0. **Pooled optimum is
keep_frac = 1.00 — no pruning.**

The preregistered kill rule (optimum ≥ 0.99 on *both* families) did not literally trigger, because
6bba's optimum is 0.975 — but that "gain" is **+0.00010**, one ten-thousandth, while 44b6 loses
**−0.00641**, sixty-four times larger. On the primary pooled objective the channel is closed.

**THE MECHANISM, now settled across three independent measurements.**

| substrate | node ratio | node-budget Δ @0.975 | basis |
|---|---:|---:|---|
| E0c arm A | **+0.0832** (over) | **+0.00157** | corpus, 199 crops |
| v122 arm D | **−0.1598** (under) | **−0.00088** | corpus, 199 crops |
| P0-B (deployment) | **−0.1028** (under) | **−0.00001** | 4 placeholder movies |

**The sign flips with the sign of the node ratio.** Node budget's arm-A gain is a *count-multiplier*
effect available only to a graph that OVER-predicts nodes. Arm D already under-predicts at −0.1598;
pruning drives it to −0.1810, the multiplier term is saturated, and only the edge-quality cost
remains. This is independently consistent with the arm-A decomposition (~116% of +0.00157 was the
multiplier, ~−16% edge quality) and with the direct P0-B measurement from the other lane.

**DECISION: KILL the "P0-B + node budget" candidate.** Not a mechanism that transfers off arm A.

**DEFECT FOUND AND FIXED.** `scripts/win_bet/phaseb_node_budget.py` accepted `--arm` and ignored
it: `prune_one()` unpacked `arm` and never read it, and the graph path was hardcoded to the arm-A
cache. **Running `--arm D` would have scored arm A while writing `"arm": "D"` into the output.**
Caught only by the per-crop parity cross-check. Fixed by `arm_graph_path()`, which now raises on a
missing arm cache instead of silently falling back. Recorded as trap 14.

Artifacts: `reports/inventory/node_budget_armD.json`.

### 2026-07-31 (ssl × geometry EXACT REPLAY — first ever run; headline was ~26% high)

**Basis: CORPUS, 199 crops, exact patched scorer, canonical pooled objective, E0c substrate.**
Runtime ~39.8 min, 3 workers. Baseline arm reproduces the published anchors exactly
(44b6 **0.759549**, 6bba **0.648965**, pooled **0.665404**) — the check that makes the delta
trustworthy.

| | pooled | 44b6 | 6bba |
|---|---:|---:|---:|
| Δ composite | **+0.001042** | +0.003153 | **+0.000097** |
| 95% CI | [−0.00056, +0.00293] | [−0.00086, +0.00819] | [−0.00143, +0.00201] |
| P(Δ > 0) | 0.890 | 0.933 | **0.520** |
| P(Δ > +0.005) | 0.0001 | 0.207 | 0.0001 |

Divisions TP 4→9, FP 675→202, FN 147→142. Edge cost −0.00102 pooled, uniform across families.
211 admitted → 211 forks retained, 226 parent steals.

**The headline was overstated.** `ssl_fuse_v2.json` carried two arithmetic projections spanning
**43×** — `h0c_measured` **+0.001413** (which became the HANDOFF §4 headline) and `conservative`
**+0.000033**. The exact scorer says **+0.001042**: the optimistic convention was ~26% high, the
conservative one 32× low. This is the fourth reporting correction of the cycle and the same
pattern each time — a projection quoted as a measurement.

**DECISION: does not change deployment.** Min-fold **+0.000097** with **P(6bba gain) = 0.52**, a
coin flip, against a +0.005 bilateral gate; the entire pooled effect is carried by 44b6. It is also
measured on **E0c**, not the P0-B base that scores 0.914, and it is not a bolt-on veto — deploying
it requires the whole H0c cascade at test time.

**TWO DEFECTS FIXED (trap 15).** The script had **never actually run**: `main()` called
`cached_crops()` with no argument against a `cached_crops(split: int)` signature, raising
`TypeError` immediately — despite being recorded as "built, smoked, not run". And `agg()` computed
the multiplier as `abs(N_pred_arm − N_pred_baseline)/N_pred_baseline`, identically **0** for an
edges-only replay, so results were **unpenalised** and incomparable to the published anchors. Now
computed signed against `N_est`. Verified: the job list builds 199 crops (71 + 128).

### Open reconciliation — the 0.8986 vs 0.7595 gap is NOT a weighting convention

Lane 2 checked directly: `scripts/score_oof.py` and `scripts/score_loeo_submission.py` **both** build
rows via `per_sample_metrics(..., estimated_nodes(gt_geff), ...)` and aggregate with the same
`tracking_cellmot.metrics.summarise`. **A convention mismatch between the two entry points is ruled
out.** So the fold-0 adj_edge_jaccard of 0.89859 versus E0c's published 44b6 0.7595 is either a
genuine substrate difference or a config/leakage issue. The manifest audit argues against leakage
(secondary off, DeepCenter off, `split_0` is LOEO-clean on fold 0). **Unresolved — do not build on
the comparison until someone scores E0c and this artifact through one entry point in one pass.**

### CPU queue items NOT run, with reasons (no fabricated numbers)

- **`agent5_utility.py --max-admit 6000`** — not attempted. Joins 24 float32 features onto a
  **14,371,002-row** shortlist, ~3–5 GB peak against 2.7–5.7 GB free while three jobs held the
  worker cap. Runnable on a free box; note `out_dir = ledger.parent`, so redirect the output.
- **`h1n_exact_replay.py`** — not run, **and the "~82 min" estimate is suspect**: the existing
  smoke artifact records **4 crops in 590 s**, which scales naively to ~8 h for 199. Re-derive the
  estimate before budgeting. Its `pooled_delta = +0.00859` is a **4-crop smoke** and must not be
  quoted as a headline.
- **`phaseb_h2a_hybrid_oracle.py`** — not run; interface confirmed working. From its own artifacts
  (3 crops in 133.4 s) the full 199 crops is **~2.5 h at 3 workers**.

### 2026-08-01 (CORRECTION — the node-ratio mechanism was wrong; verdicts unchanged)

The 2026-07-31 arm-D entry above says "**The sign flips with the sign of the node ratio**" and calls
the arm-A gain a count-multiplier effect "available only to a graph that OVER-predicts nodes".
**The empirical results are unchanged and node budget stays CLOSED — but that explanation is
falsified.**

The per-node count cost is `0.1·tp_i/N_est_i`. `N_est_i` is **GT metadata**, so the cost is
**exactly invariant** to whether the substrate over- or under-predicts. The node ratio enters the
decision only through `w_i = 1 − 0.1·r_i`, which shifts the threshold by **1.1%** across the whole
±0.16 range — nowhere near enough to flip a sign. And per-crop node ratios are **mixed-sign on both
substrates**, so "E0c over-predicts, P0-B under-predicts" was never the clean dichotomy I described.

**What actually flipped it** `[4 movies, exact scorer]` is the `d_tp/d_fp` composition of the
components being deleted:

| substrate | d_tp | d_fp | q_net | deleting is | measured Δ |
|---|---:|---:|---:|---|---:|
| E0c arm A | −5 | +8 | −1.67 | correct | **+0.006339** |
| P0-B | +5 | 0 | +1.00 | wrong | **−0.0000103** |

P0-B already runs `filter_short_track_components`, so its weakest surviving components are *correct
tracks* and there is nothing left worth deleting. Lane 3 actually reported this correctly at the
time ("the sign of the node ratio is not the cause… substrate quality is"); I then wrote the
node-ratio story into the arm-D entry and the ledger anyway. **That is a synthesis error of mine,
not a measurement error**, and it is the fifth reporting correction of this cycle.

**Closed form now available and verified:** retain iff
`q = d_tp/(d_tp+d_fp) > (Jbar + 0.1·n·ρ_i/a)/(w_i + Jbar)`, floor **0.3994**, which independently
reproduces the project's separately-derived **40.6%** detection-precision bar. Implementation:
`decision_kernel.py` (research store); if adopted it belongs at `src/biotrack/decision.py`.

**Standing rule extended:** the reporting-discipline section covers *numbers*. It now also covers
*mechanisms* — a causal story attached to a correct number is itself a claim, and must be stated as
a hypothesis until something tests it. "The sign tracks r" was never measured; it was pattern-matched
from three data points that happened to line up.

### 2026-08-01 (DIVISION BIOLOGY — measured on our own 151 GT divisions, not cited)

Measured directly over all 199 crops (151 GT divisions: 26 in 44b6, 125 in 6bba — the same
denominators as the division track). Grade A unless noted. Raw material and scripts:
`..._RESEARCH/agent_runs/research_division_biology_2026-07-31/`; research.sqlite 90 → 102.

**Two facts that reframe the division problem.**
1. **The label is a histone** (`tg(h2afva:h2afva-mCherry)`). Mitosis here is a **condensation**
   event, not an envelope event — which is exactly why compaction features work and
   rounding/envelope features have nothing to key on. Stop looking for envelope signatures.
2. **The acquisition interval is NOT in the released metadata** — the zarr declares the T axis as
   `unit: "second", scale: 1.0`, a placeholder. Two independent estimates (a 4–6 frame condensation
   window against a ~25 min zebrafish NEB→daughters mitosis, and the 1.82 µm/frame median
   displacement) put the true cadence at **~2–3 min/frame** `[grade C]`. **Express every temporal
   window in FRAMES, never in physical time.** Worth asking the host to confirm.

**The free win: every appearance channel is strictly better at t−1 than at the annotated fork
frame.** peak-above-background 0.679 vs 0.604 · anisotropy 0.658 vs 0.583 · half-max volume
**0.657 vs 0.496**. By the fork frame the object is already re-expanding, so the compaction signal
has cancelled. Same features, one frame back: **+0.05 to +0.08 AUC for zero cost.**

**A cross-family appearance channel that transfers.** `[peak(t−1), halfmax_vol(t−2),
anisotropy(t−1), peak(t−2)]` in an L2 logistic reaches **leave-one-embryo-out AUC 0.719** —
held-out 44b6 **0.746**, held-out 6bba **0.728**, both independently above 0.72
`[cross-family-LOFO]`. The geometry-only candidate surface has no such channel.

**Geometry, measured.** `sister_um` median **10.57 µm**, p10 6.36, p90 14.36, **max 20.30**.
A 15 µm cap keeps **93.4%** of true pairs, 18 µm keeps **99.3%**, and the old 8.5 µm cap kept only
**29.1%** — which quantitatively explains why that cap destroyed 75–79% of reachable divisions.
The frozen 15/15 surface is sound; **18 µm is the only justified widening**. Flow-midpoint residual
median **2.33 µm** against a **1.82 µm** single-frame noise floor — the tightest invariant we have.

**Killed, each by measurement rather than argument:**

| proposal | result | verdict |
|---|---|---|
| local division-rate / synchrony prior | 56% of crops have ZERO divisions; per-node rate 0.0013 vs 0.0011; χ²/df 1.40 vs constant-rate Poisson; nearest-division distance 39.1 µm in a 104 µm cube (= random) | **no synchrony, no clustering.** The only variance it can absorb is per-crop annotation density ⇒ a crop-identity proxy. **Forbidden.** |
| mother→daughters mass conservation | 1.607 vs 1.000 for controls — but that is **two apertures versus one**; peak is only 0.7–1.2× background at NND ≈ 8.9 µm | aperture artefact, `[grade D]`. Correct null is non-dividing FORK candidates |
| pre-mitotic motion / IKNM | speed AUC 0.518, flow residual 0.553, track age 0.451; persistence separates in the WRONG direction | drop all motion terms |
| density-normalised sister cap | sister/NN1 median **1.16**, not the scale-free 2×; normalising raises CV 0.300 → 0.619 | raw microns wins — a noisy denominator imports more family variance than it removes |
| metaphase-plate oblateness / pre-separation bimodality | AUC 0.539 / 0.565; the plate spans ~4 z-planes at 1.625 µm | unresolvable |
| division-axis orientation | mean \|cos θ_z\| 0.455 vs 0.5 uniform | isotropic, no prior exists |

**Normalisation caveat carried forward:** peak-above-background is 0.679 raw but **0.634
crop-normalised and 44b6 collapses to 0.542** — it partially encodes an embryo gain term and must
always be crop-normalised. Anisotropy ax1/ax3 *improves* under normalisation (0.658 → 0.665),
the signature of a genuinely dimensionless quantity. Note controls sit at 1.73 from the anisotropic
PSF alone, so only the **excess** over that is biology.

### 2026-08-01 (Lane B — the 0.7595-vs-0.89859 gap is REAL; H0c is +0.0739 live on the P0-strict substrate)

**GATE 1 — RECONCILIATION. Resolved: a genuine substrate difference, not a weighting convention.**
The open caveat ("do not compare 0.89859 with E0c's published 44b6 0.7595 — weighting conventions
are unreconciled") is closed. Both arms scored through **one entry point**
(`biotrack.metric.score_pred_graph`) in **one run** over the **same 71 fold-0 crops**:

| arm (71 crops, 44b6, fold 0) | raw edge_J | adj_edge_J | node_recall | divJ | composite | N_pred |
|---|---:|---:|---:|---:|---:|---:|
| E0c cached post-wrapper graphs | 0.767340 | 0.759549 | 0.948158 | 0.0000 (0/93/26) | **0.759549** | 2,864,419 |
| P0-strict recovered LOEO f0 | 0.882245 | 0.898589 | 0.984573 | 0.01587 (2/100/24) | **0.900177** | 1,900,633 |
| **delta** | +0.114905 | +0.139040 | +0.036415 | +0.01587 | **+0.140628** | −963,786 |

PARITY: the E0c arm reproduces the published anchor to |d| = 4.9e-05 (rounding of `0.7595`), and
per-crop it matches branch A's published `44b6_d29c9ab2` adjJ **0.8821** exactly. The P0 arm
reproduces `reports/inventory/loeo_f0_strict.json` at **|d| = 0.000e+00 on every field**. CSV
sha256 verified `6880f2fa04969f4007908738cbc88045015f95142c3aee7f8da95392164ce6be`. 255.7 s,
3 workers. 66/71 crops have node recall >= E0c's; only 2/71 regress on adjJ.

**Attribution of the +0.140628** `[exact-pooled-OOF, fold 0 / 44b6 only]`:

- **+0.114905 (81.7%) real matching quality.** Edge TP 17,292 -> 18,723, FP 2,709 -> 1,396,
  FN 2,534 -> 1,103. Precision 0.8646 -> 0.9306, recall 0.8722 -> 0.9444. P0-strict emits
  **one million fewer nodes** and finds **more** GT nodes.
- **+0.024136 (17.2%) count multiplier.** Edge-mass-weighted node ratio +0.1048 -> -0.1852, so the
  unclipped multiplier goes 0.98952 -> 1.01852. Stated as a share of the observed gap only — the
  metric agent's correction stands: the per-node count cost is `0.1*tp_i/N_est_i`, invariant to
  over/under-prediction.
- **+0.001587 (1.1%) division term.**

**No leakage signature, but the pack's provenance is undocumented and this must be said.**
P0-strict fold 0 runs the *support pack's* `split_0`, not ours. FOR cleanliness: only 2/71 crops
regress; per-crop adjJ still spans 0.5591–1.0640 and node recall bottoms at 0.9256; divisions are
still TP2/FP100/FN24; and the held-out-family OOF composite (0.9002) sits **below** P0-A's public
0.913 — memorisation would invert that. AGAINST certainty: the pack ships **no training record at
all** — no `train_datasets`, no held-out declaration — and its own `ARTIFACT_MANIFEST.json` names it
**`biohub-tracking-support-pack-400ep-snapshot-v1`** while the Kaggle dataset is called "50ep"
(`source: "public learned baseline artifact, repackaged locally"`). **The only evidence that 44b6
was held out is the directory name `split_0`.** Treat 0.900177 as clean-pending-provenance.

**The two 44b6 numbers are also not the same system.** Pack `split_0` is 8,363,159 B, sha256
`12f6881ee3620a83…`; our LOEO `split_0` is 8,357,783 B, sha256 `d3e89eb361eeadef…` — E0c ran ours.
Fold 0 therefore conflates *model vintage* with *pipeline*. **Fold 1 does not**: it runs our
`split_1` on both sides, so fold 1 vs E0c 6bba isolates the pipeline alone.

---

**GATE 2 — FOLD 1 (6bba) LAUNCHED, RUNNING.** `aryaarun07/biohub-loeo-f1-strict` v2, slug read back
after push (trap 12), `PYTHONUTF8=1` throughout (trap 8).

*Leak-freeness of `split_1`, verified before spending GPU:* `data/dataset_splits.json` split 1 =
train 71x44b6 / test 128x6bba; `notebooks/kaggle_train_oof_f1/kaggle_train_oof_f1.py` sets
`FOLD = 1` ("HOLD OUT 6bba (train on 44b6)"), 45 epochs, and picks its best epoch on a **6-crop
validation subset of the TRAINING embryo** — never the held-out one. The exported filenames are
exactly those in `aryaarun07/biohub-oof-weights`. **LOEO-clean for 6bba.** Kernel log confirms the
override resolved to sha256 `2e4ebf616b3d4fb5…`, secondary DISABLED, DeepCenter DISABLED, 128
crops, both GPU shards on `/kaggle/working/loeo_weights/edge_predictor.pth`.

*Serving check:* `pool_kernel_um` comes from `PredictConfig` (3.0), **not** `config.json` (which
supplies only `window_size`/`downsample`), so both folds serve at 3.0 — no asymmetry.

**v1 died at t = 628 s -> new trap 16.** `weights glob '/kaggle/input/*/edge_predictor_best_split_1.pth'
matched []`. **Not trap 9:** the same run had already mounted 128 train `.zarr` crops and resolved
the pack at `/kaggle/input/datasets/pilkwang/…` — datasets mount **owner-qualified, one level
deeper**. Fixed in `scripts/kaggle_edits/loeo_retarget.py::_loeo_find` (declared pattern, then a
**bounded** `*/`-ladder on the basename; never `recursive=True`, which would descend the 79 GB
competition zarr tree), and the failure path now prints the mount tree. Cost ~10 GPU-minutes.
No shards needed: fold 0 was 71 crops / 2.86 M E0c-nodes in 2.70 h against a 9 h session; fold 1 is
128 crops / **2.25 M** E0c-nodes — 6bba crops are individually smaller.

---

**GATE 3 — H0c AND H0d ON THE P0-STRICT SUBSTRATE.** Frozen cascade, config hash `04eeac97500d`,
K and ranking inherited from H0b and **not retuned**. Two independent implementations agree:
`h0c_replay_p0strict.py` (loader rebound, `replay_one` imported verbatim) gives +0.073419;
`phaseb_h0d_livefilter.py --surfaces p0strict` gives +0.073418. Baseline parity |d| = 0.000e+00 on
every field in both. 1418 s / 1944 s at 3 workers.

| arm (71 crops, 44b6, fold 0) | composite | Δ vs base | adj_edge_J | raw edge_J | divisions |
|---|---:|---:|---:|---:|---|
| base | 0.900177 | +0.000000 | 0.898589 | 0.882245 | TP2/FP100/FN24 |
| `supp_post` suppress-all, no re-filter | 0.899097 | −0.001080 | 0.899097 | 0.882756 | TP0/FP0/FN26 |
| `supp_refilt` + live wrapper filter | 0.899558 | −0.000619 | 0.899558 | 0.883081 | TP0/FP0/FN26 |
| `h0c_post` published H0c form | 0.973595 | **+0.073418** | 0.900518 | 0.884155 | TP19/FP0/FN7 |
| `h0c_refilt` **LIVE PATH** | **0.974054** | **+0.073877** | 0.900977 | 0.884475 | TP19/FP0/FN7 |
| `h0c_refilt_guard` | 0.974054 | +0.073877 | 0.900977 | 0.884475 | TP19/FP0/FN7 |

`[GT-oracle, fold 0 / 44b6 only — 14.94% of corpus edge mass. NOT a pooled claim.]`

Retention **19/22 reachable = 86.4%**; shortlist 5,267,771 from 28,956,390 pre-top-K over
1,833,541 mothers; only **12 parent steals**; `gt_collisions = 0`; 5,300 existing forks suppressed.
This reconciles with the circulated **+0.0830**: that is the analytic k=22/m=0 ceiling, while the
frozen flow-midpoint top-3 shortlist actually retains 19 of 22 -> **+0.0734**. The missing +0.0096
is exactly the 3 reachable divisions the ranker drops — i.e. **the ranker, not the substrate, is
now the first loss.**

**The edge side is POSITIVE here, unlike E0c.** adj_edge_J +0.001929 (`h0c_post`), +0.002388 (live).
On E0c it was +0.0010 (44b6) and **−0.0006** (6bba). Suppress-all alone is edge-**positive**
(+0.000508); add-replace of the 19 true forks adds +0.001421.

**The live-filter cross-term must NOT be inherited from E0c — measured, it shrinks 3.5x:**

| surface | `h0c_refilt − h0c_post` | `supp_refilt − supp_post` | division-exempt short components |
|---|---:|---:|---|
| E0c 44b6 | +0.001610 | +0.001617 | 134 comps / 652 nodes, **all lost** |
| E0c 6bba | +0.004894 | +0.004865 | 122 comps / 586 nodes, **all lost** |
| **P0-strict 44b6** | **+0.000459** | **+0.000461** | **0 comps / 0 nodes** |

The mechanism is structurally absent: on P0-strict **no component survives the wrapper only through
its division exemption**, so the D0′ hazard cannot fire and the GT-free retention guard is a literal
no-op (`h0c_refilt == h0c_refilt_guard` to 1e-6). The 3,767–3,784 nodes the live filter does delete
are ordinary short components exposed by edge removal, and deleting them *helps* slightly.
**E0c's 6bba +0.0049 — larger than the whole +0.005 promotion gate — has no support here and must be
re-measured on fold 1, not inherited.** `live-filter idempotent on the unedited graph: True` per
crop, which is what validates `min_track_len = 6` (P0-A's own `output_min_track_len`, with
`output_keep_division_components: true`, `output_prune_isolated: true`).

*Invariance, measured not assumed:* under the edges-only replay N_pred and node_recall are identical
on every crop **by construction** (one immutable node frame) — that arm proves nothing, exactly as
trap-14 thinking predicts. H0d is the authority; it re-runs the real `filter_short_track_components`
and reports the deletions above. 15/71 crops regress on adj_edge_J (worst `44b6_9be80b04` −0.0139).

---

**GATE 4 — the cost side of the mother gate is now MEASURED, and two GT-only falsifications fire.**

**Marginal cost of a FALSE fork** (`fp_cost_p0strict.py`, 1828 s, base parity PASS). After
suppress-all, admit the per-crop top `ceil(q x mothers)` shortlist pairs ranked by the frozen
flow-midpoint residual, with GT used **only** to exclude true dividers, so every admission is false
by construction and the delta is a pure cost:

| q | forks admitted | composite | Δ vs suppress-all | per fork |
|---:|---:|---:|---:|---:|
| 1e-5 | 71 | 0.899097 | **+0.000000** | 0.000e+00 |
| 1e-4 | 220 | 0.898827 | −0.000270 | −1.228e-06 |
| 1e-3 | 1,847 | 0.896159 | −0.002937 | −1.590e-06 |

**The top-ranked 71 false forks are exactly metric-free** (off-annotation edits cost nothing), and
the asymptotic cost is ~**1.6e-06** composite per false fork. Against a **+0.003921 per true fork**
gain (`h0c_post − supp_post = +0.074498` over 19), the *edge-side* value ratio is ~2,470:1.
**So the edge cost is NOT the binding constraint on this substrate** — at k=19 the division-FP term
is `0.1*(19/27 − 19/26) = −2.707e-03`, **1,700x larger** than the edge cost. The gate must be tuned
against the division-J denominator, exactly as the metric agent's kernel says, and the "unconditional
edge cost" caution from the E0c era is quantitatively small here.

Marginal admit thresholds from that kernel with **`pi_vis` PINNED at 1.0** (never fitted — fitting it
is an annotation-coverage exploit), `[exact-pooled-OOF restricted to fold 0 / 44b6]`: edge-neutral
edit **1.55% at k=0 -> 15.89% at k=22**; pessimistic edit (steal 1 TP, add 1 FP) **12.76% -> 25.47%**.
A single fixed bar over-admits early and under-admits late — do not use one.

**`gt_lineage_census.py` over all 199 GT crops** (133,318 nodes, 128,883 edges, **151 divisions** =
26 + 125, matching the known denominators):

- **A hazard head over cell age is NOT MEASURABLE. 0 of 151 mothers has an observed birth** — no
  division mother's parent is itself a division anywhere in the corpus. Every "age" is
  left-censored at an annotation start, so `P(divide | age = k)` cannot be estimated. **Do not build
  the discrete-time competing-risks head.**
- **A GT-fitted refractory window is NOT MEASURABLE: 0 observed inter-division intervals.** Temporal
  NMS along a track is still worth building, but only as a **structural** constraint on our own
  predicted tracks (one admitted fork per track per window, keep the argmax) — never as a
  GT-calibrated threshold.
- Positive: GT tracklets are long — median **20** frames, mean 28.1, max 100, only 0.2% of length 1
  — so there is ample within-track context for temporal NMS to act on. Express windows in FRAMES:
  the zarr T axis declares `unit: "second", scale: 1.0`, a placeholder.

Artifacts (scratchpad `laneB/`): `reconcile_e0c_vs_p0strict.json`, `h0c_p0strict_f0.json`,
`h0d_p0strict_f0.json` + per-crop parquet, `fp_cost_p0strict.json`, `gt_lineage_census.json`.
The P0-strict fold-0 graphs are rematerialised as a first-class H0d surface at
`artifacts/kaggle/p0strict_f0_cache/graphs/0/` (71 parquet, E0c schema, gitignored).

### 2026-08-01 (LANE B — reconciliation RESOLVED, H0c measured on P0-strict, cost side priced)

#### Gate 1 — the 0.8986 vs 0.7595 gap is REAL, and it is mostly genuine matching quality

Both arms through **one entry point** (`biotrack.metric.score_pred_graph`), **one run**, the **same
71 fold-0 crops**. 255.7 s, 3 workers.

| arm (71 crops, 44b6, fold 0) | raw edge_J | adj_edge_J | node_recall | divJ | composite | N_pred |
|---|---:|---:|---:|---:|---:|---:|
| E0c cached post-wrapper | 0.767340 | 0.759549 | 0.948158 | 0 (0/93/26) | **0.759549** | 2,864,419 |
| P0-strict recovered LOEO f0 | 0.882245 | 0.898589 | 0.984573 | 0.01587 (2/100/24) | **0.900177** | 1,900,633 |
| **delta** | +0.114905 | +0.139040 | +0.036415 | +0.01587 | **+0.140628** | **−963,786** |

**Parity:** E0c reproduces the published `0.7595` at |d| = 4.9e-05 and matches branch A's
independently published per-crop `44b6_d29c9ab2` adjJ **0.8821** exactly; P0-strict reproduces
`reports/inventory/loeo_f0_strict.json` at **|d| = 0.000e+00 on every field**.

**Attribution** `[exact-pooled-OOF, fold 0 / 44b6 only]`: **+0.114905 (81.7%) genuine matching
quality** — edge precision 0.8646→0.9306, recall 0.8722→0.9444, **a million fewer nodes with HIGHER
node recall**, 66/71 crops ≥ E0c and only 2/71 regressing. Count multiplier contributes +0.024136
(17.2%), division +0.001587. So this is not a scoring artifact and not mostly a count effect.

**PROVENANCE CAVEAT — state this whenever the 22/26 or 0.9002 numbers are quoted.** Fold 0 runs the
**support pack's** `split_0`, not ours (8,363,159 B `12f6881e…` vs our 8,357,783 B `d3e89eb3…`), so
it conflates model vintage with pipeline. **The pack ships no training record at all** — no
`train_datasets`, no held-out declaration — and its own manifest says `…-400ep-snapshot-v1` while
the dataset is named "50ep". **The only evidence that 44b6 was held out is the directory name
`split_0`.** Evidence against memorisation: per-crop adjJ still spans 0.559–1.064, node recall
bottoms at 0.9256, divisions remain TP2/FP100/FN24, and the held-out OOF (0.9002) sits **below**
P0-A's public 0.913. Verdict: **clean-pending-provenance**. Fold 1 uses our own `split_1` and has no
such confound.

#### Gate 3 — H0c/H0d on the P0-strict substrate. Two implementations agree to 1e-6.

| arm (71 crops, 44b6, f0) | composite | Δ | adj_edge_J | divisions |
|---|---:|---:|---:|---|
| base | 0.900177 | — | 0.898589 | TP2/FP100/FN24 |
| suppress-all (live filter) | 0.899558 | −0.000619 | 0.899558 | TP0/FP0/FN26 |
| h0c_post (edges-only) | 0.973595 | +0.073418 | 0.900518 | TP19/FP0/FN7 |
| **h0c_refilt (LIVE path)** | **0.974054** | **+0.073877** | 0.900977 | TP19/FP0/FN7 |

`[GT-oracle, fold 0 / 44b6 = 14.94% of edge mass. NOT a pooled claim.]` Retention **19/22 = 86.4%**.

- **The circulated +0.0830 was the analytic k=22/m=0 ceiling. The frozen ranker drops 3 of the 22,
  and that +0.0096 gap IS the ranker. The ranker, not the substrate, is now the first loss.**
- Edge side is **positive** here (+0.0019 / +0.0024) where E0c was +0.0010 / **−0.0006**; even
  suppress-all alone is edge-positive.
- **Do NOT inherit E0c's live-filter cross-term.** P0-strict **+0.000459** vs E0c +0.001610 (44b6)
  and **+0.004894** (6bba). Mechanism: P0-strict has **zero** division-exempt short components
  (E0c has 134/122), so the D0′ hazard cannot fire and the retention guard is a literal no-op.
  E0c's 6bba figure is larger than the whole promotion gate and has **no support here**.
- Methodological note: the edges-only arm is node-invariant **by construction** and therefore proves
  nothing — the same shape as trap 14. H0d is the authority because it re-runs the real
  `filter_short_track_components` (3,767–3,784 nodes deleted, `refilter_identity: True` per crop).

#### Gate 4 — the cost side is priced, and the edge cost is NOT the binding term

Marginal cost of a false fork (post-suppression; GT used only to exclude true dividers; base parity
PASS): the top-ranked **71 cost exactly 0.000000** — off-annotation edits are metric-free — 220 cost
−1.23e-06/fork, 1,847 cost **−1.59e-06**/fork. Against **+0.003921 per true fork**. At k = 19 the
division-J denominator term is **−2.707e-03**, i.e. **1,700× larger than the edge cost**.

**Marginal admit threshold, `pi_vis` PINNED at 1.0: 1.55% (k=0) → 15.89% (k=22)** edge-neutral,
12.76% → 25.47% pessimistic. **One fixed precision bar is wrong in both directions** — this
supersedes any use of the 10.15% average.

#### Gate 4b — GT lineage census (199 crops, 151 divisions) kills two proposed heads

- **Age-hazard / competing-risks head is NOT MEASURABLE: 0 of 151 mothers has an observed birth.**
  Every age is left-censored. **Do not build it.**
- **A GT-fitted refractory window is NOT MEASURABLE: 0 inter-division intervals exist.** Temporal
  NMS remains worth building, but only as a **structural** constraint on our own tracks (argmax per
  track per window) — never GT-calibrated.
- Tracklets are long enough for NMS to act on: median 20 frames, mean 28.1. Express windows in
  **frames**; the zarr T unit is a placeholder.

H2a was not run; that slot went to H0d, which invalidated an inherited constant — the better trade.
Fold 1 (`aryaarun07/biohub-loeo-f1-strict` v2) is RUNNING on GPU; `split_1` was verified LOEO-clean
before spend (split 1 = train 71×44b6 / test 128×6bba, best epoch chosen on a validation subset of
the *training* embryo). New **trap 16** recorded.

### 2026-08-01 (RED TEAM — both selector feature sets fail a leakage budget; set (A) fails transfer)

Full audit: `..._RESEARCH/agent_runs/research_redteam_2026-07-31/REDTEAM_FEATURE_AUDIT.md`.

**The corpus fact that drives everything.** 44b6 is cell-DENSE (404 nodes/frame) and SPARSELY
annotated (label_fraction 0.0077); 6bba is cell-SPARSE (94.9) and DENSELY annotated (0.0971).
Background intensity differs **5.3×**, nuclear peak 2.1×, `edge_prob` 0.652 vs 0.819. Meanwhile
**division rate per 1000 GT nodes is 1.29 vs 1.11** — division biology is the *one* thing that
matches across families. Almost everything else our selectors are fed is **regime, not biology**.

**Family-discriminability** (per-fold HGB, GroupKFold by crop, budget ≤ 0.65):

| feature set | rows / crops | family AUC | verdict |
|---|---|---:|---|
| crop-level descriptors | 199 crops | **0.9955** | (reference) |
| component set (A), 34 feats | 40k / 78 | **0.865** | **FAIL** |
| single candidate edge, 10 feats | 139k / 199 | 0.793 | FAIL |
| fork geometry (B), 14 feats | 20k / 199 | 0.709 | FAIL |
| appearance-4 RAW | 22k / 199 | 0.699 | FAIL |
| (A) "hardened" to 7 feats | 40k / 78 | 0.697 | FAIL |
| **appearance-4, per-crop rank-normalised** | 22k / 199 | **0.532** | **PASS** |

Mean |SMD| for set (A) is **0.151** and for a single edge 0.118 — **the SMD trap reproduced on our
real deployment features**, exactly as the prior synthetic-patch result predicted.

**Set (A) does not transfer** `[cross-family-LOFO, GT proxy, 23 crops]`. Base rate differs **11.3×**
(0.357% vs 4.034%), so thresholds are mis-set before features matter. In **3 of 4 direction × label
cells the transferred selector is at or below random-selection utility** (lift 0.88× / 1.31× /
0.77×); cross-family AUC **inverts below 0.5** in the 44b6→6bba direction; 44b6's in-family CV AUC
is **0.354 on 21 positives**. Also measured: only **52% of GT edges inside deleted components are
uncontested**, halving the 5,311 ceiling to **~2,760**.

**The division appearance channel survives — but only normalised.** Family AUC **0.699 raw →
0.532 after per-crop rank normalisation**. The leak is entirely the two `peak_above_bg` terms
(0.673 alone, |SMD| 0.575); **`aniso(t−1)` alone is 0.504**, independently confirming it as
physically invariant. **The LOEO division AUC 0.719 is NOT yet a transfer claim** — it must be
re-measured on the rank-normalised set. If it holds ≈0.72 at family AUC 0.53, adopt it; if it drops,
the 0.719 was partly family-conditional.

**Boundary flag: SPLIT IT, two booleans, never fused, never continuous.**
Temporal (movie start/end) rate ratio **0.98**, family AUC **0.4979** — the cleanest feature in the
whole audit. Spatial (volume boundary) rate ratio 0.54, AUC 0.4108 — 6bba's deleted components are
**1.84× more likely** to touch the boundary, i.e. crop framing. Continuous `bdist_min_um` is worst
(0.5915).

**Normalisers.** Required: N1 per-crop rank (0.699→0.532); N2 density ÷ crop-median NN
(|AUC−0.5| 0.142→0.012); N3 `(I−p50)/(p99.9−p50)` (0.516). **Explicitly REJECTED: peak/background**
— the intuitive choice and measurably *worse* than raw (0.343). Standing rule, generalising the
`sister_um` counterexample: **report a normaliser's effect on BOTH family AUC and target AUC before
adopting it.** A noisy predicted denominator imports more variance than it removes.

**The methodological headline, from the red team's own self-correction.** It classified trajectory
smoothness PHYSICALLY INVARIANT on distributional grounds (block leakage 0.506, the lowest it
measured) and then **falsified it**: straightness 0.545/0.416, cos_mean 0.527/0.460 cross-family,
**reversed on both labels**. Conversely, several features it had flagged as family proxies
(`prob_mean`, `nn5_mean_um`, `cnt10_mean`) *passed* the sign test.
**MARGINAL FAMILY-BLINDNESS DOES NOT IMPLY SIGN-STABILITY AGAINST THE TARGET, AND VICE VERSA.**
Neither direction of that inference is safe. Only bilateral LOFO sign-stability counts.

**Drop list (16)**, headed by: absolute intensity/mass · forward persistence (`n_persist` 0.175 →
0.755, a complete reversal) · absolute local density (0.520 → 0.467) · crop aggregates (0.998) ·
raw model probabilities · `nuc_snr` · division-rate/synchrony priors · and
**`BIOHUB_SHORT_TRACK_MIN_LEN_BY_DATASET`** — a **live** env-driven per-dataset override in
`src/biotrack/wrapper.py` that keys the short-track filter on the crop stem. It defaults to `{}`
and is inert, but nothing prevented it being set. **Now locked by `tests/test_no_family_routing.py`**
(5 tests: override empty, env unset, every stem resolves to the global constant, and no `44b6_` /
`6bba_` literal gates control flow in the wrapper). Suite 33 → 38.

**Pre-registration adopted — 8 gates before either selector goes near a submission:**
Gate 0 leakage ≤0.65 · Gate 1 per-feature sign stability + base-rate ratio ≤3× (**necessary, not
sufficient** — 9/28 features pass while the model still transfers at 0.77×) · Gate 2 bilateral
pooled Δ ≥ +0.002 with the two LOFO directions agreeing within **3×** · Gate 3 family-oracle
ablation ≤ +0.0005 · Gate 3b normalisation control · Gate 4 P(Δ>0) ≥ 0.80 in the *weaker* family ·
Gate 5 unit economics against a **random equal-sized retention control**, not against nothing ·
Gate 6 count-multiplier ≤50% of the gain · Gate 7 basis tags on every number.

### 2026-08-01 (P0-CR SCORED 0.906 — reverse-time is BASE-DEPENDENT, and the structural read called it)

`ref 55147215` landed at **0.906** after ~4.5 h PENDING (kernel COMPLETE, no error throughout —
trap 13 at its extreme; P0-A and P0-B both scored within minutes on the same day).

| base | without reverse-time | with reverse-time | Δ | divisions |
|---|---:|---:|---:|---:|
| clean 0.913 | P0-A **0.913** | P0-B **0.914** | **+0.001** | −9 |
| v122 | v122 **0.908** | P0-CR **0.906** | **−0.002** | +10 |

**The signs differ. Reverse-time is not a general mechanism.** Same source-locked extract, same
`w = 0.20`, opposite outcomes on two bases. The slot bought a real answer rather than a confirmation.

**The pre-registered structural read predicted the direction before any score existed.** The
CANDIDATE_LEDGER recorded, from graph structure alone, that divisions moved **−9** on the clean base
and **+10** on v122, and flagged the mechanism as base-dependent on that basis. The arm whose
divisions moved *up* is the one that lost 0.002. **Reusable rule: measure the division-count
direction before spending a slot** — it is cheap, offline, and it worked.

**Confound, named:** P0-CR bundles the line-fit volume guard with reverse-time, so the −0.002 is
strictly (reverse-time + guard) vs neither. The guard repaired exactly **one** out-of-volume node
(`44b6_0b24845f` node 15274, z=64) out of ~120,673 and cannot plausibly carry −0.002, but this is
not a pure single-variable contrast and must not be quoted as one.

**What changes.** Deployment is unchanged — P0-B remains the base at **0.914**, and P0-CR was never
a climb candidate. What changes is a *rule*: **a mechanism validated on one substrate is unvalidated
everywhere else.** That is now the third independent instance this cycle — node budget (+0.00157 on
E0c → −0.00088 on v122 → −0.00001 on P0-B), the H0d live-filter cross-term (+0.004894 on E0c 6bba →
+0.000459 on P0-strict, because P0-strict has zero division-exempt short components), and now
reverse-time (+0.001 → −0.002). **Substrate transfer is this project's dominant failure mode, ahead
of overfitting and ahead of compute.**

It also retires the P0-B ambiguity in the honest direction: P0-B's +0.001 is still one LB quantum
and still not established as a real gain, but the mechanism is now known to be *capable* of moving
score in both directions, so the +0.001 is not automatically noise either. Do not re-litigate it
with another blend weight — that remains closed.

### 2026-08-01 (FOLD 1 LANDS — THE 22/26 SUBSTRATE ADVANTAGE DOES NOT TRANSFER)

`aryaarun07/biohub-loeo-f1-strict` COMPLETE in 6.89 h (Lane B estimated 2.70 h; 9 h session limit).
Manifest audit PASS: fold 1, arm strict, 128 unique 6bba crops, **our own `split_1`** weights
override, secondary OFF, DeepCenter OFF. The export fix worked — 4 clean files, gzip and cleanup
both ran, no 500-file dump.

| quantity (128 crops, 6bba, fold 1) | value |
|---|---:|
| adj_edge_jaccard | 0.7025 |
| node_recall | **0.8547** |
| division_jaccard | 0.0016 (TP 1 / FP 491 / FN 124) |
| composite | 0.7026 |
| **reachable GT divisions** | **66 / 125 (0.528)** |

**THE HEADLINE OF THIS CYCLE IS OVERTURNED.**

| substrate | 44b6 | 6bba | **POOLED** |
|---|---|---|---|
| E0c | 20/26 = 0.769 | 93/125 = 0.744 | **113/151 = 0.748** |
| clean903 | 20/26 = 0.769 | 96/125 = 0.768 | **116/151 = 0.768** |
| v122 | 15/26 = 0.577 | 68/125 = 0.544 | **83/151 = 0.550** |
| **P0-strict** | **22/26 = 0.846** | **66/125 = 0.528** | **88/151 = 0.583** |

P0-strict is the **best substrate on 44b6 and the WORST on 6bba** — below even v122's 68/125.
Node recall collapses **0.9846 → 0.8547** across the family boundary. Since 6bba carries **85.06%
of edge mass**, the pooled reach is **58.3%**, far below E0c's 74.8% and clean903's 76.8%.

**What this retracts.** The 2026-07-31 conclusion "the 0.914 platform and the division track
multiply rather than compete", and the follow-on "the ranker, not the substrate, is now the first
loss", were both drawn from **fold 0 only — 14.94% of edge mass**. On the 85% family the substrate
is the worst we have measured. The +0.073877 H0c figure is a **44b6 GT-oracle** number and must
never be pooled or extrapolated. The honest statement is now: **on the deployment substrate the
division track has LESS pooled headroom than on E0c, not more.**

I flagged the 44b6-only bound every time I quoted it, and it still produced a wrong strategic read
for a full cycle. **Naming a caveat is not the same as acting on it.** The fold-1 measurement should
have been ranked above the H0c replay, the cost-side pricing and the cascade design — all of which
were built on a 15%-mass result.

**This is the FOURTH substrate-transfer failure of the cycle**, after node budget, the H0d
live-filter cross-term, and reverse-time. It is now the project's defining failure mode.

### 2026-08-01 (LANE A — CLOSED. Headroom exists; the GT-free selector cannot find it.)

Exact replay, 116 of 199 crops (80.1% of target signal after re-ordering by signal-per-second).
**Parity PASSES:** pooled baseline **0.645667332** vs ledger 0.645667332, |d| = **1.11e-16**,
**0/116** per-crop disagreements, and per-family exact to **0.00e+00** (44b6 0.722882164,
6bba 0.639929760).

| arm | components retained | corpus-scaled Δ | 44b6 | 6bba | bar |
|---|---:|---:|---:|---:|---:|
| **selector (GT-free)** | 11,683 | **−0.007761** | +0.001230 | **−0.011504** | +0.002 |
| **oracle (GT)** | 763 | **+0.008727** | +0.011003 | +0.012475 | +0.002 |

**VERDICT: CLOSE component retention.** The headroom is real — the GT oracle clears the bar more
than 4× over at **+0.008727**, and it is bilaterally positive. But the deployable selector is
**−0.007761**, and its two families **disagree in sign** (+0.0012 vs −0.0115), which is precisely
the failure the red team predicted from family AUC 0.865 and transfer at or below random utility.

The mechanism is stark: the selector retains **11,683 components where the oracle retains 763** —
15× too many — adding **49,688 nodes** and **1,297 false edges** to achieve edge TP **−117**. The
oracle adds 3,257 nodes for edge TP **+1440**. The signal exists and is worth having; nothing we can
build from deployment-observable features locates it.

Recorded for anyone who reopens this: the honest ceiling is **+0.0087 corpus-scaled**, so the lane
is not closed for lack of headroom — it is closed because **selection is the binding constraint**,
and three independent corrections all deflate the naive estimate (per-component attribution
overstates the joint effect ~2.4×; only 52% of GT edges in deleted components are uncontested;
retention is additive so the unit is 6.57e-06, not 1.095e-05).

Artifacts: `reports/inventory/loeo_f1_strict.json`, `loeo_f1_strict_manifest.json`,
`laneA_component_retention_replay.json`. Fold-1 CSV preserved outside the worktree at
`..._RESEARCH/exports/loeo_f1_strict/` (33 MB).

---

## 2026-08-02 — Arm-B blocker: the diagnosis was wrong, the real defect is in safe divisions

**Execution.** Picked up the arm-B deployment blocker (`biohub-p2-armb-flowgate` ERROR:
`6bba_05db0fb1: invalid lineage degree`, one node in 121,003 at out-degree 3, zero in-degree
violations). The inherited handoff attributed it to `motion_relink_edges` and prescribed an
out-degree ≤ 2 guard at relink edge admission. **Read the deployment source before implementing
it — the attribution is false and the prescribed fix is a silent no-op.**

**Why the relink cannot be the site.** `motion_relink_edges` runs FIRST and REPLACES the whole
edge list (`edges = motion_edges`). Its `linear_sum_assignment` is one-to-one per frame pair,
`unmatched_sources` blocks a second match in the relaxed pass, and a node is a source for exactly
one frame pair. Out-degree ≤ 1 by construction. Measured over 300 randomized dense frames:
**max out-degree 1, max in-degree 1**. `close_single_frame_gaps` guards `source_id in outgoing`
and also cannot raise a source past 1. `OUTPUT_SINGLE_CHILD_REPAIR` and
`OUTPUT_DIVISION_GEOMETRY_FILTER` both default `"0"` and are never set in the deployment env,
so neither is live (same dead-code class as correction 6).

**The real defect.** `add_safe_divisions_postlink` builds `proposals` over
`source_ids × candidate_ids`, so several candidates may share a source. The admission loop dedupes
on the **target** only (`used_targets` / `incoming`) and never advances `out_by_source`. With
`frame_cap ≥ 2` and `global_cap ≥ 2`, one source with an existing child takes **two** safe
divisions in one frame → **out-degree 3**. This is the only mechanism that yields the observed
signature — one violating node, **zero** in-degree violations — because the target-side dedupe is
correct and there is no source-side dedupe at all. Arm B's role is real but indirect: it changes
which targets are left unlinked, shifting the candidate population until the latent defect fires.

**Repair, at admission rather than post-processing.** Live `out_degree_now` map; reject any
proposal whose source already holds 2 children; count `safe_division_skipped_outdegree`; advance
on admission. Existing edges preserved, existing cost order (`proposals.sort` by score) untouched,
target in-degree ≤ 1 still enforced. Added `assert_degree_invariants(edges, stage)` after the
relink/repair, gap-close, safe-division and export stages so a future breach names its own stage.
Also pre-initialised the new stats key — `stats` is a plain dict and `+=` on a missing key would
have raised `KeyError` inside the kernel.

**Numbers.** Pre-fix reproduction: out-degree 3 on exactly one node, in-degree clean. Post-fix:
**exactly one edge rejected, zero edges added**, node set / coordinates / times identical, and
**bit-inert** when only one candidate exists (P0-B path untouched). Tests **42 → 49**; claims 53
still resolve. Both notebooks rebuilt: blast radius 3 cells, and the arm-B and baseline notebooks
still **differ in exactly one character** (`BIOHUB_ARMB_FLOW_GATE`), so the causal pair survives.

**Decision.** The baseline arm **must be re-run**, reversing the previous handoff's "do not re-run
the baseline". The fix is shared code, so the baseline's byte-identity to
`4c285cae0c220a11…` is now the regression gate that proves the fix is inert on the deployed path.

**Next step.** Push both kernels, audit, then submit arm B SOLO. **BLOCKED:** the Kaggle push was
denied by the harness permission classifier this session; it needs the user to run the push or
grant the permission. No GPU spent, no submission slot spent.

### Same day — real-data replay confirms the diagnosis, and corrects one of my own claims

**Execution.** 40 cached P0-strict crops (stride 5 over both folds), complete wrapper, arm-B gate
ON, 6 CPU workers. One pipeline pass per crop; the state entering `add_safe_divisions_postlink` is
captured and BOTH the pre-fix and post-fix implementations run on that identical state, so the
comparison isolates the fix exactly. `prob = 0` proxy, per the established finding.

| | pre-fix | post-fix |
|---|---:|---:|
| crops with out-degree > 2 | **1** (`44b6_a2bb48bb`, node 16693) | **0** |
| in-degree violations | 0 | 0 |
| violations at export | — | **0** |
| invariant assert fired | — | 0 |

**The defect is real on real data**, not only in the synthetic reproduction — one crop in 40 on the
training substrate, consistent with one node in 121,003 on the test substrate.

**CORRECTION to my own earlier claim in this session.** I wrote that the fix "rejects exactly one
edge and adds zero". That is true only when the safe-division cap is slack. On `44b6_a2bb48bb` the
**global cap binds at exactly 166**, and because the guard `continue`s rather than spending the
slot, the rejection frees budget for the next-ranked valid proposal: removed `(16693, 17368)`,
**added** `(38172, 38903)`, total 166 safe divisions before and after. **The fix is a one-edge SWAP
under a binding cap, not a pure removal.**

Kept deliberately — an invalid proposal must not consume budget, and the alternative (charging the
cap for a rejected proposal) would be an extra semantic change, not a smaller one. Now locked by
`test_rejected_proposal_does_not_consume_the_cap_budget`, which reconstructs the binding-cap
reallocation from first principles and reproduces it exactly. Tests **49 → 50**.

This does not touch the baseline regression gate: the guard is inert when no violation exists, and
live P0-B passed its own degree guard, so its artifact must still hash to `4c285cae0c220a11…`.

**Still blocked** on the Kaggle push permission. No GPU spent, no submission slot spent.

---

## 2026-08-02 (later) — Arm B submitted; residual census; sister-gate relaxation FALSIFIED

**Kaggle submission `55181562`, pending.** Codex ran both kernels. I did the independent
verification and the productive CPU work while the scorer runs. No GPU, no further slots.

### Independent artifact audit — the fix is verified in production

Baseline sha256 `4c285cae0c220a11…` — **byte-identical to deployed P0-B with the out-degree fix
applied**. Since the fix is shared code, that is the proof it is inert on the P0-B path, so the
arm-B delta stays attributable to the gate change.

Arm B `83498f9e27d4453212f1d2e75b2b4999939733d1ce4fa183b0e4d670cc6ef6dd`, 237,916 rows.
Against the failed 2026-08-01 run: same 121,003 nodes, **116,914 → 116,913 edges**, **max
out-degree 3 → 2**, violating nodes **1 → 0**. **Net production effect: exactly one edge removed.**
The safe-division cap was slack on `6bba_05db0fb1`, so a pure removal rather than the cap-bound
swap seen locally. Degrees keyed on `(dataset, node_id)` per trap 24; 0 dangling, 0 non-consecutive,
0 duplicates, 0 negative coordinates.

Churn vs P0-B **7.148%** (caveat predicted 7.2%). Divisions **305 → 318**. Edge length max
7.312 → 7.846 µm, >6 µm 24 → 49, **>10 µm and >14 µm both zero** — the "one property to watch"
(arm B can in principle exceed the relaxed radius) is now **closed empirically**.

### Residual-error census — 50 OOF crops, exact scorer

`score 0.734918 = adj_edge_J 0.734418 + 0.1·div_J 0.005000` ·
edge TP/FP/FN `28904/4193/6704` · division TP/FP/FN `1/151/48`.

**Corrected oracles** (a repaired FN becomes a TP; my first pass wrongly scaled FN out of the
denominator and understated every recall ceiling):

| oracle | Δ |
|---|---:|
| edge recall perfect | **+0.170959** |
| division perfect (both) | +0.099500 |
| edge precision perfect | +0.082648 |
| division recall perfect | +0.024000 |
| **edge: 10% of FN recovered** | **+0.017096** |
| division precision perfect | +0.001541 |
| node ratio → 0 | **−0.008205** |

6bba carries **95.2%** of missed edges (6,380 vs 324) at score 0.7012 vs 44b6 0.9014.

`total_node_ratio = −0.1476` — we under-predict nodes by 15% and the count adjustment **pays us
+0.0082 for it**. Adding nodes surrenders that bonus first. Not a free axis.

### Sister-gate: designed, measured, closed as a relaxation

The live gate is `SAFE_DIV_SISTER_MAX_UM = 8.5`, **not** the `DIV_SISTER_MAX_UM = 8.0` that
correction 6 killed (re-confirmed dead: `OUTPUT_DIVISION_GEOMETRY_FILTER` defaults "0", never set).

GT geometry, 151 true divisions over all 199 crops: **sister p50 = 10.570 µm**. The gate sits at
8.5 — **below the median** — and rejects **70.9%** of true divisions, versus 42.4% and 40.4% for the
two parent gates. It is genuinely mis-centred, which is a real and previously unquantified fault.

But relaxing it loses. Break-even precision for added candidates is only 0.498% *on the division
term alone* — however each safe division also emits an edge that lands in edge FP. Measured both
sides: added candidates need **≈20–25% precision** to be net positive. Relaxing 8.5 → 12.0 µm
unlocks **+23 of 151** true divisions (1.58×) while candidate volume grows as
**(12/8.5)³ = 2.81×** → achievable precision **≈2.7%** → division gain ≈+0.0011 against edge cost
≈−0.0047, **net ≈ −0.0036. FALSIFIED, on CPU, for free.**

Seventh instance of oracle-clears/selector-fails: oracle +0.024, achievable 2.7% vs required 20–25%.

**A re-aim at constant admission count is NOT falsified** — no extra edge cost, so any precision
gain is pure. Ceiling is +0.024 even at perfect recall.

**WITHDRAWN:** I first tested `sister_dist / local_spacing` on **GT** density and it looked
strongly falsified (CV 0.2997 → 0.4845, family transfer worse). Invalid — GT annotates ~670 nodes
per crop against ~25,000 predicted, so GT spacing (p50 22.9 µm) is not what a deployed gate would
compute. It falsifies a variant nobody would ship.

### Recommendation

**Next primitive attacks 6bba edge recall, not divisions.** +0.171 at oracle, only 10% capture
needed to clear +0.011 to 0.925, 95% of mass in one family. Divisions cap at +0.024.

Reproduce: `scripts/armb_census.py`, `scripts/gt_division_gates.py`,
`scripts/audit_armb_artifact.py`. Full writeup `reports/ARMB_RESIDUAL_CENSUS.md`.

---

## 2026-08-02 (result) — Arm B scores 0.914. Zero public movement.

`55181562` COMPLETE at **0.914**, identical to P0-B `55136908`. Expected +0.0045, delivered <0.001.

Arm B was the only mechanism ever to clear every gate here: P0-strict +0.0079822, E0c +0.0088059,
P(Δ>0)=1.000, min-fold over +0.005, 144/144 crops parity-exact. It changed **7.148%** of the edges
on the test movies. The public score did not move.

**The build was not at fault.** Baseline re-run byte-identical to deployed P0-B
(`4c285cae0c220a11…`) *with the fix applied*, proving the repair inert on the deployed path. Audit
clean: max out-degree 2, in-degree 1, zero dangling/non-consecutive/duplicate/negative rows. Churn
7.148% against a pre-registered 7.2%.

**Diagnosis: the public instrument cannot resolve the effect.** Measured GT annotation density
across all 199 training crops:

| family | crops | med annotated nodes | med annotated edges | med est_nodes | coverage |
|---|---:|---:|---:|---:|---:|
| 44b6 | 71 | 214 | 209 | 32,681 | **0.655%** |
| 6bba | 128 | 826 | 800 | 9,691 | **8.529%** |

Total annotated edges over 199 crops: **128,883**. Edge Jaccard scores only annotated cells, so the
4 public movies carry roughly **2,000 scored edges**: 1 edge ≈ 0.05% Jaccard, and +0.008 needs ~16
net correct. Arm B's ~8,335 churned edges intersect ~143 annotated ones; the needed ~56/44 split is
inside the noise of which cells were annotated.

This also explains correction 3 (6bba is 85% of edge mass) mechanistically: 6bba is **13× better
annotated** than 44b6.

**Two conclusions.**

1. **Retain P0-B; do not adopt arm B; do not discard it.** OOF P(Δ>0)=1.000 over 144 crops may
   still pay on the private set, which has more movies and therefore more resolution. Public
   flatness is not evidence of worthlessness.
2. **Stop using the public LB as a measurement instrument.** OOF spans 128,883 annotated edges and
   is ~64× more sensitive. Never spend a slot to resolve an effect below ~0.005.

Caveat: the density figures are from *training* GT; there is no test GT. The inference is
consistent with everything observed but is not proven.

Substrate transfer now stands at **seven failures and zero confirmed successes** — arm B was the
one claimed success and it went flat. Next cycle's first job is the instrument question, then 6bba
edge recall (+0.171 oracle, 10% capture clears +0.011). HANDOFF.md rewritten as the cold-start brief.

---

## 2026-08-02/03 — Substrate error, retraction, two deployment defects, four lanes

Consolidated entry covering commits `b0ce6c2`..`cf673d0`. The journal had drifted 12 commits
behind the reports; that is a standing-workflow failure and this closes it.

**The cycle's dominant event: I measured on the wrong substrate.** Built four instruments and three
lanes on `artifacts/kaggle/p0strict_cache`, which `reports/inventory/wsf_ROUTE1_VERDICT.json` had
already examined and rejected in writing before I started — "NO. It is a POST-wrapper surface and
arm B edits a PRE-wrapper stage." Zero `edge_prob` against 1,801,603 pre-wrapper edges, 83,260
relink-population nodes missing, 92% of coordinates linefit-shifted, and the cache edge list IS the
relink output. **Retracted**: "arm B is +0.000435". `+0.0079822` stood unrefuted.

**P0 parity, correct substrate, 199 crops:** arm B = **+0.0084877** pooled (44b6 +0.0139120, 6bba
+0.0076907, bootstrap CI **[+0.00745, +0.01365]**). WS-F's 144-crop +0.0079822 was right; the
+0.0005 residual is crop population.

**Two latent deployment defects, both found only because the substrate was fixed.**
Safe divisions could emit out-degree 3 (target-side dedupe, no source-side). Gap close could give a
node two parents (`used_starts` and `used_isolated` disjoint, neither guard consulting the other) —
this one killed the census at crop 185/199 and **never fired on the wrong substrate**, which lacks
83,260 of the nodes the gap-closer sees. Same defect class both times: admission dedupes against one
bookkeeping record while another path writes a different one. Both fixed at admission, both locked.
Tests 42 → 51.

**Lane O — sampling variance FALSIFIED.** With the true +0.008488 a 4-movie panel moves UP with
p=0.8995, DOWN 0.0683, FLAT **0.0322**. We observed the 3.2% outcome. Family mix (0.82–0.93),
weighting (0.906 vs 0.882) and concentration (top-20 = 49.6%, ESS 142/199) all fail to rescue it.
Remaining hypotheses B/C/D not separable without test GT. **Decision rule: promotion ≥ +0.015,
submission ≥ +0.020, trust OOF ranking not magnitude.**

**Lane B — detection owns the loss.** 72.2% of edge FN is detection (17,444, oracle **+0.10332**);
association-addressable 19.8%, ceiling **+0.032931**. Denominators only became trustworthy on the
correct substrate: relaxed surface 2,292 → **77,599**, starvation base rate 1.544 (impossible) →
**0.040245**, i.e. 1 in 25 vs 1 in 834 for the undifferentiated surface.

**Lane D0 — new detector or nothing.** 15,296 missing GT nodes (11.47%), 28,732 edges at stake.
44b6 misses **1.37%**; 6bba misses **13.28%** with **40.5% having nothing within 15 µm**. Class C
(wrapper removed it) ≈ 0. Priced: +0.103322 against −0.008633 of surrendered node-ratio bonus, net
≈ **+0.095**, ~3× association.

**D1/M2 settled without GPU.** The A/B/D partition is exactly computable from the peak rule. The
detector grid is **isotropic at 1.625 µm** (downsample [1,4,4] cancels the 4× voxel anisotropy), so
the pool suppresses only ±1.625 µm against ~6.5 µm spacing — class B small, arms N/TN likely dead.
`compute_detection_loss` confirmed: `neg_weight=0.1`, no ignore mask, single-voxel targets →
**91.5–99.3% of real nuclei explicitly trained as background.** Two sub-hypotheses killed free:
target-voxel collisions **0 of 133,318**, and perfect-heatmap recall ceiling **1.0000 at every
sigma** → the Gaussian-target line is dead as a recall fix; the LOSS is the mechanism.

**C0-FULL never completed** — first attempt was on the wrong substrate and stopped; the relaunch was
killed by my own process cleanup using a 30-minute window. C1 never started.

**Decision: no submission.** Nothing cleared +0.020. Slots consumed remain 10.

---

## 2026-08-05 — Leaderboard re-baseline: the field moved and we did not

Public top is now **0.948**; 0.924 is ~15th. **P0-B at 0.914 is well off the pace** and the
0.920–0.925 target this project was pursuing is obsolete.

Pulled public CC0 kernel `raykkretzschmar/biohub-harmonic-bidirectional-association-v1`. It is
**our own base again** — clean913, same three datasets, wrapper cell byte-identical in size — and
its reverse-time patch is character-for-character our `_bi_new` with **one changed expression**.

We blend forward/reverse as an **arithmetic mean of logits**:
`edge_logits_pair = (1-w)*forward + w*reverse_aligned`.

It blends as a **weighted harmonic mean in probability space**:
`harmonic_prob = 1/((1-w)/forward_prob + w/reverse_prob)`, renormalised, logged, then affinely
rescaled back onto the forward logit scale so the downstream threshold and ILP see the same range.
λ = 0.20, identical to our `BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT`. The harmonic mean is dominated by the
**lower** of the two directions, so a link must be supported both ways; the arithmetic mean lets one
confident direction carry a bad edge.

Provenance: transcribed from `yusuketogashi/no-hack-biohub-cell-another-approch-3rd` v18, **CC0**.
Satisfies the external-asset rule with attribution.

**Consequence for the planned work.** The C0-FULL/C1 association programme is measured against P0-B.
Harmonic fusion rewrites the edge logits, which changes which targets are left unlinked, which
changes the entire edge-FN population that Lane B's base rates and Lane C's ceiling are computed
from. **Running C0-FULL on the old base would repeat the substrate error with a different
substrate.** Overnight run held pending re-baseline.

Carry forward: P0-CR showed reverse-time is **base-dependent** (+0.001 on clean913, −0.002 on v122).
Harmonic sits on the same pathway, and arm B's relink cost consumes `prob`, which harmonic fusion
directly rewrites — so arm B and harmonic **will interact** and must be measured together, never
assumed additive.

---

## 2026-08-05 (result) — P3 harmonic scores 0.915. The re-baselining theory is falsified.

`55274582` COMPLETE at **0.915**, against P0-B 0.914. **+0.001.** New best; slots consumed 11.

Harmonic mutual-support fusion is worth **+0.001** on our base — the same magnitude the arithmetic
reverse-time blend gave over clean913. A marginal refinement of the same mechanism, not a different
class of thing.

**The theory this submission was built to test is dead.** The hypothesis was that the field's move
to 0.93–0.948 was carried by this public CC0 rule and that adopting it would put us near 0.93. It
put us at 0.915. **Whatever separates the 0.93+ teams from us is not harmonic fusion**, and it is
not in the public notebooks we have examined — the Kretzschmar kernel and the "clean 913" frontier
kernel both turned out to be our own ancestor with one change.

This supports the instinct that the leaders hold genuine proprietary edge, and it reframes the
remaining 0.033 gap as a research problem rather than an adoption problem.

**Calibration, scored honestly.** Predicted 0.913–0.928 with a modal band of 0.916–0.921 at 55%.
Actual 0.915: inside the range, below the mode. Third consecutive optimistic central estimate.
The signal that worked was **structural churn against a known-null reference** — arm B churned
7.148% for 0.000, P3 churned 4.368%, which correctly implied a small gain. The signal that failed
was net cardinality (+1% nodes and edges), which I used to push the estimate up and which carried
nothing. Use churn-vs-null, not cardinality, for the next structural pre-registration.

**Standing implication unchanged and now sharper:** the only asset large enough to close 0.033 is
detection — +0.095 oracle ceiling with the root cause confirmed in the shared training code
(`neg_weight=0.1`, no ignore mask, 91.5–99.3% of real nuclei supervised as background). Association
caps at +0.033 at oracle with an unproven selector.

**C0-FULL/C1 must now re-base onto P3, not P0-B.**

---

## 2026-08-06 — v5 D1 audited: three blockers, the full-199 launch stopped

**Execution.** Pulled both v5 smoke kernels (`aryaarun07/biohub-p3-d1-smoke-f0`/`-f1`, both
COMPLETE). Downloaded manifests are sha256-identical to the archived copies. Ran the acceptance
audit that the previous handoff listed as "not yet run". It had never been run because the data
to run it on does not exist in the export.

**Three blockers.**

1. **The kernel never emitted the partition.** `{crop}__rows.parquet` has no `d1_class` and no
   `matched` column — raw spherical statistics only. `matched` can only come from the scorer's
   bipartite matching, a CPU step. Not a defect; a missing stage believed to exist.

2. **`scripts/d1f_probe.py` cannot load a v5 artifact.** Reads `__feat_near.npy` (kernel writes
   `__feat_max.npy`) and columns `d1_class`/`near_dist_um` (neither exists). `main()` *fits* H0
   instead of reading the checkpoint head. `build_arm(..., pi_crop=None)` is hardcoded with no
   temporal logic, so **H1/H2/H3/H4 are byte-identical arms** — four identical rows that would
   have read as convergent evidence. Never run.

3. **FATAL — features are identity-view, logits are post-TTA.** `predict_unet_transformer.py:372-388`:
   `unet_out, det_logits = model.encode(imgs)` binds features once from the identity view; the
   TTA loop then does `_, det_flip = model.encode(imgs_flip)`, **discarding every flipped feature
   map**, and averages only logits over 4 views (identity + 3 planar flips, `/4`).
   `d1_inject.py:264` passes the post-TTA logits and the identity-view features to the audit as a
   matched pair, and `d1_response_audit.py:5` states the misconception in a comment. Therefore
   `checkpoint_detect_head(feat)` cannot reproduce the deployed logit and **H0 parity is
   mathematically unavailable from a v5 export.**

**Damage is bounded, and the boundary is exact.** Every neighbourhood statistic is computed from
`logits_1zyx`, which *is* the post-TTA tensor. So the partition is sound; only the 32-D features
are contaminated. v5 = PASS on wiring/manifests/statistics/serialization, NOT RUN on M/C/T/L/D,
INVALID on D1-F.

**Numbers.** Built `scripts/d1_postprocess.py` (versioned `d1-derived-1`, deterministic,
manifest-driven, raw inputs read-only, calls the scorer's own matching). Match authority is
*measured*, not assumed: the pregraph reproduces the recorded parity target **52/52** on
`44b6_0113de3b`; the post-wrapper submission gives only 49/52.

| crop | GT | M | C | T | L | D |
|---|---:|---:|---:|---:|---:|---:|
| `44b6_0113de3b` | 52 | 52 | 0 | 0 | 0 | 0 |
| `6bba_57b7cc1e` | 1,659 | 1,314 | 242 | 39 | 64 | 0 |
| `6bba_6feb10f0` | 1,368 | 155 | 47 | 810 | 356 | 0 |
| **total** | **3,027** | **1,469** | **289** | **849** | **420** | **0** |

Of 1,558 unmatched GT: **C 18.5% · T 54.5% · L 27.0% · D 0.0%.** Basis: 3-crop smoke,
IN-FAMILY, **diagnostic only** — one crop was deliberately selected as extreme. Not a census.

**Class D is exactly zero.** Every missed GT node has a local maximum within 15 µm; at this
radius there is no evidence of a response-poor representation on these crops. T dominates at
54.5% — response present within 7 µm, nothing clearing `det_threshold = 0.96875` (logit 3.434),
i.e. the **calibration/operating-point** cause, which explicitly does not justify encoder work.
This points away from encoder retraining, which is the cheap direction, so the bar for believing
it goes up, not down. v6 first.

**Side finding, recorded not chased.** Pregraph-matched 1,521 vs submission-matched 1,505: the
post-wrapper stage (short-track filter, isolated-node prune) **destroys 16 GT matches, 1.05%**,
on every crop measured (52→49, 1314→1307, 155→149).

**Acceptance checks all pass** on the 3 crops: partition invariants; every distance ≤ 15 µm
(max observed **14.982 µm**, so v4's 24.4 µm cube-corner leak is genuinely fixed); peak parity
`voxel_accepted == is_local_max & over_threshold`; monotonic neighbourhood counts;
`subthr_localmax` rows all local maxima and all under threshold; features `(n,32)` finite;
terminal per-crop manifests agree with the aggregate.

**Decision.** Full-199 export **not launched**. Order: v6 TTA-consistent feature export
(inverse-transform and accumulate features over the same view list; assert
`detect_head(mean(aligned_features)) == mean(aligned_logits)` numerically) → rewrite
`d1f_probe.py` with a capability registry that hard-fails unimplemented arms → re-run the
3-crop smoke as v6 → **one** combined full-199 launch. Do not run full v5 then repeat 199 for v6.

**Next step.** v6 export patch. Tests 90 → **103**; new locks in `tests/test_d1_postprocess.py`
cover the partition rules and pin the TTA defect so a v6 fix must be a deliberate change.

---

## 2026-08-06 (correction) — I audited a file that never runs. Views are 8, not 4. Plus a fourth blocker.

**My error, corrected.** I reported the deployed TTA view list as "exactly 4 (identity + 3
planar flips), `/4`", citing `predict_unet_transformer.py:379-388`. **That block is replaced
at build time.** The generated notebook substitutes a full D4 set — identity + `flip(-1)` +
`flip(-2)` + `flip(-2,-1)` + `rot90(k=1)` + `rot90(k=3)` + `transpose(-1,-2)` +
anti-transpose — dividing by the runtime counter `_nv` = **8**. Lane 5 caught it; verified
directly against the generated notebook and locked by tests.

The instruction to derive the view list from the *generated notebook* rather than assume
four-view was explicit, and I did not follow it. The B3 conclusion is unchanged and in fact
strengthened: the feature/logit mismatch is 8-fold, not 4-fold.

**Meta-rule, now the important part: the deployment artifact is `base file + notebook string
patches`. Auditing the base alone audits a file that never runs.** Every constant quoted from
`vendor/` in the reports is subject to this. Mirror image of correction 6 (dead-code
constant).

Two further facts from the same read: the TTA patch guard is `print("TTA WARNING: block not
found - using default 4-way")` — **a print, not a raise** — so the view count depends on an
exact string match at runtime and appears in **no manifest field**. And there are **two**
8-view blocks: a primary and a secondary detection model.

**B4 — a fourth blocker that would have destroyed the full-199 export.** `_d1_flush` built
`_pl.DataFrame(_rows)` from heterogeneous dicts with no declared schema. polars infers from
the first 100 rows; emission is 96 non-GT rows per frame (64 uniform + 32 subthr). So any
crop whose first GT lands at frame >= 2 puts the first GT row past the window and **every
gt-only column silently disappears** while the terminal record still reports the right
`gt_rows` and `status: complete`.

Reproduced at pinned polars 1.42.1: 99 lead rows fine, **100 → total loss**. Measured over
all 199 GT geffs: **28 crops (14.1%)** would fire — 44b6 14/71 (19.7%), 6bba 14/128 (10.9%);
worst `6bba_767a1e17` (first GT t=46), `44b6_e29f0176` (t=30), `44b6_90724892` (t=29).
**All three smoke crops start at frame 0, so the smoke could never have caught it**, and the
composition gate checks cells and edit hashes, not artifact schema. Found by the red-team
lane, independently reproduced here.

**Fixed:** `infer_schema_length=None` plus a `_D1_REQUIRED_COLS` contract that raises. Locked
by three tests. Tests 103 -> **109**.

**Decision unchanged:** full-199 still not launched. v6 must additionally raise (not print) if
the TTA patch fails to apply, and record `n_views` / `tta_view_set` in the manifest.

**Next step.** v6 export patch: accumulate inverse-transformed features over the same 8 views
as the logits, assert `detect_head(mean(aligned_features)) == mean(aligned_logits)`.

---

## 2026-08-06 (correction) — the "3-crop" partition was a 2-crop 6bba-only figure

The v6 build swarm was terminated mid-flight by a session limit. Before dying, the
postprocessor-repair lane reported a baseline that contradicted a number I had propagated into
three documents.

**The bug was mine and it was arithmetic.** The total row of the partition table summed GT and
M over the **two fold-1 crops only** (3,027 / 1,469) while summing submission-matched over
**all three** (1,505). Verified from the artifacts: `44b6_0113de3b` GT 52, `6bba_57b7cc1e`
1,659, `6bba_6feb10f0` 1,368; 1,659 + 1,368 = 3,027 but 52 + 1,659 + 1,368 = **3,079**, and
M is **1,521**, not 1,469.

**The unmatched analysis is unaffected** — 44b6 contributes 52 GT and 52 M, so 1,558 unmatched
and the C 18.5% / T 54.5% / L 27.0% / D 0.0% split hold exactly as recorded.

**But the reason it is unaffected is the actual finding, and I had not stated it: every
unmatched GT node in the smoke comes from 6bba.** `44b6_0113de3b` is 52/52 matched. So `D = 0`
and the whole class distribution are **6bba-only measurements from two crops**, carrying zero
44b6 representation — on the family whose corpus miss rate is 1.37% against 6bba's 13.28%.
Anywhere the partition was described as covering the smoke's families, it did not.

This is correction C7 biting on my own numbers one commit after I wrote it down: the two
transfer directions must be reported separately, and a pooled label hid the fact that one
direction had no data at all.

Corrected in `reports/D1_V5_STATUS.md`, `HANDOFF.md` and the decision package. The earlier
journal table at 2026-08-06 retains 3,027 as the fold-1 subtotal it actually was.

**Swarm status:** all nine agents terminated by the session limit. Partial work salvaged to
their branches (`d1_postprocess.py` +504, `d1f_probe.py` +2,194, `d1_response_audit.py` +405)
as unreviewed WIP commits. Nothing merged to master.

---

## 2026-08-06 (halt) — cycle stopped at Stage A on instruction

Full record: `reports/CYCLE_OUTPUT_2026-08-06.md`.

**Score movement 0.000. GPU 0. Kaggle pushes 0. Submissions 0.** Public stays 0.915.
Tests 90 -> 224. Commits `2c91bc6` -> `6cdd016`, pushed.

The cycle set out to spend ~11.5 T4-hours on a 199-crop D1/D1-F export and did not. Five
instrument defects were found, **four of which would have returned a plausible answer rather
than an error**: byte-identical H1-H4 arms reading as convergent evidence; identity-view
features paired with post-TTA logits; a polars schema drop voiding 28/199 crops while
reporting COMPLETE; and family/checkpoint confounding that returns "representation deficit"
with probability ~1 from a basis mismatch.

Two further traps were caught while writing the fixes. Every research lane proposed
accumulating the TTA mean into `unet_out`, which `predict_edges` reads downstream -- that would
have destroyed the 0.915 association substrate under cover of a bug fix. And
`augpath_oracle.json` turned out to be the retracted post-wrapper run (40 crops, not 199)
sitting at the script's own default `--out`.

**A defect in the deployed detector.** The view written as the anti-transpose,
`rot90(imgs,1,(-2,-1)).transpose(-1,-2)`, is exactly `flip(-1)` -- already view 1. So the
deployed detector makes 8 encode calls over 7 distinct views, `flip(-1)` weighted 2/8 and the
true anti-transpose 0/8, divisor 8. Non-uniform on D4; not a group average. Against a true
uniform D4 average: max |Dlogit| 1.33 and 201/1,780 accepted peaks (~11%) change identity on
one frame. Decision: ship unchanged, because v6 must describe the deployed detector and
correcting the view set is a detector change that voids the anchors. Locked by test.

**Ten claims corrected, most of them mine.** The two that matter most: the TTA view count (I
audited `vendor/`, which is replaced at build time -- the base file is not the run), and the
partition, whose total row summed GT/M over two crops but submission-matched over three. Real
totals GT 3,079 / M 1,521. The deeper point that had never been stated: `44b6_0113de3b` is
52/52 matched, so **every unmatched GT node is 6bba** and D=0 is a 2-crop 6bba-only result with
zero 44b6 representation.

Also corrected: M1 is not CPU-exact (a changed accepted set changes existing edge logits
through cross-attention and source-softmax); |T u L| is not a ceiling and +0.015 needs ~2,221
perfect recoveries, not 3,400; `Y == X` is not the safety property; and a signed-residual test
does not catch wrong inverses, which come out exactly zero-mean.

**Unmerged, untested work is preserved on two branches** with resume points in the cycle
output. Agents 7-10 never started; Agent 9's recall-ceiling re-audit at true nuclear density
(9.68 um measured spacing vs ~10 um axial PSF) is the highest-leverage unstarted item, since it
can re-price the +0.10332 detection oracle before any GPU.

**Two decisions await the host:** a `.gitignore` negation for `data/d1_factorial/` (not made --
`.gitignore` is under a do-not-touch instruction), and the scope of the full run. FULL is
22.44 T4-h against ~11.5 remaining and cannot be sharded away, because the LOEO retarget binds
one fold at module level. Recommended: **split 1 only, 7.88 T4-h**, ~85% of the objective.

**On 0.930:** not on an evidenced path. +0.015 public would be 15x the largest public gain this
project has recorded (P3 +0.001; arm B +0.0085 OOF -> 0.000 public). Nothing on the shelf
clears +0.020.
