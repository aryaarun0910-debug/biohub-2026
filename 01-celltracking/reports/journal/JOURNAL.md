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
