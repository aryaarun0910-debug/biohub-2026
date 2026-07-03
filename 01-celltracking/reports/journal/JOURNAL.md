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
- **V3 reproduction.** `notebooks/kaggle_dog_infer.py` implements the verified public 0.842
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
