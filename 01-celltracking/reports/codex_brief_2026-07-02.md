# Codex Tasking Brief — Biohub Cell Tracking (build ON TOP of existing research)

**Purpose:** Codex extends, red-teams, and operationalizes work already done in Claude Code.
Do NOT re-derive what is listed under "Already established" — spend effort on the OPEN
QUESTIONS and the SYNTHESIS deliverable.

---

## Already established (do not repeat)

**Built + verified locally (Claude Code):**
- Exact metric harness wrapping the organizer's `tracking_cellmot.evaluate` (validated: GT-vs-GT = 1.0).
- Lossless graph <-> `submission.csv` converter (round-trip verified).
- Leave-one-embryo-out CV split over all 129 crops (fold0 test=44b6: 71 crops/19.8k edges/26 div;
  fold1 test=6bba: 58 crops/48.7k edges/56 div). Full labels + inventory downloaded.
- Two Kaggle inference notebooks, machine-validated end-to-end on synthetic:
  `kaggle_dog_infer.py` (multiscale DoG + Hungarian ANCHOR) and `kaggle_daxi_infer.py`
  (DAXI detector, repositioned as external residual source).
- DAXI U-Net (`unet-daxi.pt`, 20,420,030 B) I/O VERIFIED by running on CPU:
  in `(N,1,Z,Y,X)` [0,1]; out `(N,2,Z,Y,X)` sigmoid probs; ch0=foreground, ch1=contour.
- Full 87 GB dataset mirrored locally (in progress) for CPU DoG sweeps + exact local scoring.

**Verified data facts:** 129 crops from exactly 2 embryos (44b6, 6bba); hidden test = different
embryo; labels ~1-6% sparse; count adjustment uses `estimated_number_of_nodes`; scale
(1.625,0.40625,0.40625). Metric = weighted_avg(adj_edge_jaccard) + 0.1*division_jaccard,
7 um bipartite node matching.

**Two methods reports already written** (methods_research_2026-06-30.md + a deeper pass in progress):
DAXI weights, Ultrack API/CBC solver, PAC-MAP heatmap math, Trackastra ctc, Linajea masked-loss,
DoG Section-5 config, Spotiflow (high level).

**Latest competitive state (from Codex's own 2 Jul delta):** 518 teams; leader 0.875; public
plateau 0.839-0.842; classical multiscale DoG dominant; divisions hurt early; a from-scratch
3D U-Net reportedly failed to generalize from 2 embryos.

### SLOT A — Biohub/Royer ecosystem deep-dive (full: reports/research/biohub_ecosystem_deepdive_2026-07-02.md)
- **Host baseline = ONE jointly-trained UNetNodeTransformer** (temporal U-Net + bi-cross-attn edge
  transformer), detection+linking end-to-end. NO ILP in default train/inference (opt-in flag only).
- **Detection loss** = BCE-with-logits, single-voxel positives, `det_neg_weight=0.01` (negatives
  downweighted 100x) — the host's sparse-GT mechanism. **Edge loss** = focal(gamma=2) on
  `softmax(logits,dim=0)` (each t+1 target picks one parent; a source may win two => divisions),
  masked to annotated rows/cols.
- **TWO host-code levers**: (a) division up-weight `weight[div_rows]=1.0` is a NO-OP; (b) checkpoint
  selection uses `acc*recall`, NOT the LB metric. Aug is bare (brightness+flips); model under-trained.
- **METRIC now exactly reimplementable in numpy**: per-timepoint OPTIMAL bipartite match via
  `min_weight_full_bipartite_matching(maximize=True)` on weight `1/(1+d)`, scaled dist, 7um gate;
  matched_edge_mask = directed GT-edge inner-join on matched ids; adj = max(0, J*(1-0.1*(N_pred-N_true)/N_true));
  exact division-TP rule documented. -> we can score locally on Kaggle WITHOUT tracksdata.
- **NEW offline assets**: `unet-simview.pt` (84 MiB) 2nd pretrained U-Net; ZSNS001-005 same-domain
  DaXi embryos + full track CSVs + tracks_benchmark graphs for pretraining.
- Inference default `--det-threshold 0.99` exists specifically to avoid over-detection FPs.

### SLOT B — Competitive/recent methods deep-dive (full: reports/research/competitive_methods_deepdive_2026-07-02.md)
- **Spotiflow-3D** (Nat Methods 2025) is the top learned-detector candidate: multiscale Adaptive-Wing
  heatmaps + stereographic-flow sub-voxel offsets; offline 3D pretrained (`synth_3d`,`smfish_3d`,
  fine-tune only). **CRITICAL**: it does NOT ignore unlabeled voxels — naive training punishes detecting
  the 94-99% unlabeled real nuclei. Must inject Linajea soft-mask.
- **Linajea soft-mask (THE sparse recipe)**: per-voxel loss weight = 1 within a nucleus radius of each
  annotated point, **0.01 (confocal) / 1e-6 (light-sheet) elsewhere** — soft down-weight, not hard ignore.
  Plus division oversampling (>=25% of iters contain a division).
- **CPV auxiliary loss** (Hirsch-Kainmueller): every fg voxel regresses the vector to its nucleus center
  (SSD) -> robustly separates merges/splits = the exact errors adjusted-edge-Jaccard punishes. Cheap add-on.
- **motile** = offline division-aware ILP on Kaggle via **SCIP** (no Gurobi): MaxChildren=2/MaxParents=1,
  EdgeDistance gate at 7um anisotropic, Split cost; built-in `fit_weights` sSVM tunes costs to OUR metric.
- **ITEC** (bioRxiv 2026): iterative "if split/merge lowers tracking cost, correct segmentation & re-track"
  loop (min-cost circulation). No code, but implementable as a post-hoc booster.
- **CTC analog** = Fluo-N3DL sparse light-sheet sets; winners KIT-GE/EmbedTrack (embedding+offset), Ultrack,
  Linajea++. Public Kaggle solutions still thin (~0.842 DoG); top 0.875 unpublished.
- Recommended offline stack: **Spotiflow-3D (masked loss)+CPV -> KNN candidate graph -> motile ILP
  (fit_weights) -> optional ITEC retrack**.

---

## OPEN QUESTIONS for Codex (competitive-intel + synthesis strengths)

1. **Leaderboard delta since 2 Jul:** refresh top-20; note score movement, submission-count
   patterns, any team that jumped. Is 0.875 holding or climbing? Any new leader?
2. **Public notebook diff:** find the EXACT current best public notebook (the ~0.842 recipe) and
   summarize its precise detector/linker params so we can diff against our DoG config. Any NEW
   public notebook above 0.842? What specifically differentiates it?
3. **Top-5 signal:** from discussions/submission patterns, what are the 0.86-0.875 entries
   plausibly doing beyond the DoG baseline (learned residual? count calibration? better linking?
   embryo-specific tuning?). Evidence, not speculation.
4. **Count calibration in practice:** what N_pred / estimated_N ratios and per-frame peak caps do
   the strong public notebooks actually use? The metric's count term is mild — quantify the sweet spot.
5. **Division reality:** is ANY public notebook scoring division_jaccard > 0 meaningfully? What
   fraction of the 0.842->0.875 gap is edges vs divisions?
6. **Test-set structure:** any host confirmation of how many embryos are in the hidden test, or
   its size ratio to train? Affects how much we trust the 2-direction embryo CV.
7. **Rule/eligibility:** confirm external-data + pretrained-model rules still allow DAXI/Ultrack/
   Spotiflow weights; any Gurobi/solver license concerns for winner reproducibility.

## DELIVERABLE for Codex

A single **prioritized experiment queue** merging (our built assets + both methods reports +
your competitive intel) into ranked experiments, each with: hypothesis, expected score delta,
which embryo-held-out fold(s) validate it, Kaggle-runtime feasibility (<12 h), and a
go/no-go gate on the exact local metric. Put the highest expected-value experiment first.
Explicitly flag where our current plan (DoG anchor -> Spotiflow residual detector -> learned
association -> conservative divisions) is most likely WRONG given the live competitive state.
