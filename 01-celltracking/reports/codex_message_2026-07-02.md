# Codex research mission — Biohub Cell Tracking (Kaggle) — go maximal

You are a competitive-intelligence + research agent. Mission: exhaustively sweep the internet
and synthesize a WINNING plan for the Kaggle "Biohub – Cell Tracking During Development"
competition. Two Claude research agents already did deep methods/repo dives (condensed below) —
your job is to go BEYOND them: live competitive state, reverse-engineer top solutions, verify
the unverified, and turn everything into a ranked, buildable experiment queue. Cite every
concrete claim with a URL. Flag anything unverifiable. Be dense; no filler.

## Competition facts (ground truth)
- Task: 3D+time zebrafish light-sheet (DaXi) microscopy. Detect cell centroids, link across time,
  identify divisions, reconstruct lineages. CC0. Notebook-only, internet disabled, <=12h runtime.
- Data: zarr (T,Z,Y,X) uint16, ~(100,64,256,256). Voxel scale (z,y,x)=(1.625,0.40625,0.40625) um
  (Z ~4x coarser). Labels = sparse GEFF point graphs (~1-6% of true cells annotated; count
  adjustment uses metadata `estimated_number_of_nodes`). TRAIN = exactly 2 embryos (44b6, 6bba),
  129 crops; hidden TEST = different embryo(s) (embryo-disjoint).
- Metric: score = weighted_avg(adjusted_edge_jaccard) + 0.1*division_jaccard. Node matching =
  per-timepoint OPTIMAL bipartite within 7 um physical distance. adjusted = max(0, J*(1 - 0.1*
  (N_pred - N_est)/N_est)). Edges dominate ~10:1.
- Live state (2 Jul 2026): ~518 teams; leader ~0.875; public plateau 0.839-0.842; dominant public
  method = multiscale DoG + Hungarian; a from-scratch small 3D U-Net reportedly failed to
  generalize from the 2 embryos; divisions hurt early. Public LB = only 29% of test (71% private).

## Already established / built (DO NOT REPEAT — build on it)
- We have: exact-metric local harness (validated), lossless graph<->submission.csv converter,
  leave-one-embryo-out CV over all 129 crops, DoG anchor notebook + DAXI notebook (machine-tested),
  full 87GB data mirrored locally, DAXI U-Net I/O verified (in (N,1,Z,Y,X); out (N,2,Z,Y,X) probs;
  ch0=foreground ch1=contour).
- Host baseline internals (read from royerlab/kaggle-cell-tracking-competition code): ONE jointly
  trained UNetNodeTransformer (temporal U-Net + bi-cross-attn edge transformer), detection+linking
  end-to-end, NO ILP by default (opt-in flag). Detection loss = BCE, single-voxel positives,
  neg_weight=0.01. Edge loss = focal(g=2) on softmax(dim=0) (each t+1 target picks one parent; a
  source can win two => divisions), masked to annotated rows/cols. TWO latent host-code levers:
  division up-weight is a no-op; checkpoint selection uses acc*recall NOT the LB metric. Aug is bare
  (brightness+flips); model admittedly under-trained. Inference --det-threshold default 0.99.
- Metric is exactly reimplementable in numpy: per-timepoint min_weight_full_bipartite_matching(
  maximize=True) on weight 1/(1+d), scaled dist, 7um; matched_edge_mask = directed GT-edge inner-join
  on matched ids; adj Jaccard w/ count penalty; documented division-TP rule.
- Learned-detector plan (from methods dive): Spotiflow-3D (multiscale Adaptive-Wing heatmaps +
  stereographic-flow sub-voxel offsets; offline synth_3d/smfish_3d weights, fine-tune only) — BUT it
  does NOT ignore unlabeled voxels, so must inject the Linajea soft-mask (loss weight 1 within a
  nucleus radius of each point, 0.01-1e-6 elsewhere) + CPV aux head (regress voxel->nucleus-center
  vector) to separate merges/splits. Linker = motile ILP via SCIP (offline, MaxChildren=2/MaxParents=1,
  fit_weights sSVM tuned to our metric). Optional ITEC-style split/merge-then-retrack booster.
- Offline same-domain assets: unet-daxi.pt (20MB), unet-simview.pt (84MB), ZSNS001-005 DaXi embryos +
  full track CSVs + tracks_benchmark graphs (public.czbiohub.org/royerlab/), Ultrack, Trackastra ctc.

## YOUR RESEARCH TARGETS — obliterate every source
Search hard across: Kaggle (leaderboard/notebooks/discussions), GitHub, arXiv, bioRxiv, SSRN,
PubMed/PMC, Nature/eLife/Cell, image.sc forum, X/Twitter, lab pages. For EACH, extract concrete,
cited, actionable intel:

1. LIVE LEADERBOARD DELTA: refresh top-20 vs 2 Jul; score movement, submission-count patterns, any
   new leader or jump. Is 0.875 climbing? Winner trajectory.
2. TOP PUBLIC NOTEBOOK EXACT DIFF: open the current best public notebook(s) (the ~0.842 DoG recipe
   and anything above). Extract EXACT detector params (DoG scales, thresholds, NMS radius, count cap),
   linker params (gate, gap-closing), and how they calibrate node count. What precisely separates the
   >0.842 notebooks?
3. REVERSE-ENGINEER THE 0.86-0.875 TIER: from discussions, submission cadence, teaser comments — what
   are they plausibly doing beyond DoG (learned residual? ensemble? count calibration? embryo-specific
   tuning? Ultrack/motile ILP?). Evidence, not speculation.
4. COUNT CALIBRATION IN PRACTICE: what N_pred/N_est ratios and per-frame peak caps do strong public
   solutions use? Quantify the sweet spot given the mild 0.1 count penalty.
5. DIVISION REALITY: is anyone scoring division_jaccard>0 meaningfully? Fraction of 0.842->0.875 gap
   from edges vs divisions? Best public division-handling approach?
6. TEST-SET STRUCTURE: any host confirmation of #test embryos or size ratio to train? Affects trust in
   2-direction embryo CV.
7. RULES/ELIGIBILITY: confirm external-data + pretrained-model rules still permit DAXI/Ultrack/
   Spotiflow/simview weights; Gurobi/solver license concerns for winner reproducibility; team-merge and
   final-submission-selection deadlines.
8. VERIFY THE UNVERIFIED: read the Ultrack (Nat Methods 2025) and Zebrahub (Cell 2024) methods PDFs for
   exact ILP objective + division handling; find ITEC (bioRxiv 2026.03.12.711203) full method + any
   code release; confirm CTC Fluo-N3DL winners + methods (KIT-GE/EmbedTrack, Ultrack, Linajea++).
9. NEW LEVERS: any 2025-2026 paper/repo/tool (sparse-point 3D detection, learned association,
   division classifiers, SAM-for-tracking, foundation detectors) that transfers and is offline-usable.

## DELIVERABLE
1. A ranked EXPERIMENT QUEUE: each item = hypothesis, expected score delta, which embryo-held-out fold
   validates it, Kaggle <12h feasibility, and a go/no-go gate on the exact local metric. Highest EV first.
2. An explicit RED-TEAM: where is our current plan (DoG anchor -> numpy metric harness -> Spotiflow-3D
   + Linajea-mask + CPV residual detector -> motile ILP -> conservative divisions) most likely WRONG or
   suboptimal given the live competitive state? What would you do differently to reach 0.88+ on PRIVATE?
3. A short list of the single highest-value actions for the next 72 hours.
