# Codex — red-team the 0.94 WIN pathway + FIND THE ORTHOGONAL EDGE

Mission (two jobs):
1. **Red-team the pathway to ~0.94** below — is the arithmetic sound, which phase is mis-weighted, what's the
   realistic private-board ceiling, and where does the edge-Jaccard climb *actually* come from?
2. **Find the ORTHOGONAL EDGE** — deploy a parallel research team to sweep EVERYTHING (papers, preprints,
   repos, issues, other Kaggle comps, cross-domain tracking, talks, theses, "silly little things") for a
   high-EV lever **nobody in bioimage cell tracking uses.** We are playing to WIN (1st, private board), not
   place. We have until 29 Sep 2026.

## Hard filter (every finding)
NEW + ACTIONABLE + high-EV, each with: the edge, why it's orthogonal, a cited source URL, offline-Kaggle
feasibility (notebook-only, ≤12h, internet OFF at submission, T4×2), and the first experiment to test it.
Rank by EV × novelty × feasibility. Legal only (no private-label reconstruction, no ToS/ rules breach).

## The competition (self-contained)
Kaggle "Biohub - Cell Tracking During Development" (Royer Lab / CZ Biohub). 3D+time light-sheet zebrafish
embryos. Detect nuclei per frame → link across time → detect divisions. Metric = `weighted_avg(adjusted
edge-Jaccard) + 0.1·division-Jaccard`, where `adj_edge_J = max(0, J·(1 − 0.1·(N_pred−N_est)/N_est))`
(ONE-SIDED count penalty, no upper clip → under-count with perfect edges scores >1.0). Node match =
one-to-one within 7µm physical, weight 1/(1+d). Voxel (z,y,x)=(1.625,0.40625,0.40625)µm (z 4× coarser).

KEY: the **private LB (71%) is a DISJOINT hidden embryo**; the public LB (29%) is 4 visible movies with
labeled train copies (leakable). **Our embryo-held-out OOF IS the private-LB measurement** (same ruler).

## Where we are (MEASURED, 2026-07-06)
- Only 2 train embryos (`44b6`, `6bba`), ~1% sparse annotations. Full 199 crops local + Kaggle.
- Organizer learned stack (vendored `vendor/kaggle-cell-tracking/src/tracking_cellmot/`): TemporalUNet3D
  detector + SimpleNodeTransformer edge model + `tracksdata.solvers.ILPSolver` (SCIP-backed, division-native).
- **Detection ~95% node recall on held-out embryos = essentially SOLVED. Not the bottleneck.**
- **Edge Jaccard ~0.70 (ILP) — THIS is the entire gap.** Greedy min-fold OOF 0.559 → ILP ~0.67 (+0.11
  proven; ILP also killed a greedy division-FP catastrophe, 10,786→~1-28/crop).
- **Divisions currently 0.00 of the 0.1 term — untouched.**
- **More training OVERFITS:** 45-epoch (1.9×) model scored WORSE than 30-epoch on all held-out crops
  (0.60-0.64 vs 0.64-0.70) + lower recall. Training on our 2 embryos is a NEGATIVE lever → the bottleneck
  is GENERALIZATION to the disjoint embryo.
- Rules: external public data/models ALLOWED (cleared). Compute: Kaggle T4×2 free, ≤12h, internet-off submission.

## The 0.94 arithmetic (tear this apart)
`~0.88 edge-Jaccard + 0.1·(~0.6 division-Jaccard) ≈ 0.94.` So winning = push edge-J 0.70→0.88 AND turn
divisions into a real ~0.6 term, on the DISJOINT embryo.

## The proposed pathway (critique + reorder + tell us what's missing)
1. **Data = the ceiling.** Large multi-source pretraining corpus (NIS3D + Cell Tracking Challenge Fluo-N3DL
   developmental + Zebrahub + more) → detector that has seen dozens of embryos, not 2. OOF 0.67→~0.75.
2. **Best-in-class linking** (where 0.88 edge-J is won): tuned ILP + stronger learned edge model + gap-closing
   + motion priors + detection-precision cleanup. OOF →~0.85.
3. **Divisions as a weapon** → division-J 0.5-0.6 = +0.05-0.06. OOF →~0.90.
4. **Ensemble + TTA + count-calibration (exploit the one-sided penalty) + sub-voxel refine + free
   unmatched-edge quirk.** OOF →~0.92-0.94.
5. **Orthogonal edge** (your job).

## What I want from you
- **Is the 0.94 arithmetic realistic on a DISJOINT embryo?** What's the honest private-board ceiling, and
  what's the single biggest reason edge-J might NOT reach 0.88?
- **Where does edge-J 0.70→0.88 actually come from?** Linking architecture (global ILP vs learned vs
  transformer), gap recovery, motion, detection precision — rank them.
- **Divisions:** cheapest reliable path to division-J 0.5+ on a disjoint embryo (learned classifier vs ILP
  division-cost vs motion/appearance cues). Quantify FP risk under the metric.
- **THE ORTHOGONAL EDGE:** sweep cross-domain tracking (MOT: MOTR/ByteTrack/OC-SORT/GHOST; particle-physics
  track-finding / GNN / exa.trkx; optimal transport / graph matching for association; SLAM data-association;
  RAFT/scene-flow motion priors), metric/submission-format exploits (read tracksdata + tracking_cellmot
  adversarially for LEGAL degenerate wins), data-leverage edges, and analogous winning Kaggle bio-comp
  solutions (Sartorius, HuBMAP, cell-instance-seg). What transfers to 3D+t sparse cell association and beats
  the field?
- **The ONE thing we are structurally blind to.** End with the single bet you'd make that could be decisive.

## Where to look (repo, self-contained)
reports/WIN_PLAN.md; reports/journal/JOURNAL.md; HANDOFF.md; vendor/kaggle-cell-tracking/ (tracking_cellmot +
metrics.md + predict/train scripts); src/biotrack/; reports/research/gap_*.md;
reports/codex_lateral_sweep_results_2026-07-03.md (prior orthogonal-edge sweep — dedupe against it).
