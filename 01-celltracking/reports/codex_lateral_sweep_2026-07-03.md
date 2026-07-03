# Codex LATERAL SWEEP — deploy a research team; find edges the whole field will miss

Mission: find high-EV, ORTHOGONAL edges nobody in this competition is using. Spin up a large
parallel research team (one agent per track below + a synthesis agent) and sweep EVERYTHING —
papers, preprints, blogs, docs, repos, issues, forum/Discord/Reddit threads, talks, theses,
patents, tweets, YouTube, "silly little things." We are NOT looking for more competitive intel
(that is saturated). We are looking for the non-obvious lever.

## HARD FILTER (every agent obeys)
- **DO NOT REPEAT what we already have** (below). If a finding restates known material, drop it.
- Return ONLY items that are NEW + ACTIONABLE + high-EV, each with: the edge, why it's orthogonal,
  a cited source URL, offline-Kaggle feasibility (notebook-only, 12h, internet OFF at submission),
  and the first experiment to test it. Rank by EV x novelty x feasibility.
- Legal only: no DQ moves (no private-label reconstruction via submissions, no ToS breaches).
- Flag anything unverifiable.

## ALREADY COVERED (saturated — do not re-surface)
Metric = weighted adj_edge_Jaccard + 0.1 div_Jaccard; per-timepoint optimal bipartite match at 7um on
1/(1+d); count penalty vs estimated_number_of_nodes; assignment-stealing; V3 DoG recipe (0.842);
our anchor 0.807; 77-80% of misses are no-candidate; DoG recall ceiling ~0.90; Spotiflow-3D + Linajea
mask; PAC-MAP; Ultrack/motile/Trackastra; DAXI/unet-simview weights; ZSNS embryos; host U-Net+transformer
internals; 2-embryo trap; count-band 0.95-1.05; sparse-division break-even. Data = 199 crops / 2 embryos,
tensorstore on Kaggle (no zarr). See reports/research/, reports/experiment_plan.md, reports/SPRINT_2026-07-03.md.

## RESEARCH TEAM — one agent per track (run in parallel)

**A. Cross-domain association / tracking.** The hard problem is association under a MATCH-FIRST sparse
metric. Sweep fields bioimaging never touches: multi-object tracking (MOTR, TrackFormer, ByteTrack,
OC-SORT, BoT-SORT, GHOST, QDTrack, joint-detection-embedding), particle-physics tracking (CERN ACTS, GNN
track-finding, exa.trkx), point-cloud registration & scene flow, SLAM loop-closure/data-association,
optical flow (RAFT) as motion prior, graph matching / optimal transport for linking. What transfers to
3D+t sparse cell association and beats Hungarian/ILP?

**B. Metric & submission-format exploits.** Read tracksdata + tracking_cellmot (metrics.py,
division_metrics.py, _matching.py) adversarially for LEGAL degenerate exploits: submission-parsing edge
cases, edge-validity loopholes, division-TP structural tricks, how ties/duplicates/ordering are handled,
whether node id / coordinate encoding can be gamed. Quantify each with the exact formula.

**C. Data provenance & leakage.** Can the hidden-test embryo be fingerprinted to a PUBLIC volume
(Zebrahub ZSNS / DAXI / the exact-scale 522-frame Ultrack embryo)? Any signal in zarr attrs,
estimated_number_of_nodes, file/crop naming, chunking, image_statistics that competitors ignore?
What EXACTLY do the rules permit re: public external data / source identification (quote verbatim)?

**D. Newest sparse/PU 3D detection (2024-2026).** Beyond Spotiflow/PAC-MAP: new sparse-point / PU /
self-supervised / foundation detectors usable offline — SAM-3D/SAM2, Cellpose-SAM, uSAM, CellViT,
diffusion-based detection, DINOv2/feature-based point detection, any zebrafish-nuclei-specific model.
Which genuinely lifts recall on anisotropic light-sheet and ships offline.

**E. Organizer full footprint.** Every Royer Lab / CZ Biohub artifact: all repos (issues, PRs, commits,
branches, wikis), papers + supplements, theses, talks/slides, tweets, blog posts, the tracksdata/geff/
ultrack/zebrahub docs. Hunt for undisclosed hints about the data, the metric's intent, or the "intended"
winning approach. The silly things count.

**F. Kaggle meta + analogous comps.** Every discussion post/notebook/competitor profile here (accidental
leakage?); host comments; and WINNING-SOLUTION writeups from analogous past Kaggle comps (cell instance
seg, HuBMAP, Sartorius, tracking, 3D bio) — recurring winning patterns that transfer.

**G. Label-efficiency / domain generalization (the 2-embryo trap).** Newest semi/self-supervised,
pseudo-labeling, teacher-student, consistency, test-time adaptation, domain-generalization, style-transfer
augmentation methods to generalize a detector from 2 embryos to a disjoint one without overfitting.

**H. Wildcards.** image.sc, Discord, Reddit, YouTube, preprints, patents, blogs mentioning this data /
DAXI / Zebrahub / tracksdata / this competition. Any obscure trick, any accidental disclosure.

## SYNTHESIS AGENT (final)
Dedupe against ALREADY COVERED; merge all tracks; output a RANKED list of orthogonal edges
(EV x novelty x feasibility), each with the first experiment + offline-Kaggle feasibility + legality.
End with: the ONE non-obvious bet you'd make that the entire field will miss, and why — plus which 2-3
edges are worth interrupting execution for vs. which are "later."
