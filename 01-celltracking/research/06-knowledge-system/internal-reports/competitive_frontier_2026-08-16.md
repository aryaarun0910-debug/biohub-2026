# Competitive frontier sweep (agent report) — 2026-08-16

Live via authenticated Kaggle CLI/API (leaderboard, 72 discussion threads, public notebooks, rules).
Evidence store (outside Git). Full LB CSV + pulled notebooks in the session scratchpad.

## Leaderboard (2426 teams; notebook-only; $60k)
Top-3 boundary = **0.948**. Leader Mark Cooper 0.950; TWEAK 0.949; Soheil Ayati 0.948; then 0.933-0.947 to rank 20.
Distribution: >=0.950:1, >=0.948:3, >=0.945:7, >=0.940:10, >=0.935:16, >=0.930:27 (top 1.1%), >=0.920:73.
Then a CLIFF: **0.915 = 302 teams**, >=0.915 = 447 teams (top 18.4%) = the PUBLIC-NOTEBOOK PLATEAU. **We (0.915) sit here,
tied with ~300 copy-of-public teams, ~rank 447.** Gap to top-3 = **+0.033**. Frontier actively rising 08-15/16.
Signatures: z7777 0.945 in 7 subs / Seine 0.930 in 1 sub = strong external/pretrained asset, minimal LB probing.

## The public plateau is real and exhausted at ~0.911-0.916
Host baseline = TemporalUNet3D detector + SimpleNodeTransformer linker + lineage ILP (royerlab tracking_cellmot).
Best public notebook ~0.915-0.916 (Togashi "no-hack"). Whole field runs ONE shared engine: pilkwang
50-epoch weights (8,827 downloads) + 2-seed detector blend + 22-feature UNET300 ranker + edge-TTA + ILP +
motion relink + gap repair + harmonic bidirectional fusion (= our "P3 harmonic"). Community-measured:
single-seed 0.908 -> 2-seed 0.910-0.911; ILP probes 0.908; edge-TTA HURTS (0.885). Post-processing EXHAUSTED
~0.911-0.916 (independently reproduces our "cheap levers closed"). Metric hack (-10000 hub fork) PATCHED
+ re-scored ~2026-07-20; today's 0.93-0.95 is legitimate.

## The undisclosed 0.93-0.950 edge (discussions collapse the mystery)
adj_edge_jaccard plateaus ~0.90-0.91 on the public stack, division_J ~0 for nearly everyone -> the separation
is the EDGE TERM = a genuinely better/RETRAINED model. Data facts: only 2 physical embryos, 199 chunks, ~2.8%
nuclei annotated, ~304 division events total -> it is a DATA/GENERALIZATION problem, not post-processing.

- **H1 (STRONGEST) Retrain detector/associator on external ZEBRAHUB data.** Host EXPLICITLY OPENED the full
  Zebrahub corpus (imaging + *_tracks.csv dense Ultrack lineages) on **2026-08-13**: "free to use... no overlap
  with the test set" (#734330). Same-domain light-sheet zebrafish. Only days old -> under-exploited. Tang (top-15):
  "retrain the model instead of the public ckpt... use external/synthetic data." Kaggle artifacts already appearing:
  Spotiflow variant, DeepCenter epoch-400 (vs public 50ep), fine-tuned cellmot detector.
- **H2 (STRONG, = our known gap) Learned detector to KILL the FP candidate pool.** Failure is PRECISION not recall:
  "detector finds real cells but produces an enormous junk candidate pool; linker chooses wrong" (#734604). DoG
  recovers 0.91-0.94 of nuclei already. 2-seed blend helped ONLY by cutting nodes ~1.7%; adding candidates HURTS.
  => learned FP-suppressing ranker = our "re-acceptance must be a ranker".
- **H3 (STRONG) Dense pseudo-labels / synthetic tracks.** GM hengck23 advocates dense-track + affinity-flow-field
  training. José Freitas released 18.5GB CC0 synthetic 3D microscopy, **165k labeled divisions (~540x real)**,
  pooling-matched to the evaluator's vol[:, ::4, ::4] stride (#732103).
- **H4 (MODERATE) Division recovery** +~0.02 (mikelou1 claims div_J ~0.2), but 304 events -> SHAKEUP-PRONE.
- **H5 pretrained-plugin (Trackastra/CoTracker) — likely a DETOUR unless retrained.** Community tested: CoTracker
  loses morphology through divisions; **Arul tested the host's HOCT division-tracker and it UNDERPERFORMED a tuned
  ILP** (#728551). => HOCT is NOT the edge as a drop-in.
- H6 longer training (~400ep vs public 50ep) = hygiene, not sufficient; diminishing returns past ep400.

## Rules (verified)
Deadline 2026-09-29; team-merge + final-entry 2026-09-22; 5 subs/day; team<=5; notebook-only internet-off rerun.
External data ALLOWED incl. Zebrahub imaging + tracks (host-confirmed 2026-08-13, #734330). Pretrained/self-trained
models ALLOWED (attach as dataset, reproducible if win). Verbatim rules legal text not directly quoted (JS-gated).

## Named competitors
hengck23 (GM) = dense pseudo-label + affinity flow, "lineage decides the winner". TWEAK (rank2, 164 subs) = "universal
bio-cell plugin" AI-agent company. Felipe Kitamura (rank13) = radiology-AI GM, detector-transfer player. Tang/mikelou1
= candid: retrain + external/synthetic + modest division. Hosts = Royer lab (Ultrack) -> intended solution is a
well-trained detector feeding ILP/Ultrack.

## Bottom line
The edge is a RETRAINED/GENERALIZING edge model enabled by EXTERNAL same-domain data unlocked only 2026-08-13
(Zebrahub) + synthetic division supervision, attacking the learned-FP-suppression axis our own state identified.
Highest-EV, cheapest-to-falsify: **H1+H2 on leave-one-embryo-out CV with the PATCHED scorer, before any GPU submission.**
Division (H4) = small secondary, shakeup-prone. Off-the-shelf pretrained linkers = detour unless retrained.

## KEY TIMING INSIGHT (added in synthesis)
We CLOSED the project 2026-08-07. External Zebrahub data was unlocked 2026-08-13 -- SIX DAYS AFTER we closed.
The single biggest lever the top tier now rides did not exist as a legal option when we quit. The reopen is well-timed.
