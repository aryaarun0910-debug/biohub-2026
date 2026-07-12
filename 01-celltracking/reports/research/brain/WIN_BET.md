# WIN_BET — the generalization lever (swing-for-the-win track)

**Date:** 2026-07-12. Runs in parallel with E0. Authoritative alongside
`SYNTHESIS.md`. This is the bet that can actually win, not the incremental
count/consistency hygiene.

## Thesis (why this wins and pruning doesn't)

The private board is a disjoint-embryo **generalization** contest where ~1000 teams
forked ONE 0.889 baseline trained on the **two** competition families. Everything
overfits for the same reason: two embryos is not enough diversity to learn
embryo-agnostic association. Fusion won 2-embryo OOF and lost hidden for exactly
this reason. **The only structural fix is training breadth.** Whoever trains an
association model that generalizes across embryo density/velocity/intensity regimes
degrades least on the hidden embryo — and wins. Count-pruning buys ~+0.005–0.01 of
hygiene; it does not change the transfer gap.

## The differentiated asset: the breadth lake (acquired; eligibility still gated)

Dense, multi-embryo, same-modality (DAXI zebrafish, same voxel scale) lineage
supervision that no public fork uses. Downloading to `data/external/zebrahub/`
(gitignored) with SHA256 provenance:

| Source | Rows/size | Schema | Use |
|---|---|---|---|
| ZSNS001_tracks.csv | 890 MB | `track_id,t,z,y,x,parent_track_id` (µm) | dense embryo 3 |
| ZSNS003_tracks.csv | 209 MB | same | dense embryo 4 |
| ZSNS004_tracks.csv | 378 MB | same | dense embryo 5 |
| ZSNS005_tracks.csv | 341 MB | same | dense embryo 6 |
| tracks_benchmark/2024_03_14_daxi_tracks.zarr | (later) | GEFF-like | March dense set |
| competition 44b6 / 6bba sparse GT | local | GEFF | held-out targets |

`parent_track_id = -1` marks a birth; a `track_id` that is the parent of ≥2 child
tracks marks a division. ZSNS002 is a 9-byte placeholder — skip. Net: **4 dense
embryos + March set → 3× the embryo diversity**, dense not sparse.

Provenance gate: every file needs URL + SHA256 + license before it feeds a
prize-submission model. Source host `public.czbiohub.org/royerlab/zebrahub`
(Royer lab / CZ Biohub public release) — confirm the exact license text before the
final submission uses a model trained on it (Tier-3 eligibility open question in 01).

## The model: a transfer-first association scorer (NOT appearance, NOT from-scratch 4D)

Score candidate frame-to-frame edges with features that are **scale-free by
construction**, so the model cannot key on any one embryo's density/velocity scale
(appearance is the least-transferable axis — we deliberately avoid it):

- displacement normalized by **local median neighbor displacement** (velocity-invariant);
- candidate distance **rank among kNN** alternatives (density-invariant);
- forward/backward (cycle) consistency of the link;
- **local tissue-flow residual** — motion relative to the smoothed neighbor velocity field;
- kNN-geometry preservation — does the local neighborhood survive the link;
- competition_baseline `edge_prob`/`edge_dist` (already in the wrapper GEFFs) as one input.

Label = true lineage edge (from dense tracks). **Train across ALL embryos at once**
(4 Zebrahub + one competition family), so embryo-specific nuisance scale averages
out. Model = gradient-boosted trees first (low-variance, fast, CPU, lane-04
recommended); GNN/MLP only if GBDT saturates.

## Deployment = selective repair over the 0.889 wrapper

Never swap the association graph wholesale (that already failed twice). The transfer
scorer overrides a baseline edge ONLY where: baseline confidence low AND transfer
score high AND forward/backward consistent AND no lineage/collision constraint
violated AND node-count neutral-or-better. Worst case = fall back to the wrapper.
This unifies the red-team's safe posture with the generalization gain.

## Validation = honest reverse-fold (the gate that catches the false positive)

Train on {ZSNS001,003,004,005 + 6bba}, hide **44b6**, run the full wrapper + repair,
score 44b6. Then reverse (hide 6bba). Because external embryos are in training, this
directly tests whether breadth closes the hidden-embryo gap.
**Promotion gate:** min-fold adj-J ≥ +0.005 vs the E0 wrapper baseline, both folds
positive, no regime-slice regression. Pre-registered; ≤2 confirmatory tests/round.

## Milestone 0 — the CHEAP kill-gate first (CPU, ~2–3 days, no GPU)

Before building the repair/deployment machinery, falsify the thesis cheaply:

> Can an association scorer trained on the 4 external dense embryos + one competition
> family rank true-vs-false candidate edges on the **held-out** competition embryo
> better than the wrapper's own `edge_prob`?

External pre-gate status: training on ZSNS003/004/005 and testing on held-out
ZSNS001 produced pooled ROC-AUC 0.9967 and hard-subset ROC-AUC 0.769 versus 0.754
for normalized distance. This supports transfer, but **does not pass Milestone 0**:
pooled AUC is dominated by easy negatives, and the hard AUC pools candidates across
sources. The decision-relevant external checks are candidate recall plus per-source
top-1/MRR; the decisive gate remains the held-out competition-embryo comparison.

Stricter two-frame audit (added after red-team review): candidate recall@6 was
0.9596; overall per-source top-1 was 0.9880 versus 0.9860 for nearest-neighbour.
For sources whose truth was not the nearest candidate, model top-1 was 0.2935
versus 0.0000 and MRR was 0.5807 versus 0.4297. But pooled hard AUC reversed to
0.7576 versus 0.7965. This is **mixed, preliminary evidence**: the model can repair
some ambiguous links, but the result must be reproduced over a larger window and
then on competition OOF before promotion.

Steps: (1) parse the 4 Zebrahub CSVs into the unified µm schema; (2) build
positive/negative candidate edges (true lineage edges vs. plausible-but-wrong
kNN alternatives) with the scale-free features; (3) train GBDT on external + 6bba;
(4) measure **edge-classification AUC / PR on held-out 44b6 candidate edges**, and
whether it beats the baseline `edge_prob` ranking; reverse.

**KILL GATE:** if the external-trained scorer cannot beat the wrapper's own edge
ranking on the held-out competition embryo (no AUC lift on either fold), the breadth
thesis is dead — stop, do not build deployment. If it lifts, proceed to the selective
repair + reverse-fold score gate. This is the falsification the plan owes: it costs
days, not GPU-weeks, and it decides the whole bet.

## Sequencing / compute
- Milestone 0 is 100% CPU-local (feature extraction + GBDT). No Kaggle GPU.
- Runs in parallel with E0 (E0 = the baseline the score gate measures against).
- Only if Milestone 0 passes do we spend GPU on a learned feature branch or a
  breadth-pretrained detector (Tier-3), gated separately.

## Open items
- Confirm Zebrahub license text for prize eligibility (blocks final submission use, not R&D).
- Align Zebrahub coordinate convention/scale to the competition `(1.625,0.40625,0.40625)`
  µm — verify axis order and origin before mixing embryos.
- Export the wrapper's per-crop candidate edges + `edge_prob` for the AUC comparison
  (couples to E0's exact-replay output).
