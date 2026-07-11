# Division (mitosis) prediction lane — research + ranked plan (2026-07-06)

Competition: Biohub "Cell Tracking During Development" — 3D+t light-sheet zebrafish nuclei.
Metric = weighted adjusted edge-Jaccard + **0.1 * division-Jaccard**, scored on a DISJOINT hidden
embryo. Division = one parent node at t with TWO outgoing edges to daughters at t+1. Inputs are
POINT detections (no masks). Kaggle T4x2, internet-off, <=12h, open solvers only (SCIP ok).

Current state: 0% of the 0.1 term captured. 151 labeled divisions, severe embryo imbalance
(26 in 44b6 vs 125 in 6bba). A single global threshold is unsafe. Simple GEOMETRIC proposals
FAILED a preliminary cross-embryo test. D = r / [1 + r(1-p)/p]; D>=0.6 needs ~80% recall @ 70.6%
precision. Realistic private target D = 0.25-0.50 (+0.025-0.05 score).

Evidence tags: **[V]** verified from a primary paper/repo I read; **[C]** claim reported by a
source; **[I]** my inference for this competition.

---

## 1. Bottom line (read first)

1. **Predict divisions AFTER association, as an explicit fork POSTERIOR gate — not by lowering a
   global ILP division cost.** [I] Rationale below (§4). It decouples division EV from edge-J risk,
   permits per-embryo calibration, and by construction cannot remove a correct continuation edge.
2. **The one cheapest credible path to D>=0.25 with edge-J degradation <=0.005** is a two-stage
   proposal+classifier that only ADDS a second daughter edge on top of the frozen best linking
   (§6). This mirrors the dominant, repeatedly-winning design in the mitosis-detection literature:
   high-recall candidate generator → learned classifier to kill false positives [C]
   (YOLO11x→ConvNeXt two-stage, arXiv 2509.02627; "birth-event detection + local tracking",
   Huh et al.).
3. **Why simple geometric proposals failed cross-embryo, and what fixes it:** geometry thresholds
   implicitly encode the local length/velocity/DENSITY scale and the division RATE, and both are
   strongly stage-dependent in zebrafish (progressive metasynchrony → desynchronization by cycle
   ~10) [C, Mendieta-Serrano 2013; bioRxiv 2025.02.03.636134]. The features that TRANSFER are
   scale-free/physics-based ratios, not absolute geometry (§3).

---

## 2. Ranked method table

EV(+score) is my expected contribution to the 0.1*division-Jaccard term at a realistic private
operating point, net of edge-J risk. All are usable from POINT detections + the raw image
(patch crops around each point) unless noted.

| # | Method | Mechanism | Mask-free? | EV (+score) | Effort | Few-label / imbalance robustness | Cross-embryo risk | Evidence | Repo | First falsification experiment |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Two-stage POSTERIOR fork gate** (high-recall triplet proposals → GBDT/small-MLP classifier on mask-free ratio features → add 2nd daughter edge only) | Propose (parent, d1, d2) triplets at every base-track terminus / near-tie; score; accept only high-precision calls that add an edge without removing one | **Yes** (points + patch crops) | **+0.025 to 0.045** | **Low (1-2 d)** | **High** — GBDT on <=10 features; embryo-stratified LOEO calibration + class weights | **Low** — scale-free features + per-embryo isotonic threshold | [C] two-stage proposal+classifier is the field standard (2509.02627; Huh birth-detection); [I] the "add-only" edge-J guarantee | uses motile/Ultrack you already have | LOEO: train on 6bba, calibrate+test on 44b6 (and reverse). Report division-J and Δedge-J on the held-out embryo. Kill if Δedge-J < -0.005 or division-J < 0.15 |
| 2 | **motile / Ultrack division-native ILP with a LEARNED per-triplet division cost** (feed the §1 classifier score as the Split cost instead of a constant) | ILP jointly links + divides under MaxChildren=2/MaxParents=1; division cost = -log posterior from the fork classifier | **Yes** | +0.02 to 0.04 (higher ceiling, higher edge-J coupling) | Med (3-4 d) | Med — ILP weight fit needs care under 26-vs-125 imbalance | **Med** — global objective can trade edge-J for divisions | [V] motile SCIP fallback, MaxChildren; [V] Ultrack Nature Methods 2025 division-aware | funkelab/motile, royerlab/ultrack | Ablate constant vs learned division cost on LOEO; check edge-J does not regress vs the frozen linker |
| 3 | **Mask-free biological ratio features** (daughter-displacement symmetry, daughter separation, combined-intensity/"mass" conservation, mitotic-rounding/intensity spike, local-flow divergence) | Handcrafted, scale-free cues fed to methods 1/2 | **Yes** | (enables 1/2; standalone +0.01-0.02) | Low | High (few params) | **Low if ratios; High if absolute geometry** | [C] rounding + prophase intensity spike are near-universal (PMC7423972; Science adu9628); [C] mass/count cues (Huh) | n/a (feature code) | Rank features by LOEO permutation importance; drop any whose sign flips between embryos |
| 4 | **GNN-DOL — GNN + differentiable optimization layer enforcing mother-daughter constraint** | Edge-classification GNN whose DOL enforces "a node maps to <=2 nodes next frame", improving division recall over plain link-prediction | Yes (graph of points) | +0.02-0.04 if it trains | **High** | Med — needs enough division examples; 3D untested | **High** — published 2D only; must port to 3D + anisotropy | [V] Nat. Sci. Rep. 2023 (s41598-023-41562-y); code exists | 95-HaishanZHANG/GNN-DOL | Reproduce 2D result, then train 3D on 6bba, test 44b6; abort if it needs >125 divisions to converge |
| 5 | **Trackastra (division-aware transformer, ctc 3D weights)** | Transformer predicts pairwise + parental associations incl. divisions; tokens = morphology (intensity, area, inertia) + Fourier pos-enc | Partly — tokens want area/inertia (mask-derived); can degrade to point+patch tokens | +0.01-0.03 | Med | Low-Med — pretrained, but division is its weak spot on long/proliferative seqs | Med | [V] ECCV 2024 (2405.15700); [C] "performs well on short seqs, not large ones where division is frequent" | weigertlab/trackastra (ctc.zip 3D, offline) | Run pretrained ctc on our data; measure division recall vs method 1 with zero extra training |
| 6 | **Mitosis-aware multi-hypothesis tracker w/ aleatoric uncertainty (EmBAtT/Kaiser)** | MHT that scores division hypotheses with learned uncertainty; strong CTC mitosis metrics | Yes (detections) | +0.02-0.03 | High | Med | High — heavy integration, not point-native | [C] arXiv 2403.15011 v5 | (research code) | Skip unless 1-4 all stall; smoke-test on 44b6 only |
| 7 | **Track-age / cell-cycle-length prior as a division gate** | Forbid/penalize division unless track age ~ expected cycle length | Yes | **Negative-to-+0.01** | Low | Low | **HIGH — stage-specific, do NOT transfer** | [C] zebrafish cycles desynchronize with stage (2504.21818; 2025.02.03.636134) | n/a | Only as a soft refractory "no back-to-back divisions"; test that it does not suppress true divisions on LOEO |

---

## 3. Strongest mask-free features (and which transfer across embryos)

We have POINT detections but ALSO the raw light-sheet image, so we can crop nucleus-centered
patches around each point without any segmentation mask. Compute ALL distances in physical µm with
voxel (z,y,x) = (1.625, 0.40625, 0.40625) — z is ~4x coarser, so isotropic geometry is wrong.

**TRANSFER across embryos (scale-free / physics-based — USE THESE):** [I], grounded in [C]
- **Displacement symmetry** of the two daughters about the parent: cos-angle between (d1-parent)
  and (d2-parent) near -1, and the RATIO |d1-parent| / |d2-parent| near 1. Ratio/angle are
  scale-free, so they survive stage/density change. [C] daughters emerge near-symmetrically from
  the division plane (Science adu9628).
- **Combined nuclear "mass"/intensity conservation:** sum of patch-integrated intensity of d1+d2
  ≈ intensity of the parent patch just before split (a RATIO ~1). Conservation is embryo-agnostic.
  [C] mother→two similar high-intensity telophase blobs (Huh et al.).
- **Mitotic rounding + prophase/metaphase intensity spike** of the parent, measured as a local
  intensity/compactness Z-score within a short temporal window (self-normalized per track, so no
  global scale). [C] rounding is a near-universal animal-cell mitosis feature and a chromatin/DNA
  sensor peaks at M-phase (PMC7423972; Science adu9628).
- **Daughter-pair separation NORMALIZED by local nearest-neighbour spacing** (separation/median-NN
  distance), not raw µm. Normalization removes the density/stage dependence that sank raw geometry.
- **Deviation from local tissue flow (divergence):** a dividing site is a local source in the
  velocity field; measure divergence of the smoothed neighbour-flow around the candidate. Scale-free
  after normalizing by local speed. [I], motivated by tissue-flow/mechanics work
  (bioRxiv 2023.11.29.569235).

**STAGE-SPECIFIC (do NOT transfer as absolute thresholds — this is why geometric proposals failed):**
- Absolute division **rate** per frame, absolute cell-cycle **length**, mitotic **synchrony/wave**
  structure, absolute daughter separation in µm, absolute local density. Zebrafish divisions go
  from highly synchronous → metasynchronous → fully desynchronized by ~cycle 10, and rate is
  strongly stage/timing dependent [C] (Mendieta-Serrano 2013; bioRxiv 2025.02.03.636134). The two
  labeled embryos (26 vs 125 divisions) almost certainly sit at different stages/rates, so any
  feature carrying absolute scale over-/under-fires on the hidden embryo.

Design rule: **every feature fed to the classifier must be a ratio or a self-normalized Z-score.**
Then calibrate the decision THRESHOLD, not the features, per embryo.

---

## 4. Before or after association? Two-stage fork posterior vs lowering the ILP division cost

**Recommendation: AFTER, as an explicit fork posterior on a frozen high-quality linking.** [I]

- **Edge-J risk isolation.** Lowering a single global ILP division cost lets the solver re-route
  continuation edges into forks globally; that perturbs the adj-edge-Jaccard (0.9 of the metric) to
  chase the 0.1 term — a bad trade at these weights. A posterior gate that only ADDS a second
  daughter edge to an already-correct parent leaves the base edges untouched (edge-J neutral, often
  slightly positive because it adds a true edge).
- **Imbalance/calibration.** A global division cost cannot be embryo-stratified; the 26-vs-125
  imbalance means one constant either floods 44b6 or starves 6bba. A separate posterior lets us fit
  a per-embryo isotonic/Platt threshold (LOEO) — the only imbalance-safe option (§5).
- **Field precedent.** The consistently strong mitosis pipelines are explicitly two-stage:
  high-recall proposal → learned classifier posterior (2509.02627; Huh birth-detection), and the
  GNN-DOL result shows that adding an explicit mother-daughter CONSTRAINT on top of link-prediction
  beats folding it into a soft cost [V, s41598-023-41562-y].
- **Still keep the lineage constraints regardless of stage:** MaxChildren=2, MaxParents=1, and a
  short refractory "no back-to-back divisions in the same lineage." These are structural, not
  rate-based, so they transfer. Use motile/Ultrack to ENFORCE them; use the posterior to SCORE.

Superior ordering: frozen best linker → propose forks → score with mask-free ratio features →
accept high-precision calls → (optionally) hand accepted division costs back to the ILP so it
re-optimizes the local neighbourhood under the hard constraints.

---

## 5. Calibrating with few labels + 26-vs-125 imbalance

- **Embryo-stratified leave-one-embryo-out (LOEO) is the ONLY honest validation** here, because the
  test embryo is disjoint. Train the classifier on one embryo, fit the threshold + calibration on
  the other, and report division-J and Δedge-J on the held-out one. Any tuning that touches both
  embryos leaks the cross-embryo generalization you are trying to measure. [I]
- **Calibrate probability, then choose an operating point for PRECISION, not F1.** Isotonic or
  Platt on the held-out embryo's scores; pick the threshold at ~70% precision (recall follows) so
  D lands ~0.25-0.35 with bounded false-division damage. [I], target math from
  D = r/[1 + r(1-p)/p].
- **Handle imbalance with class weights / focal loss, not resampling that distorts geometry.**
  Oversample division examples in the loss only (Linajea's ">=25% division iterations" recipe from
  MEMORY) rather than duplicating points. [C]
- **Weak/synthetic division supervision from public dense trajectories** (Zebrahub ZSNS001-005
  lineage CSVs; Biohub public embryo — both same DAXI modality + exact voxel scale, per MEMORY):
  these have explicit lineage graphs, so you can mine THOUSANDS of real division triplets to
  pre-train the fork classifier, then fine-tune the threshold on the 151 competition labels. This
  is legal (public data, no test-label transfer) and directly attacks the few-label problem. [I]
  Guard: only transfer the scale-free features (§3); the public embryos are at their own stages.
- **Report a confidence interval:** with 26 divisions in 44b6, a single LOEO number is noisy —
  bootstrap the held-out divisions to get a CI before trusting an EV.

---

## 6. THE cheapest credible path to D>=0.25 with edge-J degradation <=0.005

**"Add-only" two-stage posterior fork gate (method #1). 1-2 days, reuses current linker.**

1. **Freeze** the current best linking (greedy/velocity/motile). Edge-J is now fixed.
2. **Propose** division candidates cheaply: for every parent node that the base tracker terminates,
   OR that has two near-tie outgoing edge candidates, form (parent, d1, d2) from the 2 nearest
   otherwise-unmatched detections at t+1 within a µm gate. High recall, cheap.
3. **Score** each triplet with a small GBDT on the mask-free RATIO features of §3 (symmetry angle,
   distance ratio, mass-conservation ratio, parent rounding/intensity Z-score, NN-normalized
   daughter separation, local-flow divergence).
4. **Pre-train** the GBDT on division triplets mined from Zebrahub/Biohub public lineage CSVs
   (thousands of examples), then **calibrate + threshold per-embryo via LOEO** on the 151 labels at
   ~70% precision.
5. **Accept** only high-precision calls, and add ONLY the second daughter edge (the parent's first
   daughter edge already exists from step 1). Because no existing edge is removed, adj-edge-J cannot
   drop by construction — the only edge-J effect is the added true/false edge, bounded well under
   0.005 at 70% precision. Enforce MaxChildren=2 / MaxParents=1 / refractory as a post-filter.

**First falsification experiment (do this before anything else):** LOEO both ways —
train+pretrain, calibrate on 6bba, test on 44b6, then swap. Success gate: division-J >= 0.25 AND
Δedge-J >= -0.005 on the held-out embryo, with a bootstrap CI that excludes 0.15. If it fails, the
mask-free ratio features are not transferring — re-rank features by cross-embryo permutation
importance and drop sign-flippers before adding model capacity (GNN-DOL, method #4).

---

## Sources

- Two-stage mitosis (proposal→classifier): https://arxiv.org/html/2509.02627v2 ;
  birth-event detection + local tracking (Huh et al., ScienceDirect mitosis-detection overview):
  https://www.sciencedirect.com/topics/computer-science/mitosis-detection
- GNN-DOL (differentiable optimization layer, mother-daughter constraint), Sci. Rep. 2023:
  https://www.nature.com/articles/s41598-023-41562-y ; code https://github.com/95-HaishanZHANG/GNN-DOL
- GNN edge-classification tracking (ECCV 2022, 2D+3D lineage):
  https://dl.acm.org/doi/10.1007/978-3-031-19803-8_36
- Trackastra (ECCV 2024, division-aware transformer): https://arxiv.org/html/2405.15700v1 ;
  code https://github.com/weigertlab/trackastra ; 3D ctc weights: github.com/weigertlab/trackastra-models
- Mitosis-aware multi-hypothesis tracker w/ aleatoric uncertainty: https://arxiv.org/html/2403.15011v5
- Ultrack (Nature Methods 2025, division-aware ILP, zebrafish light-sheet):
  https://www.nature.com/articles/s41592-025-02778-0 ; code https://github.com/royerlab/ultrack
- motile (ILP over candidate graph, MaxChildren, SCIP fallback): https://funkelab.github.io/motile/
- Mitotic rounding (near-universal): https://pmc.ncbi.nlm.nih.gov/articles/PMC7423972/
- Interphase morphology → mitosis mode/symmetry (Science 2024): https://www.science.org/doi/10.1126/science.adu9628
- Zebrafish division synchrony / stage dependence (why absolute geometry fails cross-embryo):
  https://anatomypubs.onlinelibrary.wiley.com/doi/10.1002/ar.22692 ;
  https://www.biorxiv.org/content/10.1101/2025.02.03.636134v2.full ;
  tissue mechanics / asynchrony: https://www.biorxiv.org/content/10.1101/2023.11.29.569235.full.pdf
- EmbedTrack (CTC seg+track): https://www.researchgate.net/publication/360164254
- Prior lane report (linker choice, motile/Ultrack wiring): reports/research/gap_tracking_division_2026-07-03.md
```
