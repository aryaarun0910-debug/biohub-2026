# Gap research: cell tracking / linking + division detection (2026-07-03)

Competition: Biohub "Cell Tracking During Development" — 3D+t light-sheet zebrafish, detected
nuclei given, LINK across time + detect DIVISIONS. Metric = weighted adjusted edge-Jaccard +
0.1 * division-Jaccard. Constraints: inference notebook, internet OFF, <=12h, T4x2, open-license
solvers only (pyscipopt/SCIP ok; NOT Gurobi).

Our state: greedy/velocity-aware NN linker; divisions OFF (0% of the 0.1 term captured).

---

## 1. Bottom line (read this first)

- **Divisions are free money and we take none.** 0.1 of the metric sits untouched. A pure
  post-hoc division heuristic on top of our existing greedy tracks (no new model, no retrain)
  can plausibly capture a meaningful fraction of that 0.1 in a day of work. **Do this first.**
- **The single best "SOTA that fits our constraints" is `motile`** (Funke lab): a combinatorial
  ILP tracker that takes a *candidate graph* (our detections + candidate edges) as input, models
  appearance/disappearance/**division**, needs **no training**, is pip-installable, offline, and
  runs on **SCIP** through `ilpy` when no Gurobi license is present. It is essentially "our greedy
  linker, but globally optimal, and division-aware." This is the highest-leverage upgrade and it
  is directly SCIP-compatible.
- **Ultrack** is the strongest published method on exactly our data (zebrafish light-sheet,
  Nature Methods 2025) and is also ILP-based, division-aware, BSD-3, no-training-required — but
  it wants to build its own segmentation hypotheses from foreground+contour maps, so it is a
  heavier integration than motile if all we have is point detections.
- **Learned linkers (Trackastra, EmbedTrack, ELEPHANT, Linajea)** are all division-aware but each
  has a friction point for us: 3D pretrained weights are not clearly published (Trackastra,
  EmbedTrack), require a server/annotation loop (ELEPHANT), or were built around Gurobi + training
  (Linajea). Usable, but not the cheapest win.

---

## 2. Linker comparison table

| Method | Division-aware lineage? | Offline / Kaggle-feasible? | License | Needs training? | Solver | Notes for us |
|---|---|---|---|---|---|---|
| **motile** (funkelab) | **Yes** — `MaxChildren(2)` + Split/Appear/Disappear costs | **Yes**, pip, pure-python graph in | MIT (open) | **No** (combinatorial) | **ilpy → SCIP fallback** when no Gurobi | Takes a networkx candidate graph of our detections + candidate edges. Drop-in global replacement for greedy. **Top pick.** |
| **Ultrack** (royerlab) | **Yes** — models division/death/exit in ILP | Yes, pip, offline, BSD-3 | BSD-3-Clause | **No** (multi-hypothesis combinatorial) | Gurobi optional; open MIP fallback | Nature Methods 2025, validated on zebrafish/fly/nematode light-sheet. Wants foreground+contour maps to build seg hypotheses; more integration if we only have points. |
| **Trackastra** (weigertlab) | **Yes** — blockwise parental softmax; `greedy` or `ilp` mode | Yes if 3D weights ship; pretrained runs offline | BSD-3-Clause | Pretrained exists (**2D confirmed; 3D weights unclear**) | ilp mode reuses motile → SCIP fallback | Transformer learns pairwise assoc from image+seg crops. Input = images + instance segs. 3D "promised to scale" but public 3D checkpoint not confirmed. |
| **EmbedTrack** | Yes (offset + clustering, CTC entrant) | Heavy — trains per-dataset CNN | open (research) | **Yes**, per-dataset | graph matching | Mostly 2D CTC; 3D not its strength. Skip for Kaggle. |
| **ELEPHANT** | Yes (incremental DL, 3D nuclei) | **No** — client-server Fiji/Mastodon, annotation loop | open | **Yes** (incremental) | — | Great tool, wrong shape for an offline inference notebook. |
| **Linajea** (funkelab) | **Yes** — learned division detector + ILP, beats CTC on Fluo-N3DH-CE | Offline possible but built around **Gurobi** + trained U-Net | open (research) | **Yes** (U-Net + SSVM weights) | ILP (Gurobi) | Best-in-class *divisions* on developmental embryos, but needs training + was Gurobi-oriented. Mine it for ideas, not as a drop-in. |
| **TrackMate/u-track LAP** | Yes — splitting events in 2nd cost matrix | Java/Fiji; concept portable | open | No | LAP (Hungarian) | Jaqaman two-step LAP: frame-to-frame, then gap-closing/merge/**split** matrix. Good cheap blueprint for a division-aware assignment cost. |

---

## 3. Cheapest high-yield plan to capture part of the 0.1 division term

Divisions are scored by a **division-Jaccard**: we get credit for correctly predicting which
mother→two-daughter events happen. We currently predict **none**, so recall = 0 and this whole
term = 0. Any correct division we emit is pure upside; false divisions only hurt the division term
(weighted 0.1) and, mildly, the edge term.

**Plan A — post-hoc division heuristic on existing greedy tracks (1 day, no model):**

1. Run our current greedy linker as-is to get tracks.
2. Detect division candidates: for each track that **terminates** at frame t (no assigned
   successor), look in frame t+1 for **two unmatched detections** both within the ~7µm matching
   gate (in µm using the anisotropic voxel size z,y,x = 1.625, 0.40625, 0.40625). Classic
   1-to-2 NN test (this is exactly the u-track "split" cost and the generalized-NN 1→2 search).
3. Score each candidate cheaply with handcrafted cues that need no training:
   - **symmetry**: the two daughters roughly equidistant from the mother (|d1 - d2| small);
   - **proximity**: daughters close to each other and straddling the mother position;
   - **count/appearance**: mother nucleus size/intensity drop, or two new detections appearing
     where one track ended (telophase = two similar high-intensity blobs).
4. Accept a division when the combined score passes a threshold **tuned on the training set to
   maximize division-Jaccard** (precision/recall trade against the 0.1 weight). Attach the two
   daughters as children (they inherit the two child edges).

This reuses the detections we already trust and adds two edges + one division event per accepted
call. Because the metric weights divisions at 0.1 and we start at 0, even modest recall at
moderate precision is net-positive. Tune the threshold on the provided training lineages.

**Plan B — let a global solver emit divisions for free:** switch the linker to **motile** (see §4)
with `MaxChildren(2)` and a **Split/division cost**. The ILP then decides divisions jointly with
linking under a global objective, so we do not hand-tune a separate detector; we tune one division
cost weight. This is strictly better than Plan A if we are already paying to adopt motile, and it
also improves the edge term. Plan A is the zero-risk floor; Plan B is the real answer.

**Guardrail:** cap children at 2, forbid a detection being a daughter of two mothers (MaxParents=1),
and forbid divisions on consecutive frames for the same lineage — these are the standard Linajea
"consistency/split" constraints and they stop runaway false divisions that would erode the edge term.

---

## 4. Is replacing the greedy linker worth it, and how?

**Yes — with motile, and it is the highest-leverage single change.** Rationale:

- Greedy NN is locally optimal and myopic; it makes irreversible frame-by-frame choices under
  detection noise and anisotropy. A **global min-cost formulation** (min-cost flow / ILP) resolves
  swaps, gaps, and divisions jointly and is the consensus SOTA backbone (Ultrack, Linajea,
  Trackastra-ilp all reduce to an ILP over a candidate graph).
- motile is the version of that we can actually run: **no training, pure-python, offline, MIT,
  and SCIP-backed via ilpy** (motile explicitly falls back to SCIP when no Gurobi license is
  found — that is exactly our situation). SCIP/pyscipopt is on the allowed list.

**How to wire it (SCIP-compatible, offline):**

1. Build a **candidate graph**: nodes = the given detections (with z,y,x in µm); candidate edges =
   each detection to every detection in t+1 within the ~7µm gate (optionally velocity-gated using
   our existing velocity estimate to prune edges and keep the ILP small).
2. Costs: `NodeSelection` (favor keeping real detections), `EdgeSelection` with cost =
   f(anisotropic distance, velocity agreement) so shorter/consistent links are cheaper,
   `Appear`/`Disappear` costs to penalize track birth/death, and a **Split (division) cost**.
3. Constraints: `MaxParents(1)`, `MaxChildren(2)` (2 = divisions allowed).
4. Solve with ilpy on **SCIP**. Scale: solve **per spatial tile or per time-window** (sliding
   window, stitch overlaps) to keep each ILP tractable within the 12h / T4x2 budget — this is the
   standard trick Ultrack uses for terabyte volumes. GPU is not needed for the solve.

**Risk / fallback ladder (cheapest-first):**
1. Keep greedy linker, add **Plan A division heuristic** (guaranteed small win on the 0.1 term).
2. Swap edge assignment to a **Jaqaman/u-track two-step LAP** (Hungarian) with a **split matrix** —
   gets global-per-frame optimality + divisions without an ILP, if motile integration slips.
3. Full **motile ILP** with division cost (best expected score; SCIP-backed).
4. Only if time remains and images are readily loadable: try **Trackastra** pretrained (verify a
   3D checkpoint exists first) or **Ultrack** from foreground+contour maps.

---

## 5. Robust linking under detection noise / anisotropy (applies to all options)

- **Always compute distances in physical µm** using voxel (z,y,x)=(1.625, 0.40625, 0.40625).
  z is ~4x coarser — an isotropic-voxel gate is wrong and will over/under-link along z. The ~7µm
  gate must be applied in µm space.
- **Velocity-aware gating** (we have it) as an edge prior/prune, not a hard rule — predict next
  position, cost on residual. Helps swaps in dense regions.
- **Allow gap-closing** (skip-one-frame edges) for missed detections — cheap recall on the edge
  term; standard in LAP/ILP formulations.
- **Divisions must be gated tighter than normal links** (daughters are close + symmetric) to avoid
  converting dense-cluster confusion into false divisions.

---

## Sources

- Ultrack (Nature Methods 2025): https://www.nature.com/articles/s41592-025-02778-0 ,
  preprint https://arxiv.org/pdf/2308.04526 , code https://github.com/royerlab/ultrack
- Trackastra (ECCV 2024): https://arxiv.org/html/2405.15700v1 ,
  code https://github.com/weigertlab/trackastra
- motile / ilpy (Funke lab, ILP over candidate graph, SCIP fallback):
  https://funkelab.github.io/motile/ , https://github.com/funkelab/ilpy ,
  https://pypi.org/project/motile/ , https://github.com/funkelab/motile_tracker
- Linajea — whole-embryo lineages, learned division detector + ILP (beats CTC on Fluo-N3DH-CE):
  https://www.biorxiv.org/content/10.1101/2021.07.28.454016v1.full ,
  https://arxiv.org/pdf/2208.11467
- ELEPHANT (incremental DL 3D tracking, eLife 2022): https://elifesciences.org/articles/69380
- EmbedTrack: https://www.researchgate.net/publication/360164254
- TrackMate / u-track LAP (Jaqaman split/gap-closing cost matrices):
  https://imagej.net/plugins/trackmate/trackers/lap-trackers ,
  LapTrack https://academic.oup.com/bioinformatics/article/39/1/btac799/6887138
- Cheap handcrafted mitosis cues: https://www.nature.com/articles/s41598-025-87180-8 ,
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4845473/
- Cell Tracking Challenge datasets / benchmark: https://celltrackingchallenge.net/3d-datasets/ ,
  https://www.nature.com/articles/s41592-023-01879-y
