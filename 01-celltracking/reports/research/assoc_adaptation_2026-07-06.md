# Association + Domain-Adaptation research lane (2026-07-06)

Competition: Biohub "Cell Tracking During Development" — 3D+t light-sheet zebrafish. Detections
~solved (95%/90% recall); **the dominant error is ASSOCIATION** (measured linking ~0.67 vs
detection ceiling ~0.885 → ~+0.18 of pure-linking headroom). Metric = weighted adjusted
edge-Jaccard + 0.1·division-Jaccard, on a **DISJOINT HIDDEN embryo** (the private board is the
prize). Constraints: T4x2, internet-OFF at submit, ≤12h, open licenses, no Gurobi.

**Scope of THIS lane:** generalizable association under embryo domain shift, and test-time
adaptation to the unlabeled hidden embryo. We assume a candidate-edge graph already exists
(detections + gated edges + organizer edge-head scores); we do **not** retrain the big detector.
The solver/ILP/division-mechanics comparison is covered in `gap_tracking_division_2026-07-03.md`
and is not repeated here — this report is about *what score each edge should get* and *how to make
that scoring survive the domain shift*.

Evidence tags: **[V]** verified from primary paper/repo; **[A]** author's own claim; **[I]** my
inference for our setting.

---

## 1. The one bet I'd fund

**Path-consistency test-time adaptation of the association head** (CVPR 2024, Lu et al.,
"Self-Supervised MOT with Path Consistency"), adapted to our SimpleNodeTransformer edge head and
run *on the unlabeled hidden embryo's own frames at inference time*.

Why this and not anything else:
- It is the **only** family that directly attacks our decisive lever — association that must
  generalize to a disjoint embryo — using **only unlabeled target images and no identity labels**.
  Its supervisory signal is exactly right for us: track an object t→t+k with *different* sets of
  skipped intermediate frames; every path must yield the **same** association. This needs no
  ground truth, no appearance identity, only motion/geometry — which is all we have. **[V]**
- It is **safe by construction.** Unlike TENT/DARTH (which minimize the model's own entropy /
  confidence and can collapse onto a wrong-but-confident domain), path/cycle consistency is a
  *relational invariant* the true tracking must satisfy regardless of domain. It cannot be gamed by
  becoming over-confident. **[V for the invariant; I for the safety-vs-entropy argument]**
- It reuses infrastructure we already have (a differentiable pairwise edge head + gated candidate
  graph) and the skipped-frame edges we should be building anyway for gap-closing.

Concrete first cut (falsifiable in a day, offline): compute forward–backward consistency of the
**existing** edge scores with *no* gradient step first (§3, method C). If the consistency-gated
edges already beat raw scores cross-embryo, the full learned TTA (a few gradient steps on the edge
head's affine/last layers, minimizing path-consistency loss over the target embryo) is very likely
to pay. If forward–backward gating does *nothing*, abandon the whole TTA branch — that is the kill
criterion.

**RULES FLAG (needs clearance before any submission uses it):** path-consistency TTA optimizes
model parameters on the *provided test images*. This is transductive/self-supervised (no test
labels touched), which most TTA papers treat as legitimate, but Kaggle host rules on "training on
test data" must be confirmed. The forward–backward *gating* variant (method C, no gradient step) is
almost certainly safe since it only post-processes our own predictions; the *learned* variant is
the one to clear. Design so the learned step is a toggle we can disable.

---

## 2. Ranked table

EV = expected private-board value for us. Effort = eng days to a testable prototype.
Gen-risk = risk it fails to transfer to the disjoint embryo. Evidence = strength that the *idea*
works (not that it works for us).

| # | Method | Mechanism (for OUR edge graph) | EV | Novelty | Effort | Cross-embryo gen-risk | Evidence | License | Repo / weights | First falsification experiment |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Path-Consistency TTA** (CVPR24) | Adapt edge head on target embryo; skipped-frame paths must give same association. No labels, motion-only. | **High** | High | Med (3–5d) | **Low** (invariant is domain-agnostic) | **[V]** strong, SOTA self-sup MOT | Check repo (MIT/Apache typ.) | arxiv 2404.05136; openaccess CVPR24 | Forward–backward gate (§3C) with 0 grad steps: does it beat raw scores cross-embryo on 44b6→6bba? |
| 2 | **Trackastra `ctc` + organizer agreement fusion** | Pretrained transformer linker scores edges from geometry+shallow feats; fuse w/ organizer head. Already **passed** bilateral cross-embryo test. | **High** | Low (in-stack) | Low (done) | **Low** — won CTC "generalizable linking" 2024 | **[V]** ISBI24 CTC winner, generalization is its thesis | BSD-3 | github.com/weigertlab/trackastra | Ablate fusion vs organizer-only on held-out embryo; if no lift, drop. |
| 3 | **Forward–backward / cycle-consistency edge gating** (no training) | Keep an edge iff the reverse-time match agrees; downweight non-cyclic edges. Pure post-proc on existing scores. | **High** | Med | **Very low (1d)** | **Very low** | **[V]** cycle-consistency is standard self-sup MOT signal | n/a (our code) | concept: Wang/Jabri cycle-time; QUB self-sup MOT | Compute FB-consistency on train pair; does removing non-cyclic edges raise edge-Jaccard? |
| 4 | **Optimal-transport association layer** (entropic + unbalanced/partial; 1→2 for division) | Replace greedy/argmax with Sinkhorn doubly-stochastic soft-assignment over the gated cost matrix; unbalanced slack rows/cols = births/deaths; duplicate a mother column to allow one-to-two = division. | Med-High | Med | Med (2–4d) | Med (cost design transfers if in µm + motion) | **[V]** unbalanced/partial OT proven for particle tracking | POT: MIT | PythonOT (`ot`); arxiv 2407.04583 (unbalanced OT particle tracking) | Swap assignment only (fixed costs); does Sinkhorn+unbalanced beat greedy on train edge-Jaccard? |
| 5 | **OC-SORT motion cues** (OCM direction + ORU re-update) | Add observation-centric *momentum* (velocity-direction agreement over a history window) to edge cost; re-update after gaps. Motion-only, no appearance. | Med | Low | Low (1–2d) | Low (pure geometry) | **[V]** CVPR23 SOTA on assoc metrics (HOTA/IDF1) | MIT | github.com/noahcao/OC_SORT | Add OCM term to edge cost; ID-switch / edge-Jaccard drop on dense regions? |
| 6 | **CPD / registration pre-warp** (global rigid + local non-rigid) | Register frame t point-set → t+1 before gating so distances are deformation-corrected; CPD's GMM posterior itself yields soft correspondences + a coherent velocity field. | Med | Med | Med (2–3d) | Med (over-warp risk in mitotic bursts) | **[V]** CPD is the standard non-rigid point-set method | pycpd: MIT | pycpd; NeurIPS'06 CPD; CVPR22 revisit | Rigid-align consecutive frames; does residual-distance gating cut long-range mismatches? |
| 7 | **Neural Scene Flow Prior** (test-time MLP flow, cycle-consistent) | Per frame-pair, fit fwd+rev flow MLPs at test time (no training) → dense deformation field to warp predicted positions before gating. Source-free by design. | Med | High | Med-High (3–5d, GPU/pair) | Med (per-pair overfit; regularize) | **[V]** NeurIPS'21 spotlight, runtime-only optimization | repo license (check) | github.com/Lilac-Lee/Neural_Scene_Flow_Prior | Fit NSFP on one frame pair; does warping reduce t→t+1 NN mismatch vs raw? Budget check on T4. |
| 8 | **Fused Gromov-Wasserstein / kNN-constellation** | Match on local *neighborhood structure* (intra-set distance matrices), not absolute position — invariant to global drift; FGW blends feature + structure cost. | Med | High | High (4–6d) | Low-Med (structure is drift-robust) | **[V]** FGW proven for graph/point matching | POT: MIT | PythonOT FGW; hal-02971153 | Build kNN constellation cost; does it beat plain-distance cost under simulated global drift? |
| 9 | **TENT / adaptive-BN test-time** | Update only BN stats + affine on target frames by entropy-min. Cheap source-free adaptation of any conv backbone. | Low-Med | Low | Low (1–2d) | **High** — entropy-min can collapse; mostly helps *detector*, weak for association | **[V]** ICLR21, but for classification/seg | MIT | github (tent) | Recompute BN stats on target; any lift, or confidence collapse? Likely weak for edges. |
| 10 | **Exa.TrkX object-condensation GNN** | MLP-embed each nucleus so same-track points cluster; radius-graph + edge-classifier GNN = learned linker. | Low-Med | High | High (retrain, 6d+) | **High** (needs training on 2 embryos → overfit) | **[V]** works at LHC scale, different domain | Apache-2.0 | exatrkx.github.io; arxiv 2504.04670 | Only if §1–5 saturate: train tiny GNN on 44b6, test 6bba — does it generalize with 2 embryos? |
| 11 | **DARTH** (ICCV23 MOT TTA) | Detection-consistency + patch-**contrastive appearance** TTA. | Low | — | Med | **High** — its core loss is *appearance* re-ID, which we do not have | **[V]** but off-domain | check repo | github.com/mattiasegu/darth | Mine the detection-consistency idea only; appearance branch N/A → skip. |

---

## 3. Cross-domain translation: their vocabulary → our metric

The competition scores **adjusted edge-Jaccard** (did we predict the correct set of temporal
mother→daughter/self edges) + 0.1·**division-Jaccard**. Mapping each field:

- **hits / detections** → given, ~solved. Not our lane except as OT slack (a detection with no
  good partner should go to a birth/death dummy, not force a wrong edge).
- **edges** → the numerator/denominator of the main metric. Every method here is ultimately a
  *cost/score on a candidate temporal edge*. OT (row 4), OC-SORT momentum (5), FGW (8) all produce
  edge scores; consistency methods (1,3) *filter* them.
- **tracks** → chains of accepted edges; global consistency (path/cycle) is what stops the local
  greedy swaps that silently destroy edge-Jaccard in dense regions.
- **divisions** → in OT this is a **one-to-two** transport: allow a mother column to be matched by
  two daughter rows (duplicate the column, or use a capacity-2 unbalanced formulation). In LAP
  terms it's the "split" cost. Feeds the 0.1 term (currently 0 for us — pure upside).
- **births / deaths** → **unbalanced / partial OT** slack: dummy source/sink rows absorbing
  unmatched nuclei at a fixed cost, so entering/leaving cells don't get force-linked. Directly
  reduces false edges → raises precision side of edge-Jaccard.
- **motion / deformation** → CPD (6), NSFP (7), scene-flow: correct the *coordinates* before any
  gating so the "distance" that all edge costs are built on is measured in a deformation-stabilized
  frame. Global specimen drift + local tissue flow otherwise inflate every edge distance and
  break the µm gate — worst exactly on the unfamiliar embryo. This is upstream of everything.

**A. Method C (forward–backward gating)** is our concrete C-consistency mapping: an edge
(u@t → v@t+1) is trustworthy iff, matching backward t+1→t, v maps back to u. Keep/upweight cyclic
edges, downweight the rest → fewer false edges, higher edge-Jaccard, zero training, almost
certainly rules-safe.

**A. Method 4 (OT) integration:** build the cost matrix in **physical µm** (voxel z,y,x =
1.625, 0.40625, 0.40625; z is 4× coarser — never gate isotropically), add velocity-residual and
optional Trackastra/organizer edge-score terms, run entropic Sinkhorn with unbalanced marginals +
a duplicated-column division slot. It degrades gracefully to soft assignment where the graph is
ambiguous — which is exactly the failure mode on a new embryo — and it slots *under* the ILP as a
better per-window cost or *replaces* greedy directly.

---

## 4. What actually transfers with motion/geometry only (no appearance identity)

The hidden embryo gives us images but **no re-ID / appearance identity signal that survives the
shift** (nuclei look generic; intensity/scale differ per embryo). That immediately re-ranks the
field:

- **Transfers well:** Trackastra (trained precisely to generalize across modalities via *shallow*
  geometric feats — won CTC generalizable-linking) **[V]**; OC-SORT (pure motion) **[V]**; OT/FGW
  and kNN-constellation (geometry/structure) **[V]**; CPD/NSFP (deformation is physical, not
  domain-specific) **[V]**; path/cycle consistency (a *relational invariant*, domain-agnostic)
  **[V]**.
- **Transfers poorly / risky:** DARTH's patch-contrastive **appearance** TTA (no stable appearance
  identity here) **[V, off-domain]**; TENT-style entropy-min (can collapse to confidently-wrong on
  a new domain; its natural target is the detector's BN, not the edge head) **[I]**; a from-scratch
  GNN linker trained on **2** embryos (high overfit/generalization risk) **[I]**.

Design rule for this lane: prefer methods whose correctness criterion is a *physical or relational
invariant* (cycle closes, mass is conserved, motion is coherent) over methods whose criterion is
*model confidence*. The former can't be gamed by the domain shift; the latter is exactly what the
shift breaks.

---

## 5. Recommended sequencing (cheapest-falsifiable first)

1. **Day 1 — Method 3 (forward–backward gating).** Zero training, rules-safe, tells us whether
   consistency signal exists at all in this data. Gate for the whole TTA branch.
2. **Confirm Trackastra fusion (row 2)** stays net-positive on a held-out embryo split — it's
   already in-stack and low-risk; keep as the association backbone.
3. **Method 4 (OT assignment)** as a drop-in for greedy: unbalanced (births/deaths) + one-to-two
   (division) in µm cost. Improves both metric terms without retraining.
4. **Add OC-SORT momentum (row 5)** as an extra edge-cost term — a 1-line-of-physics robustness
   boost for dense/nonlinear motion.
5. **If §1 gate passes → Method 1 (learned path-consistency TTA)** on the target embryo. This is
   the private-shuffle weapon. **Clear the rules question first.**
6. **Deformation pre-warp (CPD row 6, else NSFP row 7)** only if residual global/tissue drift is
   measurably breaking the µm gate on the unfamiliar embryo.

FGW (8), TENT (9), Exa.TrkX GNN (10), DARTH (11) are reserve/low-priority for the reasons in the
table.

---

## Sources

- Path Consistency (the bet): arxiv.org/html/2404.05136v1 ; openaccess.thecvf.com/content/CVPR2024/papers/Lu_Self-Supervised_Multi-Object_Tracking_with_Path_Consistency_CVPR_2024_paper.pdf
- Trackastra (CTC generalizable-linking winner 2024): github.com/weigertlab/trackastra ; arxiv.org/html/2405.15700v1 ; ecva.net/papers/eccv_2024/papers_ECCV/papers/09819.pdf
- Cycle-consistency self-sup MOT: pure.qub.ac.uk/en/publications/self-supervised-multi-object-tracking-with-cycle-consistency ; arxiv.org/html/2605.30211
- Unbalanced/partial OT for particle tracking: arxiv.org/abs/2407.04583 ; PythonOT (POT) pythonot.github.io
- Fused Gromov-Wasserstein (structure matching): hal.science/hal-02971153 ; arxiv.org/html/2502.09934 (fused partial GW)
- OC-SORT (motion-only association): github.com/noahcao/OC_SORT ; openaccess.thecvf.com/content/CVPR2023/papers/Cao_Observation-Centric_SORT_..._2023_paper.pdf ; arxiv.org/pdf/2203.14360
- Coherent Point Drift: papers.nips.cc/paper/2962-non-rigid-point-set-registration-coherent-point-drift ; CVPR22 revisit openaccess.thecvf.com/content/CVPR2022/papers/Fan_Coherent_Point_Drift_Revisited_..._paper.pdf ; pycpd
- Neural Scene Flow Prior (test-time, source-free): github.com/Lilac-Lee/Neural_Scene_Flow_Prior ; proceedings.neurips.cc/paper/2021/hash/41263b9a46f6f8f22668476661614478-Abstract.html
- DARTH (MOT TTA): github.com/mattiasegu/darth ; arxiv.org/abs/2310.01926
- TENT (entropy-min TTA / adaptive BN): openreview.net/forum?id=uXl3bZLkr3c ; arxiv.org/abs/2006.10726
- CMTT-JTracker (cell-tracking fully-TTA framework): pmc.ncbi.nlm.nih.gov/articles/PMC11570544 ; academic.oup.com/bib/article/25/6/bbae591/7902784
- Exa.TrkX / object-condensation GNN track finding: exatrkx.github.io ; arxiv.org/pdf/2504.04670
