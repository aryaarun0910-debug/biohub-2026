# Novel / cross-domain portable mechanisms (agent report) — 2026-08-17

Raw agent output; evidence store (outside Git). Mandate: NOVEL / underexplored, cross-domain
mechanisms that port to our **candidate-graph LINKER / division head / node-count calibration**
operating on **existing detections + geometric/learned features on 2 embryos, no imaging retrain**.

Explicitly goes BEYOND `methods_frontier_2026-08-16.md` — does NOT re-list D2D-Rescore
(arXiv:2606.03568), HOCT (arXiv:2607.11754), Cellpose-SAM, LSM-FM (arXiv:2605.26026), SELMA3D.
Where a lever is adjacent to that report's TTA/SSL entry (#4) it is folded, not repeated.

Ranked by **portability × EV × cheapness**. Target metric levers named per entry:
`EDGE` = adj_edge_J plateau (~0.90-0.91); `DIV` = division_jaccard +0.06 ceiling;
`NODE` = the `(1 - 0.1*(N_pred-N_est)/N_est)` multiplier; `XFAM` = 2-embryo cross-family trap.

---

## Ranked summary

| # | Mechanism | Attacks | GPU? | Cheapest test surface |
|---|-----------|---------|------|-----------------------|
| 1 | **Unbalanced/Partial OT linker** (Sinkhorn transport plan replaces greedy/threshold) | EDGE + NODE + DIV | **No** | POT / uotod on existing candidate edges |
| 2 | **Post-hoc meta-classifier** quality estimate on frozen candidate features (MetaDetect/LMD) | EDGE + NODE | **No** | GBM on existing geffs, secs |
| 3 | **PU-learning loss** (nnPU) for the re-scorer under 2.8% annotation | EDGE + NODE + XFAM | **No** | Loss swap in #2, ablation |
| 4 | **Conformal / FDR-controlled acceptance** for node-count | NODE + EDGE + XFAM | **No** | Post-hoc threshold, few lines |
| 5 | **Mitosis-aware MHT + aleatoric TTA** (BiologicalNeeds, public code) | DIV + EDGE | **No** | Port Erlang mitosis cost onto our detections |
| 6 | **Forward-backward cycle-consistency** as unsupervised FP signal + pseudo-labels | XFAM + EDGE + sparse-anno | **No** | Geometry on existing linker, in-notebook |
| 7 | **Heterophily-aware GNN encoder/decoder** (SAGE/SIGN + MLP/DistMult) | EDGE (non-homophilic sibling edges) | Light | Ablation if a linker GNN is built |
| 8 | **MoTT / learnable graph matching** context-aware association ranker | EDGE | Light T4 | Adapt public MoTT head to 3D µm coords |
| 9 | **Ensemble diversification / model soups** that survive cross-family | XFAM + EDGE (abstention) | **No** | N diverse seeds of #2, blend + gate |

Top-5 for immediate cheapest-falsification on existing OOF geffs without any GPU retrain:
**1, 2, 3, 4, 5** (6 is the cheapest XFAM/sparse-annotation lever and enables 2/3).

---

## 1. Unbalanced / Partial Optimal Transport as the frame-to-frame LINKER  *** flagship novel-portable ***

**Mechanism.** Replace greedy/threshold linking on the candidate graph with an **entropic
unbalanced-OT transport plan** (Sinkhorn). Cost matrix `C[i,j]` = our current per-edge cost
(physical-µm displacement, motion residual, learned edge score). **KL-relaxed marginals**
(soft mass conservation) let sources appear/disappear at a tuned penalty τ; a **relaxed target
marginal that permits one source → two targets is exactly cell DIVISION as mass-splitting**.
Critically, the **total transported mass / marginal-relaxation parameter directly sets how many
edges are accepted → a principled knob on N_pred vs N_est** (the metric's node-count multiplier),
which greedy thresholding controls only indirectly.

**Why portable.** Operates on the EXISTING candidate edges + features; pure CPU; two turnkey
libraries: **POT** (`ot.sinkhorn_unbalanced`, `ot.unbalanced.sinkhorn_knopp_unbalanced`,
`ot.partial.entropicwasserstein`; BSD/MIT-style, CPU) and **uotod** (`uotod.match.UnbalancedSinkhorn`
/ `BalancedSinkhorn` / `Hungarian`; params `background_cost` default 10, `reg_dimless` default 0.12;
returns `(B, num_pred, num_targets+1)` matching matrix; **LGPL-3.0**, `pip install uotod`, PyTorch
CPU-capable). uotod literally unifies Hungarian↔Sinkhorn↔unbalanced as one continuum — swap our
linker's assignment step and sweep one parameter.

**EV.** Direct hit on all three failing levers simultaneously; the OT global optimum can beat greedy
on precisely the ambiguous/crossing edges where adj_edge_J plateaus.

**Cheapest falsification.** On existing OOF candidate edges for embryo A and B: build `C` from the
current edge cost; run entropic unbalanced Sinkhorn; **sweep the marginal-relaxation τ per embryo
direction** (calibrate on A, freeze, apply to B and reverse). Report adj_edge_J AND realized
N_pred/N_est vs the current greedy linker, both directions separately. CPU, hours. **GPU: No.**

Sources: De Plaen et al., *Unbalanced Optimal Transport: A Unified Framework for Object Detection*,
CVPR 2023, arXiv:2307.02402, code github.com/hdeplaen/uotod (LGPL-3.0);
*Unbalanced optimal transport for stochastic particle tracking*, arXiv:2407.04583 (partial
Wasserstein of Gaussian measures, auto-detects #matched particles via hyperparameter optimization);
POT — pythonot.github.io.

**License flag:** uotod is **LGPL-3.0** (linking a modified version triggers copyleft on that
component). POT is permissive — prefer POT for a clean-license reimplementation of the same math.

---

## 2. Post-hoc meta-classifier / prediction-quality estimation on FROZEN candidate features (MetaDetect / LMD)

**Mechanism.** The cheapest possible instantiation of "learned RANKER not a threshold": a small
gradient-boosted tree / shallow MLP **meta-classifier** ingests the *already-computed* per-candidate
and per-edge hand-crafted features and predicts P(true) / an IoU-like quality, discriminating TP vs
FP **entirely post-hoc on a frozen detector+linker** — no deep net, no imaging retrain. MetaDetect
reports meta-classification AUROC up to 99.93% and meta-regression R² up to 0.92 from transparent
hand metrics; LMD (IJCV 2025) is the LiDAR-sparse analogue, a light post-processing head on a frozen
detector giving improved uncertainty/IoU on sparse points — the closest published domain-match to our
sparse-nuclei regime.

**Attacks.** EDGE (FP filtering restores precision on ambiguous edges) + NODE (rank-then-cut to drag
N_pred toward N_est). This is the D2D re-scorer's payload with the deep model removed.

**Cheapest falsification.** Train a GBM on embryo-A edges using the features already in the geffs;
threshold/rank on B; sweep the operating point; report adj_edge_J + N_pred/N_est both directions.
Seconds to train on CPU. If a GBM on existing features already lifts adj_edge_J, it de-risks (and may
obviate) the heavier transformer re-scorer. **GPU: No.**

Sources: Schubert, Kahl et al., *MetaDetect*, arXiv:2010.01695; *LMD: Light-Weight Prediction Quality
Estimation for Object Detection in Lidar Point Clouds*, IJCV 2025, doi:10.1007/s11263-025-02377-8.

---

## 3. Positive-Unlabeled (PU) learning loss for the re-scorer — corrects the 2.8%-annotation bias

**Mechanism.** Our regime is textbook PU: annotated nuclei/edges = **positive**; every other candidate
= **unlabeled** (a mix of true-unannotated + false). Any re-scorer/linker trained with naive BCE
treating unlabeled as negative is **systematically biased** — it learns to reject true-but-unannotated
edges and, worse, **miscalibrates the score that #2 and #4 threshold on, corrupting the node-count
multiplier**. The **non-negative PU (nnPU)** loss corrects this with a single estimated class-prior π,
recovering an unbiased risk. MICCAI-2021/MELBA-2022 work applies exactly this to cell detection with
incomplete point annotations; a 2025 Micromachines paper reports 90% annotation reduction on small-object
localization via PU.

**Attacks.** EDGE + calibration→NODE + XFAM (unbiased scores transfer better across the 2 families than
prior-contaminated ones).

**Cheapest falsification.** Retrain the #2 meta-classifier with nnPU loss vs BCE on identical features;
estimate π from the annotated fraction; compare cross-embryo AUROC and realized N_pred/N_est. Pure
ablation, CPU. **GPU: No.**

Sources: *Positive-unlabeled learning for cell detection in histopathology images with incomplete
annotations*, MICCAI 2021, arXiv:2106.15918; binary+multiclass extension arXiv:2302.08050 (MELBA 2022);
nnPU — Kiryo et al., NeurIPS 2017; *Small Object Localization with 90% Annotation Reduction by PU
Learning*, Micromachines 2025, doi:10.3390/mi16121379.

---

## 4. Conformal / FDR-controlled selective acceptance — makes NODE calibration a provable control problem

**Mechanism.** The metric penalizes N_pred > N_est — i.e. **spurious accepted edges are false
discoveries**. Rather than hand-tuning a global score threshold, use **conformal selection / conformal
risk control** to choose the acceptance threshold that **provably controls the false-discovery rate**
of accepted edges at a target level, calibrated on one embryo, applied to the other. Distribution-free,
single scalar, no retrain. MOT-CUP is the tracking-domain proof that conformal uncertainty can be
propagated through association (+2% accuracy, 2.67× uncertainty reduction, +4% under heavy occlusion).

**Attacks.** NODE directly (bounds the over-prediction term) + EDGE (calibrated cut) + gives a
**built-in XFAM transfer test** (calibrate A → guarantee holds on exchangeable B?).

**Cheapest falsification.** Take the #2/#3 scores on embryo A as calibration; set an FDR/FSR target;
derive the conformal threshold; apply to B; compare adj_edge_J and N_pred/N_est against a fixed global
threshold. If the empirical false-discovery on B tracks the target, we have a principled, transferable
node-count knob. **GPU: No.**

Sources: Jin & Candès, *conformal selection* (FDR-controlled), 2023; Angelopoulos et al., *Conformal
Risk Control*, 2022; *Selective Conformal Risk Control*, arXiv:2512.12844; *Conformal Selective
Prediction with General Risk Control*, arXiv:2603.24704; *Collaborative MOT with Conformal Uncertainty
Propagation* (MOT-CUP), arXiv:2303.14346 (IEEE T-IV 2024).

---

## 5. Mitosis-aware Multi-Hypothesis Tracking + aleatoric TTA (BiologicalNeeds — PUBLIC CODE)

**Mechanism.** Two drop-in pieces, both retrain-free on our existing detections+motion: (a) lift
single-point motion into **probabilistic spatial densities via problem-specific test-time augmentation**
(aleatoric uncertainty) — no encoder training; (b) a **mitosis-aware assignment** whose split cost is
derived from the **Erlang distribution of expected cell lifetimes**, combined with local position/motion
densities, and an MHT that resolves false associations and mitosis via **long-term conflicts** rather
than per-frame greedy decisions. This is the closest published, code-released attack on our +0.06
DIVISION ceiling that does not require an image encoder.

**Attacks.** DIV (high-precision, lifetime-aware fork selection) + EDGE (long-term consistency removes
identity switches greedy linking can't see).

**Cheapest falsification.** Port the Erlang mitosis-cost + TTA-density scoring onto our existing
detections/motion; A↔B; measure division_jaccard **precision at the 10% break-even** and adj_edge_J
delta from the long-term MHT vs current greedy. **GPU: No** (CPU MHT; TTA is cheap forward passes).

Sources: Kaiser, Schier, Rosenhahn, *Cell Tracking according to Biological Needs — Strong Mitosis-aware
Multi-Hypothesis Tracker with Aleatoric Uncertainty*, IEEE TMI 2025, arXiv:2403.15011; **code
github.com/TimoK93/BiologicalNeeds** (builds on EmbedTrack).

**License flag:** verify the BiologicalNeeds/EmbedTrack license before vendoring code; the Erlang-cost
+ TTA *mechanism* is reimplementable independently of their repo.

---

## 6. Forward-backward cycle-consistency as an unsupervised FP signal + pseudo-label engine

**Mechanism.** A true link should survive **forward AND backward** linking; disagreement is a cheap,
label-free false-positive detector — usable on the **~97% unannotated nuclei** and, crucially, on the
**hidden test embryo at inference** (the volumes are local; using them isn't a leak). Two payoffs:
(a) **pseudo-labels** that expand the PU/meta-classifier training set far beyond the 2.8% annotations
(feeds #2/#3); (b) a **test-time self-training / gating signal** on the unseen family (feeds XFAM)
with no labels. This is the cheapest lever that simultaneously attacks the sparse-annotation and
cross-family constraints, and it composes with everything above.

**Cheapest falsification.** Compute forward-vs-backward link agreement on embryo A; **correlate
cycle-consistency with GT edge correctness (AUROC)**. If predictive, (i) add it as a feature to #2,
(ii) use it to mine pseudo-positives/negatives for #3, (iii) test as an inference-time gate on B.
Pure geometry on the existing linker, in-notebook, CPU. **GPU: No.**

Sources: Baade et al., *Self-Supervised Cross-View Correspondence with Predictive Cycle Consistency*,
CVPR 2025; *Decoupled Spatio-Temporal Consistency Learning for Self-Supervised Tracking*,
arXiv:2507.21606; classic forward-backward — Kalal et al., *TLD*, 2010.

---

## 7. Heterophily-aware GNN encoder/decoder for the candidate graph — NEW angle on HOCT's non-homophilic finding

**Mechanism.** HOCT observed sibling edges are non-homophilic (H_adj~0.01), concluding node-GNNs
"can't aggregate." NeurIPS-2024 link-prediction theory gives the *fix* HOCT implied was impossible:
under feature heterophily, link probability correlates *negatively* with feature similarity, and
**no single linear/dot-product decoder can separate edges from non-edges** (their Theorem 2). The
prescription: an encoder with **ego/neighbor-embedding separation** (GraphSAGE or SIGN — keeps a
distinct self-representation, up to +44% over GCN on synthetic heterophily) plus an **expressive
decoder (MLP, or DistMult, NOT dot-product)** (+55% over dot-product in negative-similarity regions).
So a *node-GNN linker is viable* if architected for heterophily — a distinct route from HOCT's
edge-token transformer.

**Attacks.** EDGE (the division-entangled, non-homophilic sibling edges specifically).

**Cheapest falsification.** If/when a linker GNN is built, run the ablation SAGE+MLP-decoder vs
GCN+dot-product on A↔B candidate-edge classification; measure adj_edge_J. Light GPU; prototypable CPU.

Sources: *On the Impact of Feature Heterophily on Link Prediction with Graph Neural Networks*,
NeurIPS 2024, arXiv:2409.17475.

---

## 8. Context-aware learned association ranker — MoTT (SPT) / Learnable Graph Matching (MOT)

**Mechanism.** MoTT: a transformer attends over each live tracklet's past + **hypothetical future**
tracklets to output matching probability, existence probability, and next position — a multi-hypothesis
association ranker that beats classical JPDA/MHT on the ISBI SPT challenge. Learnable Graph Matching
(GMTracker): differentiable **quadratic** graph matching injecting higher-order structure into vertex
features, solved with Sinkhorn — data association as a learned QP. Both learn the ranker on top of
detections (no imaging retrain) and are small.

**Attacks.** EDGE (context/higher-order association, identity switches under crossings).

**Cheapest falsification.** Adapt MoTT's public tracklet-matching head to our 3D physical-µm coords on
existing detections; A↔B. Heavier than 1-6 (needs training) but small models; T4-scale, CPU-prototypable.
**GPU: Light T4.**

Sources: Zhang et al., *A Motion Transformer for Single Particle Tracking in Fluorescence Microscopy
Images*, MICCAI 2023, bioRxiv 2023.07.20.549804, **code github.com/imzhangyd/MoTT**; He et al.,
*Learnable Graph Matching*, arXiv:2303.15414 (GMTracker).

---

## 9. Cheap ensembling that survives cross-family — ensemble diversification / model soups

**Mechanism.** With only 2 families, member **diversity is the OOD signal**: Scalable Ensemble
Diversification (SED) trains members to disagree off-distribution, so member-agreement itself flags
cross-family-fragile edges (→ abstain/down-weight); **model soups** weight-average re-scorer seeds for
a robust merge at constant inference cost. Applied to the small #2/#3 re-scorers, not the imaging net.

**Attacks.** XFAM + EDGE (disagreement-gated abstention raises precision on the unseen family).

**Cheapest falsification.** Train N diverse meta-classifier seeds on A, blend, eval B vs single seed;
use inter-seed disagreement to gate low-confidence edges; report adj_edge_J + N_pred/N_est. CPU.
**GPU: No.**

Sources: *Scalable Ensemble Diversification for OOD Generalization and Detection*, arXiv:2409.16797;
Wortsman et al., *Model Soups*, ICML 2022.

---

## Cross-cutting synthesis — these compose into one cheap CPU pipeline

Over-propose (drop DoG threshold) → **#2 meta-classifier (features already exist), trained with #3 nnPU
loss and #6 cycle-consistency pseudo-labels** → **#1 unbalanced-OT assignment** whose mass knob and
**#4 conformal-FDR threshold** jointly calibrate N_pred to N_est → **#5 Erlang mitosis-cost** on the
resulting graph for the division head → **#9 diverse-seed blend + #6 inference-time cycle gate** for the
hidden family. Every stage is CPU, trains on existing OOF geffs/candidate features, and reports both
embryo directions separately. Nothing here needs the imaging retrain. Falsify cheapest-first: #2 and #1
are the two independent load-bearing bets; #6 de-risks #2/#3 and #9; #4 makes #1's mass knob principled.

## Flags / licenses
- **uotod LGPL-3.0** — copyleft on linked modifications; prefer permissive **POT** to reimplement the
  same unbalanced-Sinkhorn math cleanly.
- **BiologicalNeeds / EmbedTrack** — verify license before vendoring; Erlang-cost + TTA mechanism is
  independently reimplementable.
- **MoTT** — verify repo license before code reuse; MICCAI method reimplementable from paper.
- LMD / MetaDetect numbers are automotive/LiDAR — **mechanism transfers, numbers do not**; re-measure
  on our 2 embryos.
- Heterophily-LP (#7) has no released repo — reproduce from paper appendix; it is an *architecture
  prescription*, cheap to A/B if a linker GNN already exists.
- All "cheapest tests" assume the official patched scorer + pooled objective, both directions reported.
