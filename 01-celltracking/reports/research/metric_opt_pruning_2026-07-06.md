# Metric-Aligned Optimization + Component Pruning — Research Lane

Date: 2026-07-06 · Lane: make the SCIP ILP optimize the DOCUMENTED `adj_edge_J`, not a generic cost; plus cross-embryo-safe pruning/existence.
Evidence tags: **[V]** verified from primary source · **[C]** claim (secondary/plausible) · **[I]** my inference for THIS problem.

> Scope note: this lane optimizes the *documented* metric using only valid lineage-constrained predictions. No evaluator-defect exploitation. The one-sided count factor is part of the *published* scoring rule, so calibrating N_pred against it is a legal modeling choice, not an exploit.

---

## 1. The metric, restated as a linear-fractional program

Per-sample score (given):

```
adj_edge_J = max(0, J · (1 − 0.1 · (N_pred − N_est)/N_est))
J          = TP / (TP + FP + FN) = TP / (G + FP)          [since FN = G − TP]
```

- `G` = number of ground-truth (GT) edges on the hidden embryo (**hidden**).
- `N_est` = estimated node count (**hidden**), `N_pred` = our node count (**we choose**).
- Run score = weighted mean of per-sample `adj_edge_J` + `0.1 · division_J`.

Decision variables: `x_e ∈ {0,1}` over candidate edges, subject to the existing tracksdata lineage constraints (≤1 predecessor, ≤2 successors at a division, flow/appearance/disappearance, division consistency). Let

- `pTP(e)` = P(e is scored as a true positive edge),
- `pFP(e)` = P(e is scored as a false positive edge).

Because annotations are **sparse**, every selected edge falls in **three classes**: (a) matches a GT edge → **TP**, (b) contradicts an annotated GT edge → **metric-FP**, (c) lies in an unannotated region → **IGNORED** (neither). Hence **`pFP(e) ≠ 1 − pTP(e)`** — the central modeling fact of this lane. **[I]**

Under a calibrated model, the plug-in ("ratio-of-expectations") approximation of expected `J` is a **linear-fractional function of x**:

```
E[J](x) ≈  N(x)/D(x),   N(x) = Σ_e pTP(e) x_e ,   D(x) = G + Σ_e pFP(e) x_e   > 0
```

This is exactly the object Nowozin optimizes for expected IoU/Jaccard with a *parametric-LP* algorithm **[V]** (Nowozin, CVPR 2014, "Optimal Decisions from Probabilistic Models: the Intersection-over-Union Case") — direct academic precedent: IoU is combinatorially hard as an expected loss, and he replaces `E[IoU]` by a statistical approximation solved by parametric linear programming, which is the Dinkelbach scheme below. Note the plug-in swaps `E[N/D]` for `E[N]/E[D]`; Nowozin shows this approximation works well in practice. **[V/I]**

---

## 2. Dinkelbach derivation for THIS metric

Dinkelbach (1967, *Management Science* 13(7):492–498, "On Nonlinear Fractional Programming") **[V]**: to maximize a ratio `N(x)/D(x)` with `D(x)>0` over a feasible set `X`, solve the **parametric** problem

```
g(λ) = max_{x ∈ X} [ N(x) − λ D(x) ]
```

Facts (Dinkelbach 1967) **[V]**: `g` is convex, continuous, strictly decreasing in λ, and has a **unique root** `λ*`; `g(λ*) = 0` and the maximizer of the ratio is the `x` attaining `g(λ*)`. The fixed-point iteration `λ_{k+1} = N(x_k)/D(x_k)` converges **superlinearly**.

Substituting our `N, D`:

```
g(λ) = max_{x∈X} [ Σ_e pTP(e)x_e − λ(G + Σ_e pFP(e)x_e) ]
     = max_{x∈X} [ Σ_e ( pTP(e) − λ·pFP(e) ) x_e ]  −  λG
```

**Key structural result [I, verified algebra]:** the term `−λG` is **constant in x**. Therefore the argmax over `x` at a fixed λ depends **only** on the per-edge coefficient

```
w_e(λ) = pTP(e) − λ·pFP(e)          →  ILP edge cost  c_e = −w_e(λ)
```

Consequences:
1. **The hidden `G` is not needed to pick edges at a fixed λ.** `G` only shifts `g(λ)` vertically, i.e. it moves the root λ* (the *achieved* J value). For a **fixed, OOF-calibrated λ** we can select optimally without ever knowing `G`. This is what makes the scheme deployable on a disjoint hidden embryo.
2. At the optimum `λ*` equals the achieved expected-J, and each edge enters (constraints permitting) iff `pTP(e)/pFP(e) > λ*` — a principled **ratio threshold**, not a generic cost.

**Algorithm (drop-in for the current SCIP solve):**

```
choose λ0 (OOF-calibrated, ≈ current OOF adj_edge_J, e.g. 0.85)
repeat for k = 0,1,2…:
    solve  x_k = argmax_{x∈X} Σ_e (pTP(e) − λk·pFP(e)) x_e     # existing lineage-constrained ILP, new costs
    λ_{k+1} = (Σ_e pTP(e)x_k) / (Ĝ + Σ_e pFP(e)x_k)             # Ĝ = OOF proxy for G, ONLY for stopping
until |λ_{k+1} − λk| < ε   (typically 2–4 iterations)
```

`Ĝ` (a per-embryo GT-edge-count proxy from OOF, e.g. ≈ N_pred·mean_degree) is used **only** in the stop test, never in edge selection. Even a rough `Ĝ` is fine because the fixed-point is contractive. If you prefer zero dependence on `Ĝ`, **skip the iteration entirely** and use a single OOF-tuned constant λ — the selection ILP is identical, and grid-searching λ on the two OOF embryos to maximize measured `adj_edge_J` is the most robust deployment. **[I]**

**Three-class handling of `pFP`.** With sparse labels:
```
pFP(e) = (1 − pTP(e)) · P(e is a metric-FP | e is not TP)      # the "not-ignored given wrong" mass
```
- **Two-class conservative approximation:** set `P(metric-FP | not TP) = 1`, i.e. `pFP = 1 − pTP`. This *over*-penalizes selection → conservative/shrunken tracks. Safe first cut, guaranteed no worse than generic cost if λ is tuned. **[I]**
- **Sparse-aware:** most wrong-looking edges are IGNORED, so `P(metric-FP|not TP) ≪ 1`. Estimate it from OOF: among selected non-TP edges, the fraction that actually landed on an annotated GT node's alternative. This shrinks `pFP`, raising the effective λ budget → more aggressive (correct) selection. This is the payoff over the two-class version. **[I]**

---

## 3. Why isolated detections steal optimal 1-to-1 matches (and count-penalty synergy)

The evaluator matches predicted edges/tracks to GT **one-to-one**. A spurious **isolated detection** contributes an edge that can be greedily matched to a GT node, **consuming** that GT, so the *true* predicted node then has no GT left to match → **two** errors from one bad detection (a stolen TP + an added FP). Removing the isolated detection frees the GT for its correct partner. This is the mechanism behind the observed **+0.018 / +0.011 OOF** from isolated-node pruning. **[I, consistent with observed result]**

**Legal count-penalty synergy.** The published factor `(1 − 0.1·(N_pred − N_est)/N_est)` is **one-sided with no upper clip**: predicting *fewer* nodes than `N_est` yields a factor **> 1**. Pruning a low-value node therefore does **two** good things at once: (i) raises `J` (frees 1-to-1 matches), and (ii) raises the count factor by ≈ `0.1/N_est` per removed node. A node should be pruned whenever its **marginal** contribution to `J` is below the count-factor gain — a clean, documented decision rule:

```
prune node v  iff   ΔJ_keep(v)  <  0.1 · J / N_est
```
where `ΔJ_keep(v)` is the (OOF-estimated) J gain from keeping v's edges. Because `N_est` is hidden, use the OOF `N_est`≈`N_pred` regime and tune the RHS as a single global threshold. **[I]** This is calibration to a *published* rule, not a defect.

---

## 4. Calibration (prerequisite for §2–3)

All of the above assume `pTP`, `pFP` are **true probabilities**. Raw edge scores from Trackastra/Ultrack/GNN are typically miscalibrated. Do **out-of-fold** calibration on the two held-out embryos **before** the fractional solve:

- **Isotonic regression** — flexible, monotone; needs ≳1000 calibration points (we have thousands of edges) — good fit here. Risk: overfits on tiny folds. **[V]** (scikit-learn `CalibratedClassifierCV`, method="isotonic").
- **Temperature/Platt (sigmoid)** — 1–2 params, robust on small data, smoother. Safer cross-embryo. **[V]**
- Calibrate `pTP` and `pFP` **separately** (they are different events under sparse labels), each with its own OOF isotonic/Platt map. **[I]**

Recommendation: **Platt/temperature as the default (cross-embryo-robust), isotonic as an OOF challenger**; pick by measured OOF `adj_edge_J`, not by ECE. **[I]**

---

## 5. Ranked methods

| # | Method | Mechanism | EV (OOF pts) | Effort | Risk | Evidence | Repo / cite | First experiment |
|---|--------|-----------|--------------|--------|------|----------|-------------|------------------|
| **1** | **Dinkelbach metric-aligned edge cost** | replace SCIP cost with `c_e=−(pTP−λ·pFP)`; 2–4 outer iters or fixed OOF λ | **high (+0.01–0.03)** | **med** (reuse ILP) | low–med | Dinkelbach 1967 **[V]**; Nowozin 2014 **[V]** | SCIP/ilpy (have) | fixed λ=OOF-J, two-class pFP, grid λ∈[0.7,0.95] on 2 OOF embryos |
| 2 | Sparse-aware 3-class pFP | estimate `P(metric-FP\|not TP)` from OOF, feed into #1 | med-high (+0.005–0.015 over #1) | med | med | this doc §2 **[I]** | — | after #1: measure ignored-fraction of selected non-TP edges |
| 3 | Selective node pruning + count-penalty rule | prune node iff `ΔJ < 0.1·J/N_est`; single global τ | **high (already +0.018/+0.011)** | **low** | low | observed result; §3 **[I]** | existing pruning code | sweep global keep-threshold on OOF; add count-factor term |
| 4 | OOF calibration (Platt / isotonic) | monotone map raw score→prob before #1–3 | med (enables 1–3) | low | low | scikit-learn calibration **[V]** | sklearn (have) | isotonic vs Platt on `pTP`,`pFP`, score by adj_edge_J |
| 5 | Conformal / selective abstention on nodes | distribution-free keep/abstain threshold w/ coverage guarantee | med | med | low-med | conformal risk control **[V]** | MAPIE-style | calibrate node-keep quantile on OOF, apply to hidden |
| 6 | Min-cost-flow reformulation (muSSP) | exact/faster data-association if ILP too slow at scale | low (speed, not score) | med | low | Zhang-Li-Nevatia CVPR 2008 **[V]**; muSSP **[V]** | github yu-lab-vt/muSSP | only if SCIP >12h; keep #1 costs |
| 7 | Decision-focused learning (SPO+/diff-opt) | train scores to minimize decision loss directly | uncertain | **high** | high | DFL survey arXiv:2307.13565 **[V]** | — | **defer** — not tractable offline on Kaggle T4x2 in <12h |

Notes: EV ranges are **[I]** estimates anchored on the one verified data point (+0.018/+0.011 from isolated-node pruning). #6 is a fallback for tractability only — the existing ILP already solves, so min-cost flow buys speed not score; use only if the metric-aligned ILP blows the 12h budget. #7 is explicitly **out of scope for this competition's compute** — decision-focused training needs the solver in the training loop for many epochs, infeasible internet-off on T4x2.

---

## 6. THE ONE BET

**Replace the generic SCIP edge cost with the Dinkelbach metric-aligned cost `c_e = −(pTP(e) − λ·pFP(e))`, where `pTP`/`pFP` are OOF-calibrated (Platt) and λ is a single constant grid-tuned on the two OOF embryos to maximize measured `adj_edge_J`.**

Why this and not the others:
- It optimizes the **documented metric directly** with the **solver we already have** — surgical, ~a cost-vector swap plus a λ grid.
- It **does not need the hidden `G`** (the `−λG` term is constant in x), so it transfers to the disjoint hidden embryo — the property that kills most metric-alignment attempts here.
- It **subsumes pruning**: an isolated-detection edge has low `pTP`, non-trivial `pFP` → negative `w_e` → dropped automatically, unifying the +0.018/+0.011 pruning win with edge selection under one λ.
- Verified precedent (Nowozin 2014 parametric-LP for expected Jaccard; Dinkelbach 1967 convergence) de-risks the math.

**First experiment (half-day):** (1) OOF Platt-calibrate `pTP`; set `pFP = 1−pTP` (two-class, conservative); (2) for λ in `{0.70,…,0.95}` step 0.05, re-solve the existing lineage ILP with `c_e=−(pTP−λ·pFP)`; (3) plot OOF `adj_edge_J` vs λ on both embryos, pick λ*; (4) then add the sparse-aware `pFP` (#2) and the count-penalty prune rule (#3) as increments. Ship if OOF `adj_edge_J` at λ* beats the current generic-cost baseline on **both** embryos.

---

## Sources
- Dinkelbach, "On Nonlinear Fractional Programming", Management Science 13(7), 1967. https://pubsonline.informs.org/doi/10.1287/mnsc.13.7.492 (algorithm/analysis: https://pubsonline.informs.org/doi/10.1287/mnsc.22.8.868)
- Nowozin, "Optimal Decisions from Probabilistic Models: the Intersection-over-Union Case", CVPR 2014. https://openaccess.thecvf.com/content_cvpr_2014/papers/Nowozin_Optimal_Decisions_from_2014_CVPR_paper.pdf
- You, Castro, Grossmann, "Dinkelbach's Algorithm as an Efficient Method for Solving a Class of MILFP". https://egon.cheme.cmu.edu/Papers/MILFP_YouCastroGrossmann.pdf
- Analysis of Dinkelbach for 0-1 fractional programming (METR92-14). https://www.keisu.t.u-tokyo.ac.jp/data/1992/METR92-14.pdf
- Zhang, Li, Nevatia, "Global Data Association for Multi-Object Tracking Using Network Flows", CVPR 2008. http://vision.cse.psu.edu/courses/Tracking/vlpr12/lzhang_cvpr08global.pdf
- muSSP exact min-cost-flow for data association. https://github.com/yu-lab-vt/muSSP
- scikit-learn probability calibration (isotonic / sigmoid / CV). https://scikit-learn.org/stable/modules/calibration.html
- Decision-Focused Learning survey, arXiv:2307.13565. https://arxiv.org/html/2307.13565v4
- Selective Conformal Risk Control, arXiv:2512.12844. https://arxiv.org/html/2512.12844
