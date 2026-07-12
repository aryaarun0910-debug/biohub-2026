# Decision Theory for Directly Maximizing the Biohub Adjusted-Edge-Jaccard Metric

**Date:** 2026-07-12
**Scope:** How to select the final set of edges / nodes / divisions from calibrated candidate probabilities so as to maximize the *expected* competition metric out-of-sample, under lineage constraints and a node-count budget, when labels are sparse and positive-only.

---

## 0. The metric, restated precisely

Per sample `s` (an embryo / field / crop with its own supplied `N_est`):

```
adj_J_s = max(0,  J_s * (1 - 0.1 * (N_pred_s - N_est_s)/N_est_s))
J_s     = TP_s / (TP_s + FP_s + FN_s)         # edge Jaccard, nodes matched within 7 µm
```

Overall:

```
score = ( Σ_s w_s * adj_J_s ) / ( Σ_s w_s )   +   0.1 * division_jaccard
w_s   = TP_s + FP_s + FN_s                     # "edge volume" weight
```

**Sparse / positive-only labels ⇒ three outcomes per candidate edge, not two.** A predicted edge that is not matched to a labelled GT edge is *not* automatically a false positive: it may fall in an *unannotated* region and be **ignored**. So for any candidate edge `e` we have three mutually exclusive latent events if we select it:

- **TP(e)**: matches an annotated GT edge within 7 µm  → prob `pTP(e)`
- **counted-FP(e)**: lies in an annotated neighbourhood but matches nothing → prob `pFP(e)`
- **ignored(e)**: lies in an unannotated region → prob `pIG(e)`

with `pTP + pFP + pIG = 1` and, crucially, **`pFP(e) ≠ 1 − pTP(e)`**. Any method that uses a single binary edge probability and treats `(1 − p)` as false-positive mass is mis-specified for this metric.

---

## Executive summary — the 5 theoretical levers

1. **Volume-weighting collapses the weighted average of per-sample Jaccards into ONE global (micro-averaged) fractional program.** Because `w_s · J_s = (TP_s+FP_s+FN_s)·TP_s/(TP_s+FP_s+FN_s) = TP_s`, the numerator of the weighted average is just `Σ_s TP_s·(1−0.1Δ_s)` and the denominator is `Σ_s (TP_s+FP_s+FN_s)`. One global ratio ⇒ **one global Dinkelbach parameter λ**, not one per sample. This is the single most useful structural fact about the metric.

2. **Maximize the ratio by Dinkelbach's parametric method, which turns "maximize Jaccard" into an iterated *linear* edge objective** `c_e = (1−0.1Δ_s)·pTP(e) − λ·pFP(e)`. Select edge iff `c_e > 0`, subject to lineage constraints; then update `λ ← achieved objective`. (Nowozin CVPR 2014 uses exactly this ratio-of-expectations + parametric-threshold structure for expected IoU; Dinkelbach 1967 gives the convergence.)

3. **λ is a dimensionless shadow price in [0,1] equal, at optimum, to the achieved global Jaccard-like value — so it transfers across embryos and does NOT require the hidden denominator.** You never need to know `G_s` (GT edge count) or the hidden node total; you calibrate λ out-of-fold as "the value one TP is worth relative to one counted-FP," a *scale-free* quantity.

4. **The count penalty is a per-node Lagrangian tax `κ_s = 0.1 · E[TP_s] / N_est_s`, and it is the *only* force that disciplines ignored edges.** Ignored edges add 0 to Jaccard but +1 to `N_pred` — without the penalty you would dump every non-counted-FP edge. The penalty also actively drives `N_pred → N_est` (a bonus factor `>1` below `N_est`, a tax above), so **`N_pred ≈ N_est` is the robust anchor** the organizer hands you for free.

5. **Everything must be built on a genuine three-class calibrator (TP / counted-FP / ignored) with out-of-fold + hidden-embryo prior-shift correction (Saerens EM / MLLS).** The largest deployment shift is *annotation density*, which moves the ignored-vs-counted-FP split; correct it with EM re-estimation of the class priors on the hidden embryo, and make the whole selection *distributionally robust* (min-fold over the 2 labelled embryos) because 2 embryos is a tiny calibration set.

---

## 1. Prior art and how it maps onto our problem

### 1.1 Nowozin — Optimal Decisions from Probabilistic Models: the IoU case (CVPR 2014)
- PDF: https://openaccess.thecvf.com/content_cvpr_2014/papers/Nowozin_Optimal_Decisions_from_2014_CVPR_paper.pdf
- Abstract / repo: https://openaccess.thecvf.com/content_cvpr_2014/html/Nowozin_Optimal_Decisions_from_2014_CVPR_paper.html
- IEEE: https://ieeexplore.ieee.org/document/6909471/

**Core result we reuse.** IoU/Jaccard is a *set-level fractional* score; its exact Bayes-optimal decision is combinatorial (the loss is not decomposable over pixels/edges). Nowozin makes it tractable with two moves:

1. **Ratio-of-expectations approximation** of the expected score for a *fixed* decision set `A`:
   `E[ IoU(A, Y) ] ≈ E[|A ∩ Y|] / E[|A ∪ Y|]`.
   With independent marginals `p_i = P(i ∈ Y)`, `E[|A∩Y|] = Σ_{i∈A} p_i`, and `E[|A∪Y|] = |A| + Σ_{i∉A} p_i`.
2. **Parametric LP / threshold sweep.** The optimizer of the proxy over `A` is a *threshold on the marginals*: include element `i` iff `p_i > θ`. Sweep `θ` (a one-parameter family — a parametric linear program), evaluate the true expected IoU (by sampling `Y`) at each candidate `A`, keep the best. Reported to beat max-posterior-marginal (per-pixel MAP) decisions on 3 benchmarks.

The threshold sweep over `θ` is operationally a Dinkelbach sweep: at the optimal set the threshold equals a function of the achieved IoU. We inherit the *structure* (ratio-of-expectations + one scalar parameter that couples numerator and denominator) and extend it to (a) the three-outcome / ignore model, (b) a multiplicative count penalty, and (c) hard lineage constraints.

Related follow-ups confirming the threshold/soft-Jaccard structure: *Jaccard Metric Losses* (https://arxiv.org/pdf/2302.05666), *Lovász-Softmax* (https://openaccess.thecvf.com/content_cvpr_2018/papers/Berman_The_LovaSz-Softmax_Loss_CVPR_2018_paper.pdf).

### 1.2 Dinkelbach — nonlinear fractional programming (1967); Schaible, *Fractional Programming II: On Dinkelbach's Algorithm* (Management Science 1976)
- https://pubsonline.informs.org/doi/10.1287/mnsc.22.8.868 · https://dl.acm.org/doi/abs/10.1287/mnsc.22.8.868
- Modern acceleration: https://arxiv.org/abs/2510.26257

**Result.** To maximize `Num(x)/Den(x)` over feasible `x` with `Den(x) > 0`, define the parametric problem `F(λ) = max_x [Num(x) − λ·Den(x)]`. `F` is convex, strictly decreasing, and has a unique root `λ*`; `Num/Den` is maximized exactly at that root, with maximum value `λ*`. Newton iteration `λ_{k+1} = Num(x_k)/Den(x_k)` (where `x_k = argmax` of the inner problem) converges superlinearly / locally quadratically. **This is the engine that turns Jaccard maximization into a sequence of *linear* edge-selection problems.**

### 1.3 F-measure / plug-in threshold theory (same fractional-score family)
- Lipton et al., *Thresholding Classifiers to Maximize F1* — https://arxiv.org/abs/1402.1892 (optimal threshold = ½·F1\*).
- Waegeman et al., *On the Bayes-Optimality of F-Measure Maximizers* (JMLR 2014) — https://jmlr.org/papers/volume15/waegeman14a/waegeman14a.pdf
- Ye/Nan/Chai *et al.*, *Optimizing F-measure: plug-in vs structured loss* — https://www.researchgate.net/publication/289725154

**Why relevant.** These prove that for fractional metrics of the `TP/(…)` family, a *calibrated-probability plug-in rule with a single global threshold/parameter* is Bayes-consistent, and the threshold is a function of the achieved score — the F-measure analogue of our λ. Justifies "calibrate → one global λ → threshold" instead of end-to-end surrogate training.

### 1.4 Constrained lineage selection (the feasibility layer)
- Ultrack ILP (multi-hypothesis, division/appear/disappear constraints): https://arxiv.org/pdf/2308.04526
- Haubold, Jug, Hamprecht — *Generalized Successive Shortest Paths Solver for Tracking Dividing Targets* (MICCAI 2016): https://link.springer.com/chapter/10.1007/978-3-319-46478-7_35
- Coupled minimum-cost-flow cell tracking: https://link.springer.com/chapter/10.1007/978-3-642-02498-6_31

**Result.** Lineage tracking = min-cost flow / ILP with: node capacity 1, ≤1 incoming continuation edge, ≤2 outgoing (division), flow conservation, appear/disappear arcs. Division constraints are enforced by a residual-graph modification (Haubold) so successive-shortest-paths stays feasible. **We feed our Dinkelbach edge costs `−c_e` as arc costs into exactly this solver.**

### 1.5 Three-class calibration + prior/label shift
- Dirichlet calibration (natively multiclass): https://arxiv.org/pdf/1910.12656 · NeurIPS PDF https://papers.neurips.cc/paper/9397-beyond-temperature-scaling-obtaining-well-calibrated-multi-class-probabilities-with-dirichlet-calibration.pdf
- Calibration survey: https://arxiv.org/pdf/2112.10327
- Saerens, Latinne, Decaestecker, *Adjusting the Outputs of a Classifier to New a Priori Probabilities* (Neural Computation 2002): https://pubmed.ncbi.nlm.nih.gov/11747533/ · code https://github.com/aesuli/em-prior-adjust
- Alexandari, Kundaje, Shrikumar, *Maximum Likelihood with Bias-Corrected Calibration is Hard-to-Beat at Label Shift* (ICML 2020): https://arxiv.org/pdf/1901.06852
- One-vs-rest ROC-regularized isotonic: https://proceedings.mlr.press/v238/berta24a/berta24a.pdf

### 1.6 Distributional robustness (2 embryos → hidden embryo)
- Group-DRO / worst-group loss: https://arxiv.org/pdf/2007.13982
- Label-shift DRO: https://www.researchgate.net/publication/344878106
- Decision-focused worst-case shift: https://arxiv.org/pdf/2407.03557

---

## 2. The DERIVED objective for OUR metric

### 2.1 Reduction to a single global fractional program

Under one-to-one matching, each GT edge is matched at most once, so `TP_s + FN_s = G_s` (the fixed, unknown number of GT edges in sample `s`), giving `denominator_s = G_s + FP_s`. Using `w_s J_s = TP_s`, the volume-weighted average edge-Jaccard component becomes an exact identity (before the `max(0,·)` clip):

```
E_edge  =  [ Σ_s (1 − 0.1 Δ_s) · TP_s ]  /  [ Σ_s (G_s + FP_s) ] ,
          Δ_s = (N_pred_s − N_est_s) / N_est_s .
```

This is a **micro-averaged Jaccard modulated per-sample by the count penalty** — a single ratio, not an average of ratios. (The `max(0,·)` clip only binds when the penalty factor goes negative, i.e. `N_pred_s > 11·N_est_s`; keep every sample far from that and the clip is inactive — treat it as a feasibility guardrail, not part of the smooth objective.)

### 2.2 Decision variables and expectations

- `x_e ∈ {0,1}`: select candidate edge `e` (in sample `s(e)`).
- `y_n ∈ {0,1}`: node `n` is present in the prediction; `N_pred_s = Σ_{n∈s} y_n`.
- Calibrated three-class probs `pTP(e), pFP(e), pIG(e)`.

Taking expectations over the latent outcome of each selected edge (edges independent given features):

```
E[TP_s] = Σ_{e∈s} x_e · pTP(e) =: T_s
E[FP_s] = Σ_{e∈s} x_e · pFP(e) =: P_s
G_s      constant w.r.t. decisions.
```

Apply Nowozin's ratio-of-expectations approximation to the whole global ratio (exact in the large-volume concentration limit; error is `O(Var/mean²)` by the delta method — good for high-volume samples, weak for tiny ones, but volume-weighting already down-weights tiny samples):

```
        Σ_s (1 − 0.1 Δ_s) · T_s
E_edge ≈ ───────────────────────── ,      Δ_s = ( Σ_{n∈s} y_n − N_est_s ) / N_est_s .
          Σ_s ( G_s + P_s )
```

Full program (add the division term, §2.6):

```
maximize_{x,y ∈ Feasible}   E_edge(x,y)  +  0.1 · E_div(x,y)
subject to  lineage constraints on (x,y)  (§2.5),  and clip guardrail N_pred_s ≤ N_est_s·(1+β).
```

**Assumptions.** (A1) Edge outcomes conditionally independent given features (needed for `E[TP]=Σ pTP`; correlated errors inflate variance but not the mean). (A2) One-to-one GT↔pred matching so `TP+FN=G_s` is constant. (A3) `pTP,pFP,pIG` are well-calibrated OOF and prior-corrected for the hidden embryo. (A4) Ratio-of-expectations proxy (large-volume). (A5) `N_est_s` is a trustworthy anchor (organizer-supplied).

### 2.3 Dinkelbach iteration

Introduce global `λ ≥ 0` (current estimate of `E_edge`). Solve, dropping the constant `−λ Σ_s G_s`:

```
F(λ) = max_{x,y}   Σ_s (1 − 0.1 Δ_s) · Σ_{e∈s} x_e pTP(e)   −   λ · Σ_e x_e pFP(e)
Update:  λ ← [ Σ_s (1−0.1Δ_s) T_s ] / [ Σ_s (G_s + P_s) ]
Stop when F(λ) ≈ 0  (⇒ λ = optimal objective value).
```

`G_s` appears **only** in the λ-update denominator and only as a constant additive term. In practice you do **not** run the λ-update at test time on the hidden embryo (you can't, `G_s` is hidden): instead you **fix λ at its OOF-calibrated value** (§3.4). This is legitimate precisely because `λ*` = achieved Jaccard value, a scale-free ratio you *can* estimate on labelled folds and it transfers.

### 2.4 The per-edge linear coefficient and the per-node count cost

Hold `Δ_s` (hence the factor `φ_s := 1 − 0.1 Δ_s`) and `T_s` at their current-iterate values (block-coordinate on the bilinear count term). Then the objective is **linear in `x`** with coefficient

```
   c_e  =  φ_s · pTP(e)  −  λ · pFP(e) ,     φ_s = 1 − 0.1 (N_pred_s − N_est_s)/N_est_s .
```

Select `e` iff `c_e > 0` (subject to constraints). Interpretation: **λ = "how many units of expected counted-FP one unit of expected TP is worth"**, the shadow price; `φ_s` up-weights TP when you are below the count target and down-weights it above.

**Per-node count cost (closed form).** The count penalty couples every node in `s` to the whole sample's TP mass. Differentiating the numerator term `−0.1 (Σ_n y_n / N_est_s) · T_s` w.r.t. adding one node:

```
   κ_s  =  0.1 · T_s / N_est_s  =  0.1 · E[TP_s] / N_est_s .        (per-node Lagrangian tax)
```

**Marginal-value rule for a NODE** `n` (with best incoming edge `e_in` and its outgoing/division edges `E_out(n)`):

```
 ΔObj(n) = φ_s·( pTP(e_in) + Σ_{e∈E_out} pTP(e) ) − λ·( pFP(e_in) + Σ pFP(e) ) − κ_s .
 Add node n  ⇔  ΔObj(n) > 0.
```

Consequences:
- A **purely ignored** node/edge has `pTP=pFP=0` ⇒ `ΔObj = −κ_s < 0` ⇒ excluded. **The count cost is the sole mechanism excluding ignored edges** (they never hurt Jaccard, only inflate `N_pred`).
- Because `T_s` grows and `φ_s` shrinks as you add nodes, `ΔObj(n)` is **decreasing** in `N_pred_s` — diminishing returns give a natural stopping point.
- Below `N_est` (`Δ_s<0`) the factor `φ_s>1` rewards adding nodes; above it (`φ_s<1`) it taxes them. The equilibrium sits near **`N_pred_s ≈ N_est_s`**, i.e. the metric *hands you the count target*. Use `Σ_{n∈s} y_n = N_est_s` as a strong prior / near-equality constraint and let `c_e` decide *which* nodes.

### 2.5 Constraints (feasibility layer)
Standard lineage ILP / min-cost-flow-with-branching:
- node capacity: each node used ≤ 1;
- continuation: ≤ 1 incoming continuation edge per node;
- division: ≤ 2 outgoing child edges; a division event requires exactly 2 selected children;
- flow conservation with appear/disappear arcs;
- count budget: `Σ_{n∈s} y_n ≤ B_s` (set `B_s = N_est_s`) — enforced *either* as a hard knapsack constraint *or* implicitly via the tax `κ_s` (Lagrangian relaxation; sweep the multiplier to hit `N_pred=N_est`).

Feed `−c_e` (plus `κ_s` on node-open arcs) as arc costs into the Haubold successive-shortest-paths dividing-target solver, or solve the ILP directly (Ultrack-style) for small crops. The bilinearity (via `φ_s`, `κ_s`, `T_s`) is handled by the **outer Dinkelbach + block-coordinate loop**: solve linear min-cost-flow with fixed `φ_s,κ_s,λ`; recompute `Δ_s, T_s`; refresh `φ_s,κ_s`; update `λ`; repeat to convergence (typically 3–6 iters).

### 2.6 Division term (the `+0.1·division_jaccard`)
Divisions have their *own* Jaccard `J_div = TPdiv/(TPdiv+FPdiv+FNdiv)` with its own three-class calibration `pTPdiv, pFPdiv, pIGdiv` per candidate division event (a node with 2 selected children). Because it is **added**, the full objective is a **sum of two ratios** `A/B + 0.1·C/D`. Sum-of-ratios is NP-hard in general, but with two ratios use **two Dinkelbach parameters** `(λ, λ_div)`:

```
F(λ,λ_div) = max_{x,y}  Σ_s φ_s Σ x_e pTP(e) − λ Σ x_e pFP(e)
                       + 0.1 [ Σ z_d pTPdiv(d) − λ_div Σ z_d pFPdiv(d) ]
```

with `z_d∈{0,1}` the division indicators coupled to the two child edges, both λ's updated by their own ratio. Practical caveats: division counts are **small** ⇒ `J_div` is high-variance and the ratio-of-expectations proxy is weak; be **conservative** — only declare a division when both child edges have high `pTP` *and* `pTPdiv(d)` is high, because a false division costs a division-FP *and* corrupts two edges. The 0.1 weight means divisions are a tie-breaker, not the main prize.

---

## 3. Three-class calibration recipe (TP / counted-FP / ignored)

### 3.1 Labelling OOF candidates
For each candidate edge produced by the base tracker on a labelled embryo, assign a ground-truth *outcome* class using the sparse GT:
- **TP** if it matches an annotated GT edge (both endpoints within 7 µm, respecting the one-to-one matcher);
- **counted-FP** if both endpoints lie inside an annotated region/neighbourhood but it matches no GT edge;
- **ignored** if it lies (partly) in an unannotated region.
The annotated-region mask is what separates counted-FP from ignored — build it explicitly from the GT node footprint + a 7 µm dilation. This masking definition is the single most consequential modelling choice; sanity-check it against a few known counted-FP and known-ignored edges.

### 3.2 Fit a native 3-class calibrator, out-of-fold
- Base model outputs a score vector per candidate → fit a **multiclass calibrator**: **Dirichlet calibration** (multinomial logistic on `log`-probs, with ODIR off-diagonal regularization — strong regularization is mandatory with only 2 embryos) or **one-vs-rest isotonic** (lower ECE per the Dirichlet paper but needs more data). Temperature/vector scaling is the low-data fallback.
- **Leave-one-embryo-out**: fit the calibrator on embryo A, apply to embryo B and vice-versa. This yields genuinely OOF `(pTP, pFP, pIG)` per candidate — never calibrate and evaluate on the same embryo (2-sample overfit is severe).

### 3.3 Correct for the hidden embryo's prior shift (label shift)
The hidden embryo differs mainly in **annotation density and cell density**, which shifts the *class priors* `π=(πTP,πFP,πIG)` (especially the ignored fraction) even if the per-candidate *likelihoods* are stable — i.e. a label-shift, `P(feature|class)` fixed, `P(class)` changed. Use **Saerens EM / MLLS** (Alexandari 2020) to re-estimate the test priors from the classifier's own outputs on the hidden embryo, unsupervised:

```
EM fixed point (per class k):
  ŵ_k^(t+1) = (1/M) Σ_i  [ ŵ_k^(t) p_k(x_i) ] / [ Σ_j ŵ_j^(t) p_j(x_i) ]
  ŵ_k = π'_k / π_k   (source→target prior ratio),  init ŵ=1
Re-weighted posterior used everywhere downstream:
  p'_k(x) = ŵ_k p_k(x) / Σ_j ŵ_j p_j(x) .
```

Alexandari et al. show **bias-corrected calibration + this MLLS EM is hard-to-beat** for label shift. Apply it before computing `c_e, κ_s`. Because `N_est` is given, you also have a *supervised* anchor on the ignored/present split — use it to constrain the EM (the TP+FP present-mass should be consistent with `N_est`).

### 3.4 Estimate λ and κ that transfer
- **λ**: on each held-out embryo, run the Dinkelbach loop *with* the known GT (you can compute the true ratio) to find the `λ` that maximizes the *actual* metric on that embryo. Alternatively λ = the achieved Jaccard value at optimum. Take the **two per-embryo λ's** and, for robustness (§5), deploy `λ_deploy = max(λ_A, λ_B)` or the upper confidence limit (higher λ ⇒ more FP-averse ⇒ safer under-prediction).
- **κ** is *not* free-estimated: it is `0.1·T_s/N_est_s`, recomputed each outer iteration from current `T_s`. The only tunable is the node-budget multiplier if you enforce `N_pred=N_est` by Lagrangian sweep.
- Both λ and the threshold on `c_e` are **ratios / dimensionless** → they transfer across embryos far better than any absolute count would. This is why you never need `G_s`.

---

## 4. Findings table

| Result / method | Source URL | What it gives us | First experiment + cheap kill-gate | Cost |
|---|---|---|---|---|
| Ratio-of-expectations proxy for expected Jaccard/IoU + parametric threshold sweep | https://openaccess.thecvf.com/content_cvpr_2014/papers/Nowozin_Optimal_Decisions_from_2014_CVPR_paper.pdf | Justifies `E[J]≈E[TP]/E[den]` and a single-parameter threshold decision rule | Sweep a scalar threshold on `c_e` on one embryo vs. per-edge MAP baseline. **Kill-gate:** if sweep never beats MAP metric, the proxy/independence assumption is broken for this data | Hours |
| Dinkelbach parametric fractional programming | https://pubsonline.informs.org/doi/10.1287/mnsc.22.8.868 | Exact `λ`-iteration converting Jaccard-max into linear edge selection; superlinear convergence | Implement `F(λ)=max Σ c_e x_e`, Newton update on OOF embryo. **Kill-gate:** if `λ` oscillates / doesn't converge in <10 iters, denominator sign or `Den>0` assumption violated | Hours |
| Volume-weight identity `w_s J_s = TP_s` ⇒ single global ratio | this report §2.1 | One global λ instead of per-sample; big simplification | Verify numerically on labelled embryos that weighted-avg equals `ΣTP(1−.1Δ)/Σ(G+FP)` | <1 day |
| Per-node cost `κ_s=0.1 E[TP_s]/N_est_s` + `N_pred≈N_est` anchor | this report §2.4 | Closed-form node budget; a robust count target | Fix `N_pred=N_est` as constraint; compare metric to free `N_pred`. **Kill-gate:** if forcing `N_pred=N_est` loses >1–2% metric, the penalty is not the binding term | 1 day |
| Lineage ILP / min-cost flow with dividing targets | https://link.springer.com/chapter/10.1007/978-3-319-46478-7_35 · https://arxiv.org/pdf/2308.04526 | Feasible selection under ≤1-in / ≤2-out / conservation with `−c_e` costs | Plug `−c_e` into existing Ultrack/flow solver on a crop. **Kill-gate:** infeasible or worse than greedy ⇒ constraint encoding bug | 1–3 days |
| Dirichlet / OvR-isotonic 3-class calibration | https://arxiv.org/pdf/1910.12656 | Calibrated `(pTP,pFP,pIG)`; the whole objective depends on these | Reliability diagrams + ECE per class, leave-one-embryo-out. **Kill-gate:** ECE not improved over raw scores ⇒ recheck 3-class labelling/mask | 1–2 days |
| Saerens EM / MLLS label-shift prior correction | https://pubmed.ncbi.nlm.nih.gov/11747533/ · https://arxiv.org/pdf/1901.06852 | Corrects ignored-vs-FP prior on hidden embryo without labels | Run EM on embryo B using calibrator from A; compare corrected vs uncorrected metric. **Kill-gate:** EM diverges or worsens B ⇒ shift is not pure label-shift | 1 day |
| F-measure plug-in threshold theory (consistency, threshold=f(score)) | https://jmlr.org/papers/volume15/waegeman14a/waegeman14a.pdf · https://arxiv.org/abs/1402.1892 | Theoretical guarantee that "calibrate → global λ threshold" is Bayes-consistent for fractional scores | Conceptual check; no separate experiment | — |
| Group-DRO / min-fold robust selection | https://arxiv.org/pdf/2007.13982 · https://arxiv.org/pdf/2407.03557 | Worst-case λ/prior over the 2 embryos for hidden-embryo safety | Deploy `max(λ_A,λ_B)` vs mean; check the *worse* fold's metric. **Kill-gate:** if robust choice tanks the good fold, uncertainty set is too large | 1 day |

---

## 5. Failure modes and the robust variant

**Where estimating on 2 embryos breaks:**
1. **λ / prior overfit.** Two embryos ⇒ 1 effective OOF point per fold; λ and priors are high-variance. *Fix:* min-fold / worst-case deployment — `λ_deploy = max(λ_A, λ_B)` and priors chosen as the worst case within an f-divergence (KL) ball around the pooled empirical `π̂`. Formally a DRO objective `max_{x,y} min_{π∈U, λ∈[λ_lo,λ_hi]} objective(x,y;π,λ)`; with box/KL uncertainty this reduces to using the FP-pessimistic corner (higher λ, higher πFP).
2. **Annotation-density shift dominates.** The ignored fraction `πIG` is the least stable quantity; if the hidden embryo is more sparsely annotated, uncorrected `pFP` is over-stated and you under-predict. *Fix:* Saerens/MLLS EM (§3.3), anchored by `N_est`.
3. **Ratio-of-expectations proxy fails on small-volume samples** (delta-method error `~Var/mean²`), and on **divisions** (tiny counts). *Fix:* volume-weighting already down-weights small samples for the edge term; for divisions, be conservative (raise `λ_div`, only fire high-confidence divisions) and cap their influence (weight is only 0.1).
4. **Correlated edge errors** violate independence (A1) — a whole mis-tracked lineage flips together, so realized `TP` variance ≫ Poisson. *Fix:* don't trust per-sample `J` point estimates; optimize the *global* ratio (averages out) and validate with block-bootstrap over lineages, not edges.
5. **`max(0,·)` clip / over-prediction cliff.** Pushing `N_pred` far above `N_est` can drive the factor negative and zero a whole sample. *Fix:* hard guardrail `N_pred_s ≤ N_est_s·(1+β)` with small `β`, plus the natural `N_pred≈N_est` equilibrium.
6. **Block-coordinate (φ_s, κ_s, λ) non-convergence** on the bilinear count term. *Fix:* damp the `Δ_s` update, or solve at fixed `N_pred=N_est` (removes the bilinearity — `φ_s=1`, `κ_s` constant).

**Recommended robust deployment:** three-class Dirichlet calibration (leave-one-embryo-out) → MLLS EM prior correction on the hidden embryo anchored to `N_est` → Dinkelbach with `λ_deploy = FP-pessimistic corner` → min-cost-flow-with-branching at the hard count target `N_pred_s = N_est_s` → conservative division firing. Scale-free knobs (λ, thresholds) + organizer-given count anchor (`N_est`) are the load-bearing robustness, not any absolute estimate.

---

## 6. Open questions (need confirmation from metric code / organizer)
1. **Matching cardinality:** is GT↔pred strictly one-to-one (needed for `TP+FN=G_s` constant)? If many-to-one, the denominator reduction changes.
2. **Ignored semantics:** does an ignored edge truly contribute 0 to *both* numerator and Jaccard denominator, but still +1 to `N_pred`? (My whole "count-penalty is the only discipline on ignored edges" argument hinges on this.)
3. **Node vs edge counting:** is `N_pred` a *node* count or effectively an *edge/track* count? `κ_s` derivation assumes nodes; confirm the definition of `N_est`.
4. **`N_est` granularity/trust:** is `N_est` per-sample, per-frame, and how noisy? The robustness backbone assumes it is trustworthy.
5. **Division-Jaccard matching:** does `division_jaccard` use the same 7 µm / one-to-one matching, and does it include a division-ignore class over unannotated regions?
6. **Sample = embryo or crop?** The volume-weight identity and one-global-λ claim assume samples are the units of the weighted average; confirm the aggregation granularity.
7. **Is the ratio-of-expectations gap material at your volumes?** Worth a one-off Monte-Carlo: sample `Y`, compare true `E[J]` vs proxy, to size the approximation error before trusting it.

---

*Primary sources cited inline; all URLs verified reachable as of 2026-07-12 except the Nowozin CVF PDF (open-access, occasionally rate-limits WebFetch but the HTML abstract page and IEEE/ResearchGate mirrors resolve).*
