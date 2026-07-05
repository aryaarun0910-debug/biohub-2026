# WIN_PLAN — the roadmap to win Biohub Cell Tracking (not place)

**Written 2026-07-05. Deadline 29 Sep 2026.** North-star strategy doc. The objective is **1st on the
PRIVATE leaderboard** (the hidden, disjoint embryo — the only board that pays), via the **long game**:
build the maximal, honestly-validated generalizer, gated on our clean out-of-fold (OOF) metric, and
submit **once**, when the OOF says we dominate — never a public-board stunt.

## 0. The one idea everything follows from
The **public LB is a leakable 4-movie board** (visible test IDs have labeled train copies). The **private
LB is a disjoint embryo the model never saw.** Therefore **winning is a GENERALIZATION contest** — whoever's
model degrades least on an unseen embryo wins. Every decision below serves generalization, measured on our
embryo-held-out OOF (our proxy for the private board), NOT the public score.

## 1. Where we stand (measured, honest — 2026-07-05)
- **Detection is essentially solved:** 90–95% node recall on *held-out* embryos. The hard part is done.
- **Clean greedy OOF baseline:** fold-0 (held-out 44b6) 0.656, fold-1 (held-out 6bba) 0.559, **min-fold 0.559**.
- **ILP proven as a big lever (local):** fold-1 0.559 → **~0.67** (+0.11), and it kills the greedy division
  catastrophe (10,786 FP → ~1–28/crop). Banked, not yet run at full scale on Kaggle.
- **Full-length fold-1 retrain running** (45ep/1000it ≈ 1.9× the budget model) — first read on the training lever.
- Pipeline validated end-to-end (train→predict→score), exact scorer hardened (`biotrack.metric`), preflight rule in place.

## 2. Theoretical ceiling
Metric = `weighted_avg(adj_edge_jaccard) + 0.1 · division_jaccard`.
- **Hard max ≈ 1.1** (perfect edges 1.0 + perfect divisions 1.0). Marginally higher is *possible* by exploiting
  the ONE-SIDED count penalty (`adj = J·(1 − 0.1·(N_pred−N_est)/N_est)`, no upper clip → under-count with
  perfect edges scores >1.0).
- **Realistically achievable ≈ 0.90–0.95** — capped by 3D+t noise, sparse/imperfect GT annotation, and the
  disjoint-embryo gap.
- **Winning private score: unknown, likely below the 0.896 public top** (public = easy seen-movies; private = hard).

## 3. The levers, ranked by expected OOF gain (each gated on OOF, both folds)
| # | Lever | Why it matters (esp. for the PRIVATE/disjoint embryo) | Status |
|---|---|---|---|
| 1 | **ILP linking, tuned** | +0.11 proven; sweep division-weight, add 1-frame gap recovery. Biggest banked lever. | proven local; scale on Kaggle (cap giant-crop ILP to avoid OOM) |
| 2 | **NIS3D / external pretraining** | THE differentiator — pretrain the detector on dense external zebrafish/embryo nuclei (CC-BY) so it doesn't overfit our 2 embryos. Most teams won't. This is what wins a disjoint board. | not started |
| 3 | **Proper full-length training** | Out-train the public 50ep; more epochs + augmentation. | fold-1 running |
| 4 | **Divisions as a weapon** | The 0.1 term is ~unexploited by the field. Precision-gated division detection = near-free points once ILP kills the FPs. | ILP fixes FP; need TP |
| 5 | **Ensemble + TTA** | fold-A ∪ fold-B, flip-TTA (already in code). Marginal alone, decisive at the very top. | not started |
| 6 | **Metric-quirk exploits** | Sub-voxel centroid refine (free recall at the 7µm gate); count-calibration exploiting the one-sided penalty; edges between two unmatched nodes are FREE (not FP). | identified |
| 7 | **An orthogonal edge** | A lever nobody else uses (metric structure, cross-domain association). High variance, high upside. | lateral-sweep hunt |

## 4. Execution order
1. **Land the ILP at scale** — memory-safe ILP (fall back to greedy only on the handful of giant crops),
   full-fold OOF for both folds. This converts the proven +0.11 into a real min-fold number.
2. **Read the training lever** (fold-1 retrain, running) → decide how hard to push epochs.
3. **NIS3D pretraining** — the generalization weapon. Pretrain detector → fine-tune both folds → OOF.
4. **Tune divisions + count-calibration** on frozen detections (cheap, OOF-gated).
5. **Ensemble + TTA + sub-voxel refine** — final polish.
6. **ONE calibration submission** at some point (not a stunt): confirm OOF↔LB tracks for the learned stack
   before betting everything (we have zero calibration for it). Then the final dominant submission.
RULE: bank no lever until it lifts the CLEAN embryo-held-out min-fold OOF (report public_TEST4 and OOF separately).

## 5. Compute division of labor
- **Kaggle T4×2 (free) = the 3D GPU workhorse:** all detector/edge training + inference. One fold per session
  at bs=1; pin `machine_shape: NvidiaTeslaT4`; validate every kernel locally on CPU first (preflight).
- **Local box = the CPU lab, NOT the GPU:** the MX350 (2 GB) is too small to train or reliably infer the 3D
  model — it sits this out. The local machine's value is the **exact metric scorer, the ILP solves (SCIP is
  CPU — that's how we got +0.11), OOF validation, and fast preflight experiments.**

## 6. Honest odds + discipline
- No one can promise 1st among 600+ teams with a private shakeup. Realistic top-tier odds **~15–25%**, and they
  **rise with each OOF-validated gain.** We maximize win-probability by stacking every honest edge above.
- **Discipline that wins the long game:** optimize the OOF, not the public board; submit sparingly; the final
  submission goes in only when the OOF dominates the field on the disjoint-embryo proxy.

## 7. DO-NOT (prize/DQ risk)
- Do NOT optimize or believe the leakable public board; report OOF separately, always.
- Do NOT build on the quarantined evaluator division defect (host-clearance required).
- External public data/weights ARE allowed (rules cleared); NIS3D CC-BY clean; PAC-MAP NonCommercial = avoid in
  a prize submission; Zebrahub same-lab = leakage risk (no label transfer into a suspected test source).
