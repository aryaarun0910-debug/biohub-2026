# Red-team: blind spots, artifacts, and premature kills — 2026-08-17

Adversarial re-examination of our own conclusions. Deployed = **P3 harmonic, public 0.915**;
leader 0.950, top-3 0.948. Every critique below carries a concrete cheapest experiment and a
verdict distinguishing **LIKELY WRONG** from **WORTH A CHEAP CHECK** from **SOLID (do not reopen)**.

New measurements run for this report (CPU, exact patched scorer):
- Identified the locally-scoreable OOF substrates. `artifacts/kaggle/oof_clean` and
  `artifacts/kaggle/e0c_cache` are the **E0c** substrate (over-predicting: 44b6 pooled ratio
  **+0.094**, multiplier 0.991; score anchors 0.7595/0.6490). `artifacts/kaggle/p0strict_cache/graphs/{0,1}`
  is the **deployed-adjacent P0-strict** substrate (all 199 crops; 44b6 pooled ratio **−0.274**,
  multiplier **1.027** — already at OR BELOW public-plateau calibration, i.e. more aggressively
  pruned than the plateau's −0.10). So on 44b6 there is no multiplier headroom left; any residual
  is on 6bba (node_recall 0.855 vs 0.985 on 44b6). The two substrates are genuinely different
  pipelines, not the same graphs: same crop `44b6_0113de3b`, E0c ratio +0.094 vs P0-strict −0.021.
- Ran a per-family weakest-component-first node-budget sweep directly on P0-strict (see §2/§6).

---

## Claim 1 — "Divisions are a minority of the gap and the division pair-ranker is dead."

**VERDICT: SOLID.** The pair-ranker is dead and it was tested on BOTH families, not just n=16.

- The task's premise that laneD "used n=16 mothers (44b6)" is only half right. laneD
  (`h1t_conditional_ranker.py`, lab-notebooks 2026-07-31) conditioned on the 92 mothers whose
  true pair survived the top-3 shortlist: **16 in 44b6 AND 76 in 6bba**. On BOTH, the learned
  MLP/linear/GBDT **lost to frozen geometry** (44b6 14/16 geometry vs 13/16 MLP; 6bba 58/76 vs
  55/76). McNemar p≥0.39.
- **Structural, not threshold, closure:** in 44b6 **b=0** — there is not one mother where the MLP
  ranks the true pair first and geometry does not. The MLP's correct set is a STRICT SUBSET of
  geometry's, so the union oracle over any blend/gate/ensemble/cascade = geometry alone (14/16).
  No operating point can rescue it.
- **Within-mother vs global framing:** the within-mother rank is the SECOND factor
  `P(pair | mother divides)` and it is NOT the bottleneck. D-01 shows within-mother rank carries
  zero information about WHICH mother divides (every mother has a rank-0 pair). The FIRST factor
  `P(mother divides)` is where it dies: **0/100 true dividers in the top-100 mothers, both
  families**. A perfect second factor × a zero first factor = zero. A perfect conditional ranker
  adds at most 2 (44b6) / 18 (6bba) divisions and only on mothers the first factor admits (~0).
- **Could a DIFFERENT mechanism clear 4.07%/6.38% break-even?** Arithmetically almost impossible.
  Base rate of true forks among metric-visible candidates is ~0.1% (8/7,514; 38/37,591). Reaching
  4–7% precision at 50% recall is a 40–70× lift. Required classifier AUC at these base rates
  (measured across three independent mechanisms): flat-mother **0.983–0.9992**, branch-emergence D1
  **0.9695**, conditional critic **0.9386**. Best measured anywhere: flat-mother 0.86–0.92,
  appearance-at-t-1 LOEO **0.719**, D1 0.856/0.777. Every attempt is 0.1–0.2 AUC short of the bar.

**The one genuine crack (WORTH A CHEAP CHECK, low expected yield):** nobody has cross-fit the
STRONGEST available mother-gate features together — appearance-at-t−1 (DB-05/06, +0.05–0.08 AUC,
LOEO 0.719) stacked with the geometric midpoint-residual — and read its precision-at-50%-recall on
the metric-visible candidates. Cheapest experiment: fit a LOEO logistic/GBM on the existing
`phaseb_h1g_features` + h0c candidate cache, report precision at TP=16/76. **Predicted result:
< 1% precision (fails the 4–7% bar by >5×).** Arithmetic says dead; this closes the last door.
Cost ~30 min CPU.

**Caveat that cuts the OTHER way (do not use n=16 to REOPEN):** D-03 is right that 44b6's 16
positives give ±0.0086 SD — but that means 44b6 cannot CONFIRM a small POSITIVE, not that division
is under-exploited. 6bba (76 positives, SD 0.0039) is well-powered and also negative.

---

## Claim 2 — "Node-calibration (the (1−0.1r) multiplier channel) is already captured in P3, worth ~+0.02–0.03."

**VERDICT: MOSTLY RIGHT, but the number is MISread and there is a real per-family residual to check.**

- The **+0.02–0.03 is the E0c→plateau gain, ALREADY BANKED** in the deployed pipeline, NOT live
  headroom on P3. Direct measurement this session: E0c over-predicts (pooled ratio +0.17,
  multiplier ~0.98); P0-strict/deployed already sits at plateau calibration (ratio −0.10 to −0.17,
  multiplier ~1.01+). RT-04's headline "our whole deficit to the plateau is the node-count
  multiplier" is an **E0c** statement about the 0.889→0.913 route we already travelled. Reading
  RT-05's "+0.08 channel capacity" as recoverable on P3 would be the error.
- **BUT the channel is not provably closed on the deployed substrate per-family.** A 20-crop
  P0-strict pilot (weakest-component-first, division-protected, deployment-observable) gives:
  keep_frac 0.975 → pooled **+0.0024**, **6bba +0.0031**, 44b6 flat (best at 1.00). The
  preregistered kill rule ("optimum ≥0.99 on BOTH families") would **NOT trigger** here. The
  full-199 result is in §6.
- Whether that residual is real or **selection optimism** (D-04: best-of-T over the same two
  families manufactures ~+0.010 at T=5; here T=7 keep-fracs) is the open question. Even if real it
  is ~+0.002–0.003, below the +0.005 bar, and concentrated on the fold that is 85% of edge mass.
- The **deeper** multiplier headroom RT-05 describes (drive ratio from −0.10 toward −0.42 by
  deleting metric-invisible nodes) is NOT node-BUDGET; it is **learned FP-suppression** — the same
  `bet-learned-ranker`/retrain lever. It cannot be realized by length-ranked pruning because the
  deployed weakest components are already correct tracks (q_net +1.0 on P0-B).

**Cheapest experiment (settles it):** the full P0-strict per-family node-budget sweep (§6, running).
If 6bba optimum ≥0.99 → channel closed on the deployed substrate, RT-04/05 headroom fully banked.
If 6bba optimum <0.99 with a robust positive → a small min-fold trim was closed on pooled/E0c
evidence and deserves a preregistered single-point confirmation.

---

## Claim 3 — "The remaining ~0.029 to the leader is not attributable to any recoverable PUBLIC mechanism."

**VERDICT: SOLID as a conclusion; one framing flaw and one under-scoped mechanism.**

- The elimination (RT-01/02 hub-fork negative; RT-09 foreign detectors falsified 0.36–0.51;
  RT-07 plateau = one 50-epoch weights pack; A-20 threshold tuning = noise) is corroborated
  INDEPENDENTLY by the live 2026-08-16 competitive sweep: adj_edge_J plateaus 0.90–0.91, div_J≈0
  for nearly everyone, so the leader edge is a **retrained/generalising model** on external
  Zebrahub (host-opened 2026-08-13) + synthetic division supervision. The pivot to H1 retrain is
  well-founded.
- **Framing flaw in RT-12:** its decomposition anchors on "plateau implies raw J≈0.9037" and
  assigns 40–60% of the gap to node-count/detection-precision. That share is the **E0c→plateau**
  portion — already ours. For the DEPLOYED substrate the multiplier is already calibrated, so the
  0.915→0.950 gap is **almost entirely the edge term (better weights) + divisions**. Do not use
  "node-count is 40–60% of the recoverable gap" to justify more post-hoc pruning on P3.
- **Under-scoped public mechanism — multi-seed / multi-checkpoint ENSEMBLING.** The whole field
  shares ONE pack (RT-07). Nobody has trained N independent seeds and blended logits. This is a
  legitimate raw-Jaccard lever the field has NOT swept (edge-TTA hurts is a DIFFERENT thing:
  augmented VIEWS of one model, not independent models). It requires GPU retraining, so it folds
  into `bet-zebrahub-retrain`/`bet-learned-ranker` rather than being a cheap CPU win — but it should
  be listed as an explicit arm of the retrain bet, not treated as closed by "TTA hurts."
- Calibration, ILP tuning, gap-repair, TTA: correctly dismissed (all measured negative/noise).

**Cheapest experiment:** when the H1 pilot trains its first retrained checkpoint, also save a second
seed and measure the 2-seed logit-blend LOEO delta — near-free once one retrain exists; do NOT
retrain a second model solely for this before the first clears its falsification.

---

## Claim 4 — Substrate confounds: which E0c conclusions do not transfer to the deployed substrate?

**VERDICT: THE BIGGEST SYSTEMIC RISK. Nearly every "positive lever" we hold is an E0c number.**

Confirmed this session — the locally-scoreable "positive" levers are all measured on E0c or a
non-deployed arm:

| lever | headline | substrate | deployed? |
|---|---|---|---|
| motion-gate (arm B) | **+0.0088 pooled, +0.0074 min-fold, P(d>0)=1.0** | **E0c** (claims-table flags "NOT a deployment claim") | NO — P3+armB kernels built, GPU scoring PARKED |
| suppress-all | +0.0027 bilateral | E0c; ports to P0-strict as **+0.0017 with 44b6 NEGATIVE**, GT-free control **−0.00004/−0.00204** | NO — P3 = harmonic only |
| node-budget arm A | +0.00157 | E0c (over-predicting); P0-B direct −0.00001 | NO |

- The single strongest un-shipped lever, **motion-gate**, has its +0.0088 on E0c and its
  deployment-substrate confirmation PARKED behind a GPU green-light. We may be wrong that
  "P3-harmonic-alone" is our best deployable: **P3+armB has stronger prior evidence than harmonic
  and is already built.**
- **Reverse asymmetry on the deployed substrate:** P0-strict has the WORST 6bba division
  reachability of any substrate — **66/125** (E0c 93, clean903 96, v122 68); pooled reach
  88/151 = 58.3% vs E0c 74.8%. The aggressive pruning that BUYS the good node calibration THROWS
  AWAY 27 of 93 E0c-reachable 6bba divisions, and 6bba is 85% of edge mass. So the division
  ceiling on the DEPLOYED substrate is materially below the E0c-measured +0.06 oracle. Suppress-all
  +0.0027 measured on E0c (where 44b6 div TP=0) does NOT hold on P0-strict (44b6 div TP=2) — already
  caught (D-11) but worth re-stating as the template failure.

**Cheapest experiments:**
1. **Score the two already-built P3+armB LOEO-export kernels** (the only blocker is one parked
   GPU inference run; scoring the exported graphs is CPU via `score_oof.py`). This is the
   highest-upside item in the whole report: it either promotes a +0.007 deployable or kills the
   strongest E0c lever on the real substrate. Falsification: either fold ≤ its P3-alone anchor.
2. Re-run suppress-all GT-free on P0-strict per-family (cache present) to confirm it stays
   ≤0 before any thought of shipping it.

---

## Claim 5 — Metric-scoring subtleties: any legitimate exploitable asymmetry we under-use?

**VERDICT: No NEW legitimate asymmetry beyond FP-suppression; the one real asymmetry IS the
learned-ranker lever. (The patched hub-fork hack is not proposed.)**

- The only load-bearing asymmetry is the one `concepts-and-definitions.md` already states: because
  annotation is 0.8–5.4% dense, deleting a predicted node that carries no metric-visible edge
  cannot lower edge TP, can only lower counted FP, and raises the unclipped multiplier. That is
  exactly RT-05's channel and it reduces to **detection precision** — rank "false detection" below
  "true-but-unannotated cell." No free lunch: random deletion is provably negative (d(adjJ)/df<0).
- division_jaccard scoring only on annotated cells, the 7 µm bipartite one-to-one match, and
  ignored off-annotation edges are all correctly characterised; none yields a legitimate lever we
  aren't already chasing. The bipartite one-to-one property is the mechanism that KILLS candidate
  breadth (extra nodes steal true matches, branch A −0.16), not one we can exploit.

**Cheapest experiment (mechanism check, transfers):** on the E0c OOF geffs (which carry
`edge_prob`), sort predicted nodes by detector/edge confidence and measure the metric-visible
fraction by decile (RT-05's own falsification). If the bottom decile is NOT enriched for
metric-invisible nodes, confidence-ranked pruning cannot beat length-ranked pruning and the
FP-suppression channel needs a LEARNED signal (not a threshold) — which is what we already believe.
~20 min CPU.

---

## Claim 6 — Are any "closed" levers closed on flawed evidence?

**VERDICT: One plausible case (node-budget per-family on the deployed substrate); the rest are
soundly closed.**

- **`bet-node-budget` "lost — optimum = no pruning" is the weakest closure in the graveyard.**
  The E0c corpus sweep (`node_budget_sweep.json`, 199 crops) actually shows keep_frac 0.975 →
  pooled **+0.00157** (6bba +0.0022, 44b6 −0.0021). The "closed on the deployed substrate" verdict
  rests entirely on `p1_node_budget_p0b_substrate.json`, which by its own header used
  **"FOUR VISIBLE PLACEHOLDER MOVIES ONLY … keep_frac inherited FROZEN … no sweep was run here"**
  and warns the proxy **"is NOT the leaderboard and cannot predict it … biased in the node
  budget's favour."** That is a selection substrate CLAUDE.md rule 2 explicitly forbids, at one
  frozen point, with no per-family sweep. The P0-strict per-family pilot (§2) shows a 6bba-specific
  optimum at keep_frac 0.975; the full 199-crop per-family sweep (§6) is the FIRST proper test on
  the deployed-adjacent substrate. Even if reopened, the ceiling is ~+0.002–0.003 (below the
  +0.005 bar) and the E0c decomposition says ~116% of it is count-multiplier, ~−16% edge quality.
- Scalar re-acceptance (−0.023), GT-free component selector (−0.008, but GT-oracle +0.0087 → the
  headroom is real and the CONSTRAINT is selection, i.e. the learned-ranker bet), split/merge
  (+0.0006), edge-TTA (0.885), branch-A breadth (−0.16, structural via one-to-one matching),
  synthetic-division transfer (synth→real AUC 0.664): all soundly closed with mechanisms, not
  operating points.
- **Not "closed on flawed evidence" but "OPEN with un-deployed evidence":** the motion-gate
  (§4). That is the inverse risk and the bigger EV miss.

---

## §6 — Live experiment result (node-budget on the deployed-adjacent substrate)

**P0-strict 20-crop pilot (10/fold), weakest-component-first, division-protected, no GT:**

| keep_frac | pooled | Δpool | 44b6 | Δ44b6 | 6bba | Δ6bba | ratio |
|---|---|---|---|---|---|---|---|
| 1.000 | 0.77904 | +0.00000 | 0.91076 | +0.00000 | 0.75084 | +0.00000 | −0.166 |
| 0.990 | 0.77853 | −0.00051 | 0.90778 | −0.00298 | 0.75079 | −0.00005 | −0.175 |
| **0.975** | **0.78140** | **+0.00236** | 0.90891 | −0.00185 | **0.75394** | **+0.00310** | −0.187 |
| 0.950 | 0.77712 | −0.00192 | 0.90779 | −0.00297 | 0.74893 | −0.00191 | −0.208 |
| 0.900 | 0.76764 | −0.01140 | 0.89866 | −0.01210 | 0.73920 | −0.01164 | −0.250 |

Optima: pooled 0.975 (+0.0024), 44b6 **1.00** (no pruning), 6bba **0.975** (+0.0031). The
preregistered kill rule ("optimum ≥0.99 on BOTH families") is **NOT triggered** → nominal
"selective pruning pays on 6bba." This exactly mirrors the **E0c full 199-crop** sweep
(`node_budget_sweep.json`): keep 0.975 → pooled +0.00157, **6bba +0.0022, 44b6 −0.0021**. Same
per-family sign on both substrates.

**Full 199-crop P0-strict per-family sweep — COMPLETED (854s CPU, 199/199 crops):**

| keep_frac | pooled | Δpool | 44b6 | Δ44b6 | 6bba | Δ6bba | ratio |
|---|---|---|---|---|---|---|---|
| 1.000 | 0.73178 | +0.00000 | 0.90018 | +0.00000 | 0.70264 | +0.00000 | −0.206 |
| 0.990 | 0.73257 | +0.00079 | 0.90086 | +0.00069 | 0.70341 | +0.00077 | −0.214 |
| **0.975** | **0.73277** | **+0.00099** | 0.90128 | **+0.00110** | 0.70351 | **+0.00087** | −0.226 |
| 0.950 | 0.73001 | −0.00178 | 0.89779 | −0.00239 | 0.70073 | −0.00191 | −0.246 |
| 0.900 | 0.72177 | −0.01001 | 0.89318 | −0.00700 | 0.69164 | −0.01100 | −0.286 |

**Optima: keep_frac 0.975 on ALL THREE (pooled, 44b6, 6bba).** The preregistered kill rule
("optimum ≥0.99 on BOTH families") is **NOT triggered**. Unlike the 20-crop pilot (which had 44b6
optimum at 1.00), the full run is **bilaterally positive** at 0.975: +0.00110 (44b6) / +0.00087
(6bba) / +0.00099 pooled. Most of it (+0.00079) is already captured by the first 1% trim (0.990).

**Interpretation (the honest read):** this is the rare lever that is **sign-consistent across both
families** on the deployed substrate — but the magnitude is **~+0.001**, an order below the +0.005
promotion bar and comparable to one LB resolution unit. It is picked as best-of-7 keep-fracs
(D-04: selection optimism manufactures ~+0.010 at T=5), and the E0c decomposition attributes ~116%
of the sibling +0.00157 to the count multiplier, ~−16% to edge quality — a count artifact, not a
tracking gain. **Verdict: NOT promotable, but the "closed" claim was false.** The deployed
substrate's node channel yields a small bilateral positive from a 2.5% weakest-component trim, which
the 4-placeholder-movie "closure" (one frozen point, forbidden substrate) entirely missed. Correct
status: **a real but sub-bar +0.001 bilateral effect, not a closed lever.** A single preregistered
confirmation (keep 0.975, both folds, bar +0.001 not +0.005) is all it merits — the EV does not
justify more.

---

## Bottom line

- **Divisions (Claim 1): soundly dead**, both families, both factors (pair-ranker structurally,
  mother-gate arithmetically). Do not reopen without a mother-gate signal at AUC ≥0.97.
- **Node-count (Claims 2/6): the E0c→plateau +0.02–0.03 is already banked in P3; further multiplier
  headroom on P3 is a LEARNED-FP-suppression problem, not a pruning problem.** The full 199-crop
  deployed-substrate sweep confirms the "closed" claim was false: a 2.5% weakest-component trim gives
  a **bilateral +0.001** (44b6 +0.0011 / 6bba +0.0009) — sign-consistent but an order below the bar
  and ~count-multiplier artifact. Not promotable; the prior "closure" rested on forbidden 4-movie
  evidence and simply missed it.
- **The 0.029 leader gap (Claim 3): correctly attributed to retrained weights + external data**;
  RT-12's node-count share is E0c-relative and must not be read as P3 headroom. Add multi-seed
  ensembling as an explicit arm of the retrain bet.
- **The systemic risk (Claim 4): every "positive" lever we hold (motion-gate, suppress-all,
  node-budget) is an E0c number.** The strongest of them — motion-gate, +0.0088 E0c, P(d>0)=1.0 —
  is built as two P3+armB kernels and PARKED. Scoring them is the single highest-EV action.
- **Metric (Claim 5): no new legitimate asymmetry**; the one real one is FP-suppression = the
  learned-ranker bet.
