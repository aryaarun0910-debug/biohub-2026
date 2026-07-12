# 06 — Red Team: adversarial critique of the three-layer architecture

**Date:** 2026-07-12
**Lane:** ruthless re-ranking of the 8-bet portfolio by *realistic, transfer-adjusted* EV/cost.
**Attacks:** [THREE_LAYER_WIN_ARCHITECTURE_2026-07-12.md](../../THREE_LAYER_WIN_ARCHITECTURE_2026-07-12.md), [WIN_PLAN.md](../../WIN_PLAN.md), [HANDOFF.md](../../../HANDOFF.md), [JOURNAL.md](../../journal/JOURNAL.md).

---

## Executive summary — the 5 sharpest criticisms

1. **The gates measure a quantity that has already pointed the WRONG WAY.** The fusion bet improved held-out OOF by **+0.0353 / +0.0364** (WIN_PLAN §intro) — it *passed* the "both folds positive" gate cleanly — and then **lost on the real target: 0.865 vs 0.889 public** (HANDOFF). That is a **measured sign inversion**: a +0.035 OOF gain became a −0.024 deployment loss. The Trackastra dummy-movie score did the same thing (0.9065 dummy → 0.865 hidden). Since **every kill-gate in the plan is an OOF delta**, and OOF deltas have already been shown not to preserve sign to the hidden set, **passing the gates is neither necessary nor sufficient**. The plan has not internalized its own strongest negative result.

2. **We have ~2 effectively-independent samples and are trying to select among 8 bets.** Two embryos, edge-weighted **~5:1** toward 6bba (109k vs 19.8k edges; JOURNAL §4), crops within an embryo correlated (same acquisition, biology, cells persisting across frames of one crop). The domain shift that the private board actually tests is *embryo-level*, so the real degrees of freedom ≈ **2**. Running 8 bets × 2 folds at a +0.005 gate, with per-fold noise, you expect **~2 bets to pass by luck**. You cannot safely rank 8 hypotheses on 2 embryos.

3. **~45% of the compute (bets 1+2) attacks a bottleneck the oracle says is nearly CLOSED.** Measured candidate-edge oracle is **0.935 / 0.885**, endpoint oracle **0.9179 / 0.8756**, learned node recall **0.9536 / 0.8966** — detection is near-ceiling on the *production* stack; current linking is **0.656 / 0.559**, so the ~+0.18–0.26 of headroom is **association, not detection** (WIN_PLAN §"research-backed", assoc lane §1). Bet 1 (track-before-detect) and bet 2 (Lagrangian Lineage Field, a track-before-detect *detection* architecture) both spend on endpoint recovery. **Naïve TBD was already killed at 5.5% recovery vs a 20% gate** (JOURNAL 2026-07-03 "TBD: NO-GO"). Reviving it as "raw pre-NMS response volumes" ignores the journal's diagnosis: the problem is *separability* (DoG's bright-background tail overlaps dim nuclei), which a learned detector fixes — i.e. it is bet 2, not bet 1.

4. **Baseline confusion inflates every headroom estimate.** OOF deltas are quoted against the **organizer greedy (0.6562/0.5593)** or fork-suppressed organizer (0.6595/0.5680). But the deployed **0.889 wrapper already contains** D4 TTA, learned+ILP links, motion relinking, gap repair, safe divisions, adaptive short-track recovery, **and isolated-node pruning** (WIN_PLAN §1; HANDOFF). So bet 5 (pruning) reinvents something already in production (its +0.018/+0.011 is vs the organizer, *already captured* by the wrapper), and bets 3/4 partly re-derive the wrapper's relinking. **No gate specifies the baseline it is measured against.** Until they say "delta vs the deployed 0.889 wrapper on identical data," the numbers are not comparable to the thing we ship.

5. **The main bet is the least defensible one.** A from-scratch 7-head, ~16-channel 4D model with deformable warping + temporal attention, trained on 2 embryos + Ultrack pseudo-labels, that must run offline ≤12h on T4/T4x2. The journal already found **"full-length retrain (45ep) OVERFIT… more training on 2 embryos is negative"**. A model an order of magnitude larger will overfit harder, **imitate Ultrack's linking** (its training targets), and its ≤12h dense-4D inference on hidden volumes is unproven. Highest cost (25%), worst transfer-adjusted EV. **Kill it as the main bet.**

---

## The OOF→hidden transfer problem, quantified

The plan treats embryo-held-out OOF as ground truth for hidden performance. The data says the mapping is **non-monotone in deltas**:

| System | OOF 44b6 | OOF 6bba | Public/hidden | Sign of (OOF delta → public delta) |
|---|---:|---:|---:|---|
| 400-ep anchor | 0.6562 | 0.5593 | **0.889** | reference |
| fork-suppressed organizer | 0.6595 | 0.5680 | — | — |
| direct Trackastra + pruning (fusion) | **0.6948** | **0.6044** | **0.865** | **+0.035 OOF → −0.024 public (INVERTED)** |

Two facts fall out:

- **The OOF↔public offset is ~+0.2 and nonlinear.** 0.6562/0.5593 OOF maps to 0.889 public because the visible movies are *easier* than the held-out training embryos (JOURNAL 2026-07-03: "hidden test embryo is EASIER than our hard 44b6 fold"). A +0.2 offset that is not even sign-preserving for deltas cannot be used to rank bets.
- **The fusion counterexample is the whole plan's Achilles heel.** It passed the exact gate structure the portfolio uses ("both folds positive, +0.035") and lost the real target. Any bet whose only evidence is "both folds OOF positive" is in the same epistemic position fusion was in the day before it was submitted.

**Why fusion inverted (root cause the plan must address):** (a) the fusion threshold (cached 0.8) and the agreement weighting were tuned to the two training embryos' geometry; (b) it *replaced* an association component the wrapper's downstream machinery (ILP, relinking, gap repair) depends on, and on the easier hidden regime the wrapper's own association was already good, so the swap only injected error. Both mechanisms are shared by **any bet that (i) tunes a scalar threshold/weight on 2 embryos or (ii) replaces rather than augments the wrapper.** That is bets 1, 2, 4, 7, and the learned variant of 6.

**How many hypotheses can we safely test?** With n≈2 and a target family-wise false-pass rate <0.2, you can afford roughly **one or two confirmatory tests**, not eight. Everything else must be either (a) *pre-registered and pooled* into one decision, or (b) validated on **constructed pseudo-folds** (cluster 6bba's 128 crops by the L1.4 regime index into ≥3 acquisition-regime groups and hold each out) to manufacture validation diversity the two embryos don't provide. The plan builds the regime index but never uses it to multiply folds — a missed, cheap defense.

---

## Per-bet risk table

Transfer risk = P(a 2-embryo OOF gain fails to reach hidden). Gate adequacy = is the stated kill-gate falsifiable, cheap, and measuring the right baseline/quantity.

| Bet | Transfer risk | Overfitting / failure mechanism | Gate adequacy | Realistic EV | Verdict |
|---|---|---|---|---|---|
| **1. TBD / motion-comp endpoint recovery** | **High** | Already killed at 5.5% vs 20% (JOURNAL). Attacks near-closed detection ceiling. "Marginal precision" is unmeasurable under positive-only sparse labels. | Gate is falsifiable+cheap but **the bet already failed it**; re-run with pre-NMS won't fix a *separability* problem. | Low (detection ~solved) | **KILL** (fold into bet 2 only if a learned evidence field is used) |
| **2. Lagrangian Lineage Field** | **Very High** | From-scratch 4D on 2 embryos + Ultrack pseudo-labels → overfits (45ep already did) and caps at Ultrack linking. ≤12h dense-4D inference unproven. Scalar/threshold tuning on 2 embryos = fusion's failure mode at 10× cost. | Gates the **cheap prototype** (= bet 1, already failed), then commits weeks before the real artifact is gated. Cost/EV inverted. | Low–negative net of cost | **KILL as main bet** |
| **3. Selective baseline repair** | **Medium-Low** | Only real risk is threshold-on-2-embryos; mitigated because it **augments, never replaces**, and requires FB agreement + no constraint violation. | **Best gate in the plan**: +0.005 both folds *measured on the 0.889 graph itself*, no stress regression. Right baseline, cannot regress by construction. | **Highest realistic** | **DO NOW** |
| **4. Dense-track edge/fork posterior** | **High** | Trained on Ultrack pseudo-labels → **imitates Ultrack**, cannot exceed it. "same-source" March-22 may overlap visible-public regime, not hidden. | "**+0.010 both folds OR +0.015 min-fold**" — the *min-fold OR clause is a loophole*: passes on the easy embryo alone (op_bright showed min-fold gains ≠ edge-weighted gains). | Medium, capped | **LATER**, drop the min-fold clause |
| **5. Component/count optimization** | **Medium** | Isolated-node pruning **already in the 0.889 wrapper** → most of +0.018/+0.011 is not novel above production. Count-penalty trick needs `N_est` which is **hidden at inference** → can't target the ratio. | "**transferable component-confidence curve**" — **not falsifiable, no threshold, would pass noise.** Rewrite as a quantitative OOF adj-J delta. | Low–Med (the *Dinkelbach λ* piece is the only novel, transferable part — `G` not needed) | **LATER** (only the λ increment; re-gate) |
| **6. Forward/backward target adaptation** | Gate-variant Low; learned-variant **High + rules** | Safe FB-gating removes edges → can cut recall/count. Learned variant **trains on test images** — eligibility unconfirmed. | Zero-gradient gate is a **good cheap kill-gate**; but passing it does **not** license the learned variant (needs host clearance). Plan conflates the two. | Gate: Low; Learned: blocked | **DO the safe gate NOW; treat learned variant as BLOCKED** on clearance |
| **7. Local unbalanced OT / FGW** | Medium | Overlaps what the ILP already does; cost design tuned on 2 embryos. | "**+0.003 min-fold on ambiguous SUBSET**" — **weakest gate in the plan**: single fold, cherry-picked subset, tiny n → passes noise trivially. | Low (3% alloc) | **KILL for now** |
| **8. Dedicated division posterior** | **Very High** | ~151 divisions across 2 embryos, **26 vs 125 imbalance**; geometric proposals gave **0 TP from 7**; organizer div-FP swings 594 vs 10,786 across embryos → wildly embryo-sensitive. Upside capped at 0.1×J. | Div-J ≥0.25 gate ok, but works **adjacent to the quarantined evaluator defect** — accidental exploitation = DQ risk. | Low, capped ≤0.025, high variance | **LATER**, hard-guarded, only after edge score is frozen |

---

## Kill-gate realism — where the gates are too weak or mis-targeted

- **All gates are OOF-delta gates.** Given the demonstrated OOF→hidden sign inversion, add a **transfer-stability gate** to every bet: the gain must be **stable across L1.4 regime slices** (freeze/jump/density/depth/intensity). A delta that concentrates in one regime is the fusion failure mode and will not transfer. This is the single most valuable new gate and it costs nothing beyond re-slicing existing OOF.
- **Min-fold is statistically fragile with 2 folds.** "min-fold ≥ X" is one number from one embryo; it says nothing about the mean and cannot be bootstrapped at the embryo level (n=2). Where a bet allows "min-fold OR both-fold" (bet 4), **the min-fold clause should be deleted** — it is the loophole through which noise passes.
- **"Transferable component-confidence curve" (bet 5) and "stress-test robustness" (major-arch gate) are undefined.** Both would pass anything. Replace with explicit numbers: e.g., "adj-J delta ≥ +X on **both** embryos **and** on **each** regime slice, measured against the 0.889 wrapper."
- **"Marginal precision ≥70%" (bet 1)** cannot be honestly measured: with positive-only sparse labels, a recovered node in an unannotated region is neither a confirmed TP nor FP — it becomes count-penalty pressure. Precision-on-annotated overstates true precision.
- **The bilateral fusion gate already passed and still failed.** Therefore raise the confirmatory bar: **no promotion without a wrapper-relative OOF gain AND regime-slice stability AND a runtime/eligibility pass** — and accept that even then, hidden transfer is a coin-flip, so the *deployment* decision must keep the 0.889 wrapper as one of the two final submissions.

---

## Compute / time realism

- **Opportunity cost is the story.** Bet 2 (25%) + bet 1 (20%) = **45% of the portfolio** on detection recovery the oracle says is nearly closed, one arm of which already failed its gate. The two cheap, right-baseline bets (3 and the Dinkelbach part of 5) are 25% combined. The allocation is **backwards** relative to measured headroom.
- **≤12h offline inference for the LLF is hand-waved.** Deformable 4D warping + multi-scale temporal attention over full hidden volumes ((100,64,256,256) typical), internet-off, no Gurobi, is a real risk of blowing the 12h budget — a *submission-invalidating* failure, not just a score miss. The plan's own §3.6 asks for a <9h projection but the LLF has none.
- **The plan's caching layer (L3.2) is genuinely good** and under-credited: it turns most experiments into CPU reselection. It should be built **first**, because it is what makes the n=2 multiple-comparisons problem *cheaper to police*, not because it enables more bets.

---

## The baseline paradox — who has real headroom above the 0.889 wrapper

The 0.889 wrapper already does: 400-ep temporal model, D4 detection TTA, learned+ILP links, motion relinking, gap repair, safe divisions, short-track recovery, isolated-node pruning.

- **Reinvents the wrapper (low novel headroom):** bet 5 pruning (already in), bet 1 gap-filling (gap repair already in), parts of bet 4/7 linking (relink + ILP already in).
- **Real headroom ABOVE the wrapper:** (a) **selective, uncertainty-gated repair of the wrapper's own low-confidence edges** (bet 3) — augments what the wrapper can't self-correct; (b) **Dinkelbach metric-aligned λ** (subset of bet 5) — the wrapper optimizes generic likelihood, not the count-penalized adj-J, and `G` is provably not needed (metric lane §2), so this transfers; (c) **safe FB-consistency gating** (bet 6 safe variant) as a precision filter the wrapper lacks.
- Everything else is either inside the wrapper already or is a *replacement* — and replacement is exactly what produced 0.865.

---

## Rules / eligibility landmines

- **Learned target-time adaptation trains on the test images.** WIN_PLAN §4/§F asserts the rules permit it but concedes it is **unconfirmed** ("optionally seek organizer confirmation"). If a *prize* solution uses it and it is later ruled "training on test data," that is a DQ. **Get written host clearance BEFORE building the learned variant**, or build only the gating variant. The plan currently designs toward the learned variant while treating clearance as optional — reverse that.
- **Division work (bet 8) sits next to the quarantined evaluator defect** (distant unmatched fork qualifies a GT division while dodging division-FP). Correctly quarantined in the journal/HANDOFF, but a division-posterior that "adds a second daughter edge" can trip it accidentally. Require a **hard automated guard/test** in the submission pipeline that rejects any graph exhibiting the unmatched-fork pattern.
- **External-data licenses unverified in these docs.** DAXI weights, ZSNS/Zebrahub, the "522-frame exact-scale" embryo with **identity unproven** (JOURNAL §2). Two risks: (1) a **non-commercial** weight/dataset in a prize solution; (2) the "same-source" March-22 embryo *being* (or overlapping) the hidden test embryo → that is the plan's own red line ("public-source label transfer into an identified test crop"). **Confirm March-22 is disjoint from the hidden embryo before training on it**, and log the license per artifact as L1 already requires.

---

## Over-engineered vs under-specified

**Over-engineered (cut or defer):**
- **L1 "unified trajectory lake"** with full provenance schema, a 9-family corruption taxonomy, and a ~10-feature regime index. The binding constraint is **n=2 validation**, not data volume. Provenance is needed for eligibility; the elaborate warehouse is gold-plating for a 2-embryo problem.
- **The LLF's 16-channel, 7-head 4D architecture** — massive surface area, each head an overfit vector, for a detection headroom the oracle says is nearly gone.
- **The L3 6-branch cascade + test-time calibration of 7 statistics** — each branch is another threshold tuned on 2 embryos.

**Under-specified (must fix before any gate is meaningful):**
- **Which baseline every gate measures against** (organizer greedy 0.656 vs fork-suppressed 0.6595 vs the deployed 0.889 wrapper). Mandate: **delta vs the 0.889 wrapper on identical held-out crops.**
- **"Stress-test robustness"** and **"transferable component-confidence curve"** — no metrics, no thresholds.
- **How pFP / "ignored" mass is calibrated** with 2 embryos of positive-only labels (metric lane needs per-embryo `P(metric-FP|not TP)`; with n=2 that is 2 estimates).
- **LLF ≤12h offline inference feasibility** — no runtime projection exists.

---

## Missing high-EV ideas (what a top-10 team does that this plan doesn't)

1. **Re-baseline everything against the deployed 0.889 wrapper's OOF** — and if the wrapper's true embryo-held-out OOF isn't yet measured on identical crops, measuring it is the **highest-information single action available**, because it re-scales every headroom claim in the plan.
2. **A transfer-audit gate**: promote only deltas that are **stable across regime slices** (use the L1.4 index you already build). This directly targets the fusion failure and is nearly free.
3. **Manufacture folds**: cluster 6bba's 128 crops into ≥3 regime groups and hold each out, turning n≈2 into n≈4–5 pseudo-folds for *selection* (not for final honesty, but to break ties less noisily).
4. **Two-submission hedge strategy**, explicit: final slots = {0.889 wrapper (safe)} + {best wrapper-augmenting challenger}. Given the sign-inversion evidence, **never spend both slots on challengers.**
5. **Adopt the live public frontier faithfully.** The board is 1000+ teams forking one LB897 baseline; the hidden board is *wider* than public suggests and the hidden embryo is *easier* than 44b6. The highest-EV move may simply be **the frontier wrapper + one surgical, wrapper-compatible robust delta** — not a from-scratch 4D architecture. The plan under-weights "adopt the best public baseline and add one transferable increment."
6. **Bound divisions by their arithmetic**: max contribution 0.1×J, realistically ~0.02, against extreme 2-embryo variance and DQ adjacency. Any division effort must be justified against that ceiling before compute is spent.

---

## Final re-ranked portfolio (realistic transfer-adjusted EV / cost)

| Rank | Bet | Action |
|---|---|---|
| 1 | **Selective baseline repair (3)** + its cheap kill-gate = **safe FB-consistency gating (6-safe)** | **DO NOW.** Right baseline, augments-not-replaces, cannot regress by construction. |
| 2 | **Dinkelbach metric-aligned λ** (the transferable subset of 5) | Cheap, `G`-free, optimizes the real metric. DO after re-baselining. |
| 3 | **Dense-track edge/fork posterior (4)** | LATER. Drop the min-fold clause; guard against Ultrack imitation and March-22 leakage. |
| 4 | **Learned target-time adaptation (6-learned)** | **BLOCKED** on host clearance. Do not build until cleared. |
| 5 | **Division posterior (8)** | LATER, hard-guarded, only after edge score is frozen; upside ≤0.025. |
| — | **Lagrangian Lineage Field (2)** | **KILL as main bet.** Revisit only if a learned-evidence prototype clears 20% endpoint recovery *and* beats the wrapper on both embryos + all regime slices. |
| — | **Track-before-detect (1)** | **KILL as standalone** (already failed 5.5% vs 20%); survives only folded into a learned evidence field. |
| — | **Local OT / FGW (7)** | **KILL for now** (weakest gate, overlaps the ILP). |

### The ONE thing to do first

**Measure the deployed 0.889 wrapper's own embryo-held-out OOF on identical held-out crops, then run selective, uncertainty-gated repair (bet 3) — augmenting, never replacing — against *that* baseline, promoting only deltas that are positive on both embryos AND stable across regime slices.**

Rationale: this is the only near-term, GPU-free action that simultaneously (a) fixes the baseline confusion that inflates every headroom number, (b) refuses to repeat the wholesale-replacement move that already cost us 0.024 on the hidden set, and (c) installs the regime-slice transfer audit that the fusion failure proves we need. Do this before spending a single GPU-hour on the 4D model.
