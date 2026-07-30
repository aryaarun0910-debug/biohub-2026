# Next decision — attack the family boundary, not the same recipe

**Preregistered:** 2026-07-30
**Status:** superseded 2026-07-30 by the division track (D0' GREEN). Branch A CLOSED;
branch B demoted behind divisions.
**Compute state:** idle

## Primary track — Jaccard-optimal joint fork selection (D0' GREEN)

D0' measured the composed operation Oracle C never tested: remove existing false forks, then
reconstruct reachable true ones. Five parity-controlled arms, 199 crops, exact patched scorer,
edges-only edits (`N_pred` and node recall identical in every arm).

| arm | 44b6 | Δ | 6bba | Δ |
|---|---:|---:|---:|---:|
| baseline (parity OK) | 0.7595 | +0.0000 | 0.6490 | −0.0000 |
| suppress_all | 0.7630 | +0.0035 | 0.6517 | +0.0027 |
| suppress_all_then_add_replace | 0.8413 | +0.0818 | 0.7273 | +0.0783 |
| selective_suppress_then_add_replace | 0.8413 | +0.0818 | 0.7273 | +0.0783 |
| add_replace_then_selective_suppress | 0.8413 | +0.0818 | 0.7273 | +0.0783 |

GT-free child-retention control (`--fallback-only`): **+0.0783 / +0.0737**. Order does not
matter and the operations do not interfere. Division goes `TP0/FP93/FN26 → TP20/FP0/FN6` and
`TP4/FP582/FN121 → TP93/FP0/FN32`.

GREEN on both conditions: composed min-fold `+0.0737` (bar `+0.03`); suppression contributes
`+0.0601 / +0.0599` over Oracle-C-alone (bar `+0.01` bilateral). Strongly super-additive —
`0.0035 + 0.0182 = 0.0217` apart versus `0.0783` composed.

**Everything above is a GT-informed oracle ceiling and is not submittable.** Fork selection is
oracle in every arm.

### D0P — GT-free proposer audit: RED on frozen surfaces, but the cap is the limiter

Generation never consults GT; the oracle below is "proposable-only" (`suppress_all` then
add-replace restricted to forks the proposer generated).

| surface | 44b6 Δ | reachable covered | 6bba Δ | reachable covered | candidates |
|---|---:|---|---:|---|---|
| geometric_core 10.5/8.5 µm | +0.0195 | 5/20 | +0.0142 | 20/93 | 4.20M / 3.21M |
| native (candidate-edge cache) | +0.0116 | 3/20 | +0.0036 | 7/93 | 0.30M / 0.27M |
| outer_diag 15/15 µm (fixed) | **+0.0783** | **20/20** | **+0.0646** | 82/93 | 64.7M / 44.1M |

**RED by the preregistered rule** — 6bba `geometric_core` is `+0.0142`, below the `+0.015`
floor. No frozen deployable surface passes.

**But the diagnostic did its job.** `outer_diag` recovers 20/20 reachable divisions on 44b6 and
reproduces the GT-free D0′ ceiling exactly (`+0.0783`). The prize *is* present in a GT-free
surface; the `10.5 µm` cap discards it. That cap is E0c's motion-relink gate — calibrated on
**migration**. At division the daughters separate, so parent→daughter displacement is
systematically larger than ordinary motion, which makes a migration-calibrated cap the wrong
prior for a division proposer. Branch A corroborates: ordinary recoverable continuations sit at
median 8.29 / 7.62 µm, just under the cap.

Effective cost is far below the raw count: division FP accrues only at annotated mothers, so
`outer_diag`'s metric-visible denominator is 405,212 / 965,478 reliable negatives, not 64.7M /
44.1M. With `J = k/(26+m)`, holding `m ≤ ~50` at `k = 20` keeps `J ≈ 0.29`.

Caps were frozen before execution and have **not** been re-selected afterwards. Any successor
surface must be preregistered on principle, not fitted to these numbers.

**Open decision — RED says close reconstruction, but the diagnostic that was built to
distinguish "cap-limited" from "approach-limited" says cap-limited.** Resolving that tension is
a command decision, not a threshold to quietly retune.

Next gates, in order (D0R remains conditional and is NOT started):

- **D0** — Jaccard-optimal operating-point reanalysis of the existing division posterior. No
  saved v4 predictions exist under `artifacts/` (`divevents/*.npz` are balanced training events
  only), so this needs the authorised inference-only re-run. Threshold on exact composite after
  fork edits, cross-fitted between embryos — never precision `0.9`. Gate: `≥+0.005` exact
  composite on both families, or `≥30%` of the D0' bilateral upside.
- **D1** — covariance fork audit if D0 falls short. Candidate population is the union of E0c's
  existing forks, the reachable proposals, and their competing parent assignments; the scorer
  jointly decides suppress/retain/reconstruct/steal. Note the retain decision is near-vacuous:
  only `0` and `2` of ~20k existing forks sit on a true divider.
- **D2** — joint fork/parent re-optimiser, only after D1 passes.

Missed-node displacement forensics run in parallel; the broad atlas stays deferred.

## Decision

Do not accept `0.889` as the ambition, but stop spending GPU on generic training. Seven
methods now show the same failure: apparent progress within one embryo family evaporates or
reverses across the family boundary.

The next round has two cheap falsification branches. At most one may graduate to full OOF
compute. Branch A ran on 2026-07-30 and failed; branch B is the only one still open.

## Branch A — CLOSED 2026-07-30 (failed Gate A1.2 on CPU)

Executed in full without a Kaggle session: `scripts/branchA_gate.py --stage all`.

- **A0.2** — the cache does not contain the mechanism. Both OOF GEFFs are hard-pruned
  upstream at `p >= 0.5` (observed minimum `0.500008 / 0.500015`), with in-degree exactly
  `1` for every target and zero targets carrying two parents. Top-two-to-`0.25` cannot be
  replayed from cache.
- **A1.1 PASS** — of E0c's matched-edge FN (`60 / 215`): recoverable within the cap
  `73.3% / 49.8%`; not already a transformer edge `60.0% / 29.8%`; never enumerated by E0c
  at all `36.7% / 18.1%`. Every level clears the 10% bar.
- **Mechanism** — 100% of those never-enumerated candidates are beyond E0c's 6 µm tight gate
  (median `8.29 / 7.62 µm`). Enumerating them needs no transformer, so the breadth half
  was testable on CPU.
- **A1.2 FAIL** — one global 10 µm enumeration, exact patched scorer:
  44b6 adjJ `0.8821 -> 0.7225` (`-0.1596`), 6bba `0.8068 -> 0.6572` (`-0.1496`).
  Edge TP *falls* (`1268 -> 1177`, `1664 -> 1532`) while FP roughly triples/doubles.
  Re-running at the E0c gate reproduced the cached graphs exactly, so the harness is sound.
- **Ceiling under a perfect probability** — granting the widened surface an oracle
  `p = 1` on true pairs still gives `-0.1319 / -0.1281`. No attainable edge scoring
  rescues it. Not a downstream artifact: with the short-track filter off, `-0.1281 / -0.1125`.
- **Cause** — the per-frame assignment is one-to-one; ~1550 / ~940 extra relink edges per
  crop mean each false assignment can displace a true one.

Gate A1 requires all bullets, so branch A is closed with zero GPU spent. Residual: this
falsifies expanded candidates inside E0c's Hungarian assignment, not inside the public
notebooks' global ILP. That combination rests on two independently negative components
(this result and closed lever 5, v122 coupled ILP at `-0.0633` on 44b6). Reopening needs an
argument for why the interaction beats both parts, not just that it is untested.

Evidence: `inventory/branchA_surface_audit.json`, `branchA_oracle_ceiling.json`,
`branchA_widen10um_twocrop.json`, `branchA_oracle_prob_twocrop.json`,
`branchA_control_nofilter.json`.

<details>
<summary>Original branch A preregistration (kept verbatim for audit)</summary>

### Branch A — clean extraction of the current public candidate-breadth idea

Two newly popular public notebooks report scores near `0.95`, but their scored submissions
append negative-time, out-of-volume hub/fork structures. That output is an evaluator exploit
and is permanently quarantined.

Sources inspected 2026-07-30:

- `https://www.kaggle.com/code/boristown/dark-agi-biohub-cell-tracking-solution`
  — downloaded notebook SHA256
  `3072d128e03e7ae4a874570bb38aea33eff1d7bb6ed00e58522de2a5be66b1d4`.
- `https://www.kaggle.com/code/yoikoarmor/biohub-modular-last-call-turned`
  — downloaded notebook SHA256
  `d14fa960c7e7d532185690e1476831982b746c918fb00db21ec61f08154ac6da`.

The clean pre-exploit pipeline contains one potentially unmeasured mechanism:

- retain each target's top two transformer parents down to probability `0.25`;
- apply a `10 µm` motion cap;
- solve globally with ILP afterward.

Threshold changes, D4 TTA, survival-cost ILP, and gap closing have already been measured.
Only the expanded pre-ILP edge topology may be new.

### Gate A0 — zero/low compute audit

1. Diff the clean pre-exploit inference path against the parity-proven E0c/coupled cache.
2. Prove whether the existing cache contains the top-two/`0.25` candidates.
3. If it does, replay locally. If it does not, build only a two-crop T4 cache:
   `44b6_d29c9ab2` and `6bba_bb9f20c3` (the highest-edge-mass crops in each family).
4. No exploit cell, negative time, out-of-volume coordinate, artificial hub, or synthetic
   division is retained.

### Gate A1 — candidate/oracle

Proceed to full 199-crop OOF only if both preregistered crops satisfy all of:

- expanded candidates recover at least 10% of baseline matched-edge false negatives;
- exact clean graph score is positive versus E0c on both crops;
- node recall does not fall by more than `0.005`;
- candidate growth and runtime remain compatible with a 9-hour Kaggle session.

Failure closes the public candidate-breadth idea. Do not tune thresholds on the two crops.

### Full gate

One global configuration, both LOEO folds, exact patched scorer. Promotion remains:
both folds positive, min-fold at least `+0.005`, no major regime collapse.

</details>

## Branch B — CPU-only family-boundary decomposition

Use existing E0c, coupled, selector, and M1 artifacts. Do not train a model.

Decompose each per-crop delta into:

1. node population/recall;
2. raw edge matching;
3. wrapper changes;
4. division term;
5. count multiplier.

Then test whether any deployment-observable descriptor has the same directional relationship
to error within both families. Family/crop identity and hidden `N_est` are forbidden.

Branch B graduates only if it identifies:

- a bilateral oracle ceiling of at least `+0.01`; and
- a feature/mechanism whose sign is stable within both families under leave-family-out
  evaluation.

Otherwise it is documentation, not another selector search.

## Stop rule and final hedge

Branch A has already failed. Branch B is the last open gate: if it does not clear a
bilateral `+0.01` oracle ceiling with a leave-family-out sign-stable mechanism, the stop
rule fires.

If neither branch passes, stop research compute. Preserve two final candidates:

- E0c for private robustness;
- clean v122 for public strength.

Do not spend on full-data training, additional M1 seeds, logit ensembles, or external
pretraining until a family-boundary mechanism clears its cheap gate.
