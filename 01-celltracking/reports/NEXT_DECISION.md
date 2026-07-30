# Next decision — attack the family boundary, not the same recipe

**Preregistered:** 2026-07-30
**Status:** active — branch A CLOSED 2026-07-30 (CPU, no GPU spent); branch B open
**Compute state:** idle

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
