# Experiment ledger

**Frozen evidence through:** 2026-07-30
**Full historical tree:** Git tag `pre-lean-2026-07-30` (`7897511`)

## Authoritative floor

E0c is the private-safe baseline: public `0.889`, exact patched LOEO OOF
`0.7595 / 0.6490`. The full wrapper is parity-proven and scored on all 199 crops.

Canonical result: `inventory/e0c_score_full.txt`.

## Closed methods

| Method | Exact decisive result | Verdict | Recovery point |
|---|---|---|---|
| Breadth candidate reranking | beat the wrapper ranking on neither held-out family | saturated | journal 2026-07-13 |
| Learned division posterior | mean recall at precision 0.9 ≈ `0.045`, near zero on 3/4 embryos | closed | journal 2026-07-13 |
| Isolated DAXI redetection | oracle recovery `0% / 8.8%`, below 20% gate | closed | journal 2026-07-13 |
| Temporal accumulation v3 | 44b6 null; 6bba negative with bootstrap lower/upper evidence below zero | closed | journal 2026-07-14 |
| v122 coupled ILP | 44b6 `-0.0633`, 6bba `+0.0507` | bilateral fail | `inventory/coupled_score_2026-07-29.txt` |
| A/D selector | perfect oracle min-fold only `+0.0056`; learned leave-family-out rules harmful | closed | `inventory/selector_audit_2026-07-29.json` |
| M1 domain-randomised model | same-family `+0.0074`; cross-family `+0.0010`, CI `[-0.0102,+0.0132]` | stopped after one seed | `inventory/m1_selection.json`, `inventory/m1_heldout_result.json` |
| Pre-ILP candidate breadth (10 µm) | 44b6 `-0.1596`, 6bba `-0.1496`; still `-0.1319 / -0.1281` under a perfect oracle edge probability | closed on CPU, no GPU spent | `inventory/branchA_*.json`, journal 2026-07-30 |

## Reopened method

| Method | Exact decisive result | Verdict | Evidence |
|---|---|---|---|
| Jaccard-optimal joint fork suppression + reconstruction | composed oracle ceiling `+0.0783 / +0.0737` (GT-free child retention); suppression contributes `+0.0601 / +0.0599` over Oracle-C-alone | GREEN, primary track | `inventory/phaseb_oracle_d0prime.json`, journal 2026-07-30 |

Distinct from the killed "high-precision trajectory posterior": that was gated at precision
`0.9`, whereas the metric rewards maximising exact composite. After suppression the division
count starts at `TP0/FP0/FN26`, so `J = k/(26+m)` for `k` true and `m` false forks added — a
detector at 30–50% precision clears the `+0.005` gate. Realizability is unproven; the oracle
selects forks with ground truth in every arm.

## Mechanistic conclusions

- The v122 improvement is survival pruning: it improves the count multiplier but collapses
  node recall on sparse 44b6.
- No deployment-observable selector transferred the sign of that pruning benefit.
- M1 made raw held-out linking worse (`0.6499 -> 0.6421`) and emitted 20% fewer nodes. Its
  `+0.0010` composite change was count credit, not improved tracking.
- The recurrent obstacle is embryo-family conditional shift, not insufficient compute on
  the same training recipe.
- E0c's fork layer is essentially pure noise: 11,441 forks on 44b6 and 9,012 on 6bba, of which
  `0` and `2` sit on a true GT divider. Only 93 / 584 are metric-evaluable; the rest fall in
  unannotated regions. Suppressing all of them is edge-neutral (`-0.0000 / -0.0020`), which is
  why suppression and reconstruction are worthless apart and strongly super-additive together.
- Candidate breadth fails for a structural reason, not a scoring one: the per-frame
  assignment is one-to-one, so widening the gate to 10 µm adds ~1550 / ~940 extra relink
  edges per crop and each false assignment can displace a true one. Edge TP falls below
  baseline even when every true pair is given probability 1.
- Exact graph-level checkpoint selection is retained as good infrastructure: M1 epoch 10
  scored `0.7963` internally while later epochs fell as low as `0.6820`, despite monotonically
  improving training loss.

## Public deployment evidence

- E0c: public `0.889`, private-safe OOF floor.
- Clean v122: public `0.908`, but fails bilateral OOF and is a hedge rather than a promoted
  private model.
- Public notebooks advertising roughly `0.95` on 2026-07-30 were inspected and contain a
  scored negative-time hub/fork augmentation stage. Their leaderboard score is not evidence
  of a clean tracking advance and that stage is quarantined.

## Recovery

Use Git history rather than keeping dead code in the active tree:

```powershell
git show pre-lean-2026-07-30:<path>
git worktree add ..\Biohub-CellTracking-2026-historical pre-lean-2026-07-30
```

Do not restore an entire historical plan into the active tree. Recover only the file needed
to reproduce or audit a specific result.

## Cycle 2026-07-31 — deployment programme

**Best public moved 0.908 -> 0.914.** P0-A reproduced the public 0.913 exactly; P0-B (clean base +
source-locked reverse-time w=0.20) reached 0.914. The `+0.001` delta is exactly one unit of LB
resolution, so reverse-time is not harmful but not established as beneficial.

### Objective corrected

The leaderboard POOLS with edge-volume weighting; 44b6 is only **14.94%** of edge mass. The old
bilateral-delta gate rejected every better pooled arm (v122 +0.0337, C +0.0315, Bp +0.0262 pooled
over E0c). Primary metric is now the exact pooled composite; min-fold is a robustness constraint.
Proof `scripts/verify_pooled_objective.py`, locked by `tests/test_pooled_objective.py`.
The objective is closed-form, verified to 2.27e-13 at corpus scale.

### Newly closed methods

| method | decisive result | verdict |
|---|---|---|
| Hub/fork exploit | -0.0027 / -0.0007 under the patched scorer | score-NEGATIVE, not merely illegitimate |
| Detector diversity (cheap) | threshold variants strictly nested, 0 new nodes; union of 5 gains +0/+1 GT nodes | empty |
| Appearance x appearance stacking | FPs concentrate on the SAME mothers (7-232x independence), lift 0.00 | closed with mechanism |
| Zebrahub as a division corpus | 5.7-11.7 terminations per division; oracle over 30 anchor x stride pairs still wrong-signed | dead on lineage |
| CTC replacement corpus | "cloning of datasets or their parts, including reference annotations, is strictly forbidden" | licence-blocked |
| H1-T conditional pair ranker | b = 0 in 44b6 -- correct set is a strict SUBSET of geometry's | closed structurally |
| Synthetic split patches | real-vs-synth CV AUC 0.9888; synth-trained -> real AUC 0.664 vs 0.867 | dead |

### Deployable and corpus-verified

| mechanism | pooled delta |
|---|---:|
| node budget (keep_frac 0.975, arm A) | +0.00157 |
| ssl x geometry veto | +0.00141 |

### Oracles (real ceilings, not deployable)

- H0c cascade **+0.06012** pooled; live-filter variant +0.0641/+0.0646 per-family.
- Hybrid substrate: 6bba reachable divisions **68 -> 101** for ~50 aux nodes; reachability sets are
  NOT nested, so the prior table understated the ceiling.

### Reporting corrections (all mine)

H1-M pooled ~+0.0023 -> **+0.00007 / -0.00131** (an in-family CV ceiling probe quoted as
cross-family, ~30x); node budget +0.00822 -> **+0.00157** (5.2x); FN association share 63.5% ->
**43.3%** (1.5x). Standing rule: corpus numbers only, basis named explicitly.

### Method findings

- SMD audits drastically understate multivariate domain separability (passed at mean |SMD| 0.302
  while 98.9% separable).
- FN attribution: 56.7% never detected, 43.3% detected then discarded.
- Break-even: detection needs 40.6% precision, a division action 10.15% -- and the 10.15% is at
  full recall.
- E0c's published baseline contains thousands of out-of-volume coordinates, hidden by
  `max(0, int(round(v)))`. Count disputed (7,349 vs 14,319); volume guard defaults OFF.
