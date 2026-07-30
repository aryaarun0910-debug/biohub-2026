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

## Mechanistic conclusions

- The v122 improvement is survival pruning: it improves the count multiplier but collapses
  node recall on sparse 44b6.
- No deployment-observable selector transferred the sign of that pruning benefit.
- M1 made raw held-out linking worse (`0.6499 -> 0.6421`) and emitted 20% fewer nodes. Its
  `+0.0010` composite change was count credit, not improved tracking.
- The recurrent obstacle is embryo-family conditional shift, not insufficient compute on
  the same training recipe.
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
