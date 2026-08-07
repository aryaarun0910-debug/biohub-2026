# Current handoff

**Updated:** 2026-08-07

**Commit before this reset:** `c939bbf`

**Public:** **0.915** (P3 harmonic)

**GPU running:** split-1 D1 smoke source and target cells

**Submission ready:** none

## Decision

Proceed with the detector-calibration branch, but do not launch the 7.88 T4-hour split-1
run yet.

The latest corpus audit found:

- at most 22.3% of detector misses appear optically unresolvable;
- 85.94% of GT centres have a local maximum within 7 um;
- only 58.27% have an accepted maximum;
- 66.30% of unmatched GT retain an unaccepted maximum inside 7 um;
- the remaining estimated detection ceiling is +0.062 to +0.069 pooled;
- true eight-view D4 is dead at +0.00068;
- the synthetic corpus is rejected for the active path;
- association-score proposals await a free probability-at-gate oracle.

This is evidence for an acceptance/head problem, not permission to build a detector bank,
HOCT stack, graph ensemble, or full retraining programme.

## One execution sequence

1. Complete the probability-at-the-deployed-gate oracle for association. The corpus-wide
   census is complete; split/merge arbitration is closed at +0.000581.
2. Verify the rebuilt 2x2 smoke artifacts:
   - both checkpoints x both families;
   - TTA-consistent features;
   - no mutation of the association feature tensor;
   - complete manifests and exact H0 parity.
3. Finish the four-cell three-crop smoke. Split-1 is running; split-0 is queued behind
   Kaggle's two-session limit.
4. If structurally green, run the existing 19-crop two-direction pilot.
5. If the pilot shows portable ranking, run a one-basis candidate replay with the fitted
   33-parameter head through P3, the complete wrapper, and the exact scorer.
6. Only a graph-scored, homogeneous pilot gain may launch public test inference. The
   7.88-hour full split-1 export is confirmatory and no longer the automatic next spend.

## Pilot decision logic

- Large **T** mass with useful frozen-feature ranking: build a tiny re-acceptance head.
- Large **T** mass but no portable ranking: close re-acceptance. The scalar-threshold
  fallback is measured at about 0.3% marginal precision and -0.0234 pooled, so it is dead.
- Large **L/D** mass or chance-level frozen representation: detector representation work is
  justified, beginning with a paired minimal training ablation.
- Revised net ceiling below +0.020 pooled: close the detection attack.
- Cross-crop coefficient of variation above 1.22: retain as private hedge, not a public-LB
  promise.

## Verification

```powershell
git status
git rev-list --left-right --count origin/master...master
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\claims_table.py --check
```

Expected user-owned dirty files: `.claude/settings.json`, `.gitignore`.

Historical detail is recoverable from tag `pre-lean-2026-08-07`; do not reconstruct it in
new Markdown files.
