# Current handoff

**Updated:** 2026-08-07

**Commit before this reset:** `c939bbf`

**Public:** **0.915** (P3 harmonic)

**GPU running:** none

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

1. Complete the two CPU measurements:
   - corpus-wide C/T/L/D census on the scorer-exact partition;
   - probability-at-the-deployed-gate oracle for association.
2. Verify the rebuilt 2x2 smoke artifacts from `c939bbf`:
   - both checkpoints x both families;
   - TTA-consistent features;
   - no mutation of the association feature tensor;
   - complete manifests and exact H0 parity.
3. Run the three-crop smoke.
4. If structurally green, run a 16-24 crop split-1 pilot spanning median, p90, extreme,
   dense, and sparse 6bba regimes.
5. Return the pilot report. Launch the 7.88-hour split-1 run only if the mechanism, not just
   the oracle, shows portable ranking or re-acceptance value.

## Pilot decision logic

- Large **T** mass with useful frozen-feature ranking: build a tiny re-acceptance head.
- Large **T** mass but no portable ranking: test a per-crop scalar threshold under a fixed
  node budget; do not build a larger network.
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
