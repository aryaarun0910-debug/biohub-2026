# Current handoff

**Status:** REOPENED 2026-08-16 (reverses the 2026-08-07 closure) by explicit user decision.
**Branch:** `master`.

## Mission

Aggressive climb toward **top-3** on the public leaderboard, private-set-honest. We hold a
reproducible **0.915** system (P3 harmonic); the leader is **0.948** (gap **0.033**), rank at
reopen ~143. Public score is a deployment signal; model selection stays embryo-held-out (LOEO).

## The central problem (from the submission ledger, verified)

The field moved to **0.93–0.948**. That jump is **NOT** harmonic fusion (tested: adopting the
public CC0 rule gave 0.915, not 0.93). **The 0.93+ teams hold something undisclosed.** Closing the
gap is a RESEARCH problem, not an adoption problem. A fresh frontier analysis (what that edge is)
is the first work of the reopen.

## Baseline and what is proven (gate all deltas vs this, both folds, no regime-slice regression)

- **P3 harmonic = 0.915 public** (deployment hedge; keep it live).
- LOEO substrate anchors: 44b6 `0.759549`, 6bba `0.648965` (reproduced exactly; the trustable base).
- **Closed levers (measured negative/trivial — do not reopen without a new mechanism):** split/merge
  arbitration (+0.0006), node-budget pruning (optimum = no pruning), GT-free component selector (−0.008),
  scalar-threshold re-acceptance (−0.023). Detection recovery is NOT a threshold — "it must be a ranker"
  (marginal precision ~0.003 where the ceiling needs ~0.55).
- **Positive, NOT-yet-shipped mechanisms (harvest first — cheap):** motion-residual gate (+0.0088 pooled,
  min-fold +0.0074, clears the ≥+0.005 gate); suppress-all division (+0.0027 on E0c, must be re-measured
  on P0-B where forks are 8 not 20k). Division GT-oracle ceiling is +0.06 but needs a high-precision fork
  selector (break-even ~10% precision at full recall).

## Immediate queue

1. **Fresh frontier analysis (IN PROGRESS, agents):** current top-20; the top-3 boundary; and the crux —
   what do the 0.93–0.948 teams have that public notebooks do not. Methods since the July break.
2. Harvest the two un-shipped positive mechanisms into a candidate (motion-residual gate; re-measured
   division suppression) and submit if bilateral.
3. Architecture decision + a learned-ranker / detector escalation (GPU) once the frontier read lands.

## Guardrails (unchanged, prize-critical)

- The unmatched-fork division-evaluator pathology is diagnostic ONLY — never in a submission.
- Exact public-source trajectory transfer into an identified hidden crop needs written host clearance;
  generic public-data pretraining is permitted.
- Never infer hidden-set quality from the four visible placeholder movies (in-sample, biased).
- Preserve user-owned `.claude/settings.json` and `.gitignore`. Stage explicit paths; never `git add -A`.
- Smoke → representative pilot → full LOEO; a passing smoke is not scientific evidence.

## Directly-relevant surface

- Core: `src/biotrack/` (wrapper, metric, metric_numpy, d1_partition, decision, submission).
- Docs: this file, `SYSTEM_DESIGN.md`, `reports/CLAIMS.md`, `reports/EXPERIMENT_LEDGER.md`,
  `reports/METRIC_SEMANTICS_VERIFIED.md`, `reports/submissions/CANDIDATE_LEDGER.md`.
- Scripts: `scripts/claims_table.py`, `d1_postprocess.py`, `d1f_probe.py`, `kaggle_factory.py`,
  `score_oof.py`, `win_bet/`, the `d1_*`/`build_*` family.
- Active notebook: `notebooks/kaggle_p3_harmonic` (the 0.915 deployment). Stale kernels → `notebooks/_archive/`.

## Preservation / recovery

- Large research artifacts: `C:\Users\aryaa\Documents\Biohub-CellTracking-2026_RESEARCH\`
  (sibling `research.sqlite`, 66 findings — evidence lives OUTSIDE the worktree).
- Recoverable git tags: `pre-lean-2026-07-30`, `pre-lean-2026-08-07`, `biohub-closed-2026-08-07`.

## Verification

```powershell
git status
git rev-list --left-right --count origin/master...master
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\claims_table.py --check
```
