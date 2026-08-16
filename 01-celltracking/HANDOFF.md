# Current handoff

**Status:** REOPENED 2026-08-16 (reverses the 2026-08-07 closure) by explicit user decision.
**Branch:** `master`.

## Mission

Aggressive climb toward **top-3** on the public leaderboard, private-set-honest. We hold a
reproducible **0.915** system (P3 harmonic); the leader is **0.948** (gap **0.033**), rank at
reopen ~143. Public score is a deployment signal; model selection stays embryo-held-out (LOEO).

## The central problem — CRACKED (frontier swarm, 2026-08-16; full reports in _RESEARCH/agent_reports/)

Top-3 boundary = **0.948** (leader 0.950). Our 0.915 = the PUBLIC PLATEAU (302 teams at exactly
0.915; ~rank 447). The 0.93–0.950 tier's edge is NOT post-processing (exhausted ~0.911–0.916,
confirming our own finding), NOT the patched metric hack, and NOT a drop-in pretrained linker
(community tested Trackastra/CoTracker and even the host's HOCT — HOCT UNDERPERFORMED a tuned ILP,
discussion #728551, so HOCT is NOT the edge). **The edge is a RETRAINED/GENERALIZING edge model
enabled by EXTERNAL same-domain data.** Specifically:
- **H1 (strongest): retrain the detector/associator on external ZEBRAHUB data** (imaging + dense
  Ultrack `*_tracks.csv` lineages), host-unlocked **2026-08-13** ("no overlap with test set", #734330).
- **H2 (= our own gap): a learned FP-suppressing candidate ranker** — the failure is PRECISION not
  recall (DoG already finds 0.91–0.94 of nuclei; the junk candidate pool mis-links). "A ranker, not a threshold."
- **H3: dense pseudo-labels / synthetic tracks** (José Freitas 18.5GB CC0, 165k labeled divisions, ~540x real).
- H4: division recovery +~0.02, but 304 events → shakeup-prone secondary lever.

**TIMING:** we closed 2026-08-07; Zebrahub external data unlocked 2026-08-13 — SIX DAYS AFTER we quit.
The biggest lever the top tier rides did not exist as a legal option when we closed. The reopen is well-timed.

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

## Immediate queue (post-frontier; LOEO-gated on the PATCHED scorer before any GPU submission)

0. **Quick bump (free, CPU):** harvest the un-shipped motion-residual gate (+0.0088 pooled, min-fold
   +0.0074) into a candidate on P3 and submit — banks a small real gain off 0.915 while the retrain spins up.
1. **THE program (H1+H2 — GPU T4x2):** retrain the detector + a learned FP-suppressing candidate ranker on
   external ZEBRAHUB (+ synthetic H3), validated leave-one-embryo-out with the patched scorer. Target: push
   adj_edge_jaccard past the ~0.91 public wall. This is the only measured path off the plateau.
   - Detector recall lever: LEAD "over-propose + learned re-scoring" (learned-3D-NMS/D2D) — the ranker mechanism.
   - Data: Zebrahub imaging + tracks (dense same-domain); Freitas 18.5GB synthetic (division supervision);
     leave-one-embryo-out CV is the community-validated protocol.
2. **Division (H4), late + precision-gated:** small (+~0.02), shakeup-prone — only after edges clear, watch CV/LB divergence.
3. Keep P3 harmonic 0.915 as the frozen hedge; every candidate gates vs it on both folds, patched scorer.

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
