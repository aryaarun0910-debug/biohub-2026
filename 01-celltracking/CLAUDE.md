# Biohub Cell Tracking 2026 — project operating instructions

## Standing workflow: record + commit after every experiment

**After any test, experiment, or meaningful change that produces a result — without
being asked — do BOTH of these before moving on:**

1. **Update the research record.**
   - Add a dated, terse entry to `reports/journal/JOURNAL.md` (match the existing
     `### YYYY-MM-DD (short title)` bullet style): what was run, the numbers, the
     decision, and the next step. Convert relative dates to absolute.
   - If the result changes a conclusion, also update the relevant brain doc
     (`reports/research/brain/SYNTHESIS.md`, `ROADMAP.md`, `WIN_BET.md`, or
     `METRIC_SEMANTICS_VERIFIED.md`) and `HANDOFF.md`.
2. **Commit to the git repo** with a clear, descriptive message summarising the result
   (not just "update"). End commit messages with the Co-Authored-By trailer. Include
   the code, the updated docs, and any small canonical result files (CSVs/summaries);
   do not commit ignored data/weights/artifacts.

Applies to: OOF scoring runs, kill-gates, ablations, kernel results, wrapper/repair
changes, new scripts, and research-doc/synthesis updates. A failed or negative result
is still recorded and committed — negative evidence is kept, never discarded.

If a run is still in progress, commit the code/setup now and append the numbers to the
same journal entry (and commit again) once it completes.

## The authoritative plan

`reports/research/brain/ROADMAP.md` is the current battle plan; `SYNTHESIS.md` +
`METRIC_SEMANTICS_VERIFIED.md` are the evidence beneath it. `HANDOFF.md` is the
start-here for scores/decisions. These supersede the older
`THREE_LAYER_WIN_ARCHITECTURE` portfolio.

## Non-negotiable measurement rules

- Measure every delta against the **E0b full-0.889-wrapper OOF** baseline
  (`scripts/win_bet/e0_replay.py`), never the weaker organizer greedy.
- Promote only on **embryo-held-out** evidence: both folds up, min-fold ≥ +0.005, no
  regime-slice regression. Never tune on the four visible placeholder test movies.
- With ~2 effectively-independent samples, run **≤2 confirmatory tests per round**.
- **Augment-never-replace**: the wrapper is the fallback; new methods override edges
  only under gate. No wholesale association replacement (it has failed twice).
- Prize eligibility: external data/models need URL + license + checksum; gradient-free
  target-time adaptation only (weight-update TTA needs written host clearance); the
  unmatched-fork division evaluator defect stays quarantined.

## Environment

- Metric/dev: `.venv` (Python 3.12). Trackastra: `.venv-trackastra`.
- Run tests: `.\.venv\Scripts\python.exe -m pytest -q`.
- Score OOF: `.\.venv\Scripts\python.exe scripts\score_oof.py --pred-dir <dir> --gt-dir data\train`.
