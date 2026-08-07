# Project closed

**Closed permanently:** 2026-08-07

**Final verified public score:** **0.915** (P3 harmonic)

**Final public rank at closure:** **143**

**Active local processes:** none

**Active Kaggle kernels:** none

**Submission ready:** none beyond the existing P3 result

## Decision

Biohub is archived and must not receive further research, training, Kaggle compute, or
submission work. The project produced a reproducible 0.915 system, but the demonstrated
public return no longer justified the additional investment required to reach the current
top-10 boundary.

The unfinished probability-at-the-deployed-gate oracle was stopped deliberately at
closure. Its partial rows remain in the sibling research store and are not an experiment
result. Do not resume or interpret them.

## Preservation

- Repository history, tests, notebooks, weights, and reports remain intact.
- Large research artifacts remain under
  `C:\Users\aryaa\Documents\Biohub-CellTracking-2026_RESEARCH\`.
- Historical material removed by the lean reset remains recoverable from Git tag
  `pre-lean-2026-08-07`.
- The closure tag is `biohub-closed-2026-08-07`.
- User-owned `.claude/settings.json` and `.gitignore` changes remain untouched.

## Reopening rule

There is no automatic reopening path. Resume only after an explicit user decision to
reverse this closure; no stale task, agent handoff, monitor, or promising oracle is
authority to restart the project.

## Read-only verification

```powershell
git status
git rev-list --left-right --count origin/master...master
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\claims_table.py --check
```
