# Biohub — Cell Tracking During Development (2026)

Everything from my entry to the Kaggle competition *Biohub — Cell Tracking During
Development* (closed 2026-09-29): tracking cells through 3D+t light-sheet movies of
developing embryos. Final private leaderboard: **rank 1488 of 4017**, best private
score 0.914 (best public 0.947). Full results and all 34 submissions are in
[results/](results/RESULTS.md).

The work ran as several campaigns, each in its own repository. They are merged here
with full commit history, one folder per repository, in chronological order.

| folder | period | what it is |
|---|---|---|
| [01-celltracking](01-celltracking/) | Jul – 7 Sep | Campaign 1. The long first run (540+ commits): pipeline, research registry, facts and levers. Best public 0.925. |
| [02-colab-relay](02-colab-relay/) | 26 – 27 Aug | Laptop ↔ Colab worker control channel, plus Colab/Kaggle parity-failure notebooks and measured compute burn rates. |
| [03-biohub-x](03-biohub-x/) | 1 – 7 Sep | Campaign 2. A blind, object-centric restart. Best 0.496. |
| [04-sprint-handoff](04-sprint-handoff/) | 7 Sep | Five-document handoff written between campaigns (start at `01-START-HERE.md`). |
| [05-biohub](05-biohub/) | 11 – 15 Sep | Campaign 3. Consolidated base: provenance-tracked knowledge base, local scorer, rank-1 plan. Reached 0.947. |
| [06-final](06-final/) | 19 – 20 Sep | Final sprint: pre-registered submissions, evidence gates, learned division ranker. Start at `docs/HANDOFF.md`. |

## Not included

Model checkpoints (`.pt`, `.pth`, `.ckpt`), `.npz` arrays, vendored pip wheels, the
generated synthetic training set, and any file over 50 MB were removed from history so the
repository fits on GitHub. Paths in the notes that point at them will not resolve. The
competition data itself was never committed.
