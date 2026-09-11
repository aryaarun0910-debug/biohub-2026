# AGENTS.md — contract for any agent working on Biohub

Read by Codex automatically; `CLAUDE.md` points here so there is one file, not two that drift.

## The task

Kaggle **Biohub — Cell Tracking During Development** (id 136605). Detect cells in 3D+time
zebrafish microscopy, link them across time, identify divisions. **Closes 2026-09-29 23:59 UTC.**
Aim is rank 1. Kernels-only: the submission is a Kaggle notebook, internet disabled, ≤12 h.

    score = adjusted_edge_jaccard + 0.1 × division_jaccard

## The one thing that matters right now

~400 teams sit at 0.9465 with `division_jaccard ≈ 0`. Rank 1 is 0.970. **The entire gap is the
division term**, and it is a two-constant bug the whole public field inherited:

| Gate | Deployed | Ground truth | Effect |
|---|---|---|---|
| `SAFE_DIV_MAX_UM` | 4.7 µm | median 5.7–6.4, p90 8.4–9.9 | excludes >50% of true divisions |
| `SAFE_DIV_SISTER_MAX_UM` | 7.2 µm | median 8.98 / 11.47 | excludes ~90% in embryo 6bba |

The ranking score `parent_dist + 0.15*sister_dist` is increasing in both — it ranks tightest
first, exactly backwards. Widening to (10, 15) takes admissible divisions from **10 to 81 of 89**.
CPU-only, hours. Full detail: `python3 tools/ask.py facts division`.

## Where knowledge lives

    python3 tools/ask.py status | gap | topics | kernels | expts | facts <topic> | todo
    sqlite3 db/biohub_base.db

`db/biohub_base.db` — 376 facts, experiments, decisions, the leaderboard, 240 public notebooks.
`db/DATABASE.md` — how it stores things and why. **Read this before writing a row.**
`knowledge/` — provenance-tracked notes. Start at `hub-biohub.md`.
`notes/RANK1-PLAN.md` — the plan and the day-by-day gates.
`repos/` — the three archived campaigns, cloned. Source of code to port, not to resume.

## Rules — non-negotiable

1. **A measured value lives in the DB once.** Prose cites it; prose never restates it. The prior
   campaign measured the cost: a superseded score appeared 244 times across 36 files while the
   live one appeared 31 times across 7.
2. **Every fact needs a `source_id` and a verbatim `quote`.** No quote, no fact.
3. **Never delete a contradicted fact.** Set `status='superseded'`, point `superseded_by` at the
   replacement, and fix its description.
4. **Declare the falsifier before the run.** A lever with no kill condition cannot be stopped.
5. **Run `python3 tools/sync_source_rows.py export`** after adding facts, decisions or
   experiments — `db/*.db` is gitignored and those rows exist nowhere else. `check` fails if the
   export has drifted.
6. **Never write to `~/.claude/`.** That store is machine-agnostic and Claude-session-scoped.
   Project knowledge belongs here.

## Already closed — do not re-pay for these

Measured negative by the prior campaign:

| Lever | Result |
|---|---|
| scalar-threshold re-acceptance | −0.023 |
| GT-free component selector | −0.008 |
| node-budget pruning | optimum was **no pruning** |
| split/merge arbitration | +0.0006 (noise) |
| edge-TTA | hurts (0.885) |
| detection-threshold lowering on 6bba | dead — 0.1% of GT is threshold-limited |

HOCT is not a drop-in linker. CoTracker loses morphology through divisions.

## Traps that have already cost time

- **Never select on `data/test`.** Its four movies are byte-identical copies of train volumes,
  verified by tree digest, and are swapped out at rerun.
- **The offline instrument was structurally broken** because a component trained on all 199
  labelled crops. Keep one embryo unseen by *every* component or the measurement is worthless.
- **An edgeless graph is unscorable** — `evaluate()` returns early and raises. It is a bug, not
  a low score.
- **`torch.autocast` is thread-local** — with `DataParallel` on T4 x2 it silently does not reach
  the replica threads.
- **Attach every Kaggle datasource before Save Version.** The save run cannot attach new ones.
- Public notebooks inject a hub node at `t = -1000` to game the metric. Check any borrowed
  pipeline for nodes at implausible timepoints before believing its numbers.

## Compute

Colab A100 for long training (`colab run --gpu A100 train.py`). Kaggle T4 x2 for free parallel
runs and submission-truthful numerics. The Mac for everything with no marginal cost — it is the
*slowest* of the three. Measured Colab burn on this account: ~1.4–1.65 units/hour.

**PyTorch only.** Not MLX: it cannot produce a submission, and torch 2.14 erased its conv3d
advantage. Keep `bias=True` on every `Conv3d`; build decoders from `Upsample`+`Conv3d`, never
`ConvTranspose3d` (unusable on MPS).

## Working alongside another agent

SQLite is single-writer even in WAL mode. **Do not run two agents writing to the base at the
same time.** Before starting: `git pull`, then `python3 tools/sync_source_rows.py check`. After
finishing: `export`, commit, push. If `check` fails, someone else's rows are unexported — export
them before adding your own, or they are lost on the next rebuild.

Scratch work goes in `~/Work/TestBench/BiohubTestBench/`, never here.
