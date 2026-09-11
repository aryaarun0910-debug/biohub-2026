# Biohub Cell Tracking 2026 — Development Base

One queryable place holding everything the three campaigns learned plus live
competition intel, so the sprint on the MacBook starts from a base instead of
from scratch.

    python3 tools/ask.py status     # situation report: deadline, LB shape, repo state
    python3 tools/ask.py gap        # what separates the 0.947 plateau from the podium
    python3 tools/ask.py topics     # triaged discussion threads + our takeaways
    python3 tools/ask.py kernels    # public notebooks ranked by score, with family labels
    python3 tools/ask.py expts      # every lever already tried, and what killed it
    python3 tools/ask.py facts prior-campaign
    python3 tools/ask.py todo       # open decisions

## Knowledge lives here too

Split out of the Claude memory store on 2026-09-11 so competition knowledge does not auto-load
into unrelated sessions, and does not become permanent noise after the deadline.

- `knowledge/` — provenance-tracked notes. Start at `knowledge/hub-biohub.md`.
- `db/` — the measurements. Notes cite the base; they never restate a number from it.

Machine-level facts that outlive this project (what the Mac/Kaggle/Colab can run, the torch-2.14
conv3d finding) stay in the portable store at `~/.claude/projects/<slug>/memory/`. Links that
cross that boundary appear as explicit paths like `claude-memory/macbook-m5-pro-ml.md`.

## Layout

| Path | What |
|---|---|
| `db/biohub_base.db` | SQLite. The whole base. Schema in `db/schema.sql`. |
| `tools/ingest_kaggle.py` | Competition intel from frozen evidence → DB. Idempotent. |
| `tools/ingest_repos.py` | Clones the 3 private repos, loads commits/files → DB. Re-run after each session. |
| `tools/ingest_registries.py` | Prior campaigns' `facts.yaml` / `levers.yaml` / ledgers → DB. |
| `tools/ask.py` | Read-only views. |
| `tools/sync_source_rows.py` | Export/restore the rows no ingester can rebuild. **Run `export` after any session that adds facts, decisions or experiments.** |
| `evidence/raw/` + `evidence/index.jsonl` | Frozen Kaggle responses, sha256-indexed. Nothing here is a guess. |
| `reference/royerlab-baseline/` | The organizers' own baseline + the **authoritative metric source**. |
| `repos/` | Working clones of the three private repos. |

## What is in the DB

| Table | Rows | Holds |
|---|---|---|
| `fact` | 317 | Competition facts + all 270 prior-campaign registry facts, with provenance, confidence and superseded/invalid status preserved |
| `experiment` | 57 | 46 levers, 5 closed dead-ends with measured deltas, 6 submissions |
| `lb_snapshot` | 3375 | Full public leaderboard, dated — re-run the ingester to track movement |
| `forum_topic` | 102 | Every discussion thread, 16 triaged with takeaways |
| `public_kernel` | 200 | Public notebooks with score, runtime and a `family` label |
| `artifact` | 243 | The Biohub-X archive branch: 20 checkpoints, 5 submission CSVs, 218 measurement reports |
| `repo` / `repo_commit` / `repo_file` | 3 / 672 / 1940 | The three repos' history; `reusable=1` flags files worth porting |
| `decision` | 5 | Open strategic calls, with rationale |

## Rules this base enforces

Carried from the prior campaigns, because they are why their numbers can be believed:

1. **A measured value lives in the registry once.** Prose cites it; prose never restates it.
   The monolith measured the cost of breaking this: a superseded score appeared 244 times
   across 36 files while the live one appeared 31 times across 7.
2. **Declare the experiment and its falsifier before running it.**
3. **Bind every artifact by content digest and verify before loading.**
4. **Provenance and validity are separate axes.** A number can be correctly computed from an
   invalid run — that is exactly how the `EXP-0019` leave-one-embryo-out leak propagated to
   seven downstream facts before it was caught.

## Refresh

    python3 tools/ingest_repos.py        # after every work session
    python3 tools/ingest_kaggle.py       # after re-harvesting Kaggle (LB moves daily)
    python3 tools/ingest_registries.py   # if the registries change
    python3 tools/sync_source_rows.py export   # ALWAYS, after adding rows by hand

`db/*.db` stays gitignored - it is a 1.8 MB binary that churns and will not merge. But 63 rows
(12 decisions, 51 analysis facts) exist nowhere else, so they are mirrored to the tracked
`db/source_rows.jsonl`. `sync_source_rows.py check` fails if that export has drifted; `import`
restores them after a rebuild. Round-tripped on 2026-09-11: 63 out, 63 back.
