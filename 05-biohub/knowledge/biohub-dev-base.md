---
name: biohub-dev-base
description: ~/Work/biohub - the SQLite development base consolidating all three Biohub repos, prior-campaign registries and live Kaggle intel; query it with tools/ask.py before re-deriving anything
metadata:
  type: project
provenance:
  claim_type: observation
  status: active
  observed_at: 2026-09-11
  valid_from: 2026-09-11
  valid_until: null
  superseded_by: null
  review_after: null
  confidence: high
  sources:
    - method: measured
      command: "python3 tools/ingest_repos.py && python3 tools/ingest_registries.py"
      quote: "commits=672  files=1940  reusable=200 ; facts.yaml -> 270 facts ; levers.yaml -> 46 levers"
      at: 2026-09-11
  author: claude
  verified_at: 2026-09-11
---

Built 2026-09-11 so the MacBook sprint starts from measured knowledge rather than scratch.
Lives at **`~/Work/biohub`**; the base itself is `db/biohub_base.db`.

    python3 tools/ask.py status | gap | topics | kernels | expts | facts | todo

Holds 322 facts (270 of them the monolith's own registry, with `provenance` *and* `validity`
preserved so superseded and leak-tainted numbers stay flagged), 57 experiments (46 levers,
5 measured-negative dead ends, 6 submissions), the full 3,375-row public leaderboard, 102
discussion threads with 16 triaged, 200 public notebooks labelled by family, 243 archived
artifacts including 20 checkpoints, and all 672 commits / 1,940 files across the three repos.

**The three repos** (`aryaarun0910-debug`, all private, clones under `repos/`):
`Biohub-CellTracking-2026` is the monolith — 535 commits, best **0.925**; `Biohub-X` is the
blind restart — 136 commits, best 0.496, whose `archive/evidence-2026-09-07` orphan branch
carries the 243 artifacts that `.gitignore` excluded (never merge it); `Biohub-Sprint-2026`
is a 6-file handoff pack written 2026-09-07 — **read `01-START-HERE.md` first.**

**Already closed, with measured deltas — do not re-pay for these:** scalar-threshold
re-acceptance −0.023; GT-free component selector −0.008; **node-budget pruning: optimum was
no pruning at all**; split/merge arbitration +0.0006 (noise); edge-TTA hurts; HOCT is not a
drop-in linker; CoTracker loses morphology through divisions.

**Two traps the monolith paid for.** `EXP-0019` scored the `6bba` embryo with weights trained
on `6bba` — a leave-one-embryo-out leak that propagated to seven facts before it was caught,
so provenance and validity are tracked separately. And its offline proxy repeatedly failed to
predict leaderboard *direction*: DeepCenter read +0.0009 offline against +0.003 on the LB,
`icom` read +0.002 offline and scored −0.014.

Refresh with `tools/ingest_repos.py` after each session. See [[biohub-competition-2026]] and
[[biohub-rank1-plan]].
