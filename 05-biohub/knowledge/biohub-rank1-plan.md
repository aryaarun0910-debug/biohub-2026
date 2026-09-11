---
name: biohub-rank1-plan
description: Pointer - the Biohub sprint plan lives at ~/Work/biohub/notes/RANK1-PLAN.md in git, not in memory; this note says what it contains and what supersedes what
metadata:
  type: project
provenance:
  claim_type: decision
  status: active
  observed_at: 2026-09-11
  valid_from: 2026-09-11
  valid_until: null
  superseded_by: null
  review_after: 2026-09-30
  confidence: high
  sources:
    - method: user-stated
      quote: "Our aim is Rank 1"
      at: 2026-09-11
    - method: measured
      command: "git -C ~/Work/biohub log --oneline"
      at: 2026-09-11
  author: claude
  verified_at: 2026-09-11
---

Arya's stated aim is **rank 1**. The plan is a versioned file, not a memory note, because it
changes daily and is edited alongside the code it governs:

    ~/Work/biohub/notes/RANK1-PLAN.md      (git-tracked)

It supersedes the day-by-day shape in `repos/Biohub-Sprint-2026/05-SPRINT-PLAN.md`, which was
written against a different machine-arrival date. The rest of that handoff pack still stands and
should be read first — `01-START-HERE.md` through `05`.

It carries the runway, the strategic decision not to ship a metric exploit and why, the primary
honest lever, the per-day gates with their falsifiers, and a hardware section covering what
actually maximises the Mac.

Numbers in it are snapshots. The live ones are in the base — see [[biohub-dev-base]] and
[[hub-biohub]]. Competition constraints: [[biohub-competition-2026]].
