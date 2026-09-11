---
name: hub-biohub
description: HUB - entry point for the Biohub cell-tracking campaign; says which questions memory answers and which ones the SQLite base answers
metadata:
  type: reference
provenance:
  claim_type: inference
  status: active
  observed_at: 2026-09-11
  valid_from: 2026-09-11
  valid_until: null
  superseded_by: null
  review_after: null
  confidence: high
  sources:
    - method: unrecorded
      quote: "navigation only - this note asserts no measured values"
      at: 2026-09-11
  author: claude
  verified_at: 2026-09-11
---

**This is a hub. It holds no scores, no dates and no measurements** — those rot daily and belong
in the base, not here.

**The split that makes this work.** Experiment records are high-churn, numeric and want to be
queried, ranked and filtered. Markdown memory is none of those things. So:

- **Measurements live in SQLite** — `~/Work/biohub/db/biohub_base.db`, described by
  [[biohub-dev-base]]. Every run, lever, submission, leaderboard row and public notebook, each
  with provenance *and* validity. Query it with `python3 tools/ask.py`. Never copy a number out
  of it into a note; cite the table instead.
- **Durable conclusions live in memory** — the handful of things that stay true after the
  campaign ends, and that you would want recalled in an unrelated session a year from now.

If you find yourself about to write a score into a note, that is the signal you wanted a row.

**The campaign.** [[biohub-competition-2026]] — the task, the metric's shape, the submission
constraints, and why the leaderboard looks the way it does. [[biohub-dev-base]] — the three
repos, what the two prior campaigns closed off, and the two methodological traps they paid for.
[[biohub-rank1-plan]] — the plan and where it lives.

**Hardware for it:** `claude-memory/hub-compute.md`. Two facts in `claude-memory/macbook-m5-pro-ml.md` constrain the model
architecture directly and should be read before any code is written.

**Method carried from the prior campaigns**, and the reason their numbers can be believed at all:
one registry is the only home for a measured value and prose cites it rather than restating it;
declare the experiment's falsifier before running it; bind every artifact by content digest; and
keep provenance separate from validity, because a number can be correctly computed from an
invalid run.
