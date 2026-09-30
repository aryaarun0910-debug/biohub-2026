---
name: biohub-competition-2026
description: Kaggle "Biohub - Cell Tracking During Development" (id 136605) - kernels-only, $60k, closes 2026-09-29; metric, data shape and dated evidence navigation
metadata:
  type: project
provenance:
  claim_type: claim
  status: active
  observed_at: 2026-09-11
  valid_from: 2026-06-29
  valid_until: null
  superseded_by: null
  review_after: 2026-09-29
  confidence: high
  sources:
    - method: curl
      url: "https://www.kaggle.com/api/i/competitions.CompetitionService/GetCompetition {competitionId:136605}"
      sha256: 895b66b7bd23
      quote: '"deadline":"2026-09-29T23:59:00Z","onlyAllowKernelSubmissions":true,"maxGpuRuntimeMinutes":720,"totalTeams":3372'
      at: 2026-09-11
    - method: curl
      url: https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md
      sha256: d8a2d3ff21b5
      quote: "adjusted_jaccard = max(0, jaccard - (1 - a - (T_pred - T_true) / T_true))"
      at: 2026-09-11
  author: claude
  verified_at: 2026-09-11
---

The task is cell detection, tracking and division identification in 3D time-series microscopy.
Query `python3 tools/ask.py status` for dated competition metadata and
`python3 tools/ask.py facts metric` for recorded metric findings. These are frozen evidence,
not a live confirmation of rules, deadlines or the leaderboard.

Aggregate leaderboard scores do not establish competitors' component scores or methods.
The current approach is in [[biohub-rank1-plan]]. Preserve embryo separation through every
learned component, and never select on visible test data. See [[biohub-dev-base]] for
historical corrections and provenance. Verify current rules at the competition before
submitting or introducing external data.
