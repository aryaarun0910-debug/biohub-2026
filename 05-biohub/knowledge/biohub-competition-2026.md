---
name: biohub-competition-2026
description: Kaggle "Biohub - Cell Tracking During Development" (id 136605) - kernels-only, $60k, closes 2026-09-29; metric, data shape and the leaderboard's exploit-driven plateau
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

Arya's target is **rank 1**. Competition id **136605**, forum id 10656304.

**Hard shape.** Kernels-only (`onlyAllowKernelSubmissions: true`) — the submission *is* a
Kaggle notebook, internet disabled, <=12 h GPU, output `submission.csv`. Locally trained
weights can only reach it as an attached Kaggle Dataset/Model. Apple frameworks do not
exist there, which is why MLX cannot be on the submission path. 5 subs/day, 2 final
selections, max team 5, public LB = 29% of test, scores truncated to 3 decimals.
Deadline 2026-09-29; entry and team-merge cutoff 2026-09-22. $60k over 7 prizes.

**Task.** 3D+time zebrafish light-sheet, Zarr v3 `(T,Z,Y,X)`, typically `(100,64,256,256)`
uint16. Voxels are **4:1 anisotropic in Z** (z=1.625, y=x=0.40625 um). Ground truth is
**sparse** — unannotated voxels are not background, which is why both prior campaigns
converged on positive-unlabelled losses. Train/test are **embryo-disjoint** and train holds
only ~2 embryos (`44b6`, `6bba`), so generalisation is the core risk.

**Metric.** `score = adj_edge_jaccard + 0.1 * division_jaccard`, higher is better. Node
matching is optimal bipartite on centroid distance within 7 um. The adjustment is
`max(0, J * (1 - 0.1*(N_pred - n_total)/n_total))` and **has no clamp for `N_pred < n_total`**,
so under-predicting nodes multiplies the score by up to 1.1 — which is why scores can exceed
1.0. Predicted nodes matching no GT node are *not* edge FPs, so a spurious detection costs
only multiplier. The host already patched a division "weakly connected component" exploit
(commit `aa65e90`, 2026-07-17) and rescored.

**The leaderboard is exploit-shaped.** Notebooks labelled "metric hack" score 0.950-0.966 in
**6-16 minutes**; the best honest ML notebooks reach 0.947 in ~110 minutes. ~400 teams sit on
the 0.946-0.947 public-notebook line. Only 63 teams are >=0.948, 11 >=0.957, 6 >=0.960, and
one at 0.970. See [[biohub-dev-base]] for the measured detail and [[biohub-rank1-plan]].

Rules §3.148 permits disqualification for "unfair playing practices or abuses", and winners
must deliver training code, inference code and a reproducible methodology under MIT, possibly
defended on a recorded call. External data *is* allowed if public, free and equally accessible.
