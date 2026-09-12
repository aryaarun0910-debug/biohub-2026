# Current campaign plan — validation before compute

The active decision is to hold paid compute until the correctness and packaging gates in
[FULL-REVIEW-2026-09-12.md](FULL-REVIEW-2026-09-12.md) pass. Query the DB topic
`review-2026-09-12` for the evidence and research behind this decision.

1. Fix validation, NMS scale, cache provenance and fold-ancestry blockers; pin the environment.
2. Prove a minimal raw-image detector → linker → CSV → official-score path locally.
3. Benchmark linking on contiguous dense physical crops with intact reference trajectories;
   label Zebrahub results as pseudo-label agreement and include both dense and sparse scoring.
4. Validate an offline Kaggle package and its complete dataset coverage before training.
5. After those gates, measure an explicitly budgeted A100 pilot on the actual installed stack.
6. Compare masked Gaussian detection with PAC-MAP targets and their matching decoder; add
   refinement only after a held-out composite-score gain. Revalidate all downstream thresholds
   on predicted dense detections.

There is no Mac-arrival dependency. Oracle results are conditional diagnostics, not a
submittable ceiling; EXP-18 is a confounded stress test, not a realistic-density benchmark.
The public notebook's local validation decomposition is not its hidden leaderboard decomposition.
Sparse annotation can support masked detector training. No full training, remote compute,
submission or rank improvement is claimed by this review.

The previous plan is preserved as
[RANK1-PLAN.superseded-2026-09-12.md](RANK1-PLAN.superseded-2026-09-12.md).
Its settled-linker, hardware-runtime and benchmark-ceiling conclusions are superseded.
