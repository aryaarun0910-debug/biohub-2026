# Campaign plan — post-processing on a reproduced 0.947

Supersedes [RANK1-PLAN.superseded-2026-09-14.md](RANK1-PLAN.superseded-2026-09-14.md), whose
plan was "beat it on divisions". Divisions are now closed by measurement (EXP-38b: divTP never
moved across eight gate relaxations). The full evidence trail is in
[STATE.md](STATE.md) — read that first; this file is only the forward plan.

## Position

| | |
|---|---|
| Public LB | **0.947**, reproduced exactly (#27) |
| Best measured variant | ppgrid + OUTPUT_MIN_TRACK_LEN=9, **+0.0082** on the four real test movies |
| Deadline | 2026-09-29 23:59 UTC; entry/merge cutoff 2026-09-22 |

## What is closed, and why

All four scoring terms were attacked and three are shut by base-rate arithmetic:

- **divisions** — EXP-38b, no gate relaxation adds a single true division
- **edge numerator** — EXP-40/41, only 5.5% of missing edges have both endpoints free
- **edge rewiring** — EXP-43, the linker beats nearest-neighbour 53:7 on disagreements
- **N_pred by node selection** — EXP-36, annotation is not predictable (AUC 0.641/0.557)
- **detection** — EXP-37, node recall 0.9979; and better detection *lowers* the multiplier

The residual has one shape every time: ~1-2% base rate against a 24-48% precision bar.

## What is open

**The multiplier, via component pruning.** It is the only lever whose gain is
annotation-independent: it depends solely on N_pred and n_total, both known exactly for all four
test movies. Currently 1.0138 of a possible 1.1000, though the ceiling is not attainable (it sits
at zero predicted nodes).

**The sweep's selection objective.** This is the live idea. Their in-kernel sweep picks a config
by weight-averaging over 8 held-out TRAIN stems, which mix to 44b6 25.9% / 6bba 74.1%. The real
test set weights **44b6 4.6% / 6bba 95.4%**. Every public fork has therefore been selecting the
config that wins on an embryo mix the test set does not contain. `kernels/biohub-rw-ml9` scales
44b6 weights by 0.1379 so the proxy's mix matches the test's.

This is also the principled version of an accident: ppgrid+minlen9 gained +0.0082, more than the
+0.0063 its parts predict, because pruning perturbed the landscape enough that the sweep
*stumbled* onto a better config. Reweighting looks for those deliberately.

## Method rules earned the hard way

1. **Score variants with `tools/score_submission_local.py`** on a REAL kernel submission, against
   the four test movies. Their in-kernel proxy is mis-weighted; our cached train graphs are a
   different pipeline; pre-filter geffs are a different stage. Substrate errors cost three
   experiments in one day.
2. **Validate an instrument against a known answer before trusting it.** `tools/fastpp.py` looked
   like a 6x speedup and under-read the one case we could check by 3x.
3. **Check the break-even** `eps/J < delta/(m+delta)` before spending a run on pruning.
4. **The multiplier is exactly computable offline** — `tools/multiplier.py`. Never infer it from
   leaderboard deltas.
5. **A kernel run is not a submission**, and `machine_shape="NvidiaTeslaT4"` is what binds.

## Next

- Read #29 / #30 / #31 when they score; they test whether the local proxy SIZES gains or only
  detects nulls.
- Run `biohub-rw-ml9` (built, waiting on a slot).
- If reweighting works, sweep a wider PP_CANDIDATES table under the corrected objective.
