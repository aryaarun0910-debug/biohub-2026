# Where we actually are

Last updated 2026-09-14. This file is the single honest answer to "what is the
state of this campaign". It supersedes nothing -- the review documents keep
their dated snapshots -- but if you read one file, read this one.

## Position

| | |
|---|---|
| Public LB | **0.947** |
| Rank | ~100-200 of 3,425 |
| Reproducible | yes -- submission #27 reproduced the public 0.947 kernel exactly |
| Calibration points | #27 (0.947), #28 (0.947, null), #29 (pending) |
| Deadline | 2026-09-29 23:59 UTC; entry/merge cutoff 2026-09-22 |

## The metric, exactly

    score = adj_edge_jaccard + 0.1 * division_jaccard
    adj    = max(0, J * (1.1 - 0.1 * N_pred / n_total))

The multiplier is bounded at **1.1** (as N_pred -> 0) and is exactly **1.0** at
full density. Theoretical max 1.1972. The leaderboard resolves **3 decimals**,
so anything under 0.001 is unobservable and must not be chased.

Exchange rates, measured:

- division break-even precision **24%** (one true division is worth 3.2 false ones)
- edge break-even precision **48.1%** (a false edge and a true edge move J nearly symmetrically)

## The structural law

Every candidate region in this problem has a base rate near **0.1%** against a
metric that needs 24-48% precision. That is a required lift of **150-320x**.
The features available give about **2x**. This is arithmetic, and it is why the
division and edge routes are closed rather than merely difficult.

## Sparse annotation -- the fact that governs everything

Only **2.82%** of cells are annotated (133,318 of 4,725,117). The asymmetry
between embryos is **7x** (44b6 = 0.77%, 6bba = 5.37%). Annotated density is
6.7 cells/frame; real density is **203-237** cells/frame -- a factor of **35**.

Two consequences that have each cost us real time:

1. **The sparse-annotation trap.** Counting unannotated-but-real items as false
   positives. It bit three separate times (a 98.86% "swap" base rate; edgeJ of
   0.03 against a true 0.85; a competitor's own `precision_sparse` reading 0.065).
   Any precision number computed against GT must be restricted to *judgeable*
   items.
2. **Wrong-regime tuning.** Everything tuned before EXP-29 was tuned at oracle
   density, 35x away from production. EXP-17 was overturned on exactly this.

## The four levers, and their status

| Lever | Pool | Status |
|---|---|---|
| edge_jaccard numerator (find missed true edges) | +0.05 | **closed** -- 0.53% detection / 5.3% linking base rate; 316x lift needed |
| edge_jaccard denominator (drop false edges) | +0.03 | **closed** -- EXP-31 cleared break-even, EXP-32 showed appearance adds nothing, #28 returned null on LB |
| 0.1 * division_jaccard | +0.04 | **closed** -- 2.0% detection / 32.5% selection / 65.6% contention; all three fail the 24% bar |
| N_pred multiplier | **+0.068** | **open** -- never directly attacked with a trained classifier |

### Why N_pred is hard rather than easy

97.3% of predicted nodes match no GT node. So **chance alone is 97.3% "safe"**
at removal. A classifier does not need to be good, it needs to beat 97.3%
precision. Removing 25% of unmatched nodes gives multiplier 0.9727 (enough to
win); removing all of them gives 1.0376. Their current multiplier is 1.0038 of
a possible 1.1000.

Every rule tried so far loses edge Jaccard faster than it gains multiplier:
spatial targeting (EXP-22), track-length pruning (EXP-34), their own output
scorer (v-outgrid). **EXP-36** is the first attempt to ask the question as a
supervised problem with the right target: not "is this node spurious?" but
"did a human annotator label this cell?".

## The false ceiling -- stated plainly

Everything in this repository is **post-hoc surgery on DeepCenter's output**.
The detector has never been changed. The post-processing ceiling is real and
has been mapped thoroughly across ~35 experiments. The **detection** ceiling is
completely unmeasured.

The leader at ~0.96 is very unlikely to be post-processing better. The public
forum evidence (dense finetune, elastic augmentation, dzyx bias,
velocity extrapolation) is a *training* story. The open technical problem is:

> how do you finetune a 3D U-Net centre-detector when only 2.8% of instances
> are labelled and unlabelled does not mean negative?

That is partial-label / positive-unlabelled learning for dense detection, and
it is the thing most likely to be worth more than 0.001.

## Reverse-engineering the 0.947 kernel

- Reproduced byte-identically (`kernels/repro-947/`).
- Their `ppsweep` tunes **in-kernel** on **8 held-out train stems containing 12
  GT divisions**, then applies its winner *over* the environment -- so env vars
  for swept parameters are silently ignored. This cost us two runs before it
  was understood.
- Their proxy over-reads the LB by **+0.0041**. This is a **constant offset**
  and therefore **cancels in a difference**. An earlier rule of mine ("a variant
  must beat baseline by >0.004") was wrong and would have discarded the only
  positive result in 38.
- Their sweep builds combinations only from *positive singles*, so it
  structurally cannot find a combination whose parts individually score badly.
  Submission #29 is exactly such a combination (`tight55+vel025+tight50relax9`,
  proxy +0.00132, where `vel025` alone is -0.00061). This is the one genuine
  structural edge we have over the public kernel.

## What remains

1. **EXP-36** -- matched-vs-unmatched node classifier. The only untested idea.
2. **v-fine1 / v-fine2** -- 34 configs around the winning combination. ~+0.001.
3. **Repair re-test** -- `gap_max_frames=0`, `min_track_len=1` were set at
   oracle density and never re-tested at real density. Expected ~0.000.

There is no fourth item. If EXP-36 returns AUC near 0.6, the post-processing
route is finished and the only remaining path is retraining the detector.

## Discipline that must not be relaxed

- **LOEO.** Train and test embryos are disjoint by host statement. Fit on one
  embryo, report on the other, take the **minimum**.
- **Paired bootstrap over datasets**, not over divisions. Unpaired, 151
  divisions resolve only +/-0.0068 of score -- wider than every effect we have
  ever measured.
- **Never `sorted(files)[:N]`.** That slice is entirely 44b6. It bit three
  times (EXP-18, EXP-26, EXP-32). Balance embryos explicitly.
- **Pre-check every submission against the in-kernel proxy.** This has already
  prevented two bad submissions.
- **A kernel run is not a submission.** Submit explicitly via the CLI.
- **`machineShape="NvidiaTeslaT4"`** selects the GPU. `acceleratorType` is
  accepted and silently ignored, and a P100 (sm_60) is fatal -- the pinned torch
  needs sm_70+.
