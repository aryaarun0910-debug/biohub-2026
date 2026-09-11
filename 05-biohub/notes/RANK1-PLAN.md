# Campaign plan

This plan supersedes the former date-by-date plan in this file. The reasoning and evidence
pointers are in [APPROACH-REVIEW.md](APPROACH-REVIEW.md). Scores and competition metadata belong
in the knowledge database; `tools/ask.py status` labels the frozen snapshot date.

## Gate 1: reproduce the instrument and a runnable baseline

Start now on available hardware. Port the smallest useful path from the archived campaigns:
data contracts, classical detector, linker, official metric adapter, submission writer and
validator. Commit this path into the active repository before extending it. Pin the scorer,
its transitive dependencies, training/inference environment and checkpoint digests. Keep the
vendored upstream source intact and put campaign adaptations outside `reference/`.

Record the training embryos of every checkpoint and fitted transform. Validate both
leave-one-embryo-out directions with no component exposed to the evaluation embryo. Reject
unknown training membership, missing digests, stale caches, or mismatched scorer versions.
Do not use visible test movies for selection. Preserve predictions and per-crop score counts.

**Stop condition:** if the input/artifact audit or scorer fails, the run is invalid and cannot
support any model conclusion. An edgeless graph is an explicit evaluation failure, not zero.

Package and run the baseline with Kaggle's actual offline constraints, attached artifacts and
required output schema. Record acceptance and runtime before investing in long training.

## Gate 2: measure the error budget

Report official adjusted edge, division and total scores separately for each embryo direction,
alongside division TP/FP/FN, detection reachability and runtime. Use the official aggregation,
not an average of crop Jaccards. Compare candidate and baseline on identical evaluation inputs.

Treat archived findings as substrate-specific. Consult `prior-campaign/FACT-0161`, `FACT-0162`
and `FACT-0174` before making an edge-parity or division-ranking argument. The current
leaderboard does not expose a component decomposition for arbitrary competitors.

**Decision:** prioritize the largest demonstrated, recoverable score loss on the clean baseline.
Do not require local score levels to equal leaderboard levels from another population.

## Gate 3: one division ablation at a time

The historical `division/gate_widening_table` motivates a proposer experiment. It does not
establish its final precision or composite-score benefit. Freeze a small candidate set before
the run. Tune thresholds/ranking only on the training embryo for each direction; apply them
unchanged to the evaluation embryo. Mark any choice already informed by both embryos as
exploratory validation, not an untouched test.

Compare the incumbent with gate widening alone; only then consider angular or persistence
features. Keep the detector, node set and unrelated linking rules fixed where possible.
Record forks at matched mothers as a diagnostic, plus final TP/FP/FN and both score terms.

**Falsifier:** reject a candidate that fails to improve the official total score in either
embryo direction, violates graph contracts, or exceeds the submission runtime budget. A rise
in admissible candidates or proposed forks alone does not pass. Preregister the minimum
practical score gain and compute budget with the experiment before running it.

## Gate 4: promote and freeze

Promote only a reproducible paired improvement that passes both directions and a submission
smoke run. Keep the incumbent ready to submit throughout the campaign. Choose final artifacts
using the declared validation policy, not repeated public-leaderboard tuning. Leave a buffer
before the recorded competition deadline for a full offline rerun and artifact recovery.

## Compute discipline

Use PyTorch for the submission path. Select devices at runtime. Benchmark the installed
versions and actual operators before hardware optimization; verify numerical parity and runtime
on the submission hardware. No hardware purchase or assumed billing rate gates CPU work.
Pin a working environment before optimizing it. Record actual paid-session cost and explicitly
verify termination; a local CLI exit is not evidence that a remote session stopped.
