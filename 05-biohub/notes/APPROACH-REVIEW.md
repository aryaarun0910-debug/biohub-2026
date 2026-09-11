# Approach review and change rationale

## Assessment

Keeping archived measurements and avoiding an immediate model rewrite are sound choices.
The active plan nevertheless treats selected historical inferences as established results.
The highest priority is a reproducible baseline with trustworthy evaluation, followed by
paired ablations. A rank target does not establish which model component needs work.

### The division-first conclusion outruns its evidence

The active instructions attributed the entire leaderboard gap to divisions. An aggregate
leaderboard score cannot identify its edge and division components. It also cannot show
which constants another team uses. Notebook titles and runtimes are clues, not an audit.

More seriously, the archived registry explicitly withdraws the earlier edge-parity argument:
see `prior-campaign` keys beginning `FACT-0161` and `FACT-0162`. The latter says:

> The strategic conclusion it produced is withdrawn.

The division forensic report remains useful; the later registry explicitly reconciles it
with honest out-of-fold measurements. Its gate-admissibility census is an oracle diagnostic,
not a measured gain from a deployable proposer. It does not account for all added false
positives, candidate contention, or damage to the edge term. The ranking experiment is
already addressed by `FACT-0174`; rerunning it without a changed premise wastes runway.

**Change:** replace the old calendar and assumed component targets with ordered validation
gates. Keep widening as a preregistered hypothesis, with promotion determined by the official
composite score on both embryo directions. Preserve contradicted records and link corrections.

### A leave-one-embryo-out label is insufficient

Every learned component must exclude the evaluation embryo: detector, edge model, calibration,
normalization fitted from data, and any learned postprocessing. Cached predictions inherit
their producing checkpoints' training exposure. A different manifest name does not make an
all-data checkpoint out of fold. Repeated tuning against both embryos also makes them
validation data; it is not an independent final generalization test.

**Change:** require immutable input/checkpoint/code/scorer identities and training membership
before scoring. Keep crop-level paired diagnostics, but do not claim crop bootstrap intervals
measure uncertainty across new embryos. Report both directions; do not hide one in a pool.

### The persistence contract did not hold

The former source-only export omitted source IDs, supersession links, experiment start times,
repository links, and artifacts. Decisions and experiments had no exported identity, so repeat
imports appended duplicates. Topic and string-prefix ownership guesses also omitted records
that had no implemented rebuild path. The schema conflated validity with confidence/status.

**Change:** snapshot the complete knowledge core: sources, repositories, facts, experiments,
artifacts, decisions, and competition snapshots/annotations. Keep only repository history/file
indexes disposable.
Export every column, deterministically and atomically. Validate restores in a temporary SQLite
database, preserve IDs and foreign keys, permit identical repeat restores, and refuse to merge
into different existing knowledge. Add a separate validity field with UNKNOWN as the honest
default for legacy records. Missing quotes and sources are review debt, not silently repaired.

### Refreshing could fabricate freshness or erase corrections

Kaggle ingestion dated frozen evidence using the clock at rebuild time and did not verify the
index. Replacement inserts could delete a fact's identity; registry refresh deleted experiments
that artifacts might reference. Repository fetch failures were ignored, then reported as success.

**Change:** verify frozen evidence hashes and use its acquisition timestamps; use conflict updates
for facts and stable experiment identities; retain curated forum/kernel annotations. Fail repo
refreshes visibly and roll back each failed repository's ingest. Show the selected leaderboard
snapshot date, and select every rank from that same snapshot.

### Execution and packaging remain work, not demonstrated capabilities

This repository tracks a knowledge base and an upstream reference implementation. The campaign
pipelines are ignored working clones. It has no current, pinned campaign training/inference
package or competition data here. The reference package follows a moving dependency branch,
so its presence alone does not establish a reproducible scorer environment. The Colab guard
also depended on an untracked sibling script and overclaimed that script exit status proved
CPU parity or automatic teardown.

**Change:** document these boundaries and require a pinned, executable submission baseline as
the next milestone. No model-quality improvement, Kaggle acceptance, GPU parity, or billing
behavior is claimed by this maintenance review. Validation for these code changes is the
repository's knowledge-tool regression suite and an actual clean restore/rebuild.

## Next experiments and stopping rules

Follow [RANK1-PLAN.md](RANK1-PLAN.md). Historical negative results remain scoped to their
substrate and validity. Reopen one only for a recorded change in premise or corrected evidence.
Hardware purchase dates, billing estimates and copied leaderboard levels are not experiment
acceptance criteria. Verify current platform constraints before scheduling a paid run.
