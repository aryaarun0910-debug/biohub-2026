# SYSTEM

Architecture, contracts, and the honest current state of Biohub-X.

## 1. Current state

**Phase 2 complete. The system detects, associates and emits its own legal lineage graph, on a synthetic fixture only.**

What exists:

| Component | Module | State |
| --- | --- | --- |
| Content identity | `biohubx.hashing` | implemented, contract-tested |
| Artifact records and atomic writes | `biohubx.artifacts` | implemented, contract-tested |
| Typed CLI | `biohubx.cli` | five commands implemented |
| Coordinate contract | `biohubx.contracts.coordinates` | implemented, contract-tested with a negative control |
| Legal-lineage contract | `biohubx.contracts.lineage` | implemented, one mutation test per required refusal |
| Pinned official metric | `biohubx._vendor.official_competition` | vendored byte-identical, digests locked |
| Metric adapter | `biohubx.evaluation.official_metric` | implemented, fifteen calibration fixtures |
| Read-only data validation | `biohubx.data.validation` | implemented, understands the official split layout |
| Dataset content identity | `biohubx.hashing` (tree digest) | implemented, contract-tested |
| Synthetic fixture | `biohubx.data.synthetic` | deterministic movie with a known lineage |
| Instance, representation, candidate, prediction contracts | `biohubx.contracts.*` | implemented, one mutation test per rule |
| Classical detector | `biohubx.proposals.classical` | deterministic, physical-radius suppression |
| Geometry representation | `biohubx.representation.geometry` | implemented, no defaulted channel |
| Candidate graph and reach | `biohubx.tracking.candidate_graph` | implemented, reach measured not assumed |
| Untrained matcher | `biohubx.tracking.matcher` | explicit no-parent option, hand-set constants |
| Constrained decoder | `biohubx.tracking.graph_decoder` | greedy, emits a legal graph |
| Vertical slice | `biohubx.tracking.pipeline` | detect to score, one owner |

What does not exist yet: any trained model, any learned representation, any
division model, temporal context beyond one frame, alternative decoders, and
deployment. Those modules are absent from `src/biohubx/`, not stubbed. A module
is created when its first real consumer exists.

No score has been measured on competition data. No model has been trained or
downloaded. No GPU has been used. No Kaggle notebook exists. Every number this
repository has produced comes from synthetic fixtures whose purpose is to
characterise the scorer and to prove the software owns its graph. The matcher is
an untrained rule with hand-set constants, so its score is a property of those
constants and of a fixture far easier than real data.

### 1.1 What Phase 1 established

The metric mathematics are not reimplemented. `src/biohubx/_vendor/` holds a
byte-identical copy of the official `tracking_cellmot` scorer at commit
`075fc5f5a52d11077f9dc2b074644618f26939e2`, with typed digests recorded in
`registry/official_source.yaml` and re-derived by `biohubx official
verify-source`. The adapter validates Biohub-X contracts, converts to the
official graph type, and calls the official functions; it computes nothing.

The node-count denominator of the adjusted Jaccard is an explicit,
provenance-tagged quantity (`EstimatedTotalNodes`) with no default. The official
metric divides by a coarse estimate of *every* cell, which on real data is the
dataset's `estimated_number_of_nodes`. Annotations are sparse, so the annotated
count is a different and smaller number. Substituting one for the other would
report an adjusted Jaccard that is not comparable with the competition's, so the
substitution has to be named at every call site.

The official scorer silently drops edges that span more than one frame. Biohub-X
refuses them at construction instead, so an illegal edge can never reach the
scorer and be quietly forgiven. The same holds for backward edges, in-degree
above one, out-degree above two, dangling endpoints, cross-dataset edges,
duplicate identities and non-integer identities.

### 1.2 What Phase 2 established

The whole chain runs without competition data, a GPU or any external artifact:

```
synthetic volume -> candidate instances -> representation -> sparse T=2
candidate graph -> scored options including no-parent -> legal lineage graph
-> official score
```

Three properties are enforced rather than hoped for. Candidate generation and
candidate scoring are separate contracts, and the candidate graph reports how
many true edges it actually offered, so a low score can be attributed to the
detector, the radius or the matcher rather than blamed on whichever is nearest.
Abstention is a scored row competing on the same axis as the parents, computed
from the evidence of the target itself, so it can express that a cell is new
rather than only that the best parent is not good enough. The decoder returns a
`LineageGraph`, which makes legality a type rather than a convention.

The constants in the untrained matcher live in one named object, so the learned
matcher that replaces them has a contract to satisfy and a baseline to beat.

## 2. Target pipeline

```
raw microscopy volumes
    -> multi-hypothesis instance/center proposals
    -> per-cell appearance and morphology representation
    -> physical geometry and local tissue-motion representation
    -> sparse temporal candidate graph
    -> learned association with explicit abstention
    -> joint continuation/division reasoning
    -> global lineage decoder
    -> official metric
    -> thin frozen Kaggle inference artifact
```

The organising principle is that irreversible decisions are delayed until
complementary evidence has been combined: appearance, instance morphology,
physical geometry, local tissue deformation, temporal identity, higher-order
parent/daughter structure, explicit no-link uncertainty, and global lineage
constraints.

The system owns the graph it emits. It is not a chain of post-processing
scripts operating on someone else's output.

## 3. Systems

At most three systems exist at once.

| System | Role | State |
| --- | --- | --- |
| A - Biohub-X matcher | multi-source representation, learned abstention, continuation, division, full graph decode | not started |
| B - Trackastra-style owner | temporal-window graph owner consuming the identical frozen proposal/representation export | not started |
| C - promoted hybrid | does not exist until A or B wins independently and the loser shows complementary errors | forbidden to create |

System B consumes the same frozen export as System A and emits its own final
graph. It is never used as a pruner or post-processor for System A.

A fourth top-level system may not be added without replacing one of these.

## 4. Contracts

Contracts are the interfaces that downstream components may rely on. They live
in `src/biohubx/contracts/` and each arrives with the component that first
consumes it.

### 4.1 Content identity (implemented)

Three identities, never interchangeable:

- `raw_artifact_sha256` - SHA-256 of the exact bytes on disk. The identity of
  weights, exported graphs, built notebooks, submissions, and hash-bound
  snapshots.
- `canonical_text_sha256` - SHA-256 after canonicalization `v1`. The identity
  used for source and configuration drift.
- `tree_sha256` - SHA-256 over a canonical listing of a whole directory tree
  under tree canonicalization `v1`. The identity of a dataset artifact, which is
  a directory of many thousands of chunk files and has no single-file digest.

Canonicalization `v1`: strict UTF-8 decode, strip BOM, CRLF and lone CR to LF,
append one trailing LF if the text is non-empty and lacks one. Trailing
whitespace and blank lines are preserved. Bumping the version invalidates every
recorded canonical digest and requires a DECISIONS.md entry.

Serialised form always states the kind, and the canonicalization version when
one applies:

```
raw_artifact_sha256:sha256:<64 hex>
canonical_text_sha256:sha256/v1:<64 hex>
tree_sha256:sha256/v1:<64 hex>
```

Tree canonicalization `v1`: walk the tree, judging every entry before descending
into it; refuse any reparse point at the root or below it, whether a symlink, a
Windows directory junction, or any other reparse tag, and refuse anything that
is neither a regular file nor a real directory; record each file as its relative
POSIX path, size and content digest, and each EMPTY directory as itself; sort by
path; join with LF. The tree must be quiescent for the whole pass: its structure
is snapshotted before and after, and a digest computed over a moving tree is
refused rather than reported. Paths are relative to the root, so moving or renaming an artifact
does not change what the data is, while moving a chunk inside it does. The text
and tree canonicalization versions are independent and are not interchangeable,
which is why the kind is part of the token rather than implied by the version.

A bare hexadecimal string is not an identity and is refused by
`biohubx.hashing.Digest.parse`.

Repository byte policy is fixed in `.gitattributes` and was set before the first
digest was recorded: text is stored and checked out with LF everywhere, and
`research/primitives-dossier.md` is marked `-text` so its recorded raw digest is
an assertion about the original source bytes.

### 4.2 Coordinates (implemented)

Physical `(z, y, x)` derived from official voxel spacing `(1.625, 0.40625,
0.40625)` micrometres. The module preserves both voxel and physical coordinates,
declares axis order on every value, round-trips within a stated tolerance, and
refuses an isotropic scale at construction. No downstream component receives an
anonymous three-column coordinate tensor.

The negative control is concrete rather than nominal: five voxels along z is
8.125 micrometres and falls outside the 7 micrometre matching radius, while the
identical five voxels along x is 2.03 micrometres and falls inside it. An
isotropic convention collapses the two onto the same value and onto the wrong
side of the radius, so a system built on it would be scored against a different
graph than the one it believes it emitted.

### 4.3 Lineage (implemented)

`LineageGraph` is a complete legal graph ready for integer export and official
scoring. It refuses, at construction: duplicate node identities, dangling
endpoints, duplicate candidate pairs, in-degree above one, out-degree above two,
cross-dataset edges and nodes, edges that do not advance exactly one frame
(which covers backward, same-frame and frame-skipping edges), non-integer and
negative identities and frames, and an edge whose declared kind disagrees with
the topology at its source. Division is a set of two daughters, so daughter
order does not change the export.

### 4.4 Instances, representation, candidates, predictions (implemented)

- `CandidateInstance` carries both coordinate systems and validates that they
  agree, plus the detector that proposed it and that detector's confidence.
- `InstanceFeatures` names every channel and defaults none. A missing feature
  raises, because a zero-filled channel is indistinguishable from a genuine
  measurement of zero.
- `CandidateGraph` separates generation from scoring and carries a `ReachReport`
  whose counts must add up, distinguishing a detector miss from a radius miss.
  An absent reach report is absent, not a reach of zero.
- `TargetPrediction` requires a no-parent score alongside the parent scores, and
  abstention wins ties. The no-parent state is structurally explicit here. It is
  not yet learned, and the constants that fill it are not evidence.

Still to come: appearance and morphology embeddings, and the learned weights
that make the hand-set matcher constants unnecessary.

### 4.5 CLI (partially implemented)

Every command accepts a typed configuration, prints a positive heartbeat,
writes outputs atomically, writes a manifest, refuses missing or ambiguous
inputs, records source and artifact digests, and never silently falls back to a
default data path.

Implemented:

- `biohubx artifacts digest PATH --kind {raw_artifact_sha256,canonical_text_sha256}`
- `biohubx artifacts verify [--registry PATH] [--deep]`
- `biohubx artifacts register --from REPORT [--registry PATH]`
- `biohubx artifacts clear --reviewed-by NAME --source-url URL --access-restrictions TEXT --eligibility {eligible,not-eligible} --id-prefix PREFIX [--data-license L] [--code-license L] [--weight-license L]`
- `biohubx official verify-source`
- `biohubx evaluate fixture`
- `biohubx data validate {--root PATH | --synthetic}`
- `biohubx data fingerprint --root PATH [--dataset ID] [--plan]`
- `biohubx infer synthetic [--seed N] [--noise F] [--annotated-fraction F]`
- `biohubx infer real --dataset ID [--root PATH] [--split S] [--first-frame N] [--frames N] [--crop-z A:B] [--crop-y A:B] [--crop-x A:B] [--detection-threshold F]`
- `biohubx infer reference --weights PATH [--dataset ID] [--root PATH] [--split S] [--first-frame N] [--frames N] [--crop-z A:B] [--crop-y A:B] [--crop-x A:B] [--detection-threshold F]`
- `biohubx evaluate slice [--graph PATH]`
- `biohubx evaluate retention [--root PATH] [--retentions LIST] [--seed N]`
- `biohubx evaluate mask-audit [--root PATH] [--max-frames N] [--compare-maxpool]`
- `biohubx package audit [--out PATH] [--expect-kernel ID] [--allow-dirty]`
- `biohubx package kaggle --owner SLUG [--expect-kernel ID] [--fold ID] [--out PATH] [--epochs N] [--max-movies N] [--gpus N] [--runtime-ceiling S] [--smoke-id ID] [--wheelhouse PATH] [--expect-published DIGEST] [--expect-device NAME] [--allow-dirty]`
- `biohubx package preflight [--out PATH] [--wheelhouse PATH] [--expect-published DIGEST] [--allow-dirty]`
- `biohubx package wheelhouse [--path PATH] [--remote PATH] [--out PATH]`
- `biohubx package retrieve --kernel ID --only NAME... --out DIR [--allow-missing]`
- `biohubx train preflight [--dataset ID] [--root PATH] [--seed N] [--peak-quantile F] [--learning-rate F]`

`infer synthetic` writes the emitted graph and a stage-by-stage report
atomically and records the raw digest of the graph. `evaluate slice` refuses a
stored graph that does not match a fresh run of the same configuration, so a
stale artifact is never scored against a freshly generated ground truth.

`data validate` takes an explicit root or the single named environment variable
`BIOHUB_DATA_ROOT`, and refuses when neither is given. It never guesses a
location, never writes into the data, and refuses an ambiguous layout rather
than picking one. It understands the official split layout: `train/` must pair
each volume with its ground truth, `test/` is unannotated by design, and ground
truth appearing in `test/` is refused rather than assumed to be a layout change.
Every dataset records whether its role was declared by an official split name or
inferred from a leaf directory's contents.

`artifacts verify` is tiered. Files inside the repository are always re-derived
from their bytes. Dataset trees live outside it and take minutes to read, so
they are checked for presence, file count and total size, and the report says
how many results were deep, shape-only or absent. A shape check is not identity:
a byte flipped in place changes neither count nor size, and only `--deep` sees
it. Absence fails for an in-repository artifact, because the repository claimed
it, and is merely reported for an external one, because a machine need not hold
every dataset.

`artifacts register` merges a completed fingerprint report into the registry
atomically, so no digest is ever retyped. It is idempotent and refuses an id
that already records a different identity, writing nothing.

Identity and clearance are separate. Fingerprinting and registration always
record `external_uncleared`, because they read bytes and check no terms.
`external_cleared` is entered separately by a person and requires the evidence of
a real review: a source, a licence, the access restrictions, an explicit
eligibility decision and a named reviewer. Licence, restrictions and eligibility
are three fields because they are three facts: data can be permissively licensed
and still carry a rule against passing it on, and neither settles whether a
competition allows its use. A recorded clearance survives re-registration.

`artifacts clear` writes down a review that has already happened. It does not
perform one and cannot judge eligibility, so every piece of evidence is a
required option with no default, and the eligibility decision must be stated
explicitly rather than implied by omission. It verifies the selected artifacts
before writing: a clearance says someone read the terms covering those bytes, so
it may not be attached to an artifact whose recorded identity no longer holds.

`data validate` establishes layout integrity only. `data fingerprint` establishes
content identity, by computing a `tree_sha256` for every dataset artifact. The
two answer different questions and neither substitutes for the other: a correct
shape says nothing about a flipped byte.

`data fingerprint --plan` performs the same walk with the same refusals and reads
nothing, so the cost of a large pass is known before committing to it and a
layout or reparse problem surfaces in seconds rather than after minutes of
reading. Its report states no identity, because it has not looked at the bytes.
Every report names the selection it covers, so a single-dataset run is never
read later as a claim about the whole root.

Planned, absent until a consumer exists: `data make-splits`, `proposals infer`,
`representation acquire`, `train fold`.

`infer real` runs the same chain as `infer synthetic` on a bounded window of
one registered training movie. The window is explicit in every dimension
because a preflight whose cost is unknown before it starts is not a
preflight, and the shared `run_chain` means a stage that behaves differently
on real data behaves differently in exactly one place. Its node-count
estimate is prorated and therefore `declared`, never official metadata.

`evaluate retention` reads ground-truth graphs only, never the four
duplicated public-test fixtures, and never a volume.

`evaluate mask-audit` measures the intensity percentile of every annotated
cell in the corpus and reports what each candidate background band would
cost. It is the instrument that falsified the band ([[F-0020]]) and forced
the positive-unlabelled objective ([[D-0024]]). Bands are selected per fold
from the training embryo alone; the evaluation embryo measures failure
afterwards and never selects the rule.

`package audit` stages a CPU-only diagnostic that measures the Kaggle runtime:
no accelerator, no internet, no dataset, model or competition sources, and no
Biohub-X import, so it runs on an image where Biohub-X cannot. It reports the
interpreter, platform tag and ABI, every installed distribution, the runtime
closure's importability, and whether the target decodes the competition's own
blosc codec. Every probe is wrapped, so an absent package is reported rather
than ending the run ([[D-0029]]).

`package kaggle` runs a mandatory pre-push gate before it finishes: the generated
notebook must pass `nbformat.validate`, convert through nbconvert, carry every
required kernelspec and cell field, and the packaged entry point must run locally
and emit at least one heartbeat. All of it happens before any network call,
because a notebook that fails on Kaggle costs a whole GPU session and produces
nothing to read ([[D-0026]]). The kernel id is derived from the title so the two
cannot disagree.

`package kaggle` stages a fold for a machine this repository cannot watch, and
sends nothing. The guards travel with it: the package pins the commit and the
configuration digest, ships the registry's tree digests and shapes for its
training embryo, and its entry point re-verifies mounted identity, asserts fold
membership, refuses public-test paths and quarantined checkpoints, checks the
GPU count, heartbeats every stage and writes an atomic checkpoint with a typed
digest. `--smoke-local` runs that exact entry point here on CPU first.

Its package now also carries the wheelhouse, because `biohubx` imports zarr and
the image does not have it ([[R-0006]]). The mount is located by the identity the
package carries, verified before pip reads a byte of it, and installed offline,
all before `import biohubx`, since installing after that import would install too
late ([[D-0032]], [[D-0033]]). The accelerator is requested as the canonical
`NvidiaTeslaT4`, and because a request is not an allocation, the run refuses
unless the device it actually got reports a Tesla T4, including when it got no
accelerator at all. Torch and CUDA versions, the device name, total VRAM and peak
allocated and reserved memory are reported, so the T4-versus-P100 question and the
memory headroom for a larger window stop being unmeasured.

`package retrieve` fetches only the outputs an authorisation named. `kaggle
kernels output` has no per-file mode, so this is containment rather than
prevention: everything the platform sends lands in a temporary directory, only the
named files are kept, the rest are deleted with it, and the report lists every
file that arrived and what happened to it. It exists because retrieving a
preflight also brought back an artifact nobody had asked for ([[R-0008]]).

`package preflight` stages the wheelhouse installation and readability check, and
sends nothing. It asks two questions in order: does the wheelhouse install offline
on the measured image, and can the result decode a real competition chunk. It
imports no part of Biohub-X, so it runs on an image where Biohub-X cannot, and its
notebook is validated and converted at build time but never executed, because it
installs packages and reads the corpus.

Before either question it establishes that the tree it mounted is the tree it was
authorised against. The published payload's identity and canonical records are
embedded at build time, the mounted tree is recomputed under canonicalization `v1`
before pip is invoked, and any extra, missing, renamed or modified file refuses
with both identities printed into the log ([[D-0032]]). The anchor is carried
rather than read from the mount, because `requirements-offline.txt` cannot
authenticate itself: a substituted dataset shipping its own matching requirements
file satisfies `--require-hashes` exactly. Since the package may not import
Biohub-X, canonicalization `v1` exists a second time inside the notebook, and a
test requires that copy to return exactly what `biohubx.hashing.tree_digest`
returns for the same tree. `--expect-published` refuses to build unless the local
wheelhouse publishes the identity an authorisation named.

That same identity is how the input is found. The package does not assume a mount
path, because assuming one cost an authorised run ([[R-0007]]): it enumerates what
is mounted under `/kaggle/input`, bounded to four levels with single-level globs
and never a recursive one, and selects the tree whose canonical digest equals the
embedded identity ([[D-0033]]). Shape is compared first with an early abort, so a
candidate larger than the tree being sought is abandoned rather than hashed and the
competition corpus is never read to find a directory of four wheels. A miss refuses
with a listing of what is actually mounted, because the run before it could not
distinguish an unattached dataset from one mounted elsewhere.

`package wheelhouse` records what a wheelhouse is, as the two different trees it
actually is. Kaggle consumes `dataset-metadata.json` as configuration rather than
storing it, so the upload bundle this machine sends and the published payload a
kernel mounts differ by exactly that file, and one recorded digest names neither
tree ([[D-0031]]). Both are computed from one walk of one directory. The published
payload is staged and walked as a tree in its own right rather than derived by
deleting a line from the bundle's listing, because removing the last file from a
directory turns it into an empty directory that canonicalization `v1` records,
so an edited listing can describe a tree that could not exist. `--remote` compares
a downloaded copy of the published dataset with the published payload by relative
path, size and content digest, then by tree digest, and refuses on any difference.
Content rather than inventory: a same-size substitution moves neither the file
count nor the total bytes.

`train preflight` proves the E03 loop runs before a GPU is asked for: forward,
masked loss, backward, optimizer step, atomic checkpoint, strict reload,
identical output, peak extraction, suppression and a legal graph. Its detector
starts from deterministic random weights and no quarantined checkpoint touches
it ([[D-0022]], [[D-0023]], [[F-0019]]).

`infer reference` loads a quarantined external checkpoint on CPU and runs its
detection path only. It needs the optional `model-cpu` dependency group;
every other command runs without Torch, which is why the group exists. Both
published weights are `reference_only`, so everything this command reports is
`integration_only` and none of it may support a held-out finding ([[D-0020]],
[[D-0021]]).

## 5. Evaluation policy

The official competition metric is the only end-to-end promotion metric. Every
evaluation reports, separately and never only pooled:

node recall; node precision and count behaviour; candidate-edge reach; raw edge
Jaccard; adjusted edge Jaccard; division TP/FP/FN; division Jaccard; combined
official score; runtime; peak memory; and each held-out domain and direction on
its own.

Retrieval@k, AUROC, t-SNE, UMAP, PHATE and segmentation IoU are diagnostics.
They cannot promote anything on their own.

### 5.0 Provenance gate

**Passed 2026-09-02.** All 402 competition artifacts record `external_cleared`:
data licence CC0, eligibility `true`, reviewer Arya Arun, and the access
restrictions summarised with a pointer to the authoritative wording.

Entry was evidenced by a leaderboard placement under the reviewer's name and by
the competition rules page showing acceptance. An earlier clearance recorded the
same day was withdrawn because its confirmation had not come from someone able
to give it, and because the restrictions had been recorded as though quoted when
they were a paraphrase. That history stays on every record.

The restrictions bind this repository: the data may not be transmitted,
duplicated, published, redistributed or made available to anyone who has not
agreed to the competition rules. Nothing derived from it may be published in a
form that carries the data itself.

A dataset registered later does not inherit this review; a contract test
requires every competition artifact to carry its own.

The public leaderboard is not a validation signal either. The reference system
Biohub-X reads for engineering detail was fitted and selected on movies that are
byte-identical to the public test volumes, so its public score measures
memorisation as much as method ([[D-0020]], R-0002). Every Biohub-X target is
defined on a held-out split of the 199 annotated train movies.

### 5.1 What the corpus permits

The registered corpus constrains evaluation more tightly than the movie count
suggests, and all three constraints are measured rather than assumed.

The local test split is not an evaluation set. It holds four volumes, carries no
ground truth, and each volume is byte-identical to a train volume of the same
dataset id. Nothing is ever scored against it ([[F-0010]], D-0015).

The corpus spans two embryos, 6bba with 128 fields of view and 44b6 with 71.
Splits are grouped by embryo, so leave-one-embryo-out yields exactly two folds
and every cross-embryo claim rests on n=2 ([[F-0011]], D-0016). The 199 annotated
datasets are fields of view, not independent subjects.

Annotation is sparse and unevenly so. Across the 199 annotated datasets there
are 133,318 nodes and 128,883 edges against 4,725,117 estimated cells, 2.821
percent overall, ranging 0.13 to 20.21 percent per dataset ([[F-0013]], which
supersedes the earlier byte-ratio proxy [[F-0012]]). The two embryos differ about
twelvefold in density, so the two folds are different annotation regimes and not
merely different subjects.

Division supervision is scarcer still: 151 annotated divisions in the whole
corpus, with 112 of 199 datasets containing none, split 26 and 125 between the
embryos ([[F-0014]]). This bounds any learned division model far more tightly
than the metric's 0.1 weight suggests.

The node-count adjustment offers roughly a ten percent uplift on the raw edge
Jaccard to a system predicting the annotated count rather than the true cell
count, and is nearly flat between half and twice that count ([[F-0015]]). Which
nodes are predicted therefore matters far more than how many.

### 5.2 What the node-count adjustment is worth

The adjustment is a real lever and a bounded one, and both halves are measured.

A system emitting roughly every cell receives a multiplier of one. The same
edges emitted at the annotated node count receive about nine percent more, and
that gain survives only while a filter destroys less than about eight percent of
the correct edges reaching it ([[F-0018]]). Both embryo folds agree on the
break-even point and on the retention needed to buy the step from a 0.936 system
to 0.950. The measurement is an oracle: retained edges are correct by
construction, so it states the retention a real filter must beat, not that such
a filter exists ([[D-0018]]).

The lever is cheap to over-claim, because unmatched predictions cost nothing in
edge terms ([[F-0003]]) and nothing in division terms either ([[F-0016]]). What
they cost is the node count, and that is the whole trade.

Nothing filters yet. The first real chain on competition bytes is limited by
detection rather than association: every annotated edge it could not reach was
unreachable because an endpoint was never proposed, not because the candidate
radius was too small ([[F-0017]]). A retention policy has nothing to filter
until proposals exist that are worth keeping.

## 6. Phase gates

| Phase | Gate | State |
| --- | --- | --- |
| 0 | repository foundation, identity contract, registries, CLI heartbeat, contract tests | done |
| 1 | official metric from its authoritative source, pinned revision, adapter, synthetic graphs with known scores, coordinate and lineage contracts | done |
| 2 | end-to-end software vertical slice: fixture to candidate instances to representation to sparse T=2 graph to minimal matcher to legal final graph to official scorer | done |
| 3 | real representation smoke, CPU preflight, one protected GPU smoke, stop and report before launch | CPU preflight done ([[F-0017]]); GPU smoke awaiting approval |
| 4 | T=2 learning, depth ladder D = 4/6/8/10 as controlled arms, select one depth | not started |
| 5 | System A against System B on identical frozen inputs, error complementarity | not started |
| 6 | temporal context ladder, coordinated division learning, uncertainty routing once earned | not started |
| 7 | freeze, thin notebook, runtime and structural audit, report before push | not started |

Transformer depth and temporal context are different axes. Depth is `D`.
Temporal context is `T`, and the `T` ladder opens only after `T=2` is valid.
Layers are never described as frames.

## 7. Research inputs

`research/primitives-dossier.md` is a hash-bound snapshot with status
`reference_only`. It is an input, not an instruction file, and it confers no
standing on any primitive it lists.

Public research intake is intentionally broad under [[D-0035]]. Public Kaggle
notebook source and metadata, papers, repositories, technical reports, blogs and
competition discussions may be pulled and read without a per-source approval.
They enter through `registry/reference.yaml` as external assertions, not through
`registry/findings.yaml` as results. Raw captures stay outside Git unless a
hash-bound snapshot has a named consumer. Intake creates neither a new Markdown
file nor a bespoke test per source; mechanisms are deduplicated into a compact
hypothesis queue and earn code only through a preregistered experiment.

This does not relax the prior-campaign quarantine or authorize weights,
datasets, third-party code execution, GPU work, Kaggle pushes, submissions or
other external writes.

`research/shortlist.yaml` is empty at Phase 0 by construction: a primitive may
enter it only when a live Biohub-X component consumes it.
