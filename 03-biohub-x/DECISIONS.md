# DECISIONS

Short architectural decision records. Newest last. A decision is recorded when
it constrains future work; routine implementation choices are not decisions.

---

## D-0001 - Biohub-X is founded independent of any prior campaign

**Date:** 2026-09-01
**Status:** accepted

Biohub-X is a blind, independent attempt. No prior campaign repository is
opened, read, imported, depended on, or compared against during design. Beliefs
come only from the hash-bound dossier snapshot, official competition sources,
primary papers and official project repositories, and measurements produced
here.

**Consequence:** the repository will re-derive facts that may already be known
elsewhere. That cost is accepted deliberately, because an imported belief cannot
be falsified inside this repository and would contaminate every downstream
promotion decision.

**Enforced by:** `tests/contracts/test_repository_isolation.py`.

---

## D-0002 - Two content identities, never interchangeable

**Date:** 2026-09-01
**Status:** accepted

Every piece of content has a `raw_artifact_sha256` (exact bytes) and, when it is
valid UTF-8, a `canonical_text_sha256` (canonicalization `v1`). Raw identity is
used for weights, exported graphs, built notebooks, submissions and hash-bound
snapshots. Canonical identity is used for source and configuration drift.

Digests are serialised as typed tokens that state their kind and, where it
applies, the canonicalization version. A bare hexadecimal string is refused.

**Rejected alternative:** a single hash. Under one hash, a Windows checkout that
rewrites line endings is indistinguishable from a real content change, and a
weight file with a coincidentally matching text hash is indistinguishable from a
verified one. The two questions ("did the file change" and "did the content
change") are genuinely different and both are needed.

**Consequence:** bumping `CANONICALIZATION_VERSION` invalidates every recorded
canonical digest and requires a new decision record here.

---

## D-0003 - Repository byte policy fixed before the first digest

**Date:** 2026-09-01
**Status:** accepted

`.gitattributes` sets `* text=auto eol=lf` so text is stored and checked out
with LF on every platform, making raw digests of source files
platform-independent. It was written and committed in the same commit that
records the first digest, and before any digest was computed.

`research/primitives-dossier.md` is marked `-text` so Git never normalises it.
Its recorded raw digest is an assertion about the original source bytes, and
normalisation would silently break that assertion.

**Consequence:** changing `.gitattributes` invalidates recorded raw digests of
text files. It is on the stop-and-report list in AGENTS.md.

---

## D-0004 - One typed CLI, no `scripts/` directory

**Date:** 2026-09-01
**Status:** accepted

Every operation is a subcommand of `biohubx.cli`. Each command accepts a typed
configuration, prints a heartbeat, writes atomically, writes a manifest, refuses
missing or ambiguous inputs, and never falls back to a default data path.

**Rejected alternative:** a `scripts/` directory. Loose scripts accumulate
untested branching logic that the package never sees, and they are where hidden
absolute paths and silent defaults survive.

---

## D-0005 - Modules are created only when a consumer exists

**Date:** 2026-09-01
**Status:** accepted

The proposed repository tree is a target, not a checklist. `src/biohubx/`
currently holds `hashing.py`, `artifacts.py` and `cli.py` because those three
have real consumers today. `contracts/`, `data/`, `proposals/`,
`representation/`, `tracking/`, `training/`, `evaluation/` and `deployment/` are
absent rather than stubbed.

**Rejected alternative:** creating the full tree with `pass` bodies. An empty
module is indistinguishable from an unimplemented one at a glance, invites
`from x import y` against nothing, and makes the repository look further along
than it is.

`artifacts.py` at the package root is a deliberate addition to the proposed
tree: `hashing.py` owns identity only, and atomic materialisation plus the
artifact registry model needed a home whose consumer (`cli.py`) already exists.

---

## D-0006 - The dossier is `reference_only` and the shortlist starts empty

**Date:** 2026-09-01
**Status:** accepted

`research/primitives-dossier.md` is a hash-bound snapshot with provenance status
`reference_only`. It is a research input, not an instruction file, and listing a
primitive in it confers no standing.

`research/shortlist.yaml` is empty at Phase 0. A primitive enters it only when a
live Biohub-X component consumes it, with the component named in the entry.

**Consequence:** the shortlist can never become a second copy of the dossier,
and the count of shortlisted primitives is a measure of how much system exists.

---

## D-0007 - Python 3.12 pinned for development

**Date:** 2026-09-01
**Status:** accepted

`.python-version` pins 3.12. `requires-python` allows `>=3.11,<3.14` so the
package can be installed into whatever interpreter the eventual Kaggle image
provides.

**Revisit when:** the competition runtime image is inspected in Phase 7. The
declared range is a guess about the target environment and has not been measured
against it.

---

## D-0008 - The metric is vendored and pinned, never reimplemented

**Date:** 2026-09-02
**Status:** accepted

`src/biohubx/_vendor/official_competition/` holds a byte-identical copy of the
official scorer at one commit, with typed digests in
`registry/official_source.yaml`. `biohubx.evaluation.official_metric` validates
Biohub-X contracts, converts to the official graph type, and calls the official
functions. It computes no metric arithmetic of its own. The vendored copy is
excluded from lint and format so that a tidying pass cannot silently fork the
authority, and its digests are re-derived by `biohubx official verify-source`.

**Rejected alternative:** reimplementing the metric from its documentation. A
reimplementation agrees with the original exactly until the day it does not, and
the disagreement would surface as an unexplained gap between local and
leaderboard scores, at which point every earlier promotion decision becomes
suspect.

**Consequence:** moving the pin is a stop-and-report action. It supersedes every
finding whose validity names the pin, until the instruments are re-run.

---

## D-0009 - The node-count denominator is explicit and provenance-tagged

**Date:** 2026-09-02
**Status:** accepted

The official adjusted Jaccard divides by a coarse estimate of *every* cell,
which on real data is the dataset's `estimated_number_of_nodes`. Annotations are
sparse, so the annotated count is a different and smaller number.
`EstimatedTotalNodes` carries the value together with where it came from, has no
default, and every evaluation records the annotated count, the estimate and
their ratio side by side.

**Rejected alternative:** defaulting to the annotated ground-truth count. The
adapter did exactly that before this decision. It is not a harmless
approximation: it pins `total_node_ratio` at zero regardless of the true
sparsity, which silently deletes the whole node-count term and reports an
adjusted Jaccard that is not comparable with the competition's. Measured in
[[F-0004]]: the effect is invisible on a complete ground truth and worth 0.09 of
adjusted Jaccard at a tenfold sparsity on the same graph.

**Consequence:** using the annotated count remains available for a fixture whose
ground truth is complete by construction, but the call site has to say so.

---

## D-0010 - Illegal graph structure is refused, never left to the scorer's silence

**Date:** 2026-09-02
**Status:** accepted

`LineageGraph` refuses at construction every structural defect the mission
names. This is deliberately stricter than the official scorer, which was
measured in [[F-0005]] to drop a frame-skipping edge rather than charge it: such
an edge is neither rewarded nor penalised, while the annotated edge it failed to
reproduce still counts as a false negative.

**Rejected alternative:** emitting whatever the decoder produces and relying on
the scorer to clean it up. Leniency that is silent is worse than leniency that
is loud, because a system tuned against it learns to depend on behaviour that no
document promises and that the organisers can change without notice. The metric
was already patched once, in July 2026, to close an exploit of exactly this kind.

**Consequence:** Biohub-X can score lower than a system that games the scorer's
silences, and that is accepted.

---

## D-0011 - Node count is a measured quantity before it is an architecture choice

**Date:** 2026-09-02
**Status:** accepted

[[F-0003]] and [[F-0004]] together show that the number of nodes a system emits
has two separate consequences: unmatched nodes carry no edge penalty, and the
node-count adjustment moves the score in both directions with no upper clamp. No
component whose purpose is to change how many nodes are emitted is added to
`main` until that lever has been measured on real data with an experiment that
declares its falsifier.

This binds proposal banks, rescue detectors, threshold sweeps and minimum track
lengths equally. It is not a judgement about any of them.

**Rejected alternative:** adding a multi-hypothesis proposal bank first, on the
argument that recall is the ceiling. Recall is a ceiling on edges, but the same
extra proposals move the adjustment, and the two effects have opposite signs.
Adding the bank before measuring the trade would make the result uninterpretable
whichever way it came out.

**Revisit when:** the node-count experiment reports, at which point this either
becomes a specific calibration decision or is superseded.

---

## D-0012 - Dataset identity is a tree digest, not a name or a size

**Date:** 2026-09-02
**Status:** accepted

A dataset artifact is a Zarr-backed directory of many thousands of chunk files,
so it has no single-file digest. `tree_sha256` under canonicalization `v1` is
the SHA-256 over a canonical listing of every file's relative path, size and
content digest, plus every empty directory. No extracted data may enter an
experiment without one.

**Rejected alternatives, and why each is not enough:**

- *The competition slug.* A mutable name chosen by the host. It says where data
  was meant to come from, not what arrived.
- *The downloaded archive's digest.* The tool that fetches it normally deletes
  the archive after extracting, so the identity would name something that no
  longer exists and that nothing later reads.
- *Directory kinds and entry counts.* These establish layout integrity, which is
  a different question. They cannot detect a flipped byte in a chunk.
- *Paths and sizes without content.* Same objection: a same-size corruption is
  invisible.
- *A digest over file contents alone.* Would call two trees identical after two
  files swapped contents, and would miss a chunk moved to the wrong path. The
  path is part of the identity because a chunk in the wrong place is different
  data.

**Consequence:** registration reads every byte once, which on a large dataset is
the dominant cost. That is accepted, because it buys the ability to say that the
data an experiment used is the data that was registered. The listing is kept
alongside the digest so a later mismatch names the file that changed rather than
only reporting that something did.

Any reparse point is refused, at the root and at every descendant: a symlink, a
Windows directory junction, or anything else carrying the reparse-point
attribute. Following one could leave the artifact and record foreign files under
relative paths that look local, and skipping one would lose information, so
neither happens silently.

Checking the predicate is not sufficient on its own; the traversal order matters
as much. An implementation built on `rglob` enumerates the whole tree before a
caller can inspect any of it, so a junction is already followed and its contents
already listed by the time any check could run. The walk is therefore explicit
and judges every entry before descending into it.

On Windows a directory junction is invisible to the obvious test: `is_symlink()`
reports False for one while `is_dir()` reports True, and a junction needs no
special privilege to create. That combination is why this was a live hole rather
than a theoretical one, and why the junction case is covered by a real test
rather than only a mocked one.

**The tree must also be quiescent for the whole pass.** A digest is an assertion
about a state the tree was actually in. Hashing a tree that is still being
written produces one describing a state that never existed as a whole: some
files read before a change and some after, with anything created midway missing
from the walk entirely and therefore invisible to any content check. The
structure is snapshotted before and after the pass and the digest is refused if
it moved, with a distinct error, because the response is to wait for the writer
rather than to investigate a malformed tree.

The snapshot compares path, size and modification time. It cannot detect a
change reverted inside the window, nor two writes leaving both size and
modification time identical. Those are not the realistic case, which is a writer
that has not finished. This is why registration waits for extraction to be
complete rather than merely for a download to reach its final byte.

---

## D-0013 - One registry, verification tiered by what each artifact is

**Date:** 2026-09-02
**Status:** accepted

Competition datasets are registered in `registry/artifacts.yaml` alongside
everything else, keeping the four-registry rule intact. They differ from every
other artifact in three ways at once: they live outside the repository, their
location is machine-local, and reading one takes minutes rather than
milliseconds.

Verification is therefore tiered, and the tier is recorded on every result.
A file inside the repository is always re-derived from its bytes. A dataset tree
is checked for presence, file count and total size by default, and re-derived
only under `--deep`. The report states how many results were deep, how many were
shape-only and how many were absent, and prints a line naming the shape-only
count when it is not zero.

**The shape tier is not identity, and nothing pretends otherwise.** A byte
flipped in place changes neither the file count nor the total size, so the cheap
tier passes it and only `--deep` catches it. There is a test asserting exactly
that divergence, so the limitation is a measured property rather than a caveat
in prose.

**Rejected alternatives:**

- *A second registry for datasets.* Cleanest separation, but it adds a fifth
  registry file against the explicit four-registry design, and it would split
  the answer to "what identities does this repository assert" across two places.
- *Registering everything at one depth.* Deep everywhere makes the gate read
  81 GiB on every run. Shallow everywhere silently weakens the guarantee for the
  in-repository artifacts that can afford the strong one.
- *Skipping datasets during the ordinary gate.* A skipped check that is not
  reported is indistinguishable from a passing one, which is the failure this
  repository is built to avoid.

**Absence is reported, not failed, for external artifacts only.** The repository
must contain what it claims, so a missing in-repository artifact is a broken
claim. A machine need not hold every dataset, so a missing external one is a
fact about the machine. The two are distinguished by which location field the
record carries.

**Registration is a command.** `biohubx artifacts register --from <report>`
merges a completed fingerprint into the registry atomically. Hand-editing would
mean retyping digests, which is the manual duplication forbidden elsewhere. It
is idempotent, it preserves records it did not add, and an id already present
with a different identity is refused with nothing written: that means the data
changed under a name the registry already claims, and only a person can decide
whether that is a re-download or a problem.

Registration records `external_uncleared`. Fingerprinting establishes identity,
not licence or competition eligibility, and the record must not imply a review
nobody has done.

---

## D-0014 - Clearance is a separate act from identity, and must name its evidence

**Date:** 2026-09-02
**Status:** accepted

Byte identity answers what the data is. Clearance answers whether it may be
used. Nothing about a digest establishes the second, so the two are recorded
independently and neither can be mistaken for the other.

Fingerprinting and registration always record `external_uncleared`. They read
bytes; they check no terms. A cleared status is entered separately, by a person,
after an actual review.

`external_cleared` now requires the evidence of that review: a source, a licence,
the access restrictions, an explicit eligibility decision, and the name of who
reviewed it. Without this the status was decorative, and anything could be marked
cleared with every terms field empty. A later reader had no way to tell a real
review from a forgotten default.

**Licence, access restrictions and eligibility are three separate fields because
they are three separate facts.** Competition data can be permissively licensed
and still carry a rule against redistributing it to non-participants, and neither
of those says whether a competition's own rules allow a particular use. A single
licence field would record the first and silently lose the other two.
`data_license` exists because for a dataset neither a code licence nor a weight
licence is the licence that matters.

**A recorded clearance survives re-registration.** A fingerprint always reports
`external_uncleared`, so merging one into a record that had been cleared would
erase a completed review during a routine re-run. Registration leaves an
unchanged artifact alone, and a test now holds that property rather than leaving
it to be an accident of how the comparison happens to be written.

---

## D-0015 - The local test split is never evaluated against

**Date:** 2026-09-02
**Status:** accepted

Measured in [[F-0010]]: the local test split holds four volumes, carries no
ground truth, and every one of the four is byte-identical to a train volume of
the same dataset id. It is a format example for the submission pipeline. The
genuinely held-out data is the hidden set Kaggle swaps in at rerun.

No Biohub-X component may compute a score against it. Doing so would be scoring
training data, and because the duplication is exact the result would look
plausible rather than obviously wrong, which is the dangerous kind of mistake.
The test split is used only to exercise the shape of the inference and export
path.

All held-out evaluation is carved out of the 199 annotated train datasets.

## D-0016 - Splits are grouped by embryo, and the fold count is two

**Date:** 2026-09-02
**Status:** accepted

Measured in [[F-0011]]: dataset ids are `{embryo}_{field_of_view}` and the corpus
spans two embryos, 6bba with 128 fields of view and 44b6 with 71.

Splits are grouped by embryo. Fields of view from one embryo share its biology,
its imaging session and its annotator, so a random split over the 199 would put
near-siblings on both sides and report a generalisation number that is really a
memorisation number.

**The honest consequence is that leave-one-embryo-out yields exactly two folds,
each training on a single embryo.** Every cross-embryo claim this project can
make rests on n=2. That is a property of the data, not something a better
experiment design can fix, and it is recorded here so that no later report
describes 199 movies as 199 independent observations.

Where a finer split is needed, grouping by field of view within an embryo is
permitted for model selection, but a result obtained that way may never be
described as evidence of holding up on a new embryo.

## D-0017 - External work is registered as reference, never as finding

**Date:** 2026-09-02
**Status:** accepted

A run log from a competitor's notebook arrived carrying a great deal of usable
detail: thresholds, weight digests, emitted node counts, hardware, runtime. None
of it is a Biohub-X measurement, and `registry/findings.yaml` exists precisely to
hold measurements this repository can reproduce from its own instruments.

Putting a third party's numbers there would make them indistinguishable from
measured ones at a glance, which is the failure `AGENTS.md` section 2 is written
to prevent. Leaving them only in prose would violate the rule that a number
outside a registry is not a result, and would lose them.

So they go in `registry/reference.yaml`, a third registry whose every entry is
`status: reference_only` and whose schema requires an `unresolved` list. The
distinction it enforces is between what a source states and what Biohub-X knows.
A reference entry may motivate an experiment; it may never support a claim, and
no threshold recorded in one may be adopted as a prior.

The first entry records that the run's configuration was tuned against public
leaderboard feedback. That alone disqualifies every constant in it from being
copied, because Biohub-X does not use that leaderboard as validation.

## D-0018 - The node-count trade is measured as an oracle before it is engineered

**Date:** 2026-09-02
**Status:** accepted

The official node-count adjustment rewards a system that emits roughly the
annotated cell count over one that emits roughly every cell. Acting on that
means building a filter, and a filter that removes predictions also removes
correct edges. Whether the trade is worth making is a question about the metric,
not about any particular filter, so it is answered without building one.

`biohubx evaluate retention` constructs predictions from each dataset's own
ground truth under two node budgets, scores them through the pinned official
scorer, and reports where the curves cross. Retained edges are correct by
construction, so the result is an upper bound: it establishes the retention a
real filter would have to beat, and cannot establish that such a filter exists.
Every report of it must carry that limit.

Three choices inside it constrain later work.

The decisive quantity is the adjusted edge Jaccard, not the combined score. The
division term is additive and independent of the node budget, so folding it in
would attribute division behaviour to a retention decision.

The dense budget is reached by padding with isolated nodes placed far outside
any annotated coordinate. That is only legitimate because [[F-0003]] and
[[F-0016]] establish that unmatched predictions are free of edge and division
penalty, and the command re-checks it on real data before the sweep relies on
it. If that guard ever fails, the sweep refuses rather than reporting.

Folds are reported separately and never pooled, per [[D-0016]] and [[F-0013]].

## D-0019 - Two subcommands were added where the brief allowed one

**Date:** 2026-09-02
**Status:** accepted

Recorded as a deviation rather than left implicit. The turn's anti-sprawl budget
allowed one new CLI subcommand. Two were added: `infer real`, which runs the
chain on a bounded window of real data, and `evaluate retention`, which measures
the node-count trade across the corpus.

They were not merged because they share no input. One reads volume bytes for a
single window; the other reads ground-truth graphs for 199 movies and reads no
volume at all. A single command spanning both would take a mode flag and two
disjoint option sets, which is worse than two commands. `AGENTS.md` section 3
forbids the third option, a script.

One experiment configuration was added, `configs/phase3-cpu.yaml`, holding both
E01 and E02, so the budget was met there.
