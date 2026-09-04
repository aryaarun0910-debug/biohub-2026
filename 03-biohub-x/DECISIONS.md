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
## D-0020 - The public leaderboard is not a target, and the reference score is not evidence

**Date:** 2026-09-02
**Status:** accepted

G-01 acquired the reference notebook's source and its weight packs' provenance
manifests without downloading a weight. Two of the three models it blends record
their own training splits, and both splits contain all four public-test movie
ids: the secondary seed trained on all 199 annotated movies, and the centre-prior
model split them 71 against 128, which is the embryo split, holding none of the
four out of both sides (R-0002).

Because each public-test volume is byte-identical to a train volume carrying
ground truth ([[F-0010]]), those models were fitted and selected on the public
test set. The reference's public score is therefore not evidence that its method
generalises, and neither is any score built on the same weights.

Three consequences bind future work.

No Biohub-X target is defined against the public leaderboard. The 0.950 goal is
restated as a score on a held-out split carved from the 199 annotated train
movies under [[D-0016]], and a public-leaderboard number may be reported as an
observation but never as a promotion criterion.

Reproducing the reference is now an engineering exercise, not a validation one.
It may establish node counts, runtime, memory and stage behaviour, and it may not
establish that the approach works. The experiment that reproduces it must say so
in its hypothesis.

Where the reference's constants disagree with its own runtime receipt, the
notebook's environment block is authoritative. The receipt is stale in at least
three fields and describes the notebook's parent fork rather than the run that
produced the graphs (R-0002).

This does not make the reference worthless. Its training code is CC0 and it
describes a proposal source, which [[F-0017]] identifies as the thing Biohub-X
does not yet have.
## D-0021 - Torch is an optional group, and the reference definition is a boundary not a framework

**Date:** 2026-09-02
**Status:** accepted

Loading the reference checkpoints needs Torch. Making it a core dependency would
have made every Biohub-X operation pay for it: `artifacts verify`, the official
metric, the retention oracle and the whole data path do not touch a tensor, and
none of them should import two hundred megabytes to prove it.

So Torch lives in a `model-cpu` dependency group, pinned to `2.10.0+cpu` from
the PyTorch CPU index. The build is verified CPU-only in R-0004: no CUDA, no
NVIDIA packages. Whether to add a CUDA build is a separate decision that belongs
to training our own folds, and keeping the two apart is the point of the group.

The architecture itself is vendored byte-exact from CC0 source, for the same
reason the official metric is vendored: an architecture retyped from a
description is a different architecture, and `strict=True` would either fail
against it or, worse, succeed against the wrong shapes.

What Biohub-X owns is only the composition the checkpoints imply and the
vendored files do not contain, plus a strict loader. That module has one named
consumer and must not acquire a second without a decision. **It is not a
reference-model framework and must not become one.** When Biohub-X trains its
own folds, those models get their own definitions and their own registry
entries; they do not go here.

One detail is worth keeping. The positional embedding width appears in no
published configuration, so the loader derives it from the checkpoint rather
than hardcoding the upstream constant. A checkpoint built differently is refused
at load instead of failing somewhere inside inference.
## D-0022 - The reference is an engineering interface, not a performance baseline

**Date:** 2026-09-02
**Status:** accepted

The contaminated 0.936 pipeline will not be reproduced end to end.

Two of the three models it blends were fitted or selected on movies that are
byte-identical to the public-test volumes (R-0002), so reproducing it would
confirm that Biohub-X can run someone else's contaminated pipeline and would
establish no baseline anything could honestly improve on. The work it would take
is not small, and what it buys is a number that already exists and does not mean
what it appears to mean.

What the reference remains is useful and bounded: a loadable proposal interface
(R-0004), a CC0 training script whose objective can be audited, and a record of
what a dense detector emits. It informs hypotheses. It is never a target, never a
baseline, and never evidence.

**The campaign target is 0.950 on each embryo fold separately**, measured on the
held-out fold under [[D-0016]], never pooled and never against the public
leaderboard ([[D-0020]]). The first honest number this project produces will be
lower than 0.936, and it will be the first one that means anything.

## D-0023 - Unlabelled cells are ignored, and the published objective is rejected

**Date:** 2026-09-02
**Status:** accepted

The CC0 trainer builds its detection target as `torch.zeros_like(logits)`, marks
the annotated node voxels positive, and lightly penalises every other voxel. Its
own docstring says so. There is no ignore class.

Against this corpus that is not a small approximation. On `44b6_0c582fdc` it
supervises 71 cells as positive and 27,887 visually identical cells as
background, and the ratio spans 4x to 393x across the sample, tracking the
twelvefold density difference in [[F-0013]]. The two folds would not be solving
the same problem. AGENTS.md section 5 forbids it outright: unlabelled and ignore
regions are respected, never treated as background.

So Biohub-X does not copy it. The target is three-valued. Positives are the
annotated node voxels. Negatives are voxels the image says are empty. Everything
bright but unlabelled is ignored, because that is where the unannotated cells
are and nothing in the data says which of them is one.

The band is set by intensity quantile on the volume being trained, not by a
tuned constant, and the reason to believe it is measured rather than asserted:
at the ninetieth percentile it covers about a tenth of voxels and contains every
annotated cell in the preflight window ([[F-0019]]). A contract test holds that
claim, because if the band that hides unannotated cells did not also contain the
annotated ones, there would be no argument that the two populations look alike.

The cost is real and taken deliberately: supervision on bright non-cell
structure is given up, which the published objective did have. A detector
trained to call 393 real cells background is the worse trade.
## D-0024 - There is no background class, because the data will not support one

**Date:** 2026-09-02
**Status:** accepted

[[D-0023]] replaced the published objective's implicit background with an
intensity band, on the strength of a single window where the band contained
every annotated cell. E03-MASK-AUDIT measured the same band at every annotated
node in the corpus and it does not survive ([[F-0020]]).

The numbers are not marginal. The band this repository had adopted would call a
fifth of 44b6's annotated cells and a seventh of 6bba's background. Selecting per
fold from training data alone, 6bba admits no band in the set at all, and the one
44b6 admits withholds half the volume from supervision while still misfiring on
the held-out embryo. Max-pooling instead of striding makes it slightly worse, so
the dim annotated cells are real rather than an artefact of sampling every fourth
voxel.

Widening the band until it stops contradicting anything would be fitting the rule
to the complaint. **So the negative class is removed.** Voxels are positive or
unlabelled. The unlabelled ones are modelled as a mixture whose positive share is
the dataset's own recorded `estimated_number_of_nodes`, which is official
metadata rather than a tuned constant, and the risk is the non-negative
positive-unlabelled form so that the clamp firing is visible rather than silent.

Three consequences bind later work.

`build_detection_target` and its masked loss are deleted from `main` rather than
left behind a flag. They are a falsified implementation, and AGENTS.md keeps
those in history, not in production files. The band survives only as the audit
that rejected it, because a rejected rule is worth being able to re-measure.

The remaining uncertainty is named rather than resolved. Biohub-X does not know
how many dim unannotated cells exist, only that annotated ones reach percentile
0.09. The class prior counts every cell the dataset estimates, so if that
estimate is itself biased against dim cells the prior is too low and the
detector will under-predict. That is E03's stated open risk, not a solved
problem.

A finer grid is the one route that could bring a background class back. At
(1, 4, 4) a cell centre can fall between sampled voxels; at full resolution it
cannot. If a later experiment trains at a finer grid, [[F-0020]] must be
re-measured there before any band is reconsidered.
## D-0025 - nnPU is used without claiming it is unbiased

**Date:** 2026-09-02
**Status:** accepted

The positive-unlabelled objective [[D-0024]] adopted is unbiased only under
SCAR: labelled positives must be selected completely at random from all
positives. [[F-0022]] measured that assumption and it does not hold. Annotation
density correlates with the brightness of the annotated cells at r = +0.354
across the corpus, so labelling is not independent of the feature the detector
reads.

The estimator is kept, because the alternatives are worse. A negative class is
falsified outright on both folds ([[F-0021]]). Estimating a per-dataset labelling
propensity would need a model of how the annotators chose, which this project
does not have and cannot validate. What changes is the claim, not the code:
**E03 states nnPU as a bias-bounded choice, never as an unbiased one**, and the
residual bias is an open risk carried into its results rather than a footnote.

Two consequences are binding.

E03's contract lists SCAR as an explicit assumption with its own falsifier, so a
result that depends on it cannot be read as if it did not. The falsifier is
recorded alongside the fold falsifier and is checked with it.

The class prior is per dataset, from that dataset's own recorded cell estimate,
and it is valid at every scale the pipeline uses ([[F-0021]]): 1.443e-4 to
3.000e-3 per grid voxel, every dataset inside the open unit interval. A single
corpus-wide prior would have been wrong by up to twentyfold on individual
movies, and the per-dataset form is what keeps the misspecification bounded even
where SCAR fails.
## D-0026 - A generated notebook is validated before it is allowed to cost a GPU

**Date:** 2026-09-02
**Status:** accepted

E03-SMOKE was authorised, pushed once, and produced nothing. The generated
notebook failed `nbformat` validation before a single line of packaged code ran:
the kernelspec had no `display_name` and the cell had no `id`. Every guard the
package carried, every heartbeat, every falsifier, was irrelevant, because none
of them executed.

That is the worst shape a failure can take here. A guard that refuses is cheap
and informative. A run that dies before the guards do is a spent GPU session with
nothing to read.

Two rules follow.

The notebook's own structure is now part of what the builder checks, not
something discovered on the far side of a push. A test asserts the fields
`nbformat` requires, because the cost of that assertion is nothing and the cost
of omitting it was a whole authorised run.

The kernel address is checked at build time. Kaggle derives a slug from the title
and prefers it over the requested id, so a title that does not slugify to its own
id silently relocates the kernel; `kernel_metadata` now refuses that, and refuses
an id with no owner. The owner itself still cannot be verified offline: the
`username` field of a local credentials file is not necessarily the account's URL
slug, and on this machine it was not. A future request states the owner as the
slug from the account's own URL, and the discrepancy is recorded rather than
guessed at.

## D-0027 - The notebook carries its own package and imports from nowhere else

**Date:** 2026-09-02
**Status:** accepted

The generated notebook used to begin `sys.path.insert(0,
"/kaggle/input/biohubx-package/src")`. Nothing supplies that package. Kaggle
documents `/kaggle/input` for attached inputs and `/kaggle/working` as the
writable runtime directory, and the CLI uploads the kernel folder without
promising to mount it under an input path. The package declares no dataset or
model sources at all, deliberately, so the path could not have existed.

It would have been found only by spending another GPU session, which is how the
last one was spent.

So the tested source travels inside the notebook as a deterministic archive:
entries sorted, timestamps fixed, permissions normalised, so the same tree always
yields the same bytes and a digest recorded at build time means something at run
time. The notebook verifies that digest before extracting anything, refuses
entries that escape their root, extracts under `/kaggle/working`, imports from
there, and then asserts the imported module actually came from the verified tree.

The payload carries source and nothing else, checked by reading the archive
rather than trusting the filter that wrote it.

The gate proves this by running the notebook's own code as a subprocess from a
temporary directory, with the interpreter isolated and `PYTHONPATH` removed.
That is not ceremony: `biohubx` is installed in this environment, so any weaker
check would pass whether or not the bootstrap worked. The run must also announce
the whole stage sequence, from `bootstrap-verify` to `done`, as a subsequence.

**The general rule this turn earns: a packaged run may not depend on any path the
package itself does not create or verify.**

One remote run has been lost, not two. E03-SMOKE was pushed once and errored
during notebook conversion; the unmounted import path was never reached, because
the notebook died before executing a line, and it was found locally afterwards
rather than by spending a second session. An earlier revision of this record said
two, which overstated the cost and would have made the history unreadable later.
The distinction matters: one assumption was paid for and one was caught.

## D-0028 - The Kaggle image is an input, and it was never checked

**Date:** 2026-09-02
**Status:** accepted

E03-SMOKE-02 ran. The bootstrap worked exactly as designed: the payload verified
against its recorded digest, extracted, and imported from the verified tree, and
the kernel returned that tree so the round trip could be confirmed here rather
than assumed. Then it died on `import zarr`.

The Kaggle image does not carry zarr, and the kernel runs with internet disabled
by design, so nothing could install it. The reference notebook shipped 322 MB of
offline wheels for exactly this reason, and this repository looked at that list
and recorded that a local run "does not need" them. That was true of a local run
and irrelevant to a Kaggle one.

**The environment a package runs in is an input, and inputs get verified.** The
pre-push gate checks the notebook and the bootstrap against this machine, which
has every dependency installed, so it cannot see a gap that only exists over
there. Nothing in the gate models the target image at all.

The same run exceeded its authorised scope in a second, unrelated way. It was
approved for one T4; Kaggle allocated a P100. `kernel-metadata.json` requested
`nvidiaTeslaT4` and that did not take effect. The packaged guard checked
`torch.cuda.device_count()`, which was 1 and passed, and never looked at the
device model. Counting devices is not checking hardware. The guard now compares
`torch.cuda.get_device_name(0)` against an expected substring and refuses
otherwise, and the run prints the name it found so a mismatch is visible in the
log rather than in a screenshot afterwards.

Both failures share a shape worth naming: a check that looks adjacent to the
thing that matters. Device count next to device model, local imports next to
remote imports. Neither is a substitute for the other.

## D-0029 - The target environment gets measured before it gets depended on

**Date:** 2026-09-02
**Status:** accepted

Two remote runs have failed for reasons that had nothing to do with the science.
Both were assumptions about a machine this repository cannot see: a notebook
format, then an import path, then a missing package. The pre-push gate closed the
first two by running the real conversion and the real bootstrap locally. It
cannot close the third, because it runs on a machine that has every dependency
installed, so it is structurally blind to a gap that exists only over there.

So the environment becomes an input, and inputs get measured. `package audit`
stages a CPU-only diagnostic with no accelerator, no internet and no sources,
which reports the interpreter, the platform tag, the ABI, every installed
distribution, the importability of the whole runtime closure, and whether the
target can decode the competition's actual blosc/zstd/bitshuffle codec.

Two properties make it a diagnostic rather than another guess. It imports no part
of Biohub-X, so it runs on an image where Biohub-X cannot. And every probe is
wrapped in BaseException, so one absence is reported rather than ending the run;
a diagnostic that stops at the first gap measures only that gap.

**A wheelhouse is not designed from a traceback.** A wheel is matched to a Python
version, a platform tag and an ABI, and until those are measured any wheel set is
a guess with a download attached. The provisional plan is a minimal Biohub-X-owned
wheelhouse locked to the measured ABI, with source, version, licence, SHA-256 and
named consumer recorded per wheel, and internet still disabled. That plan is not
started, because its first input does not exist yet.

## D-0030 - Environment measurements are facts, and still not evidence

**Date:** 2026-09-03
**Status:** accepted

E03-ENV-AUDIT-01 ran and produced what three tracebacks could not: the target is
CPython 3.12.13 on Linux x86_64 with glibc 2.35, platform tag `linux-x86_64`,
SOABI `cpython-312-x86_64-linux-gnu`, carrying 874 distributions. Nine of the
twenty-eight names in the runtime closure are absent. That is a real measurement,
made by this repository's own instrument on the machine in question.

It is not scientific evidence, and the registry now says so in a way that cannot
be misread. `registry/reference.yaml` gains a second status,
`measured_environment`, for facts Biohub-X measured about an external system.
Such an entry may inform a plan and may never appear in `registry/findings.yaml`
or support a promotion, because what a runtime contains says nothing about cell
tracking.

The measurement narrows the wheelhouse sharply. The smoke path imports
`data.competition`, `reference.architecture` and `training.targets`; only the
first reaches beyond torch and the standard library, and it imports zarr alone.
It never touches the official metric, so tracksdata, geff, rustworkx, bidict and
imagecodecs are not needed to make a smoke run. zarr pulls numcodecs, donfig and
crc32c. Four wheels against the reference's 322 MB bundle, and the difference is
entirely because the requirement was measured rather than copied.

One probe was weaker than it looked, and it is recorded as such. blosc2 is
present at 4.1.2, but the audit only reported its version; it never asked it to
decode the Blosc1 fixture. Blosc2 is a different container format, so whether the
image can read competition chunks without numcodecs is still open in both
directions. A probe that reports a version where a decode was needed is a probe
that answers an adjacent question, which is the same shape of mistake as counting
devices where the model mattered.

## D-0031 - A published dataset has two identities, and recording one is how they get confused

**Date:** 2026-09-03
**Status:** accepted

A wheelhouse is uploaded as a directory and published as a dataset, and those are
not the same tree. The Kaggle CLI reads `dataset-metadata.json` to learn what to
create and does not keep it among the data, so the published dataset holds one
file fewer than the bundle that produced it.

Only one digest was recorded, over the upload bundle, and the authorisation
therefore named 8 files and 9,304,348 bytes. The dataset holds 7 files and
9,304,073 bytes. The gap is exactly that 275-byte configuration file, but a single
recorded identity gave nobody a way to say so: with one digest, a benign platform
behaviour and a substituted file look identical, and the only available response
to a differing inventory is to stop. Stopping was correct. Having to reason about
it in prose was not.

So both trees are recorded, and they are named apart. The **upload bundle** is
what this machine sends and can re-derive from its own staging directory. The
**published payload** is what a kernel mounts, and it is therefore the identity an
authorisation names and the only one a running package could ever check.

Three consequences bind later work.

`biohubx package wheelhouse` computes both from one walk of one directory, because
recording an identity is an operation and AGENTS.md section 3 puts operations in
the CLI. The wheelhouse had been assembled and digested outside it, which is how a
digest came to exist with no instrument that could reproduce or extend it.

**The published payload is digested as a tree in its own right, never by deleting
a line from the bundle's canonical listing.** Subtraction is wrong in general:
removing the last file from a directory turns it into an empty directory, which
tree canonicalization `v1` records in its own right, so an edited listing can
describe a tree that could not exist. The payload is staged and walked with the
same contract-tested walk every other tree identity uses.

Verification of a published dataset is by content, not by inventory. Kaggle
reports filenames and sizes, and a same-size substitution changes neither those
nor the file count, so `--remote` compares relative path, size and content digest
and then the tree digest. This is the same distinction D-0013 draws between the
shape tier and identity, applied to a tree this repository published rather than
one it was given.

**What this does not fix.** The preflight mounts the published payload and does
not verify it. It reads only `wheels/` and `requirements-offline.txt`, so it
carries no upload-bundle expectation and meets a seven-file mount correctly, and
`pip --require-hashes` refuses any wheel the requirements file does not list. But
the requirements file arrives with the wheels, and the tree's own identity is
never checked, which leaves the same shape of gap D-0027 closed for the notebook's
payload and this package still has for its mount. Closing it means embedding the
published tree digest and checking it before installing. That changes the package
and needs its own authorisation, so it is recorded as an open gap rather than
quietly accepted.

## D-0032 - A packaged run verifies its inputs against an anchor it carried, not one it mounted

**Date:** 2026-09-03
**Status:** accepted

The wheelhouse preflight installed from whatever it mounted. Its only integrity
check was `pip --require-hashes` against `requirements-offline.txt`, and that file
arrives inside the tree being checked. A substituted dataset shipping its own
matching requirements file satisfies the hash check exactly, so the check bound
nothing it had not also been handed.

Verifying the published dataset remotely, as [[D-0031]] did, is not a substitute.
It establishes what was published at one moment. It says nothing about the bytes a
kernel reads at another, and the gap between those two moments is the whole
window.

So the identity travels with the code. The package embeds the published payload's
tree token and the canonical records behind it, recomputes the mounted tree under
canonicalization `v1` before pip is invoked, refuses on any extra, missing,
renamed or modified file, and prints the expected and observed identities into the
log so a mismatch is readable in the run rather than reconstructed from it. A
rename appears as one missing path and one extra path, which is what a rename is.

Three consequences bind later work.

**The trust anchor may not come from the thing being trusted.** This is the rule
[[D-0027]] earned for the notebook's own payload, now applied to its inputs
instead of only to itself. A packaged run verifies what it mounts against
something it brought.

**Canonicalization `v1` now exists twice, and the copy is held to the original.**
The preflight may not import Biohub-X, because it has to run on an image where
Biohub-X cannot ([[D-0029]]), so the authoritative walk cannot travel as code. It
travels as behaviour: a test executes the notebook's own verifier and requires it
to return exactly what `biohubx.hashing.tree_digest` returns for the same tree,
including the empty-directory record and the zero-byte-file size that are the two
cases easiest to get subtly wrong. Without that test this would be a
reimplementation that agrees with the original until the day it does not, which
[[D-0008]] rejected for the metric and which is no safer for an identity.

**The identity checked is the published payload, never the upload bundle.** Kaggle
consumes `dataset-metadata.json`, so no kernel ever mounts the bundle, and
embedding the bundle's identity would refuse every correct mount ([[D-0031]]).
Both are recorded in the package manifest, each marked with whether it is verified
at runtime or kept for provenance.

**What this does not do.** It does not authenticate the dataset's origin, and it
is not a defence against the account holder. Anyone able to replace the dataset
can still replace it. What they can no longer do is have the run proceed, which is
the difference between a substitution that is caught and one that installs.

## D-0033 - The identity locates the input as well as verifying it

**Date:** 2026-09-03
**Status:** accepted

E03-WHEELHOUSE-PREFLIGHT-01 was authorised, pushed, and refused in 8.752 seconds
because `/kaggle/input/biohubx-wheelhouse-zarr-cp312-linux` does not exist on the
image. The dataset was attached: Kaggle's own recorded kernel metadata lists it
and the dataset reports ready. The path was the guess that failed ([[R-0007]]).

The evidence was already in hand and went unused. E03-SMOKE-02 established that
competition data lives at `/kaggle/input/competitions/<slug>` on this image, so
inputs are not mounted at `/kaggle/input/<slug>`. The competition path was checked
against that measurement before the push and the wheelhouse path three lines above
it was not, which is the same adjacent-check failure this repository keeps
recording, committed by the agent that had just written up the previous instance.

The response is not a better guess. **The package already carries the tree's
identity for [[D-0032]], and an identity answers a stronger question than a path
does: a name says where to look, a digest says when you have actually found it.**
So the search enumerates what is mounted and selects the tree whose canonical
digest equals the embedded one. Locating and verifying then rest on one fact
rather than two, and the layout stops being something this repository has to know.

Three properties keep it honest.

**Shape is checked before anything is hashed, with an early abort.** A candidate
larger than the tree being sought cannot be that tree, so the walk abandons it the
moment the file count or byte total is exceeded. The competition mount is a corpus
of hundreds of chunked volumes and must never be read to find a directory of four
wheels. Measured on a simulated mount with a 480-file corpus-shaped decoy: three
trees hashed, none of them the decoy, 0.369 seconds.

**The enumeration is bounded and single-level, never a recursive glob**, four
levels deep, which covers a plain mount, a mount under a kind, and a mount under a
kind and an owner, with one spare. A test asserts no `rglob` and no `**` reaches
the generated notebook.

**A miss refuses with a listing of what is actually mounted.** The previous
refusal named the path it wanted and not the paths it had, so its log could not
distinguish a dataset that was never attached from one mounted elsewhere. That
distinction is worth a run, and this is what stops it costing one.

A directory containing only the wheelhouse has the wheelhouse's shape, is
therefore hashed, and is rejected on identity. That is not waste; it is the reason
the criterion is the digest and not the shape.

**What this does not settle.** Where Kaggle actually mounts a private dataset on
this image remains unmeasured, and this decision deliberately removes the need to
know. If the next run resolves, the path it reports becomes a Biohub-X
measurement rather than an assumption.

## D-0034 - A requested accelerator is not an allocated one, and the run must know the difference

**Date:** 2026-09-03
**Status:** accepted

E03-SMOKE-02 was authorised for one T4. `kernel-metadata.json` requested
`nvidiaTeslaT4`, Kaggle allocated a P100, and the packaged guard checked
`torch.cuda.device_count()`, which was 1, and passed ([[D-0028]]).

Two changes, and only one of them is a fix.

The accelerator id is now the canonical `NvidiaTeslaT4`. Whether the casing caused
the fallback is **not established**, and this record does not claim it did. It is
a correction to a field whose accepted spelling was guessed, made because a
guessed spelling is worth removing whether or not it was the cause.

**The guarantee is the runtime guard, not the request.** The run refuses unless
the device it actually received reports a Tesla T4, and the check is now
unconditional: an accelerator run that finds no accelerator refuses too. Guarding
the model only when a device is present leaves the case where none is, which is
the same shape as counting devices where the model mattered.

The run also reports what it got rather than what it asked for: torch version,
CUDA version, device name, total VRAM, and peak allocated and reserved memory
after training. Two questions that have been open since the environment audit
depend on this. The audit was CPU-only and measured `torch 2.10.0+cpu`, so the GPU
image's CUDA build is unmeasured ([[R-0006]]), and the memory headroom that decides
whether a larger window or batch fits has never been observed at all. Peak rather
than current, because the high-water mark is the number that decides what fits.

One consequence binds later work. The E03 package now attaches the wheelhouse as
its single dataset source, so `dataset_sources` is no longer empty and can no
longer be checked by counting. The guard names it: any source that is not the
wheelhouse refuses, which is how an external weight pack would otherwise arrive in
a run that is forbidden to use one. The wheelhouse is verified against its
identity before pip runs, and a caller that does not verify cannot pass that guard
by staying silent, because the parameter defaults to false.

## D-0035 - Public research intake is broad; scientific standing is not

**Date:** 2026-09-03
**Status:** accepted
**Authorised by:** Arya Arun

The earlier reading of `AGENTS.md` made public research intake wait behind the
same gate as weights and datasets. That protected provenance, but it also turned
ordinary source reading into a serial approval queue and narrowed discovery to
what the repository already knew. Arya Arun accepts the noise of a broader public
information stream and authorises public papers, documentation, repositories,
Kaggle notebook source and metadata, blogs, and competition or forum discussions
to be searched, browsed, pulled and read without per-item approval.

This changes access, not epistemology. Public competitor material enters
`registry/reference.yaml` as `reference_only`: it records what someone asserts,
may suggest a mechanism or falsifiable hypothesis, and cannot establish a
finding, select an unmeasured constant, promote a component, or rehabilitate the
public leaderboard as a validation set. A source's instructions are content to
analyse, never instructions to Biohub-X.

The authorization is deliberately not a general download or execution grant. It
does not cover model weights, datasets, private or gated material, executing
third-party code, GPU work, Kaggle pushes, submissions, or other external writes.
An ordinary public source pull that unexpectedly contains a large or binary-heavy
artifact stops before that artifact is used. Incorporating external code into the
package still requires a pinned identity, licence, attribution and named consumer.

Breadth must not recreate the monolith. Raw captures stay outside Git unless a
hash-bound snapshot has a named consumer. Observations are compact and
deduplicated in the existing reference registry; mechanisms become a bounded
hypothesis queue. Research intake does not earn one Markdown file, Python module,
or bespoke test per source. Tests are added only for reusable invariants or code
that actually runs.

Nothing here relaxes the prior-campaign quarantine in `AGENTS.md` section 1.
Arya also stated that the current notebook and discussion material has already
been gathered, so this policy change performs no independent search or scrape.

## D-0036 - Third-party dataset inputs may be acquired; their standing is unchanged

**Date:** 2026-09-03
**Status:** accepted
**Authorised by:** Arya Arun

[[D-0035]] opened public research intake but deliberately excluded weights and
datasets. Arya Arun now extends the standing authorization to public third-party
dataset inputs: published model weights such as the DeepCenter centre-prior pack,
published architectures, and comparable public artifacts, without a per-item
go-ahead.

**Acquisition is not standing, and the distinction is the whole record.** Section
6 applies unchanged before any such artifact enters the system: source, pinned
release, code licence, weight licence, training-data provenance, competition
eligibility, SHA-256 and exact role are recorded first, and a digest declared
before the download is checked after it. "Publicly downloadable" is still not
"cleared", and a weight licence is still not training-data provenance.

Three consequences bind.

An artifact whose training data is unknown or contaminated stays `blocked` in
`registry/models.yaml` and may not produce a held-out finding, be evaluated
against any split carved from the training data, or promote a component. That is
[[D-0021]] and [[D-0022]], and access does not touch it. The two `pilkwang` packs
Biohub-X already holds are exactly this case: one records training on all 199
annotated movies, which include the four byte-identical to the public-test
volumes ([[F-0010]], [[R-0002]]).

Downloading a thing is not a reason to use it. [[R-0010]] records that all three
public notebooks read this turn load those same quarantined packs, so their
leaderboard scores are contaminated in the way [[D-0020]] already established for
the reference. Acquiring DeepCenter would let Biohub-X study a centre-prior
interface; it would not make any score built on it evidence.

Large weights are never committed to Git, and raw captures stay outside it under
[[D-0035]]. What enters the repository is the registry record, not the bytes.

## D-0037 - Exploration runs as a factory; promotion remains a gate

**Date:** 2026-09-03
**Status:** accepted
**Authorised by:** Arya Arun

Per-source approval was only one serial bottleneck. Treating every cheap question
as a promotion experiment imposed the same registry packet on a CPU probe and a
held-out model comparison, while requiring a fresh authorization for every
package digest turned packaging repairs into human scheduling work. Arya Arun
authorises a research-and-test factory that continuously turns noisy public
signals into cheap falsification, without lowering the standard for a finding.

Biohub-X therefore has two execution lanes.

**The probe lane is deliberately light.** A local or CPU-only probe that cannot
promote anything declares only its ID, question, falsifier or stop condition,
inputs, split, budget and expected output. Multiple arms may share one declaration
and configuration. Local CPU probes on registered, cleared inputs need no
per-run approval. Private CPU-only Kaggle diagnostic and integration pushes also
have standing authorization when their local package gate passes, internet is
disabled, no submission is created, every input is registered and the declared
CPU budget is bounded. Every remote attempt remains a separate record. A probe
ends `integration_only`, `killed` or `invalid`; a useful signal graduates into a
promotion experiment rather than acquiring standing by enthusiasm.

**The promotion lane keeps the full contract.** Anything that can enter
`registry/findings.yaml` or promote a component declares the existing hypothesis,
falsifier, typed input identities, clean split, frozen configuration, budget,
outputs, forbidden changes, official promotion metric, ceiling and provenance.
Results remain per held-out embryo and never pooled only. Shared fields belong to
one experiment-family declaration instead of being copied into a row for every
arm.

**GPU approval is an envelope, not necessarily one digest.** A human go-ahead may
name an objective, allowed inputs, hardware class, maximum pushes or runs, total
compute budget, permitted arms and repair policy. Within that envelope, a
package may be rebuilt and an integration failure retried without a new approval
when the local gate passes and the scientific configuration, inputs and budget
do not change. Every digest and attempt is still recorded. Work outside the
envelope stops. A competition submission always retains its own explicit gate.

This removes ceremony, not controls: the prior-campaign quarantine, provenance,
unlabelled-region semantics, fold isolation, official metric, artifact identity
and submission gate are unchanged. It also does not reward repository growth.
Failed probes leave machine-readable reports and Git history, not dead modules,
notebooks, Markdown packets or one bespoke test per idea.

## D-0038 - A guard handed the expected value cannot check the observed one

**Date:** 2026-09-03
**Status:** accepted

E03-SMOKE-03 was approved for one T4. Kaggle allocated two. The packaged guard
compared the GPU count and passed, because the entry point handed it
`spec.expected_gpu_count` in place of the observed count whenever smoke mode was
on. It compared one to one. **The check was structurally incapable of failing in
the only mode that ever used it**, and every smoke this project has run was in
that mode.

The substitution had a reason, which is why it survived review: the pre-push gate
runs the same entry point on a machine with no accelerator, where an honest count
of zero would refuse. Rather than tell the guard that this was a local exercise,
the code told it the answer it wanted to hear.

So the fix is not to delete the exemption but to make it declare itself. The
observed count and the observed device name now always reach the guard. A local
CPU exercise sets one environment variable that the pre-push gate owns, the guard
skips the two hardware checks, and **it records in the report which checks it
skipped**. A manifest from a local exercise can no longer be mistaken for one
whose hardware was actually verified.

This is the third instance of one shape. [[D-0028]] recorded a device count
standing in for a device model. [[D-0033]] recorded a path assumed where an
identity was available. Here an expected value stood in for an observed one. In
all three the check sat next to the thing that mattered rather than on it, and in
all three it passed.

**What this does not excuse.** The device model guard worked: the run refused
nothing because it genuinely got Tesla T4 hardware, and the envelope named a model
rather than a count. The deviation cost nothing on this run, since the model
trains on one device and peak memory fits one card with 11.9 GB spare. What was
lost was the ability to notice, and noticing is the entire purpose of a guard.

**Open, and named rather than guessed.** Both `accelerator` and `machine_shape`
were set to `NvidiaTeslaT4` and the command line carried `--accelerator
NvidiaTeslaT4`, and two cards still arrived. Whether an accelerator request can
pin a count at all is unmeasured, and no further request will be described as
pinning one until it is.

## D-0039 - Research may roam and execute in isolation; integration and external writes remain gated

**Date:** 2026-09-04
**Status:** accepted
**Authorised by:** Arya Arun

[[D-0035]] broadened what Biohub-X could read and [[D-0037]] made cheap
falsification a factory, but neither defined autonomous citation traversal,
scraping, third-party execution, shared resource limits or durable raw-source
provenance. Leaving those to interpretation made a broad authorization behave
conservatively in one turn and expansively in another.

Research autonomy is now explicit. A controller may decompose a question, spawn
parallel branches, follow citations recursively, cross disciplinary boundaries,
compare public implementations, abandon weak routes and repeat retrieval passes
without an artificial search-depth or agent-hop limit. The stopping rule is
marginal information gain relative to a shared request, transfer, time and disk
budget, not a fixed paper count.

Bounded scraping is authorized for genuinely public, unauthenticated resources
reachable through ordinary HTTP. Authentication barriers, paywalls, CAPTCHAs,
private APIs, credential reuse, IP rotation and anti-bot circumvention stay out
of scope. Agents use conservative per-host concurrency, back off on 429 and 503,
honour explicit rate limits and share counters so a hard limit cannot be evaded
by splitting work. The three-tier resource envelope in `AGENTS.md` distinguishes
autonomous work, continued work after a marginal-value check, and a hard stop.

Public notebook-linked resources may be fetched automatically within that
envelope. Research-cache acquisition is deliberately separated from system
incorporation: acquiring bytes permits inspection, not use as evidence or a
Biohub-X input. Each artifact receives a machine-readable ledger record and
important claims trace to an exact location. Metadata survives payload eviction;
raw captures remain outside Git unless a hash-bound snapshot has a named
consumer. Parallel agents check the ledger and cache before downloading, so
autonomy does not multiply large transfers.

Third-party code may execute, including installation, tests, notebooks,
inference and instrumentation, but only in a disposable sandbox with no secrets,
no general host mount, no external writes, bounded processes and resources, and
networking disabled or separately declared. The user's ordinary workstation
shell is not such a sandbox. Trust escalates from static inspection to dependency
inspection to isolated execution and then, only when separately controlled, to
network access.

External constants may freely seed provenance-labelled exploratory grids. They
remain hypotheses until the target data, a clean validation experiment, physical
reasoning or an official constraint independently supports them. This removes a
discovery bottleneck without turning a neighboring paper's configuration into a
biological fact.

The autonomy ends at consequence boundaries. Private material, hard resource
limits, unscoped GPU work, submissions and external writes still stop. Existing
explicit standing grants, such as bounded private CPU-only Kaggle diagnostics,
remain explicit grants rather than becoming an implied general write authority.
A successful experiment only qualifies a component for integration; changing
the canonical pipeline or merging it into `main` requires Arya Arun or a named
delegated controller to approve promotion. The prior-campaign quarantine remains
absolute.

## D-0040 - An optimisation that changes an answer is not an optimisation

**Date:** 2026-09-04
**Status:** accepted

Physical-radius suppression compared each candidate with every point accepted so
far, in a Python loop. On the small windows it was written for that was fine. On a
full-extent frame at a permissive threshold it is ten thousand candidates against
a thousand accepted, and E04's first run did not finish.

It is now bucketed: accepted points are indexed on a grid whose cell is the
suppression radius, and a candidate is compared only with the twenty-seven cells
around it. **This is exact rather than approximate**, because any point within the
radius necessarily falls in one of those cells, and that is the only reason it is
acceptable here.

The reason it has to be exact is that four recorded findings rest on this
detector's output. [[F-0006]] counts its instances on the synthetic fixture,
[[F-0017]] its node recall and candidate reach on real bytes, [[F-0019]] its
proposals after peak extraction, and [[F-0025]] its reachability against DoG. A
suppression that returned a different set would silently invalidate all four while
every test that only checks shapes kept passing.

So the equivalence is asserted rather than argued. A test builds a random frame,
runs the bucketed implementation, and independently reimplements the brute-force
version inside the test rather than refactoring the original into it, then
requires the two to return the same points in the same order. The end-to-end slice
that F-0006 rests on runs unchanged alongside it.

**The other half of the fix was not the algorithm.** E04's first attempt also
passed the classical detector an absolute intensity threshold of 0.30, which on a
normalised full-extent frame admits millions of candidates and means something
different on every movie. Both sources are now cut at a quantile of their own
response. The threshold stops being the thing that differs between them, which is
what lets the node budget be the thing that is matched.


## D-0041 - The DoG operating configuration is strict peaks on two scales, stated explicitly

**Date:** 2026-09-04
**Status:** accepted

[[F-0030]] measured that greedy physical-radius suppression packs the
above-threshold region rather than returning maxima, and [[F-0032]] measured what
changes when a candidate has to be a strict 26-neighbourhood maximum: reach near
0.92 and 0.90 on the two embryos at a proposal count at or below the official cell
estimate, against 0.72 and 0.68 for the packed detector at the same counts, and
with the two-scale bank (2.0, 3.0) micrometres beating three scales on both
embryos wherever the budget is at least twice the estimate.

**Going forward, the DoG configuration carried into scored experiments is
`local_maxima_only=True` with radii (2.0, 3.0).** It is stated as an explicit
argument on every experiment declaration and every command, not baked into a
default.

The defaults do not change, and that is deliberate. `detect_instances` keeps
packed suppression and the three-scale bank as its defaults, and `evaluate
proposals` keeps them as its option defaults, because [[F-0025]], [[F-0026]],
[[F-0027]], [[F-0028]], [[F-0029]] and [[F-0031]] record commands that relied on
those defaults, and a finding whose command no longer reproduces is a finding
that has quietly been rewritten. The cost is that a caller who omits the flags
gets the old detector. That is accepted, because the registry says which
configuration each number came from and the alternative silently changes what
six recorded numbers mean.

Two things this does not decide. It does not promote DoG: reachability is not the
official metric and no scored fold exists. And it does not settle the bank at the
neutral budget on the dense embryo, where three scales still lead by 0.015;
whether a per-scale peak union recovers that without re-merging neighbours is
H-11c and is cheap to measure.

The chain that led here is worth naming once. A proposal count identical across
three different scale banks was the tell ([[F-0030]]); the first fix admitted
plateaus as peaks, the second lost every peak on a volume face to replicated
padding, and each was caught by a synthetic blob before a corpus number was
recorded. Three instrument defects in one afternoon, none of them in the science,
all of them the kind that would have passed a shape check.
