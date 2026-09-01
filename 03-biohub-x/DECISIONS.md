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
