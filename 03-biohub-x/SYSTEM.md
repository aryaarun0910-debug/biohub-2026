# SYSTEM

Architecture, contracts, and the honest current state of Biohub-X.

## 1. Current state

**Phase 0 complete. The system has no scientific capability.**

What exists:

| Component | Module | State |
| --- | --- | --- |
| Content identity | `biohubx.hashing` | implemented, contract-tested |
| Artifact records and atomic writes | `biohubx.artifacts` | implemented, contract-tested |
| Typed CLI | `biohubx.cli` | two commands implemented |

What does not exist yet: data loading, coordinate contract, proposals,
representation, candidate graph, association, division, decoder, metric,
deployment. Those directories are absent from `src/biohubx/`, not stubbed.
A module is created when its first real consumer exists.

No score has been measured. No model has been trained or downloaded. No GPU has
been used. No Kaggle notebook exists.

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

Two identities, never interchangeable:

- `raw_artifact_sha256` - SHA-256 of the exact bytes on disk. The identity of
  weights, exported graphs, built notebooks, submissions, and hash-bound
  snapshots.
- `canonical_text_sha256` - SHA-256 after canonicalization `v1`. The identity
  used for source and configuration drift.

Canonicalization `v1`: strict UTF-8 decode, strip BOM, CRLF and lone CR to LF,
append one trailing LF if the text is non-empty and lacks one. Trailing
whitespace and blank lines are preserved. Bumping the version invalidates every
recorded canonical digest and requires a DECISIONS.md entry.

Serialised form always states the kind, and the canonicalization version when
one applies:

```
raw_artifact_sha256:sha256:<64 hex>
canonical_text_sha256:sha256/v1:<64 hex>
```

A bare hexadecimal string is not an identity and is refused by
`biohubx.hashing.Digest.parse`.

Repository byte policy is fixed in `.gitattributes` and was set before the first
digest was recorded: text is stored and checked out with LF everywhere, and
`research/primitives-dossier.md` is marked `-text` so its recorded raw digest is
an assertion about the original source bytes.

### 4.2 Coordinates (not implemented)

Physical `(z, y, x)` derived from official voxel spacing. The module must
preserve both voxel and physical coordinates, declare axis order, round-trip
within a stated tolerance, refuse isotropic substitution, and ship a negative
control proving the isotropic convention fails. No downstream component may
receive an anonymous three-column coordinate tensor.

### 4.3 Instances, representation, candidates, predictions, lineage (not implemented)

Specified in the mission and written when their first consumer exists. Two rules
already bind them:

- Candidate generation and candidate scoring are separate contracts, and the
  graph reports true-edge reach so a scorer is never blamed for an edge it was
  never offered.
- The no-parent state is a learned output, not a threshold on the best edge.

### 4.4 CLI (partially implemented)

Every command accepts a typed configuration, prints a positive heartbeat,
writes outputs atomically, writes a manifest, refuses missing or ambiguous
inputs, records source and artifact digests, and never silently falls back to a
default data path.

Implemented:

- `biohubx artifacts digest PATH --kind {raw_artifact_sha256,canonical_text_sha256}`
- `biohubx artifacts verify [--registry PATH]`

Planned, absent until a consumer exists: `data validate`, `data make-splits`,
`proposals infer`, `representation acquire`, `train`, `infer`, `evaluate`,
`package kaggle`.

## 5. Evaluation policy

The official competition metric is the only end-to-end promotion metric. Every
evaluation reports, separately and never only pooled:

node recall; node precision and count behaviour; candidate-edge reach; raw edge
Jaccard; adjusted edge Jaccard; division TP/FP/FN; division Jaccard; combined
official score; runtime; peak memory; and each held-out domain and direction on
its own.

Retrieval@k, AUROC, t-SNE, UMAP, PHATE and segmentation IoU are diagnostics.
They cannot promote anything on their own.

## 6. Phase gates

| Phase | Gate | State |
| --- | --- | --- |
| 0 | repository foundation, identity contract, registries, CLI heartbeat, contract tests | done |
| 1 | official metric from its authoritative source, pinned revision, adapter, synthetic graphs with known scores, coordinate and lineage contracts | next |
| 2 | end-to-end software vertical slice: fixture to candidate instances to representation to sparse T=2 graph to minimal matcher to legal final graph to official scorer | not started |
| 3 | real representation smoke, CPU preflight, one protected GPU smoke, stop and report before launch | not started |
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

`research/shortlist.yaml` is empty at Phase 0 by construction: a primitive may
enter it only when a live Biohub-X component consumes it.
