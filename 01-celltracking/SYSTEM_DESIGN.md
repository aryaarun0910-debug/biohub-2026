# Lean system design

## Goal

Move the official score with the smallest reliable experiment. The repository is an
execution product, not a transcript of every research conversation.

## Four layers

### 1. Competition core

`src/biotrack/` owns metric integration, graph invariants, conversion, and the deployed
wrapper. It changes only for a measured scoring mechanism or correctness defect.

### 2. One active scoring pipeline

P3 harmonic is the deployment baseline. The only active research branch is D1 detector
acceptance diagnosis and re-acceptance. Historical detectors, selectors, division cascades,
motion experiments, and ILP variants are not part of the runtime surface.

### 3. Experiment factory

Every experiment has one machine-readable spec, one immutable input manifest, one output
artifact directory outside Git, and one compact ledger row containing:

- hypothesis and parent;
- code/config/input hashes;
- validation basis;
- pooled and per-family metrics;
- runtime and resource use;
- decision and falsification reason.

Smoke proves plumbing. A representative pilot estimates a mechanism. Full LOEO confirms it.
These stages must never be conflated.

### 4. Evidence store

Git stores code, small manifests, canonical summaries, and tests. The sibling research store
holds raw sources, caches, generated features, agent reports, and full experiment output.
Git tag `pre-lean-2026-08-07` preserves the pre-reset active tree.

## Test architecture

Tests may block execution only for software invariants:

- official scorer parity;
- graph degree and coordinate validity;
- deterministic manifests and joins;
- checkpoint/family provenance;
- serialization schemas;
- baseline byte parity.

Tests must not encode a scientific conclusion such as "method X can never work", a score
promotion threshold, or a temporary research policy. Those belong in evidence and decision
records. A new hypothesis may run in an isolated experimental path without weakening core
invariants.

## Language policy

Python remains the control plane, model language, scorer language, and Kaggle packaging
language. Polars/NumPy/PyTorch already execute their expensive kernels in native code.

Rust or C++ is permitted only when all of the following hold:

1. profiling shows one stable CPU function consumes at least 30% of end-to-end wall time;
2. algorithmic and vectorised Python improvements are exhausted;
3. a pure Python reference and property/parity tests exist;
4. the compiled artifact can be reproduced inside the internet-off Kaggle environment;
5. expected saved compute exceeds integration and packaging cost.

Use Rust for safe parallel graph/candidate kernels; use C++/CUDA only for an unavoidable
PyTorch extension. TypeScript has no role in the scoring pipeline. It is suitable only for a
separate dashboard if one becomes necessary.

Polyglot code is therefore an optimisation result, never an architectural starting point.

## Research priorities

1. Detector acceptance/head diagnosis on 6bba.
2. Minimal re-acceptance policy under exact node/edge economics.
3. Only if representation failure is demonstrated: paired sparse-supervision training.
4. HOCT or CoTracker only after the deployed-gate oracle leaves material association value
   and a mask/detection bridge is demonstrated cheaply.

The proposed multi-detector + HOCT + ILP + graph-ensemble championship stack is a useful
long-range menu, not the next implementation. Several components have already failed on this
dataset or sit below the detection ceiling. They must earn entry independently.
