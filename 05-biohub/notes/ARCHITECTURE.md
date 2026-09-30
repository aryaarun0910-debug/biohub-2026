# System architecture

Written 2026-09-11, after the five pre-Mac experiments. Every seam below exists because a
measurement demanded it, not because it is tidy. Where a measurement is cited, it is in the base.

## The shape

    volume ──▶ detect ──▶ refine ──▶ score_edges ──▶ resolve ──▶ repair ──▶ write
                          └──────── Graph (MUTABLE coords) ────────┘

Six stages. Each takes a `Graph` and a `Config`, returns a `Graph` and a `StageDelta`.

## Why these seams and not others

**`detect` / `refine` are separate** because refinement is gated. EXP-3 measured the centroid
cliff at 2.0→2.5 µm — loss quadruples across half a micron — and the refinement head buys an
18–24% error reduction. That is worth more than anything else a detector can do, but it must
prove ≥15% reduction at ≥99% recall before it ships. A gate needs a seam.

Supervision differs across the seam, which is the real reason it exists:
`node_head` is supervised **densely** (FOCUS-3D pseudo-centres), `refine_head` **sparsely**
(Kaggle, conditional on detected peaks). Sparse labels are only safe where they are evaluated at
peaks that already exist. Training a detector on sparse labels teaches every unannotated real
cell to be a hard negative — the documented cause of loss blowing up at ~10 epochs.

**`score_edges` / `resolve` are separate** and this is the most load-bearing split in the
system. Measured: accumulating candidates to a 0.99 cutoff reaches **0.966** edge recall, while
greedy top-1 selection gets **0.942**. The right edges are already in the candidate set; the
selector loses them. Fusing scoring and selection makes that +0.024 unreachable by construction.

**`resolve` must be allowed to fork.** EXP-5, oracle detection over all 199 datasets: a strict
1:1 Hungarian scores `division_jaccard` **0.0000, tp/fp/fn 0/0/151** — it misses every division
by construction. Permitting a second child within 12/18 µm recovers **142 of 151** for an
edge-term cost of **0.0002**. Forking is not a post-process; it is a property of the resolver.

**`repair` runs after `resolve` and may not delete forks.** The public stack's defining bug is
that its motion-relink rebuilds the edge list from a 1:1 assignment *after* `safe_div` has
inserted forks, destroying them. This is enforced, not merely documented — see the delta ledger.

**`write` validates before it writes.** `tools/validate_submission.py`, stdlib-only so it runs
inside the internet-disabled notebook. A non-consecutive edge silently scores 0.0 with no error.

## Nodes are mutable

The `Graph` carries **mutable coordinates**. Detection and localisation are not separable: many
false positives sit very close to a true node and are recoverable by moving them —
*"during linking, my zxy must change… you cannot detect and fix a location."* Any stage may
refine a coordinate; no stage may silently drop a node without recording it.

## The delta ledger

Every stage returns:

    StageDelta(nodes_added, nodes_removed, nodes_moved,
               edges_added, edges_removed,
               forks_created, forks_destroyed,
               wall_s)

Printed as a table at the end of every run. A line reading

    resolve   forks_created  +187
    repair    forks_destroyed -184

is the entire public field's bug, visible in one glance. Without this the only observable is the
final score, which is exactly how a frozen division term went unnoticed by ~400 teams.

## Configuration

One frozen dataclass, environment-overridable — the same pattern the 0.940–0.947 band uses, and
it demonstrably works: that whole band is one codebase differing only in `BIOHUB_*` values.
Every constant that tonight's experiments touched is a field: detection threshold, NMS radius,
motion gate, fork parent/sister gates, fork acceptance features, min track length.

No plugin registry. Stages are selected by name from a dict. The prior campaign's own warning:
drop the ceremony a two-month campaign could afford and twelve days cannot.

## Devices

One function, `resolve_device()`, returns `mps` / `cuda` / `cpu`. Every stage takes `device` as
an argument and nothing else ever inspects it. **PyTorch only** — MLX cannot produce a
submission, and torch 2.14 erased its conv3d advantage. `bias=True` on every `Conv3d`; decoders
are `Upsample`+`Conv3d`, never `ConvTranspose3d`, which is unusable on MPS.

## Scoring is a stage, not an afterthought

Every run self-scores against held-out ground truth and reports **`adj_edge_jaccard` and
`division_jaccard` separately**. The leaderboard only ever returns their sum, and that is
precisely how the field mistook a frozen division term for a working one — and how our own p15
ablation was misread as +0.019 of division value when it was in fact −0.0041.

Scoring uses the **patched** scorer at `support/`. The pre-patch copy is quarantined under
`support/PRE-PATCH-DO-NOT-USE/`; anything validated against it is calibrated to a closed exploit.

## Layout

    src/biohub/
      contracts.py     Graph, Nodes, StageDelta, Config
      device.py        resolve_device()
      detect.py        dual-head UNet + peak extraction
      refine.py        refinement head + learned peak prune
      edges.py         candidate scoring (all-pairs, gated)
      resolve.py       assignment, fork-permitting
      repair.py        gap-node interpolation, track filters
      submit.py        graph -> CSV, validates before writing
      score.py         patched scorer, adj_edge and divJ separately
    tools/             harnesses that already exist and stay outside the package

## Invariants the writer enforces

Derived from measurement, not convention:

1. Every edge spans exactly one frame. All 7,998 GT edges do; the scorer drops anything else.
2. No node has out-degree > 2. Every GT fork is strictly binary.
3. No node sits outside `t 0–99, z 0–63, y 0–255, x 0–255`. Verified against the GT.
4. The graph is never edgeless — `evaluate()` returns early and raises.
5. Coordinates are integer voxel indices. GT is `int64`; there is no sub-voxel signal to emit.
