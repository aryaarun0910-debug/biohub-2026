---
tags:
  - stage
---

# ILP Linker

Integer linear program producing a global frame-to-frame assignment. Its output
`.geff` files are what the [[Local Harness]] reads — so everything downstream is
testable in seconds, and everything upstream is not.

Weights `BIOHUB_ILP_APPEARANCE_WEIGHT` (0.0) and
`BIOHUB_ILP_DISAPPEARANCE_WEIGHT` (2) are [[Upstream Knobs]].

The ILP already produces forks (divisions). [[Motion Relink]] then wipes them —
the first instance of the [[Ordering Bug Class]].

Graph shape note: no isolated nodes, minimum component size 4 in every film — the
ILP enforces this, which is why [[Prune Isolated]] is inert.
