---
tags:
  - stage
---

# Prune Isolated

Drops nodes with no incoming and no outgoing edge.

**Completely inert on every chain tested** — `prune=False` returns exactly
0.00000. The [[ILP Linker]] already guarantees no isolated nodes and a minimum
component size of 4, so there is nothing for this stage to remove.

Related: [[Inert Variables]], [[Parametric Search Closed]]
