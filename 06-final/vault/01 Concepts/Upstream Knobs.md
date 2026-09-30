---
tags:
  - concept
---

# Upstream Knobs

Everything before the [[ILP Linker]]'s output. The [[Local Harness]] **cannot
test any of it**, because it starts from that output.

| knob | deployed | stage |
|---|---|---|
| `DET_THRESHOLD` | 0.965 | [[Temporal UNet3D Detector]] |
| `ILP_APPEARANCE_WEIGHT` | 0.0 | [[ILP Linker]] |
| `ILP_DISAPPEARANCE_WEIGHT` | 2 | [[ILP Linker]] |
| `BIDIRECTIONAL_EDGE_WEIGHT` | 0.15 | [[Node Transformer Edge Scorer]] |
| `SECONDARY_EDGE_FEATURE_TTA_WEIGHT` | 0.75 | [[Node Transformer Edge Scorer]] |

All five sit under the [[Drift Guard]]. [[make_env_variant]] can now build them.

**Untested, and deliberately not yet pushed.** Changing the edge scorer moves the
ILP graph itself — the topology [[s08]] and [[s09]] were measured on — so by the
[[Transfer Lesson]] it would invalidate the post-processing gains and require
re-validating them on top. Each probe also costs a full kernel run with no local
preview.

This is the **only axis left** after [[Parametric Search Closed]].
