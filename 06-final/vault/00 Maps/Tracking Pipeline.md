---
tags:
  - moc
---

# Tracking Pipeline

Nothing here trains a model. The whole entry is a public Kaggle notebook running
[[Frozen Weights]], all code in cell index 2 (~214k chars).

**Detection and linking:**
[[Temporal UNet3D Detector]] → [[Node Transformer Edge Scorer]] → [[ILP Linker]]

**Then a rule-based post-processing chain** — this is where every change we have
made lives, driven by `BIOHUB_*` environment variables in the config cell:

1. [[Motion Relink]] ← **removed in [[s05]]**
2. [[Single Parent Repair]]
3. [[Gap Closing]]
4. [[Gap2 Recovery]] ← **moved after safe division in [[s08]]**
5. [[Safe Division]]
6. [[Prune Isolated]]
7. [[Short Track Filter]]
8. [[Linefit Smoothing]] ← **retuned in [[s09]]**

[[DeepCenter]] supplies vetoes at two points.

The chain is now **[[Parametric Search Closed]]** — every knob re-priced, nothing left.
