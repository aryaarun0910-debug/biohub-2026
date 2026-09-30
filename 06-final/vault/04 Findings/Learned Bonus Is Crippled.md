---
tags:
  - finding
---

# `MOTION_RELINK_LEARNED_BONUS` Is Structurally Crippled

Every `solution` flag in the prediction `.geff` is True, so the file stores
**only ILP-selected edges**. Therefore `prob.get(pair, 0.0)` returns 0 for every
alternative edge.

The `−β·prob` term is not a likelihood at all — it is a flat ~0.9 µm
**incumbency discount** applied to whichever edge the [[ILP Linker]] already chose.

A board submission moved this knob 1.0→2.0 and scored 0.932.

Moot in the current lineage anyway, since [[Motion Relink]] is off from [[s05]]
onward.

Related: [[Node Transformer Edge Scorer]]
