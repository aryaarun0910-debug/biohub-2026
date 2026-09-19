---
tags:
  - gotcha
---

# Gotcha: the wrong metric module fails silently

`src/biohub/metric.py` and `src/biohub/postprocess.py` contain lookalikes of
[[metric2]] and the stage ports — but they assume the **downsampled isotropic
64³ grid** with `GRID_UM = 1.625`.

On the [[Local Harness]] `.geff` graphs, which are original-voxel and
**anisotropic** (`SCALE = (1.625, 0.40625, 0.40625)`), they are **4× wrong in y
and x**.

**They do not raise.** They return plausible numbers.

Both now carry warning comments. Use [[metric2]] and
[[Script 91 Other Stages]] for anything touching those graphs.
