---
tags:
  - infra
---

# `scripts/24_keep_ilp_edges.py`

Supplies `load_pred()`, `load_gt()` and **`safe_div()`** — the deployed
[[Safe Division]] rule reimplemented on original-voxel coordinates.

`safe_div` is fully parameterised (`parent_max`, `sister_max`, `child_max`,
`tau`, `diverge`, `frame_cap`, `glob_cap`), which is what made
[[Script 101 Safediv Gates]] possible.

Other scripts import it by `exec`-ing the module prefix — a pattern with a
sharp edge, see [[Docstring Shadowing]].
