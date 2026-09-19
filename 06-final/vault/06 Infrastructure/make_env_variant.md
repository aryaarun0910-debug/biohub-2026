---
tags:
  - infra
---

# `scripts/make_env_variant.py`

Adds or substitutes **one** env line on **any parent notebook**, so variants
chain: [[s08]] is [[s05]] + a reorder, [[s09]] is [[s08]] + one line.

Auto-detects its mode:
- **ADD** — key unset in the parent (sits at its `os.environ.get` default)
- **SUBSTITUTE** — key already set, rewritten in place

When the key is under the [[Drift Guard]], it rewrites the guard entry **in the
same edit** and proves the result is exactly those two lines with no other
guarded key disturbed. That is what unlocks [[Upstream Knobs]].

Five proofs, including two that catch silent no-ops: the key must actually be
**read** somewhere, and the value must **differ** from the current one.

Also enforces `slug == slugify(title)` — see [[Kaggle Slug From Title]].
