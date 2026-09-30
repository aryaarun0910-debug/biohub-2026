---
tags:
  - infra
---

# `scripts/make_variant.py`

Builds one-change notebook variants by **substituting** an existing
`os.environ[...]` line. Refuses to write unless the changed-line count matches
what was requested.

Its limitation is why [[make_env_variant]] exists: most interesting knobs are
never set in the notebook at all, so changing one means *adding* a line.

Warns when a key is covered by the [[Drift Guard]].
