---
tags:
  - gotcha
---

# Gotcha: Kaggle slugs come from the TITLE

Kaggle derives the live kernel slug by **slugifying the title**, ignoring the
`id` field in `kernel-metadata.json`.

Evidence:
- [[s05]] shipped `"id": "aryaarun07/biohub-s05-no-relink"` and went live at
  **`biohub-s05-no-motion-relink`** = slugify("Biohub S05 no motion relink").
- [[s06]]'s log file is `biohub-s06-linefit-weight-0-4.log`, from
  "Biohub S06 linefit weight 0.4".

**Worse, the failure is misleading:** querying the metadata id returns
*"Permission 'kernels.get' was denied"*, which reads like a private-notebook or
auth problem, not a wrong name.

Both builders now **abort** unless `slug == slugify(title)`.

Related: [[make_env_variant]], [[Script 96 Reorder Variant]]
