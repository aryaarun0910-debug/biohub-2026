---
title: Knowledge layer - how to use this
tags: [moc, readme]
generated: true
---

# Knowledge layer - how to use this

> [!info] Generated navigation. Do not edit.
> Built by `scripts/core/knowledge.py` from the canonical catalog. The registry is the
> only source of truth for numbers; every value below is a LINK, never a copy.


This directory is GENERATED. Do not edit it; edit the registry or the code it describes
and regenerate. Everything here is a VIEW over `research/00-system/registry/generated/catalog/`.

## Obsidian
1. Open the REPOSITORY ROOT as an Obsidian vault (not this folder).
2. Start at [[00-command-center]].
3. Entity notes live in `knowledge/entities/` and are named for their STABLE ID, so a
   renamed display title cannot break an edge.
4. Note bodies carry NO VALUES. A fact note gives provenance and validity and points at
   `facts.yaml`, which is the only source of truth for numbers.

## Regenerate everything
ORDER MATTERS. The catalog catalogues these generators, and `knowledge.py` reads the
catalog - so after editing any generator, run `catalog.py` FIRST, then `knowledge.py`,
then `catalog.py` again to pick up the generator's own new digest. Running the pair twice
reaches the fixed point; the `--check` drift locks fail until it does.
```
python scripts/core/catalog.py            # the canonical catalog
python scripts/core/receipt_envelope.py   # sanitized receipt envelopes
python scripts/core/knowledge.py          # this layer
python scripts/core/rag_index.py --build  # the local RAG index
```
Each has a `--check` mode used by the test suite as a drift lock.

## Ask the index a question
```
python scripts/core/rag_index.py --query "what is the operational base"
python scripts/core/rag_index.py --validate    # the standing validation queries
```

## Compare two notebooks
```
python -c "import json;c=json.load(open('research/00-system/registry/generated/catalog/notebooks.json'))['notebooks'];print(c['kaggle_p38_relink_bonus_b2']['delta_from_parent'])"
```

## Script lifecycle and test ownership
See [[40-scripts]] and [[50-tests]], both generated from the
catalog's `lifecycle` / `test_class` fields.
