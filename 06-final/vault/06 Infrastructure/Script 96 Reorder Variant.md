---
tags:
  - infra
---

# `scripts/96_reorder_variant.py`

Builds the [[Gap2 Recovery]] reorder and **proves** it seven ways. Targets:
`s07` (wrong base, kept for comparison) and `s08`.

1. **line diff** — exactly 4 lines, 2 statements moved
2. **AST** — 558 top-level statements, exactly 1 differs; within
   `filter_output_graph`, `sorted(old) == sorted(new)`, i.e. a strict permutation
3. **compile**
4. **[[Drift Guard]]** — the diff sets no `os.environ` value and touches no guarded key
5. **behavioural** — runs the notebook's *own* stage functions on two toy graphs,
   confirming a contested orphan goes to safe division and uncontested gap2
   pairs are bit-identical, synthetic node ids included
6. **env insertion** — the [[s05]] line adds exactly one top-level statement
7. **diff vs the shipped [[s05]] notebook** — exactly the two moved statements

Proof 7 is what makes "[[One Change Per Submission]]" *demonstrated* rather than
asserted.
