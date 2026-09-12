# Foundations

What the system can and cannot tell us about itself. Written 2026-09-12, when the honest answer
to "why did we miss that division?" was "no idea, but the scalar went up."

## The problem this fixes

Every threshold in `Config` was set by grid search against a single number. That is blind
hill-climbing. At score 1.1412 we had 25 missed and 81 false divisions and no way to name one of
them, so we could not distinguish *"the gate is wrong"* from *"the candidate was never offered"* —
which are opposite fixes.

## 1. Decision trace (`src/biohub/trace.py`)

Stages record the individual decisions they make, not just their aggregate effect. `resolve`
reports every rejected fork with the rule that rejected it (`cos`/`sister`/`parent`/`diverge`/
`degenerate`) and the value that failed.

Off unless a stage is handed a `Trace`, so the submission path pays nothing.

```python
tr = Trace(); g = resolve(g, cfg, trace=tr)
tr.reasons("fork_reject")        # Counter({'diverge': 46, ...})
```

## 2. Error atlas (`tools/error_atlas.py`)

Attributes every ground-truth division to the stage that lost it. Exact, not distance-matched:
under oracle detection node index `i` **is** ground-truth node `i`.

| verdict | meaning |
|---|---|
| `RECOVERED` | both daughter edges survive to the final graph |
| `NOT_PROPOSED` | `score_edges` never offered a daughter edge as a candidate |
| `LOST_IN_ASSIGNMENT` | the candidate existed; the Hungarian pass gave the target away |
| `FORK_REJECTED/<rule>` | the fork pass tried and a named rule refused it |
| `REPAIR_DESTROYED` | `resolve` produced it and `repair` removed it |

It **self-validates**: `RECOVERED` must equal the scorer's TP count. It does (126 = 126).

First run paid for itself twice — it confirmed `repair` destroys nothing (`REPAIR_DESTROYED 0`,
agreeing with EXP-10) and put a number on the divergence gate's cost (8 real divisions), which
Zebrahub had independently flagged as our most fragile default.

## 3. Knowledge base interface (`tools/kb.py`)

Facts were written with hand-rolled SQL at every call site. In one session that cost **four**
silently rolled-back transactions — a missing bind parameter, a `validity` CHECK violation, a
`repo_id` column that does not exist, and a `superseded_by` given as a key where an id was
required. Each printed a traceback *after* the work and lost it.

`kb.py` is now the only way in. It validates before writing, commits before printing, refuses a
fact with no source or no quote, and stamps the git SHA.

    kb.py record --topic ... --key ... --value ... --quote ... --source ...
    kb.py supersede --key old --by new --why "..."
    kb.py ask <topic> | kb.py open | kb.py check

`kb.py check` is the provenance audit. Its first run found **66 violations** of the standing rule
— 45 active facts with no source, 15 with no quote, 6 superseded rows with no successor. Five were
back-filled from frozen host evidence, 6 dangling rows retracted, and the remaining 40 pre-rule
facts **quarantined at `confidence=low` / `validity=UNKNOWN`** rather than given invented
provenance. They are reported, never presented as evidence.

## What is still naive — do not mistake this for done

- **No mechanistic interpretability of a model, because there is no model.** `detect` and
  `refine` are stubs. Everything above interprets the *linker*. When the detector exists it needs
  its own layer: which voxels drove a peak, what the refinement head learned, where the two
  supervision regimes disagree.
- **The atlas needs oracle detection to be exact.** Under a real detector the index identity
  breaks and attribution becomes distance-matched, with its own failure modes.
- **No cross-stage counterfactuals.** We can say a division was `NOT_PROPOSED`; we cannot yet say
  what minimum change to `score_edges` would have proposed it.
- **False divisions are barely attributed.** The atlas classifies misses well and false positives
  crudely. 81 FP is the larger number and the weaker analysis.
- **No versioned data contract.** Nothing pins which `data/` build a result came from.
