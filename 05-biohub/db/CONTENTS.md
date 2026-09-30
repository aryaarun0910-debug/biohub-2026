# What is in this database, and how to read it

Generated 2026-09-11. Regenerate the counts with `python3 tools/ask.py facts <topic>`.

## The one-minute version

```
python3 tools/ask.py status      # deadline, leaderboard shape, repo state
python3 tools/ask.py gap         # what separates us from the podium
python3 tools/ask.py todo        # open decisions
python3 tools/ask.py facts division    # START HERE - the highest-value topic
python3 tools/ask.py facts strategy    # the board, decomposed
```

## Topics

`fact.topic` partitions the base. CURATED topics are judged and written by hand; `*-forum`
and `code-*` topics are raw agent proposals, every one carrying a verbatim quote.

| topic | facts | what it is |
|---|---|---|
| `prior-campaign` | 272 (11 sup.) | The monolith's own registry, ingested verbatim with its provenance AND validity preserved. Superseded rows include FACT-0001 (0.925 was true only to 2026-08-24; we reached 0.932). |
| `meta-forum` | 147 | Raw quoted claims from the 40 leaderboard/submission/rules threads. |
| `method-forum` | 90 | Raw quoted claims from the 18 detection/linking threads. |
| `division-forum` | 82 | Raw quoted claims from the 10 division threads, plus an original dense measurement on zebrahub ZSNS003. |
| `public-stack` | 79 | Teardown of the honest 0.947 family - one 4,200-line codebase behind five titles. |
| `sweep-forum` | 70 | Raw quoted claims from the all-102-topic sweep (bodies, not titles). |
| `code-scored` | 64 | Teardown of 63 of the 203 scored notebooks, incl. output-level exploit census. |
| `data-forum` | 57 | Raw quoted claims from the 17 data/ground-truth threads. |
| `metric-exploit` | 39 (2 sup.) | The hub-node exploit: mechanism, the 2026-07-17 patch, the 07-23 rescore, and proof it is dead. |
| `hardware` | 30 (4 sup.) | M5 Pro, Colab, Kaggle: what each can run, measured burn rates, the torch-2.14 conv3d fix. |
| `division` | 27 (3 sup.) | CURATED division findings - the highest-value topic in the base. Read this first. |
| `strategy` | 23 (5 sup.) | The board decomposed, the north star, and what the gap actually consists of. |
| `method` | 22 | CURATED technique findings, incl. the unused PU loss and the dead division-upweight hook. |
| `metric` | 22 | How the scorer really behaves, read from source rather than prose. |
| `data` | 20 | Corpus shape, annotation bias, the frozen-frame defect, external-data verdicts. |
| `competition` | 19 | Hard competition facts from the Kaggle API: deadlines, limits, prizes. |
| `bx-skipped` | 10 | The 10 Biohub-X findings judged NOT worth ingesting, WITH the reason, so nobody re-reads them. |
| `rules` | 10 | Rules text and host rulings: external data, hand-labelling, disqualification (none stated). |
| `page` | 8 | The competition's own pages, frozen. |
| `parity` | 6 | Measured Colab-vs-Kaggle environment differences. |
| `tooling` | 3 | Agent setup, the Colab cost gate, ponytail hook audit. |
| `validation` | 2 | Why the prior campaign's offline instrument was structurally broken. |
| `ops` | 1 | Submission mechanics - chiefly the silent-zero trap. |

## Reading a fact

Every row carries `claim_type` (observation / claim / inference / decision / profile),
`confidence`, a `source_id`, and a **verbatim `quote`**. A fact with no quote is not a fact.

`status` is the one to watch:

- `active` — current.
- `superseded` — kept, never deleted, with `superseded_by` pointing at the replacement and the
  reason written into the key. **35 facts were superseded on 2026-09-11 alone**, eight of them
  mine from earlier the same evening. Read the reason before reusing anything adjacent.
- `retracted` — recorded as deliberately NOT ingested, with why.

## The corrections that matter most

If you read nothing else, read why these were superseded:

- `THE_DECOMPOSITION_plateau_already_has_divisions` — the plateau is edge 0.926 + divJ 0.231,
  not edge 0.945 + divJ 0. This broke the original north-star arithmetic.
- `exploit_is_DEAD_since_2026_07_23` — the 0.950-0.966 notebook badges are frozen pre-rescore
  ghosts. Their authors now sit at 0.881-0.939.
- `gates_are_already_correct_on_true_geometry` — widening division gates buys +0.002, not the
  0.02 I first claimed.
- `support_dir_was_the_PRE_PATCH_scorer` — a landmine that shipped in this repo for hours.

## Open contradictions — do not resolve silently

Two live disagreements are recorded rather than papered over:

1. **Node-count pruning.** A competitor measured a 44.7% node cut as score-neutral (the
   multiplier compensates). Our monolith measured pruning's optimum as *no pruning*. Likely
   reconciliation: whole tracks versus scattered nodes. Test locally.
2. **`estimated_number_of_nodes` on test.** One agent says test ships it, another says it does
   not. If it does not, node-count calibration is uncomputable at inference. Inspect a real test
   `.geff` before betting on it.

## Rules the base enforces

1. A measured value lives here once; prose cites it and never restates it.
2. Every fact carries a source and a verbatim quote.
3. Contradicted facts are superseded, never edited or deleted.
4. Rejections are recorded too, with the reason (see `bx-skipped`).
5. `tools/sync_source_rows.py export` after any change — `db/*.db` is gitignored and the
   tracked `db/source_rows.jsonl` snapshot is what actually survives.
