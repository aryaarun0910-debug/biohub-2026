# The Biohub knowledge base — how it stores things, and why

One file: `db/biohub_base.db`, SQLite. ~1.8 MB, 376 active facts.

## Why SQLite and not a graph DB, a server, or more markdown

The store answers questions like *"what has already been measured about divisions?"* and
*"which levers are closed and by how much?"*. Those are **filter-and-rank** questions over a few
thousand rows, which is exactly SQL's shape and exactly what a document store is bad at.

It is not Memgraph or any graph database. The graph-ish part — notes linking to notes — lives in
markdown under `knowledge/`, where `[[wikilinks]]` are edges. That split is deliberate:

| | Lives in | Because |
|---|---|---|
| Measurements, scores, counts | SQLite | They are queried, ranked, filtered, and compared |
| Conclusions, decisions, orientation | Markdown | They are read by humans and by recall, and they link |

A server was rejected because the data is 1.8 MB and single-writer. A graph DB was rejected
because nothing here traverses more than one hop.

## The core technique: provenance and validity are separate axes

This is the part worth copying. It came from the monolith, which paid for it.

**Provenance** grades how strongly a number was *derived*:
`VERIFIED` → `MEASURED` → `EXTERNAL` → `UNVERIFIED`.

**Validity** asks whether the run it came from was a *legitimate measurement of what it claims*:
`VALID` / `SUSPECT` / `INVALID`.

They are orthogonal, and conflating them is how false facts survive. The monolith's `EXP-0019`
scored the 6bba embryo using weights trained on 6bba — a leave-one-embryo-out leak. Those facts
were `MEASURED` and *deserved* that grade: they were correctly computed. They were also
worthless, because the run was invalid. Seven downstream facts and eight work packets inherited
the leak before anyone noticed.

In this schema that becomes `claim_type` + `confidence` + `status`, and every row carries a
`source_id` and a verbatim `quote`. A fact with no retrievable source is not a fact.

## The four row classes

Every row is one of these, and knowing which is what makes the rebuild rules obvious:

1. **Derived-from-Kaggle** — `lb_snapshot`, `forum_topic`, `public_kernel`, plus `fact` rows on
   topics `competition` and `page`. Rebuilt by `tools/harvest_kaggle.sh`. Disposable.
2. **Derived-from-repos** — `repo`, `repo_commit`, `repo_file`, and `fact` rows on
   `prior-campaign`. Rebuilt by `ingest_repos.py` / `ingest_registries.py`. Disposable.
3. **Curated** — the judged ingests (`ingest_bx_findings.py`). Reproducible *only because the
   judgement is committed as code*: the keep/skip decision for all 35 Biohub-X findings lives in
   a dict in that file, with a reason per line, including the rejects.
4. **Source** — `decision` rows and our own analysis facts. **No ingester regenerates these.**
   Mirrored to the tracked `db/source_rows.jsonl` by `tools/sync_source_rows.py`.

That last split is why `db/*.db` can be gitignored safely: the binary churns and will not merge,
but nothing irreplaceable rides on it.

## Tables

| Table | Holds |
|---|---|
| `source` | Every artifact a claim rests on: url, sha256, method, fetched_at |
| `fact` | topic, key, value, claim_type, confidence, source_id, **quote**, status, superseded_by |
| `experiment` | Levers and runs: hypothesis, config, cv/lb scores, outcome (`win`/`neutral`/`dead-end`) |
| `decision` | Strategic calls with rationale and open/decided status |
| `lb_snapshot` | The full public leaderboard, dated — re-harvest to track movement |
| `forum_topic` | All 102 threads, with our triage and takeaway |
| `public_kernel` | 240 notebooks with score, runtime, and a `family` label |
| `artifact` | 243 archived items incl. 20 checkpoints, addressed by git blob sha |
| `repo` / `repo_commit` / `repo_file` | Repo history; `reusable=1` flags files worth porting |

## The rules it enforces

1. **A measured value lives here once.** Prose cites it; prose never restates it. The monolith
   measured the cost of breaking this: a superseded score appeared 244 times across 36 files
   while the live one appeared 31 times across 7.
2. **Every fact carries a source and a verbatim quote.** No quote, no fact.
3. **Never delete a contradicted fact** — set `status='superseded'` and point `superseded_by` at
   the replacement.
4. **Skips are recorded too.** When 10 of 35 findings were judged not worth ingesting, the
   reasons went in as `status='retracted'` rows, so nobody re-reads them to rediscover that.

## Using it

    python3 tools/ask.py status | gap | topics | kernels | expts | facts <topic> | todo
    sqlite3 db/biohub_base.db

    python3 tools/sync_source_rows.py export   # after adding rows by hand
    python3 tools/sync_source_rows.py check    # fails if the export drifted
