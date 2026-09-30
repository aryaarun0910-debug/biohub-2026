# Agent operating contract

Every agent — human, primary, or sub — reads this file before doing anything else. It is
short on purpose. The long-form direction lives in `research/00-system/handoff.md`.

`CLAUDE.md` remains the execution contract and outranks this file wherever they overlap.

---

## 1. The registry is the only source of truth for numbers

`research/00-system/registry/` holds four files. Nothing else is authoritative.

| file | what it holds | id |
|---|---|---|
| `facts.yaml` | every measured value, once | `FACT-####` |
| `experiments.yaml` | binds spec ↔ kernel ↔ submission ↔ evidence ↔ score | `EXP-####` |
| `levers.yaml` | every hypothesis and its status | `LEVER-####` |
| `packets/PKT-####.yaml` | a unit of work an agent owns | `PKT-####` |

**Cite ids, do not restate numbers.** Write "the deployed score (`FACT-0001`)", not "0.925".
This is not style. It was measured on 2026-08-25 that the superseded score `0.915` appeared
**244 times across 36 files** while the live score appeared 31 times across 7 — so an agent
opening a doc at random was **8× more likely to read a dead number than a live one**. Every
restatement is an independent chance to go stale. A citation cannot.

If you need a number that is not in `facts.yaml`, that is a finding: add it with its
provenance, or say explicitly that it is unknown. Never infer one from prose.

### Provenance is a claim about evidence, not confidence

| label | means |
|---|---|
| `VERIFIED` | re-derived this cycle at a named `file:line`, or by rerunning committed code |
| `MEASURED` | computed from a named artifact by code that is **committed** |
| `EXTERNAL` | a third party asserted it; we have **not** checked it |
| `UNVERIFIED` | in our own history, never re-derived, not currently reproducible |
| `SUPERSEDED` | replaced; `superseded_by` names the successor |

`VERIFIED` and `MEASURED` **require** an `instrument`. An oracle that is not committed as
code is not a result, it is an anecdote — learned when the recorded `3.44%` nearest-parent
anchor proved unreproducible and its script had never been committed (`FACT-0043`).

---

## 2. One lever, one owner — the anti-duplication rule

Two agents independently discovering the same thing is pure waste, and it has happened.
The lock is mechanical, not a convention you are asked to remember:

1. Pick a lever from `levers.yaml` with `status: open`.
2. Create `packets/PKT-####.yaml` from `PKT-TEMPLATE.yaml`, set `lever:` and
   `lock: claimed`, and put your agent name in `owner:`.
3. `validate_registry.py` **fails** if two packets hold the same lever (check R6). That is
   the guard. Run it before you start work, not after.

If the lever you want is already held, do not start a parallel investigation. Either extend
the existing packet or pick another lever.

**Never reopen a `killed` or `closed` lever** without (a) a new mechanism and (b) a stated
falsification test, both written into the packet. And note the inverse failure, which cost
this project weeks: the Trackastra pruning lane was closed by *association* with HOCT
evidence — a different model. **A lever may only be closed by evidence about itself.**

---

## 3. What a packet must declare before it runs

A packet is a dossier, not a to-do. It states, before any work begins:

- `claim` — what would be true if this succeeds
- `falsifier` — **the result that kills it.** Written first, so the outcome cannot be
  rationalised afterwards. Predictions before scores have a measured track record here;
  P9's stated band hit while the naive estimate was still optimistic.
- `inputs` — artifacts and facts it consumes, by path and id
- `budget` — GPU sessions and submission slots. Both are scarce and separately capped:
  **max 2 concurrent GPU batch sessions** (`FACT-0061`), ~5 submissions/day.
- `forbidden` — what it must not touch
- `outputs` — where evidence lands, and which facts it will add

---

## 4. Standing rules that have each cost us something

- **Verify a premise at `file:line` before building on it.** Five premises failed this way
  in one cycle, two of them our own computations. The handoff's own "one-line glob fix" was
  false (`FACT-0063`).
- **Calibrate any derived quantity against an independently known value.** Distance code
  must reproduce the GT displacement median (`FACT-0040`) before its output is trusted.
  Atlas coords are full-res `(z,y,x)` at `(1.625, 0.40625, 0.40625)` — **not** isotropic.
  Two of three distance analyses in one day started wrong.
- **An empty grep is not evidence.** ripgrep `-E` means `--encoding`; a malformed flag
  produced empty output that reached two reports as a verified negative. `_evidence/` and
  `artifacts/` are gitignored and skipped silently by default.
- **Check whether it is already on disk.** Three times in one cycle we concluded we lacked
  something that was sitting there.
- **Does the state cross a process, thread, or replica boundary?** Ask this at build time.
  Two failures of this exact class: `torch.autocast` not reaching DataParallel replica
  threads, and parent-process `builtins` not reaching the prediction subprocess
  (`FACT-0060`), which silently wasted a full GPU session.
- **A silent no-op is worse than a crash.** If an instrument can do nothing, make it say so
  loudly. Prefer a positive heartbeat whose *absence* is the alarm.
- **Report both embryo directions separately.** Pooling has hidden a full inversion — the
  lever ranking flips by fold, and essentially all headroom is in fold 1 (`FACT-0031`).

---

## 5. Environment and compute

- Python is `.venv\Scripts\python.exe`; set `PYTHONIOENCODING=utf-8` or Windows cp1252 will
  crash on the first non-ASCII character in any report.
- Kaggle CLI is **`python -m kaggle`** — the `.exe` is blocked by Windows App Control. The
  factory's generated `submitcmd` names `kaggle.exe`; substitute.
- **THIS COMPETITION IS NOTEBOOK-ONLY. A local CSV CANNOT BE SUBMITTED.** A submission must bind
  the kernel and version: `competitions submit -c <comp> -k <owner/slug> -v <version> -f
  submission.csv -m "..."`. Uploading the file directly returns HTTP 400 from `CreateSubmission`
  after a fully successful upload, which looks like a size or format problem and is not.
  **Read `kaggle_factory.py submitcmd` and use what it prints** — it states this in a comment and
  carries the receipt and artifact hashes. Measured 2026-08-31: two attempts lost to improvising
  past it. A failed `CreateSubmission` spends NO submission slot, but verify that against the
  submissions list before retrying rather than assuming it.
- Kaggle only. Colab is declined (host decision). Deadline **2026-09-29**.
- Hardware is deterministically **T4x2, CC 7.5**, so fp16 + `GradScaler` is correct and the
  DataParallel/autocast trap is guaranteed on every run: `torch.autocast` is thread-local
  and does not reach replica threads. See also `FACT-0060` — the same class, one boundary
  over.
- **Max 2 concurrent GPU batch sessions** (`FACT-0061`); ~5 submissions/day. Pipeline pushes.
- Kernel work goes through `scripts/core/kaggle_factory.py`:
  `build -> verify -> push -> status -> fetch -> audit -> submitcmd`.
- **Watch for slug divergence on push** — the factory reports it, and the spec must be
  updated or `status`/`fetch` break. Related trap: a spec copied without changing `out_dir`
  silently overwrites its sibling's `kernel-metadata.json` and repoints that kernel's id.

## 6. Before you finish

```
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\core\validate_registry.py
.\.venv\Scripts\python.exe scripts\core\validate_research_tree.py
.\.venv\Scripts\python.exe scripts\core\claims_table.py --check
```

Update the registry, set your packet's `lock: done`, append one compact row to
`research/06-knowledge-system/experimental-records.md`, and stage explicit paths — never
`git add -A`. No GPU launch or submission is ever automatic.
