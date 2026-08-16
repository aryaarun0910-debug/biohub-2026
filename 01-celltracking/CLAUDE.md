# Claude operating contract

Read `research/00-system/handoff.md`, `research/00-system/system-design.md`, and
`research/README.md`. `research/00-system/handoff.md` is the live entry point; the current
direction in full is `research/01-research-direction/directional-updates.md`. This file
(`CLAUDE.md`) is the only load-bearing doc left at the repo root.

## Mission

Climb toward top-3 (see `research/01-research-direction/scientific-mission.md`): convert the
current finding into a scored candidate, or kill it with the cheapest valid experiment. Do not
reopen a closed lever (`research/06-knowledge-system/failed-experiments.md`) without a new
mechanism and a stated falsification test.

## Execution rules

1. Preserve `.claude/settings.json` and `.gitignore` unless the host explicitly authorises
   editing them.
2. Never use the four visible placeholder movies for model or threshold selection.
3. Use the official patched scorer and the pooled objective; always report both embryo
   directions separately.
4. Tests enforce software contracts only. Scientific promotion decisions belong in an
   experiment result, not in a unit test.
5. Start with smoke, then representative pilot, then full evaluation. A passing smoke is
   not scientific evidence.
6. Do not create a new Markdown plan for each cycle. Update `research/00-system/handoff.md` and
   the relevant `research/` files; append one compact row to
   `research/06-knowledge-system/experimental-records.md`; store raw output in `_evidence/`
   or `/temp` (outside Git). RAG artifacts (embeddings, indexes) go in `.claude/rag/`
   (gitignored). Every `research/**.md` needs conforming frontmatter and a `system.yaml`
   entry — `scripts/core/validate_research_tree.py` enforces it.
7. Stage explicit paths. Never use `git add -A` or `git add -u`.
8. No GPU launch or submission is automatic. Return the measured pilot report first unless
   the host explicitly authorises the next stage.

## Active code surface

- `src/biotrack/`: immutable scorer/graph core and deployed wrapper.
- `scripts/d1/d1_postprocess.py`: scorer-exact M/C/T/L/D partition.
- `scripts/d1/d1f_probe.py`: representation-versus-head diagnosis.
- `scripts/core/kaggle_factory.py`: reproducible Kaggle build/push tooling (never auto-submits).
- `scripts/core/score_oof.py`: score exported `*.geff` vs `data/train`.
- `scripts/core/claims_table.py`: generates `research/06-knowledge-system/claims-table.md` from
  `inventory/*.json`; `--check` fails on drift.
- `scripts/core/validate_research_tree.py`: research-machine integrity (manifest + frontmatter).
- `scripts/win_bet/`: learned-ranker / Zebrahub workstream.
- `scripts/kaggle_edits/`, `scripts/kaggle_specs/`: active kernel patches and specs.
- Full instrument map: `research/03-experimentation/instrumentation.md`.

Everything else is support, immutable evidence (`_evidence/`), or history. If a file is not
named by `research/00-system/` or the `research/` machine, do not treat it as an instruction.
