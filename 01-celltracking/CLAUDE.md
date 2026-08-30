# Claude operating contract

Read `AGENTS.md` first — it is short, and it is the operating contract for every agent.
Then `research/00-system/handoff.md`, `research/00-system/system-design.md`, and
`research/README.md`. `research/00-system/handoff.md` is the live entry point; the current
direction in full is `research/01-research-direction/directional-updates.md`.

Only two docs are load-bearing at the repo root: this file and `AGENTS.md`. Both must stay
at the root — Claude Code discovers the PROJECT contract there, while `.claude/` holds the
user-global `CLAUDE.md` and harness machinery (`settings.json`, agents, skills). Moving this
file into `.claude/` would silently stop it loading as project instructions.

**`research/00-system/registry/` is the only source of truth for numbers.** Cite ids
(`FACT-0001`), do not restate values. Measured 2026-08-25: the superseded score 0.915
appeared 244 times across 36 files while the live score appeared 31 times across 7 — an
agent reading a random doc was 8x more likely to hit a dead number than a live one.
`scripts/core/validate_registry.py` enforces this and must pass before you finish.

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
7a. Claim a `LEVER-####` by opening a packet before working a lever. Two packets holding one
   lever is a hard failure (`validate_registry.py` R6) — that is the anti-duplication lock.
   A lever may only be closed by evidence **about itself**, and never on `UNVERIFIED` facts.
8. No GPU launch or submission is automatic. Return the measured pilot report first unless
   the host explicitly authorises the next stage.

## Active code surface

- `notebooks/kaggle_<name>/biohub-<name>.ipynb`: **the deployed artifact and the one that must
  be audited.** Built by `kaggle_factory` from `scripts/kaggle_specs/*.json`. The CHAMPION of record is
  `notebooks/kaggle_p35_dcveto_on_931/` (`FACT-0412`); `notebooks/kaggle_p3_harmonic/` is the older
  harmonic platform and is **not** the champion. Corrected 2026-08-30 — they differ in two ACTIVE
  post-processing flags (`GAP2_RECOVERY`, `ADAPTIVE_SHORT_TRACK_RESCUE`), so an agent sent to audit
  `p3_harmonic` reads the wrong chain configuration. Verify against the notebook the claim is about,
  and say which one. Any claim about deployed behaviour is verified in a built notebook, at
  `file:line`, and nowhere else.
- `src/biotrack/`: immutable scorer/graph core, plus a **partial** wrapper mirror — it is NOT the
  deployed program. Measured 2026-08-22: `DEEPCENTER` 0x in `wrapper.py` vs 73x in the built
  notebook; `VOLUME_GUARD` 4x vs 0x. Neither is a superset of the other, so auditing the division
  path here finds no filter stage and wrongly concludes there is none.
- `scripts/d1/d1_postprocess.py`: scorer-exact M/C/T/L/D partition.
- `scripts/d1/d1f_probe.py`: representation-versus-head diagnosis.
- `scripts/core/kaggle_factory.py`: reproducible Kaggle build/push tooling (never auto-submits).
- `scripts/core/score_oof.py`: score exported `*.geff` vs `data/train`.
- `scripts/core/claims_table.py`: generates `research/06-knowledge-system/claims-table.md` from
  `inventory/*.json`; `--check` fails on drift.
- `scripts/core/validate_research_tree.py`: research-machine integrity (manifest + frontmatter).
- `scripts/core/validate_registry.py`: registry integrity — id/xref hygiene, provenance rules,
  the superseded-value guard on `record_kind: state` docs, and the one-lever-one-owner lock.
- `research/00-system/registry/`: facts, experiments, levers, packets. The source of truth.
- `scripts/win_bet/`: learned-ranker / Zebrahub workstream.
- `scripts/kaggle_edits/`, `scripts/kaggle_specs/`: active kernel patches and specs.
- Full instrument map: `research/03-experimentation/instrumentation.md`.

Everything else is support, immutable evidence (`_evidence/`), or history. If a file is not
named by `research/00-system/` or the `research/` machine, do not treat it as an instruction.
