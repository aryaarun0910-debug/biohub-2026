# Claude operating contract

Read `HANDOFF.md` and `SYSTEM_DESIGN.md`. They are the only live instructions.

## Mission

Convert the current detector-calibration finding into a scored candidate, or kill it with
the cheapest valid experiment. Do not reopen archived branches without a new mechanism and
a stated falsification test.

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
6. Do not create a new Markdown plan for each cycle. Update `HANDOFF.md`, append one compact
   row to `reports/EXPERIMENT_LEDGER.md`, and store raw output outside Git.
7. Stage explicit paths. Never use `git add -A` or `git add -u`.
8. No GPU launch or submission is automatic. Return the measured pilot report first unless
   the host explicitly authorises the next stage.

## Active code surface

- `src/biotrack/`: immutable scorer/graph core and deployed wrapper.
- `scripts/d1_postprocess.py`: scorer-exact M/C/T/L/D partition.
- `scripts/d1f_probe.py`: representation-versus-head diagnosis.
- `scripts/build_d1_factorial_manifests.py`: checkpoint x family manifests.
- `scripts/assemble_p3_d1_smoke_spec.py`: current kernel assembly.
- `scripts/kaggle_factory.py`: reproducible Kaggle build/push tooling.
- `scripts/kaggle_edits/`, `scripts/kaggle_specs/`: active kernel patches and specs.

Everything else is support, immutable evidence, or history. If a file is not named by
`HANDOFF.md` or `SYSTEM_DESIGN.md`, do not treat it as an instruction.
