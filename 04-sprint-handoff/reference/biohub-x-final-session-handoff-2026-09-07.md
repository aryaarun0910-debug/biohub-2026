# Biohub-X handoff, 2026-09-07 (X1 parity, E10 rule v2, E10 smoke package)

Branch `codex/research-rx03-focus3d`. Start from `3242d0a`; tree clean.
Commits this session: `4b8d6a8` (code, declarations, registries), `3242d0a` (clean package record, R-0064).
Repository `c:\Users\aryaa\Documents\Biohub-X`. Read AGENTS.md, SYSTEM.md, DECISIONS.md D-0049, registry R-0063 (Arya's review) and R-0064 (this session) before editing.

## What was done, in Arya's order (R-0063)

1. **X1 CPU parity, complete.** The support pack's `predict_unet_transformer.py` and `train_unet_transformer.py` were acquired by digest (RL-0218 to RL-0222, `research/cache/RL-supportpack/`) and transcribed into `biohubx.inference.reference_chain.track_reference_edges`: integer feature indexing on the U-Net feature maps, sinusoidal positional features (eight per axis over the strided shape, window-relative time), softmax over sources per target, one parent above the pack's threshold 0.5, decoded by the unchanged baseline decoder. A parity guard refuses a checkpoint whose positional width is not 32; unit tests hold the formula and the indexing to the transcription. Run on the four local test movies: N0 rebuilt equal to X0's frozen digest on every movie, CSV validated and round-tripped, no label read. Numbers in the EXTERNAL-REFERENCE-01 entry's `x1_local_cpu` block. No score computed; engineering only.
2. **E10 collapse rule v2, measured.** `heatmap.collapse_verdict(rule)` with rules `saturation` (historical, kept so the kill reproduces) and `v2` (non-finite or constant map, zero gradient, failed reload, usable peaks below the annotated count, ties at the matched-budget cut, coverage below A0 at that budget; background saturation reported as a diagnostic). The declared 200-step fit check re-run under v2 on the declared window: D1 and D2 both survive (learned recall 8/11 against A0 truncated 7/11 at 79 and 82 peaks, no ties, background saturated fraction 0.88 and 0.84 as diagnostics). Reports `artifacts/e10-probe-D{1,2}-v2.json`; registry block `stage1_v2`. The 2026-09-06 saturation-rule kill stays registered as that rule's kill.
3. **E10 GPU smoke, packaged and gated, NOT pushed, NOT authorised.** `biohubx package heatmap-smoke` stages `entry.run_heatmap_smoke` (dispatched from `run_fold` on experiment `E10`) behind the E07 package's bootstrap, wheelhouse identity and guards; pre-push stage tuple `prepush.HEATMAP_SMOKE_STAGES`. The packet is the one Arya proposed: one T4, `44b6_5f15d135` and `44b6_808952d6` first three frames, seed 0, D1 and D2, 20 steps per arm, runtime ceiling 900 s, two attempts at most, 30 GPU-minutes, no paid spend. Built clean from `4b8d6a8`, package manifest digest `canonical_text_sha256:sha256/v1:7c864d5112983a7fcb163f5f088b0f7756bad629d33505ce65f8e7d79fc7f2bf`, kernel `aryaarun07/biohub-x-e10-smoke-01`, local gate passed (20 stages, entry exercised in isolation on one movie, two frames, two steps). Declared in `configs/e10-learned-detector.yaml` `compute_packet.smoke_packet` and the E10 entry's `smoke_package` block. It is a proposal. If Arya authorises it, the push goes through `package transport --push` under an envelope; the permission classifier has blocked pushes from this harness before (memory note), so record the command for Arya.
4. **E09-ABSENCE-01 declared, not implemented, not run.** `configs/e09-absence-counterfactual.yaml` and its registry entry: the annotated parent removed from one target's offered set is candidate-set absence, never a birth; correct matches with the parent present and correct abstentions with it removed are measured apart, ranking apart from acceptance, on E09-DECODE-01's inner split only; no training, no threshold, no held-out read. Instrument name `biohubx evaluate correspondence-absence`, to be written.
5. **Fixture.** `tests/conftest.py` no longer copies `artifacts/cache` (10.7 GB) into memory; cache files are tracked by size and mtime. Full pytest is now minutes, not tens of minutes.

## Open for Arya

- The E10 smoke push and run (item 3) need an explicit authorisation naming the kernel and the packet; nothing was sent.
- The workstation manifest (D-0043) disables Colab and forbids competition data in consumer Colab while later plans mention Colab: needs an explicit reconciliation before any enablement or transfer. Not touched.
- DeepCenter's declared repair-veto role (R-0063) is not assessed; it stays a blocked, disabled arm.
- E09-ABSENCE-01 instrument: implement only under the declaration as written.

## Standing rules that bit this session

- Registry scalars carrying `: ` must be quoted or block scalars; a record script must parse before it writes (one status line broke and was re-quoted).
- `grep -c $'\r'` in this Git Bash reports every line as CR; check endings with python bytes or `git ls-files --eol`.
- Never run pytest while a background biohubx run is writing artifacts (session fixture deletes new files).
- Run `gate_and_commit.sh` with `BIOHUB_DATA_ROOT` UNSET: a CLI test expects the refusal without a root, and `artifacts verify` passes without it. Export the variable on individual biohubx runs only.
- Pass `BIOHUB_DATA_ROOT="C:/Users/aryaa/.cache/kagglehub/competitions/biohub-cell-tracking-during-development"`; use `./.venv/Scripts/biohubx.exe`; gate commits inside scripts; no `&&` across heredocs.

## Commands

```
BIOHUB_DATA_ROOT=... ./.venv/Scripts/biohubx.exe infer reference-chain --arm X1 --frozen-from artifacts/external-reference-01/X0/reference-chain-manifest.json
BIOHUB_DATA_ROOT=... ./.venv/Scripts/biohubx.exe train heatmap-probe --arm D1 --dataset 44b6_5f15d135 --steps 200 --rule v2 --out artifacts/e10-probe-D1-v2.json
BIOHUB_DATA_ROOT=... ./.venv/Scripts/biohubx.exe package heatmap-smoke --owner aryaarun07 --movies 44b6_5f15d135,44b6_808952d6 --smoke-id E10-SMOKE-01 --arms D1,D2 --frames 3 --steps 20 --seed 0 --runtime-ceiling 900
```
