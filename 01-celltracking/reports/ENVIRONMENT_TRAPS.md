# Environment traps — read before writing a Kaggle kernel or touching the trainer

Each of these cost real time or real GPU on this project. They are not in the code
comments because they are properties of the *environment*, not of our code.

## Trainer

- **`train_epoch()` has different loss-weight defaults than the baseline path.**
  Its own signature defaults are `det_loss_weight=0.1 / det_neg_weight=0.1`, but
  `train()` — the path that produced the baseline — passes **`1e1` / `1e-2`**. Calling
  `train_epoch` directly without passing them trains a 100× wrong objective and looks
  perfectly healthy in the loss curve. Locked by `tests/test_m1_driver.py`, which reads
  `train()`'s call site via AST rather than hardcoding the numbers.
- **DataLoader position must be checkpointed to resume.** Model + optimizer state is not
  enough; without the sampler permutation you silently re-train on a different data
  stream. `scripts/m1/m1_driver.py:ResumableSampler` derives the permutation from
  `(seed, epoch)` and aborts on a hash mismatch. A control-to-control run measured a
  divergence floor of exactly 0.0, so there is no GPU nondeterminism to blame.

## Kaggle image

- **polars ships with a compiled backend that does not load.** `import polars` succeeds
  while `polars._plr.PySeries` is undefined, so any import-only check passes and the
  first real call dies with `NameError: PyDataFrame`. Use a *functional* probe
  (construct a DataFrame) and force-reinstall on failure.
- **Dependency bootstrap must install by spec via `--find-links`, unconditionally.**
  Installing the raw wheel set replaces numpy/scipy and breaks the ABI; installing
  only-if-missing leaves the broken polars in place. Copy v122's proven `PACKAGE_SPECS`
  block rather than writing a new bootstrap.
- **No `zarr`, no `tracksdata`** in the image.

## Support pack

- **The pack names the package `biohub_tracking`; our vendored checkout calls it
  `tracking_cellmot`.** Accept either at import.
- **The pack predates metric patch `075fc5f`.** It is inference-only. It may produce
  coords/edges, but it must never score, validate or select. Authoritative scoring is
  local, on the pinned patched scorer.

## numpy

- **`np.savez_compressed` appends `.npz` unless the name already ends in it.** A
  `foo.npz.tmp` temp path is written as `foo.npz.tmp.npz`, so the subsequent
  `os.replace` fails — *after* the GPU work is done. Keep atomic temp names ending in
  `.npz`.

## Process

- **Do not preflight a script, modify it, then scale the modified version.** The
  `savez` bug above burned a full GPU session exactly this way.
- **Module-global config leaks across replay helpers.** `coupled_replay_shared`
  inherited `TTA = "d4"` while globbing 4-view files; the run reported "ok" throughout
  and was only caught by cross-checking node counts against the cache. Cross-check a
  cheap invariant (node/edge counts) against a known-good artifact after any replay.
- **Score only over the intersection of crops present in every arm.** An arm with 199
  crops compared against an arm with 1 fabricates a promotion delta.

## Kaggle submission and kernel defects (discovered 2026-07-31)

**7. THIS COMPETITION ONLY ACCEPTS SUBMISSIONS FROM NOTEBOOKS.** A direct CSV upload uploads the
whole file successfully and then fails on `CreateSubmission`:

```
400 {"error":{"code":400,"message":"Submission not allowed:  This competition only accepts
Submissions from Notebooks.","status":"FAILED_PRECONDITION"}}
```

The CLI truncates this to a bare `400 Client Error`, so it looks like a transport failure. The
working form requires a COMPLETED kernel:

```powershell
kaggle competitions submit -c biohub-cell-tracking-during-development `
  -k <owner>/<kernel-slug> -v <version> -f submission.csv -m "<message>"
```

Consequence for planning: **every candidate must exist as a completed Kaggle kernel before it can
score.** A locally-produced CSV can never be submitted, however well audited.

**8. `kaggle kernels push` reads notebooks with the locale codec.** Dies on non-ASCII content with
`'charmap' codec can't decode byte 0x9d`. Set `PYTHONUTF8=1` on every Kaggle CLI call.

**9. A kernel created during an SSL-error window is permanently broken.** A push whose response was
lost to `SSLEOFError` still created the kernel; that kernel then failed three consecutive runs with
`/kaggle/input` entirely unmounted while the API reported its datasets correctly attached. An
identical-metadata probe mounted fine, and the same notebook under a FRESH SLUG worked first time.
Cost: three wasted GPU sessions. Rule: if `/kaggle/input` is empty, do not debug the notebook --
re-create the kernel under a new slug.

**10. Never parse kernel status by substring.** Matching `"ERROR"` in CLI output makes any transient
`SSLError` look like a failed kernel. Use the typed
`get_kernel_session_status(...).status.name`.

**11. `kaggle kernels output` pulls the entire working directory** (168 files including weights) and
times out. Fetch `submission.csv` by URL via `list_kernel_session_output`.

**12. Kaggle slugifies the TITLE, not the id.** `"Biohub P0A Clean 913 Repro"` becomes
`biohub-p0a-clean-913-repro`, silently diverging from whatever `id` is in `kernel-metadata.json`.
Always read the slug back after a push.

**13. Public scoring is slow.** Submissions can stay `PENDING` for hours. Do not poll; submit a
coherent batch and check back later.
