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

## Silent wrong-arm scoring (discovered 2026-07-31)

**14. `phaseb_node_budget.py` accepted `--arm` and ignored it.** `prune_one()` unpacked `arm` and
never read it again; the graph path was hardcoded to the arm-A cache. Running `--arm D` therefore
**scored arm A while writing `"arm": "D"` into the output JSON** — a wrong result labelled as the
right one, with no error and no warning. Lane 2 caught it only by cross-checking `keep_frac=1.0`
per-crop counters against the cached `coupled_cache/scores/D__<split>__<crop>.json` anchors.

Fixed by `arm_graph_path()`, which resolves arm A to `e0c_cache/graphs/` and B/Bp/C/D to
`coupled_cache/arms/<arm>/`, and **raises** on a missing arm cache instead of falling back.

Generalisation worth internalising: **an argument that is accepted but unused is more dangerous
than one that is rejected.** Before trusting any per-arm/per-fold/per-seed number, verify the
selector actually changes the bytes read — cheapest check is that two settings produce different
node counts. The parity habit that caught this (compare a no-op setting against a known-good
cached anchor, per crop) is the general defence and costs seconds.

**15. `h4_ssl_gate_replay.py` had never run, and its aggregator was unpenalised.** Two defects,
both found 2026-07-31 when it was executed for the first time:

- `main()` called `cached_crops()` with no argument, but it is defined `cached_crops(split: int)`
  and yields crop *stems*, not `(split, crop)` pairs — so the script raised `TypeError` before
  doing any work. It had been recorded in the queue as "built, smoked, not run"; in fact it
  **could not run**. Treat "smoked" as unverified unless a result artifact exists.
- `agg()` computed the count multiplier as `abs(N_pred_arm − N_pred_baseline) / N_pred_baseline`,
  which is **identically zero** for an edges-only replay, so every arm was scored with **no count
  penalty at all** and its `adj_edge_jaccard` was not comparable to the published 0.7595 / 0.6490
  anchors. The canonical multiplier is **signed** and defined against **`N_est`**
  (`metrics.py:440`), not absolute and not against our own baseline.

Both fixed. **The general check both defects fail: a replay's baseline arm must reproduce the
published anchor before any delta from it is quoted.** That single assertion catches this whole
class and takes seconds.

## Kaggle input mount depth (discovered 2026-08-01)

**16. Datasets do not always mount at `/kaggle/input/<slug>/`.** The fold-1 LOEO kernel died at
t = 628 s on `weights glob '/kaggle/input/*/edge_predictor_best_split_1.pth' matched []` while the
*same run* had already resolved the support pack at
`/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1/` — i.e. an
`owner`-qualified, one-level-deeper layout. A depth-1 glob therefore misses a correctly attached
dataset.

**This is not trap 9.** `/kaggle/input` was fully populated (128 train `.zarr` crops mounted, the
competition dir present); only the depth assumption was wrong. Distinguish the two by printing the
mount tree before raising — `scripts/kaggle_edits/loeo_retarget.py::_loeo_find` now tries the
declared pattern and then a **bounded** `*/` ladder (depths 1–4) on the basename. Do **not** use
`glob(..., recursive=True)` over `/kaggle/input`: it descends the 79 GB competition zarr tree.

Cost: ~10 GPU-minutes, cheap only because the assertion fired before `predict`. The general rule:
**any hard assertion on an input path must print the actual mount tree in its failure message**,
otherwise "wrong path" and "broken kernel" look identical.

**16. Kaggle datasets mount OWNER-QUALIFIED, one level deeper than competition data.** A fold-1
kernel died at t = 628 s on `weights glob '/kaggle/input/*/…' matched []` — **after** the same run
had successfully mounted 128 zarrs and resolved the support pack. This is **not trap 9**: the mount
was healthy. Attached *datasets* appear at `/kaggle/input/datasets/<owner>/<slug>/…` while
competition data sits at `/kaggle/input/<slug>/…`, so a single-star glob silently matches nothing.

Fixed in `scripts/kaggle_edits/loeo_retarget.py::_loeo_find` with a **bounded depth ladder**.
**Never use `recursive=True` here** — it walks the 79 GB zarr tree. The failure path now prints the
mount tree so the two failure modes stay distinguishable. Cost ~10 GPU-minutes.

Distinguishing rule: `/kaggle/input` **empty** ⇒ trap 9, re-create under a fresh slug.
`/kaggle/input` **populated but your glob matched nothing** ⇒ trap 16, widen the depth ladder.

**17. `linefit_smooth_output_graph` mutates node dicts IN PLACE.** A shallow copy therefore does not
isolate the arm from the baseline: the second graph is silently **double-smoothed**, while a parity
check on the *first* call still passes, so the defect hides behind a green check. Found 2026-08-01
during component-retention replay.

Fix applied in the affected replay scripts: build the **arm first and the baseline second**, so the
parity assertion itself exercises the mutation path. General rule for this codebase: **any replay
that constructs two graphs from one source must deep-copy, and must order its arms so parity covers
the mutating call** — a parity check that only ever runs on a pristine first invocation proves
nothing about the second.

**Ordering also matters for cost, not just correctness.** The same lane initially sorted crops
size-descending and spent 2 CPU-hours covering **9.9%** of its target signal, because 91.2% of the
target edges live in *small* `6bba` crops. Re-ordering by *target-count ÷ predicted runtime* reached
**80.1%** coverage in a further **9 wall-minutes**. Before any long corpus sweep, sort by expected
signal per second — not by crop size, and not by directory order.

**18. TWO different "published anchors" are in circulation, and they differ by one edge.**
Found 2026-08-01 by Lane 1's per-crop parity check.

| source | pooled | 6bba |
|---|---|---|
| **`inventory/pooled_objective_parity.json` — CANONICAL, locked by `tests/test_pooled_objective.py`** | **0.6653932886896151** | **0.6489523829566957** |
| `fn_attribution_ceilings.json`, `h4_ssl_gate_replay_pooled.json`, `phaseb_h0c_replay`, `phaseb_oracle_d0prime` | 0.6654043056779476 | 0.6489651953216343 |

**Cause:** those replay scripts rebuild the edge list as a Python **set** — `{(a,b) for a,b in edges}`.
The cache contains **zero duplicate edges** (4,916,122 checked), so nothing is dropped; what changes
is **insertion order**. The metric's out-degree>2 cap and its merge-dedup both keep the **lowest
EDGE_ID**, so a different edge survives. Corpus edge weight 151,615 vs 151,614; corpus edge FN
**27,706 vs 27,705**.

**Magnitude 1.10e-05 pooled = 0.55% of a +0.002 gate.** No verdict on record flips. The damage is
epistemic: a lane can truthfully say "reproduces the published anchor exactly" while matching the
*other* anchor, so "parity passed" stops being a single well-defined claim.

**Rules from this:**
- Parity-check against the **canonical** triple, and state which anchor you used.
- Better: check **per-crop counts** against the independent cached artifacts
  `artifacts/kaggle/coupled_cache/scores/A__*.json`. That is what caught this; a single scalar
  would not have.
- **Never rebuild an edge list with a bare set comprehension.** Use a list plus a seen-set so
  insertion order is preserved — the scorer is order-sensitive through its tie-breaks even when the
  edge *set* is identical.

**19. A zarr chunk IS a frame, so an exact-duplicate census is nearly free.** Each timepoint is
exactly one chunk file and blosc/zstd is deterministic, so **raw identity ⟺ compressed-byte
identity**. A size prefilter over 19,900 files plus sha256 on the ~1,800 size collisions settles
duplicate detection across all 199 crops in **110 seconds**, against ~160 GB of decoding for the
naive approach. Check file-level identity before writing a decoder.

**20. GT node-id encoding DIFFERS BY FAMILY.** 44b6 encodes `(t + offset) * 1e9`; 6bba encodes
`(t + 1) * 1e6`. Any code that derives a timepoint from a node id arithmetically will be silently
wrong on one family — and, because 6bba is 85% of edge mass, "wrong on one family" is usually wrong
on the answer. **Always build the id→t map from the geff itself; never infer it.**

**21. polars schema inference silently drops conditionally-populated columns.** If a column is
absent from the first inference window it is dropped without warning, so a field that only appears
in some rows vanishes from the frame. Declare the schema explicitly when rows are heterogeneous.

**22. `list_kernel_session_output` is PAGINATED and caps a page at 500 entries.**
`scripts/kaggle_factory.py::session_outputs()` made a single call and returned `resp.files`. For any
kernel that also writes a zarr/geff tree the page fills with chunk files and the **named outputs are
absent** — which is indistinguishable from "the kernel produced nothing". Verified on
`biohub-loeo-f0-strict`: the unpaginated call returned **500 files / 15 distinct geff crops**; following
`next_page_token` returns **2,377 files / 71 crops** (the true count) and surfaces `submission.csv`.

Fixed 2026-08-01 — `session_outputs()` now follows `next_page_token` with `page_size=500` and a
200-page hard stop. **A truncated listing is worse than an error, because it looks like data.**

**23. A glob is not a manifest.** A WS-A kernel silently processed **66 of 71** fold-0 crops (8.2% of
the fold missing) because its crop list came from a corrupt-filtered glob rather than the manifest.
Nothing failed; the run simply covered less than it claimed. Caught only by comparing counts.
**Rule: the manifest is authoritative — every declared item must resolve or the job dies naming the
missing ones.** Never let a filter silently shrink a declared work set.

**24. `node_id` is NOT unique across movies in the submission format.** Counting node degree with a
bare `node_id` key instead of `(dataset, node_id)` reports **15,471 "hub" nodes on the clean P0-B
artifact**, whose true maximum out-degree is 2. Found 2026-08-01 in a freshly-written diagnostic
*and* in the audit tool meant to check it — the same defect in both, because both were written from
the same wrong mental model. **Always key graph structure on `(dataset, node_id)`.**

Related hardening adopted at the same time: **a diagnostic cell must never be able to fail the
kernel.** Wrap appended provenance/diagnostic cells in `try/except` that writes a `*_FAILED.txt`
rather than raising — this competition can only submit from a **COMPLETE** kernel, so a diagnostic
that raises converts a good run into an unsubmittable one.
