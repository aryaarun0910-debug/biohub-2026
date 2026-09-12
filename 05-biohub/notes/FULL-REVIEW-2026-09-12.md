Biohub review at `46713c5bae637400a8a5f7afd16848ff131a927b`

The decision is **hold paid compute**. The sparse oracle is useful as a linker diagnostic,
but neither it nor EXP-18 establishes performance at realistic density. The proposed order—
validate density, prove packaging, then train—is sensible after correcting the benchmark.
This review changes knowledge and planning; it does not promote or retrain a model.

Scope: active `src/biohub` stages, experiment/evaluation/cache tools, database maintenance,
submission validation, compute guards, pinned-in-tree scorer, existing tests, strategy notes,
relevant frozen competition evidence, and primary research. Ignored historical campaign trees
and every downloaded third-party notebook were not independently re-audited. Existing notebook
censuses are historical evidence, not a new exhaustive census or a live leaderboard verification.

The pre-existing SQLite drift and untracked EXP-18 script were backed up under
`~/Work/TestBench/BiohubTestBench/review-2026-09-12/` before export. EXP-18 was left unchanged.
Research sources and short verbatim excerpts are in `evidence/review-2026-09-12/research.json`;
all new conclusions are queryable under the DB topic `review-2026-09-12`.

**Corrections to the supplied assessment**

1. **The node-count adjustment can be lost.** The actual multiplier is
   `max(0, 1.1 - 0.1 * N_pred/N_total)` on a nonnegative edge Jaccard. At estimated full density
   it is exactly one; the sparse annotation bonus disappears. The limit at vanishing count is
   bounded, not unlimited. The narrow range reported by EXP-18 is a property of its chosen
   subset and inflation range, not every realistic detector. The official aggregate weights
   samples by edge denominators, so a pooled annotation ratio is illustrative, not an exact
   aggregate correction. See `review_metric_density` and the metadata census in `checks.json`.

2. **EXP-18 is a confounded stress test.** `inflate()` samples independent nodes and offsets;
   it does not propagate a consistent offset along a track. It permits coordinates outside
   the volume. `hash(p.stem)` changes between ordinary Python processes. Its first sorted
   subset contains only one embryo. The shipped classifier has seen both embryos. No frozen
   original output, deterministic seed or full run manifest supports reproducing the precise
   reported collapse. This is enough to reopen downstream calibration, but not to attribute
   the exact loss to genuine dense cells. Annotation-only oracle scores remain conditional
   measurements; they are not mathematically established upper bounds either.

3. **The notebook decomposition is local CV.** The archived source explicitly identifies an
   eight-sample validation table; it is not Kaggle's hidden-score decomposition. Its edge term
   is adjusted edge Jaccard, not raw edge Jaccard. Similarity of its composite to the displayed
   leaderboard score does not establish the hidden components. Existing post-rescore evidence
   supports retaining the clean notebook family as a comparator; it does not support a universal
   edge ceiling or a conclusion about which term every competitor is bottlenecked on.
   The badge/rescore census was not independently refreshed here. See `review_notebook_scope`.

4. **Zebrahub is not perfect truth.** The provider identifies Ultrack as the tracking tool.
   Without independent curation, the CSV is a pseudo-label reference. Whole-embryo counts do
   not determine local density. Randomly subsampling points destroys the very neighbour
   competition the benchmark should preserve. Crop contiguous physical volumes and time windows;
   retain all centres within each crop. Use several embryos and spatial regions, audit boundary
   divisions and temporal sampling, and score reference agreement separately from manually
   verified biological accuracy. The archived host reply permits use and states no test overlap;
   it does not certify label quality. See `review_zebrahub_label_provenance`.

5. **PAC-MAP is not inverse-distance Gaussian weighting.** The authors' implementation passes
   nearest-neighbour distances as amplitudes and has a corresponding proximity-aware decoder.
   Sparse missing neighbours corrupt that target. Treat it as an ablation on suitably dense
   labels, not a ready-made sparse-label fix. The published successor is identified in
   `review_pacmap_target_and_decoder`. [Authors' implementation](https://github.com/DeVosLab/PAC-MAP).

6. **Hardware estimates are not measured runtime.** A100 peak arithmetic is supported by the
   vendor; the Mac/T4/A100 ranking and units-to-hours conversion for this model are unmeasured.
   Colab availability varies. EXP-11 uses assumed effective FLOPS and an assumed hidden-set size;
   EXP-12 adds that estimate to local disk timing and extrapolates ideal worker scaling. Neither
   demonstrates Kaggle headroom. Prepare CUDA work now, then measure the actual allocated stack.
   [Colab FAQ](https://research.google.com/colaboratory/faq.html),
   [NVIDIA specification](https://www.nvidia.com/en-us/data-center/a100/).

**Code findings, ordered by what blocks the next run**

| Priority | Location | Finding and required gate |
|---|---|---|
| P0 | `src/biohub/detect.py:38`, `refine.py:37`, `tools/run_pipeline.py:24` | Detection is unimplemented, refinement is a no-op without a model, and the runner always loads oracle nodes. There is no current image-to-submission entry point. Require an offline raw-volume run before paid training. |
| P1 | `detect.py:24` | NMS uses native voxel scale while claiming a downsampled grid; it uses a rectangular maximum filter rather than spherical physical suppression, and equality admits flat plateaus. Explicitly define grid scale, convert output coordinates, and test close nuclei, plateaus and borders. |
| P1 | `submit.py:22`, `tools/validate_submission.py:16` | Writer validation misses nonfinite coordinates and negative endpoints (NumPy negative indexing aliases real nodes); the CSV validator accepts header-only input and truncates fractional values. Coverage is optional; expected dataset shapes are hardcoded. Reject malformed/empty output and require runtime-discovered coverage before writing. |
| P1 | `tools/_eval_common.py:32`, `score_submission.py:44` | Shared evaluation substitutes zero node recall for edgeless predictions rather than enforcing the repository's failed-run policy. Standalone scoring skips datasets without GT and does not enforce complete expected coverage; missing count metadata can make adjusted scores NaN and omit rows. Fail closed on incomplete evaluation and require all score terms finite. |
| P1 | `resolve.py:133`, `run_pipeline.py:28`, `exp7_fork_sweep_loeo.py:65` | Current defaults load coefficients trained on both embryos. The general runner and old sweep do not enforce fold ancestry. EXP-16 supplies separate paths, but a missing model silently falls back to probability one. Require checkpoint existence, content hashes and train/evaluation disjointness. |
| P1 | `resolve.py:144` | The first Hungarian pass has no birth/death option and consumes targets before forks compete. Equal frame counts can eliminate a true fork when an unrelated parent should disappear. The second pass only sees unclaimed daughters; rejected forks do not reopen assignment. Compare explicit unmatched costs/joint fork alternatives on identical dense detections before declaring proposal solved. |
| P1 | `tools/build_train_cache.py:103` | Cache reuse checks only existence. A partial-frame or old-target cache silently survives a full rerun; no data/code/fold digest is checked. If sampled frames contain no labelled peaks, threshold infinity marks every finite voxel as negative. Fail or retain unknown regions; bind caches to full configuration and provenance. |
| P1 | `tools/kb.py:54`, `kb.py:85`, `tools/ingest_registries.py:49` | `kb.record` labels every inserted item VALID and measured; `kb.ask` omits validity filtering/display. Registry conflict updates can restore active status over a curated supersession. Keep claim validity separate, and preserve adjudications across refresh. This review uses an explicit transaction to retain those distinctions. |
| P2 | `tools/train_fork_model.py:32`, `exp15_fork_classifier_probe.py:61` | Training with `fork_accept_p=-1` still applies the divergence gate, so it does not harvest all proposals. The original probe now uses learned-default filtering and thus is not a clean rerun of its earlier candidate population. Freeze harvesting independently of acceptance; exact-edge labels also differ from the scorer's tolerant division definition. |
| P2 | `tools/exp17_lost_edge_recoverability.py:91` | Lost-edge distances include cases without alternatives; kept distances do not. Slicing both lists to the shorter length does not restore pairing. Its final margin/comparison is invalid when an unpaired loss appears before a paired one. Record paired tuples. |
| P2 | `repair.py:86`, `contracts.py:46`, `refine.py:25` | Fork preservation counts forks rather than verifying their identities; the stage ledger reports net size differences rather than actual replacements. Refinement adds dlogit to an unspecified score domain, prunes by whole-graph quantile, and drops edges. Define ordering and score units before wiring it after linking. |
| P2 | `tools/colab_guard.sh:48`, `tools/fetch_geff.py:20` | Parity receipts attest successful command exit, not numerical parity, and hash only the entry script. Data download depends on a machine-local path outside the repository. Require a portable manifest and machine-readable remote parity report; receipt success alone cannot release compute. |

These are review findings, not applied production fixes. The diagnostic script preserves
small runnable reproductions; failing readiness checks are intentional findings, not a claim
that the current test suite fails. The dependency-pinned campaign package remains a gate.

**Independent research that changes the options**

Linajea is the strongest additional sparse-supervision precedent found here. It trains only
inside small masks around labelled centres and predicts backward motion so daughters can share
one parent. That directly contradicts the categorical prohibition on sparse detection training.
It leaves background unconstrained, so filtering remains necessary. Test its masking idea in
the simple detector first; adopting its complete distributed stack is unnecessary.
[Primary paper](https://www.janelia.org/sites/default/files/Malin-Mayor%202022.pdf).

Trackastra's registry has a `ctc` checkpoint supporting 3D, while its common `general_2d`
example does not. It expects segmented objects; centroid-to-feature conversion needs checking.
Use it as a matched-input comparator after that adapter is validated, not as a promised gain.
[Checkpoint registry](https://github.com/weigertlab/trackastra/blob/main/trackastra/model/pretrained.json).

Ultrack's multiple segmentation hypotheses provide a reason to test irreversible early
assignments. They do not prove we need a wholesale solver replacement. Its own generated
tracks cannot independently establish its superiority or our superiority over it.
[Authors' implementation](https://github.com/royerlab/ultrack),
[ECCV algorithm paper](https://arxiv.org/abs/2308.04526).

**Validation sequence before compute**

1. Fix the identified validation, scale and provenance blockers. Pin a clean install and scorer
   dependency commit; reproduce graph → CSV → reload → official score on synthetic and real
   training fixtures. Include empty output, missing datasets, nonfinite values, bad endpoints,
   close peaks, plateau ties, forks, boundaries and serialization rounding.
2. Implement a minimal raw-intensity detector path with robust normalization and physical NMS.
   Run the complete image pipeline on training data in both embryo groups. This proves plumbing,
   not competitive accuracy. Package offline weights/dependencies and all attached input sources.
3. Preregister density validation: contiguous Zebrahub crops, full native trajectories, fixed
   physical extents/time cadence, multiple embryos, boundary handling, immutable hashes and
   independent train/evaluation membership. Select density strata from geometry, not score.
   Compare the current linker, one-to-one baseline and a joint-fork/unmatched-cost alternative
   on identical detections. Also compare against a frozen public baseline on Kaggle training
   crops where checkpoint ancestry permits; otherwise label the comparison in-sample.
4. Report two different questions: agreement with dense pseudo-labels, and official sparse-label
   scoring with a fixed lineage-preserving annotation mask. Include imperfect detections with
   misses, duplicates, jitter and coherent extra tracks. Density-only success does not validate
   detector noise. Never interpret fully dense reference scores as a Kaggle leaderboard forecast.
5. Once these checks pass, run an offline Kaggle smoke notebook and confirm its artifact and
   acceptance. Full GPU training remains deferred. Benchmark actual device/precision/I/O and
   verify checkpoint save/resume before a bounded A100 pilot. Require a confirmed budget and
   stop rule, remote parity evidence, and teardown verification.
6. Train a masked Gaussian baseline first; then one PAC-MAP target+decoder ablation with the
   same split and budget. Add refinement only if it improves held-out composite performance
   while preserving recall. Recalibrate motion, fork acceptance and repair on predicted dense
   detections. No parameter is “settled” solely from oracle runs.

The intended outcome is an auditable compute decision, not another optimistic proxy score.

**Validation actually completed**

The existing unittest suite passes in the pre-existing scoring environment. The focused
readiness diagnostics expose the failures listed in `review_readiness_probes`; the synthetic
CSV roundtrip succeeds. The separate edgeless scoring attempt exceeded its bounded timeout,
so it is a failed diagnostic, not a zero score. The vendored scorer pytest suite could not run
because this environment lacks pytest. Package versions and scorer hashes are frozen in the
review evidence; these record this machine, not a validated clean-install dependency lock.

Reproduce from the repository root with the scoring environment (or an equivalent environment
matching `evidence/review-2026-09-12/environment.json`):

```bash
POLARS_MAX_THREADS=2 OMP_NUM_THREADS=2 python evidence/review-2026-09-12/review_checks.py
python -m unittest discover -s tests -v
python tools/sync_source_rows.py check
```

The diagnostic command emits JSON with explicit `passed` fields; its exit code means the
report was generated, not that readiness passed. It is not a compute-approval preflight.
The reviewed EXP-18 bytes are frozen alongside it for reproducibility, while the user's
original untracked script remains untouched. No production algorithm fixes were applied,
and raw-image inference, Kaggle acceptance and accelerator parity remain unverified.
