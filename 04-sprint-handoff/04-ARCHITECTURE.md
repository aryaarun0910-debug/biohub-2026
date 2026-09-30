# 04 WHAT TO BUILD, AND WHAT TO PORT

## One framework, three devices

Write **PyTorch**. Select the device at runtime.

| Where | Device | Used for |
|---|---|---|
| MacBook | `mps` | training and all local experiments |
| Kaggle notebook | `cuda` | the submission run, and any training that MPS cannot do |
| Anywhere | `cpu` | tests, and the fallback that must always work |

Do not write MLX. It is Apple-only, and the submission is a Kaggle notebook where
Apple frameworks do not exist. Choosing MLX means either two model implementations or
no submission. If a specific MLX kernel later proves dramatically faster for one
bounded job, treat that as a measured optimisation with a conversion and agreement
check, not as the architecture.

Device selection belongs in exactly one function. Everything else takes a `device`
argument. Biohub-X got this roughly right and then leaked `cuda` assumptions into its
packaging layer; do not repeat that.

## The first thing to measure, before anything is designed

The detector is a 3D U-Net: `Conv3d`, `MaxPool3d`, `ConvTranspose3d`. Historically
those have been the weakest part of PyTorch's Metal backend.

On day one, run each of the three operations forward and backward on `mps`, compare
outputs against `cpu` for numerical agreement, and time both. Then decide:

- **They work and are fast.** Train locally. This is the plan the sprint assumes.
- **They work but fall back to CPU silently.** You will see it in the timings, not in an error. Then the Mac's advantage is its 18 cores and 48 GB, which is still several times the old laptop, and GPU training moves to Kaggle.
- **They error.** Train on Kaggle, use the Mac for data preparation, evaluation and packaging, both of which were the real bottleneck on the old machine anyway.

Whichever it is, record the measurement and move on the same day. Do not discover
this in the second week.

## Port these from Biohub-X. Do not retype them.

These are hardware-independent, tested, and cost real time to get right.

- **The contracts.** Coordinates with an explicit voxel scale, the lineage graph types, the candidate instance types. Everything downstream depends on them and they encode the competition's conventions correctly.
- **The official metric adapter.** A fail-closed wrapper around the vendored organizer scorer, pinned by commit, that refuses to score an unscorable prediction rather than returning a misleading zero.
- **The submission writer and validator**, and its round-trip check. It produced an accepted submission. That is worth more than its size suggests.
- **The digest system.** Two content identities, raw bytes and canonicalised text, serialised as typed tokens so a bare hex string is refused. This is what makes "the checkpoint I evaluated is the checkpoint I shipped" checkable.
- **The registry pattern and the commit gate.** Numbers live in registries only; the gate runs lint, types, tests and artifact verification before a commit lands.
- **The classical detector.** Difference-of-Gaussians at two scales with strict peaks and suppression. It is the baseline every learned detector must beat at a matched budget, and it is already correct.
- **The Kaggle packaging path**, with its guards: the notebook carries its own source, the wheelhouse is verified by identity before install, inputs are checked against registered digests, and the whole thing is exercised locally before anything is pushed.

## Leave these behind

- **The bounded streaming cache, the window census and the resume machinery.** All of it exists because 16 GB could not hold a training window. With 48 GB of unified memory it is complexity with no purpose. If a dataset later does not fit, add the simplest thing that works then.
- **Every Windows accommodation.** The line-ending contract test, the `.venv/Scripts/*.exe` paths, the environment-variable trap where the test suite required a variable to be unset while the tools required it set.
- **The absolute-path provenance fields** in reports. They made the old artifacts non-portable. Record paths relative to a declared root.
- **The blind-isolation contract.** It was the right rule for its campaign and is now dissolved. Do not carry a test that forbids reading the repositories you are meant to learn from.

## Compute, in order of preference

1. **The Mac**, for everything that runs locally: data preparation, the classical baseline, evaluation, packaging, and training if MPS holds.
2. **Kaggle**, for the submission always, and for training if MPS does not hold. Two concurrent GPU sessions, T4 x2, roughly five submissions a day. Budget those five.
3. **Colab** is unresolved and should stay that way unless you need it. The old workstation manifest disabled it and forbade processing competition data in consumer Colab, while later plans assumed it. That conflict was never reconciled. If you want Colab, reconcile it as an explicit decision first, and check the competition's rules on where data may be processed.

## Storage

The competition data is about 98 GB. With a 1 TB internal drive and a 1 TB external,
put the data on the external and keep the repository, environments and artifacts
internal. Do not let derived caches grow unbounded on the internal drive; the last
campaign accumulated 10 GB of patch cache and 4 GB of fetched kernel outputs without
noticing.
