# Rank 1 — the plan

Written 2026-09-11. Supersedes the day-by-day shape in
`repos/Biohub-Sprint-2026/05-SPRINT-PLAN.md`, which assumed the Mac landed on the 17th.
Everything else in that handoff pack still stands and should be read first.

## What changed since the handoff was written (2026-09-07)

| | Handoff assumed | Actually true on 2026-09-11 |
|---|---|---|
| Mac arrives | 17 Sep | **14 Sep** — three extra days, 15-day runway |
| Leader | 0.962 | **0.970** |
| Our 0.925 was worth | rank 207 / 2,693 | **rank 1,125 / 3,372** |

The field gained ~0.022 in eighteen days. That was not eighteen days of better science —
it was public "metric hack" notebooks propagating. **0.925 is now a below-median score.**

## The one number that defines the task

    score = adjusted_edge_jaccard + 0.1 x division_jaccard

    public plateau (~400 teams)  0.9465
    rank 1                       0.970
    gap                          0.0235
    gap / 0.1                    0.235   <- implied division_jaccard at rank 1

A competitor publicly reported measuring their own `division_jaccard` at **0.22**
(discussion 739516). A one-to-one linker scores **0.000** on divisions (discussion 733877).

**Read that together: the plateau is ~400 teams with working edges and no divisions, and the
entire podium gap is the division term.** That is an honest lever. It needs no metric gaming,
it survives a rules review, and it is defensible on the winners' recorded call.

This is a hypothesis with one cheap falsifier, and it is the first thing to test on day 1:
score any prediction locally with the organizers' own scorer and read `edge_jaccard` and
`division_jaccard` *separately*. If our edge term is already ~0.945 and divisions are ~0, the
plan is correct. If the edge term is well below 0.945, fix edges first.

## The strategic fork, decided

Do **not** ship a metric hack. Concretely: no hub node at `t = -1000`, and no node-count
manipulation. Reasons, in order of weight:

1. The host has already patched an exploit mid-competition (`aa65e90`, 2026-07-17) and
   rescored the whole leaderboard. Precedent says they will do it again, and the private
   rerun is the moment to do it.
2. Winners must deliver training code, inference code and a reproducible methodology under
   MIT, and may defend it on a recorded call. A 7-minute notebook with no model does not
   survive that.
3. Rules §3.148 allows disqualification for "unfair playing practices or abuses".
4. Arya's own prior campaign already wrote this exclusion into `levers.yaml`: node-count
   manipulation is "excluded from submissions entirely".

The legitimate neighbour of the exploit is still available and should be used: the detection
threshold is a real calibrated parameter, and `N_pred` relative to `n_total` is a real
quantity to tune. Tune it honestly on held-out folds — but note the monolith already measured
that **node-budget pruning's optimum was no pruning at all**, so expect little here.

## Runway: 14 -> 29 September

Treat **28 September** as the deadline. Assume one day is lost to something.

### Before the Mac arrives (11-13 Sep, on the HP laptop)

**Revised after the Colab account came to light: these are not filler days.** Colab and Kaggle
are both reachable from a browser, so the critical path — *be submittable* — can start now. The
Mac adds a zero-cost iteration box on the 14th; it is not the starting gun. Waiting for it costs
3 of 18 days.

0. **Verify the Colab burn rates on your own billing page.** My figures are from a secondary
   source and Google changes them. Everything downstream assumes ~65-85 h of A100; confirm it
   before planning around that number.
0b. **Open one Colab session and record what it gives you** — `torch.__version__`, the GPU you
   are assigned, and whether an A100 is actually available on your tier. Kill the session the
   moment you have the answer; idle A100 time bills at ~15 units/hour.
0c. **Kaggle API token** (`kaggle.json`) so the download and any notebook push can be scripted
   rather than clicked.

These need no GPU and remove a whole day from the critical path.

1. **Kaggle identity verification.** `requiresIdentityVerification: true` — it gates
   submission and prizes and can take days. Do this today.
2. **Confirm entry status** — the rules must be accepted before 22 Sep, and the team-merge
   deadline is the same day.
3. **Start the ~98 GB data download onto the external drive now.** It is network- and
   disk-bound, not CPU-bound, so the dead fan does not matter. Day 1 on the Mac should not
   be spent downloading.
4. Read `repos/Biohub-Sprint-2026/01-START-HERE.md` through `05`.

### Day 1 — Sun 14 Sep — make the machine real

- `uv`, Python 3.12, Xcode CLT. Clone all three repos.
- **The MPS gate.** `Conv3d`, `MaxPool3d`, `ConvTranspose3d`, forward *and* backward, on
  `mps` vs `cpu` — numerical agreement and wall-clock. The detector is built entirely from
  those three ops. A silent CPU fallback shows up in the timings, not in an error.
  Decide the compute plan today, not on day nine.
- Vendor and pin the organizers' scorer at `075fc5f5a52d11077f9dc2b074644618f26939e2`.
  It is already sitting in `reference/royerlab-baseline/`.

### Day 2 — Mon 15 Sep — be submittable

Port from Biohub-X, do not retype: the contracts, the metric adapter, the submission writer
and validator, the digest system, the classical difference-of-Gaussians detector, the Kaggle
packaging path. Run end to end, submit.
**Gate: an accepted score on the leaderboard tonight.** If not, all science stops until there is.

### Days 3-4 — 16-17 Sep — reach the public honest line

We are chasing 0.947, which public honest notebooks already achieve and we never did. Absorb
what the 0.947 tier does on **edges** — checking any borrowed pipeline for nodes at
implausible timepoints before believing a single number it produces.
**Gate: edge term >= 0.94 on held-out folds.**

### Days 5-9 — 18-22 Sep — divisions, the main event

This is where rank 1 is won or lost, and it gets half the runway.

- Divisions are ~1 link in 853. This is rare-event detection, not a tracking tweak.
- The metric's division rules are *local*: a fork one timepoint early or late still counts,
  but branches must be topologically distinct and unmerged. Read `metrics.md` closely and
  build directly against those rules.
- External data is rules-legal if public and free. The community **18.5 GB synthetic set with
  165k labelled divisions** (discussion 732103) exists precisely for this scarcity. Verify its
  licence and accessibility first. Zebrahub and the Cell Tracking Challenge are further candidates.
- Measure ranking and acceptance separately — Biohub-X's matcher ranked well and accepted
  badly and that was hidden for days by reporting one number.
- **Note 22 Sep is the entry and team-merge deadline.** Nothing should need action, but confirm.

**Gate: `division_jaccard` > 0.10 held out by end of day 9.** Below that, the plan has failed
and the remaining days go to the edge term instead.

### Days 10-12 — 23-25 Sep — whichever term the numbers say is weaker

Change one thing at a time. The monolith's best score came from a bundle of three changes,
and one knob it believed in turned out to have exactly zero effect.

### Day 13 — 26 Sep — freeze
Code freeze. Training runs and packaging only. A bug found today is fixed only if it changes
the output.

### Days 14-16 — 27-29 Sep — final submissions
Two selections. Take one high-public-LB run and one low-variance run — the public LB is only
29% of the test set, ~400 teams are tied inside 0.002, ties break on earliest submission, and
a shakeup is near-certain. Never let a day pass in the last week with the best local model
unsubmitted.

## Standing rules

- Five submissions a day. Keep two in hand on each of the last two days.
- Never select anything on the four visible test movies — they are byte-identical copies of
  training movies and are replaced at rerun.
- Never trust an offline proxy's *direction*. The monolith measured DeepCenter at +0.0009
  offline against +0.003 on the LB, and `icom` at +0.002 offline against −0.014 actual.
- One registry is the only home for numbers: `db/biohub_base.db`. Prose cites, never restates.
- Declare the falsifier before the run.

---

# The machine — what actually maximises it

Researched 2026-09-11 against primary sources. Full claims with URLs and verbatim quotes are
in the database (`python3 tools/ask.py facts hardware`), including ~17 explicitly unverified
inferences kept labelled as such.

## The finding that changes day 1

`F.conv3d` on Apple Silicon ran at **~7% of conv2d throughput** up to PyTorch 2.12. **2.14.0,
released 2026-09-02 — nine days ago — moved it to hand-written Metal kernels and it now reaches
97% of matmul peak.** That is 3.63 → 59.13 TFLOP/s, about **16x**, on the operator that
dominates 3D U-Net training.

Two conditions attach, and both are easy to miss:

- **`bias=True` on every `nn.Conv3d`.** A bias-free conv3d fell to the slow SIMD path; the
  no-bias fix (#192229) also landed in 2.14 but the measured gap was 10x.
- **macOS 26+.** The fast path needs Metal Performance Primitives. Do not upgrade to macOS 27
  mid-competition.

`04-ARCHITECTURE.md` tells you to test Conv3d / MaxPool3d / ConvTranspose3d on day 1. That test
is still right, but its premise — "historically the weakest part of PyTorch's Metal backend" —
expired nine days ago. **The gate now is the torch version, not the backend.** And the failure
mode is slowness, not an error, so a wrong pin costs 16x silently.

## The architecture change this forces

**`ConvTranspose3d` is still unusable on MPS.** It rejects fp16/bf16 outright, which also
crashes `torch.amp.autocast` for any model containing it, and the fix PR is still open. This is
exactly why nnU-Net does not run on Apple Silicon.

**Build the U-Net decoder from `Upsample` + `Conv3d`.** It costs nothing in model quality and
removes the only hard blocker. Decide this before writing the detector, not after.

## The uncomfortable part

The Mac is **not** faster than the machine you submit from. Estimated ~33–35 TFLOPS dense FP16
(low confidence) against a single T4's 65. **Kaggle's T4 x2 is roughly 4x this machine for raw
training throughput**, and its bandwidth (300 GB/s) is at parity with the M5 Pro's 307 GB/s.

So "abuse its computational strength" is the wrong frame. What the Mac actually buys:

- **48 GB of unified memory**, no host/device copies, no gradient checkpointing where a 16 GB
  T4 needs it. nnU-Net's presets stop at a 40 GB "XL" tier; this machine can plan past it.
- **No queue and no 12-hour wall.** Kaggle gives you two concurrent sessions and five
  submissions a day. The Mac gives unlimited ablations.

Use it for data preparation, the classical baseline, ablations, evaluation and packaging — and
run the long training jobs on Kaggle's T4 x2 in parallel. That is two machines working at once,
which is the real throughput win.

One caveat that cuts the other way: **plan the patch size for the T4, not the Mac.** A
configuration tuned to 48 GB will not run at inference on a 16 GB GPU — nnU-Net warns about
exactly this.

## Train on Mac → infer on Kaggle: the handoff

1. `sd = {k: v.detach().to("cpu", torch.float32).contiguous() for k, v in model.state_dict().items()}`
   The `.cpu()` is load-bearing: `state_dict()` does **not** strip the device, MPS tensors carry
   an `"mps"` location tag, and deserialising calls `obj.mps()`, which cannot succeed on Kaggle.
   The `.float()` avoids T4/P100 having no bf16 tensor-core path.
2. Save as **safetensors** — no pickle, no device tag, no torch-version gate.
3. Ship a **frozen parity fixture** with the weights: fixed-seed input plus the fp32 CPU
   reference output.
4. In the notebook, assert with `torch.testing.assert_close(..., check_device=False,
   check_stride=True)` and set **`torch.backends.cudnn.allow_tf32 = False`**. That flag defaults
   to **True**, so Conv3d silently runs at 10 mantissa bits on an L4 (cc 8.9) but full fp32 on a
   T4 (7.5) — same notebook, different numerics depending on what you're assigned.
   `set_float32_matmul_precision` does **not** control convolutions; it is not the lever.
5. Order matters: `load_state_dict` **first**, then `.to(memory_format=torch.channels_last_3d)`.

Version skew runs the breaking direction: the Mac gets 2.14, Kaggle's live image is the 2.10
line, and 2.11 is already merged to their `main`. Pin with `docker_image_pinning_type:
"original"` so the image cannot move under you mid-competition.

## Kaggle operational traps worth more than any tuning

- **Attach every data source before Save Version.** The save run cannot attach new sources or
  different versions, and this is a leading cause of rerun failure.
- Kaggle's **Dependency Manager** now does offline pip installs for internet-disabled
  competitions — the manual wheels-in-a-dataset trick is superseded.
- **P100 was dropped from Kaggle's GPU CI on 2026-09-05; T4 x2 is the sole test bed**, which
  matches the handoff's T4 x2 / cc 7.5.
- Ship weights as a Kaggle **Model**, not a Dataset — Models version, Datasets don't — and read
  them via `kagglehub`, not hardcoded `/kaggle/input` paths.

## Setup, in order of payoff

1. `torch >= 2.14.0`, then run the conv3d benchmark from pytorch#192213 as an acceptance test.
2. `bias=True` on every Conv3d.
3. Decoder = `Upsample` + `Conv3d`.
4. Stay on macOS 26.x.
5. CPU fp32 safetensors + parity fixture + `allow_tf32 = False`.
6. Measure `torch.mps.recommended_max_memory()` — do not assume 36 GB. Cap with
   `set_per_process_memory_fraction`; the allocator's default high watermark is 1.7, which will
   happily swap the machine.
7. DataLoader: `pin_memory=False` (meaningless on unified memory), `persistent_workers=True`,
   `multiprocessing_context="forkserver"`, and benchmark `num_workers=0` honestly — macOS
   defaults to `spawn`, so workers cost seconds each.
8. Zarr v3 with sharding on the internal SSD, outside iCloud, excluded from Spotlight. Avoid
   HDF5 — its process-wide lock and fork hazard are exactly wrong here.
9. A/B `torch.compile` before trusting it. Compiled *training* on MPS was up to 4.35x **slower**
   than eager until August 2026; if it regresses try `TORCHINDUCTOR_LAYOUT_OPTIMIZATION=0`.
10. Turn `PYTORCH_ENABLE_MPS_FALLBACK` **off** during parity testing so divergences throw
    instead of hiding.

**Skip:** MLX (its one real advantage was conv3d, and 2.14 erased it — and it costs you MONAI,
TorchIO and every forkable notebook); High Power Mode (~0.4% on GPU work — it only raises the
CPU ceiling and the fans to 7500 rpm); the Neural Engine (cannot train); a bigger charger (input
is hard-capped at ~97 W); clamshell operation (heat exits through the keyboard deck — lid open).

**No float64 on MPS at all** — a hard error, not a downcast. Voxel spacings and affines are
commonly float64; cast them at the boundary.

---

# Compute: three machines, one framework

Revised 2026-09-11 after the Colab budget came to light.

## One framework. PyTorch. Not MLX.

The instinct — "use the 20-core GPU properly" — is right. The conclusion doesn't follow, because
**PyTorch already reaches the M5's Neural Accelerators.** Apple says so directly: *"Open source
tools like MLX, llama.cpp, and PyTorch already leverage neural accelerators under the hood."*
There is no Metal performance sitting behind an MLX-shaped door.

Against adopting it, specifically:

- **MLX cannot produce a submission.** Kernels-only, and Kaggle has no Metal. MLX could only ever
  be a local trainer whose weights you convert afterwards — a conversion step plus a numerical
  agreement check, bought for nothing.
- **Its one real advantage evaporated nine days ago.** MLX's conv3d edge existed because
  PyTorch's was broken. torch 2.14 took conv3d to 97% of matmul peak.
- **It is not reliably faster.** An open MLX issue has it **10–20% slower** than PyTorch on M5
  bf16 GEMM, flagged as hurting training.
- **There is no official MLX→PyTorch exporter.** Every conversion tool runs the other way.
- **You lose the ecosystem** — MONAI, TorchIO, nnU-Net, timm 3D encoders, and every forkable
  public notebook. In an 18-day runway that is the whole argument by itself.

So: **PyTorch, device chosen at runtime, in exactly one function.** `mps` at home, `cuda` on
Colab and Kaggle, `cpu` for tests. This is what `04-ARCHITECTURE.md` already decided, and 2.14
made it more right, not less.

## Where GPUs actually accelerate this problem

Be precise about it, because the answer is "one stage out of five":

| Stage | Character | GPU helps? |
|---|---|---|
| Data loading, Zarr decompression, patch extraction | I/O + CPU | No — 18 cores matter |
| **Detector (3D U-Net) training** | **conv3d-bound** | **Yes. This is the only real GPU job** |
| Detector inference | GPU, but capped at Kaggle's 12 h | Yes, constrained |
| Linking / association | bipartite matching, ILP — combinatorial | Barely. CPU-bound |
| Division detection | rare-event modelling | Data-starved, not compute-starved |
| Metric evaluation | graph ops | No — but you run it constantly |

**Compute is not the binding constraint. 18 days and the division term are.** A100 hours cannot
fix a model that scores 0.000 on divisions. Budget compute to the detector; spend thinking on
divisions.

## The three machines

**Colab A100 — the training machine.** ~1000 units at ~12–15/hr is **roughly 65–85 hours** of
A100: ~312 TFLOPS bf16, 40 GB, 1555 GB/s. That is about **10x the Mac's FLOPS and 5x its
bandwidth**. Every long detector run belongs here.

Two disciplines, non-negotiable:
- **An idle session burns the budget.** 15 units/hour means one A100 left running overnight
  costs ~120 units — 12% of everything you have. Terminate sessions explicitly, every time.
- **Colab disconnects.** Checkpoint to Drive every N steps or you will pay twice for the same
  epoch.

**Kaggle T4 x2 — free parallel compute, and the only submission-truthful numerics.** Two
concurrent 12-hour sessions that cost no credits. Use it for runs that must match what the
submission will do, and to keep training moving while Colab credits are conserved.

**Mac M5 Pro — the zero-marginal-cost machine, and the one that makes Colab cheap.** It is the
*slowest* of the three for training. Its value is that nothing it does costs money or queues:
data preparation, the classical DoG baseline, metric development, small ablations, evaluation,
visual QA, packaging, and building the submission notebook.

Its highest-leverage job: **preprocess the ~98 GB into a compact, training-ready patch store on
the Mac, and upload only that.** Raw data into Colab means either a long download each session
or a Drive quota fight, and A100 minutes spent on I/O are A100 minutes burned at 15 units an
hour. Cheap local preprocessing converts directly into more effective A100 time.

## The rules question, resolved

The prior campaign left Colab "unresolved" and its manifest forbade processing competition data
in consumer Colab. **That was self-imposed policy, not a rules requirement.** The dataset is
licensed **CC0: Public Domain**, and §47 is the permissive variant: *"You may access and use the
Competition Data for any purpose, whether commercial or non-commercial."* §51 restricts giving
the data to **people** who have not accepted the rules — it says nothing about which of your own
machines you compute on.

Keep the notebook and Drive folder private and Colab is fine. Recorded as a decision so it is
not re-litigated.

## On "build from scratch"

Right instinct, with one correction: **rebuild is not retype.** In 18 days, re-deriving the
contracts, the metric adapter, the submission writer and the digest system would consume the
runway and reproduce work that is already correct and tested.

Port, from Biohub-X: the contracts, the fail-closed metric adapter, the submission writer and
validator, the digest system, the classical DoG detector, the Kaggle packaging path.

Leave behind: the bounded streaming cache, the window census and the resume machinery — all of
it existed because 16 GB could not hold a training window. Also every Windows accommodation, and
the blind-isolation contract.

What genuinely *is* new: the model (`Upsample`+`Conv3d` decoder, no `ConvTranspose3d`), the
division head, and a three-device compute layer instead of a one-machine one.
