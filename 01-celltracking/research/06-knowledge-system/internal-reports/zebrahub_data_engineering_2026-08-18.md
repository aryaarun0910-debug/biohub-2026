# Zebrahub data engineering — hands-on, 2026-08-18

Every number below was measured on this machine (CPU only) and the command that produced it is
named. Nothing here is a scored result; no GPU was launched, nothing was submitted, nothing was
uploaded, nothing was committed.

---

## Headline

**Track identity is RECOVERED for all 72 packaged `kkunizaw/biohub-zh001r` crops — exactly.**
1,357,051 of 1,357,051 crop nodes (100.000%) map to a ZSNS001 Ultrack node with a median residual
of 4e-5 µm (float32 round-off). The recovered supervision is **1,192,441 GT association edges**
and **34,179 division events** across 116,320 distinct tracks, delivered as an **8.7 MB** sidecar
that attaches to a kernel next to the existing 363 MB public dataset.

This **removes the limit** recorded on 2026-08-17 ("edge / association retrain: **NOT SUPPORTED**").
That limit was correct about the packaged bytes and wrong about what was reachable from them.

Second finding, independent: **`kkunizaw/biohub-zmnscrops` contains no labels of any kind.** It is
raw uint8 imaging windows from the Zebrahub *multi-view* embryos ZMNS001/ZMNS002 — a different
acquisition from the ZSNS single-objective embryos our tracks cover. It cannot supervise anything
on its own, and no public lineage table exists for those embryos. Its value is pretraining
substrate, not training data.

---

## 1. The unit error that blocked the first pass

The registration search was flat everywhere on the first attempt — every candidate timepoint
scored `frac<2µm` between 0.13 and 0.19, no peak anywhere across 193 timepoints. The cause was a
wrong assumption, not a wrong algorithm.

**`data/external/zebrahub/ZSNS001_tracks.csv` z/y/x are in MICRONS already, not level-0 voxel
indices.** Verified by fetching the published pyramid metadata
(`urllib` on `.../single-objective/ZSNS001.ome.zarr/.zattrs`):

| level | shape (T,C,Z,Y,X) | scale z,y,x (µm) |
|---|---|---|
| 0 | 791, 1, 448, 2174, 2423 | 1.24, 0.439, 0.439 |
| 1 | 791, 1, 224, 1087, 1212 | 2.48, 0.878, 0.878 |
| 2 | 791, 1, 112, 544, 606 | 4.96, 1.756, 1.756 |

Level-0 physical extent is 555 × 954 × 1063 µm. Measured CSV ranges are z 27.3–527.0,
y 41.7–927.2, x 7.9–1043.5 — all inside that box **as microns**. Applying the competition
`DEFAULT_SCALE = (1.625, 0.40625, 0.40625)` would place z at 812 µm, outside the volume.

Three corrections follow, and they matter beyond this task:

1. The Zebrahub level-0 voxel is **(1.24, 0.439, 0.439) µm**. The (1.625, 0.40625, 0.40625) tuple
   is the *competition zarr's* scale. These are different acquisitions and the tuples are not
   interchangeable. The brief's premise "coordinates in level-0 voxels" is incorrect.
2. Voxel scale is **per embryo**. ZSNS001 level-1 is (2.48, 0.878, 0.878) µm; the locally held
   `ZSNS003_L1.zarr` carries `scale_zyx = [0.62, 0.2195, 0.2195]`. The "2.6× z, 7.4× xy" resample
   factors in the brief are the ZSNS003 numbers; for ZSNS001 the correct factors are 1.53× z
   (an *upsample*) and 1.85× xy. Hard-coding either silently corrupts the other.
3. Consequently the "expected mapping is roughly known" premise was false in the direction that
   mattered — but the true mapping turned out to be *simpler* than expected (§2).

---

## 2. Identity recovery — the recipe, and it works exactly

Script: **`scripts/win_bet/h1r_zh001r_register.py`** (new).

### Model

```
global_um[z,y,x] = origin_um + 1.625 * crop_coord[z,y,x]
global_t         = t0 + local_t                        (local_t = 0..19)
```

Per-axis, **identity axis order, no flips, no rotation**. The crops are an axis-aligned window on
an isotropic 1.625 µm grid.

### Search

A **geometric-hash (Hough) vote over timepoint and origin jointly**. Take k = 16 seed nodes from
crop frame 0; for each candidate timepoint form every offset from every seed to every global node
at that timepoint; quantise offsets onto the 1.625 µm lattice; take the modal bin. At the true
timepoint all 16 seeds vote for one identical bin (16/16 on every crop); at a wrong timepoint the
votes scatter. Then a per-axis least-squares refit against nearest neighbours (3 frames, 6
iterations, shrinking match radius), then the acceptance gate.

**Two things the brief suggested that did not work, and why:**

- **Per-frame node counts as a first filter — too weak to use.** A 64³ crop holds ~900 nodes of
  ~27,000 per frame (measured: ZSNS001 per-timepoint counts run 18,977 at t=0 to 34,767 at t=790,
  median 27,008). Count agreement carries almost no information. Discarded.
- **FFT histogram cross-correlation (5 µm bins) as the coarse scan — works but is worse.** It
  costs 215 s/crop against 135 s for the vote, and its peak contrast is poor: on crop 0 the true
  timepoint scored 395 against 320 for the runner-up, whereas the vote scores 16/16 against
  scattered singletons. Kept in the scratchpad, not in the shipped script.

### Gate

Fraction of crop nodes within 2 µm of a global node, computed over **all 20 frames** — not only
the 3 frames used for fitting.

### Measured result — all 72 crops

Command:

```
.venv\Scripts\python.exe scripts\win_bet\h1r_zh001r_register.py \
  --root <dir-with-zh001r> --crops all --t-prior 158,316,474,632 \
  --out data\external\zebrahub\zh001r_identity.npz
```

**REGISTERED 72/72.** On every crop: votes 16/16, `frac<2µm = 1.0000`, fitted scale exactly
(1.625, 1.625, 1.625). Median residual across crops: min 2e-5, median 4e-5, max 1.2e-4 µm.
p90 residual 0.000 µm on all 72.

First three crops were solved by the **exhaustive scan over all 791 timepoints** (135–141 s each,
no prior); the remaining 69 used the four-candidate prior discovered from them (0.7–3.4 s each).
The prior is not load-bearing: `register_crop` falls back to the full 791-timepoint scan on any
crop that fails the gate, so a wrong prior cannot hide a real solution.

### Structure discovered from the exact fit

- **Voxel size is exactly 1.625 µm isotropic** — *identical* to the deployed detector input, not
  1.032× it. This **supersedes** the nucleus-size ruler's 1.677 µm estimate in
  `h1r_zh001r_audit.py` (and the earlier 1.762 µm). The ruler was biased upward ~3%, exactly as
  its own stated confound predicted (their nuclei are denser / later-stage). Crop extent is
  **104.0 µm**, matching the competition crop extent exactly.
- **Origins lie on a 78 µm lattice** (= 48 voxels). Every one of the 216 origin components across
  72 crops is an exact multiple of 78: values seen are 78, 156, 234, 312, 390, 468, 546, 624, 702,
  780, 858. A 48-voxel stride on 64-voxel crops = 16-voxel overlap.
- **t0 takes exactly four values, 18 crops each:** t0 = 158 (crops 0–17), 316 (18–35), 474 (36–53),
  632 (54–71). Spacing 158 timepoints. The packaging is 4 temporal blocks × 18 spatial windows.

### The recovered supervision (measured on `zh001r_identity.npz`)

| quantity | value |
|---|---|
| crop nodes total | 1,357,051 |
| assigned a `track_id` | 1,357,051 (**100.0000%**) |
| duplicate assignments (2 crop nodes → 1 global node), first 8 crops | **0** |
| distinct `track_id` | 116,320 |
| GT association edges (t → t+1, same track) | **1,192,441** |
| — per crop min / median / max | 10,985 / 16,505 / 24,618 |
| daughter-node appearances (parent present in previous frame) | 65,741 |
| **distinct division events** (unique parent × frame × crop) | **34,179** (474.7 per crop) |
| sidecar size on disk | **8,696,665 bytes** |

The division count is the striking one. On the deployment substrate we measured 613 division false
positives against 5 true positives across 199 crops (2026-08-18 record C). This substrate offers
34,179 *labelled* division events — four orders of magnitude more division supervision than the
competition training data exposes.

### Honest limits on this result

- The registration recovers identity **from a source we already hold**. It does not create new
  biology; it re-attaches Ultrack's ZSNS001 lineage to a third party's crops. The labels are
  therefore only as good as Ultrack's ZSNS001 tracks — which are *automated* tracking output, not
  human-curated ground truth. Training an edge model on them teaches it to imitate Ultrack.
- `zh001r` is **ZSNS001 only**, one embryo, four temporal blocks.
- The 20-frame crops give 19 transitions each; division events are concentrated in whatever
  developmental phase t = 158/316/474/632 correspond to.
- **Nothing here is scored.** No LOEO number moved. `bet-zebrahub-retrain` stays `proposed`.

---

## 3. `kkunizaw/biohub-zmnscrops` — characterised completely

Downloaded and unzipped to `data/external/zebrahub/packaged/` (3.41 GB transfer, ~5 min at
~11 MB/s, via `python -m kaggle datasets download kkunizaw/biohub-zmnscrops --unzip`).

### Structure (from `zipfile` member listing + numpy headers)

| file | bytes | members | keys |
|---|---|---|---|
| `zmns001_crops.npz` | 5,760,002,030 | 6 | `ts`, `w0`…`w4` |
| `zmns002_crops.npz` | 7,128,002,278 | 7 | `ts`, `w0`…`w5` |

Stored **uncompressed** (zip method 0) — the 3.41 GB download is the Kaggle-side zip of a 12.9 GB
uncompressed payload.

| array | shape | dtype | min | max | mean | nonzero frac |
|---|---|---|---|---|---|---|
| zmns001 `w0` | (60, 480, 200, 200) | uint8 | 0 | 255 | 14.689 | 0.3774 |
| zmns001 `w1` | (60, 480, 200, 200) | uint8 | 0 | 255 | 10.586 | 0.3284 |
| zmns001 `w2` | (60, 480, 200, 200) | uint8 | 0 | 255 | 11.584 | 0.3486 |
| zmns001 `w3` | (60, 480, 200, 200) | uint8 | 0 | 255 | 10.521 | 0.4204 |
| zmns001 `w4` | (60, 480, 200, 200) | uint8 | 0 | 255 | 6.559 | 0.1876 |
| zmns002 `w0` | (60, 495, 200, 200) | uint8 | 0 | 255 | 4.590 | 0.2229 |
| zmns002 `w1` | (60, 495, 200, 200) | uint8 | 0 | 255 | 4.569 | 0.2777 |
| zmns002 `w2` | (60, 495, 200, 200) | uint8 | 0 | 255 | 4.056 | 0.2771 |
| zmns002 `w3` | (60, 495, 200, 200) | uint8 | 0 | 255 | 2.604 | 0.1689 |
| zmns002 `w4` | (60, 495, 200, 200) | uint8 | 0 | 255 | 2.276 | 0.2450 |
| zmns002 `w5` | (60, 495, 200, 200) | uint8 | 0 | 255 | 3.616 | 0.3441 |

`ts` is `int64` (60,): zmns001 = 67…221, zmns002 = 56…187. **Non-contiguous** — successive
differences are `{1, 48, 49}` for zmns001 and `{1, 37}` for zmns002, i.e. short contiguous runs
separated by long jumps (the same "temporal blocks" packaging idea as zh001r).

**There are no other arrays. No nodes, no labels, no coordinates, no metadata, no scale.**

### What it actually is — established from the bytes

The `.npz` carries no scale, so the identity was derived and then *verified*:

**(a) Arithmetic decode.** Fetched the multi-view pyramid metadata:

| embryo | level 0 shape (T,C,Z,Y,X) | scale z,y,x (µm) |
|---|---|---|
| ZMNS001 | 270, 2, 396, 1624, 1552 | 1.97, 0.485, 0.485 |
| ZMNS002 | 225, 2, 408, 1596, 1824 | 1.97, 0.485, 0.485 |

ZMNS001 z extent = 396 × 1.97 = 780.1 µm; 780.1 / **1.625** = **480.1 → 480**. ✔
ZMNS002 z extent = 408 × 1.97 = 803.8 µm; 803.8 / **1.625** = **494.6 → 495**. ✔

Two embryos with *different* z sizes both landing on the packaged z size at the same 1.625 µm
target. The timepoint counts also fit (`ts` max 221 < 270; 187 < 225).

**(b) Byte-level verification.** Fetched ZMNS001 level-3 (50, 203, 194 at 15.76/3.88/3.88 µm,
1.2 MB) at the declared timepoint t = 67, brought `w0[0]` from the hypothesised 1.625 µm grid onto
that level-3 grid, and computed a normalised cross-correlation over all window positions:

- **NCC peak 0.8546** on channel 0, at level-3 index (0, 53, 13) → origin (0.0, 205.6, 50.4) µm
- 99.9th percentile of the NCC field 0.5738; median 0.0455 — a sharp, unique peak
- channel 1 peaks at only 0.5425 → the packaged data is **channel 0**
- the z index is 0 and the window spans 49 of 50 level-3 z slices → **full z extent**, xy window

**Conclusion:** `zmns00N_crops.npz` are **1.625 µm isotropic, full-z, 200×200 (= 325 µm) xy
windows of Zebrahub multi-view ZMNS001/ZMNS002, channel 0, rescaled to uint8**, 5 and 6 hand-picked
windows respectively (not a tiling — 5–6 windows of 325 µm cannot tile a 787 × 753 µm embryo).

**(c) A ruler measurement that disagrees, and why I discount it.** Adapting the radial-profile
ruler from `h1r_zh001r_audit.py` to detected blobs (no labels exist) gave an implied voxel size of
0.986 µm for zmns001 and 1.221 µm for zmns002 — inconsistent with each other and with 1.625.
The reason is visible in the profiles themselves:

| r (vox) | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| zh001r, real labels (known exactly 1.625 µm) | 1.000 | 0.709 | 0.390 | 0.167 | 0.093 | 0.079 | 0.066 | 0.046 |
| zh001r, *detected* blobs (same 1.625 µm data) | 1.000 | 0.836 | 0.529 | 0.253 | 0.118 | 0.064 | 0.040 | 0.022 |
| zmns001, detected blobs | 1.000 | 0.983 | 0.846 | 0.590 | 0.372 | 0.244 | 0.177 | 0.121 |
| zmns002, detected blobs | 1.000 | 0.962 | 0.722 | 0.421 | 0.242 | 0.151 | 0.111 | 0.076 |

The zmns profiles are nearly flat at r = 1 (0.98, 0.96) — the signature of mis-centred detections
on dim data (zmns means are 2.3–14.7 versus zh001r's 51.8), not of finer resolution. The zh001r
control row shows the detector alone already flattens the profile on data of *known* scale. **The
ruler is not trustworthy without real labels**; the arithmetic decode plus the 0.855 NCC is.

### What zmnscrops is worth

No labels, and no public lineage table exists for the multi-view ZMNS embryos (our four Ultrack
tables are ZSNS001/003/004/005). It supervises **nothing**. Its uses are self-supervised /
contrastive pretraining, or intensity-domain augmentation. It is 12.9 GB uncompressed for that.
**zh001r + our identity sidecar dominates it for every supervised purpose.**

Licence tags (factual, not exclusionary): `kkunizaw/biohub-zh001r` declares licence "other" with
description "Zebrahub ZSNS001 (CZ Biohub royerlab, CC BY-NC) real-label crops"; `biohub-zmnscrops`
declares "windowed crops derived from the public Zebrahub multi-view imaging dataset… No
competition data included". Upstream Zebrahub is CZ Biohub royerlab.

---

## 4. Level-1 → 1.625 µm isotropic resample pipeline

Script: **`scripts/win_bet/h1r_resample_iso.py`** (new). Reads `scale_zyx` from the source zarr's
`.zattrs` — never hard-coded, per §1 correction 2.

### Interpolation and anti-aliasing decision — measured, not assumed

`scipy.ndimage.zoom` does **not** anti-alias when downsampling. The prefilter used is a Gaussian
with `sigma = (decimation_factor - 1) / 2` input voxels per axis, zero on axes that are not
decimated.

Benchmark on the local `ZSNS003_L1.zarr` frame 0 — (129, 965, 1019) uint16 at (0.62, 0.2195,
0.2195) µm, decimation z 2.62× / xy 7.40×, output (49, 130, 138):

| method | s / frame | RMS deviation from cubic+antialias reference |
|---|---|---|
| nearest, no antialias | 0.53 | 31.24 |
| linear, no antialias | 0.61 | 28.26 |
| **linear, antialiased** | **11.15** | **1.36** |
| cubic, antialiased | 8.41 | — (reference) |

Reference intensity std is 124.53. **Skipping the prefilter costs a 28-unit RMS error against a
125-unit signal — a ~23% corruption at 7.4× decimation.** Anti-aliasing is mandatory, not
optional. Linear-plus-prefilter is within 1.1% of cubic-plus-prefilter, so **order = 1 with the
Gaussian prefilter is the chosen setting** (it is also the safer choice — cubic can overshoot
into negatives near nucleus edges).

### Cost on the real target embryo (ZSNS001)

ZSNS001 level-1 is (2.48, 0.878, 0.878) µm → zoom (1.526, 0.540, 0.540): z is **upsampled**
1.53×, xy downsampled 1.85×. Timed on a (224, 1087, 1212) uint16 frame (content-independent
compute):

- **linear, antialiased: 16.5 s/frame** → output (342, 587, 655) = 263 MB uint16
- linear, no antialias: 9.8 s/frame (rejected)

### Download cost (measured, ZSNS001 level-1, t = 158)

The pyramid chunking is `(1, 1, 128, 362, 362)` → **2 z × 3 y × 4 x = 24 chunks per frame**.

- 24 chunks = **23.7 MB compressed** (24.9× compression against the 590 MB raw frame)
- serial fetch: **63.3 s/frame**; per-request min 0.90 s, median 2.24 s, max 7.70 s
- effective 0.37 MB/s — **latency-bound, not bandwidth-bound** (the Kaggle download in §3 ran at
  11 MB/s on the same link), so parallel requests should recover most of this

**A blocker found while measuring this:** `scripts/win_bet/h1r_fetch_imaging.py` only ever
requests chunk `.../{zc}/0/0` and reshapes to `(czz, cyy, cxx)`. That is correct for ZSNS003
(1 y-chunk × 1 x-chunk) but **ZSNS001 level-1 has 3 y-chunks × 4 x-chunks**, so the assignment
`frame[z0:z0+czz] = block[:, :Y, :X]` will raise on a shape mismatch. The script **cannot fetch
ZSNS001 at level 1 as written** and needs a y/x chunk loop (it already prints a note about this
case but then proceeds). This must be fixed before any ZSNS001 stream lane starts.

### End-to-end estimate, 1-embryo edge-half training set

| timepoints | download (serial) | resample CPU | full-frame output |
|---|---|---|---|
| 50 | 1.19 GB, **53 min** | 14 min | 13.2 GB uint16 |
| 100 | 2.37 GB, **105 min** | 28 min | 26.3 GB uint16 |

So ~**1.1 h for 50 tp / 2.2 h for 100 tp** serial end-to-end, dominated by request latency;
an 8-way parallel fetch should bring 100 tp to roughly 45 min total. Disk is not a constraint
(383 GB free) but the full-frame output is far too large to package — see §5, which crops.

---

## 5. Kaggle dataset build plan (NOT executed — nothing uploaded)

Auth verified: `.venv\Scripts\python.exe -m kaggle datasets list -m` returns 13 datasets owned by
**`aryaarun07`**. Note: `.venv\Scripts\kaggle.exe` is **blocked by Windows Application Control** on
this machine ("An Application Control policy has blocked this file"); the `python -m kaggle` entry
point works and should be used everywhere instead.

### Dataset A — `biohub-zh001r-identity` (build this first)

The highest value-per-byte artifact available. It converts a public *detector-only* dataset into a
full detector + edge + division substrate.

- **Contents:** the recovered `track_id` / `parent_track_id` per crop node, plus the registration
  transform per crop (`t0`, `origin_um`, `scale_um`) and the residual stats as a provenance JSON.
- **Format:** `zh001r_identity.npz` with keys `tid_{crop}_{frame}` / `pid_{crop}_{frame}`
  (int64, aligned row-for-row with `f{crop*20+frame}` in the public `zh001r_nodes.npz`), plus
  `registration.json`.
- **Size: 8.7 MB** (already built at `data/external/zebrahub/zh001r_identity.npz`).
- **Build script:** `scripts/win_bet/h1r_zh001r_register.py --crops all --out … --report …`
  (exists and has been run; the run is reproducible in ~4 min with the `--t-prior` fast path,
  ~2.7 h without).
- **Kernel usage:** attach `kkunizaw/biohub-zh001r` and this dataset side by side; row `i` of
  `nodes[f"f{c*20+t}"]` gets identity `tid_{c}_{t}[i]`. No other join is needed.
- **Prerequisite:** it encodes ZSNS001 Ultrack lineage, which is CC BY-NC upstream. Tag it
  factually as derived from Zebrahub ZSNS001 (CZ Biohub royerlab) — do not exclude it, but the
  tag must travel with the dataset.

### Dataset B — `biohub-zsns001-iso-crops` (only if a second embryo/phase is needed)

Our own stream, needed only where zh001r's four temporal blocks are insufficient.

- **Contents:** 64³ uint8 crops at 1.625 µm isotropic on the same 48-voxel origin lattice zh001r
  uses, sampled at nucleus-dense origins (the embryos are a hollow shell — a naive origin grid
  wastes most of its budget on interior void), plus per-frame node arrays with `track_id` /
  `parent_track_id` taken directly from the Ultrack CSV.
- **Format:** mirror zh001r exactly (`*_iso.npy` `(crops, T, 64, 64, 64) uint8`, `*_nodes.npz`
  with `(N, 6) = [t, z, y, x, track_id, parent_track_id]`) so one kernel data path serves both.
- **Size:** a zh001r-shaped package (72 crops × 20 frames) is **377 MB** for the imaging plus
  ~18 MB for nodes-with-identity. Scaling to 100 timepoints × 72 windows would be ~1.9 GB.
- **Build script outline** (not yet written — proposed `scripts/win_bet/h1r_build_iso_crops.py`):
  1. fix the y/x chunk loop in `h1r_fetch_imaging.py` (§4 blocker), fetch ZSNS001 level 1 for the
     chosen timepoints in parallel;
  2. `h1r_resample_iso.resample_frame` per frame (order 1, antialiased, 16.5 s/frame);
  3. choose crop origins on the 78 µm lattice, ranked by node count from the Ultrack table;
  4. cut 64³ windows, rescale intensity to uint8, emit nodes with identity in crop coordinates;
  5. emit a manifest recording embryo, level, source scale, resample settings and the origin list.
- **Cost:** §4 gives ~2.2 h serial for 100 timepoints before cropping.

### Sequencing recommendation

Build and use **Dataset A only** until an experiment shows its four temporal blocks are the
limiting factor. It is 8.7 MB against ~2 GB, needs no download and no resample, and carries the
same 1.625 µm geometry as the deployed detector input — so unlike a level-1 retrain it does **not**
void the deployed 0.915 anchor on geometry grounds.

---

## What this does and does not change

**Changes.** The 2026-08-17 conclusion that `zh001r` is "a cheap detector-retrain substrate, not a
full H1 unblock" no longer holds: with the 8.7 MB sidecar it supervises the detector half, the
edge half (1.19 M edges) and the division half (34,179 events). The "two complementary lanes"
framing collapses to one lane, and the more expensive lane (our own level-1 stream) is now
optional rather than required for edge supervision. The recorded voxel scale for these crops is
corrected from 1.677 µm to exactly 1.625 µm.

**Does not change.** No LOEO or LB number moved. `bet-zebrahub-retrain` stays `proposed` and its
falsification is unchanged. Per CLAUDE.md rule 5 this is plumbing and data engineering, not
scientific evidence — and per the 2026-08-18 record, the LOEO promotion bar itself is falsified as
a sufficient gate, so any projection from this substrate should be discounted accordingly. The
labels recovered are Ultrack's automated output, so an edge model trained on them learns to
imitate Ultrack, which is a ceiling as well as a floor.

---

## Scripts created this session

- **`scripts/win_bet/h1r_zh001r_register.py`** — track-identity recovery for the packaged crops
  (Hough vote over timepoint+origin, LSQ refit, 2 µm acceptance gate, identity export). Run on all
  72 crops; output `data/external/zebrahub/zh001r_identity.npz` (8.7 MB, gitignored).
- **`scripts/win_bet/h1r_resample_iso.py`** — level-N → 1.625 µm isotropic resample with the
  Gaussian anti-alias prefilter, plus `--bench` to reproduce the §4 timing table.

Scratchpad-only probes (not added to the repo): tracks-CSV cache builder, the FFT
histogram-correlation registration variant, the zmns npz structure probe, the zmns blob ruler, the
zmns↔ZMNS001 NCC verification, and the identity/fetch-cost measurement scripts.

Also produced as a side effect: `data/external/zebrahub/ZSNS001_tracks_cache.npz` — the 21,697,591-row
Ultrack table t-sorted with a per-timepoint index, which cuts load time from 44 s (CSV) to 3 s.
