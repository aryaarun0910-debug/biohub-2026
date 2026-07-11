# Provenance audit: the 0.968 outlier and public Biohub movies

**Date:** 2026-07-11  
**Question:** can public/train movie identity or public Zebrahub tracks plausibly explain the new 0.968 leaderboard outlier, and which version of that idea is prize-safe?

## Bottom line

The four downloadable `test/` movies are byte-for-byte copies of four labeled `train/`
movies, but this is a deployment decoy and **cannot explain 0.968**. On 2026-07-10 the host
confirmed that those files are dummy placeholders and that scoring uses a much larger private
test set with no train overlap.

There is nevertheless a real and important provenance edge. The competition's own public code
names source crops from Biohub's public `2024_03_22_dorado` movie, and a public 122 MiB track
bundle contains dense Ultrack trajectories for that same 522-frame acquisition. I recovered two
of the organizer's original crop coordinates and matched them to anonymized training clips:
96.1% and 97.9% of sparse ground-truth nodes lie within the official 7 um radius of the public
track points, with median errors of 0.91 and 0.81 um. Directly converting those public trajectories
to competition predictions produced adjusted edge-J of 0.8490 and 0.9016 on the two clips.

That proves that public dense trajectories are genuine same-source supervision, not merely a
similar-domain dataset. It does **not** prove that the hidden leaderboard embryo is public, and
the two direct-transfer scores do not by themselves explain 0.968. The outlier is consistent with
an exact-source strategy plus substantial correction, but it is equally consistent with a private
method or an unrelated metric effect. No public evidence currently identifies Kevin's method.

## Evidence ledger

### 1. Visible test files are exact train duplicates — verified, but irrelevant to scoring

Full recursive SHA-256 comparison, not a sampled image comparison:

| Movie | Files compared | Tree delta | Byte-identical | Bytes per copy |
|---|---:|---:|---:|---:|
| `44b6_0113de3b` | 102 | 0 | yes | 456,757,564 |
| `44b6_0b24845f` | 102 | 0 | yes | 547,662,847 |
| `6bba_05b6850b` | 102 | 0 | yes | 361,668,669 |
| `6bba_05db0fb1` | 102 | 0 | yes | 540,242,928 |

This independently strengthens a public report in the welcome discussion. The decisive host
reply says these are “dummy placeholder files” for validating notebook output and that the actual
leaderboard uses a much larger deliberately private test set with no overlap with public train.
[Host discussion](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/716062)

**Conclusion:** copying the four train GEFFs is not a scored strategy and cannot produce 0.968.

### 2. The 0.968 event — verified, method unknown

Authenticated leaderboard refresh after the event:

- team: `Kevin` / `kevinzodiac`;
- score: **0.968**;
- submission time: 2026-07-11 00:25:45;
- submission count: 18;
- second place: 0.910 at the time of the snapshot.

The account exposes no public Biohub kernel or Biohub dataset through the Kaggle API. Searches of
current competition discussions, public kernels, GitHub code and the account's public assets did
not reveal a method. The 0.058 gap is evidence that this is an outlier, not evidence of its cause.

### 3. Competition crops came from the public March-22 movie — verified

Biohub publicly hosts a 522-frame nuclear zebrafish movie whose OME metadata names its origin
`2024_03_22_dorado/stabilized.zarr`, shape `522 x 1 x 505 x 2217 x 2170`, and voxel scale
`1.625 x 0.40625 x 0.40625` um — exactly the competition scale.
[Public movie](https://public.czbiohub.org/royerlab/ultrack/zebrafish_embryo.ome.zarr/)

The organizer's initial public repository contains real-data test fixtures named:

- `2024_03_22_dorado_0001_0190_1651_0467`;
- `2024_03_22_dorado_0002_0198_0184_0605`.

The suffix is the source crop's `t,z,y,x` origin for a `100 x 64 x 256 x 256` competition
clip. See the organizer's [I/O fixture](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/b7a61927707ee4c012ac87b8fb4328fb01759733/tests/test_io.py)
and [division fixture](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/b7a61927707ee4c012ac87b8fb4328fb01759733/tests/conftest.py).

### 4. The public track bundle is aligned dense supervision — verified experimentally

Biohub also serves `tracks_zebrafish_bundle.zarr.zip` (128,421,283 bytes locally). Its point
array has 522 frames, and its track-to-point CSR contains 11,851,323 memberships across 398,662
track rows. Coordinates use the movie's isotropically expanded Z convention (`z * 4`), which
maps back to the competition's native ZYX grid.
[Public track bundle](https://public.czbiohub.org/royerlab/zoo/Zebrafish/tracks_zebrafish_bundle.zarr.zip)

Using only the two source names committed by the organizer, I cropped the public points and
matched them against every anonymized competition training graph. The correct identities were
unambiguous:

| Original source crop | Anonymized crop | GT node recall at 7 um | Median nearest error | Direct public-track adjusted edge-J |
|---|---|---:|---:|---:|
| `..._0001_0190_1651_0467` | `44b6_cf8fed6b` | 0.9791 | 0.8125 um | **0.9016** |
| `..._0002_0198_0184_0605` | `44b6_587a1e22` | 0.9606 | 0.9084 um | **0.8490** |

The exact metric decomposition was:

| Crop | Pred nodes / estimated | TP | FP | FN | raw edge-J | adjusted edge-J |
|---|---:|---:|---:|---:|---:|---:|
| `44b6_cf8fed6b` | 34,621 / 43,729 | 174 | 9 | 14 | 0.8832 | 0.9016 |
| `44b6_587a1e22` | 15,102 / 18,137 | 329 | 23 | 42 | 0.8350 | 0.8490 |

These predictions used consecutive memberships within each public track and deliberately omitted
cross-track division edges. The result proves provenance and usefulness, while also showing that
raw Ultrack trajectory transfer is not automatically a 0.968 solution.

### 5. Other public same-lab embryos — verified inventory, hidden mapping unverified

The public directory exposes five dense-track CSVs (ZSNS001, its tail crop, and ZSNS003–005),
plus images for ZSNS001–005. The image metadata are:

| Asset | Frames / spatial shape | Voxel scale (Z,Y,X), um | Recorded origin | Dense CSV |
|---|---|---|---|---|
| ZSNS001 | 791 / 448×2174×2423 | 1.24×0.439×0.439 | 2021-08-02 whole embryo | yes |
| ZSNS001 tail | 791 / 420×1217×1091 | 1.24×0.439×0.439 | 2021-08-02 tail crop | yes |
| ZSNS002 | 1100 / 333×2430×5254 | 1.24×0.439×0.439 | 2020-12-02 fish tail | no CSV listed |
| ZSNS003 | 515 / 258×1929×2038 | 1.24×0.439×0.439 | `2024_02_21_daxi` | yes |
| ZSNS004 | 600 / 294×1926×2581 | 1.24×0.439×0.439 | `2024_02_28_daxi` | yes |
| ZSNS005 | 600 / 291×1910×2750 | 1.24×0.439×0.439 | `2024_03_14_daxi` | yes |

[Public image/track directory](https://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/).
The Ultrack paper describes 1.7–3.7 TB DaXi acquisitions across three embryos and makes its
imaging data and weights available publicly. [Ultrack article and data availability](https://pmc.ncbi.nlm.nih.gov/articles/PMC12615266/).

The host separately states that test embryo IDs do not overlap train and that test is roughly
similar in size. That statement does not establish or exclude overlap with an external public
Biohub acquisition. [Embryo-ID discussion](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/716793).

No retrieved metadata, discussion, kernel, paper or Git history identifies the hidden scored
embryo as ZSNS001–005 or another named public acquisition. **Hidden-public identity remains an
open hypothesis, not a finding.**

## Rules and prize-safety classification

The rules expressly allow external data when it is publicly/equally accessible at no cost. They
also prohibit information from hand labeling or human prediction of test records. There is no
retrieved clause that literally says “do not identify a test movie” or “do not reverse-engineer
the source.” [Official rules](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/rules).

That does not make every source-matching use prize-safe. Winner code, provenance and licenses are
audited, and the public Biohub directory does not visibly attach a dataset license to every image
and CSV. Use the following operational statuses:

| Use | Status | Reason |
|---|---|---|
| Train association/detection models on public movies/tracks | **PERMITTED, license record required** | Classic public external-data use; no test identity needed. |
| Use public trajectories to build motion, cadence, division and density priors | **PERMITTED, license record required** | Priors generalize and do not transfer a test label. |
| Automatically fingerprint a test movie only to select normalization/model/prior | **PERMITTED on current text; document it** | Algorithmic inference, no labels or human prediction. |
| Test-time self-supervised adaptation on the unlabeled movie | **PERMITTED on current text** | Covered separately by `RULES_TEST_TIME_ADAPTATION_2026-07-10.md`. |
| Register an identified public movie's dense trajectories and emit them as test predictions | **NEEDS WRITTEN HOST CLEARANCE / QUARANTINE** | Technically plausible under the external-data clause, but functionally transfers source labels into hidden test records and has major adjudication risk. |
| Manual matching/correction, submission-score label reconstruction | **FORBIDDEN / DQ RISK** | Hand prediction or leaderboard reconstruction. |

The exact organizer question should be sent privately:

> If a submitted notebook algorithmically matches a provided test crop to a freely public Biohub
> source movie, may public machine-generated trajectories from that movie be registered and used
> as External Data for prediction, or may they only be used for pretraining and aggregate priors?

Do not implement the direct-transfer branch until that answer is in writing.

## Does this plausibly explain 0.968?

**Technically plausible, evidentially unproven.** The competition crops are demonstrably derived
from at least one publicly released full movie, and aligned public dense trajectories already
score 0.85–0.90 with no learned correction on two recovered crops. Because the metric evaluates
sparse annotated edges while expecting a dense node count, a correctly registered dense graph is
an unusually strong prior. Better public tracks, organizer-style relinking, image refinement and
divisions could raise it further.

But the available direct-transfer measurement is materially below 0.968, the host guarantees no
train overlap, no hidden-to-public match has been established, and Kevin exposes no supporting
artifact. Assigning the outlier to “Zebrahub leakage” today would be a story, not intelligence.

## Highest-EV legal move now

Use the March-22 and ZSNS trajectories as **dense association pretraining and calibration data**.
This is stronger than generic same-domain pretraining because March-22 is proven source-aligned to
one competition embryo. Corrupt the dense graphs to competition-like missing detections,
localization noise and false candidates, then train the organizer/Trackastra fusion head and
motion residual on next-edge recovery. Gate it by training on public trajectories plus one
competition embryo and evaluating untouched on the other competition embryo, then reverse.

In parallel, build a label-free offline fingerprint index, but initially use identity only for
model/normalization/prior selection:

1. ship low-resolution temporal MIPs and gradient/quantile descriptors for every public movie;
2. match hidden clips with multiscale phase correlation and mutual information;
3. demand independent agreement from intensity texture, cell-count trajectory and physical-scale
   metadata;
4. log the match confidence and fall back to the general model when confidence is low;
5. keep the public trajectories disconnected from prediction unless the host explicitly clears
   direct registration.

This captures most of the legal generalization advantage immediately and preserves the option
value of exact-source registration without gambling prize eligibility.
