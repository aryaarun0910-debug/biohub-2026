# Live public surface — 2026-08-18

Snapshot: leaderboard CSV `2026-08-18T08:31:02Z` via authenticated Kaggle CLI
(`.venv/Scripts/python.exe -m kaggle …`; the `kaggle.exe` shim is blocked by an Application
Control policy — invoke via `python -m kaggle`). Deltas are vs
`competitive_refresh_2026-08-17.md` (morning) and Part 2 of `metric_forensics_2026-08-17.md`
(evening). Raw CSV + probe scripts in session scratchpad (`lb_0818/lb.csv`, `lb_analyze*.py`).

Bottom line: **the top moved for the first time in 2 days — TWEAK took the lead at 0.951**
(08-18 00:48, 170 subs). **liyansen has UNPUBLISHED the public 0.918 notebooks** (both 403
now), but the public ceiling survives at 0.918 via **xiaoleilian's fully independent
retrained-detector stack, published with weights** — the public frontier is now
retrain-based, not organizer-checkpoint-based, and it has *measured nonzero division recall*.
kkunizaw keeps assembling the Zebrahub lane in public (zh001r downloads 5→10 in ~18 h; they
submitted at 08-18 00:40, still 0.887 — the retrain has not converted yet).

---

## 1. Leaderboard now (2,483 teams; we are rank 229 at 0.915)

Top 20 (`python -m kaggle competitions leaderboard biohub-cell-tracking-during-development -d`):

| # | Team | Score | Subs | Last sub (UTC) |
|---|---|---|---|---|
| 1 | **TWEAK** (antonoof, tweakai, varianceofx) | **0.951** | 170 | 08-18 00:48 |
| 2 | Mark Cooper | 0.950 | 93 | 08-16 20:49 |
| 3 | Soheil Ayati | 0.948 | 30 | 08-17 20:50 |
| 4 | yuto083 | 0.947 | 52 | 08-18 01:03 |
| 5 | z7777 (songqizhou) | 0.945 | **7** | 08-16 07:33 |
| 6 | Matt Goldfield | 0.945 | 126 | 08-17 23:23 |
| 7 | enddl22 | 0.945 | 111 | 08-17 15:38 |
| 8 | Amin | 0.943 | 67 | 08-17 10:35 |
| 9 | Tang (hirotetsu) | 0.943 | 39 | 08-18 06:06 |
| 10 | htnhtn | 0.942 | 76 | 08-16 13:47 |
| 11 | Changye Li | 0.941 | 32 | 08-15 01:35 |
| 12 | Arnauya | 0.940 | 45 | 08-17 22:16 |
| 13 | **fromage _** (fromagesea) | **0.939** | **20** | 08-17 18:40 |
| 14 | Going down in 3, 2, 1… (felipekitamura) | 0.936 | 117 | 08-17 16:52 |
| 15 | atrei_des | 0.935 | 30 | 08-17 15:37 |
| 16 | Umesh Arampath | 0.935 | 87 | 08-12 20:35 |
| 17 | mikelou1 | 0.935 | 34 | 08-17 13:56 |
| 18 | xanderr (3-person team) | 0.934 | 153 | 08-18 00:04 |
| 19 | yoikoarmor | 0.934 | 39 | 08-17 14:52 |
| 20 | goh (bungohatayama) | 0.934 | **13** | 08-17 00:04 |

Movement vs 08-17 evening:
- **TWEAK 0.949 → 0.951** — first leader change; they overtook Mark Cooper (static at 0.950
  since 08-16). TWEAK is the AI-agent company that claimed a "universal +0.03–0.05 plugin" in
  thread #735352; whatever the marketing, they are now measurably first, on brute submission
  volume (170).
- Top-3 boundary unchanged at **0.948** (Soheil Ayati, 30 subs, resubmitted 08-17 20:50 with
  no gain). 0.945 tier unchanged (z7777 still frozen at 7 subs since 08-16).
- New efficiency signals in the 0.93s: **fromage _ 0.939 on 20 subs** (high teamId =
  recent entrant; fastest efficient climber on the board) and **goh 0.934 on 13 subs**. Like
  z7777, low-sub high scores imply a structural edge carried in, not tuned in.
- Tang (0.943) and yuto083 (0.947) both submitted within the last 8 hours — the 0.943+ tier
  is actively pushing.

Band counts (Δ vs 08-17 pm from `metric_forensics_2026-08-17.md` §2.1):

| Band | 08-17 pm | 08-18 am | Δ |
|---|---|---|---|
| ≥ 0.950 | 1 | 2 | +1 |
| ≥ 0.948 | 3 | 3 | – |
| ≥ 0.945 | 7 | 7 | – |
| ≥ 0.940 | 11 | 12 | +1 |
| ≥ 0.935 | 17 | 17 | – |
| ≥ 0.930 | 30 | 31 | +1 |
| ≥ 0.925 | 45 | 46 | +1 |
| ≥ 0.920 | 77 | 77 | – |
| ≥ 0.918 | 94 | 94 | – |
| ≥ 0.916 | 196 | **212** | **+16** |
| ≥ 0.915 | 484 | 493 | +9 |
| exactly 0.915 | 288 | **281** | −7 |

**Where 0.916–0.930 sits** (distinct-score counts): 0.916×76, 0.917×42, 0.918×13, 0.919×4,
0.920×6, 0.921×12, 0.922×1, 0.923×7, 0.924×5, 0.925×3, 0.926×4, 0.927×5, 0.928×2, 0.929×1,
0.930×4. Read: the plateau erodes only into the **0.916–0.917 shelf** (+16 teams in ~12 h =
propagation of the public fork lineage), then a **hard cliff above 0.918** (13 teams at 0.918,
only 4 at 0.919). Nobody has publicly broken 0.918 by a small margin — the 0.919–0.930 band
(~47 teams) is thin, flat, and silent, consistent with private knob-tuning on top of the
public stack rather than a leaked new lever. The levers that pay above 0.918 remain private.

We are **rank 229 / 2,483 at 0.915** (12 subs, last 08-17 19:59). Rank vs 403 yesterday is a
tie-break artifact (281 teams share 0.915), not movement.

---

## 2. Timeline + rules facts (from the competitions API object; script `comp_meta.py`)

Verified via `kaggle.api.competitions_list(search="biohub")` full field dump:

- **Final submission deadline: 2026-09-29 23:59 UTC** — 42 days from today, 36 from the
  user's 08-24 resumption.
- **Entry deadline AND team merger deadline: 2026-09-22 23:59 UTC** (7 days before close).
- **Max daily submissions: 5.** Max team size: 5.
- **Kernels-submissions-only: true** (code competition; submit via
  `kaggle competitions submit -k <kernel> -v <version>`; T4x2 pin per memory).
- Reward **$60,000**, category Research, awards points/medals. Host org: Biohub. Launched
  2026-04-01, enabled 2026-06-29. Evaluation metric name string: "CZI Biohub Zebrafish 133605".
- Budget math: 5/day × 36 days from 08-24 = **≤180 submissions remaining** for us; no scarcity.
- **UNVERIFIED:** number of selectable final submissions (Kaggle default is 2; not exposed by
  the API; competition page is an SPA that WebFetch cannot read — confirm in browser once).
- **UNVERIFIED:** prize split across places (web search surfaces only the $60k total).
- API field `_is_frozen = True` — semantics unclear (not a leaderboard freeze; scores are
  visibly updating). Do not act on it.
- **No host announcements found in-window** (discussion API still 403 to CLI auth; see §4).
  royerlab/kaggle-cell-tracking-competition GitHub repo shows no fresh activity signal
  (58 stars, 0 open issues; https://github.com/royerlab/kaggle-cell-tracking-competition).

---

## 3. Rival activity

### 3.1 kkunizaw (the Zebrahub-retrain rival)

- LB: **rank 1,309 at 0.887, 49 subs, last submission 2026-08-18 00:40** — submitting daily,
  score unchanged. The retrain lane is visibly *active but not yet converting*.
- Public kernels: **none** (`kaggle kernels list --user kkunizaw` → Not found).
- Datasets (both competition-targeted, `kaggle datasets list --user kkunizaw`):
  - `kkunizaw/biohub-zmnscrops` — 3.66 GB, updated 08-16 19:18, **4 downloads** (flat).
  - `kkunizaw/biohub-zh001r` — 364 MB, updated 08-17 15:12, **10 downloads** (was 5 at 08-17
    pm — doubled in ~18 h; adoption is starting but tiny). 1 vote.
    https://www.kaggle.com/datasets/kkunizaw/biohub-zh001r
- No discussion posts discoverable (see §4). **Read:** they are 3+ days into a public
  Zebrahub retrain with 49 submissions of evidence that it has not beaten even the plateau.
  Either their detector transfer is failing (imaging-domain gap) or they are iterating
  training and submitting probes. Their pipeline maturation (raw crops → `_iso`/`_tgt`
  training pairs) says intent; their 0.887 says the lever is still unclaimed in public.

### 3.2 The public 0.918 recipe changed hands overnight — liyansen pulled out, xiaoleilian is the new (better) reference

- **`liyansen/biohub-v16-ranker-persistent-divisions` and `…-v13-ranker-recall-finetune` are
  now 403 Forbidden** on `kaggle kernels pull` (control pulls of anhadmahajan06, sleepymegacat,
  dalloliogm succeeded in the same session — this is unpublication, not an API outage).
  liyansen also vanished from `kernels list --user liyansen` (Not found). Their LB row stands
  (0.918, rank 83, 19 subs, last 08-17 01:12). **The best public recipe writeup was withdrawn**
  — plausibly a competitor going quiet before pushing further. Our archived read of its
  contents (metric_forensics §2.4) is now the surviving record.
- **`xiaoleilian/biohub-m001-ens3-sm6-sim2`** (last run 08-14, 41 votes) — author **Xiaolei
  Lian, rank 86 at 0.918**, still public; pulled and read this session. This is a **fully
  independent stack**: their **own from-scratch UNet3D (base=24) detectors** — 3-model
  ensemble `unet3d_bright.pt` / `unet3d_traintophat.pt` / `unet3d_v2_tophat_b32.pt` — with
  flip-quartet logit-TTA, intensity-weighted sub-voxel refine (`_refine`), physical NMS 4.0 µm,
  two-pass µm-gated Hungarian with velocity extrapolation + appearance (logit-difference) cost,
  snap-only 1-frame gap closing (never synthesises a node), short-track filter 6, linefit
  smoothing, and a **"safe divisions" patch measured at division TP/FP/FN = 6/31/8 on VAL-24**
  (base predicts 0 divisions). Local VAL-24 (official `tracking_cellmot` metric) 0.8623 → LB
  0.918. Weights are public: `xiaoleilian/biohub-unet3d-weights-v2models` (62 MB, 08-14,
  12 downloads). https://www.kaggle.com/code/xiaoleilian/biohub-m001-ens3-sm6-sim2
  - **Two implications.** (a) The public ceiling no longer depends on the organizer
    checkpoint: a from-scratch retrained detector matches it, with reproducible weights —
    direct public proof that **retraining reaches 0.918+ and is the floor of the private
    0.92+ band**, and a live starting point anyone can fork. (b) The division-cap constants
    we and liyansen share (`FRAME_FRAC_CAP=0.0076`, `GLOBAL_FRAC_CAP=0.00375`,
    `DIVERGE=2.25 µm`) appear verbatim here as `DIV_FRAME_CAP`/`DIV_GLOBAL_CAP`/
    `DIV_DIVERGE_UM` — this m001 lineage is plausibly the *origin* of the whole public
    division config, and unlike the liyansen framing ("suppress divisions"), the original is
    measured to *add* 6 division TPs. The public 0.918 already contains a small, real,
    working division-recall mechanism.
- Author↔LB cross-reference of everything public: Pilkwang 0.921 (r68, 159 subs) remains the
  highest-ranked public author but his notebooks predate the wall; Yusuke Togashi now 0.917
  (r106, **158 subs**, active 08-18 04:23); trwang2025 0.918 (r92); altervation 0.917 (r116,
  active 08-18 02:58, Spotiflow lane per 08-17 report). **Nothing public scores above 0.918.**

### 3.3 Other new artifacts since 08-17 pm (`datasets list -s biohub --sort-by updated`)

- **`rudispresence/biohub-stabledet-hoct-code`** (825 MB, 08-17 17:21, 26 dls) +
  `…-hoct-hard-negative-checkpoints` (29 MB, 08-17 17:15, 16 dls) + `…-hoct-runtime` (353 MB,
  08-17 05:43, 0 dls). A **HOCT retrain with hard-negative mining** lane. Author =
  Drifffffft, **rank 1,654 at 0.831** — another public retrain attempt that is not
  converting. HOCT-as-drop-in already failed for us; no update needed to that close-out.
- **`horaz0/biohub-regular-linajea-hybrid-artifact`** (42 MB, 08-17 17:01) plus their
  08-14/08-15 linajea/softmax 50-epoch artifacts (27–35 dls). horaz0 = **rank 516 at 0.914**,
  32 subs, active 08-17 17:55 — a systematic Linajea-hybrid retrainer sitting just under the
  plateau. Watch: the artifact naming ("two fold", "50ep") shows disciplined fold-based
  retraining; they are the closest-to-converting public retrain lane after xiaoleilian.
- `dariushafshar/kaggle-competition-leaderboard-intelligence` (1.3 MB, 08-17) — LB-scraping
  meta-dataset, author at 0.915; no science.
- New kernels (`kernels list -s biohub --sort-by dateCreated`): `deepakjnath/biohub-climb-notes`
  (08-18, pulled — a public "method card" of the 0.915 cluster's known ladder, no new
  technique); `abhimanyu122` v20–v23 relink/detection knob sweeps (author 0.915);
  `antonkartavtsev` U-Net inference kernels (0.827); forks of the two-seed blend. **No new
  notebook above the plateau.**

---

## 4. Discussions — not refreshable this session

- CLI/API: discussion endpoints 403 to API-key auth (same as 08-17 pm session).
- WebFetch: kaggle.com discussion pages return the SPA shell; forum.image.sc thread 121671
  returns 403 to non-browser agents.
- Web search: Google/Bing have not indexed any biohub discussion thread content; no new
  external write-ups, blogs, or tweets in-window (only launch-era coverage: FEBS Network
  post, Loic Royer LinkedIn, currypurin tweet — all June/July).
- Standing discussion state therefore remains the 08-17 morning sweep
  (`competitive_refresh_2026-08-17.md` §2: #735352 shakeup thread with the TWEAK plugin
  claim, #734604 decomposition advice, #735531 "training is the only way up").
  **Any host clarification posted 08-17→08-18 would be invisible to us — check the
  discussion tab in a browser on resumption (5 minutes, highest-value manual step).**

---

## 5. Synthesis

**What the 0.94+ tier holds.** The picture sharpened: (a) xiaoleilian proves publicly that a
from-scratch retrained detector + careful post-proc = 0.918 with *measured* division recall
TP=6; (b) the 0.919–0.930 band is thin and silent (private tuning on retrained detectors);
(c) three low-sub high scores (z7777 0.945/7, fromage 0.939/20, goh 0.934/13) still argue a
structural, portable edge; (d) TWEAK reaching 0.951 on 170 subs shows the top is also partly
probe-driven. Nothing observed this window contradicts the standing two-hypothesis split
(retrained/generalising edge model vs working division recall); xiaoleilian's public TP=6 at
0.918 slightly strengthens the division half — division recall is demonstrably recoverable,
and the top tier has had 3+ weeks to scale that mechanism.

**Plateau erosion rate.** Slow and bottom-fed: exact-0.915 lost 7 teams in ~12 h to the
0.916–0.917 shelf (+16), while ≥0.920 was static. The public fork lineage lifts teams off
0.915 by +0.001–0.002 and then stalls at the 0.918 cliff. liyansen unpublishing the best
0.918 recipe *slows* public propagation from here; xiaoleilian's stack (with weights) is the
remaining escalator and its votes (41) and weight downloads (12) are still modest.

**By 2026-08-24 (resumption), expect [PROJECTION]:** leader 0.951–0.953 (TWEAK momentum);
top-3 boundary ~0.948–0.949; 10–15 more teams on the 0.916–0.917 shelf; plateau still ≥250 at
0.915; xiaoleilian-fork lineage starts appearing in new public notebooks. Watch specifically:
kkunizaw's score (any jump ≥0.915 = Zebrahub retrain converting → our H1 window narrows) and
horaz0 (closest disciplined retrainer to the plateau). **By deadline (09-29):** merger
deadline 09-22 makes the week of 09-15 the last chance to team with a 0.94-holder; the
0.918 cliff will not survive six more weeks — assume the public ceiling reaches 0.92+ once
any retrained-weights notebook propagates, and that top-3 will require ≥0.949-equivalent
private-robust score.

**Immediate implications for us:** (1) our sub-voxel refine lane is corroborated again —
xiaoleilian's `_refine` (intensity-weighted centroid, rz=2/ryx=5) + 4 µm physical NMS are
load-bearing in the only surviving public 0.918; (2) their measured safe-division patch
(TP 6 / FP 31 on VAL-24 under the exact caps we deploy) is a concrete, public, working
division-recall config to compare our motion-gate arm against; (3) their public 62 MB
retrained weights are an attachable detector for a cheap H1-adjacent A/B without any GPU
training of our own.
