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
