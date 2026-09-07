# 05 THE TWELVE DAY SPRINT

17 September to 29 September 2026. Adjust freely, but keep the two rules below.

## Two rules that decide whether this works

**Rule one: be submittable by the end of day two.** Not competitive, submittable. A
valid CSV from the classical baseline, accepted by the leaderboard. From that moment
every later day is an improvement on something real rather than a gamble on finishing.
The failure mode that ends sprints is arriving at day eleven with better science and
nothing that runs end to end.

**Rule two: change one thing at a time.** The previous campaign's own record says its
best score came from a bundle of three changes and that one knob it believed in had no
effect at all. With twelve days you cannot afford to attribute a gain to the wrong
cause and then build on it.

## The shape

| Days | Dates | Goal |
|---|---|---|
| 1 to 2 | 17 to 18 Sep | Machine ready, system ported, **first accepted submission** |
| 3 to 5 | 19 to 21 Sep | Learned detector beating the classical baseline at a matched budget |
| 6 to 8 | 22 to 24 Sep | Association: ranking and acceptance measured separately |
| 9 to 10 | 25 to 26 Sep | Divisions, and whichever of detection or association is weaker |
| 11 | 27 Sep | Freeze. Final training runs only. No new ideas. |
| 12 to 13 | 28 to 29 Sep | Final submissions and buffer. Assume you lose one day to something. |

## Day by day

**Day 1, 17 September.** Start the 98 GB data download first, it runs unattended.
Install uv, Python 3.12, Xcode command line tools. Clone the three repositories. Run
the MPS 3D-convolution test from document 04 and record the result. Decide the compute
plan on that measurement today.

**Day 2, 18 September.** Port the contracts, the metric adapter, the submission writer
and validator, and the classical detector. Run the classical pipeline end to end on
the four visible test movies. Build the Kaggle notebook, exercise it locally, submit.
**Target: an accepted submission on the leaderboard by tonight.**

**Days 3 to 5.** The learned detector. Train the 3D U-Net with the positive-unlabelled
loss and a prior built from each window's own estimate. The bar is not "it trains", it
is **higher recall than the classical detector at the same number of proposals**, on
held-out data, both directions. If it cannot clear that bar by end of day 5, keep the
classical detector and spend the time on association instead. Write that stopping
condition down now, while it is still cheap to accept.

**Days 6 to 8.** Association. Measure ranking and acceptance separately from the first
run, because a matcher that ranks well and accepts badly looks like a broken model
otherwise. Note the entry deadline falls on 22 September; confirm nothing about your
entry status needs action.

**Days 9 to 10.** Divisions, and reinforcement of whichever stage the numbers say is
weaker. Resist starting anything new after the morning of day 10.

**Day 11, 27 September.** Freeze the code. Only training runs and packaging. Any bug
found today gets fixed only if it changes the output.

**Days 12 to 13, 28 to 29 September.** Final runs, final submissions, and slack. The
deadline is 23:59 UTC on the 29th. Treat the 28th as the real deadline.

## Submission budget

About five per day. Spend them deliberately:

- One on day 2 to establish the baseline.
- One whenever a stage clears its bar, so the leaderboard confirms the local measurement.
- Keep at least two in hand on each of the last two days.

Never let a day pass in the last week with your best local model unsubmitted.

## Decision gates, written before the work

Each gate has a stated falsifier so that stopping is a result rather than a defeat.

| Gate | Passes if | Otherwise |
|---|---|---|
| MPS viability, day 1 | 3D convolutions run on `mps` at a useful speed | Train on Kaggle, use the Mac for everything else |
| Submittable, day 2 | An accepted score appears on the leaderboard | Stop all science until it does |
| Learned detector, day 5 | Beats the classical baseline at a matched proposal budget, held out, both directions | Keep the classical detector, move to association |
| Association, day 8 | Beats the untrained matcher on the official score, held out | Keep the untrained matcher, move to divisions |
| Freeze, day 11 | Best configuration is trained, packaged and submitted | Submit the best thing that exists, whatever it is |

## What to carry forward from the old process, and what to drop

Keep: one registry as the only home for numbers, the declaration before the run, the
digest binding, and the commit gate. They are cheap and they are why the previous
records can be believed.

Drop: the ceremony that a two-month campaign could afford and twelve days cannot.
Fifty pre-registered work packets and a generated knowledge graph are not the reason
that campaign scored 0.925.
