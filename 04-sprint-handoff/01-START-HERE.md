# 01 START HERE

Written 2026-09-07 by Claude, on Arya Arun's order, as the HP laptop was cleared.
Read this file first, then 02, 03, 04, 05 in order. It takes about twenty minutes.

## The situation in six lines

- The competition closes **2026-09-29 23:59 UTC**. Verify that on the competition page before planning around it.
- The MacBook Pro (M5 Pro, 18-core CPU, 20-core GPU, 48 GB unified memory) arrives **2026-09-17**.
- That leaves **twelve days** from first boot to the deadline. It is the whole runway, not a warm-up.
- Two prior campaigns exist. Both are finished and sealed on GitHub. Neither runs any more.
- The best score either campaign reached is **0.925**, rank 207 of 2,693 on 2026-08-24. The leader was 0.962.
- The HP laptop's fans are dead. It was thermally throttled to about 1.4 GHz for the whole of the second campaign. Every design decision that looks over-engineered was probably a workaround for that machine.

## Where everything is

| Repository | What it is | Visibility |
|---|---|---|
| `aryaarun0910-debug/Biohub-CellTracking-2026` | The monolith. 550 commits, 2 July to 1 September. Reached 0.925. | private |
| `aryaarun0910-debug/Biohub-X` | The blind restart. 136 commits, 1 to 7 September. Reached 0.496. | private |
| this repository | The sprint. Starts empty by design. | private |

Biohub-X also carries a branch `archive/evidence-2026-09-07`. It is an orphan commit holding the 243 measurement reports, checkpoints and submission CSVs that its `.gitignore` excluded, so they survived the laptop being wiped. It does not satisfy that repository's isolation contract and must never be merged.

Nothing else survived, deliberately. About 108 GB of competition data, agent transcripts, fetched kernel outputs and virtual environments was deleted, all of it regenerable from Kaggle or from a lockfile.

## What to do in the first hour on the Mac

Do these in order. The third one can end the sprint plan, so do not leave it until day nine.

1. Install `uv`, Python 3.12, and the Xcode command line tools. Clone all three repositories.
2. Re-download the competition data with `kaggle competitions download`. It is roughly 98 GB, so start it before anything else and let it run.
3. **Test that PyTorch's Metal backend can do 3D convolution.** Run a `Conv3d`, a `MaxPool3d` and a `ConvTranspose3d` forward and backward on `device="mps"`, and time them against `device="cpu"`. The detector is built entirely from those three operations. If MPS silently falls back to CPU, or errors, then the Mac's GPU cannot train this model in PyTorch and the whole compute plan changes on day one instead of day nine. See 04 for what to do in each case.

## The one thing that is easy to get wrong

The submission is a **Kaggle notebook**, twelve-hour runtime cap, internet disabled, producing `submission.csv`. That notebook runs on Kaggle's machines. Apple's frameworks do not exist there.

So MLX cannot produce a submission. It could only ever be a local training tool whose weights you convert afterwards, which costs you a conversion step and a numerical-agreement check for no clear gain. Write PyTorch, select the device at runtime, train on `mps` at home and run inference on `cuda` in the Kaggle notebook. One codebase, two devices.

## What Arya decided, so you do not re-litigate it

- Both prior campaigns are over. Do not resume either.
- The quarantine that separated them is dissolved. Document 03 exists because of that. The new campaign is **not** blind, and that is now a deliberate choice with known consequences, spelled out in 03.
- The sprint rebuilds the system rather than continuing either repository, because much of Biohub-X was shaped by a 16 GB throttled laptop that no longer matters.
- Rebuild does not mean retype. Port what is hardware-independent. Document 04 says exactly what.
