# Codex addendum — asymmetric-edge / red-team front (Biohub Cell Tracking)

Mission: find the moves that take the LEAD by outsmarting the metric and the data, not by
grinding the public board. Prioritize edges that are high-leverage AND keep us eligible to
collect (i.e. don't get the account banned / prize forfeited). Cite everything; label each
finding with EV (high/med/low) and STATUS (permitted / needs-rules-check / forbidden-DQ).

Context you already have (don't re-derive): exact metric known; 2 train embryos (44b6,6bba),
hidden test = different embryo(s); data is DaXi/Zebrahub-derived; Biohub publicly hosts
ZSNS001-005 embryos + full dense track CSVs + unet-daxi/simview weights; count penalty is mild;
FPs only count in annotated regions; per-timepoint optimal bipartite match at 7um on 1/(1+d).

## PRIORITY 1 — External-data provenance (potentially decisive; rules-gated)
1. Read the competition RULES verbatim WHILE LOGGED IN (anonymous scrape couldn't render the
   external-data clause). Quote exactly: is free/public external data + pretrained models allowed?
   Any clause forbidding "identifying/using test-set source data" or "reverse-engineering the
   test set"? This single clause decides whether Priority-1 is a weapon or a trap.
2. Determine whether the HIDDEN TEST embryo could be a PUBLIC Zebrahub embryo (ZSNS001-005) or a
   truly held-out unpublished one. Evidence to gather: the Zebrahub/DaXi papers' list of released
   embryos; whether 44b6/6bba map to any public embryo id; developmental-stage / frame-count /
   voxel-count fingerprints of train crops vs public embryos. If test is public, its dense track
   CSV is a massive prior.
3. If (and only if) rules permit public external data: design the fingerprinting method to match a
   test crop to a public embryo+timepoint (image phash/mutual-information registration, cell-count
   trajectory, intensity profile). Spell out the offline Kaggle-notebook mechanism (public data
   shipped as a Kaggle dataset). STATUS this explicitly.
   - NOTE the line: using PUBLIC external data = the intended, legal edge. Deanonymizing/
     reconstructing the PRIVATE labels via the submission/scoring system = DQ, do not propose.

## PRIORITY 2 — Metric-surface exploitation (all permitted; pure math)
1. Nail the count curve: for ratios N_pred/N_est in {0.7,0.8,0.9,0.95,1.0,1.05,1.1,1.2,1.5,2.0},
   compute adjusted-J multiplier AND realistic edge-J impact; identify the true optimum band per
   embryo. Quantify the undercount reward and whether a deliberate slight undercount + high-precision
   edges beats high recall.
2. Assignment gaming: since matching is optimal bipartite at 7um on 1/(1+d), does placing a single
   high-precision detection near each likely-annotated site beat many nearby proposals? Model how
   sub-voxel accuracy shifts the assignment. Where do duplicate/near proposals get wasted?
3. FP-free-zone exploitation: FPs only count in annotated regions. Quantify how aggressively we can
   over-propose in likely-unannotated space (to lift recall of the FEW annotated nodes) before the
   count penalty bites. This decouples "recall proposals" from "count".
4. Division arbitrage: division is 10% but structurally brittle; compute the exact break-even
   precision at which adding forks is net-positive on the combined score, per embryo.

## PRIORITY 3 — Host-as-competitor & interdisciplinary priors (permitted)
1. The host benchmark (~0.810) is a competitor; their code has a no-op division up-weight and a
   checkpoint metric != LB metric. Where else is the FIELD systematically weak because everyone
   forked the same public DoG notebook? Identify the blind spots the whole plateau shares.
2. Developmental-biology priors most Kagglers lack: at the imaged zebrafish stage, quantify
   cell-cycle/division timing, near-symmetric daughter displacement, tissue-level coherent motion
   (neighbor cells move together), density gradients. Which encode as hard priors / features /
   regularizers that generalize across embryos (the 2-embryo CV trap)?
3. Imaging-physics priors: DaXi anisotropy (~450nm lateral, ~2um axial), PSF, photobleaching over
   time. What preprocessing (registration, white-tophat sigma=20, quantile 0.005/0.99999 per the
   Ultrack paper) demonstrably transfers?

## DELIVERABLE
Ranked list of ASYMMETRIC EDGES, each with: the edge, EV, STATUS (permitted / needs-rules-check /
forbidden-DQ), what to verify, and the concrete first experiment to test it. Lead with the single
highest-EV permitted edge. Explicitly separate "legal weapons" from "would forfeit the prize" so we
never waste a cycle on the latter. End with: if you had to bet ONE non-obvious edge that takes us
from ~0.842 to the lead on the PRIVATE set, what is it and why.
