r"""THE DIVISION-VERIFIER SUBSTRATE - population census, leakage contract, counterfactuals.

LEVER-0040 / PKT-0032. This module builds the DATASET a false-fork verifier could be trained on,
and the leakage contract that makes a fold-0 judgement honest. It trains nothing.

WHY A VERIFIER AND NOT A GENERATOR
----------------------------------
FACT-0375's scope records 5 division true positives against 67 false positives on fold 0, on a
base of 26 annotated divisions - the false-fork population is 13x the true one, so REJECTION has
far more mass to learn from than generation. FACT-0362 separately measured that the division lane
is DATA-limited: HOCT's fork head trained on 110 positives and published zero recall. A new
generator head is not the missing piece.

THE MATCHER IS NOT RE-INVENTED
------------------------------
Every population here is enumerated by the OFFICIAL scorer, not by a second matcher:

  * per-GT-division outcome, TP forks and FP forks come from
    ``tracking_cellmot.division_metrics.score_divisions`` itself - the same function
    ``scripts/win_bet/div_reach_steal.py`` calls, whose ``fp_forks`` set IS the metric's FP tally
    (``evaluate_divisions`` returns ``len(result.fp_forks)``);
  * the FP taxonomy (evaluable / cross-component / malformed / considered) comes from the
    scorer's own ``_pred_division_fork_sets``;
  * the pred->GT node correspondence comes from the scorer's own ``_match_full`` at the official
    7 um radius with scale (1.625, 0.40625, 0.40625) - the same one-to-one rule behind
    FACT-0354/0355/0357/0381 and the parent surface in ``assoc_parent_dataset.py``.

Every crop is reconciled against ``tracking_cellmot.metrics.evaluate`` before a row is written.
A disagreement RAISES. A census that cannot reproduce the official counts is not a census.

THE LEAKAGE CONTRACT (LEVER-0040's leakage rule IS the lever, not a caveat on it)
--------------------------------------------------------------------------------
Fold 0 is embryo 44b6 held out; fold 1 is 6bba held out (FACT-0378 scope). Therefore:

  judging fold 0  ->  train on fold 1 rows + external rows.  NO fold-0 row, positive OR negative,
                      naturally occurring OR counterfactual, may enter that model.
  judging fold 1  ->  train on fold 0 rows + external rows.
  external rows   ->  Zebrahub (FACT-0296), a different organism-stage corpus with no 44b6/6bba
                      content at all, so they are admissible on both sides.

Two mechanical guards, because a written rule is not a lock:
  1. every row carries ``fold`` and ``embryo``; ``training_rows_for(judged_fold)`` is the ONLY
     sanctioned selector and it excludes the judged fold by construction;
  2. ``assert_no_leak`` re-checks the selection and RAISES on any row from the judged fold, and
     ``FEATURES`` is a closed list that contains no identifier - ``fold``, ``embryo``, ``crop``,
     ``t`` and the node ids are carried as metadata and are refused as features.

FEATURES CARRY NO FOLD-WIDE STATISTIC
-------------------------------------
Every feature is a function of ONE crop's own node cloud and edge set, computable at inference
with no labels and no cross-crop aggregate. The formula set is the one already committed in
``scripts/win_bet/phaseb_h1g_features.py`` (its Zebrahub AUCs are recorded in FACT-0296:
parent_midpoint 0.967, daughter_angle 0.838), ported here because that module reads the E0c
proposer cache while this one reads a scored champion export - different node ids, same geometry.
Deliberately ABSENT: any z-score or quantile taken over a fold, any crop-level label rate, and
the crop/embryo identity itself.

COUNTERFACTUAL NEGATIVES ARE REQUIRED
-------------------------------------
A verifier trained only on the forks the pipeline happens to emit learns the generator's habits.
So negatives are constructed as well as collected, each tagged with ``neg_kind`` so a later
analysis can separate a learned physics from a learned artifact of the generator.

Usage
-----
  .venv\Scripts\python.exe scripts\win_bet\divverify_dataset.py census ^
      --csv C:/temp/p28_f0/loeo_split0_champion.csv.gz --fold 0 --out-dir C:/temp/divverify
  .venv\Scripts\python.exe scripts\win_bet\divverify_dataset.py crossfit --out-dir C:/temp/divverify
  .venv\Scripts\python.exe scripts\win_bet\divverify_dataset.py viability --out-dir C:/temp/divverify
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")
ROOT = next(_p for _p in Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

SCALE = (1.625, 0.40625, 0.40625)   # z, y, x um per level-0 voxel (biotrack.metric.DEFAULT_SCALE)
EPS = 1e-9
DENSITY_RADIUS_UM = 15.0

# The embryo held OUT on each fold (FACT-0378 scope: fold0 held_out 44b6, fold1 held_out 6bba).
FOLD_EMBRYO = {0: "44b6", 1: "6bba"}

# The closed feature list. Ported from scripts/win_bet/phaseb_h1g_features.py lines 128-152,
# minus the four census-only columns that need the E0c proposer (flow_midpoint_residual, rank,
# resid_*/best_alt_gap, steal_required). NO IDENTIFIER APPEARS HERE - that is guard (2).
FEATURES_V1 = [
    "parent_midpoint_um",   # |midpoint(d1,d2) - mother|   strongest external feature (FACT-0296)
    "pd1_um", "pd2_um",     # mother->daughter distances, min/max ordered so d1/d2 order is inert
    "pd_ratio",             # arm symmetry
    "sister_um",            # daughter separation
    "cos_daughter_axis",    # alignment of (d2-d1) with the mother's recent velocity
    "cos_split_vs_flow",    # alignment of (midpoint-mother) with local flow
    "daughter_angle",       # cos of the angle subtended at the mother (antipodality, FACT-0182)
    "persist_d1", "persist_d2", "n_persist",
    "mother_track_age",
    "mother_speed_um",
    "vel_consistency",
    "local_density_t", "local_density_t1",
    "competing_parents",
    "bdist_um",
    "dt_d1", "dt_d2",       # frame gap mother->daughter; 1/1 for any coherent fork
    "out_degree",           # fork out-degree in the prediction (0 for a constructed triple)
]

# ------------------------------------------------------------------ PKT-0035 extension (v2)
# Six families the packet names, every one still a pure function of ONE crop's own node cloud and
# edge set, and every one symmetric in daughter order (tests/test_divverify_contract.py asserts
# that over the WHOLE of FEATURES, so a new asymmetric feature fails the suite).
#
# ASSOCIATION MARGINS are a DISTANCE-BASED PROXY and are labelled as one. The champion export
# carries no edge probability, so a margin cannot be read off the model. FACT-0271 measured that a
# pure-distance LAP reproduces the deployed linker to within 0.001, which is what makes a
# flow-corrected distance margin a faithful stand-in for the association margin rather than an
# unrelated quantity. It is still a proxy, and it is never described as the model's own margin.
#
# LOCALISATION CONFIDENCE (FACT-0380) is likewise a PROXY. The residual against the annotation is
# ground truth and does not exist at inference; crowding (nearest-neighbour spacing) and flow
# residual are the GT-free quantities that predict it.
FEATURES_V2 = [
    # daughter association margins - how contested each daughter's parent choice is
    "assoc_margin_min", "assoc_margin_max",      # 2nd-best minus mother, flow-corrected um
    "parent_rank_min", "parent_rank_max",        # mother's rank among candidate parents
    "mother_fwd_claims",                         # nodes at t+1 whose best parent is the mother
    # forward / backward agreement
    "fb_ratio_min", "fb_ratio_max",              # 1.0 iff the daughter IS the mother's forward pick
    # sister geometry, made scale-free (the FACT-0385 per-feature scale gap acts on lengths)
    "sister_over_nn", "mid_over_nn", "pd1_over_nn", "pd2_over_nn",
    "sister_over_flow", "sister_perp_frac",
    # trajectory divergence - do the two branches separate like daughters, or shadow each other
    "traj_div_2", "traj_div_3", "traj_len_min", "traj_len_max", "traj_min_sep_ratio",
    # localisation confidence proxies (FACT-0380)
    "nn_um_mother", "nn_um_dmin", "nn_um_dmax", "nn_ratio_mother",
    "flow_resid_min", "flow_resid_max",
    # temporal context
    "t_frac", "growth_ratio", "forks_at_t_excl", "frames_remaining",
]

FEATURES = FEATURES_V1 + FEATURES_V2

# ROUTE-READING FEATURES - the fix for the trap PKT-0032 DIAGNOSED but did not disarm.
# FACT-0385: pooling GT-derived tuples with emitted forks inflated the unseen-fold AUC to 0.8006
# against a deployment view of 0.5254, because out_degree and competing_parents read the
# CONSTRUCTION ROUTE rather than the physics - an emitted fork has out_degree>=2 and
# competing_parents==0 BY CONSTRUCTION, a GT-derived triple usually does not. Those features are
# not wrong, they are unusable in any fit whose positives and negatives arrive by different routes.
# Removing them is a MECHANISM, not an assertion, and `route_audit` tests it by trying to predict
# the route from the surviving features.
ROUTE_READING = (
    "out_degree",          # >=2 for an emitted fork, whatever m happens to have otherwise
    "competing_parents",   # 0 for an emitted fork by definition
    "dt_d1", "dt_d2",      # 1/1 for an emitted fork AND for a GT tuple - carries nothing, and is
                           # constant in the external corpus too
    "forks_at_t_excl",     # excludes m, but an emitted fork's frame still differs systematically
)
ROUTE_NEUTRAL = [f for f in FEATURES if f not in ROUTE_READING]

# ROUTE_READING_STRICT - what the PKT-0035 route audit actually found, and the MECHANISM behind it.
# Removing out_degree and competing_parents was not enough. With the label held fixed, an emitted
# true fork is still separated from a GT-derived tuple at AUC 0.98 by `traj_div_2`, and a natural
# false fork from a constructed negative at AUC 0.98 by `assoc_margin_min`. Neither is computed
# FROM the mother-daughter edges, so neither was caught by the first pass - but both are proxies
# for the LINKER'S DECISION:
#   * the association-margin family asks "is this mother the daughter's best available parent";
#     the pipeline linked the triple precisely when the answer was yes, so the margin's SIGN is
#     very nearly the route;
#   * the trajectory family follows each daughter's onward chain, and a daughter the pipeline
#     never attached to this mother more often has no usable chain, so the -1 sentinel pattern
#     carries the route;
#   * persistence and the daughters' own flow residual carry the same sentinel structure.
# THE CONSEQUENCE IS STRUCTURAL, NOT COSMETIC. The GT-derived positives are exactly the divisions
# the pipeline did NOT link, so any feature measuring how well the association layer supports a
# triple must separate them from the emitted true forks. The enlarged positive class and the
# association-margin / trajectory features are therefore MUTUALLY EXCLUSIVE, and this list is what
# survives if the enlarged class is chosen.
ROUTE_READING_STRICT = ROUTE_READING + (
    "assoc_margin_min", "assoc_margin_max", "parent_rank_min", "parent_rank_max",
    "mother_fwd_claims", "fb_ratio_min", "fb_ratio_max",
    "traj_div_2", "traj_div_3", "traj_len_min", "traj_len_max", "traj_min_sep_ratio",
    "persist_d1", "persist_d2", "n_persist", "flow_resid_min", "flow_resid_max",
)
ROUTE_NEUTRAL_STRICT = [f for f in FEATURES if f not in ROUTE_READING_STRICT]

IDENTIFIER_COLUMNS = ("fold", "embryo", "crop", "mother", "d1", "d2", "t",
                      "label", "neg_kind", "source", "official_label", "derived_from")

# The subset of FEATURES the EXTERNAL corpus can also supply. dt_d1/dt_d2 and out_degree are
# constants in the external table by construction (every external triple is a proposed t->t+1
# fork), so keeping them would hand any model a perfect in-domain/external discriminator - a
# DOMAIN artifact that reads as signal. They are dropped from any mixed fit.
# PINNED TO V1: the v2 features have no counterpart in the external event table, so the external
# corpus can only ever be mixed in the v1 space. That is itself a finding about the plank and not
# a reason to weaken the check - FACT-0385 already made Zebrahub inadmissible for this lever.
EXTERNAL_SHARED = [f for f in FEATURES_V1 if f not in ("dt_d1", "dt_d2", "out_degree")]
EXTERNAL_GLOB = "_evidence/agent_runs/agent4/external_events/*.parquet"
# THE EXTERNAL PLANK IS SHUT (PKT-0035, 2026-08-30). FACT-0385 measured Zebrahub anti-aligned with
# this surface - pooled AUC 0.179, an in-domain fit collapsing 0.80 -> 0.21 when its rows were
# mixed in - and LEVER-0040 forbids their use until that is solved and the solution demonstrated.
# PKT-0032's crossfit map still declared `train_external: true`, written before its own probe
# measured the anti-alignment. One constant now decides it, and every consumer reads it, so a
# stale declaration cannot quietly re-admit the rows.
EXTERNAL_ADMISSIBLE = False
# Length-scaled features: the ones FACT-0296's ~1.6x per-embryo scale difference acts on.
LENGTH_FEATURES = ("parent_midpoint_um", "pd1_um", "pd2_um", "sister_um", "mother_speed_um",
                   "vel_consistency", "bdist_um")


# ============================================================ crop context and features
class CropContext:
    """One crop's node cloud and edge set in um, with the local-flow field.

    Everything a feature needs and nothing else. Built from the SCORED EXPORT, so it is exactly
    what a verifier would see at inference: no ground truth, no probabilities, no fold identity.
    """

    def __init__(self, node_ids: np.ndarray, t: np.ndarray, pos_um: np.ndarray,
                 edges: list[tuple[int, int]]):
        from scipy.spatial import cKDTree

        self.idx = {int(v): i for i, v in enumerate(node_ids)}
        self.node_ids = node_ids
        self.t = t
        self.pos = pos_um
        self.parent_of: dict[int, int] = {}
        self.children_of: dict[int, set[int]] = defaultdict(set)
        for a, b in edges:
            self.parent_of[int(b)] = int(a)
            self.children_of[int(a)].add(int(b))

        self.by_t: dict[int, list[int]] = defaultdict(list)
        for i, a in enumerate(t):
            self.by_t[int(a)].append(i)
        self.trees = {a: cKDTree(pos_um[ii]) for a, ii in self.by_t.items()}
        self.lo, self.hi = pos_um.min(axis=0), pos_um.max(axis=0)

        # local flow: per frame, the displacement field of the crop's own single-child edges.
        src_by_t: dict[int, list[int]] = defaultdict(list)
        d_by_t: dict[int, list[np.ndarray]] = defaultdict(list)
        for a, b in edges:
            ia, ib = self.idx.get(int(a)), self.idx.get(int(b))
            if ia is None or ib is None:
                continue
            src_by_t[int(t[ia])].append(ia)
            d_by_t[int(t[ia])].append(pos_um[ib] - pos_um[ia])
        alld = [d for v in d_by_t.values() for d in v]
        self.gmed = np.median(np.stack(alld), axis=0) if alld else np.zeros(3)
        self.darr, self.ktree, self.fmed = {}, {}, {}
        for a, ds in d_by_t.items():
            self.darr[a] = np.stack(ds)
            self.fmed[a] = np.median(self.darr[a], axis=0)
            self.ktree[a] = cKDTree(pos_um[src_by_t[a]]) if len(src_by_t[a]) >= 4 else None

        # ---------------- PKT-0035: crowding, frame occupancy and the parent-cost cache.
        # All per-CROP, all inference-available. Per-crop normalisation is explicitly allowed by
        # the leakage contract (the crop is present at inference); per-FOLD is not, and none is
        # taken here.
        self.tmax = int(t.max()) if len(t) else 0
        self.n_at = {a: len(ii) for a, ii in self.by_t.items()}
        self.forks_at: dict[int, int] = defaultdict(int)
        for a_id, kids in self.children_of.items():
            ia = self.idx.get(int(a_id))
            if ia is not None and len(kids) >= 2:
                self.forks_at[int(t[ia])] += 1
        self.nn_um = np.full(len(node_ids), -1.0)
        self.nn_med: dict[int, float] = {}
        for a, ii in self.by_t.items():
            if len(ii) < 2:
                for i in ii:
                    self.nn_um[i] = -1.0
                self.nn_med[a] = -1.0
                continue
            dd, _jj = self.trees[a].query(pos_um[ii], k=2)
            self.nn_um[np.asarray(ii)] = dd[:, 1]
            self.nn_med[a] = float(np.median(dd[:, 1]))
        self._pcost: dict[int, dict[int, float]] = {}
        self._best_parent: dict[int, int | None] = {}

    def flow_at(self, i: int, a: int) -> np.ndarray:
        kt = self.ktree.get(a)
        if kt is not None:
            _, jj = kt.query(self.pos[i], k=min(16, self.darr[a].shape[0]))
            return np.median(self.darr[a][np.atleast_1d(jj)], axis=0)
        return self.fmed.get(a, self.gmed)

    def track_age(self, m: int) -> int:
        n, cur, seen = 0, m, set()
        while cur in self.parent_of and cur not in seen and n < 64:
            seen.add(cur)
            cur = self.parent_of[cur]
            n += 1
        return n

    # ------------------------------------------------------- PKT-0035 association machinery
    def parent_costs(self, di: int, k: int = 8) -> dict[int, float]:
        """Flow-corrected cost of every plausible parent of node index ``di``, keyed by index.

        cost(n) = | pos[d] - (pos[n] + flow_at(n)) |. This is a DISTANCE PROXY for the association
        cost, not the model's own score - the export carries no probability. FACT-0271 measured
        that a pure-distance LAP reproduces the deployed linker to within 0.001, which is the
        licence for the proxy. Cached per node, so the whole crop costs O(N) queries.
        """
        got = self._pcost.get(di)
        if got is not None:
            return got
        a = int(self.t[di]) - 1
        tree = self.trees.get(a)
        out: dict[int, float] = {}
        if tree is not None:
            ii = self.by_t[a]
            kk = min(k, len(ii))
            _dd, jj = tree.query(self.pos[di], k=kk)
            for j in np.atleast_1d(jj):
                c = int(ii[int(j)])
                out[c] = float(np.linalg.norm(self.pos[di] - (self.pos[c] + self.flow_at(c, a))))
        self._pcost[di] = out
        return out

    def best_parent(self, di: int) -> int | None:
        got = self._best_parent.get(di, 0)
        if got != 0:
            return got
        costs = self.parent_costs(di)
        best = min(costs, key=lambda c: (costs[c], c)) if costs else None
        self._best_parent[di] = best
        return best

    def margin_and_rank(self, di: int, mi: int) -> tuple[float, int]:
        """(2nd-best cost minus the mother's cost, the mother's rank) for daughter index ``di``.

        The mother is inserted if the k-nearest scan missed it, so a distant mother is ranked
        honestly rather than silently dropped.
        """
        costs = dict(self.parent_costs(di))
        a = int(self.t[di]) - 1
        if mi not in costs:
            costs[mi] = float(np.linalg.norm(self.pos[di] - (self.pos[mi] + self.flow_at(mi, a))))
        cm = costs[mi]
        others = [v for c, v in costs.items() if c != mi]
        margin = (min(others) - cm) if others else 0.0
        rank = sum(1 for v in others if v < cm)
        return float(margin), int(rank)

    def forward_pick(self, mi: int, k: int = 8) -> tuple[int | None, float]:
        """The mother's own forward choice at t+1: (node index, its flow-corrected cost)."""
        a = int(self.t[mi])
        tree = self.trees.get(a + 1)
        if tree is None:
            return None, -1.0
        ii = self.by_t[a + 1]
        kk = min(k, len(ii))
        proj = self.pos[mi] + self.flow_at(mi, a)
        dd, jj = tree.query(proj, k=kk)
        dd, jj = np.atleast_1d(dd), np.atleast_1d(jj)
        b = int(np.argmin(dd))
        return int(ii[int(jj[b])]), float(dd[b])

    def fwd_claims(self, mi: int, k: int = 12) -> int:
        """How many nodes at t+1 name this mother as their BEST parent. A real divider is claimed
        by two; a node that has become a magnet for a crowded neighbourhood is claimed by more."""
        a = int(self.t[mi])
        tree = self.trees.get(a + 1)
        if tree is None:
            return 0
        ii = self.by_t[a + 1]
        kk = min(k, len(ii))
        proj = self.pos[mi] + self.flow_at(mi, a)
        _dd, jj = tree.query(proj, k=kk)
        return int(sum(1 for j in np.atleast_1d(jj) if self.best_parent(int(ii[int(j)])) == mi))

    def chain(self, node: int, steps: int) -> list[int]:
        """The single-child forward chain from a node, stopping at a fork, a death or ``steps``.

        Route-neutral with respect to the triple being scored: it follows the DAUGHTER's own
        onward links, which exist whether or not the mother was linked to her.
        """
        out, cur = [], int(node)
        for _ in range(steps):
            kids = sorted(self.children_of.get(cur, ()))
            if len(kids) != 1:
                break
            cur = int(kids[0])
            out.append(cur)
        return out

    def neighbours_at(self, frame: int, centre: np.ndarray, k: int) -> list[int]:
        """The k nearest node ids at ``frame`` to ``centre`` (um). Used to CONSTRUCT negatives."""
        tree = self.trees.get(frame)
        if tree is None:
            return []
        ii = self.by_t[frame]
        k = min(k, len(ii))
        if k == 0:
            return []
        _d, jj = tree.query(centre, k=k)
        return [int(self.node_ids[ii[j]]) for j in np.atleast_1d(jj)]


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > EPS else np.zeros(3)


def triple_features(ctx: CropContext, m: int, d1: int, d2: int) -> dict | None:
    """The feature vector of ONE (mother, daughter, daughter) triple.

    A PURE FUNCTION of the triple and the crop cloud. Real forks, GT-derived positive tuples and
    constructed counterfactuals all go through this one function, so no feature can encode how a
    row was produced - which is the only way a later analysis can tell a learned physics from a
    learned artifact of the generator.
    """
    im, i1, i2 = ctx.idx.get(m), ctx.idx.get(d1), ctx.idx.get(d2)
    if im is None or i1 is None or i2 is None or d1 == d2 or m in (d1, d2):
        return None
    a = int(ctx.t[im])
    pm, p1, p2 = ctx.pos[im], ctx.pos[i1], ctx.pos[i2]
    mid = 0.5 * (p1 + p2)
    fl = ctx.flow_at(im, a)
    pd1 = float(np.linalg.norm(p1 - pm))
    pd2 = float(np.linalg.norm(p2 - pm))
    sis = float(np.linalg.norm(p2 - p1))
    par = ctx.parent_of.get(m)
    mv = pm - ctx.pos[ctx.idx[par]] if par is not None and par in ctx.idx else np.zeros(3)
    u1, u2 = _unit(p1 - pm), _unit(p2 - pm)
    tree_t, tree_t1 = ctx.trees.get(a), ctx.trees.get(a + 1)

    # ---------------------------------------------------------------- PKT-0035 v2 features
    # Everything below is symmetric in (d1, d2) - the pair is reduced by min/max or by a
    # quantity of the pair itself. tests/test_divverify_contract.py asserts that over all of
    # FEATURES, so an asymmetric addition fails the suite rather than leaking daughter order.
    mg1, rk1 = ctx.margin_and_rank(i1, im)
    mg2, rk2 = ctx.margin_and_rank(i2, im)
    fwd_i, fwd_c = ctx.forward_pick(im)
    def _fb(ii_: int) -> float:
        if fwd_i is None or fwd_c < 0:
            return -1.0
        this = float(np.linalg.norm(ctx.pos[ii_] - (pm + fl)))
        return float(fwd_c / (this + EPS))
    fb1, fb2 = _fb(i1), _fb(i2)

    nn_t = ctx.nn_med.get(a, -1.0)
    nn_t1 = ctx.nn_med.get(a + 1, -1.0)
    sc_t = nn_t if nn_t and nn_t > 0 else float("nan")
    sc_t1 = nn_t1 if nn_t1 and nn_t1 > 0 else float("nan")
    flmag = float(np.linalg.norm(fl))
    sis_vec = p2 - p1
    perp = sis_vec - np.dot(sis_vec, _unit(fl)) * _unit(fl) if flmag > EPS else sis_vec

    ch1, ch2 = ctx.chain(d1, 8), ctx.chain(d2, 8)
    def _sep(k: int) -> float:
        if len(ch1) < k or len(ch2) < k:
            return -1.0
        q1, q2 = ctx.idx.get(ch1[k - 1]), ctx.idx.get(ch2[k - 1])
        if q1 is None or q2 is None:
            return -1.0
        return float(np.linalg.norm(ctx.pos[q2] - ctx.pos[q1]) / (sis + EPS))
    seps = [_sep(k) for k in range(1, min(len(ch1), len(ch2)) + 1)]
    seps = [v for v in seps if v >= 0.0]

    def _flow_resid(dd: int) -> float:
        idd = ctx.idx.get(dd)
        kids = sorted(ctx.children_of.get(dd, ()))
        if idd is None or len(kids) != 1 or kids[0] not in ctx.idx:
            return -1.0
        ad = int(ctx.t[idd])
        step = ctx.pos[ctx.idx[kids[0]]] - ctx.pos[idd]
        return float(np.linalg.norm(step - ctx.flow_at(idd, ad)))
    fr1, fr2 = _flow_resid(d1), _flow_resid(d2)

    return {
        "mother": int(m), "d1": int(d1), "d2": int(d2), "t": a,
        "parent_midpoint_um": float(np.linalg.norm(mid - pm)),
        "pd1_um": min(pd1, pd2), "pd2_um": max(pd1, pd2),
        "pd_ratio": min(pd1, pd2) / (max(pd1, pd2) + EPS),
        "sister_um": sis,
        "cos_daughter_axis": float(np.dot(_unit(p2 - p1), _unit(mv))),
        "cos_split_vs_flow": float(np.dot(_unit(mid - pm), _unit(fl))),
        "daughter_angle": float(np.dot(u1, u2)),
        "persist_d1": int(bool(ctx.children_of.get(d1))),
        "persist_d2": int(bool(ctx.children_of.get(d2))),
        "n_persist": int(bool(ctx.children_of.get(d1))) + int(bool(ctx.children_of.get(d2))),
        "mother_track_age": ctx.track_age(m),
        "mother_speed_um": float(np.linalg.norm(mv)),
        "vel_consistency": float(np.linalg.norm(mv - fl)) if par is not None else -1.0,
        "local_density_t": len(tree_t.query_ball_point(pm, DENSITY_RADIUS_UM)) if tree_t is not None else 0,
        "local_density_t1": len(tree_t1.query_ball_point(mid, DENSITY_RADIUS_UM)) if tree_t1 is not None else 0,
        "competing_parents": int(ctx.parent_of.get(d1, m) != m) + int(ctx.parent_of.get(d2, m) != m),
        "bdist_um": float(min(np.min(pm - ctx.lo), np.min(ctx.hi - pm))),
        "dt_d1": int(ctx.t[i1]) - a,
        "dt_d2": int(ctx.t[i2]) - a,
        "out_degree": len(ctx.children_of.get(m, ())),
        # --- v2: daughter association margins (distance proxy, FACT-0271)
        "assoc_margin_min": min(mg1, mg2), "assoc_margin_max": max(mg1, mg2),
        "parent_rank_min": min(rk1, rk2), "parent_rank_max": max(rk1, rk2),
        "mother_fwd_claims": ctx.fwd_claims(im),
        # --- v2: forward / backward agreement
        "fb_ratio_min": min(fb1, fb2), "fb_ratio_max": max(fb1, fb2),
        # --- v2: sister geometry, scale-free against the crop's own cell spacing
        "sister_over_nn": float(sis / sc_t1) if sc_t1 == sc_t1 else -1.0,
        "mid_over_nn": float(np.linalg.norm(mid - pm) / sc_t) if sc_t == sc_t else -1.0,
        "pd1_over_nn": float(min(pd1, pd2) / sc_t1) if sc_t1 == sc_t1 else -1.0,
        "pd2_over_nn": float(max(pd1, pd2) / sc_t1) if sc_t1 == sc_t1 else -1.0,
        "sister_over_flow": float(sis / (flmag + EPS)),
        "sister_perp_frac": float(np.linalg.norm(perp) / (sis + EPS)),
        # --- v2: trajectory divergence
        "traj_div_2": _sep(2), "traj_div_3": _sep(3),
        "traj_len_min": min(len(ch1), len(ch2)), "traj_len_max": max(len(ch1), len(ch2)),
        "traj_min_sep_ratio": float(min(seps)) if seps else -1.0,
        # --- v2: localisation-confidence proxies (FACT-0380)
        "nn_um_mother": float(ctx.nn_um[im]),
        "nn_um_dmin": float(min(ctx.nn_um[i1], ctx.nn_um[i2])),
        "nn_um_dmax": float(max(ctx.nn_um[i1], ctx.nn_um[i2])),
        "nn_ratio_mother": float(ctx.nn_um[im] / sc_t) if sc_t == sc_t else -1.0,
        "flow_resid_min": min(fr1, fr2), "flow_resid_max": max(fr1, fr2),
        # --- v2: temporal context
        "t_frac": float(a / max(1, ctx.tmax)),
        "growth_ratio": float(ctx.n_at.get(a + 1, 0) / max(1, ctx.n_at.get(a, 1))),
        "forks_at_t_excl": int(ctx.forks_at.get(a, 0)
                               - (1 if len(ctx.children_of.get(m, ())) >= 2 else 0)),
        "frames_remaining": int(ctx.tmax - a),
    }


# ============================================================ scorer glue
def _ea_atlas():
    spec = importlib.util.spec_from_file_location("ea_atlas", ROOT / "scripts" / "win_bet" / "ea_atlas.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def crop_context(sub: pl.DataFrame) -> tuple[CropContext, list[tuple[int, int]]]:
    nodes = sub.filter(pl.col("row_type") == "node").sort("node_id")
    node_ids = nodes["node_id"].to_numpy().astype(np.int64)
    t = nodes["t"].to_numpy().astype(np.int64)
    zyx = nodes.select(["z", "y", "x"]).to_numpy().astype(np.float64)
    pos_um = zyx * np.asarray(SCALE, dtype=np.float64)
    e = sub.filter(pl.col("row_type") == "edge")
    edges = list(zip(e["source_id"].to_list(), e["target_id"].to_list()))
    return CropContext(node_ids, t, pos_um, edges), edges


def official_division_state(g, gt, dm) -> dict:
    """Everything the OFFICIAL scorer knows about divisions in one crop, in internal graph ids."""
    ds = dm.score_divisions(g, gt, SCALE, 7.0)
    evaluable, cross, malformed = dm._pred_division_fork_sets(g, gt, SCALE, 7.0)
    matched_pred = dm._match_full(g, gt, SCALE, 7.0)
    attrs = dm._matched_node_attrs(matched_pred)
    import tracksdata as td
    K = td.DEFAULT_ATTR_KEYS
    pred_to_gt = dict(zip(attrs[K.NODE_ID].to_list(), attrs[K.MATCHED_NODE_ID].to_list()))
    return {
        "scores": ds.scores, "tp_forks": ds.tp_forks, "fp_forks": ds.fp_forks,
        "evaluable": evaluable, "cross_component": cross, "malformed": malformed,
        "pred_to_gt": pred_to_gt,
    }


# ============================================================ stage 1 - population census
def census_crop(name: str, sub: pl.DataFrame, gt_geff: Path, fold: int, ea, dm, mods) -> dict:
    """One crop: the official division populations, every predicted fork, and the GT tuples.

    RECONCILIATION IS MANDATORY. The rows are only emitted if score_divisions and the official
    ``evaluate`` agree on TP/FP/FN for this crop.
    """
    load_graph, evaluate = mods
    g, i2s = ea.build_graph(sub)
    gt = load_graph(gt_geff)
    st = official_division_state(g, gt, dm)
    er = evaluate(g, gt, scale=SCALE, max_distance=7.0)

    tp = sum(st["scores"].values())
    fn = len(st["scores"]) - tp
    fp = len(st["fp_forks"])
    if (tp, fp, fn) != (er.division_tp, er.division_fp, er.division_fn):
        raise RuntimeError(
            f"{name}: census does not reconcile with the official scorer - "
            f"score_divisions {tp}/{fp}/{fn} vs evaluate "
            f"{er.division_tp}/{er.division_fp}/{er.division_fn}"
        )

    ctx, _edges = crop_context(sub)
    # internal -> submission ids for everything the scorer reports
    def s(n):
        return int(i2s[int(n)])

    import tracksdata as td
    K = td.DEFAULT_ATTR_KEYS
    all_forks = [int(n) for n in g.node_ids() if g.out_degree(n) >= 2]

    fork_rows: list[dict] = []
    for f in all_forks:
        sf = s(f)
        kids = sorted(ctx.children_of.get(sf, ()))
        if len(kids) < 2:
            raise RuntimeError(f"{name}: fork {sf} has out-degree>=2 in the graph but "
                               f"{len(kids)} children in the export - id map is broken")
        feats = triple_features(ctx, sf, kids[0], kids[1])
        if feats is None:
            raise RuntimeError(f"{name}: fork {sf} produced no feature row")
        official = ("tp_fork" if f in st["tp_forks"]
                    else "fp_fork" if f in st["fp_forks"] else "ignored")
        fork_rows.append({
            "fold": fold, "embryo": name.split("_")[0], "crop": name,
            "source": "pipeline_fork", "official_label": official,
            "label": 1 if official == "tp_fork" else 0,
            "neg_kind": "" if official == "tp_fork" else ("natural_fp" if official == "fp_fork" else "ignored_fork"),
            "derived_from": "",
            "metric_visible": official != "ignored",
            "fp_evaluable": bool(f in st["evaluable"]),
            "fp_cross_component": bool(f in st["cross_component"]),
            "fp_malformed": bool(f in st["malformed"]),
            "n_children": len(kids),
            **feats,
        })

    # --- GT-derived positive tuples: the fork configuration that WOULD be correct -------------
    # Enumerated through the scorer's own full matching, so a cell counted here is the same cell
    # counted in FACT-0359's "fully matched tuple" population.
    gt_to_pred = {}
    for p, gval in st["pred_to_gt"].items():
        if gval is not None and int(gval) != -1:
            gt_to_pred.setdefault(int(gval), int(p))
    gt_na = gt.node_attrs(attr_keys=[K.NODE_ID, "t"]).to_pandas()
    gt_t = dict(zip(gt_na[K.NODE_ID].astype(int), gt_na["t"].astype(int)))

    gtdiv_rows, pos_rows = [], []
    for divider in st["scores"]:
        children = [int(c) for c in gt.successors(int(divider))]
        pm, p1, p2 = (gt_to_pred.get(int(divider)),
                      gt_to_pred.get(children[0]) if len(children) > 0 else None,
                      gt_to_pred.get(children[1]) if len(children) > 1 else None)
        fully = pm is not None and p1 is not None and p2 is not None and len({pm, p1, p2}) == 3
        gtdiv_rows.append({
            "fold": fold, "embryo": name.split("_")[0], "crop": name,
            "divider": int(divider), "t": int(gt_t[int(divider)]),
            "scored": int(st["scores"][divider]),
            "divider_matched": pm is not None,
            "children_matched": int(p1 is not None) + int(p2 is not None),
            "fully_matched": bool(fully),
        })
        if fully:
            feats = triple_features(ctx, s(pm), s(p1), s(p2))
            if feats is not None:
                pos_rows.append({
                    "fold": fold, "embryo": name.split("_")[0], "crop": name,
                    "source": "gt_tuple", "official_label": "gt_positive",
                    "label": 1, "neg_kind": "", "derived_from": f"gtdiv:{int(divider)}",
                    "metric_visible": True,
                    "fp_evaluable": False, "fp_cross_component": False, "fp_malformed": False,
                    "n_children": len(ctx.children_of.get(s(pm), ())),
                    **feats,
                })

    # --- PKT-0035: the association-error surface, in SUBMISSION ids -------------------------
    # s2gt is the scorer's own one-to-one matching, re-keyed; gt_parent comes from the GT edge
    # table, not from a second traversal, so "the linker got this edge wrong" is the GT's verdict.
    s2gt = {s(p): int(gv) for p, gv in st["pred_to_gt"].items()
            if gv is not None and int(gv) != -1}
    gte = gt.edge_attrs(attr_keys=[K.EDGE_SOURCE, K.EDGE_TARGET]).to_pandas()
    gt_parent = {int(t_): int(s_) for s_, t_ in zip(gte[K.EDGE_SOURCE], gte[K.EDGE_TARGET])}
    from tracking_cellmot import metrics as metrics_mod
    fp_edges = charged_fp_edges(g, gt, i2s, metrics_mod)
    if len(fp_edges) != er.edge_fp:
        raise RuntimeError(
            f"{name}: charged FP edge set has {len(fp_edges)} members but the official scorer "
            f"reports edge_fp={er.edge_fp} - the association surface does not reconcile"
        )
    mis_by_t = mislinked_by_frame(ctx, fp_edges)
    # The strictly-provable subclass, measured rather than assumed: both endpoints matched and
    # the GT names a different parent. Reported so the difference between the two definitions is
    # visible in the census instead of buried in a docstring.
    strict = sum(1 for a_, b_ in fp_edges
                 if a_ in s2gt and b_ in s2gt and gt_parent.get(s2gt[b_]) != s2gt[a_])

    def is_true_pair(mm: int, nn: int) -> bool:
        gm, gn = s2gt.get(int(mm)), s2gt.get(int(nn))
        return gm is not None and gn is not None and gt_parent.get(gn) == gm

    return {
        "crop": name, "ctx": ctx, "forks": fork_rows, "gt_tuples": pos_rows,
        "gtdiv": gtdiv_rows, "mis_by_t": mis_by_t, "fp_edges": fp_edges,
        "is_true_pair": is_true_pair,
        "n_mislinked": int(sum(len(v) for v in mis_by_t.values())),
        "n_mislinked_strict": int(strict),
        "n_matched_pred": len(s2gt),
        "counts": {"division_tp": tp, "division_fp": fp, "division_fn": fn,
                   "gt_divisions": len(st["scores"]), "pred_forks": len(all_forks),
                   "forks_ignored": sum(1 for r in fork_rows if r["official_label"] == "ignored")},
    }


# ============================================================ stage 3 - counterfactual negatives
def counterfactual_rows(ctx: CropContext, seed_rows: list[dict], crop: str, fold: int,
                        embryo: str, rng: np.random.Generator, per_seed: int = 2) -> list[dict]:
    """Negatives the GENERATOR DOES NOT NATURALLY PRODUCE, built from confirmed positives.

    Three constructions, each tagged so a later analysis can separate a learned physics from a
    learned artifact of the pipeline:

      cf_wrong_parent      the two real daughters kept, the mother replaced by a NEARBY node at
                           the same frame. The daughters are perfectly plausible; the parent is
                           wrong. This is the case a verifier trained on natural FPs never sees,
                           because the ILP rarely offers a competing parent for a real sister pair.
      cf_wrong_sister      the real mother and ONE real daughter kept, the other daughter replaced
                           by a nearby non-sister at the same frame. Half-true forks are the
                           hardest class and the one the metric punishes hardest, because a fork
                           with one correct branch still scores 0 and still counts as an FP.
      cf_temporal_skip     one daughter replaced by a node one frame LATER - a fork that spans two
                           time gaps and is therefore physically impossible.
      cf_temporal_same     one daughter replaced by a node in the MOTHER's own frame - a fork with
                           no time advance at all.

    Every constructed triple goes through the same ``triple_features``; nothing marks it as
    synthetic in the feature space.
    """
    out: list[dict] = []
    for src in seed_rows:
        m, d1, d2, tm = int(src["mother"]), int(src["d1"]), int(src["d2"]), int(src["t"])
        base = {"fold": fold, "embryo": embryo, "crop": crop, "source": "counterfactual",
                "official_label": "constructed_negative", "label": 0, "metric_visible": False,
                "fp_evaluable": False, "fp_cross_component": False, "fp_malformed": False,
                "derived_from": f"{src['source']}:{m}:{d1}:{d2}"}

        # --- wrong parent, plausible daughters
        alt_m = [n for n in ctx.neighbours_at(tm, ctx.pos[ctx.idx[m]], per_seed + 4) if n != m]
        for n in alt_m[:per_seed]:
            f = triple_features(ctx, n, d1, d2)
            if f is not None:
                out.append({**base, "neg_kind": "cf_wrong_parent",
                            "n_children": len(ctx.children_of.get(n, ())), **f})

        # --- correct parent, wrong sister (replace d2, then d1)
        for keep, drop in ((d1, d2), (d2, d1)):
            td_ = int(ctx.t[ctx.idx[drop]])
            alt = [n for n in ctx.neighbours_at(td_, ctx.pos[ctx.idx[drop]], per_seed + 4)
                   if n not in (d1, d2, m)]
            for n in alt[:per_seed]:
                f = triple_features(ctx, m, keep, n)
                if f is not None:
                    out.append({**base, "neg_kind": "cf_wrong_sister",
                                "n_children": len(ctx.children_of.get(m, ())), **f})

        # --- temporally incoherent
        for frame, kind in ((tm + 2, "cf_temporal_skip"), (tm, "cf_temporal_same")):
            alt = [n for n in ctx.neighbours_at(frame, ctx.pos[ctx.idx[d2]], per_seed + 4)
                   if n not in (d1, d2, m)]
            for n in alt[:per_seed]:
                f = triple_features(ctx, m, d1, n)
                if f is not None:
                    out.append({**base, "neg_kind": kind,
                                "n_children": len(ctx.children_of.get(m, ())), **f})
    return out


# ------------------------------------------------------------ PKT-0035 - the fifth negative class
def charged_fp_edges(g, gt, i2s, metrics_mod) -> set[tuple[int, int]]:
    """The edges the OFFICIAL metric charges as false positives, in SUBMISSION ids.

    Read from the scorer's own ``_evaluate_matched_graph`` - the function whose
    ``pred_valid - MATCHED_EDGE_MASK`` difference IS ``EvaluationResult.edge_fp`` - rather than
    from a second matcher. Same discipline as ``_pred_division_fork_sets`` and ``_match_full``
    above: the metric's verdict, not our reconstruction of it.

    SAY WHAT THIS MEANS, because the obvious reading is wrong. ``pred_valid`` is true when EITHER
    endpoint is a matched annotated cell, so a charged FP edge is usually a link from an annotated
    cell to an UNANNOTATED detection (FACT-0335: 99% of fold-0 FP edges have that shape), not two
    annotated cells wired to each other. The unannotated endpoint may well be a real cell. So
    "mistake" here means EXACTLY "a link the metric charges as false" - which is the operative
    notion for a verifier whose whole purpose is to move that metric - and it is NOT a claim that
    the biology was misread. The strictly-provable class (both endpoints matched, GT names another
    parent) was measured first and is nearly empty on this substrate, which is itself why this
    definition is the one that can be populated at all.
    """
    ea = metrics_mod._evaluate_matched_graph(g, gt)
    import tracksdata as td
    K = td.DEFAULT_ATTR_KEYS
    fp = ea.filter(pl.col("pred_valid") & ~pl.col(K.MATCHED_EDGE_MASK))
    return {(int(i2s[int(s_)]), int(i2s[int(t_)]))
            for s_, t_ in zip(fp[K.EDGE_SOURCE].to_list(), fp[K.EDGE_TARGET].to_list())}


def mislinked_by_frame(ctx: CropContext, fp_edges: set[tuple[int, int]]) -> dict[int, list[int]]:
    """Targets of charged FP edges, grouped by their own frame.

    "High-confidence" means the pipeline committed to the edge - it survived the ILP, the motion
    relink and every filter and is in the final graph - and the official metric charges it.

    This is the class that connects LEVER-0040 to the association lane, and the connection is a
    mechanism rather than an analogy: FACT-0371 measured that perfecting only the EDGES takes
    fold-0 division false positives from 67 to zero. If false forks are association errors wearing
    a fork's shape, then a sister drawn from the association layer's own charged mistakes is the
    negative a verifier must reject, and it is the one the natural FP fork population under-supplies.
    """
    out: dict[int, list[int]] = defaultdict(list)
    for _a, b in fp_edges:
        ib = ctx.idx.get(int(b))
        if ib is not None:
            out[int(ctx.t[ib])].append(int(b))
    return out


def assoc_mistake_rows(ctx: CropContext, fp_edges: set[tuple[int, int]], crop: str, fold: int,
                       embryo: str, is_true_pair, rng: np.random.Generator,
                       per_crop: int = 3) -> list[dict]:
    """``cf_assoc_mistake``: THE FORK A CHARGED MISLINK WOULD MAKE IF IT ACQUIRED A SISTER.

    Seeded from the association layer's own charged errors, not from the division seeds. The first
    version of this constructor drew a substitute sister from the mislinked nodes at a division's
    own frame and produced FOUR rows on fold 0 - the charged mislinks and the annotated divisions
    barely co-occur in space and time. That is a finding, and it is recorded rather than tuned
    away; but four rows is not a class, so the construction was moved to where the population is.

    For each charged FP edge ``a -> b`` one frame apart, the triple is ``(a, b, n)`` with ``n`` the
    nearest other node at b's frame. Mothers that are ALREADY emitted forks are skipped, because
    those triples are the natural FP population and would be counted twice.

    WHY THIS CLASS EXISTS. FACT-0371 measured that perfecting only the EDGES takes fold-0 division
    false positives from 67 to zero. If false forks are association errors wearing a fork's shape,
    a verifier has to reject the shape a charged mislink makes - and the natural FP fork population
    is only 67 examples of it. ``is_true_pair`` refuses any triple that is secretly a real GT
    division, so a "negative" cannot be a mislabelled positive.
    """
    out: list[dict] = []
    cands: list[tuple] = []
    for a, b in sorted(fp_edges):
        ia, ib = ctx.idx.get(int(a)), ctx.idx.get(int(b))
        if ia is None or ib is None:
            continue
        if int(ctx.t[ib]) != int(ctx.t[ia]) + 1:
            continue                                   # not a one-frame link; not a fork shape
        if len(ctx.children_of.get(int(a), ())) >= 2:
            continue                                   # already a natural fork row
        near = [q for q in ctx.neighbours_at(int(ctx.t[ib]), ctx.pos[ib], 4)
                if q not in (int(a), int(b))]
        if not near:
            continue
        n = int(near[0])
        if is_true_pair(int(a), int(b)) and is_true_pair(int(a), n):
            continue                                   # a real division - refuse, do not mislabel
        f = triple_features(ctx, int(a), int(b), n)
        if f is not None:
            cands.append((int(a), int(b), n, f))
    if len(cands) > per_crop:
        pick = sorted(rng.choice(len(cands), size=per_crop, replace=False).tolist())
        cands = [cands[i] for i in pick]
    for a, b, n, f in cands:
        out.append({
            "fold": fold, "embryo": embryo, "crop": crop, "source": "counterfactual",
            "official_label": "constructed_negative", "label": 0, "metric_visible": False,
            "fp_evaluable": False, "fp_cross_component": False, "fp_malformed": False,
            "neg_kind": "cf_assoc_mistake", "derived_from": f"charged_fp_edge:{a}:{b}",
            "n_children": len(ctx.children_of.get(a, ())), **f,
        })
    return out


# ============================================================ stage 2 - the leakage contract
def crossfit_map() -> dict:
    """The cross-fitting map, written down BEFORE any row is used to fit anything."""
    return {
        "schema_version": 1,
        "heartbeat": "DIVVERIFY_CROSSFIT_MAP",
        "folds": {"0": {"held_out_embryo": FOLD_EMBRYO[0], "role": "the promotion gate (FACT-0261)"},
                  "1": {"held_out_embryo": FOLD_EMBRYO[1], "role": "second direction, reported separately"}},
        "rule": (
            "A model judged on fold F is fitted ONLY on rows whose fold != F, plus external rows. "
            "This covers positives, natural negatives AND constructed counterfactuals alike - a "
            "counterfactual built from a fold-0 crop is a fold-0 row."
        ),
        "assignments": {
            "judge_fold_0": {"train_folds": [1], "train_external": EXTERNAL_ADMISSIBLE,
                             "forbidden_rows": "fold == 0"},
            "judge_fold_1": {"train_folds": [0], "train_external": EXTERNAL_ADMISSIBLE,
                             "forbidden_rows": "fold == 1"},
        },
        "external_admissibility": {
            "admissible": EXTERNAL_ADMISSIBLE,
            "changed_by": "PKT-0035, 2026-08-30",
            "was": ("PKT-0032 wrote this map with train_external: true, BEFORE its own viability "
                    "probe measured the anti-alignment. FACT-0385 then recorded pooled AUC 0.179 "
                    "and an in-domain fit collapsing 0.80 -> 0.21 when Zebrahub rows were mixed "
                    "in, and LEVER-0040 forbade their use until that is solved. The map had not "
                    "been updated to match, so a later reader would have found a declaration that "
                    "contradicts the lever. It is corrected here rather than left as a footnote."),
            "reopen_when": ("the per-feature scale gap is solved AND the solution is demonstrated "
                            "on this surface - not asserted. Until then EXTERNAL_ADMISSIBLE is "
                            "False and every consumer must pass an explicit override to use the "
                            "rows at all."),
        },
        "external": {
            "corpus": "Zebrahub (FACT-0296)",
            "path": "_evidence/agent_runs/agent4/external_events/*.parquet",
            "why_admissible": (
                "a different imaging corpus with no 44b6 or 6bba content, so it is clean for both "
                "folds; the standing warning is FACT-0378, where an auxiliary trained on all 199 "
                "movies became un-measurable by any LOEO control - external must mean OUTSIDE the "
                "199, which Zebrahub is."
            ),
            "caveat": (
                "FACT-0296: usable only with PER-EMBRYO SCALE NORMALISATION - shape matches but "
                "absolute scale differs about 1.6x - and it is Ultrack automated output, not "
                "curated GT."
            ),
        },
        "leak_hunt": {
            "identifier_features_refused": list(IDENTIFIER_COLUMNS),
            "fold_wide_statistics": (
                "NONE. Every feature is a function of one crop's own node cloud and edges. There "
                "is no z-score, quantile or rate taken over a fold, and no crop-level label rate."
            ),
            "per_crop_normalisation": (
                "allowed and inference-available (the crop is present at inference); "
                "per-FOLD normalisation is not, and is not used."
            ),
            "residual_risk": (
                "t (frame index) is carried as metadata, not a feature. dt_d1/dt_d2 are frame GAPS "
                "and carry no absolute time. local_density_* are absolute counts and could in "
                "principle differ systematically between embryos - that is a DOMAIN SHIFT, not a "
                "label leak, and it shows up as transfer failure rather than optimism."
            ),
        },
    }


def training_rows_for(table: pl.DataFrame, judged_fold: int) -> pl.DataFrame:
    """THE ONLY SANCTIONED SELECTOR. Everything not from the judged fold."""
    return table.filter(pl.col("fold") != judged_fold)


def assert_no_leak(train: pl.DataFrame, judged_fold: int) -> None:
    """Fail closed. A silent no-op here would void the whole dataset (PKT-0032 falsifier (b))."""
    bad = train.filter(pl.col("fold") == judged_fold)
    if bad.height:
        raise RuntimeError(
            f"LEAK: {bad.height} training rows come from fold {judged_fold}, which this model is "
            f"judged on. Crops: {sorted(set(bad['crop'].to_list()))[:5]}"
        )
    embryo = FOLD_EMBRYO[judged_fold]
    bad2 = train.filter(pl.col("embryo") == embryo)
    if bad2.height:
        raise RuntimeError(f"LEAK: {bad2.height} training rows come from held-out embryo {embryo}")
    leaked_features = [c for c in FEATURES if c in IDENTIFIER_COLUMNS]
    if leaked_features:
        raise RuntimeError(f"LEAK: identifier columns present in FEATURES: {leaked_features}")


# ============================================================ commands
def cmd_census(args) -> int:
    ea = _ea_atlas()
    from biotrack.metric import load_graph
    from biotrack.submission import read_submission
    from tracking_cellmot import division_metrics as dm
    from tracking_cellmot.metrics import evaluate

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = read_submission(ea.open_csv(Path(args.csv)))
    names = sorted(df["dataset"].unique().to_list())
    if args.only:
        want = [c.strip() for c in args.only.split(",") if c.strip()]
        missing = [c for c in want if c not in names]
        if missing:
            raise RuntimeError(f"--only names crops not in this export: {missing}")
        names = want
    if args.stride:
        names = names[::args.stride]
    if args.max_crops:
        names = names[: args.max_crops]
    bad = [n for n in names if not n.startswith(FOLD_EMBRYO[args.fold])]
    if bad:
        raise RuntimeError(
            f"fold {args.fold} must hold out embryo {FOLD_EMBRYO[args.fold]} but the export "
            f"contains {len(bad)} crops from another embryo, e.g. {bad[:3]} - the fold label "
            f"would be wrong and the leakage contract would be built on it"
        )
    print(f"DIVVERIFY_CENSUS fold={args.fold} embryo={FOLD_EMBRYO[args.fold]} crops={len(names)}", flush=True)

    rng = np.random.default_rng(20260829 + args.fold)
    rows, gtdiv, cfs = [], [], []
    totals = defaultdict(int)
    t0 = time.time()
    for i, name in enumerate(names, 1):
        gt_geff = Path(args.gt_dir) / f"{name}.geff"
        if not gt_geff.exists():
            raise RuntimeError(f"{name}: no GT at {gt_geff} - a skipped crop is a silent no-op")
        r = census_crop(name, df.filter(pl.col("dataset") == name), gt_geff, args.fold,
                        ea, dm, (load_graph, evaluate))
        rows += r["forks"] + r["gt_tuples"]
        gtdiv += r["gtdiv"]
        seeds = [x for x in r["forks"] if x["official_label"] == "tp_fork"] + r["gt_tuples"]
        cfs += counterfactual_rows(r["ctx"], seeds, name, args.fold, name.split("_")[0], rng,
                                   per_seed=args.per_seed)
        cfs += assoc_mistake_rows(r["ctx"], r["fp_edges"], name, args.fold, name.split("_")[0],
                                  r["is_true_pair"], rng, per_crop=args.assoc_per_crop)
        totals["mislinked_nodes"] += r["n_mislinked"]
        totals["mislinked_strict"] += r["n_mislinked_strict"]
        totals["matched_pred_nodes"] += r["n_matched_pred"]
        totals["seeds"] += len(seeds)
        for k, v in r["counts"].items():
            totals[k] += v
        print(f"  [{i}/{len(names)}] {name} gtdiv={r['counts']['gt_divisions']} "
              f"tp/fp/fn={r['counts']['division_tp']}/{r['counts']['division_fp']}/"
              f"{r['counts']['division_fn']} forks={r['counts']['pred_forks']} "
              f"cf={len(cfs)} ({time.time() - t0:.0f}s)", flush=True)

    # FAIL CLOSED. The fifth negative class is the one new construction in this packet; if it
    # produced nothing the run must SAY so rather than quietly shipping a four-class dataset that
    # looks exactly like PKT-0032's.
    n_assoc = sum(1 for r in cfs if r["neg_kind"] == "cf_assoc_mistake")
    if n_assoc == 0 and totals["seeds"] > 0:
        raise RuntimeError(
            f"fold {args.fold}: cf_assoc_mistake produced ZERO rows over {len(names)} crops from "
            f"{int(totals['seeds'])} seeds, with {int(totals['mislinked_nodes'])} charged FP edges "
            f"available. The pool filter is wrong - a silent no-op here would ship a dataset "
            f"identical to PKT-0032's while claiming a new negative class"
        )
    if totals["seeds"] == 0:
        print("DIVVERIFY_NO_SEEDS: this crop selection contains no annotated division at all, so "
              "no counterfactual of any kind can be built from it. Not an error on a truncated "
              "smoke run; on a COMPLETE fold it would be one.", flush=True)

    table = pl.DataFrame(rows + cfs)
    table.write_parquet(out / f"rows_f{args.fold}.parquet")
    pl.DataFrame(gtdiv).write_parquet(out / f"gtdiv_f{args.fold}.parquet")

    summary = {
        "schema_version": 1,
        "heartbeat": "DIVVERIFY_CENSUS_COMPLETE",
        "fold": args.fold, "embryo": FOLD_EMBRYO[args.fold], "csv": str(args.csv),
        "n_crops": len(names),
        "official_counts": {k: int(totals[k]) for k in
                            ("gt_divisions", "division_tp", "division_fp", "division_fn")},
        "pred_forks_total": int(totals["pred_forks"]),
        "pred_forks_ignored_by_metric": int(totals["forks_ignored"]),
        "association_surface": {
            "matched_pred_nodes": int(totals["matched_pred_nodes"]),
            "charged_fp_edges": int(totals["mislinked_nodes"]),
            "charged_fp_edges_both_endpoints_matched": int(totals["mislinked_strict"]),
            "note": ("charged_fp_edges is the metric's OWN edge-FP tally, reconciled per crop "
                     "against evaluate().edge_fp. The strict subclass - both endpoints matched "
                     "and the GT naming another parent - is reported beside it because "
                     "FACT-0335 says the charged population is dominated by links from an "
                     "annotated cell to an UNANNOTATED detection, and the two are not the same "
                     "claim"),
        },
        "feature_space": {"v1": len(FEATURES_V1), "v2_added": len(FEATURES_V2),
                          "total": len(FEATURES), "route_neutral": len(ROUTE_NEUTRAL),
                          "route_reading_excluded": list(ROUTE_READING)},
        "rows": {
            "pipeline_fork_tp": int(sum(1 for r in rows if r["official_label"] == "tp_fork")),
            "pipeline_fork_fp": int(sum(1 for r in rows if r["official_label"] == "fp_fork")),
            "pipeline_fork_ignored": int(sum(1 for r in rows if r["official_label"] == "ignored")),
            "gt_tuple_positive": int(sum(1 for r in rows if r["source"] == "gt_tuple")),
            "counterfactual": int(len(cfs)),
            "counterfactual_by_kind": {k: int(sum(1 for r in cfs if r["neg_kind"] == k))
                                       for k in sorted({r["neg_kind"] for r in cfs})},
        },
        "fp_taxonomy": {
            "evaluable": int(sum(1 for r in rows if r["official_label"] == "fp_fork" and r["fp_evaluable"])),
            "cross_component": int(sum(1 for r in rows if r["official_label"] == "fp_fork" and r["fp_cross_component"])),
            "malformed": int(sum(1 for r in rows if r["official_label"] == "fp_fork" and r["fp_malformed"])),
            "considered_only": int(sum(1 for r in rows if r["official_label"] == "fp_fork"
                                       and not (r["fp_evaluable"] or r["fp_cross_component"] or r["fp_malformed"]))),
        },
        "gt_tuples": {
            "gt_divisions": int(len(gtdiv)),
            "fully_matched": int(sum(1 for r in gtdiv if r["fully_matched"])),
            "scored": int(sum(r["scored"] for r in gtdiv)),
        },
        "outputs": {"rows": str(out / f"rows_f{args.fold}.parquet"),
                    "gtdiv": str(out / f"gtdiv_f{args.fold}.parquet")},
    }
    (out / f"census_f{args.fold}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def cmd_external(args) -> int:
    """Map the Zebrahub event table (FACT-0296) into this row schema, and MEASURE the scale gap.

    fold = -1, so ``training_rows_for`` admits these rows on both sides by construction and
    ``assert_no_leak`` can never mistake them for held-out content. FACT-0296's caveat is not
    restated, it is measured here on THIS feature set: the per-feature median ratio between the
    external positives and our own in-domain positives.
    """
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(ROOT.glob(EXTERNAL_GLOB))
    if not files:
        raise RuntimeError(f"no external corpus at {ROOT / EXTERNAL_GLOB} - FACT-0296 names it; "
                           f"an empty glob is not evidence that it is missing, check the path")
    frames = []
    for f in files:
        df = pl.read_parquet(f)
        missing = [c for c in EXTERNAL_SHARED if c not in df.columns]
        if missing:
            raise RuntimeError(f"{f.name}: external table lacks {missing} - the shared feature "
                               f"space cannot be built and no mixed fit is admissible")
        df = df.filter(pl.col("label") != "unlabeled")
        frames.append(df.select([
            pl.lit(-1, dtype=pl.Int64).alias("fold"),
            pl.col("embryo").cast(pl.Utf8),
            (pl.col("embryo") + pl.lit("_") + pl.col("t_lo").cast(pl.Utf8)).alias("crop"),
            pl.lit("external_zebrahub").alias("source"),
            pl.col("label").alias("official_label"),
            (pl.col("label") == "positive").cast(pl.Int64).alias("label"),
            pl.when(pl.col("label") == "positive").then(pl.lit(""))
              .otherwise(pl.lit("external_reliable_negative")).alias("neg_kind"),
            pl.col("cand_id").alias("derived_from"),
            pl.col("metric_visible"),
            pl.col("mother"), pl.col("d1"), pl.col("d2"), pl.col("t"),
            *[pl.col(c).cast(pl.Float64) for c in EXTERNAL_SHARED],
        ]))
    ext = pl.concat(frames, how="vertical")
    ext.write_parquet(out / "rows_external.parquet")

    scale_gap = {}
    ours = []
    for fold in (0, 1):
        p = out / f"rows_f{fold}.parquet"
        if p.exists():
            ours.append(pl.read_parquet(p).filter(pl.col("label") == 1))
    if ours:
        ours_pos = pl.concat(ours, how="vertical")
        ep = ext.filter(pl.col("label") == 1)
        for c in LENGTH_FEATURES:
            a = float(np.median(ours_pos[c].to_numpy())) if ours_pos.height else float("nan")
            b = float(np.median(ep[c].to_numpy())) if ep.height else float("nan")
            scale_gap[c] = {"ours_median": round(a, 4), "external_median": round(b, 4),
                            "ratio_external_over_ours": round(b / a, 3) if a else None}

    summary = {
        "schema_version": 1, "heartbeat": "DIVVERIFY_EXTERNAL_COMPLETE",
        "files": [f.name for f in files],
        "rows": int(ext.height),
        "positives": int(ext["label"].sum()),
        "reliable_negatives": int(ext.height - ext["label"].sum()),
        "embryos": sorted(set(ext["embryo"].to_list())),
        "shared_feature_space": EXTERNAL_SHARED,
        "dropped_from_shared": [f for f in FEATURES if f not in EXTERNAL_SHARED],
        "scale_gap_vs_our_positives": scale_gap,
        "caveat": ("FACT-0296: Zebrahub is Ultrack automated output, not curated GT, one embryo, "
                   "4 temporal blocks - and its absolute scale differs, which is why the ratios "
                   "above are measured rather than assumed."),
        "output": str(out / "rows_external.parquet"),
    }
    (out / "external.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def cmd_crossfit(args) -> int:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = crossfit_map()
    (out / "crossfit_map.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    print(json.dumps(m, indent=2))
    return 0


# ------------------------------------------------------------ stage 5 - viability probe
def _auc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank AUC with ties handled. Returns nan when a class is empty."""
    y = np.asarray(y).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s), dtype=np.float64)
    sv = np.asarray(s)[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _fit_logistic(x: np.ndarray, y: np.ndarray, iters: int = 400, lr: float = 0.2) -> np.ndarray:
    """Plain L2 logistic regression on standardised inputs. NOT a promoted model - a probe that
    answers PKT-0032 falsifier (d): is there any separating signal at all?"""
    xb = np.hstack([x, np.ones((len(x), 1))])
    w = np.zeros(xb.shape[1])
    pos = float(y.sum())
    neg = float(len(y) - pos)
    wpos = neg / max(pos, 1.0)
    sw = np.where(y == 1, wpos, 1.0)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(xb @ w, -30, 30)))
        gradient = xb.T @ (sw * (p - y)) / len(y) + 1e-3 * np.r_[w[:-1], 0.0]
        w -= lr * gradient
    return w


def cmd_viability(args) -> int:
    out = Path(args.out_dir)
    tables = []
    for fold in (0, 1):
        p = out / f"rows_f{fold}.parquet"
        if p.exists():
            tables.append(pl.read_parquet(p))
    if not tables:
        raise RuntimeError(f"no census rows in {out} - run `census` first")
    raw = pl.concat(tables, how="vertical")

    # Two reductions, both stated rather than silent.
    # (1) Forks the metric IGNORES are dropped from every fit. They sit on nodes the official
    #     matcher did not pair, or on GT nodes with no children (end of annotation), so the metric
    #     charges nothing for them - and calling them negatives would teach "not a division" about
    #     cells whose division status is simply unannotated.
    # (2) A GT tuple and the TP fork that recovered it can be the same (mother, d1, d2). Deduped
    #     so one division cannot be counted twice as a positive.
    table = raw.filter(pl.col("official_label") != "ignored").unique(
        subset=["crop", "mother", "d1", "d2", "label"], keep="first", maintain_order=True
    )

    report = {
        "schema_version": 1, "heartbeat": "DIVVERIFY_VIABILITY_COMPLETE",
        "class_balance": {}, "per_feature_auc": {}, "crossfit_probe": {},
        "external": {}, "falsifiers": {},
    }

    report["reductions"] = {
        "rows_before": int(raw.height), "rows_after": int(table.height),
        "dropped_ignored_forks": int(raw.filter(pl.col("official_label") == "ignored").height),
        "dropped_duplicate_triples": int(
            raw.filter(pl.col("official_label") != "ignored").height - table.height),
    }
    for fold in sorted(table["fold"].unique().to_list()):
        f = table.filter(pl.col("fold") == fold)
        g = raw.filter(pl.col("fold") == fold)
        report["class_balance"][f"fold_{fold}"] = {
            "positives_scored_fork": int(f.filter(pl.col("official_label") == "tp_fork").height),
            "positives_gt_tuple": int(f.filter(pl.col("official_label") == "gt_positive").height),
            "positives_total_deduped": int(f.filter(pl.col("label") == 1).height),
            "negatives_natural_fp": int(f.filter(pl.col("official_label") == "fp_fork").height),
            "negatives_constructed": int(f.filter(pl.col("source") == "counterfactual").height),
            "ignored_forks_excluded": int(g.filter(pl.col("official_label") == "ignored").height),
            "constructed_by_kind": {
                k: int(f.filter(pl.col("neg_kind") == k).height)
                for k in sorted(set(f.filter(pl.col("source") == "counterfactual")["neg_kind"].to_list()))
            },
        }

    # per-feature AUC, natural population only (positives vs natural FP forks), per fold
    for fold in sorted(table["fold"].unique().to_list()):
        f = table.filter((pl.col("fold") == fold) & (pl.col("source") != "counterfactual"))
        y = f["label"].to_numpy().astype(int)
        report["per_feature_auc"][f"fold_{fold}_natural"] = {
            c: round(_auc(y, f[c].to_numpy().astype(float)), 4) for c in FEATURES
        }
        # THE CLEAN SEPARABILITY TEST. Both classes are forks the pipeline actually emitted, so
        # no feature can separate them by PROVENANCE - which the mixed view above cannot say,
        # because its positives are mostly GT-derived triples the pipeline never linked and
        # out_degree/competing_parents then read the construction route, not the physics.
        fd = table.filter((pl.col("fold") == fold)
                          & pl.col("official_label").is_in(["tp_fork", "fp_fork"]))
        yd = fd["label"].to_numpy().astype(int)
        report["per_feature_auc"][f"fold_{fold}_deployment_forks_only"] = {
            c: round(_auc(yd, fd[c].to_numpy().astype(float)), 4) for c in FEATURES
        }

        # Counterfactuals, BY KIND. Pooling them hides which construction carries signal:
        # cf_wrong_parent keeps both real daughters, so anything that separates it from a
        # positive is division geometry rather than a property of the daughters.
        fpos = table.filter((pl.col("fold") == fold) & (pl.col("label") == 1))
        for kind in sorted(set(table.filter((pl.col("fold") == fold) &
                                            (pl.col("source") == "counterfactual"))["neg_kind"].to_list())):
            fk = pl.concat([fpos, table.filter((pl.col("fold") == fold) &
                                               (pl.col("neg_kind") == kind))], how="vertical")
            yk = fk["label"].to_numpy().astype(int)
            report["per_feature_auc"][f"fold_{fold}_{kind}"] = {
                c: round(_auc(yk, fk[c].to_numpy().astype(float)), 4) for c in FEATURES
            }
        fc = table.filter((pl.col("fold") == fold) &
                          ((pl.col("source") == "counterfactual") | (pl.col("label") == 1)))
        yc = fc["label"].to_numpy().astype(int)
        report["per_feature_auc"][f"fold_{fold}_counterfactual_pooled"] = {
            c: round(_auc(yc, fc[c].to_numpy().astype(float)), 4) for c in FEATURES
        }

    # external rows, if built AND admissible. The gate is a constant, not a comment: with
    # EXTERNAL_ADMISSIBLE False the rows are not loaded at all unless the caller passes
    # --allow-external, and the report says so either way.
    ext_path = out / "rows_external.parquet"
    allow_ext = bool(getattr(args, "allow_external", False)) or EXTERNAL_ADMISSIBLE
    report["external_gate"] = {
        "EXTERNAL_ADMISSIBLE": EXTERNAL_ADMISSIBLE,
        "override_passed": bool(getattr(args, "allow_external", False)),
        "external_rows_used": bool(allow_ext and ext_path.exists()),
        "why": ("FACT-0385: pooled AUC 0.179 and an in-domain fit collapsing 0.80 -> 0.21 when "
                "Zebrahub rows are mixed in. LEVER-0040 forbids their use until the alignment is "
                "solved AND the solution demonstrated"),
    }
    ext = pl.read_parquet(ext_path) if (allow_ext and ext_path.exists()) else None
    if ext is not None:
        report["external"] = {
            "rows": int(ext.height), "positives": int(ext["label"].sum()),
            "fold": sorted(set(ext["fold"].to_list())),
            "feature_space": EXTERNAL_SHARED,
        }

    # cross-fitted probe: fit on the OTHER fold, judge on this one. Leakage guard is mechanical.
    for judged in sorted(table["fold"].unique().to_list()):
        test = table.filter((pl.col("fold") == judged) & (pl.col("source") != "counterfactual"))
        arms: dict[str, tuple[pl.DataFrame, list[str]]] = {}
        indomain = training_rows_for(table, judged)
        if indomain.height:
            arms["in_domain_only"] = (indomain, FEATURES)
            # THE MATCHED-PROVENANCE ARM, and the one the deployment view is actually about.
            # Trained ONLY on forks the pipeline emitted on the other fold - tp_fork against
            # fp_fork - so the model cannot separate positives from negatives by construction
            # route. It is also where PKT-0032 falsifier (a) bites hardest, because that positive
            # count is the emitted-TP count, not the GT-division count.
            emitted = indomain.filter(pl.col("official_label").is_in(["tp_fork", "fp_fork"]))
            if emitted.height:
                arms["in_domain_emitted_forks_only"] = (emitted, FEATURES)
            # And the same rows plus the constructed counterfactuals, to ask whether the
            # counterfactuals ADD anything a natural-negatives-only fit does not have.
            cf = indomain.filter(pl.col("source") == "counterfactual")
            if emitted.height and cf.height:
                arms["emitted_forks_plus_counterfactuals"] = (
                    pl.concat([emitted, cf], how="vertical"), FEATURES)
        if ext is not None:
            cols = ["fold", "embryo", "crop", "label"] + EXTERNAL_SHARED

            def _cast(df):
                return df.select(cols).with_columns(
                    [pl.col(c).cast(pl.Float64) for c in EXTERNAL_SHARED])

            ext_c = _cast(ext)
            mixed = pl.concat([_cast(indomain), ext_c], how="vertical") \
                if indomain.height else ext_c
            arms["in_domain_plus_external"] = (mixed, EXTERNAL_SHARED)
            arms["external_only"] = (ext_c, EXTERNAL_SHARED)
        if not arms:
            report["crossfit_probe"][f"judge_fold_{judged}"] = {"skipped": "no training rows"}
            continue
        entry = {}
        for arm, (train, cols) in arms.items():
            assert_no_leak(train, judged)
            xtr = train.select(cols).to_numpy().astype(float)
            ytr = train["label"].to_numpy().astype(int)
            if ytr.sum() == 0 or ytr.sum() == len(ytr):
                entry[arm] = {"skipped": "one class empty in training rows"}
                continue
            mu, sd = xtr.mean(0), xtr.std(0) + 1e-9
            w = _fit_logistic((xtr - mu) / sd, ytr)
            xte = (test.select(cols).to_numpy().astype(float) - mu) / sd
            s = np.hstack([xte, np.ones((len(xte), 1))]) @ w
            yte = test["label"].to_numpy().astype(int)
            entry[arm] = {
                "train_rows": int(train.height), "train_positives": int(ytr.sum()),
                "train_folds": sorted(set(train["fold"].to_list())),
                "n_features": len(cols),
                "auc_on_unseen_fold": round(_auc(yte, s), 4),
            }
            # THE DEPLOYMENT VIEW. The test set above includes GT-derived tuples the pipeline
            # never linked; the decision a verifier actually makes is over the forks the pipeline
            # DID emit and the metric DOES charge - tp_fork against fp_fork, nothing else. With
            # 5 true positives on fold 0 that is exactly LEVER-0040 falsifier (a) territory, so
            # the rejection curve is reported, not just an AUC.
            mask = np.array([lab in ("tp_fork", "fp_fork")
                             for lab in test["official_label"].to_list()])
            if mask.any():
                yd, sd_ = yte[mask], s[mask]
                order = np.argsort(sd_, kind="stable")   # lowest score = most rejectable
                rejected_fp_before_first_tp, seen_fp = 0, 0
                for j in order:
                    if yd[j] == 1:
                        break
                    seen_fp += 1
                rejected_fp_before_first_tp = seen_fp
                cum_fp, lost_tp, at_one_tp = 0, 0, 0
                for j in order:
                    if yd[j] == 1:
                        lost_tp += 1
                        if lost_tp == 1:
                            at_one_tp = cum_fp
                        if lost_tp == 2:
                            break
                    else:
                        cum_fp += 1
                entry[arm]["deployment_view"] = {
                    "forks_charged_by_the_metric": int(mask.sum()),
                    "true_positive_forks": int(yd.sum()),
                    "false_positive_forks": int(len(yd) - yd.sum()),
                    "auc": round(_auc(yd, sd_), 4),
                    "fp_rejected_before_losing_any_tp": int(rejected_fp_before_first_tp),
                    "fp_rejected_before_losing_a_second_tp": int(at_one_tp),
                }
        entry["test_rows"] = int(test.height)
        entry["test_positives"] = int(test["label"].sum())
        entry["note"] = "logistic probe on standardised features; a probe, never a promoted model"
        report["crossfit_probe"][f"judge_fold_{judged}"] = entry

    (out / "viability.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("census")
    c.add_argument("--csv", required=True)
    c.add_argument("--fold", type=int, required=True, choices=(0, 1))
    c.add_argument("--gt-dir", default=str(ROOT / "data" / "train"))
    c.add_argument("--out-dir", required=True)
    c.add_argument("--max-crops", type=int)
    c.add_argument("--stride", type=int)
    c.add_argument("--only", help="comma-separated crop names; for smoke runs only")
    c.add_argument("--per-seed", type=int, default=2)
    c.add_argument("--assoc-per-crop", type=int, default=3,
                   help=("cap on cf_assoc_mistake rows per crop, so the class stays comparable in "
                         "size to the other constructions instead of swamping them"))
    c.set_defaults(func=cmd_census)

    x = sub.add_parser("crossfit")
    x.add_argument("--out-dir", required=True)
    x.set_defaults(func=cmd_crossfit)

    e = sub.add_parser("external")
    e.add_argument("--out-dir", required=True)
    e.set_defaults(func=cmd_external)

    v = sub.add_parser("viability")
    v.add_argument("--out-dir", required=True)
    v.add_argument("--allow-external", action="store_true",
                   help=("override EXTERNAL_ADMISSIBLE and mix the Zebrahub rows in. LEVER-0040 "
                         "forbids this until the FACT-0385 anti-alignment is solved AND the "
                         "solution demonstrated; the flag exists so that using them is a "
                         "deliberate, recorded act rather than a default"))
    v.set_defaults(func=cmd_viability)

    args = ap.parse_args()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
