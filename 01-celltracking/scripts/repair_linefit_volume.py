"""Axis-wise repair and boundary audit for ``OUTPUT_LINEFIT_SMOOTH``.

WHY
---
``src/biotrack/wrapper.py::linefit_smooth_output_graph`` blends every node towards a
line fitted over its ``+/-OUTPUT_LINEFIT_WINDOW`` neighbourhood. At a track endpoint the
neighbourhood is one-sided, so the fit EXTRAPOLATES and can push a node outside the
acquisition volume. P0-C (v122 + reverse-time association) emitted node 15274 of
``44b6_0b24845f`` at ``t=43`` with ``z=64`` where ``z_max = 63``; the structural audit
fails on it (A3 volume), which blocks the artifact from a submission slot.

The repair is deliberately NOT a clamp. Clamping invents a boundary coordinate the
detector never proposed and manufactures a pile-up on the boundary plane -- and the
population is already boundary-heavy, so that is exactly the wrong move. Instead the
smoothed value is discarded PER AXIS and the original unsmoothed detector coordinate is
restored. That makes smoothing provably incapable of evicting a node from the volume.

The live fix is in the wrapper (``OUTPUT_VOLUME_GUARD``). This script is the offline
counterpart, for artifacts that were produced BEFORE the fix existed:

  pileup   quantify boundary occupancy and, by re-applying the smoother forward to a
           submission's own node population, measure how many nodes the unfixed vs the
           fixed smoother evicts from the volume -- by crop and by axis -- and whether
           smoothing concentrates mass on the boundary planes.

  repair   rewrite an existing submission.csv so no node lies outside the volume, by
           RECONSTRUCTING each affected node's original coordinate rather than clamping.

RECONSTRUCTION (``repair``)
---------------------------
The unsmoothed coordinates were never persisted, so they are recovered instead of read.
This is sound because the smoother is an affine operator with topology-determined
coefficients, the inputs are integer voxel indices, and the outputs are observed:

  * the smoothing neighbourhood is found by walking unique predecessors/successors, so
    it never leaves the maximal unique-degree chain containing the node -- the chain is
    a closed system that can be solved on its own;
  * per axis the map is ``u = A o`` with ``A`` known from topology alone;
  * ``o`` is integral and ``round(u)`` is observed for every chain member.

So the preimage is found by exact integer search (a DP along the chain, state = the last
four placed originals). The search returns EVERY integer ``o`` consistent with the whole
observed chain. If the affected coordinate is not identical across all solutions the
repair REFUSES rather than guessing, and if no solution exists it refuses too.

Usage
-----
  python scripts/repair_linefit_volume.py pileup <submission.csv> [...] [--json-out P]
  python scripts/repair_linefit_volume.py repair <submission.csv> -o <repaired.csv>
                                                 [--json-out P] [--search-radius N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biotrack import wrapper  # noqa: E402

AXES = ("z", "y", "x")
# Every crop in data/train (199/199) and every movie in data/test (4/4) is
# (T, Z, Y, X) = (100, 64, 256, 256).
MAXES = (63, 255, 255)
BOUNDARY_BAND = 3  # how many voxels in from each face count as "at/near the boundary"


# --------------------------------------------------------------------------- io


def load_graphs(path: Path) -> dict[str, tuple[dict[int, dict], list[dict]]]:
    """Per dataset, return (nodes_by_id, edges) in the wrapper's own dict format."""
    df = pd.read_csv(path)
    out: dict[str, tuple[dict[int, dict], list[dict]]] = {}
    for ds, grp in df.groupby("dataset", sort=True):
        nrows = grp[grp["row_type"].eq("node")]
        erows = grp[grp["row_type"].eq("edge")]
        nodes = {
            int(nid): {"node_id": int(nid), "t": int(t), "z": float(z), "y": float(y), "x": float(x)}
            for nid, t, z, y, x in zip(
                nrows["node_id"], nrows["t"], nrows["z"], nrows["y"], nrows["x"]
            )
        }
        edges = [
            {"source_id": int(s), "target_id": int(t)}
            for s, t in zip(erows["source_id"], erows["target_id"])
        ]
        out[str(ds)] = (nodes, edges)
    return out


def adjacency(nodes: dict[int, dict], edges: list[dict]):
    """The predecessor/successor maps the smoother itself builds."""
    pred: dict[int, list[int]] = defaultdict(list)
    succ: dict[int, list[int]] = defaultdict(list)
    for e in edges:
        s, t = int(e["source_id"]), int(e["target_id"])
        if s not in nodes or t not in nodes:
            continue
        if int(nodes[t]["t"]) != int(nodes[s]["t"]) + 1:
            continue
        succ[s].append(t)
        pred[t].append(s)
    return pred, succ


def neighbourhood(node_id: int, pred, succ, present, window: int) -> list[tuple[int, int]]:
    """Reproduce the smoother's neighbourhood walk exactly."""
    nb = [(0, node_id)]
    cur = node_id
    for step in range(1, window + 1):
        p = pred.get(cur, [])
        if len(p) != 1:
            break
        cur = p[0]
        if cur not in present:
            break
        nb.append((-step, cur))
    cur = node_id
    for step in range(1, window + 1):
        s = succ.get(cur, [])
        if len(s) != 1:
            break
        cur = s[0]
        if cur not in present:
            break
        nb.append((step, cur))
    return nb


def fit_coefficients(dts: list[int]) -> np.ndarray:
    """Row of the affine smoothing operator for one neighbourhood.

    ``np.polyfit`` is linear in its ``y`` argument, so evaluating it on the unit basis
    recovers the exact operator the wrapper applies -- no re-derivation, no drift.
    """
    d = np.asarray(dts, dtype=np.float64)
    coefs = np.empty(len(dts), dtype=np.float64)
    for j in range(len(dts)):
        basis = np.zeros(len(dts), dtype=np.float64)
        basis[j] = 1.0
        coefs[j] = np.polyval(np.polyfit(d, basis, 1), 0.0)
    w = float(np.clip(wrapper.OUTPUT_LINEFIT_WEIGHT, 0.0, 1.0))
    coefs *= w
    coefs[0] += 1.0 - w  # index 0 is always the node itself
    return coefs


def written(value: float) -> int:
    """The integer the P0-C kernel writer emits: ``max(0, int(round(v)))``.

    The low-side ``max(0, ...)`` is itself a silent clamp in the deployed writer. It is
    modelled here rather than corrected so that reconstruction matches what was actually
    written; the wrapper-side guard is what removes the need for it.
    """
    return max(0, int(round(float(value))))


# ------------------------------------------------------------------- pile-up audit


def boundary_profile(nodes: dict[int, dict]) -> dict:
    """Occupancy of the boundary bands, plus anything already outside the volume."""
    prof = {}
    for axis, name in enumerate(AXES):
        vals = np.array([n[name] for n in nodes.values()], dtype=np.float64)
        rounded = np.array([written(v) for v in vals])
        lo = {int(k): int(v) for k, v in Counter(rounded[rounded <= BOUNDARY_BAND - 1]).items()}
        hi_mask = rounded >= MAXES[axis] - (BOUNDARY_BAND - 1)
        hi = {int(k): int(v) for k, v in Counter(rounded[hi_mask]).items()}
        prof[name] = {
            "n": int(len(vals)),
            "min": int(rounded.min()),
            "max": int(rounded.max()),
            "outside": int(((rounded < 0) | (rounded > MAXES[axis])).sum()),
            "on_low_face": int((rounded == 0).sum()),
            "on_high_face": int((rounded == MAXES[axis]).sum()),
            "low_band": dict(sorted(lo.items())),
            "high_band": dict(sorted(hi.items())),
        }
    return prof


def smooth_population(nodes: dict[int, dict], edges: list[dict], guard: bool):
    """Re-apply the smoother forward to a node population. Returns (coords, stats)."""
    work = {nid: dict(n) for nid, n in nodes.items()}
    stats = {"linefit_smoothed_nodes": 0, "linefit_skipped_nodes": 0}
    for name in AXES:
        stats[f"linefit_volume_fallback_{name}"] = 0
        stats[f"linefit_volume_original_invalid_{name}"] = 0
    prev_guard = wrapper.OUTPUT_VOLUME_GUARD
    wrapper.OUTPUT_VOLUME_GUARD = guard
    try:
        out = wrapper.linefit_smooth_output_graph(work, edges, stats)
    finally:
        wrapper.OUTPUT_VOLUME_GUARD = prev_guard
    return out, stats


def cmd_pileup(paths: list[Path]) -> dict:
    """Boundary occupancy before/after smoothing, unfixed vs fixed, by crop and axis."""
    report = {"boundary_band": BOUNDARY_BAND, "volume_zyx_max": list(MAXES), "artifacts": []}
    for path in paths:
        graphs = load_graphs(path)
        entry = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "datasets": {},
            "totals": {
                "evicted_unfixed": dict.fromkeys(AXES, 0),
                "fallbacks_fixed": dict.fromkeys(AXES, 0),
                "original_invalid": dict.fromkeys(AXES, 0),
            },
        }
        for ds in sorted(graphs):
            nodes, edges = graphs[ds]
            before = boundary_profile(nodes)
            unfixed, ustats = smooth_population(nodes, edges, guard=False)
            fixed, fstats = smooth_population(nodes, edges, guard=True)
            after_unfixed = boundary_profile(unfixed)
            after_fixed = boundary_profile(fixed)
            evicted = {a: after_unfixed[a]["outside"] - before[a]["outside"] for a in AXES}
            entry["datasets"][ds] = {
                "nodes": len(nodes),
                "smoothed_nodes": ustats["linefit_smoothed_nodes"],
                "skipped_nodes": ustats["linefit_skipped_nodes"],
                "before": before,
                "after_unfixed": after_unfixed,
                "after_fixed": after_fixed,
                "evicted_by_unfixed_smoothing": evicted,
                "fallbacks_by_fixed_smoothing": {
                    a: fstats[f"linefit_volume_fallback_{a}"] for a in AXES
                },
                "original_invalid": {
                    a: fstats[f"linefit_volume_original_invalid_{a}"] for a in AXES
                },
                "face_occupancy_delta_unfixed": {
                    a: {
                        "low": after_unfixed[a]["on_low_face"] - before[a]["on_low_face"],
                        "high": after_unfixed[a]["on_high_face"] - before[a]["on_high_face"],
                    }
                    for a in AXES
                },
                "face_occupancy_delta_fixed": {
                    a: {
                        "low": after_fixed[a]["on_low_face"] - before[a]["on_low_face"],
                        "high": after_fixed[a]["on_high_face"] - before[a]["on_high_face"],
                    }
                    for a in AXES
                },
            }
            for a in AXES:
                entry["totals"]["evicted_unfixed"][a] += evicted[a]
                entry["totals"]["fallbacks_fixed"][a] += fstats[f"linefit_volume_fallback_{a}"]
                entry["totals"]["original_invalid"][a] += fstats[
                    f"linefit_volume_original_invalid_{a}"
                ]
        report["artifacts"].append(entry)
    return report


# ------------------------------------------------------------------ reconstruction


def maximal_chain(node_id: int, pred, succ, present) -> list[int]:
    """Maximal chain of unique-degree links through ``node_id``.

    Every smoothing neighbourhood is built by the same unique-degree walk, so no node of
    this chain has a neighbour outside it: the chain is closed under the operator.
    """
    back = []
    cur = node_id
    seen = {node_id}
    while True:
        p = pred.get(cur, [])
        if len(p) != 1 or p[0] not in present or p[0] in seen:
            break
        cur = p[0]
        seen.add(cur)
        back.append(cur)
    fwd = []
    cur = node_id
    while True:
        s = succ.get(cur, [])
        if len(s) != 1 or s[0] not in present or s[0] in seen:
            break
        cur = s[0]
        seen.add(cur)
        fwd.append(cur)
    return back[::-1] + [node_id] + fwd


def reconstruct_chain_axis(
    chain: list[int],
    rows: list[tuple[list[int], np.ndarray] | None],
    observed: list[int],
    radius: int,
    axis: int,
    target: int,
) -> tuple[list[int], int]:
    """Feasible integer originals at ``target`` reproducing ``observed`` for one axis.

    ``rows[i]`` is ``(chain_indices, coefficients)`` for node i, or ``None`` when the
    smoother skipped node i (neighbourhood < 3) and therefore wrote it through unchanged.

    Exhaustive, not heuristic. Constraints span at most ``2*window+1`` consecutive chain
    positions, so a forward DP whose state is the last ``2*window`` placed values loses
    nothing; each state carries the set of ``target`` values that can still reach it.
    Returns ``(sorted feasible values at target, number of surviving DP states)``.
    """
    n = len(chain)
    hi = MAXES[axis]
    keep = 2 * max(1, int(wrapper.OUTPUT_LINEFIT_WINDOW))
    cands = [
        list(range(max(0, observed[i] - radius), min(hi, observed[i] + radius) + 1))
        for i in range(n)
    ]
    for i in range(n):
        if rows[i] is None:  # untouched by the smoother: the written value IS the original
            cands[i] = [observed[i]] if 0 <= observed[i] <= hi else []
    # A node's constraint becomes checkable once every index it depends on is placed.
    ready_at: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        if rows[i] is not None:
            ready_at[max(rows[i][0])].append(i)

    # state -> set of target values compatible with some prefix ending in that state
    states: dict[tuple[int, ...], set[int]] = {(): set()}
    placed: dict[tuple[int, ...], tuple[int, ...]] = {(): ()}
    for pos in range(n):
        nxt: dict[tuple[int, ...], set[int]] = {}
        nxt_placed: dict[tuple[int, ...], tuple[int, ...]] = {}
        for state, tvals in states.items():
            full = placed[state]
            for v in cands[pos]:
                trial = full + (v,)
                ok = True
                for i in ready_at.get(pos, ()):
                    idxs, coefs = rows[i]
                    if written(float(np.dot(coefs, [trial[j] for j in idxs]))) != observed[i]:
                        ok = False
                        break
                if not ok:
                    continue
                new_t = set(tvals) if pos != target else {v}
                key = trial[-keep:]
                if key in nxt:
                    nxt[key] |= new_t
                else:
                    nxt[key] = new_t
                    nxt_placed[key] = trial
        states, placed = nxt, nxt_placed
        if not states:
            return [], 0
        if len(states) > 500_000:
            raise RuntimeError("reconstruction search exploded; lower --search-radius")
    feasible: set[int] = set()
    for tvals in states.values():
        feasible |= tvals
    return sorted(feasible), len(states)


def reconstruct_node(nodes, edges, node_id: int, radius: int) -> dict:
    """Certified original coordinate for one node, per axis."""
    pred, succ = adjacency(nodes, edges)
    present = set(nodes)
    chain = maximal_chain(node_id, pred, succ, present)
    pos_of = {nid: i for i, nid in enumerate(chain)}

    rows: list[tuple[list[int], np.ndarray] | None] = []
    for nid in chain:
        nb = neighbourhood(nid, pred, succ, present, wrapper.OUTPUT_LINEFIT_WINDOW)
        if len(nb) < 3:
            rows.append(None)
            continue
        coefs = fit_coefficients([d for d, _ in nb])
        rows.append(([pos_of[m] for _, m in nb], coefs))

    result = {
        "node_id": node_id,
        "chain_length": len(chain),
        "chain_position": pos_of[node_id],
        "search_radius": radius,
        "axes": {},
    }
    for axis, name in enumerate(AXES):
        observed = [written(nodes[nid][name]) for nid in chain]
        vals, n_states = reconstruct_chain_axis(
            chain, rows, observed, radius, axis, pos_of[node_id]
        )
        result["axes"][name] = {
            "observed_smoothed": observed[pos_of[node_id]],
            "surviving_dp_states": n_states,
            "candidate_originals": vals,
            "unique": len(vals) == 1,
            "original": vals[0] if len(vals) == 1 else None,
        }
    return result


def cmd_repair(path: Path, out_path: Path, radius: int) -> dict:
    graphs = load_graphs(path)
    df = pd.read_csv(path)
    report = {
        "source": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "repaired": str(out_path),
        "policy": "restore reconstructed original detector coordinate; never clamp",
        "fallbacks_by_crop_and_axis": {},
        "nodes": [],
        "refused": [],
    }
    edits: list[tuple[str, int, str, int, int]] = []

    for ds in sorted(graphs):
        nodes, edges = graphs[ds]
        per_axis = dict.fromkeys(AXES, 0)
        for nid, node in nodes.items():
            bad = [a for a, name in enumerate(AXES) if not (0 <= written(node[name]) <= MAXES[a])]
            if not bad:
                continue
            rec = reconstruct_node(nodes, edges, nid, radius)
            rec["dataset"] = ds
            rec["t"] = int(node["t"])
            rec["out_of_volume_axes"] = [AXES[a] for a in bad]
            report["nodes"].append(rec)
            for a in bad:
                name = AXES[a]
                info = rec["axes"][name]
                if not info["unique"]:
                    report["refused"].append(
                        {"dataset": ds, "node_id": nid, "axis": name,
                         "candidates": info["candidate_originals"]}
                    )
                    continue
                orig = info["original"]
                if not (0 <= orig <= MAXES[a]):
                    report["refused"].append(
                        {"dataset": ds, "node_id": nid, "axis": name,
                         "reason": "reconstructed original is itself out of volume",
                         "original": orig}
                    )
                    continue
                edits.append((ds, nid, name, written(node[name]), orig))
                per_axis[name] += 1
        if any(per_axis.values()):
            report["fallbacks_by_crop_and_axis"][ds] = per_axis

    mask_node = df["row_type"].eq("node")
    for ds, nid, name, old, new in edits:
        sel = mask_node & df["dataset"].eq(ds) & df["node_id"].eq(nid)
        assert int(sel.sum()) == 1, f"{ds}/{nid} did not select exactly one node row"
        df.loc[sel, name] = new
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    report["edits"] = [
        {"dataset": d, "node_id": n, "axis": a, "from": o, "to": v} for d, n, a, o, v in edits
    ]
    report["n_edits"] = len(edits)
    report["repaired_sha256"] = hashlib.sha256(out_path.read_bytes()).hexdigest()
    return report


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pileup")
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--json-out", type=Path)
    r = sub.add_parser("repair")
    r.add_argument("path", type=Path)
    r.add_argument("-o", "--out", required=True, type=Path)
    r.add_argument("--search-radius", type=int, default=3)
    r.add_argument("--json-out", type=Path)
    args = ap.parse_args(argv)

    if args.cmd == "pileup":
        rep = cmd_pileup(args.paths)
        for art in rep["artifacts"]:
            print(f"\n=== BOUNDARY PILE-UP: {art['path']}")
            for ds, d in art["datasets"].items():
                print(f"  {ds}  nodes={d['nodes']} smoothed={d['smoothed_nodes']}")
                for a in AXES:
                    b, u, f = d["before"][a], d["after_unfixed"][a], d["after_fixed"][a]
                    print(
                        f"    {a}: outside before={b['outside']} unfixed={u['outside']} "
                        f"fixed={f['outside']} | on high face {b['on_high_face']}->"
                        f"{u['on_high_face']} (unfixed) / {f['on_high_face']} (fixed) | "
                        f"on low face {b['on_low_face']}->{u['on_low_face']} / {f['on_low_face']}"
                    )
                print(f"    evicted by unfixed smoothing: {d['evicted_by_unfixed_smoothing']}")
                print(f"    fallbacks under fix:          {d['fallbacks_by_fixed_smoothing']}")
            print(f"  TOTALS {art['totals']}")
        if args.json_out:
            args.json_out.write_text(json.dumps(rep, indent=2) + "\n")
        return 0

    rep = cmd_repair(args.path, args.out, args.search_radius)
    print(json.dumps(rep, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(rep, indent=2) + "\n")
    return 1 if rep["refused"] else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
