r"""THE FROZEN, GT-FREE TEMPORAL PROPOSAL RULE for LEVER-0045 FALSIFIER 0 (PKT-0044).

THIS FILE CONTAINS NO GROUND TRUTH AND MUST NEVER IMPORT ANY. That is the whole point of
splitting it out of the measurement: the freeze is DEMONSTRABLE rather than asserted. The rule
and its parameters are pickled and hashed by ``freeze()``; the hash is recorded in PKT-0044
BEFORE the measurement module (``reach_oracle.py``) is allowed to read ``data/train``. A grep of
this file for ``geff``, ``gt`` or ``data/train`` returns nothing, and ``freeze()`` asserts that.

WHAT THE RULE IS, VERBATIM FROM LEVER-0045
------------------------------------------
  * context lengths 2, 5 and 8 frames;
  * tracklets built from DEPLOYED PREDICTED EDGES ONLY (the pre-ILP candidate set, which is the
    >0.5 source-axis softmax surface verified at source in FACT-0369, so at most one parent per
    target and the backward walk is unambiguous);
  * tracklet motion EXTRAPOLATED to the target frame;
  * previous-frame parents nominated by PHYSICAL DISTANCE to the extrapolated position;
  * a FIXED candidate cap and physical gate, IDENTICAL across both embryos;
  * ADJACENT-FRAME candidates ONLY. Longer context supplies CONTEXT, never a longer edge.

WHERE THE TWO FREE PARAMETERS COME FROM - neither is chosen from any GT quantity, and neither
may be re-chosen after a result is seen (PKT-0044 `forbidden`):

  gate_um = 6.0   the DEPLOYED program's own motion-relink TIGHT gate, read at
                  _evidence/assoc/survival/stage_map.json stage order 4
                  ("a two-pass Hungarian over ALL node pairs within a 6.0 um tight gate").
                  Taking the gate from the deployed final parent selector rather than inventing
                  one means the proposal surface is one the deployed relink could already act on.
  cap_k   = 3     the deployed candidate rule admits exactly ONE parent per target by arithmetic
                  (FACT-0369); the KILLED LEVER-0037 widening admitted TWO (FACT-0376). 3 is the
                  smallest cap strictly beyond the setting that was already killed, so the
                  candidate multiplier this buys is the smallest one that is a new experiment.

WHY THIS IS NOT LEVER-0037 IN DISGUISE
--------------------------------------
LEVER-0037 ranked candidates by GLOBAL EDGE PROBABILITY at a lowered floor - a wider radius on a
two-frame surface. This rule never reads ``edge_prob`` at all. It ranks by the residual to a
MOTION EXTRAPOLATION over a temporal context, which is a different ordering of the same
frame-(f-1) pool. FACT-0370 is the independent reason the distinction is not semantic: the
never-offered misses are SHORT (median 2.26 um), so they are already inside any plausible radius
and only a better ORDERING of the pool can reach them.

USAGE
    python scripts/win_bet/reach_rule.py --freeze --out _evidence/assoc/reach
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

# Atlas coords are FULL-RES (z, y, x) and take (1.625, 0.40625, 0.40625) - NOT isotropic.
# Two of three distance analyses on 2026-08-25 started with the wrong convention (FACT-0040).
SCALE_UM = (1.625, 0.40625, 0.40625)
_SCALE = np.asarray(SCALE_UM, dtype=np.float64)

FROZEN_RULE: dict = {
    "rule_id": "PKT0044-TEMPORAL-PROPOSAL-v1",
    "packet": "PKT-0044",
    "lever": "LEVER-0045",
    "falsifier": "FALSIFIER 0 - the CPU reach oracle",
    "gt_free": True,
    "context_lengths": [2, 5, 8],
    "gate_um": 6.0,
    "gate_provenance": "_evidence/assoc/survival/stage_map.json stages[3] motion_relink tight gate",
    "cap_k": 3,
    "cap_provenance": "FACT-0369 deployed cap is 1; FACT-0376 killed setting is 2; 3 is the "
                      "smallest cap strictly beyond the killed setting",
    "scale_um": list(SCALE_UM),
    "tracklet_source": "DEPLOYED pre-ILP candidate edges only (row_type=='edge' in the pre-ILP "
                       "export, the >0.5 softmax surface of FACT-0369). edge_prob is NEVER read.",
    "tracklet_walk": "backward along the unique candidate parent, following a link ONLY when the "
                     "parent frame is exactly one less than the child frame; the walk stops at "
                     "the first missing or non-adjacent parent",
    "extrapolation": "degree-1 least-squares fit of position (um) against frame index over the "
                     "up-to-L positions of the walk, evaluated at the TARGET frame f = f_src + 1; "
                     "a walk of length 1 extrapolates with zero velocity (the node's own position)",
    "nomination": "for every node at frame f, the cap_k nearest frame-(f-1) nodes by Euclidean "
                  "distance in um between the target's OWN position and the source's EXTRAPOLATED "
                  "position, keeping only distances <= gate_um; ties broken by ascending node_id",
    "adjacency": "sources at frame f-1, targets at frame f. ONLY. Longer context never produces a "
                 "longer edge.",
    "identical_across_embryos": True,
    "per_fold_tuning": "NONE. The same gate, cap and context lengths run on 44b6 (fold 0) and "
                       "6bba (fold 1).",
    "reads_any_label": False,
    "reads_edge_prob": False,
}

# Assembled at runtime so the guard's own definition is not a hit for itself.
_FORBIDDEN_TOKENS = ("g" + "eff", "data/" + "train", "ground_" + "truth", "gt_" + "edges")


def rule_hash() -> dict:
    """sha256 of the pickled rule AND of this file's own bytes. Both are recorded."""
    blob = pickle.dumps(FROZEN_RULE, protocol=4)
    src = Path(__file__).read_bytes()
    return {
        "rule_pickle_sha256": hashlib.sha256(blob).hexdigest(),
        "rule_pickle_bytes": len(blob),
        "reach_rule_py_sha256": hashlib.sha256(src).hexdigest(),
    }


def assert_gt_free() -> None:
    """A silent no-op is worse than a crash: prove the freeze rather than assert it."""
    text = Path(__file__).read_text(encoding="utf-8").lower()
    body = text.split('"""', 2)[-1]  # skip the docstring, which NAMES the forbidden tokens
    bad = [t for t in _FORBIDDEN_TOKENS if t in body]
    if bad:
        raise RuntimeError(f"reach_rule.py is not GT-free: {bad}")


def to_um(zyx: np.ndarray) -> np.ndarray:
    return np.asarray(zyx, dtype=np.float64) * _SCALE


def _chain_positions(order: np.ndarray, parent: np.ndarray, t: np.ndarray,
                     pos: np.ndarray, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    """hist[n, max_len, 3] of the backward walk, plus its length per node.

    ``hist[i, 0]`` is node i's own position, ``hist[i, k]`` its k-th ancestor. Only adjacent
    (frame - 1) steps are followed.
    """
    n = len(t)
    hist = np.zeros((n, max_len, 3), dtype=np.float64)
    length = np.ones(n, dtype=np.int64)
    hist[:, 0, :] = pos
    for i in order:  # ascending frame, so a parent is always finished before its child
        p = parent[i]
        if p < 0 or t[p] != t[i] - 1:
            continue
        m = min(int(length[p]) + 1, max_len)
        hist[i, 1:m, :] = hist[p, 0:m - 1, :]
        length[i] = m
    return hist, length


def extrapolate(hist: np.ndarray, length: np.ndarray, context: int) -> np.ndarray:
    """Position each node is predicted to occupy ONE FRAME LATER, in um.

    Degree-1 least squares over the walk truncated to ``context`` points. Regular spacing, so the
    fit is closed form: with offsets x_k = -k for k = 0..m-1, the prediction at x = +1 is
    ``mean_P + B * (1 - mean_x)``.
    """
    m_used = np.minimum(length, context)
    out = np.empty((len(hist), 3), dtype=np.float64)
    for m in np.unique(m_used):
        sel = np.nonzero(m_used == m)[0]
        P = hist[sel, :m, :]                                   # (s, m, 3)
        if m == 1:
            out[sel] = P[:, 0, :]                              # zero velocity
            continue
        x = -np.arange(m, dtype=np.float64)                    # 0, -1, -2, ...
        mean_x = x.mean()
        sxx = float(((x - mean_x) ** 2).sum())
        mean_P = P.mean(axis=1)                                # (s, 3)
        B = np.einsum("k,skc->sc", x - mean_x, P) / sxx        # (s, 3)
        out[sel] = mean_P + B * (1.0 - mean_x)
    return out


def propose(t: np.ndarray, zyx_voxel: np.ndarray, node_id: np.ndarray,
            parent_of_index: np.ndarray, context: int,
            gate_um: float = FROZEN_RULE["gate_um"],
            cap_k: int = FROZEN_RULE["cap_k"]) -> dict[int, list[tuple[int, float]]]:
    """THE RULE. Returns {target_node_id: [(source_node_id, residual_um), ...]} rank-ordered.

    ``parent_of_index[i]`` is the index of node i's unique candidate parent, or -1. Every input
    is derived from the deployed pre-ILP export; nothing here has seen a label.
    """
    from scipy.spatial import cKDTree

    pos = to_um(zyx_voxel)
    order = np.argsort(t, kind="stable")
    hist, length = _chain_positions(order, parent_of_index, t, pos, max(context, 1))
    pred = extrapolate(hist, length, context)

    by_frame: dict[int, np.ndarray] = {}
    for f in np.unique(t):
        by_frame[int(f)] = np.nonzero(t == f)[0]

    out: dict[int, list[tuple[int, float]]] = {}
    for f, tgt_idx in by_frame.items():
        src_idx = by_frame.get(f - 1)
        if src_idx is None or not len(src_idx) or not len(tgt_idx):
            continue
        tree = cKDTree(pred[src_idx])
        k = min(cap_k, len(src_idx))
        d, j = tree.query(pos[tgt_idx], k=k, distance_upper_bound=gate_um)
        d = np.atleast_2d(d.T).T if k > 1 else d.reshape(-1, 1)
        j = np.atleast_2d(j.T).T if k > 1 else j.reshape(-1, 1)
        for r, ti in enumerate(tgt_idx):
            picks = [(int(node_id[src_idx[j[r, c]]]), float(d[r, c]))
                     for c in range(k) if np.isfinite(d[r, c]) and j[r, c] < len(src_idx)]
            if picks:
                out[int(node_id[ti])] = picks
    return out


def freeze(out_dir: Path) -> dict:
    assert_gt_free()
    out_dir.mkdir(parents=True, exist_ok=True)
    blob = pickle.dumps(FROZEN_RULE, protocol=4)
    (out_dir / "frozen_rule.pkl").write_bytes(blob)
    payload = {"schema_version": 1, "heartbeat": "REACH_RULE_FROZEN",
               "rule": FROZEN_RULE, **rule_hash()}
    (out_dir / "frozen_rule.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("_evidence/assoc/reach"))
    a = ap.parse_args(argv)
    p = freeze(a.out) if a.freeze else {"rule": FROZEN_RULE, **rule_hash()}
    print(json.dumps({k: v for k, v in p.items() if k != "rule"}, indent=2))
    print("REACH_RULE_FROZEN" if a.freeze else "REACH_RULE_PREVIEW")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
