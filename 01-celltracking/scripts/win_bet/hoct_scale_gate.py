r"""THE SCALE GATE for any HOCT integration. Three parts, each able to REJECT, all fail-closed.

WHY THIS EXISTS
---------------
``FACT-0454`` blocker 1: the ``scale`` argument NEVER REACHES THE MODEL. ``scaled_`` occurs once
in the whole package, at ``graph.py:182``, into a ``node_attrs`` frame that is then discarded;
``graph.py:142-143`` defaults scale to 1.0 per axis; ``graph.py:212-218`` writes it into metadata
only AFTER ``DistanceEdges`` has already run at ``graph.py:190-196``; and
``_frame_dataset.py:53-54`` hands the RAW ``z, y, x`` columns to the batcher, which puts them
straight into ``node_pos`` at ``_batching.py:190``. HOCT is VOXEL-NATIVE, and our data is
anisotropic 4:1:1. Neither microns nor raw voxels is safe by default.

A GATE THAT ONLY PASSES IS NOT A GATE. This one carries four candidate conventions, one of which
is the mistake this project has actually made twice, and it must reject at least one on evidence.

WHAT IS NEW HERE, AND IT IS READ OUT OF THE CHECKPOINT ITSELF
--------------------------------------------------------------
``general_v1.pt`` is a TorchScript archive: it carries its own compiled source. Read as TEXT from
the zip - never ``torch.jit.load``, which would execute it - ``EdgeModel.forward`` contains

    elem_pdist = clamp_min_(baddbmm(...) , 1e-30)              # SQUARED pairwise distance
    dist_mask  = logical_or(lt(elem_pdist, 90000.), isinf(...))
    attn_bias  = masked_fill(zeros_like(...), ~mask, -inf)

``sqrt(90000) = 300``. **The paper's tau = 300 is not a configurable threshold - it is a HARD
CONSTANT COMPILED INTO THE ATTENTION MASK**, applied to ``node_pos`` and again to ``edge_pos``,
in whatever units those arrive in. It is applied at ``_api.py:206`` a second time as the default
candidate radius. So the unit choice does not merely rescale an input: it decides how much of a
crop the model can attend over, and it can silently switch the mechanism OFF.

AND IT IS WORSE THAN A THRESHOLD - 300 IS THE MODEL'S COORDINATE UNIT. The archive's
``constants/0..3`` storages (read as raw little-endian doubles/longs, again without loading)
are ``300.0``, ``1``, ``1.7320508075688772`` (= sqrt(3)), ``2``, and the compiled RoPE at
``pos_enc/___torch_mangle_3.py`` computes

    x0       = node_pos / CONSTANTS.c0          # /300.0
    freq_pos = x0 * exp(log_freq) / CONSTANTS.c2   # /sqrt(3), the 3-axis normaliser

So ``300`` fixes the wavelength of every positional frequency in the network as well as the
attention radius. There is no argument that changes it. Choosing our unit IS choosing where our
data sits inside the model's coordinate system.

One consequence worth stating because it opens a free move: the rotary encoding is applied
identically to q and k and is followed by a Householder reflection ``(eye - 2 v v^T)`` that is
also identical for q and k, so the attention logits depend on position DIFFERENCES only. A pure
TRANSLATION of ``node_pos`` is therefore free in the attention geometry, while it does move the
standardized slots 1..3. ``voxel_raw_z_recentred`` below exploits exactly that.

THE THREE PARTS
---------------
A  METRIC FIDELITY.   Convert GT inter-frame displacements into the candidate convention, then
                      back to microns using that convention's own declared micron-per-unit.
                      Must reproduce ``FACT-0040`` (1.817 um). This catches a coding error in the
                      converter. It is NOT sufficient on its own and the gate says so.
B  RECEPTIVE FIELD.   Under the convention, is the compiled tau = 300 still a live mechanism at
                      the geometry it was trained to operate on, and is the neighbourhood it
                      defines ISOTROPIC in physical space?
C  DISTRIBUTION.      ``node_pos`` also enters the feature vector at slots 1..3, where it is
                      standardized by ``_api.py:18-60``. Where do OUR coordinates land in that
                      distribution?

Part C is the one that decides the route, and it decides it against us. See ``verdict`` in the
payload.

USAGE
-----
    python scripts/win_bet/hoct_scale_gate.py --out <json> [--prefix 6bba] [--max-crops N]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hoct_feature_contract import PUBLISHER_MEAN, PUBLISHER_STD  # noqa: E402

HEARTBEAT_OK = "HOCT_SCALE_GATE_COMPLETE"
HEARTBEAT_REFUSED = "HOCT_SCALE_GATE_REFUSED"

CHECKPOINT = Path(r"C:/temp/hoct_official/general_v1.pt")
ARCHIVED_EDGE_MODEL = "general_v1/code/__torch__/slot_cell_tracking/modules/edge_model.py"

# FACT-0040, fold 1, VERIFIED. Tolerance is the committed nearest_parent_oracle gate's +/- 0.15.
ANCHOR_UM = 1.817
TOLERANCE_UM = 0.15

# AGENTS.md section 4: full-res (z, y, x) take (1.625, 0.40625, 0.40625). NOT isotropic.
UM_PER_VOXEL = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
# Measured this cycle over all 199 train crops: every one is (T, Z, Y, X) = (100, 64, 256, 256).
CROP_SHAPE_TZYX = (100, 64, 256, 256)


# ---------------------------------------------------------------------------------------------
# The compiled constant, re-derived rather than restated
# ---------------------------------------------------------------------------------------------
def _read_scalar_constants(checkpoint: Path) -> dict:
    """The TorchScript ``constants/N`` storages, as raw bytes. No unpickling, no execution.

    ``constants.pkl`` (disassembled with ``pickletools``, never loaded) declares four rank-0
    tensors over storages ``0..3`` with dtypes DoubleStorage, LongStorage, DoubleStorage,
    LongStorage. Each is 8 bytes, so the value is a single ``struct.unpack``.
    """
    import struct

    out: dict = {}
    with zipfile.ZipFile(checkpoint) as zf:
        for key, fmt, label in (("0", "<d", "c0_double"), ("1", "<q", "c1_long"),
                                ("2", "<d", "c2_double"), ("3", "<q", "c3_long")):
            name = f"general_v1/constants/{key}"
            if name not in zf.namelist():
                out[label] = None
                continue
            raw = zf.read(name)
            out[label] = struct.unpack(fmt, raw[:8])[0] if len(raw) >= 8 else None
    out["how_read"] = ("pickletools.dis on constants.pkl for the dtypes, then a raw 8-byte read "
                       "of each constants/N storage. Neither torch.jit.load nor pickle.load ran")
    return out


def read_compiled_tau(checkpoint: Path = CHECKPOINT) -> dict:
    """Extract the hard attention cutoff from the TorchScript archive, WITHOUT executing it.

    Fails closed: if the archive, the module or the constant is not found, the gate refuses
    rather than falling back to the paper's number - a number read from prose is not a number
    read from the artifact.
    """
    if not checkpoint.exists():
        return {"found": False, "why": f"checkpoint absent at {checkpoint}"}
    with zipfile.ZipFile(checkpoint) as zf:
        if ARCHIVED_EDGE_MODEL not in zf.namelist():
            return {"found": False, "why": f"{ARCHIVED_EDGE_MODEL} not in archive"}
        src = zf.read(ARCHIVED_EDGE_MODEL).decode("utf-8", "replace")

    hits = re.findall(r"torch\.lt\(elem_pdist\d*,\s*([0-9.eE+-]+)\)", src)
    if not hits:
        return {"found": False, "why": "no `torch.lt(elem_pdist, ...)` in the compiled forward"}
    vals = sorted({float(h) for h in hits})
    if len(vals) != 1:
        return {"found": False, "why": f"ambiguous cutoffs {vals}", "occurrences": len(hits)}

    squared = vals[0]
    orphan = bool(re.search(r"_\d+ = torch\.new_zeros\(x,\s*\[.*?\], dtype=None", src))
    ret = re.search(r"return \(([^)]*)\)", src)
    consts = _read_scalar_constants(checkpoint)
    return {
        "found": True,
        "squared_cutoff": squared,
        "tau_in_node_pos_units": math.sqrt(squared),
        "rope_constants": consts,
        "tau_appears_twice": bool(
            consts.get("c0_double") is not None
            and abs(consts["c0_double"] - math.sqrt(squared)) < 1e-9
        ),
        "rope_note": "CONSTANTS.c0 is the RoPE positional normaliser (node_pos / c0) and c2 is "
                     "the per-axis normaliser sqrt(3). c0 equalling the attention cutoff means "
                     "300 is the model's COORDINATE UNIT, not a tunable radius",
        "applied_to": ["node_pos (node self-attention)", "edge_pos (edge self-attention)"],
        "occurrences": len(hits),
        "where": f"{ARCHIVED_EDGE_MODEL} :: EdgeModel.forward",
        "how_read": "zipfile text read of the archived TorchScript source. torch.jit.load was "
                    "NOT called, so the graph was never executed",
        "orphan_output_is_new_zeros": orphan,
        "return_tuple": ret.group(1).strip() if ret else None,
        "orphan_note": "FACT-0453 re-derived here: output 3 is `torch.new_zeros(x, [B, N, 1])`. "
                       "_predict.py:359 then takes exp() of it, so orphan_exp == 1.0 for every "
                       "node and the 'abstain' mass at _predict.py:426-428 is a HARD-CODED "
                       "logit-0 reference, not a learned per-node quantity. An adapter that "
                       "reads orphan_prob as a learned score reads 1/(denominator+1)",
    }


# ---------------------------------------------------------------------------------------------
# The candidate conventions
# ---------------------------------------------------------------------------------------------
CONVENTIONS = {
    "um_anisotropic": {
        "factors_zyx": [1.625, 0.40625, 0.40625],
        "um_per_unit": 1.0,
        "isotropic_in_physical_space": True,
        "what_it_is": "physical microns; the convention the official scorer and every one of our "
                      "distance instruments uses",
    },
    "voxel_raw": {
        "factors_zyx": [1.0, 1.0, 1.0],
        "um_per_unit": 0.40625,          # declared on the xy pitch; z is 4x coarser per unit
        "isotropic_in_physical_space": False,
        "what_it_is": "the label array's index space - what HOCT ACTUALLY consumes today, "
                      "because graph.py:182's scaled_ columns are discarded",
    },
    "voxel_isotropic_xy": {
        "factors_zyx": [4.0, 1.0, 1.0],  # 1.625 / 0.40625 == 4.0 exactly
        "um_per_unit": 0.40625,
        "isotropic_in_physical_space": True,
        "what_it_is": "index space made metrically isotropic by scaling z to the xy pitch. "
                      "Voxel-native in magnitude, undistorted in geometry",
    },
    "voxel_raw_z_recentred": {
        "factors_zyx": [1.0, 1.0, 1.0],
        "um_per_unit": 0.40625,
        "isotropic_in_physical_space": False,
        "recentre_z_onto_publisher_mean": True,   # offset computed from the data, not assumed
        "what_it_is": "index space with z TRANSLATED onto the shipped z mean. A translation is "
                      "free in the attention geometry (RoPE + Householder are difference-only) "
                      "and it is the only lever that moves slot 1 without moving the geometry",
    },
    "um_isotropic_1625": {
        "factors_zyx": [1.625, 1.625, 1.625],
        "um_per_unit": 1.0,
        "isotropic_in_physical_space": True,
        "what_it_is": "THE KNOWN-WRONG CONVENTION. Recorded in AGENTS.md section 4 as the "
                      "mistake made in two of three distance analyses in one day, and used as "
                      "the discriminator in FACT-0447's calibration",
    },
}

# Part C's bar. A STATED CONVENTION, not a measured failure threshold - said out loud because a
# bar invented to produce a verdict is the shape this project's gates exist to prevent.
SIGMA_BAR = 3.0


def gt_voxel_deltas(geff: Path) -> np.ndarray:
    """Per-GT-edge (dz, dy, dx) in VOXELS, consecutive frames only.

    Same construction as the committed revladder_calibrate.py, deliberately - the anchor must be
    reproduced by the same quantity it was measured on.
    """
    from biotrack.metric import load_graph

    g = load_graph(geff)
    n = g.node_attrs().to_pandas()
    ids = n["node_id"].to_numpy().astype(np.int64)
    pos = n[["z", "y", "x"]].to_numpy().astype(np.float64)
    t = n["t"].to_numpy().astype(np.int64)
    row = {int(v): i for i, v in enumerate(ids)}
    e = g.edge_attrs().to_pandas()
    out = []
    for s, d in zip(e["source_id"], e["target_id"]):
        i, j = row.get(int(s)), row.get(int(d))
        if i is None or j is None or t[j] - t[i] != 1:
            continue
        out.append(pos[j] - pos[i])
    return np.asarray(out, dtype=np.float64).reshape(-1, 3)


def gt_node_coords(geff: Path) -> np.ndarray:
    from biotrack.metric import load_graph

    g = load_graph(geff)
    n = g.node_attrs().to_pandas()
    return n[["t", "z", "y", "x"]].to_numpy().astype(np.float64)


# ---------------------------------------------------------------------------------------------
# PART A - metric fidelity against FACT-0040
# ---------------------------------------------------------------------------------------------
def part_a(delta_vox: np.ndarray) -> dict:
    rows = []
    for name, cfg in CONVENTIONS.items():
        f = np.asarray(cfg["factors_zyx"], dtype=np.float64)
        d_units = np.linalg.norm(delta_vox * f, axis=1)
        d_um = d_units * cfg["um_per_unit"]
        median = float(np.median(d_um))
        passes = abs(median - ANCHOR_UM) <= TOLERANCE_UM
        rows.append({
            "convention": name,
            "median_um": median,
            "p25_um": float(np.percentile(d_um, 25)),
            "p75_um": float(np.percentile(d_um, 75)),
            "p99_um": float(np.percentile(d_um, 99)),
            "median_in_convention_units": float(np.median(d_units)),
            "passes": bool(passes),
        })
    # The real motion distribution, per axis, in microns. This is the quantity a synthetic
    # deformation must be calibrated against (PKT-0048 task 4) - a T=2 dataset whose warps are
    # not calibrated here is the packet's own falsifier.
    d_um_axis = np.abs(delta_vox * UM_PER_VOXEL)
    per_axis = {}
    for i, ax in enumerate("zyx"):
        v = d_um_axis[:, i]
        per_axis[ax] = {
            "median_um": float(np.median(v)),
            "p75_um": float(np.percentile(v, 75)),
            "p95_um": float(np.percentile(v, 95)),
            "p99_um": float(np.percentile(v, 99)),
            "max_um": float(v.max()),
            "median_voxels": float(np.median(np.abs(delta_vox[:, i]))),
            "p99_voxels": float(np.percentile(np.abs(delta_vox[:, i]), 99)),
        }

    return {
        "anchor_fact": "FACT-0040",
        "anchor_um": ANCHOR_UM,
        "tolerance_um": TOLERANCE_UM,
        "n_gt_edges": int(len(delta_vox)),
        "real_motion_per_axis_um": per_axis,
        "rows": rows,
        "discriminates": not all(r["passes"] for r in rows),
        "what_it_can_and_cannot_do": "it catches a converter that mislabels its own units. It "
                                     "CANNOT choose between two self-consistent conventions - "
                                     "parts B and C do that",
    }


# ---------------------------------------------------------------------------------------------
# PART B - receptive field, against the compiled tau
# ---------------------------------------------------------------------------------------------
def part_b(tau: float, median_um: float) -> dict:
    z, y, x = CROP_SHAPE_TZYX[1] - 1, CROP_SHAPE_TZYX[2] - 1, CROP_SHAPE_TZYX[3] - 1
    rows = []
    for name, cfg in CONVENTIONS.items():
        f = np.asarray(cfg["factors_zyx"], dtype=np.float64)
        extent = np.array([z, y, x], dtype=np.float64) * f
        diag = float(np.linalg.norm(extent))
        # tau expressed in microns per axis: how far the cutoff reaches physically along each axis
        um_per_unit_axis = UM_PER_VOXEL / f
        tau_um_axis = tau * um_per_unit_axis
        aniso = float(tau_um_axis.max() / tau_um_axis.min())
        rows.append({
            "convention": name,
            "crop_extent_units_zyx": extent.tolist(),
            "crop_diagonal_units": diag,
            "tau_covers_whole_crop": bool(tau >= diag),
            "tau_is_a_live_mechanism": bool(tau < diag),
            "tau_um_per_axis_zyx": tau_um_axis.tolist(),
            "crop_diagonal_over_tau": diag / tau,
            "receptive_field_anisotropy": aniso,
            "tau_over_median_displacement": (
                float(tau * float(um_per_unit_axis[1])) / median_um if median_um else None
            ),
            "isotropic_in_physical_space": cfg["isotropic_in_physical_space"],
            "verdict": (
                "REJECT - the cutoff never fires, so a mechanism the model was trained with is "
                "silently disabled"
                if tau >= diag else
                "REJECT - the 300-ball is 4x longer in z than in xy, so the model's isotropic "
                "neighbourhood, RoPE and learned distance prior all see a squashed volume"
                if aniso > 1.05 else
                "ADMISSIBLE on receptive field"
            ),
        })
    return {
        "tau_in_node_pos_units": tau,
        "where_tau_binds": [
            "EdgeModel.forward attention mask over node_pos (compiled, not configurable)",
            "EdgeModel.forward attention mask over edge_pos (compiled)",
            "_api.py:206 predict(distance_threshold=300.0) candidate radius (configurable)",
        ],
        "crop_shape_tzyx": list(CROP_SHAPE_TZYX),
        "crop_shape_provenance": "measured this cycle: all 199 data/train crops are "
                                 "(100, 64, 256, 256)",
        "rows": rows,
    }


# ---------------------------------------------------------------------------------------------
# PART C - where our coordinates land in the shipped standardisation
# ---------------------------------------------------------------------------------------------
def part_c(coords_tzyx: np.ndarray) -> dict:
    """Slots 0..3 of the 19-vector are t, z, y, x, standardized by _api.py:18-60.

    This is the check that decides the route. It needs no GPU, no masks and no model run.
    """
    names = ("t", "z", "y", "x")
    rows = []
    for name, cfg in CONVENTIONS.items():
        f = np.array([1.0, *cfg["factors_zyx"]], dtype=np.float64)
        c = coords_tzyx * f
        offset_z = 0.0
        if cfg.get("recentre_z_onto_publisher_mean"):
            offset_z = float(PUBLISHER_MEAN[1] - c[:, 1].mean())
            c = c.copy()
            c[:, 1] = c[:, 1] + offset_z
        per = []
        for i, nm in enumerate(names):
            mu, sd = PUBLISHER_MEAN[i], PUBLISHER_STD[i]
            sig = (c[:, i] - mu) / sd
            per.append({
                "slot": i,
                "name": nm,
                "publisher_mean": mu,
                "publisher_std": sd,
                "our_mean": float(c[:, i].mean()),
                "our_max": float(c[:, i].max()),
                "sigma_at_our_mean": float(sig.mean()),
                "sigma_at_our_max": float(sig.max()),
                "share_beyond_3_sigma": float(np.mean(np.abs(sig) > 3.0)),
            })
        worst = max(per, key=lambda r: abs(r["sigma_at_our_max"]))
        rows.append({
            "convention": name,
            "per_slot": per,
            "worst_slot": worst["name"],
            "worst_sigma": worst["sigma_at_our_max"],
            "z_offset_applied": offset_z,
            "admissible": bool(abs(worst["sigma_at_our_max"]) <= SIGMA_BAR),
        })
    return {
        "what_this_measures": "the position columns enter TWICE - raw into the attention "
                              "geometry, and standardized into the feature MLP at slots 1..3. "
                              "This is the standardized path",
        "constants_source": "hoct/_api.py:18-60 _MEAN/_STD, verbatim",
        "sigma_bar": SIGMA_BAR,
        "sigma_bar_status": "A STATED CONVENTION, NOT A MEASURED FAILURE THRESHOLD. Nothing "
                            "here shows the network breaks at 3 sigma - it shows how far OUR "
                            "data sits from the distribution the shipped constants describe. "
                            "Read the sigma figures, not the boolean",
        "rows": rows,
        "structural_finding": {
            "claim": "no convention places our z inside the shipped z distribution, and the two "
                     "paths want OPPOSITE corrections",
            "mechanism": "the shipped z statistics are mean 2.938 / std 7.600 - a distribution "
                         "concentrated within a few index units of zero, which is what "
                         "graph.py:16-40 produces for 2D input (z := 0.0). Our nodes fill z in "
                         "[1, 62]. Making the geometry metrically correct (z x 4) makes the "
                         "standardized slot WORSE, not better",
            "corroboration": "hoct_feature_contract.prove_slot_map()'s perpendicular-axis check: "
                             "the shipped inertia means satisfy I_zz = I_yy + I_xx to 1.7%, "
                             "which is the signature of a PLANAR mask corpus",
            "provenance_ceiling": "INFERENCE FROM SHIPPED CONSTANTS, not a training manifest. "
                                  "FACT-0451 stands: the training data cannot be established",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, default=ROOT / "data" / "train")
    ap.add_argument("--prefix", default="6bba", help="FACT-0040 is a fold-1 anchor")
    ap.add_argument("--max-crops", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tau_info = read_compiled_tau()
    if not tau_info.get("found"):
        payload = {
            "heartbeat": HEARTBEAT_REFUSED,
            "passes": False,
            "refusal": "the compiled tau could not be re-derived from the checkpoint; the gate "
                       "refuses to substitute the paper's number",
            "tau": tau_info,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(HEARTBEAT_REFUSED, tau_info.get("why"))
        return 2

    geffs = sorted(args.gt_dir.glob(f"{args.prefix}_*.geff"))
    if args.max_crops:
        geffs = geffs[: args.max_crops]
    if not geffs:
        raise SystemExit(f"no GT geffs under {args.gt_dir} with prefix {args.prefix}")

    deltas, coords = [], []
    for i, g in enumerate(geffs, 1):
        deltas.append(gt_voxel_deltas(g))
        coords.append(gt_node_coords(g))
        if i % 25 == 0 or i == len(geffs):
            print(f"  [{i}/{len(geffs)}] {g.stem}", flush=True)
    delta = np.concatenate(deltas)
    coord = np.concatenate(coords)

    a = part_a(delta)
    ref_median = next(r["median_um"] for r in a["rows"] if r["convention"] == "um_anisotropic")
    b = part_b(tau_info["tau_in_node_pos_units"], ref_median)
    c = part_c(coord)

    rejections = []
    for r in a["rows"]:
        if not r["passes"]:
            rejections.append({"part": "A", "convention": r["convention"],
                               "why": f"median {r['median_um']:.3f} um misses the FACT-0040 "
                                      f"anchor {ANCHOR_UM} +/- {TOLERANCE_UM}"})
    for r in b["rows"]:
        if r["verdict"].startswith("REJECT"):
            rejections.append({"part": "B", "convention": r["convention"], "why": r["verdict"]})
    for r in c["rows"]:
        if not r["admissible"]:
            rejections.append({"part": "C", "convention": r["convention"],
                               "why": f"slot {r['worst_slot']} sits at "
                                      f"{r['worst_sigma']:.2f} sigma of the shipped distribution"})

    survivors = [
        n for n in CONVENTIONS
        if not any(x["convention"] == n for x in rejections)
    ]
    payload = {
        "schema_version": 1,
        "packet": "PKT-0048",
        "instrument": "scripts/win_bet/hoct_scale_gate.py",
        "binding_restriction": {
            "fact": "FACT-0451",
            "text": "general_v1's training data CANNOT BE ESTABLISHED and overlap with the "
                    "competition movies cannot be excluded; submission-only judgement",
            "offline_scoreable": False,
        },
        "prefix": args.prefix,
        "n_crops": len(geffs),
        "n_gt_edges": int(len(delta)),
        "n_gt_nodes": int(len(coord)),
        "tau": tau_info,
        "part_a_metric_fidelity": a,
        "part_b_receptive_field": b,
        "part_c_input_distribution": c,
        "rejections": rejections,
        "surviving_conventions": survivors,
        "gate_discriminates": bool(rejections),
        "verdict": (
            "NO CONVENTION SURVIVES ALL THREE PARTS" if not survivors else
            f"surviving: {', '.join(survivors)}"
        ),
    }
    ok = bool(rejections) and a["rows"][0]["passes"]
    payload["heartbeat"] = HEARTBEAT_OK if ok else HEARTBEAT_REFUSED
    payload["passes"] = ok

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n{payload['heartbeat']}")
    print(f"  compiled tau = {tau_info['tau_in_node_pos_units']:.0f} node_pos units "
          f"(squared cutoff {tau_info['squared_cutoff']:g}, {tau_info['occurrences']} sites), "
          f"read WITHOUT torch.jit.load")
    print("\n  PART A  metric fidelity vs FACT-0040")
    for r in a["rows"]:
        print(f"    {r['convention']:<22} median {r['median_um']:8.3f} um   "
              f"{'PASS' if r['passes'] else 'FAIL'}")
    print("\n  PART B  receptive field (crop diagonal vs the compiled 300)")
    for r in b["rows"]:
        print(f"    {r['convention']:<22} diag {r['crop_diagonal_units']:8.1f}  "
              f"tau_live={str(r['tau_is_a_live_mechanism']):<5} aniso={r['receptive_field_anisotropy']:.2f}  "
              f"{r['verdict'][:58]}")
    print("\n  PART C  our coordinates in the shipped standardisation")
    for r in c["rows"]:
        print(f"    {r['convention']:<22} worst slot '{r['worst_slot']}' at "
              f"{r['worst_sigma']:7.2f} sigma  {'OK' if r['admissible'] else 'REJECT'}")
    print(f"\n  rejections: {len(rejections)}   surviving: {survivors or 'NONE'}")
    print(f"  -> {args.out}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
