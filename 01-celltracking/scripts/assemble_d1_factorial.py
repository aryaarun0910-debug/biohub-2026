"""Assemble fetched D1 factorial kernel outputs into the layout d1f_probe expects.

Four kernels run one cell each of the 2x2 checkpoint x family factorial. Each returns its
own `d1_audit/` directory. `d1f_probe.load_factorial` reads a DIFFERENT shape:

    <root>/basis_0/   one v6 audit dir holding BOTH families encoded by split 0
    <root>/basis_1/   one v6 audit dir holding BOTH families encoded by split 1

with a single merged `d1_manifest.json` per basis whose `crops` unions the source and the
target cell. Nothing performed that merge, so the four outputs could be fetched and still
be unreadable. This is that step.

Why a basis is the unit and not a fold: correction C1. The two checkpoints induce
INDEPENDENT 32-D bases (rel-L2 1.41542 ~ sqrt(2), cos(w0,w1) = -0.155). A head fitted in
one basis may never be applied in the other, so the source rows and the target rows a head
is fitted and evaluated on must have been encoded by the SAME checkpoint. `basis_N/` is
that guarantee made structural: if the merge is wrong, the directory layout is wrong, and
`assert_same_basis` fires downstream instead of quietly returning a number.

Within one basis the two cells are family-disjoint, so their per-crop filenames cannot
collide; across bases the same crop names recur, which is exactly why the namespacing
exists. A collision inside a basis means two cells claimed the same crop and is refused.

The assembler trusts nothing it can re-derive: it re-checks each cell's role against
SPLIT_SOURCE_FAMILY, re-checks the observed checkpoint hash the kernel recorded, and
validates its own output through d1f_probe.validate_manifest before returning 0. It
therefore cannot emit a directory the probe would reject.

Usage:
    python scripts/assemble_d1_factorial.py --cell <fetched_dir> [--cell ...] \
        --out <root> [--tier smoke] [--force]

Each `--cell` is a directory containing `d1_audit/`, or the `d1_audit/` directory itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import assemble_p3_d1_smoke_spec as ASM  # noqa: E402  (family/split authority)
import d1f_probe as PROBE  # noqa: E402  (the consumer defines the contract)

# Promoted per-crop declarations that must agree across the two cells of one basis. A
# basis is one measurement; two values of any of these inside it means the halves are not
# comparable and no merge can make them so.
BASIS_SCALARS = ("schema_version", "match_um", "search_um", "tta_view_set",
                 "n_encode_calls", "n_distinct_views", "checkpoint_sha256")
# Cell identity fields stamped onto every crop record, closing the loop the aggregator
# leaves open: it records fold and checkpoint but not which factorial cell produced a row.
CELL_STAMP = ("shard_id", "stage", "role", "family", "tier")


class AssemblyError(RuntimeError):
    """A structural defect in the fetched artifacts. Never downgraded to a warning."""


def _audit_dir(p: pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(p).resolve()
    if (p / "d1_cell.json").exists():
        return p
    if (p / "d1_audit" / "d1_cell.json").exists():
        return p / "d1_audit"
    raise AssemblyError(
        f"{p} holds no d1_cell.json, at its root or under d1_audit/. A kernel that ran "
        f"the identity block always writes one; its absence means this output predates "
        f"the 2x2 build or is not a factorial cell at all.")


def read_cell(path: pathlib.Path) -> dict:
    """Load one fetched cell and re-derive everything about it that is derivable."""
    d = _audit_dir(path)
    cell = json.loads((d / "d1_cell.json").read_text(encoding="utf-8"))
    man_p = d / "d1_manifest.json"
    if not man_p.exists():
        raise AssemblyError(f"{d}: no d1_manifest.json; the aggregator cell never ran")
    man = json.loads(man_p.read_text(encoding="utf-8"))

    shard = cell.get("shard_id", "<unnamed>")
    bad: list[str] = []
    if not man.get("COMPLETE"):
        bad.append(f"manifest COMPLETE={man.get('COMPLETE')!r}, "
                   f"problems={man.get('problems')}")
    split = int(cell["split"])
    if split not in ASM.SPLIT_SOURCE_FAMILY:
        raise AssemblyError(f"{shard}: split {split} is not a checkpoint split")

    # Role is a function of (split, family). Re-derive rather than read -- the kernel
    # already did this once, and an artifact can be moved after the kernel is gone.
    role = "source" if cell["family"] == ASM.SPLIT_SOURCE_FAMILY[split] else "target"
    if role != cell.get("role"):
        bad.append(f"ROLE INVERSION: split {split} trained on "
                   f"{ASM.SPLIT_SOURCE_FAMILY[split]}, crops are {cell['family']} "
                   f"=> {role}, but the artifact says {cell.get('role')}")
    for field in ("family_rederived", "role_rederived"):
        want = cell["family"] if field.startswith("family") else role
        if field in cell and cell[field] != want:
            bad.append(f"{field}={cell[field]!r} != {want!r}")
    obs = cell.get("checkpoint_sha256_observed")
    if obs is not None and obs != cell["checkpoint_sha256"]:
        bad.append(f"the kernel loaded weights {obs} but the cell pins "
                   f"{cell['checkpoint_sha256']}")
    if cell["checkpoint_sha256"] != ASM.CKPT_SHA[split]:
        bad.append(f"checkpoint {cell['checkpoint_sha256'][:16]} is not split {split}'s")
    if man.get("checkpoint_sha256") not in (None, cell["checkpoint_sha256"]):
        bad.append(f"manifest checkpoint {man['checkpoint_sha256']!r} != the cell's")

    declared = sorted(cell["crops"])
    if sorted(man.get("crops", {})) != declared:
        bad.append(f"manifest crops {sorted(man.get('crops', {}))} != declared {declared}")
    off_family = [c for c in declared if c.split("_")[0] != cell["family"]]
    if off_family:
        bad.append(f"crops {off_family} are not family {cell['family']}; a cell is pure")
    if bad:
        raise AssemblyError(f"cell {shard} is not what it claims to be:\n  "
                            + "\n  ".join(bad))
    return {"dir": d, "cell": cell, "manifest": man, "split": split, "role": role,
            "family": cell["family"], "shard_id": shard, "crops": declared}


def _merge_basis(split: int, cells: list[dict], out_dir: pathlib.Path) -> dict:
    """Copy both cells' payloads into one basis dir and merge their manifests."""
    by_role = {c["role"]: c for c in cells}
    if sorted(by_role) != ["source", "target"] or len(cells) != 2:
        raise AssemblyError(
            f"basis {split} has cells {[(c['shard_id'], c['role']) for c in cells]}; a "
            f"basis needs exactly one source and one target. A source-less basis is the "
            f"routed-only export that made the probe refuse to run in the first place.")
    fams = {c["role"]: c["family"] for c in cells}
    if fams["source"] != ASM.SPLIT_SOURCE_FAMILY[split]:
        raise AssemblyError(f"basis {split} source family {fams['source']} is not "
                            f"{ASM.SPLIT_SOURCE_FAMILY[split]}")
    if fams["target"] != ASM.SPLIT_HELDOUT_FAMILY[split]:
        raise AssemblyError(f"basis {split} target family {fams['target']} is not "
                            f"{ASM.SPLIT_HELDOUT_FAMILY[split]}")

    scalars: dict = {}
    for field in BASIS_SCALARS:
        vals = {c["shard_id"]: c["manifest"].get(field) for c in cells}
        distinct = {json.dumps(v, sort_keys=True, default=str) for v in vals.values()}
        if len(distinct) > 1:
            raise AssemblyError(
                f"basis {split}: the two cells disagree on {field}: {vals}. One basis is "
                f"one measurement; halves that disagree here are not comparable.")
        one = next(iter(vals.values()))
        if one is None:
            raise AssemblyError(
                f"basis {split}: neither cell records {field}. The probe blocks without "
                f"it, so assembling would only move the failure later.")
        scalars[field] = one

    out_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    crops: dict = {}
    for c in sorted(cells, key=lambda x: x["role"]):
        for src in sorted(c["dir"].iterdir()):
            if src.is_dir() or src.name in ("d1_manifest.json", "d1_cell.json"):
                continue
            if not src.name.startswith(tuple(f"{k}__" for k in c["crops"])):
                continue
            dst = out_dir / src.name
            if src.name in copied:
                raise AssemblyError(
                    f"basis {split}: {src.name} supplied by two cells. Within a basis the "
                    f"families are disjoint, so a collision means two cells claimed the "
                    f"same crop.")
            shutil.copy2(src, dst)
            copied[src.name] = hashlib.sha256(dst.read_bytes()).hexdigest()
        for crop, rec in c["manifest"]["crops"].items():
            rec = dict(rec)
            rec.update({k: c["cell"].get(k) for k in CELL_STAMP if k in c["cell"]})
            rec.setdefault("split", split)
            rec["encoder_split"] = split
            crops[crop] = rec

    merged = {"basis_split": split, "crops": crops,
              **{k: v for k, v in scalars.items()},
              "cells": [{"shard_id": c["shard_id"], "stage": c["cell"].get("stage"),
                         "role": c["role"], "family": c["family"], "crops": c["crops"]}
                        for c in sorted(cells, key=lambda x: x["cell"].get("stage", 0))]}
    # The consumer's own gate, run here so a bad merge cannot leave this function.
    summary = PROBE.validate_manifest(
        merged, expect_encode_calls=int(scalars["n_encode_calls"]))
    (out_dir / "d1_manifest.json").write_text(
        json.dumps(merged, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return {"split": split, "dir": str(out_dir), "files": copied,
            "n_files": len(copied), **summary}


def assemble(cell_paths, out_root: pathlib.Path, *, tier: str | None = None,
             force: bool = False) -> dict:
    out_root = pathlib.Path(out_root)
    cells = [read_cell(p) for p in cell_paths]
    seen: dict[str, str] = {}
    for c in cells:
        if c["shard_id"] in seen:
            raise AssemblyError(f"shard {c['shard_id']} supplied twice")
        seen[c["shard_id"]] = str(c["dir"])
    if tier:
        expected = {s["shard_id"] for s in ASM.load_shards()}
        missing, extra = expected - set(seen), set(seen) - expected
        if missing or extra:
            raise AssemblyError(
                f"tier {tier} expects {sorted(expected)}; missing={sorted(missing)} "
                f"unexpected={sorted(extra)}. A partial factorial is not a factorial: "
                f"the 2x2 exists to separate checkpoint from family, and three cells "
                f"cannot do that.")
    by_split: dict[int, list[dict]] = {}
    for c in cells:
        by_split.setdefault(c["split"], []).append(c)
    if sorted(by_split) != [0, 1]:
        raise AssemblyError(
            f"cells cover splits {sorted(by_split)}; the factorial needs both, because "
            f"the same checkpoint must encode the source and the target (C1).")

    for split in sorted(by_split):
        d = out_root / f"basis_{split}"
        if d.exists():
            if not force:
                raise AssemblyError(f"{d} exists; pass --force to overwrite")
            shutil.rmtree(d)
    bases = [_merge_basis(s, by_split[s], out_root / f"basis_{s}")
             for s in sorted(by_split)]
    report = {"kind": "d1_factorial_assembly", "tier": tier, "root": str(out_root),
              "sources": seen, "bases": bases}
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "assembly.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cell", action="append", required=True,
                    help="a fetched kernel output dir (repeat once per factorial cell)")
    ap.add_argument("--out", required=True, help="assembly root; basis_N/ created under it")
    ap.add_argument("--tier", help="if given, every shard of this tier must be present")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    try:
        rep = assemble(args.cell, pathlib.Path(args.out), tier=args.tier,
                       force=args.force)
    except AssemblyError as e:
        print(f"ASSEMBLY FAILED\n{e}", file=sys.stderr)
        return 1
    for b in rep["bases"]:
        cells = ", ".join(f"{c['role']}:{c['family']}"
                          for c in json.loads(
                              (pathlib.Path(b["dir"]) / "d1_manifest.json")
                              .read_text(encoding="utf-8"))["cells"])
        print(f"basis_{b['split']}: {b['n_crops']} crops, {b['n_files']} files  [{cells}]")
    print(f"\nASSEMBLED -> {rep['root']}\n"
          f"  read it with: d1f_probe.load_factorial({rep['root']!r}, "
          f"expect_encode_calls=8)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
