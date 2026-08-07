"""Local build gate for the P3+D1 smoke specs. Must be green before any push.

Verifies the COMPOSITION, not the science. Emits a compact report mapping
edit_id -> source -> cell -> match count -> before/after hash.

Runs over EVERY cell of the 2x2 checkpoint x family factorial, in the manifest's stage
order. It used to iterate `for fold in (0, 1)` and assert that no crop of the checkpoint's
own training family appeared -- which is correct for a TARGET cell and exactly wrong for a
SOURCE one, so it encoded the routed-only export that made D1-F unrunnable. The check is now
role-aware and derives the role instead of reading it.
"""
from __future__ import annotations

import ast
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PREDICT = ROOT / "vendor" / "kaggle-cell-tracking" / "scripts" / "predict_unet_transformer.py"
FAILURES: list[str] = []

sys.path.insert(0, str(ROOT / "scripts"))
import assemble_p3_d1_smoke_spec as ASM  # noqa: E402  (manifest + cell-key authority)


def check(cond, msg):
    if not cond:
        FAILURES.append(msg)
    return cond


def cells(nb_path):
    return ["".join(c["source"]) for c in json.load(open(nb_path, encoding="utf-8"))["cells"]]


def h(s):
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def gate(key: str) -> dict:
    spec_p = ROOT / "scripts" / "kaggle_specs" / f"p3_d1_smoke_{key}.json"
    spec = json.loads(spec_p.read_text(encoding="utf-8"))
    built_p = ROOT / spec["out_dir"] / spec["code_file"]
    base_p = ROOT / spec["base_notebook"]
    base, built = cells(base_p), cells(built_p)
    allb = "\n".join(built)

    # ---- 1. every changed cell must be explained by >= 1 edit -----------------------
    changed = [i for i in range(max(len(base), len(built)))
               if (base[i] if i < len(base) else None) != (built[i] if i < len(built) else None)]
    man = json.loads((ROOT / spec["out_dir"] / "build_manifest.json").read_text())
    touched: dict[int, list[str]] = {}
    for rec, ed in zip(man["edits"], spec["edits"]):
        for ci in rec["cells"]:
            touched.setdefault(ci, []).append(ed["_audit"]["edit_id"])
    unexplained = [c for c in changed if c not in touched]
    check(not unexplained, f"{key}: UNEXPLAINED changed cells {unexplained}")
    check(set(touched) <= set(changed), f"{key}: edit touched an unchanged cell")

    # ---- 2. harmonic must never touch detection -------------------------------------
    harm = [e for e in spec["edits"] if e["_audit"]["edit_id"] == "E04_harmonic"][0]
    check("det_logits" not in harm["old"] and "det_logits" not in harm["new"],
          f"{key}: harmonic edit touches det_logits")
    check("harmonic_prob = 1.0 / (" in allb, f"{key}: harmonic fusion missing from build")
    check("+ _bidirectional_weight * reverse_aligned" not in allb,
          f"{key}: arithmetic blend still present")

    # ---- 3. D1 injection present exactly once in the NOTEBOOK -------------------------
    # Notebook-level counts are about the injector being present once, not about call
    # counts: `_D1_BLOCK_B64` legitimately appears twice (definition + decode) and
    # `_d1_audit_frame` several times (idempotency guard, import payload, call payload).
    # The counts that actually matter are asserted on the PATCHED PREDICT SCRIPT below.
    for sym, want in {"_D1_BLOCK_B64 = (": 1, "_d1_mod.write_bytes": 1,
                      'if "_d1_audit_frame" in _s:': 1,
                      "expected exactly 1": 1}.items():
        check(allb.count(sym) == want,
              f"{key}: D1 injector marker {sym!r} x{allb.count(sym)} want {want}")

    # ---- 4. invariant fixes exactly once ---------------------------------------------
    for sym in ("out_degree_now", "safe_division_skipped_outdegree",
                "nid not in incoming", "target_id in incoming",
                "def assert_degree_invariants("):
        check(allb.count(sym) >= 1, f"{key}: invariant symbol {sym!r} missing")
    check(allb.count("def assert_degree_invariants(") == 1,
          f"{key}: assert_degree_invariants defined more than once")

    # ---- 5. routing, stems, checkpoint ------------------------------------------------
    cell = spec["d1_cell"]
    split = int(cell["split"])
    check(f"edge_predictor_best_split_{split}.pth" in allb, f"{key}: wrong checkpoint glob")
    check(spec["provenance"]["checkpoint_sha256"] in allb, f"{key}: ckpt sha not in build")
    for stem in spec["provenance"]["stems"]:
        check(stem in allb, f"{key}: stem {stem} missing")

    # ---- 5b. ROLE, derived from the crops and the checkpoint -- never read -------------
    # The old form of this check said "no crop of the checkpoint's own training family may
    # appear", which is the TARGET rule applied to every cell. It made a SOURCE cell -- the
    # checkpoint over its own family, which is what supplies the fitting rows -- ungateable,
    # and with it the whole C1 2x2.
    trained_on = ASM.SPLIT_SOURCE_FAMILY[split]
    held_out = ASM.SPLIT_HELDOUT_FAMILY[split]
    fams = sorted({s.split("_")[0] for s in spec["provenance"]["stems"]})
    check(fams == [cell["family"]],
          f"{key}: crops span families {fams}; a factorial cell is family-pure")
    role = "source" if cell["family"] == trained_on else "target"
    check(role == cell["role"],
          f"{key}: declares role {cell['role']} but split {split} trained on {trained_on} "
          f"and these crops are {cell['family']}, i.e. {role}")
    if role == "target":
        check(cell["family"] == held_out,
              f"{key}: TARGET cell runs the checkpoint over {cell['family']}, which split "
              f"{split} was TRAINED on -- the measurement would be in-sample")
    else:
        check(cell["family"] == trained_on,
              f"{key}: SOURCE cell runs the checkpoint over {cell['family']}, which split "
              f"{split} never saw -- fitting rows must come from the training family")
    check(cell["shard_id"] in allb, f"{key}: shard_id absent from the build")
    check(allb.count("D1 FACTORIAL CELL IDENTITY") == 1,
          f"{key}: the cell-identity block is not present exactly once")
    check(allb.count('(_d1c_dir / "d1_cell.json").write_text(') == 1,
          f"{key}: the kernel does not emit d1_cell.json")
    check("ROLE INVERSION" in allb, f"{key}: the runtime role re-derivation is missing")

    # ---- 6. kernel metadata -----------------------------------------------------------
    meta = json.loads((ROOT / spec["out_dir"] / "kernel-metadata.json").read_text())
    check(meta["enable_internet"] is False, f"{key}: internet ON")
    check(meta["is_private"] is True, f"{key}: kernel not private")
    check(meta["enable_gpu"] is True, f"{key}: GPU off")
    check(spec["expects_submission"] is False, f"{key}: expects_submission true")

    # ---- 7. retention keeps every artifact --------------------------------------------
    check('"d1_audit"' in allb, f"{key}: d1_audit not in the retention keep-set")
    check("pregraphs_split{LOEO_FOLD}.parquet" in allb, f"{key}: pregraph not retained")

    # ---- 8. patched predict script compiles + idempotent -------------------------------
    inj = (ROOT / "scripts" / "kaggle_edits" / "d1_inject.py").read_text(encoding="utf-8")
    ps_src = PREDICT.read_text(encoding="utf-8")
    simdir = ROOT / "artifacts" / "_gate_sim"
    simdir.mkdir(parents=True, exist_ok=True)

    class FakePS:
        parent = simdir
        def __init__(self, txt): self._t = txt; self.out = None
        def read_text(self): return self._t
        def write_text(self, t): self.out = t

    f1 = FakePS(ps_src)
    exec(compile(inj, "inject", "exec"), {"_ps": f1})
    patched = f1.out
    check(patched is not None, f"{key}: injection produced no output")
    try:
        tree = ast.parse(patched)
        check(True, "")
    except SyntaxError as e:
        check(False, f"{key}: patched predict script does not parse: {e}")
        tree = None
    check("@torch.no_grad()\ndef predict_video(" in patched,
          f"{key}: decorator split from predict_video")
    check(patched.count("_d1_audit_frame(") == 1, f"{key}: audit call not exactly once")
    check(patched.count("_d1_flush(") == 1, f"{key}: flush call not exactly once")
    f2 = FakePS(patched)
    exec(compile(inj, "inject", "exec"), {"_ps": f2})
    check(f2.out is None, f"{key}: injection is NOT idempotent")

    # ---- composition report -----------------------------------------------------------
    rows = []
    for rec, ed in zip(man["edits"], spec["edits"]):
        a = ed["_audit"]
        rows.append({"order": a["order"], "edit_id": a["edit_id"], "source": a["source"],
                     "cells": rec["cells"], "kind": ed["kind"],
                     "canonical_sha256": a["canonical_sha256"][:16], "intent": a["intent"]})
    return {"cell": key, "shard_id": cell["shard_id"], "stage": cell["stage"],
            "split": split, "family": cell["family"], "role": role,
            "crops": list(cell["crops"]),
            "spec_sha256": h(spec_p.read_text(encoding="utf-8")),
            "notebook_sha256": hashlib.sha256(built_p.read_bytes()).hexdigest(),
            "changed_cells": changed, "cell_to_edits": {str(k): v for k, v in touched.items()},
            "patched_predict_sha256": hashlib.sha256(patched.encode()).hexdigest(),
            "edits": rows}


def main():
    # Cell order and cell keys come from the manifest, not from a literal in this file.
    keys = [ASM.cell_key(s) for s in ASM.load_shards()]
    report = {"cells": [gate(k) for k in keys]}
    out = ROOT / "reports" / "inventory" / "p3_d1_smoke_composition.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for fr in report["cells"]:
        print(f"\n=== stage {fr['stage']} — {fr['shard_id']} — split {fr['split']} "
              f"{fr['family']} {fr['role'].upper()} — {len(fr['crops'])} crop(s) "
              f"{fr['crops']} — {len(fr['edits'])} edits, "
              f"cells changed {fr['changed_cells']} ===")
        print(f"  {'ord':>3s} {'edit_id':22s} {'cells':10s} {'kind':14s} {'sha':18s} source")
        for r in fr["edits"]:
            print(f"  {r['order']:3d} {r['edit_id']:22s} {str(r['cells']):10s} "
                  f"{r['kind']:14s} {r['canonical_sha256']:18s} {r['source']}")
        print(f"  notebook sha256 {fr['notebook_sha256'][:16]}  "
              f"patched-predict sha256 {fr['patched_predict_sha256'][:16]}")

    print("\n" + "=" * 70)
    covered = {(fr["split"], c) for fr in report["cells"] for c in fr["crops"]}
    print(f"crop-inferences covered: {len(covered)} "
          f"(want 6 = 3 crops x 2 checkpoints)")
    for split in (0, 1):
        got = sorted(c for s, c in covered if s == split)
        print(f"  split_{split}: {got}")
    check(len(covered) == 6, f"factorial covers {len(covered)} crop-inferences, want 6")

    if FAILURES:
        print(f"GATE FAILED — {len(FAILURES)} problem(s):")
        for f in FAILURES:
            print("  -", f)
        sys.exit(1)
    print("GATE PASSED — composition verified, safe to push")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
