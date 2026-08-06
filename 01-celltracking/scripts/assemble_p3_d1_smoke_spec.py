"""Deterministic assembler for the combined P3 + D1 + D1-F smoke specs.

Nothing here is transcribed. The harmonic replacement and both invariant fixes are imported
from `p3_harmonic.json`; the LOEO retarget/routing and pregraph-export edits are imported from
`loeo_f1_strict_pregraph.json`; the D1 injection is the already-verified generated cell. Source
specs are SHA-asserted so later drift cannot silently change the composition.

WHY TWO SPECS RATHER THAN ONE KERNEL. The LOEO retarget block binds a single fold at module
level -- it reads BIOHUB_LOEO_FOLD once, mounts only that fold's crops, and REBINDS the notebook
global TEST_DIR, which the predict command, the wrapper's read_test_frame and the final audit all
read. Running two folds in one kernel would require restructuring the notebook's single-pass
predict/wrapper/export flow, i.e. editing its most load-bearing cell. The requirement that each
crop is encoded exactly once, and that the two manifests are aggregated only after both complete,
is satisfied by two runs of one generated spec pair:

    fold 0 -> split_0 checkpoint -> 44b6_0113de3b                     (parity anchor)
    fold 1 -> split_1 checkpoint -> 6bba_6feb10f0, 6bba_57b7cc1e      (high-miss + stress)

Determinism: sorted keys, no timestamps, no randomness. Generating twice must be byte-identical.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPECS = ROOT / "scripts" / "kaggle_specs"

BACKBONE = SPECS / "p3_harmonic.json"
LOEO = SPECS / "loeo_f1_strict_pregraph.json"

# Frozen smoke crops. 44b6_0113de3b has 0/52 missing GT and a retained pregraph, so it is the
# parity anchor and NOT an A/B/D test. The two 6bba crops carry the diagnostic load.
SMOKE = {
    0: {"stems": ["44b6_0113de3b"], "split": 0},
    1: {"stems": ["6bba_6feb10f0", "6bba_57b7cc1e"], "split": 1},
}

# Checkpoint hashes, verified locally from artifacts/kaggle/weights_dataset.
CKPT_SHA = {
    0: "d3e89eb361eeadef06d18159d834594776174a9f8cabd81f65271abbb42a492f",
    1: "2e4ebf616b3d4fb53881eadf42c7972e04978c28f74f229c5cb30bee835063de",
}


def sha_file(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sha_edit(e: dict) -> str:
    return hashlib.sha256(json.dumps(e, sort_keys=True).encode()).hexdigest()


def tag(edit: dict, edit_id: str, source: str, order: int, intent: str) -> dict:
    """Attach audit metadata. kaggle_factory ignores unknown keys; our tests do not."""
    out = dict(edit)
    out["_audit"] = {
        "edit_id": edit_id,
        "source": source,
        "canonical_sha256": sha_edit(edit),
        "intent": intent,
        "order": order,
    }
    return out


def build(fold: int) -> dict:
    backbone = json.loads(BACKBONE.read_text(encoding="utf-8"))
    loeo = json.loads(LOEO.read_text(encoding="utf-8"))
    b_edits, l_edits = backbone["edits"], loeo["edits"]

    # --- assert imported sources are the ones we validated -------------------------------
    # Pinned per-edit hashes. If a source spec is edited later, assembly FAILS LOUDLY rather
    # than silently composing a different kernel.
    PINNED = {
        1: "d8e8bd4dac7f38676e4b8b0dbb2b8f0d",   # loeo retarget
        2: "a546af45c3057302",                    # test_stems routing
        3: "7e2712e00a82f54c",                    # export cell
        4: "5cbb94850ba34054",                    # pregraph export
        5: "b6a6c809e5d33dbf",                    # keep list
    }
    for idx, want in PINNED.items():
        got = sha_edit(l_edits[idx])
        assert got.startswith(want[:16]), (
            f"loeo edit {idx} drifted: expected {want[:16]}..., got {got[:16]}..."
        )
    imported_sha = {
        "loeo_retarget": sha_edit(l_edits[1]),
        "loeo_test_stems": sha_edit(l_edits[2]),
        "loeo_export_cell": sha_edit(l_edits[3]),
        "loeo_pregraph": sha_edit(l_edits[4]),
        "loeo_keep": sha_edit(l_edits[5]),
    }

    harmonic = [e for e in b_edits if e["kind"] == "replace" and "_bi_new" in e.get("old", "")]
    assert len(harmonic) == 1, f"expected exactly one harmonic edit, got {len(harmonic)}"
    invariants = [e for e in b_edits if e is not harmonic[0]]
    assert len(invariants) == 9, f"expected 9 invariant edits, got {len(invariants)}"

    stems = SMOKE[fold]["stems"]
    split = SMOKE[fold]["split"]

    env_vars = {
        "BIOHUB_LOEO_FOLD": str(fold),
        "BIOHUB_LOEO_ARM": "strict",
        "BIOHUB_LOEO_LIMIT": "0",
        "BIOHUB_LOEO_STEMS": json.dumps(stems),
        "BIOHUB_LOEO_WEIGHTS_GLOB": f"/kaggle/input/*/edge_predictor_best_split_{split}.pth",
        "BIOHUB_LOEO_CONFIG_GLOB": f"/kaggle/input/*/config_split_{split}.json",
        "BIOHUB_D1_FOLD": str(fold),
        "BIOHUB_D1_CKPT_SHA": CKPT_SHA[split],
        "BIOHUB_D1_EXPECTED_CROPS": str(len(stems)),
        "BIOHUB_D1_N_UNIFORM": "64",
        "BIOHUB_D1_N_SUBTHR": "32",
    }

    # keep-list: derive from the imported edit rather than transcribing it, then extend so the
    # cleanup's rmtree cannot delete the audit directory.
    keep = dict(l_edits[5])
    anchor = '"pregraphs_split{LOEO_FOLD}.json"}'
    assert keep["new"].count(anchor) == 1, "keep-list shape changed; refusing to guess"
    keep["new"] = keep["new"].replace(
        anchor, '"pregraphs_split{LOEO_FOLD}.json",\n              "d1_audit"}', 1)

    d1_inject = {
        "kind": "insert_before",
        "anchor": "def list_test_stems() -> list[str]:",
        "cell_match": "_ps = REPO_DIR",
        "code_file": "scripts/kaggle_edits/d1_inject.py",
        "expect": 1,
    }

    # --- FROZEN SEMANTIC ORDER -----------------------------------------------------------
    ordered: list[dict] = []
    n = 0

    def add(edit, edit_id, source, intent):
        nonlocal n
        ordered.append(tag(edit, edit_id, source, n, intent))
        n += 1

    add({"kind": "env", "cell_match": "BIOHUB_PRESET", "vars": env_vars, "expect": 1},
        "E01_env", "assembler", "1. environment: LOEO fold routing + D1 knobs")
    add(l_edits[1], "E02_loeo_retarget", "loeo_f1_strict_pregraph#1",
        "1. LOEO retarget: mount fold crops, rebind TEST_DIR, resolve fold weights")
    add(l_edits[2], "E03_test_stems", "loeo_f1_strict_pregraph#2",
        "1. route stem discovery at the LOEO fold")
    add(harmonic[0], "E04_harmonic", "p3_harmonic#harmonic",
        "2. harmonic edge-logit fusion (edge_logits_pair ONLY)")
    for i, e in enumerate(invariants):
        add(e, f"E05_{i:02d}_invariant", "p3_harmonic#invariant",
            "3. degree-invariant fixes and export assertions")
    add(d1_inject, "E06_d1_inject", "scripts/kaggle_edits/d1_inject.py",
        "4. D1 audit + frozen-feature injection into the predict script")
    add(l_edits[4], "E07_pregraph", "loeo_f1_strict_pregraph#4",
        "5. pre-wrapper prediction-graph export")
    add(l_edits[3], "E08_export_cell", "loeo_f1_strict_pregraph#3",
        "6. LOEO export cell replaces the submission guard")
    add({"kind": "append_cell", "code_file": "scripts/kaggle_edits/d1_aggregate.py"},
        "E10_aggregator", "scripts/kaggle_edits/d1_aggregate.py",
        "6. parent aggregator: read immutable per-crop records, fail hard on any gap")
    add(keep, "E09_keep", "loeo_f1_strict_pregraph#5 (+d1_audit)",
        "6. artifact retention: keep pregraphs AND the d1_audit directory")

    return {
        "name": f"p3_d1_smoke_f{fold}",
        "slug": f"biohub-p3-d1-smoke-f{fold}",
        "title": f"Biohub P3 D1 Smoke F{fold}",
        "code_file": f"biohub-p3-d1-smoke-f{fold}.ipynb",
        "out_dir": f"notebooks/kaggle_p3_d1_smoke_f{fold}",
        "datasets": loeo["datasets"],
        "competition_sources": loeo["competition_sources"],
        "enable_gpu": True,
        "enable_internet": False,
        "is_private": True,
        "machine_shape": "NvidiaTeslaT4",
        "expects_submission": False,
        "base_notebook": backbone["base_notebook"],
        "base_sha256": backbone["base_sha256"],
        "purpose": (
            f"STRUCTURAL SMOKE, fold {fold}, {len(stems)} crop(s). Combined P3 + D1 + D1-F "
            "export: one encoder pass per crop emits the P3 pre-wrapper graph (harmonic "
            "association), accepted detector peaks, the exact A/B/D classification of every GT "
            "node with D stratified by distance to the nearest local maximum, and 32-D frozen "
            "features at BOTH the GT voxel and that maximum. Combinable because the harmonic "
            "patch touches edge_logits_pair only and never det_logits. NOT a scientific verdict "
            "and NOT a submission."
        ),
        "provenance": {
            "backbone": f"p3_harmonic.json sha256 {sha_file(BACKBONE)}",
            "loeo_source": f"loeo_f1_strict_pregraph.json sha256 {sha_file(LOEO)}",
            "imported_edit_sha256": imported_sha,
            "d1_inject_sha256": sha_file(ROOT / "scripts/kaggle_edits/d1_inject.py"),
            "d1_block_sha256": sha_file(ROOT / "scripts/kaggle_edits/d1_response_audit.py"),
            "checkpoint_sha256": CKPT_SHA[split],
            "stems": stems,
            "two_spec_rationale": (
                "The LOEO retarget binds one fold at module level and rebinds TEST_DIR globally, "
                "so two folds in one kernel would require restructuring the notebook's "
                "single-pass predict/wrapper/export flow. Each crop is still encoded exactly "
                "once; the two manifests are aggregated only after both runs complete."
            ),
        },
        "edits": ordered,
    }


def main() -> None:
    for fold in sorted(SMOKE):
        spec = build(fold)
        out = SPECS / f"p3_d1_smoke_f{fold}.json"
        text = json.dumps(spec, indent=2, sort_keys=False) + "\n"
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out.name}  edits={len(spec['edits'])}  "
              f"stems={spec['provenance']['stems']}  sha256={hashlib.sha256(text.encode()).hexdigest()[:16]}")


if __name__ == "__main__":
    main()
