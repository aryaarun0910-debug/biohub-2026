"""Add ONE env-var line to any parent notebook, and prove it is one change.

scripts/make_variant.py can only SUBSTITUTE an existing `os.environ[...]` line,
and refuses when the key is absent. Most of the interesting knobs are never set
in the notebook at all -- they sit at their `os.environ.get(key, default)`
default -- so changing one means ADDING a line, which is what s05
(BIOHUB_OUTPUT_MOTION_RELINK) and s06 (BIOHUB_OUTPUT_LINEFIT_WEIGHT) both did by
hand.

This does it from an arbitrary PARENT notebook, so variants chain: s08 is s05 +
a reorder, s09 is s08 + one line. Each build proves it differs from its stated
parent by exactly one added line, which is what the one-change-per-submission
rule in ABORT_RULES actually requires -- one change relative to the thing you
are comparing against, not relative to the 0.947 base.

    python scripts/make_env_variant.py \\
        --parent submissions/s08_reorder_on_s05/biohub-s08-reorder-on-s05.ipynb \\
        --key BIOHUB_OUTPUT_LINEFIT_WEIGHT --value 0.4 \\
        --anchor "os.environ['BIOHUB_OUTPUT_FILTER_SHORT_TRACKS'] = '1'" \\
        --slug biohub-s09-linefit04-on-s08 --title "Biohub S09 linefit04 on s08"

NOTE ON SLUGS: Kaggle derives the live kernel slug by slugifying the TITLE, not
the `id` in kernel-metadata.json (s05 shipped id=biohub-s05-no-relink and went
live at biohub-s05-no-motion-relink). This script ABORTS if slug != slugify(title).
"""
import argparse
import ast
import difflib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUARD_KEY = "_EXPECTED_NUMERIC"
DATASETS = ["pilkwang/biohub-tracking-support-pack-50ep-v1",
            "pilkwang/biohub-temporal-unet3d-seed314159-v1",
            "pilkwang/biohub-deepcenter-unet3d-center-prior-v1"]


def die(msg):
    sys.exit(f"ABORT: {msg}")


def slugify(t):
    return re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")


def only(lines, needle, what):
    hits = [i for i, l in enumerate(lines) if needle in l]
    if len(hits) != 1:
        die(f"{what}: {needle!r} matched {len(hits)} lines, expected 1")
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--value", required=True)
    ap.add_argument("--anchor", required=True, help="line to insert AFTER")
    ap.add_argument("--slug", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--out", help="default: submissions/<slug>")
    ap.add_argument("--cell", type=int, default=2)
    a = ap.parse_args()

    if a.slug != slugify(a.title):
        die(f"slug {a.slug!r} != slugify(title) {slugify(a.title)!r}; Kaggle "
            "uses the title, so the push would land at the second one")

    parent = ROOT / a.parent
    if not parent.exists():
        die(f"parent notebook not found: {parent}")
    nb = json.loads(parent.read_text())
    before = [l for c in nb["cells"] for l in c.get("source", [])]
    cell = nb["cells"][a.cell]
    if cell["cell_type"] != "code":
        die(f"cell {a.cell} is {cell['cell_type']}, expected code")
    src_list = list(cell["source"])
    old_src = "".join(src_list)

    new_line = f"os.environ['{a.key}'] = '{a.value}'\n"
    print(f"parent : {a.parent}  ({len(before)} source lines)")
    print(f"adding : {new_line.strip()}")

    if any(f"'{a.key}'" in l and "os.environ[" in l and "=" in l.split("os.environ[")[0] + l
           and l.lstrip().startswith("os.environ[") for l in src_list):
        die(f"{a.key} is ALREADY set in the parent; use make_variant.py to "
            "substitute it instead of adding a second line")

    # the key must really be read with a get-default, or the add does nothing
    if f"os.environ.get('{a.key}'" not in old_src:
        die(f"{a.key} is never read via os.environ.get in cell {a.cell}; "
            "setting it would be a no-op")
    default = re.search(rf"os\.environ\.get\('{a.key}',\s*'([^']*)'\)", old_src)
    print(f"parent reads it as os.environ.get(...) default "
          f"{default.group(1) if default else '?'!r}")
    if default and default.group(1) == a.value:
        die(f"value {a.value!r} equals the default; this variant is a no-op")

    i = only(src_list, a.anchor, "anchor")
    new_list = src_list[:i + 1] + [new_line] + src_list[i + 1:]
    cell["source"] = new_list
    new_src = "".join(new_list)

    # ---- (a) line diff ---------------------------------------------------
    print("\n--- (a) line diff vs parent ---------------------------------------")
    diff = [l for l in difflib.unified_diff(src_list, new_list, lineterm="", n=0)
            if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]
    for l in diff:
        print("   ", l.rstrip()[:160])
    if diff != [f"+{new_line.rstrip()}"] and len(diff) != 1:
        die(f"expected exactly 1 added line, diff shows {len(diff)}")
    print(f"    inserted after line {i + 1}: {a.anchor}")

    # ---- (b) AST: exactly one added top-level statement -------------------
    print("\n--- (b) AST verification ------------------------------------------")
    old_top = [ast.dump(n) for n in ast.parse(old_src).body]
    new_tree = ast.parse(new_src)
    new_top = [ast.dump(n) for n in new_tree.body]
    if len(new_top) != len(old_top) + 1:
        die(f"top-level statement count moved by {len(new_top) - len(old_top)}")
    added = [k for k in range(len(new_top)) if new_top[k] not in old_top]
    if len(added) != 1:
        die(f"{len(added)} statements differ, expected 1: {added}")
    node = new_tree.body[added[0]]
    if ast.unparse(node) != new_line.strip():
        die(f"added statement is {ast.unparse(node)!r}, expected {new_line.strip()!r}")
    print(f"    {len(old_top)} -> {len(new_top)} top-level statements")
    print(f"    the one added statement is: {ast.unparse(node)}")
    # everything else byte-identical, in order
    rest = [s for k, s in enumerate(new_top) if k != added[0]]
    if rest != old_top:
        die("a statement other than the addition moved or changed")
    print("    every other top-level statement is unchanged and in order: PASS")

    # ---- (c) compile -----------------------------------------------------
    compile(new_src, "<cell2>", "exec")
    print("\n--- (c) compile ----------------------------------------------------")
    print("    compile(cell 2) OK")

    # ---- (d) drift guard -------------------------------------------------
    print("\n--- (d) configuration drift guard ---------------------------------")
    g = only(new_list, f"{GUARD_KEY} = ", "drift guard")
    guarded = ast.literal_eval(new_list[g].split("=", 1)[1].strip())
    if a.key in guarded:
        die(f"{a.key} is in {GUARD_KEY} (asserts {guarded[a.key]}); the guard "
            "must be updated in the SAME edit or the run aborts at cell 3")
    print(f"    {a.key} is not among the {len(guarded)} guarded keys: PASS")

    # ---- (e) whole-notebook ----------------------------------------------
    for c in nb["cells"]:
        if c.get("cell_type") == "code":
            c["outputs"] = []
            c["execution_count"] = None
    after = [l for c in nb["cells"] for l in c.get("source", [])]
    whole = [l for l in difflib.unified_diff(before, after, lineterm="", n=0)
             if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]
    if len(after) != len(before) + 1 or len(whole) != 1:
        die(f"whole-notebook diff is {len(whole)} lines / {len(after) - len(before)} "
            "added, expected 1/1")
    print("\n--- (e) whole notebook --------------------------------------------")
    print(f"    {len(before)} -> {len(after)} lines, exactly 1 added: PASS")
    print("    => ONE change relative to the parent")

    out = ROOT / (a.out or f"submissions/{a.slug}")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.slug}.ipynb").write_text(json.dumps(nb))
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": f"aryaarun07/{a.slug}", "title": a.title,
        "code_file": f"{a.slug}.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true",
        "enable_tpu": "false", "enable_internet": "false",
        "machine_shape": "NvidiaTeslaT4", "dataset_sources": DATASETS,
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "kernel_sources": [], "model_sources": []}, indent=2))
    print(f"\nwrote {out}/{a.slug}.ipynb")
    print(f"wrote {out}/kernel-metadata.json")
    print(f"slug == slugify(title) == {a.slug}: the push will land where you expect")


if __name__ == "__main__":
    main()
