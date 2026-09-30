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

Two modes, auto-detected from the parent:

  ADD         the key is not set in the parent (it sits at its os.environ.get
              default). One line is inserted. This is what s05, s06 and s09 did.
  SUBSTITUTE  the key IS already set. That one line is rewritten in place.

If the key is also in the notebook's _EXPECTED_NUMERIC drift guard, the guard
entry is rewritten in the SAME edit -- otherwise the kernel aborts at cell 3 on
a configuration-drift assertion, roughly 20 minutes in. That is a 2-line change,
and the script proves it is exactly those 2 lines. This is what unlocks the
UPSTREAM knobs (DET_THRESHOLD, the ILP weights, BIDIRECTIONAL_EDGE_WEIGHT,
SECONDARY_EDGE_FEATURE_TTA_WEIGHT), none of which the local .geff harness can
test, because it starts from the ILP output.

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

    set_lines = [k for k, l in enumerate(src_list)
                 if l.lstrip().startswith(f"os.environ['{a.key}']")]
    if len(set_lines) > 1:
        die(f"{a.key} is set on {len(set_lines)} lines; expected 0 or 1")
    mode = "SUBSTITUTE" if set_lines else "ADD"

    # the key must really be read back, or the change does nothing at all
    if f"os.environ.get('{a.key}'" not in old_src and f"os.environ['{a.key}']" not in old_src.replace(f"os.environ['{a.key}'] =", ""):
        die(f"{a.key} is never READ in cell {a.cell}; setting it would be a no-op")

    if mode == "ADD":
        default = re.search(rf"os\.environ\.get\('{a.key}',\s*'([^']*)'\)", old_src)
        print(f"mode   : ADD (parent leaves it at the os.environ.get default "
              f"{default.group(1) if default else '?'!r})")
        if default and default.group(1) == a.value:
            die(f"value {a.value!r} equals the default; this variant is a no-op")
        i = only(src_list, a.anchor, "anchor")
        new_list = src_list[:i + 1] + [new_line] + src_list[i + 1:]
    else:
        i = set_lines[0]
        cur = re.search(r"=\s*'([^']*)'", src_list[i])
        print(f"mode   : SUBSTITUTE (parent sets it to {cur.group(1)!r} on line {i + 1})")
        if cur and cur.group(1) == a.value:
            die(f"value {a.value!r} equals the parent's current value; no-op")
        new_list = list(src_list)
        new_list[i] = new_line

    # ---- drift guard: rewrite the entry in the SAME edit -----------------
    gi = only(new_list, f"{GUARD_KEY} = ", "drift guard")
    guarded = ast.literal_eval(new_list[gi].split("=", 1)[1].strip())
    guard_touched = a.key in guarded
    if guard_touched:
        want = float(a.value)
        print(f"guard  : {GUARD_KEY} asserts {a.key} == {guarded[a.key]}; "
              f"rewriting it to {want} in the same edit")
        old_entry = re.search(rf"('{re.escape(a.key)}':\s*)([0-9.eE+-]+)", new_list[gi])
        if not old_entry:
            die(f"could not locate the {a.key} entry inside {GUARD_KEY}")
        new_list[gi] = (new_list[gi][:old_entry.start(2)] + a.value
                        + new_list[gi][old_entry.end(2):])
        check = ast.literal_eval(new_list[gi].split("=", 1)[1].strip())
        if float(check[a.key]) != want:
            die(f"guard rewrite produced {check[a.key]}, expected {want}")
        if {k: v for k, v in check.items() if k != a.key} != \
           {k: v for k, v in guarded.items() if k != a.key}:
            die("the guard rewrite disturbed another key")

    cell["source"] = new_list
    new_src = "".join(new_list)
    n_expected = (1 if mode == "ADD" else 1) + (1 if guard_touched else 0)

    # ---- (a) line diff ---------------------------------------------------
    print("\n--- (a) line diff vs parent ---------------------------------------")
    diff = [l for l in difflib.unified_diff(src_list, new_list, lineterm="", n=0)
            if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]
    for l in diff:
        print("   ", l.rstrip()[:160])
    want_diff = n_expected if mode == "ADD" else 2 * n_expected
    if len(diff) != want_diff:
        die(f"expected {want_diff} diff lines for {mode}"
            f"{' + guard' if guard_touched else ''}, got {len(diff)}")
    print(f"    mode {mode}"
          + (f", inserted after line {i + 1}: {a.anchor}" if mode == "ADD"
             else f", rewrote line {i + 1}")
          + (f"; guard entry rewritten on line {gi + 1}" if guard_touched else ""))

    # ---- (b) AST: exactly one added top-level statement -------------------
    print("\n--- (b) AST verification ------------------------------------------")
    old_top = [ast.dump(n) for n in ast.parse(old_src).body]
    new_tree = ast.parse(new_src)
    new_top = [ast.dump(n) for n in new_tree.body]
    grew = 1 if mode == "ADD" else 0
    if len(new_top) != len(old_top) + grew:
        die(f"top-level statement count moved by {len(new_top) - len(old_top)}, "
            f"expected +{grew}")
    changed = [k for k in range(len(new_top)) if new_top[k] not in old_top]
    if len(changed) != n_expected:
        die(f"{len(changed)} statements differ, expected {n_expected}: {changed}")
    texts = [ast.unparse(new_tree.body[k]) for k in changed]
    if new_line.strip() not in texts:
        die(f"the env assignment is not among the changed statements: {texts}")
    if guard_touched and not any(t.startswith(GUARD_KEY) for t in texts):
        die("the guard line was expected to change and did not")
    if not guard_touched and any(t.startswith(GUARD_KEY) for t in texts):
        die("the guard changed but this key is not guarded")
    print(f"    {len(old_top)} -> {len(new_top)} top-level statements")
    for t in texts:
        print(f"    changed: {t[:120]}")
    rest = [x for k, x in enumerate(new_top) if k not in changed]
    base_rest = [x for x in old_top if x not in
                 [old_top[k] for k in range(len(old_top))
                  if mode == "SUBSTITUTE" and old_top[k] not in new_top]]
    if len(rest) != len(old_top) - (0 if mode == "ADD" else n_expected):
        die("unexpected number of untouched statements")
    print("    every other top-level statement is unchanged and in order: PASS")

    # ---- (c) compile -----------------------------------------------------
    compile(new_src, "<cell2>", "exec")
    print("\n--- (c) compile ----------------------------------------------------")
    print("    compile(cell 2) OK")

    # ---- (d) drift guard -------------------------------------------------
    print("\n--- (d) configuration drift guard ---------------------------------")
    final = ast.literal_eval(new_list[only(new_list, f"{GUARD_KEY} = ", "drift guard")]
                             .split("=", 1)[1].strip())
    if guard_touched:
        if float(final[a.key]) != float(a.value):
            die(f"guard says {final[a.key]}, notebook sets {a.value} -- cell 3 would abort")
        print(f"    {a.key} is guarded; guard now agrees with the set value "
              f"({final[a.key]}): PASS")
    else:
        print(f"    {a.key} is not among the {len(final)} guarded keys: PASS")

    # ---- (e) whole-notebook ----------------------------------------------
    for c in nb["cells"]:
        if c.get("cell_type") == "code":
            c["outputs"] = []
            c["execution_count"] = None
    after = [l for c in nb["cells"] for l in c.get("source", [])]
    whole = [l for l in difflib.unified_diff(before, after, lineterm="", n=0)
             if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]
    if len(after) != len(before) + grew or len(whole) != want_diff:
        die(f"whole-notebook diff is {len(whole)} lines / "
            f"{len(after) - len(before)} added, expected {want_diff}/{grew}")
    print("\n--- (e) whole notebook --------------------------------------------")
    print(f"    {len(before)} -> {len(after)} lines; {mode}"
          + (" + guard entry" if guard_touched else ""))
    print("    => ONE semantic change relative to the parent"
          + (" (the guard edit is bookkeeping forced by it, not a second change)"
             if guard_touched else ""))

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
