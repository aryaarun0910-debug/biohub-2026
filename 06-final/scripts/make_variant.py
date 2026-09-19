"""Build a one-change variant of the verified 0.947 notebook, and prove it.

Refuses to write unless the number of changed source lines equals the number of
substitutions requested. The drift guard (_EXPECTED_NUMERIC) is checked too: if
a key being changed also appears there, the guard must be updated in the same
edit or the notebook aborts at cell 3.
"""
import json, sys, difflib
from pathlib import Path

BASE = Path("Public Notebooks/biohub-0-947-lb-runnable-with-public-datasets.ipynb")
GUARDED = {"BIOHUB_DET_THRESHOLD", "BIOHUB_ILP_APPEARANCE_WEIGHT",
           "BIOHUB_ILP_DISAPPEARANCE_WEIGHT", "BIOHUB_GAP_CLOSE_UM",
           "BIOHUB_OUTPUT_MIN_TRACK_LEN", "BIOHUB_SAFE_DIV_MAX_UM",
           "BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD", "BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT",
           "BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT"}


def build(slug, title, subs, out_dir):
    nb = json.loads(BASE.read_text())
    before = [l for c in nb["cells"] for l in c.get("source", [])]
    n = 0
    for var, old, new in subs:
        if var in GUARDED:
            print(f"  NOTE: {var} is in the drift guard; the guard entry must change too")
        hit = 0
        for c in nb["cells"]:
            for i, line in enumerate(c.get("source", [])):
                if f"'{var}'" in line and old in line and "os.environ[" in line:
                    c["source"][i] = line.replace(old, new); hit += 1
        if hit != 1:
            sys.exit(f"  ABORT: {var} matched {hit} lines, expected 1")
        n += 1
    for c in nb["cells"]:
        if c.get("cell_type") == "code":
            c["outputs"] = []; c["execution_count"] = None
    after = [l for c in nb["cells"] for l in c.get("source", [])]
    d = [l for l in difflib.unified_diff(before, after, lineterm="", n=0)
         if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]
    changed = len(d) // 2
    if changed != n or len(before) != len(after):
        sys.exit(f"  ABORT: expected {n} changed lines, diff shows {changed}")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / f"{slug}.ipynb").write_text(json.dumps(nb))
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": f"aryaarun07/{slug}", "title": title, "code_file": f"{slug}.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": "true",
        "enable_gpu": "true", "enable_tpu": "false", "enable_internet": "false",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["pilkwang/biohub-tracking-support-pack-50ep-v1",
                            "pilkwang/biohub-temporal-unet3d-seed314159-v1",
                            "pilkwang/biohub-deepcenter-unet3d-center-prior-v1"],
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "kernel_sources": [], "model_sources": []}, indent=2))
    print(f"  OK {out}/{slug}.ipynb   {len(before)} lines, {changed} changed")
    for l in d:
        print("     ", l.strip())


if __name__ == "__main__":
    print("s02: symmetry tau 0.6 -> 1.35")
    build("biohub-s02-symmetry135", "Biohub S02 safe-div symmetry 1.35",
          [("BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU", "'0.6'", "'1.35'")],
          "submissions/s02_symmetry135")
    print("\ns03: diverge 0 AND symmetry 1.35 (the 3x case)")
    build("biohub-s03-div0-sym135", "Biohub S03 safe-div diverge 0 symmetry 1.35",
          [("BIOHUB_SAFE_DIV_DIVERGE_UM", "'2.25'", "'0'"),
           ("BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU", "'0.6'", "'1.35'")],
          "submissions/s03_div0_sym135")
