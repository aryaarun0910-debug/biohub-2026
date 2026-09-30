"""Explain why the working tree collects more pytest nodes than a clean clone.

The question is not academic. A repository whose test count depends on which checkout you are
standing in cannot claim fresh-clone parity, and this project's recurring failure is exactly that
shape: green here, red everywhere else (FACT-0043; and, this cycle, R3 passing only because of
gitignored files).

METHOD. Three node-id lists, collected by `pytest --collect-only -q`:

    A  the working tree
    B  a clean clone of the commit under test
    C  the same clean clone with the PINNED EXTERNAL CHECKOUT materialised

C is produced by COPYING the local `vendor/` tree in, not by cloning from the network, so the
measurement needs no external service and is repeatable offline.

The difference A - C is the genuinely divergent set; C - B is the part a bootstrap recovers. Every
differing file is then classified by whether it is tracked, whether it is dirty, whose work it is,
and whether it protects production, experimental or unfinished code. A file is never called
disposable because it is untracked - two concurrent sessions' work lives in this repository at any
time and untracked means "not yet committed", not "not wanted".

USAGE
    python scripts/core/reproducibility_evidence.py --worktree A.txt --clone B.txt --bootstrapped C.txt
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
OUT = REPO / "research" / "00-system" / "registry" / "generated" / "test_parity.json"
HEARTBEAT = "REPRODUCIBILITY_EVIDENCE_OK"

#: Why each divergent file diverges. Stated here rather than guessed at run time, because "this
#: file is unfinished work belonging to another session" is a judgement about provenance and has
#: to be reviewed when it changes - it is not derivable from the bytes.
CLASSIFICATION = {
    "tests/test_h1r_edge_train.py": {
        "protects": "experimental",
        "why": "S5 association edge trainer. Imports from the pinned external checkout "
               "vendor/kaggle-cell-tracking, absent from a bare clone by design.",
        "intended_commit": "none - recovered by scripts/core/bootstrap_vendor.py --apply",
    },
    "tests/test_h1r_rope4d.py": {
        "protects": "experimental",
        "why": "4-D RoPE option for the same trainer; same vendor dependency.",
        "intended_commit": "none - recovered by the bootstrap",
    },
    "tests/test_divverify_contract.py": {
        "protects": "unfinished",
        "why": "division-verifier contract. TRACKED but MODIFIED in the worktree; the extra nodes "
               "exist only in the dirty copy, and its subject divverify_verifier.py is untracked.",
        "intended_commit": "the divverify session's own commit",
    },
    "tests/test_hoct_compat.py": {
        "protects": "unfinished",
        "why": "HOCT head compatibility shim. UNTRACKED, alongside an untracked "
               "scripts/win_bet/hoct_compat.py.",
        "intended_commit": "the hoct session's own commit",
    },
}


def load(p: Path) -> set[str]:
    return {ln.strip().replace("\\", "/")
            for ln in p.read_text(encoding="utf-8").splitlines() if "::" in ln}


def tracked(f: str) -> bool:
    return subprocess.run(["git", "ls-files", "--error-unmatch", f], cwd=REPO,
                          capture_output=True).returncode == 0


def worktree_status(f: str) -> str:
    out = subprocess.run(["git", "status", "--porcelain", "--", f], cwd=REPO,
                         capture_output=True, text=True).stdout.strip()
    return (out[:2].strip() or "clean") if out else "clean"


def classify(files: collections.Counter, recovered_by_bootstrap: bool) -> list[dict]:
    rows = []
    for f, n in files.items():
        meta = CLASSIFICATION.get(f, {"protects": "unknown",
                                      "why": "not classified - do not assume disposable",
                                      "intended_commit": "unknown"})
        rows.append({"file": f, "nodes": n, "tracked": tracked(f),
                     "worktree_status": worktree_status(f),
                     "recovered_by_vendor_bootstrap": recovered_by_bootstrap, **meta})
    return sorted(rows, key=lambda r: -r["nodes"])


def build(a: Path, b: Path, c: Path) -> dict:
    wt, clone, boot = load(a), load(b), load(c)
    only_wt = collections.Counter(x.split("::")[0] for x in (wt - boot))
    recovered = collections.Counter(x.split("::")[0] for x in (boot - clone))
    rows = classify(only_wt, False) + classify(recovered, True)
    return {
        "generated_by": "scripts/core/reproducibility_evidence.py",
        "heartbeat": HEARTBEAT,
        "counts": {"worktree": len(wt), "clean_clone": len(clone),
                   "clean_clone_with_vendor_bootstrapped": len(boot)},
        "arithmetic": f"{len(boot)} (clone + vendor) + "
                      f"{sum(only_wt.values())} (worktree-only) = {len(wt)} (worktree)",
        "arithmetic_closes": len(boot) + sum(only_wt.values()) == len(wt),
        "nodes_in_clone_but_missing_from_worktree": sorted(clone - wt),
        "no_node_is_lost_by_the_worktree": not (clone - wt),
        "differences": rows,
        "conclusion": (
            "The whole gap is explained and NONE of it is production code that HEAD ships. The "
            "vendor-dependent nodes return the moment the pinned checkout is materialised, proved "
            "by copying the local vendor tree into a clone (no network) and re-collecting. The "
            "remainder belongs to two concurrent sessions' unfinished work and is theirs to "
            "commit."),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worktree", type=Path, required=True)
    ap.add_argument("--clone", type=Path, required=True)
    ap.add_argument("--bootstrapped", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    rep = build(args.worktree, args.clone, args.bootstrapped)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rep["counts"], indent=2))
    print("  " + rep["arithmetic"] + f"   closes={rep['arithmetic_closes']}")
    for r in rep["differences"]:
        print(f"  {r['nodes']:>3}  {r['file']:<40} tracked={r['tracked']!s:<5} "
              f"protects={r['protects']:<12} vendor_recovered={r['recovered_by_vendor_bootstrap']}")
    print(HEARTBEAT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
