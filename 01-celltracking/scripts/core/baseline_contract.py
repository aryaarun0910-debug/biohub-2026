"""Extract the OPERATIONAL BASE's configuration from the built notebook, mechanically.

WHY THIS IS GENERATED AND NOT TYPED
-----------------------------------
Three committed places encode a safe-division geometry, and on 2026-09-01 all three were pinned
to `p3_harmonic`'s triple - a notebook CLAUDE.md explicitly says is NOT the champion:

    tests/test_lineage_degree_invariants.py:33-35
    scripts/win_bet/constant_audit.py:241-244
    scripts/win_bet/gt_division_gates.py:8-10

Nothing anywhere encoded the operational base's. Every one of those is a hand-copied constant, so
each was an independent chance to go stale, and all three took it. The repair is to stop copying:
this module reads the built notebook and emits one contract that the tests and instruments
consume.

THE PARSING RULE, AND WHY A NAIVE ONE IS WRONG
----------------------------------------------
`kaggle_factory` appends a spec's `env` overrides to the END of the matched cell, so an
overridden variable is assigned MORE THAN ONCE in a single cell - three times for the champion's
DeepCenter checkpoint. Reading the first `os.environ["X"]` hit therefore returns a value the
process never holds. LAST ASSIGNMENT WINS, in cell order then line order.

The extractor is self-checking: `verify()` asserts it reproduces the two post-processing flags
CLAUDE.md documents as differing between the champion and `p3_harmonic`. If that assertion fails
the extractor is wrong and the contract is refused rather than emitted, because a wrong contract
that looks right is worse than no contract.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It records CONFIGURATION, never a score. Scores live in `facts.yaml` and are cited by id. It also
makes no claim that any value is correct - only that it is what the built notebook sets.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(REPO / "scripts" / "core"))

HEARTBEAT_OK = "BASELINE_CONTRACT_OK"
HEARTBEAT_DRIFT = "BASELINE_CONTRACT_DRIFT"

ROLES_YAML = REPO / "research" / "00-system" / "registry" / "baseline_roles.yaml"
OUT = REPO / "research" / "00-system" / "registry" / "generated" / "operational_base_contract.json"

_ENV = re.compile(r"""os\.environ\[\s*["']([A-Za-z0-9_]+)["']\s*\]\s*=\s*(.+?)\s*$""")

#: Groups exist so a reader can find the knob without knowing its prefix. A variable that matches
#: no group still appears under `other` - a group list must never silently drop a setting.
GROUPS = {
    "detection": ("DET_THRESHOLD", "DETECTION", "SECONDARY_DETECTION", "NODE_BUDGET"),
    "safe_division": ("SAFE_DIV",),
    "deepcenter_veto": ("DEEPCENTER", "REQUIRE_DEEPCENTER", "USE_DEEPCENTER"),
    "ilp": ("ILP_",),
    "association": ("BIDIRECTIONAL", "EDGE_", "MOTION_RELINK", "ASSOC"),
    "gap_recovery_filter": ("GAP_", "OUTPUT_", "SHORT_TRACK", "ADAPTIVE_SHORT_TRACK",
                            "FILTER", "RESCUE"),
    "export": ("EXPORT", "COORD", "INT", "FLOAT"),
    "loeo": ("LOEO",),
    "diagnostics": ("DIAGNOSTIC", "RUN_OUTPUT_DIAGNOSTICS"),
}


class ContractRefusal(RuntimeError):
    """A wrong contract that looks right is worse than none, so extraction fails closed."""


def _literal(raw: str):
    """Best-effort literal, else the raw source text. Never guesses a type."""
    txt = raw.strip().rstrip(",").strip()
    if txt.endswith("#") or "#" in txt:
        # strip a trailing comment only when it cannot be inside a string
        head = txt.split("#")[0].strip()
        if head:
            txt = head
    try:
        return ast.literal_eval(txt)
    except (ValueError, SyntaxError):
        return {"__unparsed__": txt}


def _lines(cell: dict) -> list[str]:
    """A notebook cell's `source` is EITHER a list of lines OR one string. Both are legal nbformat.

    This mattered: the first version of this extractor iterated `cell["source"]` directly, so for
    every string-valued cell it iterated CHARACTERS and matched nothing. The champion notebook has
    9 string cells and 1 list cell, so the extractor read one cell, emitted 40 variables and
    reported success. A partial read that looks complete is the defect this whole contract exists
    to prevent, and it was caught only because the cross-notebook self-check found 2 shared
    variables where the notebook plainly has 39 assignments.
    """
    src = cell.get("source", [])
    if isinstance(src, str):
        return src.splitlines()
    return [str(s).rstrip("\n") for s in src]


def extract(notebook: Path) -> dict:
    """Every `os.environ[...] = ...` in the notebook, LAST ASSIGNMENT WINS."""
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    values: dict[str, object] = {}
    where: dict[str, str] = {}
    counts: dict[str, int] = {}
    for ci, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        for li, line in enumerate(_lines(cell)):
            m = _ENV.search(line.rstrip("\n"))
            if not m:
                continue
            name, raw = m.group(1), m.group(2)
            values[name] = _literal(raw)
            where[name] = f"cell {ci} line {li + 1}"
            counts[name] = counts.get(name, 0) + 1
    return {"values": values, "defined_at": where, "assignment_counts": counts}


def group(values: dict) -> dict:
    out: dict[str, dict] = {k: {} for k in GROUPS}
    out["other"] = {}
    for name, val in sorted(values.items()):
        for gname, tokens in GROUPS.items():
            if any(tok in name for tok in tokens):
                out[gname][name] = val
                break
        else:
            out["other"][name] = val
    return out


def verify(notebook: Path, comparison: Path) -> dict:
    """The extractor must reproduce a difference the project has documented independently.

    CLAUDE.md:55-56 records that the champion and `p3_harmonic` differ in GAP2_RECOVERY and
    ADAPTIVE_SHORT_TRACK_RESCUE. If this extractor cannot see that, it is not reading the
    notebooks correctly and nothing it emits can be trusted.
    """
    a = extract(notebook)["values"]
    b = extract(comparison)["values"]
    checks = {}
    for name in ("BIOHUB_OUTPUT_GAP2_RECOVERY", "BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE"):
        checks[name] = {"operational_base": a.get(name), "comparison": b.get(name),
                        "differs": a.get(name) != b.get(name)}
    shared = sorted(set(a) & set(b))
    differing = [k for k in shared if a[k] != b[k]]
    rep = {
        "self_check": checks,
        "all_documented_flags_differ": all(c["differs"] for c in checks.values()),
        "shared_variables": len(shared),
        "shared_variables_differing": len(differing),
        "differing_names": differing,
        "only_in_operational_base": sorted(set(a) - set(b)),
    }
    if not rep["all_documented_flags_differ"]:
        raise ContractRefusal(
            "the extractor does NOT reproduce the difference CLAUDE.md documents between the "
            f"operational base and its comparison: {checks}. Refusing to emit a contract from an "
            "extractor that cannot see a known difference."
        )
    return rep


def roles() -> dict:
    import yaml
    return yaml.safe_load(ROLES_YAML.read_text(encoding="utf-8"))


def build() -> dict:
    r = roles()
    base = REPO / r["operational_base"]["notebook"]
    if not base.is_file():
        raise ContractRefusal(f"operational base notebook absent: {base}")
    digest = hashlib.sha256(base.read_bytes()).hexdigest()
    declared = r["operational_base"]["notebook_sha256"]
    if digest != declared:
        raise ContractRefusal(
            f"the operational base on disk hashes to {digest} but baseline_roles.yaml declares "
            f"{declared}. Either the notebook changed or the line-ending convention differs from "
            f"the one the digest was recorded under - see the digest_convention_warning in "
            f"baseline_roles.yaml. Refusing to emit a contract for an artifact whose identity is "
            f"unresolved."
        )
    comparison = REPO / "notebooks" / "kaggle_p3_harmonic" / "biohub-p3-harmonic.ipynb"
    ex = extract(base)
    payload = {
        "generated_by": "scripts/core/baseline_contract.py",
        "do_not_edit": "Generated. Edit the notebook or the spec, then regenerate; --check locks "
                       "this file against hand edits.",
        "operational_base": {
            "notebook": r["operational_base"]["notebook"],
            "notebook_sha256": digest,
            "spec": r["operational_base"]["spec"],
        },
        "parsing_rule": "os.environ assignments, LAST ASSIGNMENT WINS in (cell, line) order - the "
                        "factory appends spec overrides to the end of the matched cell, so an "
                        "overridden knob is assigned more than once and the first hit is a value "
                        "the process never holds.",
        "n_variables": len(ex["values"]),
        "multiply_assigned": {k: v for k, v in sorted(ex["assignment_counts"].items()) if v > 1},
        "groups": group(ex["values"]),
        "defined_at": ex["defined_at"],
        "extractor_self_check": verify(base, comparison) if comparison.is_file() else
                                {"skipped": "comparison notebook absent"},
    }
    return payload


# ==============================================================================================
# THE READ SIDE. Instruments and tests consume THIS, never a copied constant.
# ==============================================================================================
def load(path: Path | None = None) -> dict:
    """Read the generated contract. Absence is a REFUSAL, never a default.

    A default here would silently reinstate exactly the defect being repaired: three committed
    places each held their own copy of the safe-division geometry and all three had gone stale
    against the operational base without anything noticing.
    """
    p = path or OUT
    if not p.is_file():
        raise ContractRefusal(
            f"{p} does not exist. Generate it with `python scripts/core/baseline_contract.py`. "
            f"Refusing to fall back to a hard-coded value - a stale copy that runs is how the "
            f"safe-division geometry drifted three ways at once."
        )
    return json.loads(p.read_text(encoding="utf-8"))


def env(name: str, path: Path | None = None):
    """One environment value as the OPERATIONAL BASE sets it. KeyError if the base does not."""
    for grp in load(path)["groups"].values():
        if name in grp:
            return grp[name]
    raise ContractRefusal(
        f"{name!r} is not set by the operational base. If an instrument needs it, the instrument "
        f"is describing a different notebook and must say which."
    )


def as_float(name: str, path: Path | None = None) -> float:
    v = env(name, path)
    if isinstance(v, dict):
        raise ContractRefusal(f"{name} did not parse to a literal: {v}")
    return float(v)


def safe_division(path: Path | None = None) -> dict:
    """The three live safe-division radii plus the two frac caps, from the built notebook."""
    return {k: as_float(k, path) for k in (
        "BIOHUB_SAFE_DIV_MAX_UM",
        "BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM",
        "BIOHUB_SAFE_DIV_SISTER_MAX_UM",
        "BIOHUB_SAFE_DIV_FRAME_FRAC_CAP",
        "BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP",
    )}


def _canonical(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="fail on drift instead of writing")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    try:
        payload = build()
    except ContractRefusal as exc:
        print(f"{HEARTBEAT_DRIFT} REFUSED: {exc}")
        return 2
    text = _canonical(payload)
    if args.check:
        if not args.out.is_file():
            print(f"{HEARTBEAT_DRIFT} {args.out} does not exist; run without --check")
            return 1
        current = args.out.read_text(encoding="utf-8")
        if current != text:
            print(f"{HEARTBEAT_DRIFT} {args.out} differs from the built notebook. Either the "
                  f"notebook changed and the contract was not regenerated, or the contract was "
                  f"edited by hand - which is the defect it exists to prevent.")
            return 1
        print(f"  {args.out.relative_to(REPO).as_posix()}  in sync  "
              f"({payload['n_variables']} variables)")
        print(HEARTBEAT_OK)
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"  wrote {args.out.relative_to(REPO).as_posix()}  "
          f"({payload['n_variables']} variables, "
          f"{len(payload['multiply_assigned'])} multiply assigned)")
    print(HEARTBEAT_OK)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
