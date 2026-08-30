r"""CRYPTOGRAPHIC BINDING OF THE CONTROL ARM TO THE REGISTRY'S DEPLOYED FOLD CONTROL.

WHY THIS EXISTS - THE GUARD THAT DID NOT GUARD
----------------------------------------------
``FACT-0432`` defect 1, found by EXERCISING the code rather than reading it: ``assoc_report``
``verdict()`` was called on a report where every channel is favourable and the ONLY defect was
the control's identity, and it returned ``promotable: true`` with ``blockers: []``. ``build_report``
validated ``control`` in no way, and the control arrives at
``scripts/win_bet/assoc_train_harness.py:1628`` as::

    json.loads(Path(v["control"]).read_text(encoding="utf-8"))

- an arbitrary spec-supplied path. A file called "control" was therefore a control.

``FACT-0382`` is the governing rule and it is not a matter of taste: the honest full-chain
comparison is (widened surface + scorer + FINAL CONSUMER) against the **DEPLOYED** control, never
against a widened-only control. The widened arm already costs -0.00906 under the unchanged argmax
(``FACT-0376``), so substituting it as the base silently hands the candidate that entire deficit
as a gain. That substitution is invisible in every channel of the report - which is exactly why
the auditor drove an all-favourable report to ``promotable: true``.

WHAT BINDING MEANS HERE
-----------------------
Not "a field says it is the control". Five things, and the run is refused unless all five hold:

  IDENTITY     the arm names its experiment and its fold, explicitly. Missing identity FAILS
               CLOSED - it is never read as "unconstrained".
  RECOMPUTED   the artifact digest, the graph digest and the receipt digest are computed FROM THE
               BYTES on disk. A field that CLAIMS a digest is never read as the digest; if one is
               present and disagrees, that is itself a refusal, because a lying identity is worse
               than an absent one. ``FACT-0417`` is the precedent: the signed receipt's own
               ``expected_gpu_outputs`` was hardcoded, the post-run audit read that field, and
               nobody re-derived it - a correct run would have FAILED and a run that recorded
               nothing would have PASSED.
  MANIFEST     the recomputed artifact digest matches an entry in the registry-bound deployed
               control manifest, and that entry agrees about fold, experiment, graph and receipt.
               The match is on CONTENT, so a byte-identical copy at another path is the same
               control and binds; a path is not an identity.
  WIDENED      a widened-surface control is refused BY NAME, twice over: by its recorded digest in
               ``refused_controls``, and by any declared surface or candidate floor below the
               deployed floor. Refusal by name rather than by absence is deliberate - an unknown
               digest and a KNOWN-WRONG digest deserve different messages, and only the second
               tells the next agent what it did wrong.
  FOLD         the bound fold must equal the fold of the report reading it. An f0 control read
               against an f1 result is a different experiment wearing the right filename.

WHAT THIS MODULE DOES NOT DO
----------------------------
It changes what may be READ, never what counts as a win. No promotion threshold, falsifier or
gate condition lives here or is touched by it. A bound control is a PRECONDITION for reading a
report at all; it is not evidence for or against any lever.

    python scripts/win_bet/control_binding.py register --experiment EXP-#### --fold 0 \
        --control <rows.json> --graph <graph> --receipt <receipt.json> --provenance "..."
    python scripts/win_bet/control_binding.py verify --control <rows.json> \
        --experiment EXP-#### --fold 0 --graph <graph> --receipt <receipt.json>
    python scripts/win_bet/control_binding.py selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())

HEARTBEAT = "CONTROL_BOUND_TO_DEPLOYED_MANIFEST"
MANIFEST_PATH = REPO / "scripts" / "win_bet" / "deployed_controls.json"

# The deployed candidate rule: softmax over the source axis then threshold 0.5, which admits at
# most ONE parent per target by arithmetic (FACT-0369). FACT-0382 measured the surface below it:
# the widened P30/P34 acquisition floor 0.1 is a DIFFERENT surface, not a richer view of the same
# one. This constant names the deployed rule so a control acquired below it is refused by value.
DEPLOYED_CANDIDATE_FLOOR = 0.5

FOLD_EMBRYO = {"0": "44b6", "1": "6bba"}

# Identity fields a control arm must supply. Every one of them is a thing a wrong control would
# get wrong, and none of them can be inferred from the rows.
REQUIRED_IDENTITY = ("experiment", "fold", "graph", "receipt")

PLACEHOLDER = {"", "-", "n/a", "na", "none", "null", "unknown", "tbd", "todo", "?"}


class BindingRefusal(RuntimeError):
    """Raised by the CLI only. The library returns refusals so the receipt records them."""


# --------------------------------------------------------------------------------------------
# digests, recomputed from bytes
# --------------------------------------------------------------------------------------------
def sha256_path(path: Path) -> str:
    """Digest a file, or a directory tree, FROM ITS BYTES.

    A graph export may be a single file or a ``.geff``-style directory, so both must digest. The
    directory form folds in each entry's repo-relative POSIX path as well as its bytes, or two
    trees holding the same blobs under different names would collide.
    """
    p = Path(path)
    h = hashlib.sha256()
    if p.is_file():
        h.update(p.read_bytes())
        return h.hexdigest()
    if p.is_dir():
        files = sorted((q for q in p.rglob("*") if q.is_file()),
                       key=lambda q: q.relative_to(p).as_posix())
        if not files:
            raise FileNotFoundError(f"{p} is an EMPTY directory - there are no bytes to bind")
        for q in files:
            h.update(q.relative_to(p).as_posix().encode("utf-8"))
            h.update(b"\0")
            h.update(q.read_bytes())
        return h.hexdigest()
    raise FileNotFoundError(f"{p} is neither a file nor a directory")


def sha256_paths(paths) -> str:
    """Digest an ORDERED set of artifacts as one identity.

    A control is not always one file. ``finaledge_gates`` assembles its control rows from one
    payload per crop, and the deployed control's identity there is the whole ordered set - so the
    binding must be able to take the set rather than force a caller to concatenate it first and
    hash a file nobody keeps.
    """
    paths = list(paths)
    if not paths:
        raise FileNotFoundError("an EMPTY artifact list binds nothing")
    h = hashlib.sha256()
    for p in paths:
        h.update(sha256_path(Path(p)).encode("ascii"))
        h.update(b"\0")
    return h.hexdigest()


def _safe_digest(path, label: str, refusals: list[str]) -> str | None:
    if isinstance(path, (list, tuple)):
        if not path:
            refusals.append(f"{label}_missing: an empty artifact list. FAILS CLOSED")
            return None
        try:
            return sha256_paths(path)
        except (FileNotFoundError, OSError) as err:
            refusals.append(f"{label}_unreadable: {err}")
            return None
    if path is None or str(path).strip().lower() in PLACEHOLDER:
        refusals.append(f"{label}_missing: the control names no {label}, so its bytes cannot be "
                        f"hashed. Missing identity FAILS CLOSED (FACT-0432)")
        return None
    try:
        return sha256_path(Path(path))
    except (FileNotFoundError, OSError) as err:
        refusals.append(f"{label}_unreadable: {err}")
        return None


# --------------------------------------------------------------------------------------------
# the manifest
# --------------------------------------------------------------------------------------------
def empty_manifest() -> dict:
    return {"schema_version": 1, "kind": "deployed_control_manifest",
            "deployed_controls": [], "refused_controls": []}


def load_manifest(path: Path | None = None) -> dict:
    p = Path(path) if path else MANIFEST_PATH
    if not p.is_file():
        # Fail CLOSED: no manifest is not "no opinion". Nothing can bind against a manifest that
        # is not there, and an empty dict makes every lookup miss, which is the refusing answer.
        return {**empty_manifest(), "missing": str(p)}
    m = json.loads(p.read_text(encoding="utf-8"))
    if m.get("kind") != "deployed_control_manifest":
        raise BindingRefusal(f"{p} is not a deployed_control_manifest")
    m.setdefault("deployed_controls", [])
    m.setdefault("refused_controls", [])
    return m


# --------------------------------------------------------------------------------------------
# the binding
# --------------------------------------------------------------------------------------------
def bind_control(*, control_path, identity: dict | None,
                 manifest: dict | Path | None = None, n_rows: int | None = None) -> dict:
    """Bind one control arm to the registry's deployed fold control, or say why it does not.

    Returns a BLOCK, never raises on a refusal, because the refusal has to survive into the
    report: a run that was refused and a run that was never checked must not look the same in the
    JSON. ``assoc_report.verdict`` reads this block and blocks promotion unless it is bound.
    """
    refusals: list[str] = []
    man = manifest if isinstance(manifest, dict) else load_manifest(manifest)
    if man.get("missing"):
        refusals.append(f"manifest_missing: {man['missing']} does not exist, so no control can be "
                        "bound to the deployed fold control")

    ident = dict(identity or {})
    if not ident:
        refusals.append("identity_missing: the arm supplied no control identity at all. A path "
                        "named 'control' is not a control (FACT-0432)")
    for key in REQUIRED_IDENTITY:
        value = ident.get(key)
        if value is None or str(value).strip().lower() in PLACEHOLDER:
            refusals.append(f"identity_missing_{key}: fails CLOSED")

    fold = str(ident.get("fold", "")).strip()
    if fold and fold not in FOLD_EMBRYO:
        refusals.append(f"identity_fold_unknown: {fold!r} is not one of {sorted(FOLD_EMBRYO)}")

    # ---- digests, RECOMPUTED. A field that claims a digest is never read as the digest. -------
    artifact_sha = _safe_digest(control_path, "control_artifact", refusals)
    graph_sha = _safe_digest(ident.get("graph"), "graph", refusals)
    receipt_sha = _safe_digest(ident.get("receipt"), "receipt", refusals)

    for key, recomputed in (("artifact_sha256", artifact_sha), ("graph_sha256", graph_sha),
                            ("receipt_sha256", receipt_sha)):
        declared = ident.get(key)
        if declared and recomputed and str(declared) != recomputed:
            refusals.append(
                f"declared_{key}_is_false: the identity claims {str(declared)[:16]}... while the "
                f"bytes hash to {recomputed[:16]}.... The recomputed value is authoritative and a "
                "claim that contradicts it is a refusal, not a rounding error (FACT-0417)")

    # ---- the widened control, refused BY NAME rather than by absence --------------------------
    declared_surface = str(ident.get("surface", "")).strip().lower()
    if declared_surface and declared_surface != "deployed":
        refusals.append(
            f"widened_control_substitution: the arm declares surface {declared_surface!r}. "
            "FACT-0382 requires the comparison be made against the DEPLOYED control - a "
            "widened-only control hands the candidate the widening deficit as a gain")
    floor = ident.get("candidate_floor")
    if floor is not None and float(floor) < DEPLOYED_CANDIDATE_FLOOR:
        refusals.append(
            f"widened_control_substitution: candidate floor {float(floor)} is below the deployed "
            f"floor {DEPLOYED_CANDIDATE_FLOOR} (FACT-0369, FACT-0382)")

    refused = {r.get("artifact_sha256"): r for r in man.get("refused_controls", [])}
    refused_graphs = {r.get("graph_sha256"): r for r in man.get("refused_controls", [])
                      if r.get("graph_sha256")}
    hit = refused.get(artifact_sha) or refused_graphs.get(graph_sha)
    if hit:
        why = hit.get("why") or "refused by the manifest"
        refusals.append(
            f"control_is_a_registered_refused_control: this is a {hit.get('surface', 'refused')} "
            f"control - {why} (FACT-0382). It is refused by NAME, not merely unrecognised")

    # ---- the manifest match, on CONTENT --------------------------------------------------------
    entry = None
    if artifact_sha:
        matches = [e for e in man.get("deployed_controls", [])
                   if e.get("artifact_sha256") == artifact_sha]
        if not matches:
            refusals.append(
                f"control_not_registry_bound: no deployed-control entry has artifact sha256 "
                f"{artifact_sha[:16]}.... The comparison base must be the registry-bound DEPLOYED "
                "fold control (FACT-0382); an unregistered file is refused, whatever it is called")
        elif len(matches) > 1:
            refusals.append("manifest_ambiguous: two deployed-control entries share one digest")
        else:
            entry = matches[0]

    if entry is not None:
        if str(entry.get("fold")) != fold:
            refusals.append(f"fold_mismatch: the manifest binds this control to fold "
                            f"{entry.get('fold')!r}, the arm declares fold {fold!r}")
        if str(entry.get("experiment")) != str(ident.get("experiment")):
            refusals.append(
                f"experiment_mismatch: the manifest binds this control to "
                f"{entry.get('experiment')!r}, the arm declares {ident.get('experiment')!r}")
        if graph_sha and entry.get("graph_sha256") != graph_sha:
            refusals.append(
                "graph_mismatch: the rows are the registered control's, but they were scored "
                f"from a different graph ({graph_sha[:16]}... vs the bound "
                f"{str(entry.get('graph_sha256'))[:16]}...)")
        if receipt_sha and entry.get("receipt_sha256") != receipt_sha:
            refusals.append(
                f"receipt_mismatch: receipt {receipt_sha[:16]}... is not the receipt bound to "
                "this control")
        if str(entry.get("surface", "")).lower() != "deployed":
            refusals.append(
                f"registered_control_is_not_deployed: the manifest records surface "
                f"{entry.get('surface')!r} (FACT-0382)")
        ef = entry.get("candidate_floor")
        if ef is not None and float(ef) < DEPLOYED_CANDIDATE_FLOOR:
            refusals.append(
                f"registered_control_is_widened: registered candidate floor {float(ef)} is below "
                f"the deployed floor {DEPLOYED_CANDIDATE_FLOOR}")
        if n_rows is not None and entry.get("n_crops") is not None \
                and int(entry["n_crops"]) != int(n_rows):
            refusals.append(
                f"crop_count_mismatch: the manifest binds {entry['n_crops']} crops, the arm "
                f"supplied {n_rows}. The same digest cannot carry two crop counts, so one of the "
                "two numbers is describing a different object")

    bound = not refusals
    return {
        "heartbeat": HEARTBEAT if bound else "CONTROL_BINDING_REFUSED",
        "schema_version": 1,
        "bound": bound,
        "refusals": refusals,
        "fold": fold or None,
        "experiment": ident.get("experiment"),
        # RECOMPUTED, never echoed from the identity.
        "artifact_sha256": artifact_sha,
        "graph_sha256": graph_sha,
        "receipt_sha256": receipt_sha,
        "digests_recomputed_from_bytes": True,
        "control_path": ([str(p) for p in control_path]
                         if isinstance(control_path, (list, tuple)) else str(control_path)),
        "path_is_not_the_identity": "binding is on CONTENT: a byte-identical copy at another path "
                                    "binds, and the registered path is informational only",
        "manifest_entry": entry,
        "n_rows": n_rows,
    }


def unbound(reason: str) -> dict:
    """The FAIL-CLOSED block used when no binding was even attempted."""
    return {"heartbeat": "CONTROL_BINDING_REFUSED", "schema_version": 1, "bound": False,
            "refusals": [f"identity_missing: {reason}"], "fold": None, "experiment": None,
            "artifact_sha256": None, "graph_sha256": None, "receipt_sha256": None,
            "digests_recomputed_from_bytes": False, "control_path": None, "manifest_entry": None,
            "n_rows": None}


# --------------------------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------------------------
def register(*, manifest_path: Path, experiment: str, fold: str, control_path: Path,
             graph: Path, receipt: Path, provenance: str, surface: str = "deployed",
             candidate_floor: float = DEPLOYED_CANDIDATE_FLOOR, n_crops: int | None = None,
             date: str | None = None) -> dict:
    """Record a control in the manifest, with every digest recomputed here from the bytes."""
    if str(provenance or "").strip().lower() in PLACEHOLDER:
        raise BindingRefusal(
            "provenance is a placeholder. The manifest is the only place a control's identity is "
            "written down, so an unnamed provenance registers nothing")
    fold = str(fold).strip()
    if fold not in FOLD_EMBRYO:
        raise BindingRefusal(f"fold must be one of {sorted(FOLD_EMBRYO)}, got {fold!r}")
    surface = str(surface).strip().lower()
    if surface not in ("deployed", "widened"):
        raise BindingRefusal(f"surface must be 'deployed' or 'widened', got {surface!r}")

    man = load_manifest(manifest_path)
    man.pop("missing", None)
    entry = {
        "experiment": experiment,
        "fold": fold,
        "embryo_held_out": FOLD_EMBRYO[fold],
        "surface": surface,
        "candidate_floor": float(candidate_floor),
        "artifact_sha256": (sha256_paths(control_path)
                            if isinstance(control_path, (list, tuple))
                            else sha256_path(Path(control_path))),
        "graph_sha256": sha256_path(Path(graph)),
        "receipt_sha256": sha256_path(Path(receipt)),
        "n_crops": n_crops,
        "registered_path": ([str(p) for p in control_path]
                            if isinstance(control_path, (list, tuple)) else str(control_path)),
        "provenance": provenance,
        "registered": date or "",
    }
    if surface == "widened":
        entry["why"] = ("a widened-surface control. FACT-0382: the honest comparison is (widened "
                        "surface + scorer + final consumer) against the DEPLOYED control")
        man["refused_controls"] = [e for e in man["refused_controls"]
                                   if e.get("artifact_sha256") != entry["artifact_sha256"]]
        man["refused_controls"].append(entry)
    else:
        man["deployed_controls"] = [e for e in man["deployed_controls"]
                                    if e.get("artifact_sha256") != entry["artifact_sha256"]]
        man["deployed_controls"].append(entry)
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_path).write_text(json.dumps(man, indent=2), encoding="utf-8")
    return entry


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("register", help="record a control, digests recomputed from the bytes")
    r.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    r.add_argument("--experiment", required=True)
    r.add_argument("--fold", required=True, choices=sorted(FOLD_EMBRYO))
    r.add_argument("--control", type=Path, required=True)
    r.add_argument("--graph", type=Path, required=True)
    r.add_argument("--receipt", type=Path, required=True)
    r.add_argument("--provenance", required=True)
    r.add_argument("--surface", default="deployed", choices=["deployed", "widened"])
    r.add_argument("--candidate-floor", type=float, default=DEPLOYED_CANDIDATE_FLOOR)
    r.add_argument("--n-crops", type=int)
    r.add_argument("--date")

    v = sub.add_parser("verify", help="bind a control and print the refusals if it does not")
    v.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    v.add_argument("--control", type=Path, required=True)
    v.add_argument("--experiment", required=True)
    v.add_argument("--fold", required=True, choices=sorted(FOLD_EMBRYO))
    v.add_argument("--graph", type=Path, required=True)
    v.add_argument("--receipt", type=Path, required=True)
    v.add_argument("--out", type=Path)

    args = ap.parse_args(argv)

    if args.cmd == "register":
        try:
            entry = register(manifest_path=args.manifest, experiment=args.experiment,
                             fold=args.fold, control_path=args.control, graph=args.graph,
                             receipt=args.receipt, provenance=args.provenance,
                             surface=args.surface, candidate_floor=args.candidate_floor,
                             n_crops=args.n_crops, date=args.date)
        except (BindingRefusal, FileNotFoundError) as err:
            print(f"REGISTER REFUSED\n  {err}", file=sys.stderr)
            return 2
        print(f"REGISTERED {entry['surface']} control {entry['experiment']} fold "
              f"{entry['fold']}  artifact={entry['artifact_sha256'][:16]}... "
              f"graph={entry['graph_sha256'][:16]}...")
        return 0

    block = bind_control(control_path=args.control,
                         identity={"experiment": args.experiment, "fold": args.fold,
                                   "graph": str(args.graph), "receipt": str(args.receipt)},
                         manifest=args.manifest)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(block, indent=2), encoding="utf-8")
    if block["bound"]:
        print(f"{HEARTBEAT} experiment={block['experiment']} fold={block['fold']} "
              f"artifact={block['artifact_sha256'][:16]}...")
        return 0
    print("CONTROL BINDING REFUSED", file=sys.stderr)
    for why in block["refusals"]:
        print(f"  {why}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
