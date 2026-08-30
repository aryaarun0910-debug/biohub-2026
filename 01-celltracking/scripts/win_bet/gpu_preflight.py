r"""THE GPU PREFLIGHT. Runs on CPU, before any kernel is pushed, and answers on a laptop every
question that has so far been answered at GPU rates.

WHY THIS EXISTS
---------------
Two consecutive GPU sessions produced ZERO crops.
  * Attempt 1 (FACT-0387) died at a PROCESS BOUNDARY: `import predict_unet_transformer` from a
    process where that module's directory was not on `sys.path`.
  * Attempt 2 (FACT-0397) died on a DATASET MOUNT ROOT this repository had already hit, diagnosed
    and written up on 2026-08-01, in `scripts/kaggle_edits/loeo_retarget.py:86-101` - a file the
    failing notebook LOADS.
Neither was science. Both were environment discovery, and the standing GPU rule (head of
`levers.yaml`, host 2026-08-30) now forbids paying for it: GPU is never used to discover mounts,
imports, filenames, subprocess boundaries or empty inputs.

WHAT MAKES THIS DIFFERENT FROM A LINT
-------------------------------------
1. IT EXECUTES THE DEPLOYED TEXT. Resolvers are lifted out of the BUILT notebook, re-pointed at a
   simulated `/kaggle/input`, and CALLED - under both mount conventions. A regex that matches a
   resolver which still cannot find the file is the FACT-0397 mistake with extra steps.
2. IT CROSSES THE REAL SUBPROCESS BOUNDARY. The Gate-1 worker is extracted from the built
   notebook, written to a sandbox that mirrors `/kaggle/working`, and launched as a REAL
   subprocess with the notebook's own cwd and env - then run in preflight-only mode, where the
   argument parser and phase dispatch execute for real and only the GPU-bound phase body is
   stubbed. Both Gate-1 failures lived at boundaries a same-process check cannot see.
3. IT COUNTS EXACT POSITIVES. "Non-empty" is insufficient and FACT-0394 is why: a band that
   compared NOTHING passed every non-empty test it had. Every band reports an integer, and zero
   is a FAIL.
4. EVERY CHECK CARRIES THE MUTATION THAT PROVES IT FIRES. `--selftest` manufactures each defect
   and requires the matching check - not merely SOME check - to reject it. A checker that has not
   been shown to reject is not a checker.

FAIL CLOSED. An unreadable notebook, a missing dataset mirror, an unparseable spec or an
unexpected exception all produce a FAIL receipt with the reason recorded. The positive heartbeat
is the last line; its ABSENCE is the alarm.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import traceback
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_artifacts as KA          # noqa: E402
import kaggle_mounts as KM             # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HEARTBEAT = "GPU_PREFLIGHT_COMPLETE"
INSTRUMENTS = (
    "scripts/win_bet/gpu_preflight.py",
    "scripts/win_bet/kaggle_mounts.py",
    "scripts/win_bet/kaggle_artifacts.py",
    "scripts/kaggle_edits/kaggle_mount_ladder.py",
)

# Where the CPU mirror of each Kaggle dataset lives on this machine. A preflight that invents its
# inputs proves nothing about the run, so these are the SAME BYTES that were uploaded.
LOCAL_MIRRORS = {
    "aryaarun07/biohub-identity-replay-f0": Path("C:/temp/identity_ds2"),
}
LOCAL_PACK = Path("C:/temp/p7/tracking_repo")     # a materialised support pack from a real run
LOCAL_TRAIN = ROOT / "data" / "train"


# ---------------------------------------------------------------------------- result plumbing
@dataclass
class Check:
    id: str
    title: str
    passed: bool
    evidence: dict
    mutation: str          # the manufactured defect that this check is shown to reject

    def to_json(self) -> dict:
        return {"id": self.id, "title": self.title, "passed": bool(self.passed),
                "evidence": self.evidence, "proved_by_mutation": self.mutation}


@dataclass
class Ctx:
    spec_path: Path
    spec: dict
    notebook: Path
    nb_cells: list[str]
    nb_source: str
    sandbox: Path
    env: dict[str, str] = field(default_factory=dict)
    facts: dict = field(default_factory=dict)      # values later checks reuse


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True,
                           capture_output=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def _git_dirty(paths: list[str]) -> list[str]:
    """Uncommitted status for THESE paths only.

    `git status --porcelain --` with an EMPTY pathspec reports the whole worktree. That made PF01
    fail any spec with no `code_file` edits because some other agent had unrelated files open -
    the same false-negative class as PF09's doubled prefix, and just as certain to train a reader
    to ignore the check. An empty list of paths is an empty question, and its answer is empty.
    """
    if not paths:
        return []
    try:
        r = subprocess.run(["git", "status", "--porcelain", "--"] + paths, cwd=str(ROOT),
                           text=True, capture_output=True, timeout=30)
        return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    except Exception:
        return ["UNKNOWN"]


def parse_mount_path(p: str) -> tuple[str | None, str, str] | None:
    """Split a `/kaggle/input/...` literal into (owner, slug, relative-inside-the-dataset).

    WHY THIS IS A FUNCTION AND NOT THREE INLINE `split` CALLS. The first version of PF09 did
    `p.split("/", 3)[-1]` for the relative part, which LEAVES THE SLUG IN IT, and
    `p.split("/")[3]` for the slug, which returns the literal "datasets" for the nested
    convention. Simulating a mount from those two produced
    `datasets/pilkwang/datasets/pilkwang/<slug>/...` - a doubled prefix - so the check reported a
    failure caused by its own path construction rather than by the notebook under test. Agent 2
    hit it on a real spec. A checker that cries wolf is worse than no checker: it teaches people
    to ignore the one time it is right.

    Handles both observed conventions and the competition mount:
        /kaggle/input/<slug>/<rel>                        -> (None,  slug, rel)
        /kaggle/input/datasets/<owner>/<slug>/<rel>        -> (owner, slug, rel)
        /kaggle/input/competitions/<slug>/<rel>            -> (None,  slug, rel)
    `datasets/<slug>/<file>` (no owner) is disambiguated by segment count: an owner is taken only
    when at least three segments remain, which is true of every nested path observed in a fetched
    log and false of the two-segment form.
    """
    prefix = "/kaggle/input/"
    if not p.startswith(prefix):
        return None
    parts = [x for x in p[len(prefix):].split("/") if x]
    owner = None
    if parts and parts[0] in ("datasets", "competitions"):
        kind = parts.pop(0)
        if kind == "datasets" and len(parts) >= 3:
            owner = parts.pop(0)
    if not parts:
        return None
    slug = parts.pop(0)
    return owner, slug, "/".join(parts)


# The ARTIFACT CLASS a spec produces. A submission notebook has no Gate-1 report and never will,
# so failing it for lacking one is noise - and noise is how a real FAIL gets ignored. Checks
# declare the classes they apply to; anything else is SKIPPED and named in the receipt, never
# silently passed. Determining the class wrongly is the dangerous case, so an undetermined class
# runs everything.
CLASSES = ("submission", "gate_smoke", "feature_cache", "generic")


def classify(spec: dict, code_files: list[str]) -> tuple[str, str]:
    if spec.get("expects_submission"):
        return "submission", "spec.expects_submission is true"
    for cf in code_files:
        if cf.endswith("assoc_feature_cache.py"):
            return "feature_cache", f"spec injects {cf}"
        if cf.endswith("assoc_feature_parity.py"):
            return "gate_smoke", f"spec injects {cf}"
    return "generic", "no submission and no recognised gate patch"


def load_context(spec_path: Path, sandbox: Path) -> Ctx:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    nb = ROOT / spec["out_dir"] / spec["code_file"]
    cells = [""] * 0
    nbj = json.loads(nb.read_text(encoding="utf-8"))
    cells = ["".join(c.get("source", [])) for c in nbj["cells"]]
    env: dict[str, str] = {}
    for edit in spec.get("edits", []):
        if edit.get("kind") == "env":
            env.update({str(k): str(v) for k, v in (edit.get("vars") or {}).items()})
    ctx = Ctx(spec_path=Path(spec_path), spec=spec, notebook=nb, nb_cells=cells,
              nb_source="\n".join(cells), sandbox=Path(sandbox), env=env)
    code_files = [e["code_file"] for e in spec.get("edits", []) if e.get("code_file")]
    ctx.facts["code_files"] = code_files
    # The GATE PATCH is the experiment's own patch: the last code_file the factory injects.
    ctx.facts["gate_patch"] = code_files[-1] if code_files else None
    ctx.facts["profile"] = ("assoc_feature_parity"
                            if any(c.endswith("assoc_feature_parity.py") for c in code_files)
                            else "generic")
    ctx.facts["class"], ctx.facts["class_reason"] = classify(spec, code_files)
    return ctx


def _skip(cid: str, title: str, ctx: Ctx, reason: str) -> Check:
    """A check the artifact class cannot produce evidence for. Named, never counted as a PASS."""
    c = Check(cid, title, True,
              {"skipped": True, "artifact_class": ctx.facts.get("class"), "reason": reason,
               "note": "SKIPPED, not passed. It is excluded from the verdict and listed in the "
                       "receipt's `skipped` field."},
              "point the instrument at a spec of a class this check does apply to; it runs")
    c.evidence["_skipped"] = True
    return c


def _unsupported(cid: str, title: str, ctx: Ctx, needs: str) -> Check:
    """The class DOES need this check, but no profile implements it. FAIL CLOSED.

    Measured 2026-08-30: run against another agent's spec, PF05 and PF06 reported PASS while
    reading P33's dataset, because they had no notion of which spec they applied to. A preflight
    that reports PASS for a job it never examined is the defect, not the checker.
    """
    return Check(cid, title, False,
                 {"unsupported_profile": ctx.facts.get("profile"), "implemented_for": needs,
                  "artifact_class": ctx.facts.get("class"), "spec": ctx.spec.get("name"),
                  "fail_closed": "this class REQUIRES this check and no profile implements it - "
                                 "extend the profile or state the gap; do not launch"},
                 "point the instrument at an in-class spec outside the implemented profile; it "
                 "refuses instead of passing on another experiment's inputs")


def _gate_guard(cid: str, title: str, ctx: Ctx) -> Check | None:
    """Skip for classes that never carry a Gate-1 worker; fail closed for classes that should."""
    if ctx.facts.get("profile") == "assoc_feature_parity":
        return None
    if ctx.facts.get("class") in ("gate_smoke", "feature_cache"):
        return _unsupported(cid, title, ctx, "assoc_feature_parity")
    return _skip(cid, title, ctx,
                 f"artifact class {ctx.facts.get('class')!r} carries no Gate-1 feature-parity "
                 f"worker, so there is nothing here to examine")


# ------------------------------------------------------------------------------------ PF01
def check_notebook_binds_to_spec(ctx: Ctx) -> Check:
    """The thing preflighted must be the thing pushed. Otherwise the receipt certifies a ghost."""
    mdir = ctx.notebook.parent
    manifest = json.loads((mdir / "build_manifest.json").read_text(encoding="utf-8"))
    meta = json.loads((mdir / "kernel-metadata.json").read_text(encoding="utf-8"))
    nb_sha = _sha256_file(ctx.notebook)
    spec_ds = sorted(ctx.spec.get("datasets", []))
    meta_ds = sorted(meta.get("dataset_sources", []))
    ev = {
        "notebook": str(ctx.notebook), "notebook_sha256": nb_sha,
        "manifest_built_sha256": manifest.get("built_sha256"),
        "slug_metadata": meta.get("id", "").split("/")[-1], "slug_spec": ctx.spec.get("slug"),
        "datasets_spec": spec_ds, "datasets_metadata": meta_ds,
        "competition_sources": meta.get("competition_sources"),
        "enable_gpu": meta.get("enable_gpu"), "machine_shape": meta.get("machine_shape"),
        "uncommitted_patch_files": _git_dirty(
            [e["code_file"] for e in ctx.spec.get("edits", []) if e.get("code_file")]),
    }
    ok = (nb_sha == manifest.get("built_sha256")
          and ev["slug_metadata"] == ctx.spec.get("slug")
          and spec_ds == meta_ds
          and not ev["uncommitted_patch_files"])
    return Check("PF01", "built notebook is bound to the spec that will be pushed", ok, ev,
                 "rewrite one byte of the built notebook; sha256 stops matching build_manifest")


# ------------------------------------------------------------------------------------ PF02
def check_mount_resolvers(ctx: Ctx) -> Check:
    """EXECUTE the resolvers THIS SPEC injects against a simulated mount, both conventions.

    GATE-CRITICAL means "defined in a patch file this spec lists", not "matches /find/". The base
    notebook carries its own helpers (find_artifacts_root, find_offline_package_dirs, ...) whose
    signatures are not path-in/path-out and whose globals do not exist outside the notebook; a
    check that failed on those would be noise, and noise is how a real FAIL gets ignored. They are
    still scanned for boundedness, which is a static property that applies to every one of them.
    """
    slug = "biohub-identity-replay-f0"
    files = {"meta/preilp_split0.parquet": b"parquet", "ecb/44b6_0113de3b.npz": b"npz"}
    patch_sources = {}
    for edit in ctx.spec.get("edits", []):
        cf = edit.get("code_file")
        if cf and (ROOT / cf).is_file():
            patch_sources[cf] = (ROOT / cf).read_text(encoding="utf-8")
    critical = {n for cf, s in patch_sources.items() for n in KM.extract_resolvers(s)}

    resolvers = KM.extract_resolvers(ctx.nb_source)
    results: dict[str, dict] = {}
    # A spec that injects no resolver of its own (a public-base replica, say) cannot be failed for
    # not having one - that is the same false-gate the artifact CLASS split exists to remove. What
    # IS still answerable for every notebook is boundedness, so that half runs and the inherited
    # resolvers are named as un-exercised rather than quietly counted as fine.
    ok = True
    for name, src in resolvers.items():
        is_critical = name in critical
        entry = {"gate_critical": is_critical, "bounded_no_double_star": KM.is_bounded(src),
                 "resolved": {}}
        ok = ok and entry["bounded_no_double_star"]
        for conv in ("flat", "datasets"):
            if not is_critical:
                entry["resolved"][conv] = {"found": None,
                                           "hit": "NOT EXERCISED - base-notebook helper, needs "
                                                  "notebook globals; boundedness still enforced"}
                continue
            box = ctx.sandbox / "mounts" / f"{name}_{conv}"
            shutil.rmtree(box, ignore_errors=True)
            KM.simulate_mount_tree(box, slug, files, convention=conv)
            try:
                fn = KM.rebase_and_exec(src, name, box)
                hit = fn("meta/preilp_split0.parquet")
                hit = hit[0] if isinstance(hit, list) and hit else hit
                found = bool(hit) and Path(str(hit)).exists()
            except Exception as exc:                       # a resolver that raises is a miss
                found, hit = False, f"{type(exc).__name__}: {exc}"
            entry["resolved"][conv] = {"found": found, "hit": str(hit)}
            ok = ok and found
        results[name] = entry
    # Reported through parse_mount_path, not a positional split: a naive regex calls the literal
    # "datasets" a slug, which is the same mistake that produced PF09's doubled prefix.
    hardcoded = sorted({
        p[1] for p in (parse_mount_path(m) for m in
                       re.findall(r'["\'](/kaggle/input/[A-Za-z0-9_\-/.]+)["\']', ctx.nb_source))
        if p is not None})
    ev = {"gate_critical_resolvers": sorted(critical), "resolvers": results,
          "conventions_tested": ["flat", "datasets"],
          "central_ladder_sha256": KM.LADDER_SHA256,
          "notebook_uses_central_ladder": KM.ladder_matches_notebook(ctx.nb_source),
          "flat_mount_literals_still_in_notebook": hardcoded,
          "independent_ladders_in_one_notebook": len(resolvers),
          "inherited_resolvers_not_exercised": sorted(set(resolvers) - critical),
          "scope_note": ("this spec injects no mount resolver of its own, so only BOUNDEDNESS was "
                         "checked; the base notebook's own resolvers are named above and were not "
                         "executed") if not critical else
                        "injected resolvers were executed under both conventions"}
    return Check("PF02", "every mount resolver this spec injects resolves under BOTH observed "
                         "Kaggle conventions, and no resolver can `**`-walk /kaggle/input",
                 ok, ev,
                 "replace the ladder with the hardcoded /kaggle/input/<slug> form that killed "
                 "attempt 2; the `datasets` convention stops resolving (selftest M1)")


# ------------------------------------------------------------------------------------ PF03/04
def check_expected_crops(ctx: Ctx) -> Check:
    """The EXACT crop names, from the real parquet, under the notebook's own selection rule."""
    guard = _gate_guard("PF03", "expected crop COUNT and the exact crop NAMES are known before launch", ctx)
    if guard is not None:
        return guard
    import polars as pl
    mirror = LOCAL_MIRRORS["aryaarun07/biohub-identity-replay-f0"]
    parquet = mirror / "meta" / "preilp_split0.parquet"
    want = int(ctx.env.get("BIOHUB_AFP_CROPS", "2"))
    rule = "sorted(read_parquet(preilp)['dataset'].unique())[:BIOHUB_AFP_CROPS]"
    rule_in_notebook = "_afp_pl.read_parquet(_afp_preilp)[\"dataset\"].unique().to_list()" in ctx.nb_source
    all_crops = sorted(pl.read_parquet(parquet, columns=["dataset"])["dataset"].unique().to_list())
    crops = all_crops[:want]
    stems = json.loads(ctx.env.get("BIOHUB_LOEO_STEMS", "[]"))
    zarrs = {c: (LOCAL_TRAIN / f"{c}.zarr").exists() for c in crops}
    ev = {"expected_crop_count": want, "selected_crops": crops,
          "crops_available_in_parquet": len(all_crops), "selection_rule": rule,
          "selection_rule_present_in_notebook": rule_in_notebook,
          "crops_in_fold_stem_list": {c: (c in stems) for c in crops},
          "crops_have_train_zarr": {c: bool(v) for c, v in zarrs.items()},
          "fold": ctx.env.get("BIOHUB_LOEO_FOLD"), "arm": ctx.env.get("BIOHUB_LOEO_ARM")}
    ok = (len(crops) == want and rule_in_notebook
          and all(ev["crops_in_fold_stem_list"].values())
          and all(zarrs.values()))
    ctx.facts["crops"] = crops
    ctx.facts["parquet"] = parquet
    ctx.facts["ecb_dir"] = mirror / "ecb"
    return Check("PF03", "expected crop COUNT and the exact crop NAMES are known before launch",
                 ok, ev,
                 "set BIOHUB_AFP_CROPS to 3 while the fold stem list holds 2 of them; the count "
                 "and the membership check both reject")


# ------------------------------------------------------------------------------------ PF05
def check_schemas(ctx: Ctx) -> Check:
    """Every column and key the worker reads, by name and dtype, on the real uploaded bytes."""
    guard = _gate_guard("PF05", "input schemas match what the worker reads, column by column", ctx)
    if guard is not None:
        return guard
    import numpy as np
    import polars as pl
    parquet = ctx.facts["parquet"]
    schema = dict(pl.scan_parquet(parquet).collect_schema())
    need_pq = {"dataset": "String", "row_type": "String", "node_id": "Int64", "t": "Int64",
               "source_id": "Int64", "target_id": "Int64", "edge_prob": "Float64"}
    pq_ok = {k: (str(schema.get(k)) == v) for k, v in need_pq.items()}
    need_npz = ("source_id", "target_id", "edge_prob")
    npz_ok = {}
    for crop in ctx.facts["crops"]:
        f = ctx.facts["ecb_dir"] / f"{crop}.npz"
        if not f.is_file():
            npz_ok[crop] = {"present": False}
            continue
        with np.load(f, allow_pickle=False) as z:
            npz_ok[crop] = {"present": True, "keys": sorted(z.keys()),
                            "required_present": all(k in z for k in need_npz)}
    row_types = sorted(pl.read_parquet(parquet, columns=["row_type"])["row_type"].unique().to_list())
    ev = {"parquet": str(parquet), "parquet_schema": {k: str(v) for k, v in schema.items()},
          "parquet_required_columns_ok": pq_ok, "row_types": row_types,
          "ecb_required_keys": list(need_npz), "ecb": npz_ok}
    ok = (all(pq_ok.values()) and {"node", "edge"}.issubset(set(row_types))
          and all(v.get("required_present") for v in npz_ok.values()))
    return Check("PF05", "input schemas match what the worker reads, column by column", ok, ev,
                 "drop `edge_prob` from the parquet schema map; the column check rejects")


# ------------------------------------------------------------------------------------ PF06
def check_exact_positive_counts(ctx: Ctx) -> Check:
    """EXACT integers per band per crop. Zero is a FAIL - FACT-0394 is the whole reason.

    This is the check that predicts, on CPU, whether the in-kernel gate's `a_checked > 0` and
    `b_checked > 0` floors can be satisfied at the configured frame cap. If either band is empty
    in scope, the GPU session is already known to produce a refusal rather than a result.
    """
    guard = _gate_guard("PF06", "EXACT positive pair counts per band per crop, strictly greater than zero", ctx)
    if guard is not None:
        return guard
    import numpy as np
    import polars as pl
    max_frames = int(ctx.env.get("BIOHUB_AFP_MAX_FRAMES", "8"))
    df = pl.read_parquet(ctx.facts["parquet"])
    per_crop: dict[str, dict] = {}
    ok = True
    for crop in ctx.facts["crops"]:
        d = df.filter(pl.col("dataset") == crop)
        nodes = d.filter(pl.col("row_type") == "node")
        edges = d.filter(pl.col("row_type") == "edge")
        nid = nodes["node_id"].to_numpy()
        t = nodes["t"].to_numpy()
        contiguous = bool((nid == np.arange(len(nid))).all()) and bool((np.diff(t) >= 0).all())
        in_frame = set(nid[t < max_frames].tolist())
        node_t = {int(a): int(b) for a, b in zip(nid, t)}
        src = edges["source_id"].to_numpy()
        tgt = edges["target_id"].to_numpy()
        prob = edges["edge_prob"].to_numpy()
        m = np.fromiter(((int(a) in in_frame and int(b) in in_frame) for a, b in zip(src, tgt)),
                        dtype=bool, count=len(src))
        a_pairs = int(m.sum())
        a_dt = sorted({node_t[int(b)] - node_t[int(a)] for a, b in zip(src[m], tgt[m])})
        b_pairs, b_dt, b_in_scope_above = 0, [], 0
        f = ctx.facts["ecb_dir"] / f"{crop}.npz"
        if f.is_file():
            with np.load(f, allow_pickle=False) as z:
                sa = z["source_id"].astype(np.int64)
                sb = z["target_id"].astype(np.int64)
                sp = z["edge_prob"].astype(np.float64)
            ins = np.fromiter(((int(x) in in_frame and int(y) in in_frame) for x, y in zip(sa, sb)),
                              dtype=bool, count=len(sa))
            sub = ins & (sp <= 0.5)
            b_pairs = int(sub.sum())
            b_in_scope_above = int((ins & (sp > 0.5)).sum())
            b_dt = sorted({node_t[int(y)] - node_t[int(x)] for x, y in zip(sa[sub], sb[sub])})
        crop_ok = (a_pairs > 0 and b_pairs > 0 and contiguous
                   and a_dt == [1] and b_dt == [1]
                   and b_in_scope_above == a_pairs)
        ok = ok and crop_ok
        per_crop[crop] = {
            "frames_in_scope": max_frames,
            "nodes_in_scope": len(in_frame),
            "nodes_per_frame": [int((t == fr).sum()) for fr in range(max_frames)],
            "band_a_pairs_in_scope": a_pairs,
            "band_a_frame_deltas": a_dt,
            "band_a_prob_min": float(prob[m].min()) if a_pairs else None,
            "band_b_pairs_in_scope_at_or_below_0.5": b_pairs,
            "band_b_frame_deltas": b_dt,
            "sidecar_in_scope_above_0.5": b_in_scope_above,
            "sidecar_ties_to_preilp_band_a": b_in_scope_above == a_pairs,
            "node_ids_contiguous_and_time_ordered": contiguous,
            "passed": crop_ok,
        }
    ev = {"per_crop": per_crop,
          "why_zero_is_a_fail": "FACT-0394: a band that compared nothing passed every non-empty "
                                "test it had; the in-kernel gate now floors a_checked>0 and "
                                "b_checked>0, so an empty band is a refusal, not a result"}
    return Check("PF06", "EXACT positive pair counts per band per crop, strictly greater than zero",
                 ok, ev,
                 "cap the run at 1 frame so no consecutive pair is in scope; both band counts go "
                 "to 0 and the check rejects")


# ------------------------------------------------------------------------------------ PF07
_PROBE = r'''
"""Preflight-only probe. Runs in the SAME process shape as the deployed worker.

Everything up to and including argument parsing and phase dispatch executes for real; only the
GPU-bound phase bodies are stubbed. That is what makes this a preflight rather than a run, and it
is deliberately NOT a same-process check: both Gate-1 failures lived at this boundary.
"""
import importlib.util, json, os, sys, types

cfg = json.loads(sys.argv[1])
out = {"cwd": os.getcwd(), "sys_path0": sys.path[0],
       "pythonpath": os.environ.get("PYTHONPATH", "<unset>"), "steps": {}}


def step(name, fn):
    try:
        out["steps"][name] = {"ok": True, "detail": fn()}
    except BaseException as exc:
        out["steps"][name] = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
    return out["steps"][name]["ok"]


def _import_predictor():
    import predict_unet_transformer as AFP
    return AFP.__file__


imported = step("import_predict_unet_transformer", _import_predictor)

worker_mod = {}


def _import_worker():
    spec = importlib.util.spec_from_file_location("afp_worker_preflight", cfg["worker"])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # module-level imports run; main() is __main__-guarded
    worker_mod["m"] = mod
    return cfg["worker"]


step("import_worker_module", _import_worker)


def _attrs():
    import predict_unet_transformer as AFP
    missing = [a for a in cfg["afp_attrs"] if not hasattr(AFP, a)]
    if missing:
        raise AttributeError("predict_unet_transformer lacks %r" % missing)
    return sorted(cfg["afp_attrs"])


if imported:
    step("predictor_exposes_every_attribute_the_worker_uses", _attrs)


def _model_class():
    """Resolve the class load_model ACTUALLY builds, then check it.

    Do not guess the class name. The first version hardcoded TemporalUNet3D - the UNET, which is a
    CONSTRUCTOR ARGUMENT - and reported that the worker's four model calls were all missing, when
    load_model in fact returns UNetNodeTransformer, which has every one of them. Same false-negative
    class as PF09's doubled prefix, and it would have condemned a correct notebook.
    """
    import inspect, re as _re
    import predict_unet_transformer as AFP
    src = inspect.getsource(AFP.load_model)
    m = _re.search(r"^\s*model\s*=\s*([A-Za-z_]\w*)\s*\(", src, _re.M)
    if not m:
        raise RuntimeError("could not determine the class load_model builds")
    cls = getattr(AFP, m.group(1), None)
    if cls is None:
        raise AttributeError("load_model builds %r but it is not importable from the predictor"
                             % m.group(1))
    init_src = inspect.getsource(cls.__init__) if hasattr(cls, "__init__") else ""
    missing = [a for a in cfg["model_attrs"]
               if not hasattr(cls, a)
               and not _re.search(r"self\.%s\s*=" % _re.escape(a), init_src)]
    if missing:
        raise AttributeError("%s lacks %r" % (cls.__name__, missing))
    return {"class": cls.__name__, "checked": sorted(cfg["model_attrs"])}


step("model_class_exposes_every_method_the_worker_calls", _model_class)


def _argv():
    mod = worker_mod.get("m")
    if mod is None:
        raise RuntimeError("worker module never imported")
    seen = []
    mod.phase_cache = lambda a: seen.append(("cache", vars(a))) or 0
    mod.phase_verify = lambda a: seen.append(("verify", vars(a))) or 0
    for argv in cfg["argvs"]:
        sys.argv = ["afp_gate1.py"] + argv
        rc = mod.main()
        if rc != 0:
            raise RuntimeError("main() returned %r for %r" % (rc, argv[:4]))
    return [s[0] for s in seen]


step("worker_argv_parses_and_dispatches", _argv)

out["all_ok"] = all(s["ok"] for s in out["steps"].values())
sys.stdout.write("PREFLIGHT_PROBE_JSON " + json.dumps(out))
'''


def resolve_launch_pythonpath(nb_source: str, launch_txt: str) -> tuple[str | None, str]:
    """Work out what PYTHONPATH the notebook's launcher actually passes.

    PARSED AND EVALUATED, not pattern-matched. The first version searched the CALL text for a
    literal `env={**os.environ, "PYTHONPATH": "..."}`, so when the launcher was rewritten to build
    the env into a NAMED variable first - `_afp_env = {..., "PYTHONPATH": "scripts" + os.pathsep +
    "src"}` - the regex saw an `env=` it could not read and silently passed no PYTHONPATH at all.
    The probe then reproduced a failure the notebook no longer had. That is the fourth time in
    this instrument that reading code instead of resolving it produced a wrong answer, and it is
    the reason PF02, PF07 and PF09 all resolve rather than match.
    """
    m = re.search(r"env=([A-Za-z_]\w*)", launch_txt)
    if not m:
        inline = re.search(r'env=\{\*\*\w*\.?\w*environ,\s*"PYTHONPATH":\s*"([^"]+)"', launch_txt)
        return (inline.group(1) if inline else None,
                "inline literal" if inline else "no env= in the launch call")
    name = m.group(1)
    assign = re.search(rf"^\s*{re.escape(name)}\s*=\s*(\{{.*?\}})\s*$", nb_source,
                       re.S | re.M)
    if not assign:
        return None, f"env={name} but its assignment was not found"

    def _ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return _ev(node.left) + _ev(node.right)
        if isinstance(node, ast.Attribute) and node.attr == "pathsep":
            return os.pathsep
        raise ValueError(ast.dump(node)[:80])

    try:
        tree = ast.parse("_x = " + assign.group(1).strip())
        for k, v in zip(tree.body[0].value.keys, tree.body[0].value.values):
            if isinstance(k, ast.Constant) and k.value == "PYTHONPATH":
                return _ev(v), f"evaluated from {name}"
    except Exception as exc:
        return None, f"env={name} could not be evaluated: {type(exc).__name__}: {exc}"
    return None, f"env={name} sets no PYTHONPATH"


def check_worker_subprocess_boundary(ctx: Ctx) -> Check:
    """Launch the BUILT notebook's worker across the REAL subprocess boundary, preflight-only.

    Reproduces the notebook's own `subprocess.run(..., cwd=REPO_DIR)` shape exactly, including
    whether it passes `env=`. FACT-0387 was this boundary; the deployed prediction shards are
    launched with `env={**os.environ, "PYTHONPATH": "src"}` and cwd=REPO_DIR, so a launcher that
    omits `env=` inherits a PYTHONPATH that does not exist.
    """
    guard = _gate_guard("PF07", "the worker imports and dispatches ACROSS THE REAL SUBPROCESS BOUNDARY", ctx)
    if guard is not None:
        return guard
    src = ctx.nb_source
    m = re.search(r"_AFP_WORKER = r'''(.*?)'''", src, re.S)
    if not m:
        return Check("PF07", "worker crosses the real subprocess boundary", False,
                     {"error": "no _AFP_WORKER literal found in the built notebook"},
                     "delete the worker literal; extraction fails closed")
    worker_src = m.group(1)

    # How does the notebook actually launch it?
    launch = re.search(r"_afp_sub\.run\((.*?)\)\n", src, re.S)
    launch_txt = launch.group(1) if launch else ""
    passes_env = "env=" in launch_txt
    cwd_expr = (re.search(r"cwd=([^,\)]+)", launch_txt) or [None, "<none>"])[1]
    deployed_env = re.findall(r'env=\{\*\*os\.environ,\s*"PYTHONPATH":\s*"([^"]+)"', src)

    # sandbox mirrors /kaggle/working: repo dir beside the worker, exactly as on Kaggle
    box = ctx.sandbox / "subprocess"
    shutil.rmtree(box, ignore_errors=True)
    working = box / "working"
    repo = working / "tracking_repo"
    working.mkdir(parents=True)
    pack_ok = LOCAL_PACK.is_dir()
    if pack_ok:
        shutil.copytree(LOCAL_PACK / "scripts", repo / "scripts")
        shutil.copytree(LOCAL_PACK / "src", repo / "src")
    worker = working / "afp_gate1.py"
    worker.write_text(worker_src, encoding="utf-8")
    probe = working / "afp_gate1_preflight_probe.py"
    probe.write_text(_PROBE, encoding="utf-8")

    afp_attrs = sorted(set(re.findall(r"\bAFP\.(\w+)", worker_src)))
    model_attrs = sorted(set(re.findall(r"\bmodel\.(\w+)", worker_src))
                         - {"unet"}) + ["unet"]
    common = ["--weights", str(repo / "weights" / "edge_predictor_best.pth"),
              "--test-dir", str(box / "test"), "--cache-dir", str(box / "cache"),
              "--max-frames", ctx.env.get("BIOHUB_AFP_MAX_FRAMES", "8"),
              "--crops", *ctx.facts.get("crops", ["a", "b"])]
    argvs = [["--phase", "cache"] + common,
             ["--phase", "verify"] + common
             + ["--preilp", str(ctx.facts.get("parquet", "x.parquet")),
                "--ecb-dir", str(ctx.facts.get("ecb_dir", "ecb")),
                "--out", str(box / "out.json")]]
    cfg = {"worker": str(worker), "afp_attrs": afp_attrs,
           "model_attrs": sorted(set(model_attrs)), "argvs": argvs}

    # KAGGLE FIDELITY: the notebook process on Kaggle has no PYTHONPATH pointing into REPO_DIR -
    # REPO_DIR is created at runtime under /kaggle/working - so the inherited env is stripped of
    # any local PYTHONPATH before reproducing the notebook's launch.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    launch_pp, pp_how = resolve_launch_pythonpath(src, launch_txt)
    if launch_pp:
        env["PYTHONPATH"] = launch_pp
    res = subprocess.run([sys.executable, str(probe), _canonical(cfg)],
                         cwd=str(repo), text=True, capture_output=True, env=env, timeout=600)
    payload = {}
    tag = "PREFLIGHT_PROBE_JSON "
    if tag in (res.stdout or ""):
        payload = json.loads(res.stdout.split(tag, 1)[1])
    ev = {
        "notebook_launch_kwargs": " ".join(launch_txt.split())[:300],
        "notebook_passes_env_to_the_worker": passes_env,
        "pythonpath_the_launcher_passes": launch_pp,
        "how_that_was_determined": pp_how,
        "notebook_cwd_expression": cwd_expr.strip(),
        "deployed_shards_use_pythonpath": sorted(set(deployed_env)) or None,
        "sandbox_repo": str(repo), "worker": str(worker),
        "support_pack_mirror_present": pack_ok,
        "afp_attributes_required_by_worker": afp_attrs,
        "probe_returncode": res.returncode,
        "probe": payload or {"stderr_tail": (res.stderr or "")[-800:]},
    }
    ok = bool(payload.get("all_ok")) and res.returncode == 0 and pack_ok
    return Check("PF07", "the worker imports and dispatches ACROSS THE REAL SUBPROCESS BOUNDARY, "
                         "with the notebook's own cwd and env",
                 ok, ev,
                 "launch the probe with the notebook's env (no PYTHONPATH) instead of "
                 "scripts+src: `import predict_unet_transformer` raises ModuleNotFoundError, "
                 "which is FACT-0387 exactly")


# ------------------------------------------------------------------------------------ PF08
def check_output_retained(ctx: Ctx) -> Check:
    """A result the export sweep deletes is a session that produced no asset.

    CLASS-AWARE. The expected artifact is whatever this class produces: a submission notebook
    must retain submission.csv; a gate smoke must retain whatever its gate patch writes under
    /kaggle/working. The check is the same question either way - does the run keep the thing it
    was launched to make - and only the answer's name changes.
    """
    title = "the run's own output is written unconditionally AND survives the export sweep"
    keep = re.findall(r"_LOEO_KEEP\s*\|=\s*\{([^}]*)\}", ctx.nb_source)
    keep_txt = " ".join(keep)
    has_sweep = "_LOEO_KEEP" in ctx.nb_source
    gate_patch = ctx.facts.get("gate_patch")
    written: list[str] = []
    if gate_patch and (ROOT / gate_patch).is_file():
        written = sorted(set(re.findall(r'/kaggle/working/([A-Za-z0-9_\-]+\.(?:json|csv|npz))',
                                        (ROOT / gate_patch).read_text(encoding="utf-8"))))
    if ctx.facts.get("class") == "submission":
        expected = ["submission.csv"]
    elif written:
        expected = written
    else:
        return _skip("PF08", title, ctx,
                     "no submission expected and the spec's gate patch writes no named "
                     "/kaggle/working artifact, so there is no retention question to answer")
    writable = ctx.sandbox / "writable_probe"
    writable.mkdir(parents=True, exist_ok=True)
    try:
        (writable / "t.json").write_text("{}", encoding="utf-8")
        can_write = True
    except OSError:
        can_write = False
    retained = {name: (name in keep_txt or not has_sweep) for name in expected}
    unconditional = {
        name: bool(re.search(r'"/kaggle/working/%s"\)\.write_text' % re.escape(name),
                             ctx.nb_source))
        for name in expected}
    ev = {"artifact_class": ctx.facts.get("class"), "expected_outputs": expected,
          "notebook_has_export_sweep": has_sweep, "loeo_keep_unions": keep,
          "retained_by_the_sweep": retained,
          "written_outside_the_try_block": unconditional,
          "sandbox_output_writable": can_write}
    ok = can_write and all(retained.values())
    if ctx.facts.get("class") != "submission":
        ok = ok and any(unconditional.values())
    return Check("PF08", title, ok, ev,
                 "remove the gate report from _LOEO_KEEP; the retention check rejects and the "
                 "run would land with no retrievable output (selftest M12)")


# ------------------------------------------------------------------------------------ PF09
def probe_declared_mount_path(declared: str, sandbox: Path, label: str) -> dict:
    """Materialise the declared checkpoint under EACH convention and resolve it centrally.

    The path is parsed with `parse_mount_path`, so the simulated tree is built from the SLUG and
    the dataset-relative REMAINDER rather than from a naive positional split. The earlier version
    fed the whole `datasets/<owner>/<slug>/...` string in as the relative part and then mounted it
    under `<owner>/<slug>/` again, producing a doubled prefix and a FAIL that described nothing
    but its own arithmetic.
    """
    parsed = parse_mount_path(declared)
    if parsed is None:
        return {"declared": declared, "parsed": None,
                "note": "not a /kaggle/input path - nothing to resolve"}
    owner, slug, rel = parsed
    owners = tuple(o for o in {owner, "pilkwang", "aryaarun07"} if o)
    out: dict = {"declared": declared, "slug": slug, "owner": owner,
                 "dataset_relative": rel, "resolved": {}}
    for conv in ("flat", "datasets"):
        box = Path(sandbox) / "mounts" / f"{label}_{conv}"
        shutil.rmtree(box, ignore_errors=True)
        KM.simulate_mount_tree(box, slug, {rel: b"ckpt"}, convention=conv,
                               owner=owner or "pilkwang")
        try:
            hit = KM.resolve(rel, slugs=(slug,), owners=owners, input_root=str(box),
                             require=False, label=label)
        except Exception as exc:
            hit = f"{type(exc).__name__}: {exc}"
        hit_s = str(hit) if hit else ""
        out["resolved"][conv] = {
            "found": bool(hit) and Path(hit_s).exists(),
            "hit": hit_s,
            # the doubled-prefix regression, asserted rather than assumed
            "no_duplicated_slug_segment": hit_s.replace("\\", "/").count("/" + slug + "/") <= 1,
        }
    return out


def check_model_paths(ctx: Ctx) -> Check:
    """Weights must be repo-relative; every declared /kaggle/input model path must survive both
    mount conventions."""
    weights_rel = re.search(r'WEIGHTS_RELATIVE = f?"([^"]+)"', ctx.nb_source)
    wr = weights_rel.group(1) if weights_rel else None
    joined = ("_AfpPath(REPO_DIR) / WEIGHTS_RELATIVE" in ctx.nb_source
              or "REPO_DIR / WEIGHTS_RELATIVE" in ctx.nb_source)
    declared = {k: v for k, v in ctx.env.items()
                if isinstance(v, str) and v.startswith("/kaggle/input/")}
    probes = {k: probe_declared_mount_path(v, ctx.sandbox, k.lower())
              for k, v in sorted(declared.items())}
    # Does the notebook carry its OWN fallback for each declared slug, or does it rely on the
    # single hardcoded spelling that killed attempt 2?
    fallbacks = {}
    for k, pr in probes.items():
        slug = pr.get("slug")
        if not slug:
            continue
        flat = f"/kaggle/input/{slug}" in ctx.nb_source
        nested = f"/kaggle/input/datasets/" in ctx.nb_source and any(
            f"/kaggle/input/datasets/{o}/{slug}" in ctx.nb_source
            for o in ("pilkwang", "aryaarun07", "thtennant"))
        fallbacks[k] = {"slug": slug, "flat_spelling_present": flat,
                        "nested_spelling_present": nested,
                        "covers_both_conventions": flat and nested}
    ev = {"weights_relative": wr, "weights_is_repo_relative": bool(wr) and not wr.startswith("/"),
          "weights_joined_to_repo_dir": joined,
          "declared_mount_paths": probes,
          "notebook_has_its_own_datasets_fallback": fallbacks,
          "primary_weights_not_verifiable_offline":
              "the support-pack edge_predictor_best.pth is not mirrored locally; only the PATH "
              "SHAPE is checked here, and that limit is recorded rather than passed silently"}
    ok = bool(wr) and not wr.startswith("/") and joined
    for pr in probes.values():
        if pr.get("parsed", True) is None:
            continue
        for conv in pr.get("resolved", {}).values():
            ok = ok and conv["found"] and conv["no_duplicated_slug_segment"]
    return Check("PF09", "model paths are repo-relative, or resolve under BOTH mount conventions",
                 bool(ok), ev,
                 "M13 pins WEIGHTS_RELATIVE to an absolute mount path and it rejects; M14 declares "
                 "a checkpoint under a slug that is not mounted and it rejects; A3 is the ACCEPT "
                 "control - the correct nested datasets root must PASS, which is the direction "
                 "the doubled-prefix bug got wrong")


# ------------------------------------------------------------------------------------ PF10
def check_patch_heartbeats(ctx: Ctx) -> Check:
    """Every injected patch must be able to SAY it ran. Absence in the log is the alarm."""
    hb: dict[str, dict] = {}
    gate_patch = None
    for edit in ctx.spec.get("edits", []):
        cf = edit.get("code_file")
        if not cf:
            continue
        text = (ROOT / cf).read_text(encoding="utf-8")
        upper = sorted({p.strip() for p in
                        re.findall(r'print\(\s*f?["\']([A-Z][A-Z0-9_]{3,})', text)})
        any_print = len(re.findall(r"\bprint\s*\(", text))
        hb[cf] = {"uppercase_heartbeat_tokens": upper,
                  "present_in_built_notebook": [t for t in upper if t in ctx.nb_source],
                  "print_calls": any_print,
                  "can_report_itself": any_print > 0,
                  "patch_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
    # GENERIC, not AFP-specific: the gate patch is the experiment's own patch (the last code_file
    # the factory injects) and its terminal heartbeat is that patch's last uppercase print token.
    # A spec with no code_file edits has nothing to check, and says so.
    gate_patch = ctx.facts.get("gate_patch")
    if not gate_patch:
        return _skip("PF10", "the gate patch carries positive heartbeats", ctx,
                     "this spec injects no code_file patch, so there is no heartbeat to require")
    silent = sorted(cf for cf, v in hb.items() if not v["can_report_itself"])
    required = sorted(set(hb.get(gate_patch, {}).get("present_in_built_notebook", [])))
    gate_text = (ROOT / gate_patch).read_text(encoding="utf-8")
    ordered = [m for m in re.findall(r'print\(\s*f?["\']([A-Z][A-Z0-9_]{3,})', gate_text)]
    terminal = [ordered[-1]] if ordered else []
    ev = {"per_patch": hb,
          "gate_patch": gate_patch,
          "expected_log_tokens": required,
          "terminal_heartbeat": terminal,
          "terminal_present": bool(terminal) and all(t in ctx.nb_source for t in terminal),
          "patches_that_cannot_report_themselves": silent,
          "silent_patch_note":
              "reported, NOT blocking. A patch with no print cannot prove it fired, which is the "
              "silent-no-op failure AGENTS.md names - but it does not stop the session producing "
              "an asset, so it is a finding for the primary rather than a GPU gate.",
          "absence_is_the_alarm": True}
    ok = bool(gate_patch) and bool(required) and ev["terminal_present"]
    return Check("PF10", "the gate patch carries positive heartbeats, and the receipt names the "
                         "tokens whose ABSENCE from the fetched log is the alarm",
                 ok, ev,
                 "strip the prints from the gate patch; expected_log_tokens empties and the "
                 "check rejects (selftest M9)")


# ------------------------------------------------------------------------------------ PF11
def check_artifact_discovery(ctx: Ctx) -> Check:
    """Both fetch conventions, proven on the two artifacts that produced the FACT-0396 false FAIL."""
    fixtures = {}
    for name, expect in (("p24_deepcenter_best_veto", "queue"),
                         ("p32_public931_exact", "factory")):
        p = ROOT / "scripts" / "kaggle_specs" / f"{name}.json"
        d = KA.discover(KA.load_spec(p), roles=("receipt", "submission", "structural_audit"),
                        hash_files=False)
        fixtures[name] = {"expected_convention": expect, "convention": d.convention,
                          "root": str(d.root) if d.root else None, "found": d.found,
                          "roles": sorted(d.files)}
    mine = KA.discover(ctx.spec, roles=("gate_report", "log"), hash_files=False)
    ev = {"regression_fixtures": fixtures, "this_spec": mine.to_json(),
          "role_aliases": {k: list(v) for k, v in KA.ROLE_ALIASES.items()}}
    ok = all(f["found"] and f["convention"] == f["expected_convention"] for f in fixtures.values())
    return Check("PF11", "artifact discovery is unified: P24 (queue-fetch) and P32 "
                         "(factory-output) both resolve, and the convention is RECORDED",
                 ok, ev,
                 "remove the queue root from candidate_roots; P24 stops resolving, which is the "
                 "false FAIL corrected in FACT-0396")


CHECKS = (check_notebook_binds_to_spec, check_mount_resolvers, check_expected_crops,
          check_schemas, check_exact_positive_counts, check_worker_subprocess_boundary,
          check_output_retained, check_model_paths, check_patch_heartbeats,
          check_artifact_discovery)


# ------------------------------------------------------------------------------------ receipt
def sign(body: dict) -> dict:
    """Tamper-evident content signature.

    Not authentication - this repository holds no secret, and pretending otherwise would be the
    kind of decorative check this packet exists to eliminate. It binds the receipt to its own
    content AND to the exact instrument bytes that produced it, so a receipt edited after the
    fact, or produced by a modified checker, is distinguishable from one that was earned.
    """
    return {
        "algorithm": "sha256(canonical-json(body))",
        "body_sha256": hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest(),
        "instruments": {p: _sha256_file(ROOT / p) for p in INSTRUMENTS if (ROOT / p).is_file()},
        "guarantee": "tamper-evident, NOT authenticated; no secret exists in this repository",
    }


def verify(receipt: dict) -> tuple[bool, list[str]]:
    problems = []
    sig = receipt.get("signature") or {}
    recomputed = hashlib.sha256(_canonical(receipt.get("body")).encode("utf-8")).hexdigest()
    if recomputed != sig.get("body_sha256"):
        problems.append(f"body_sha256 mismatch: recomputed {recomputed[:16]} "
                        f"vs receipt {str(sig.get('body_sha256'))[:16]}")
    for p, want in (sig.get("instruments") or {}).items():
        have = _sha256_file(ROOT / p) if (ROOT / p).is_file() else None
        if have != want:
            problems.append(f"instrument {p} changed since the receipt was signed")
    return (not problems), problems


def run_preflight(spec_path: Path, sandbox: Path) -> dict:
    sandbox.mkdir(parents=True, exist_ok=True)
    checks: list[Check] = []
    try:
        ctx = load_context(spec_path, sandbox)
    except Exception as exc:
        body = {"verdict": "FAIL", "fatal": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-1500:], "checks": []}
        return {"body": body, "signature": sign(body)}
    for fn in CHECKS:
        try:
            checks.append(fn(ctx))
        except Exception as exc:                       # FAIL CLOSED, never "nothing to check"
            checks.append(Check(fn.__name__, fn.__doc__ or fn.__name__, False,
                                {"error": f"{type(exc).__name__}: {exc}",
                                 "traceback": traceback.format_exc()[-1200:]},
                                "the check itself raising is a FAIL, not a skip"))
    applicable = [c for c in checks if not c.evidence.get("_skipped")]
    skipped = [c.id for c in checks if c.evidence.get("_skipped")]
    passed = bool(applicable) and all(c.passed for c in applicable)
    body = {
        "instrument": "scripts/win_bet/gpu_preflight.py",
        "packet": "PKT-0037",
        "spec": str(spec_path.relative_to(ROOT)) if str(spec_path).startswith(str(ROOT))
                else str(spec_path),
        "spec_sha256": _sha256_file(spec_path),
        "notebook": str(ctx.notebook.relative_to(ROOT)),
        "notebook_sha256": _sha256_file(ctx.notebook),
        "code_commit": _git_commit(),
        "resolved_paths": {
            "local_mirrors": {k: str(v) for k, v in LOCAL_MIRRORS.items()},
            "support_pack_mirror": str(LOCAL_PACK),
            "train_root": str(LOCAL_TRAIN),
            "sandbox": str(sandbox),
        },
        "input_hashes": {
            str(ctx.facts.get("parquet")): _sha256_file(ctx.facts["parquet"])
            if ctx.facts.get("parquet") else None,
        },
        "crop_names": ctx.facts.get("crops", []),
        "expected_gpu_outputs": {
            "kaggle_working": ["assoc_feature_parity.json", "loeo_manifest.json", "run_stats.csv"],
            "log_tokens_required": ["AFP_PHASE cache", "AFP_PHASE verify", "AFP_CACHE crop=",
                                    "AFP crop=", "ASSOC_FEATURE_PARITY_COMPLETE"],
            "gate_report_must_contain": {"crops": len(ctx.facts.get("crops", [])),
                                         "all_passed": True},
            "per_crop_minimum_checked_pairs": {
                c: {"band_a": v["band_a_pairs_in_scope"],
                    "band_b": v["band_b_pairs_in_scope_at_or_below_0.5"]}
                for c, v in (next((ch.evidence.get("per_crop", {}) for ch in checks
                                   if ch.id == "PF06"), {})).items()},
        },
        "artifact_class": ctx.facts.get("class"),
        "artifact_class_reason": ctx.facts.get("class_reason"),
        "profile": ctx.facts.get("profile"),
        "verdict": "PASS" if passed else "FAIL",
        "checks": [c.to_json() for c in checks],
        "applicable": [c.id for c in applicable],
        "skipped": skipped,
        "failed": [c.id for c in applicable if not c.passed],
    }
    return {"body": body, "signature": sign(body)}


# ----------------------------------------------------------------------------------- selftest
def _selftest(sandbox: Path) -> int:
    """Manufacture each defect and require ITS OWN check to reject it.

    An auditor that rejects everything checks nothing, so two ACCEPT controls are included.
    """
    import numpy as np
    import polars as pl
    results: list[tuple[str, bool, str]] = []
    box = sandbox / "selftest"
    shutil.rmtree(box, ignore_errors=True)
    box.mkdir(parents=True)
    spec_path = ROOT / "scripts" / "kaggle_specs" / "p33_assoc_feature_parity_smoke.json"

    def rec(name, ok, detail=""):
        results.append((name, ok, detail))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)

    # --- M1 PF02: the hardcoded flat-mount resolver that killed attempt 2 -------------------
    hardcoded = textwrap.dedent('''
        def _afp_find(relative):
            _hit = _AfpPath("/kaggle/input/biohub-identity-replay-f0") / relative
            return _hit if _hit.exists() else None
    ''')
    hits = {}
    for conv in ("flat", "datasets"):
        b = box / f"m1_{conv}"
        KM.simulate_mount_tree(b, "biohub-identity-replay-f0", {"meta/p.parquet": b"x"}, conv)
        fn = KM.rebase_and_exec(hardcoded, "_afp_find", b)
        hits[conv] = bool(fn("meta/p.parquet"))
    rec("M1 PF02 rejects the hardcoded flat mount root (FACT-0397)",
        hits["flat"] and not hits["datasets"], f"flat={hits['flat']} datasets={hits['datasets']}")

    # --- A1 ACCEPT control: the central ladder finds it under every convention --------------
    acc = {}
    for conv in KM.CONVENTIONS:
        b = box / f"a1_{conv}"
        KM.simulate_mount_tree(b, "slug-x", {"meta/p.parquet": b"x"}, conv, owner="o")
        acc[conv] = bool(KM.resolve("meta/p.parquet", slugs=("slug-x",), owners=("o",),
                                    input_root=str(b), require=False))
    rec("A1 ACCEPT central ladder resolves under flat/datasets/competitions", all(acc.values()),
        str(acc))

    # --- M2 PF02: an unbounded `**` resolver is rejected as unbounded -----------------------
    rec("M2 PF02 rejects a `**` walk as unbounded",
        not KM.is_bounded('glob("/kaggle/input/**/x.parquet")'), "")

    # --- M3 PF06: an empty band is a FAIL, not a pass ---------------------------------------
    mirror = LOCAL_MIRRORS["aryaarun07/biohub-identity-replay-f0"]
    ctx = load_context(spec_path, box)
    ctx.facts["parquet"] = mirror / "meta" / "preilp_split0.parquet"
    ctx.facts["ecb_dir"] = mirror / "ecb"
    ctx.facts["crops"] = sorted(pl.read_parquet(ctx.facts["parquet"], columns=["dataset"])
                                ["dataset"].unique().to_list())[:2]
    good = check_exact_positive_counts(ctx)
    ctx.env["BIOHUB_AFP_MAX_FRAMES"] = "1"          # no consecutive pair can be in scope
    starved = check_exact_positive_counts(ctx)
    ctx.env["BIOHUB_AFP_MAX_FRAMES"] = "8"
    rec("M3 PF06 rejects a frame cap under which both bands compare nothing (FACT-0394)",
        good.passed and not starved.passed,
        f"8 frames passed={good.passed} / 1 frame passed={starved.passed}")

    # --- M4 PF06: a torn sidecar/pre-ILP tie is rejected -------------------------------------
    crop = ctx.facts["crops"][0]
    fake = box / "ecb_torn"
    fake.mkdir(parents=True, exist_ok=True)
    with np.load(mirror / "ecb" / f"{crop}.npz", allow_pickle=False) as z:
        np.savez_compressed(fake / f"{crop}.npz", source_id=z["source_id"][:5],
                            target_id=z["target_id"][:5], edge_prob=z["edge_prob"][:5])
    ctx2 = load_context(spec_path, box)
    ctx2.facts.update({"parquet": ctx.facts["parquet"], "ecb_dir": fake, "crops": [crop]})
    torn = check_exact_positive_counts(ctx2)
    rec("M4 PF06 rejects a sidecar that no longer ties to the pre-ILP band-A count",
        not torn.passed, "")

    # --- M5 PF11: dropping the queue root reproduces the FACT-0396 false FAIL ----------------
    p24 = KA.load_spec(ROOT / "scripts" / "kaggle_specs" / "p24_deepcenter_best_veto.json")
    real = KA.discover(p24, roles=("receipt",), hash_files=False)
    orig = KA.candidate_roots
    try:
        KA.candidate_roots = lambda spec, override=None: [
            ("factory", ROOT / spec["out_dir"] / "_out")]
        crippled = KA.discover(p24, roles=("receipt",), hash_files=False)
    finally:
        KA.candidate_roots = orig
    rec("M5 PF11 rejects a factory-only search that misses the queue-fetch champion (FACT-0396)",
        real.found and real.convention == "queue" and not crippled.found,
        f"real={real.convention} crippled_found={crippled.found}")

    # --- A2 ACCEPT control: P32 still resolves by the factory convention ---------------------
    p32 = KA.discover(KA.load_spec(ROOT / "scripts" / "kaggle_specs" / "p32_public931_exact.json"),
                      roles=("receipt",), hash_files=False)
    rec("A2 ACCEPT P32 resolves by the factory convention", p32.found and p32.convention == "factory",
        str(p32.convention))

    # --- M6 PF07: the notebook's own env (no PYTHONPATH) vs scripts+src ----------------------
    ctx3 = load_context(spec_path, box)
    ctx3.facts["crops"] = ctx.facts["crops"]
    ctx3.facts["parquet"] = ctx.facts["parquet"]
    ctx3.facts["ecb_dir"] = ctx.facts["ecb_dir"]
    as_built = check_worker_subprocess_boundary(ctx3)
    probe_dir = box / "subprocess" / "working"
    repo = probe_dir / "tracking_repo"
    fixed = subprocess.run(
        [sys.executable, str(probe_dir / "afp_gate1_preflight_probe.py"),
         _canonical({"worker": str(probe_dir / "afp_gate1.py"),
                     "afp_attrs": as_built.evidence.get("afp_attributes_required_by_worker", []),
                     "model_attrs": ["predict_edges", "_index_features", "unet",
                                     "detection_head"],
                     "argvs": []})],
        cwd=str(repo), text=True, capture_output=True, timeout=600,
        env={**{k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
             "PYTHONPATH": "scripts" + os.pathsep + "src"})
    fixed_ok = "PREFLIGHT_PROBE_JSON" in fixed.stdout and json.loads(
        fixed.stdout.split("PREFLIGHT_PROBE_JSON ", 1)[1])["steps"][
        "import_predict_unet_transformer"]["ok"]
    rec("M6 PF07 rejects the as-built launcher and accepts the same worker with PYTHONPATH "
        "=scripts+src (FACT-0387 boundary)",
        (not as_built.passed) and fixed_ok,
        f"as_built_passed={as_built.passed} with_pythonpath_import_ok={fixed_ok}")

    # --- M7 PF01: one edited byte in the built notebook breaks the binding -------------------
    ctx4 = load_context(spec_path, box)
    clean = check_notebook_binds_to_spec(ctx4)
    tamper_dir = box / "tampered"
    tamper_dir.mkdir(parents=True, exist_ok=True)
    for f in ("build_manifest.json", "kernel-metadata.json"):
        shutil.copy(ctx4.notebook.parent / f, tamper_dir / f)
    tampered_nb = tamper_dir / ctx4.notebook.name
    tampered_nb.write_text(ctx4.notebook.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    ctx4.notebook = tampered_nb
    tampered = check_notebook_binds_to_spec(ctx4)
    # Isolated to the NOTEBOOK-IDENTITY sub-fact on purpose: PF01 also rejects an uncommitted
    # patch file, and while another agent is mid-edit that would make this mutation's control
    # arm fail for a reason that has nothing to do with the mutation.
    clean_id = clean.evidence["notebook_sha256"] == clean.evidence["manifest_built_sha256"]
    tampered_id = tampered.evidence["notebook_sha256"] == tampered.evidence[
        "manifest_built_sha256"]
    rec("M7 PF01 rejects a built notebook edited after the manifest was written",
        clean_id and not tampered_id and not tampered.passed,
        f"clean_identity={clean_id} tampered_identity={tampered_id}")

    # --- M8 PF12: an edited receipt no longer verifies ---------------------------------------
    body = {"verdict": "PASS", "checks": []}
    receipt = {"body": body, "signature": sign(body)}
    ok_before, _ = verify(receipt)
    receipt["body"]["verdict"] = "PASS-but-actually-edited"
    ok_after, why = verify(receipt)
    rec("M8 receipt signature rejects post-hoc editing", ok_before and not ok_after, str(why[:1]))

    # --- M9 PF10: a patch with no heartbeat is rejected ---------------------------------------
    ctx5 = load_context(spec_path, box)
    hb = check_patch_heartbeats(ctx5)
    ctx6 = load_context(spec_path, box)
    ctx6.spec = dict(ctx6.spec)
    ctx6.spec["edits"] = [e for e in ctx6.spec.get("edits", [])
                          if e.get("code_file") != "scripts/kaggle_edits/assoc_feature_parity.py"]
    stripped = check_patch_heartbeats(ctx6)
    rec("M9 PF10 rejects a spec whose gate patch cannot say it ran",
        hb.passed and not stripped.passed,
        f"gate_patch={hb.evidence['gate_patch']} "
        f"silent={hb.evidence['patches_that_cannot_report_themselves']}")

    # --- M10 PF03: a selected crop that is not in the fold's stem list -----------------------
    ctx7 = load_context(spec_path, box)
    clean_crops = check_expected_crops(ctx7)
    ctx8 = load_context(spec_path, box)
    stems = json.loads(ctx8.env["BIOHUB_LOEO_STEMS"])
    ctx8.env["BIOHUB_LOEO_STEMS"] = json.dumps([s for s in stems if s != clean_crops.evidence
                                                ["selected_crops"][0]])
    dropped = check_expected_crops(ctx8)
    rec("M10 PF03 rejects a selected crop that the fold's stem list does not contain",
        clean_crops.passed and not dropped.passed,
        f"crops={clean_crops.evidence['selected_crops']}")

    # --- M11 PF05: a sidecar missing a required key ------------------------------------------
    maimed = box / "ecb_no_prob"
    maimed.mkdir(parents=True, exist_ok=True)
    for c in ctx.facts["crops"]:
        with np.load(mirror / "ecb" / f"{c}.npz", allow_pickle=False) as z:
            np.savez_compressed(maimed / f"{c}.npz", source_id=z["source_id"],
                                target_id=z["target_id"])          # edge_prob removed
    ctx9 = load_context(spec_path, box)
    ctx9.facts.update({"parquet": ctx.facts["parquet"], "crops": ctx.facts["crops"],
                       "ecb_dir": mirror / "ecb"})
    good_schema = check_schemas(ctx9)
    ctx9.facts["ecb_dir"] = maimed
    bad_schema = check_schemas(ctx9)
    rec("M11 PF05 rejects a sidecar missing edge_prob",
        good_schema.passed and not bad_schema.passed, "")

    # --- M12 PF08: the gate report dropped from the export keep-set --------------------------
    ctx10 = load_context(spec_path, box)
    good_keep = check_output_retained(ctx10)
    ctx10.nb_source = ctx10.nb_source.replace('_LOEO_KEEP |= {"assoc_feature_parity.json"}',
                                              '_LOEO_KEEP |= set()')
    dropped_keep = check_output_retained(ctx10)
    rec("M12 PF08 rejects a run whose gate report would be swept before fetch",
        good_keep.passed and not dropped_keep.passed, "")

    # --- M13 PF09: weights pointed at an absolute mount path ---------------------------------
    ctx11 = load_context(spec_path, box)
    good_w = check_model_paths(ctx11)
    ctx11.nb_source = re.sub(r'WEIGHTS_RELATIVE = f?"[^"]+"',
                             'WEIGHTS_RELATIVE = "/kaggle/input/pack/edge_predictor_best.pth"',
                             ctx11.nb_source, count=1)
    bad_w = check_model_paths(ctx11)
    rec("M13 PF09 rejects weights pinned to an absolute /kaggle/input path",
        good_w.passed and not bad_w.passed, "")

    # --- A3 ACCEPT: the CORRECT nested datasets root must PASS -------------------------------
    # The direction the doubled-prefix bug got wrong. A checker not shown to ACCEPT a correct
    # input is a checker nobody will trust, and PF09 was heading exactly there.
    nested = "/kaggle/input/datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1/" \
             "weights/full_frame_center/best.pt"
    flat = "/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1/" \
           "weights/full_frame_center/best.pt"
    parsed_nested = parse_mount_path(nested)
    parsed_flat = parse_mount_path(flat)
    pr_nested = probe_declared_mount_path(nested, box, "a3_nested")
    pr_flat = probe_declared_mount_path(flat, box, "a3_flat")
    a3 = (parsed_nested == ("pilkwang", "biohub-deepcenter-unet3d-center-prior-v1",
                            "weights/full_frame_center/best.pt")
          and parsed_flat == (None, "biohub-deepcenter-unet3d-center-prior-v1",
                              "weights/full_frame_center/best.pt")
          and all(v["found"] and v["no_duplicated_slug_segment"]
                  for v in pr_nested["resolved"].values())
          and all(v["found"] and v["no_duplicated_slug_segment"]
                  for v in pr_flat["resolved"].values()))
    rec("A3 ACCEPT PF09 passes the correct root in BOTH spellings, with no doubled prefix",
        a3, f"nested_rel={parsed_nested[2] if parsed_nested else None}")

    # --- M14 PF09: a genuinely wrong root must still FAIL -------------------------------------
    wrong_box = box / "m14"
    shutil.rmtree(wrong_box, ignore_errors=True)
    KM.simulate_mount_tree(wrong_box, "some-other-dataset", {"weights/best.pt": b"x"}, "datasets",
                           owner="pilkwang")
    wrong = KM.resolve("weights/full_frame_center/best.pt",
                       slugs=("biohub-deepcenter-unet3d-center-prior-v1",),
                       owners=("pilkwang",), input_root=str(wrong_box), require=False)
    ctx_w = load_context(spec_path, box)
    good_paths = check_model_paths(ctx_w)
    ctx_w.env["BIOHUB_DEEPCENTER_CHECKPOINT"] = flat
    ctx_w.nb_source = re.sub(r'WEIGHTS_RELATIVE = f?"[^"]+"',
                             'WEIGHTS_RELATIVE = "/kaggle/input/pack/w.pth"', ctx_w.nb_source, 1)
    bad_paths = check_model_paths(ctx_w)
    rec("M14 PF09 rejects a checkpoint whose slug is not mounted, and an absolute weights path",
        wrong is None and good_paths.passed and not bad_paths.passed,
        f"unmounted_slug_resolved={wrong}")

    # --- M15 class gating: a submission-class spec must SKIP, never PASS, the gate checks ------
    sub_spec = ROOT / "scripts" / "kaggle_specs" / "p32_public931_exact.json"
    if sub_spec.is_file():
        ctx_s = load_context(sub_spec, box)
        gate_checks = [check_expected_crops(ctx_s), check_schemas(ctx_s),
                       check_exact_positive_counts(ctx_s),
                       check_worker_subprocess_boundary(ctx_s)]
        all_skipped = all(c.evidence.get("_skipped") for c in gate_checks)
        none_examined = not any("per_crop" in c.evidence or "selected_crops" in c.evidence
                                for c in gate_checks)
        rec("M15 a submission-class spec SKIPS the Gate-1 checks instead of passing on P33's data",
            all_skipped and none_examined,
            f"class={ctx_s.facts['class']} skipped={[c.id for c in gate_checks]}")
        ctx_g = load_context(sub_spec, box)
        ctx_g.facts["class"] = "gate_smoke"      # a class that SHOULD have them
        forced = check_exact_positive_counts(ctx_g)
        rec("M15b an in-class spec with no implementing profile FAILS CLOSED rather than skipping",
            not forced.passed and not forced.evidence.get("_skipped"),
            str(forced.evidence.get("fail_closed", ""))[:40])

    # --- M16 / A4 PF01: the dirty-worktree question must be scoped to the spec's own patches ----
    empty_q = _git_dirty([])
    own_dirty = _git_dirty(["scripts/win_bet/gpu_preflight.py"])
    rec("A4 ACCEPT PF01 asks nothing of a spec with no patch files, and inherits no other "
        "agent's dirt", empty_q == [], f"whole_worktree_leak={len(empty_q)}")
    rec("M16 PF01 still rejects a spec whose OWN patch file is uncommitted",
        bool(own_dirty), f"{own_dirty[:1]}")

    total = len(results)
    bad = [n for n, ok, _ in results if not ok]
    print(f"\nSELFTEST {total - len(bad)}/{total} mutations behaved as required", flush=True)
    if bad:
        print("UNPROVEN CHECKS: " + ", ".join(bad), flush=True)
    return 0 if not bad else 1


# --------------------------------------------------------------------------------------- cli
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="preflight a spec's BUILT notebook")
    r.add_argument("--spec", required=True)
    r.add_argument("--sandbox", default="C:/temp/foundations/preflight")
    r.add_argument("--receipt")
    v = sub.add_parser("verify-receipt")
    v.add_argument("--receipt", required=True)
    s = sub.add_parser("selftest")
    s.add_argument("--sandbox", default="C:/temp/foundations/preflight")
    a = ap.parse_args(argv)

    if a.cmd == "selftest":
        print("PREFLIGHT SELFTEST - every check must reject the defect built for it")
        rc = _selftest(Path(a.sandbox))
        print(HEARTBEAT + f" selftest rc={rc}")
        return rc

    if a.cmd == "verify-receipt":
        receipt = json.loads(Path(a.receipt).read_text(encoding="utf-8"))
        ok, problems = verify(receipt)
        print(json.dumps({"receipt": a.receipt, "signature_valid": ok, "problems": problems},
                         indent=2))
        print(HEARTBEAT + f" verify={'OK' if ok else 'TAMPERED'}")
        return 0 if ok else 1

    receipt = run_preflight(Path(a.spec).resolve(), Path(a.sandbox))
    body = receipt["body"]
    for c in body.get("checks", []):
        mark = "SKIP" if c["evidence"].get("_skipped") else ("PASS" if c["passed"] else "FAIL")
        print(f"  [{mark}] {c['id']}  {c['title']}")
    print(f"\nartifact class: {body.get('artifact_class')} "
          f"({body.get('artifact_class_reason')})")
    print(f"VERDICT: {body['verdict']}   failed={body.get('failed')}   "
          f"skipped={body.get('skipped')}")
    if a.receipt:
        Path(a.receipt).parent.mkdir(parents=True, exist_ok=True)
        Path(a.receipt).write_text(json.dumps(receipt, indent=2, default=str), encoding="utf-8")
        print(f"receipt -> {a.receipt}")
    print(HEARTBEAT + f" verdict={body['verdict']}")
    return 0 if body["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
