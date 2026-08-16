"""Prove the gt_to_sub collision guard changed no number at the five patched sites.

The guard is first-writer-wins-then-raise; the code it replaced was last-writer-wins.
Those differ ONLY when the match is non-injective. Zero collisions were measured on the
current E0c cache, so the outputs must be identical -- this script demonstrates it per
site instead of asserting it, by running one real crop twice:

    legacy    ``gt_collision.gt_maps_from_matches`` monkeypatched back to the exact
              unguarded loop that used to be inlined at the site
    guarded   the shipped guard

and comparing the worker's full result dict with an exact JSON comparison. Each site's
worker is called directly (no ProcessPoolExecutor) so the monkeypatch is in scope.

Usage:
  .venv\\Scripts\\python.exe scripts\\verify_gt_collision_parity.py [--split 0] [--json-out P]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = next(_p for _p in Path(__file__).resolve().parents if (_p / 'pyproject.toml').exists())
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))

import gt_collision  # noqa: E402

GUARDED = gt_collision.gt_maps_from_matches
# Wall-clock fields are not results; they differ between two runs of identical code.
VOLATILE = ("runtime_s",)


def legacy(sub_to_int, int_to_gt, *, context="", allow_collisions=False):
    """Byte-for-byte the pre-fix loop: no guard, last writer wins."""
    gt_to_sub, sub_to_gt = {}, {}
    for s, iid in sub_to_int.items():
        mid = int_to_gt.get(iid)
        if mid not in (None, -1):
            gt_to_sub[int(mid)] = s
            sub_to_gt[s] = int(mid)
    return gt_to_sub, sub_to_gt, 0


def run(fn, args):
    gt_collision.gt_maps_from_matches = legacy
    try:
        before = fn(args)
    finally:
        gt_collision.gt_maps_from_matches = GUARDED
    after = fn(args)
    return before, after


def sites(split: int, crop: str):
    import phaseb_d0p_proposer as d0p
    import phaseb_h0b_rankcompress as h0b
    import phaseb_h0c_replay as h0c
    import phaseb_h1a_census as h1a
    import phaseb_oracle_d0prime as d0prime

    return [
        ("phaseb_d0p_proposer.py:287", d0p.audit_one, (split, crop, "geometric_core", False)),
        ("phaseb_h0c_replay.py:147", h0c.replay_one, (split, crop)),
        ("phaseb_h0b_rankcompress.py:138", h0b.audit_one, (split, crop)),
        ("phaseb_h1a_census.py:87", h1a.census_one, (split, crop)),
        ("phaseb_oracle_d0prime.py:215", d0prime.score_one,
         (split, crop, "suppress_all_then_add_replace", False, False)),
    ]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--crop", default=None)
    ap.add_argument("--json-out", type=Path)
    a = ap.parse_args(argv)

    from phaseb_d0p_proposer import cached_crops
    crop = a.crop or cached_crops(a.split)[0]

    report = {"split": a.split, "crop": crop, "sites": []}
    ok = True
    for name, fn, args in sites(a.split, crop):
        before, after = run(fn, args)
        for d in (before, after):
            if isinstance(d, dict):
                for k in VOLATILE:
                    d.pop(k, None)
        b = json.dumps(before, sort_keys=True, default=float)
        c = json.dumps(after, sort_keys=True, default=float)
        same = b == c
        ok &= same
        report["sites"].append({
            "site": name, "identical": same, "result_chars": len(c),
            "diff": None if same else {"legacy": b[:2000], "guarded": c[:2000]},
        })
        print(f"{'IDENTICAL' if same else 'DIFFERENT':<10} {name}  ({len(c)} chars of result)")

    report["verdict"] = "IDENTICAL" if ok else "DIVERGENT"
    print(f"\nVERDICT: {report['verdict']}  (split={a.split} crop={crop})")
    if a.json_out:
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        a.json_out.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
