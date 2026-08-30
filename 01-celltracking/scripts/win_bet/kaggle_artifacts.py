r"""ONE artifact-discovery surface for the two fetch conventions that coexist in this repository.

WHY
---
`kaggle_factory fetch` writes to `<out_dir>/_out/`. `kaggle_queue.py` writes to
`C:/temp/queue/<spec name>/`. Nothing recorded which produced which artifact, and on 2026-08-30
an audit that searched only the factory path reported the OUTGOING CHAMPION as having no bound
submission while a complete, passing artifact had been sitting at the queue path since 2026-08-27.
That false FAIL is corrected in FACT-0396, and its lesson is named in AGENTS.md: an unfinished
search became a verified negative.

So discovery is centralised, both conventions are searched, the convention that produced the hit
is RECORDED rather than assumed, and a miss returns a fail-closed result carrying every location
that was tried - never an exception that reads like "nothing to check".

ROLE ALIASES ARE PART OF THE PROBLEM, NOT A DETAIL
--------------------------------------------------
The same role has different filenames per convention, measured on disk 2026-08-30:
  C:/temp/queue/p24_deepcenter_best_veto/  -> audit_receipt.json, structural_audit.json, run_stats.csv
  notebooks/kaggle_p32_public931_exact/_out/ -> audit_receipt.json, structural_audit.json
  C:/temp/p32/                             -> audit.json
  C:/temp/p33/                             -> kernel.log
  C:/temp/p33_v2/                          -> biohub-p33-assoc-feature-parity-smoke.log
An instrument that hardcodes one spelling silently reports the artifact as absent.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUEUE_ROOT = Path("C:/temp/queue")
SCRATCH_ROOT = Path("C:/temp")

# role -> filename candidates, most canonical first. `{slug}` is substituted per spec.
ROLE_ALIASES: dict[str, tuple[str, ...]] = {
    "receipt": ("audit_receipt.json", "audit.json"),
    "structural_audit": ("structural_audit.json", "audit.json"),
    "submission": ("submission.csv", "submission.csv.gz"),
    "run_stats": ("run_stats.csv",),
    "log": ("kernel.log", "{slug}.log", "log.json"),
    "gate_report": ("assoc_feature_parity.json",),
}


@dataclass
class Discovery:
    """What was found, WHERE, and by which convention - plus everything that was tried."""
    spec_name: str
    slug: str
    found: bool
    convention: str | None
    root: Path | None
    files: dict[str, Path] = field(default_factory=dict)
    searched: list[str] = field(default_factory=list)
    sha256: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "spec_name": self.spec_name,
            "slug": self.slug,
            "found": self.found,
            "convention": self.convention,
            "root": str(self.root) if self.root else None,
            "files": {k: str(v) for k, v in sorted(self.files.items())},
            "sha256": dict(sorted(self.sha256.items())),
            "searched": list(self.searched),
        }


def candidate_roots(spec: dict, override: Path | None = None) -> list[tuple[str, Path]]:
    """Every place a fetched artifact for this spec has ever been written, labelled."""
    name = str(spec.get("name") or "")
    out_dir = spec.get("out_dir")
    roots: list[tuple[str, Path]] = []
    if override:
        roots.append(("override", Path(override)))
    if out_dir:
        roots.append(("factory", ROOT / out_dir / "_out"))
    if name:
        roots.append(("queue", QUEUE_ROOT / name))
        roots.append(("scratch", SCRATCH_ROOT / name))
        short = name.split("_")[0]
        if short and short != name:
            roots.append(("scratch_short", SCRATCH_ROOT / short))
    return roots


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discover(spec: dict, roles: tuple[str, ...] = ("receipt", "submission"),
             override: Path | None = None, hash_files: bool = True) -> Discovery:
    """Locate a fetched artifact set under EITHER convention.

    A root wins when it supplies the FIRST role in `roles` - the caller states what makes a
    directory the real artifact rather than a leftover. Every root that was looked at is recorded
    whether it hit or not, so a negative is a finished search rather than an abandoned one.
    """
    slug = str(spec.get("slug") or "")
    searched: list[str] = []
    for label, root in candidate_roots(spec, override):
        searched.append(f"{label}:{root}")
        if not root.is_dir():
            continue
        files: dict[str, Path] = {}
        for role in roles:
            for alias in ROLE_ALIASES.get(role, ()):
                cand = root / alias.replace("{slug}", slug)
                if cand.is_file():
                    files[role] = cand
                    break
        if roles and roles[0] in files:
            return Discovery(
                spec_name=str(spec.get("name") or ""), slug=slug, found=True,
                convention=label, root=root, files=files, searched=searched,
                sha256={k: _sha256(v) for k, v in files.items()} if hash_files else {},
            )
    return Discovery(spec_name=str(spec.get("name") or ""), slug=slug, found=False,
                     convention=None, root=None, searched=searched)


def load_spec(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = ["Discovery", "ROLE_ALIASES", "candidate_roots", "discover", "load_spec"]
