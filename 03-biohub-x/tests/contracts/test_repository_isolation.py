"""Repository isolation and hygiene contract.

Biohub-X is a blind, independent attempt (DECISIONS.md D-0001). A prior campaign
on this task exists elsewhere on some machines and must be treated as
nonexistent: not opened, not searched, not imported, not depended on, not
compared against during design.

Isolation is not enforceable by good intentions, so this module enforces the
properties that make it checkable:

* the repository is its own Git toplevel, not nested inside another repository;
* no quarantined name appears in the repository path, in any tracked file, or on
  the import path;
* no tracked file carries an absolute filesystem path outside the allow-listed
  provenance records, which is how a hidden dependency on another checkout would
  first appear;
* no import path entry resolves outside this repository, its virtual
  environment and the standard library;
* the repository byte policy actually holds on disk.

This file is the authoritative quarantine list. Changing it is a stop-and-report
action (AGENTS.md section 7); do not edit it as part of another change.
"""

from __future__ import annotations

import re
import subprocess
import sys
import sysconfig
from collections.abc import Iterable
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

QUARANTINED_NAMES = (
    # Prior campaign on this task. Named here so that its absence can be
    # asserted; nothing in this repository reads it.
    "Biohub-CellTracking-2026",
)

# Files permitted to contain an absolute filesystem path. Provenance records
# must state where an artifact came from; this module must name what it
# quarantines; and the negative-control tests that prove absolute paths are
# refused must contain absolute paths in order to be refused. Nothing else has a
# reason to. This set is not a convenience: each entry removes a guarantee.
ABSOLUTE_PATH_ALLOWLIST = frozenset(
    {
        "registry/artifacts.yaml",
        "research/primitives-dossier.md",
        "tests/contracts/test_repository_isolation.py",
        "tests/contracts/test_artifact_registry.py",
    }
)

# Files permitted to name a quarantined repository.
QUARANTINE_MENTION_ALLOWLIST = frozenset({"tests/contracts/test_repository_isolation.py"})

# A drive-letter path (C:\... or C:/...), with a lookbehind so that URL schemes
# such as https:// do not match.
_WINDOWS_ABSOLUTE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")
# A POSIX absolute path rooted in a user or mount directory.
_POSIX_ABSOLUTE = re.compile(r"(?<![\w.~-])/(?:home|Users|mnt|media)/")


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def tracked_files() -> list[str]:
    """Repository-relative paths of every tracked file.

    Git is required, not optional: a silently skipped hygiene test is worse than
    no test at all.
    """
    output = _git("ls-files", "-z")
    names = [name for name in output.split("\0") if name]
    # An empty index makes every content scan below pass vacuously. That is the
    # failure mode where isolation looks enforced and is not.
    assert names, "no tracked files: the hygiene scans would pass without inspecting anything"
    return names


def _tracked_text_files() -> list[tuple[str, str]]:
    files: list[tuple[str, str]] = []
    for name in tracked_files():
        raw = (REPO_ROOT / name).read_bytes()
        try:
            files.append((name, raw.decode("utf-8")))
        except UnicodeDecodeError:
            continue  # binary artifact; the byte and path checks below still apply
    return files


# --- the repository is its own root ---------------------------------------


def test_repository_is_its_own_git_toplevel() -> None:
    toplevel = Path(_git("rev-parse", "--show-toplevel").strip()).resolve()
    assert toplevel == REPO_ROOT, (
        f"Biohub-X must be a standalone repository. Git reports its toplevel as {toplevel}, "
        f"but this package lives under {REPO_ROOT}."
    )


def test_repository_path_contains_no_quarantined_name() -> None:
    parts = set(REPO_ROOT.parts)
    for name in QUARANTINED_NAMES:
        assert name not in parts, f"repository is nested inside quarantined directory {name!r}"


# --- nothing references a quarantined repository ---------------------------


def _quarantine_offenders(files: Iterable[tuple[str, str]]) -> list[str]:
    offenders: list[str] = []
    for name, text in files:
        if name in QUARANTINE_MENTION_ALLOWLIST:
            continue
        for quarantined in QUARANTINED_NAMES:
            if quarantined in text:
                offenders.append(f"{name} mentions {quarantined!r}")
    return offenders


def test_no_tracked_file_names_a_quarantined_repository() -> None:
    offenders = _quarantine_offenders(_tracked_text_files())
    assert not offenders, "quarantined repository referenced:\n" + "\n".join(offenders)


def test_the_quarantine_scan_actually_fires() -> None:
    # The scan above currently finds nothing, which is indistinguishable from a
    # scan that cannot find anything. Plant the exact shape a leak would take.
    assert QUARANTINED_NAMES, "an empty quarantine list makes the scan vacuous"
    leaked = f"from {QUARANTINED_NAMES[0]}.src.tracking import matcher\n"
    assert _quarantine_offenders([("src/biohubx/tracking/matcher.py", leaked)])
    # Exemption is by filename, not by content: this module may name what it
    # quarantines, and nothing else may.
    for exempt in QUARANTINE_MENTION_ALLOWLIST:
        assert not _quarantine_offenders([(exempt, leaked)])


def test_no_import_path_entry_touches_a_quarantined_repository() -> None:
    for entry in sys.path:
        if not entry:
            continue
        parts = set(Path(entry).resolve().parts)
        for quarantined in QUARANTINED_NAMES:
            assert quarantined not in parts, f"sys.path entry {entry!r} reaches {quarantined!r}"


def test_import_path_stays_inside_the_repository_venv_and_stdlib() -> None:
    permitted = [
        REPO_ROOT,
        Path(sys.prefix).resolve(),
        Path(sys.base_prefix).resolve(),
        Path(sysconfig.get_path("stdlib")).resolve(),
        Path(sysconfig.get_path("purelib")).resolve(),
    ]
    stray: list[str] = []
    for entry in sys.path:
        if not entry:
            continue
        resolved = Path(entry).resolve()
        if not any(resolved.is_relative_to(root) for root in permitted):
            stray.append(str(resolved))
    assert not stray, (
        "import path reaches outside the repository, its environment and the standard library: "
        + ", ".join(stray)
    )


def test_biohubx_is_imported_from_this_repository() -> None:
    import biohubx

    package = Path(biohubx.__file__).resolve()
    assert package.is_relative_to(REPO_ROOT), f"biohubx imported from {package}, outside {REPO_ROOT}"


# --- no hidden absolute paths ---------------------------------------------


@pytest.mark.parametrize("pattern", [_WINDOWS_ABSOLUTE, _POSIX_ABSOLUTE], ids=["windows", "posix"])
def test_no_hidden_absolute_paths_in_tracked_files(pattern: re.Pattern[str]) -> None:
    offenders: list[str] = []
    for name, text in _tracked_text_files():
        if name in ABSOLUTE_PATH_ALLOWLIST:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                offenders.append(f"{name}:{number}: {line.strip()}")
    assert not offenders, "absolute filesystem paths must not appear in code or configuration:\n" + "\n".join(
        offenders
    )


def test_the_absolute_path_detector_actually_fires() -> None:
    # A hygiene test that cannot fail is not a hygiene test. These are the exact
    # shapes a leaked path from another checkout would take.
    assert _WINDOWS_ABSOLUTE.search(r"C:\Users\someone\Documents\other-repo")
    assert _WINDOWS_ABSOLUTE.search("C:/Users/someone/Documents/other-repo")
    assert _POSIX_ABSOLUTE.search("/home/someone/other-repo")
    # ...and must not fire on the things it would otherwise flood on.
    assert not _WINDOWS_ABSOLUTE.search("https://example.org/x")
    assert not _WINDOWS_ABSOLUTE.search("see SYSTEM.md: architecture")
    assert not _POSIX_ABSOLUTE.search("src/biohubx/hashing.py")


# --- byte policy holds on disk --------------------------------------------


def test_no_tracked_text_file_contains_crlf() -> None:
    # .gitattributes sets `* text=auto eol=lf` (DECISIONS.md D-0003). If a file
    # on disk carries CRLF, the policy is not actually in force and every
    # recorded raw digest of a text file is platform-dependent.
    offenders = [name for name, _ in _tracked_text_files() if b"\r\n" in (REPO_ROOT / name).read_bytes()]
    assert not offenders, "CRLF found in tracked text files: " + ", ".join(offenders)


def test_gitattributes_pins_line_endings_before_any_digest_is_trusted() -> None:
    attributes = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "* text=auto eol=lf" in attributes
    # The hash-bound snapshot must never be normalised, or its recorded raw
    # digest stops being an assertion about the original source bytes.
    assert "research/primitives-dossier.md -text" in attributes


def test_governing_documents_exist() -> None:
    for name in ("README.md", "SYSTEM.md", "AGENTS.md", "DECISIONS.md"):
        assert (REPO_ROOT / name).is_file(), f"missing governing document: {name}"
    for name in ("experiments", "models", "artifacts", "findings"):
        assert (REPO_ROOT / "registry" / f"{name}.yaml").is_file(), f"missing registry: {name}.yaml"


def test_there_is_no_general_purpose_scripts_directory() -> None:
    # AGENTS.md section 3 and DECISIONS.md D-0004: operations are CLI
    # subcommands, because a scripts drawer is where untested branching logic,
    # hidden paths and silent defaults survive.
    assert not (REPO_ROOT / "scripts").exists()
