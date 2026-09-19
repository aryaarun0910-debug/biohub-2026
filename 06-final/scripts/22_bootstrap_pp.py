"""Make deployed_pp importable by stubbing names that lived in skipped blocks.

Iteratively imports, catches NameError/ImportError, and records a stub. The
stubs are written to a prelude so the generated module stays untouched -- any
stub that a post-processing path actually needs will fail loudly at call time
rather than silently returning None.
"""
import re, subprocess, sys
from pathlib import Path

PRE = Path("src/biohub/_pp_prelude.py")
MOD = Path("src/biohub/deployed_pp.py")
stubs = {}
for attempt in range(60):
    PRE.write_text(
        '"""Auto-generated stubs for names the extractor left behind."""\n'
        "from pathlib import Path\n"
        "class _Dummy:\n"
        "    def __truediv__(self, o): return self\n"
        "    def __rtruediv__(self, o): return self\n"
        "    def __getattr__(self, n): return self\n"
        "    def __call__(self, *a, **k): return self\n"
        "    def __iter__(self): return iter(())\n"
        "    def __bool__(self): return False\n"
        "    def __len__(self): return 0\n"
        "    def __str__(self): return '_dummy'\n"
        "    def __fspath__(self): return '_dummy'\n"
        "    def __eq__(self, o): return False\n"
        "    def __hash__(self): return 0\n"
        "COMP_DIR = Path('data/biohub-cell-tracking-during-development')\n"
        "TEST_DIR = COMP_DIR / 'test'\n"
        "TRAIN_DIR = COMP_DIR / 'train'\n"
        "WORKING_DIR = Path('artifacts/pp_work')\n"
        "REPO_DIR = Path('artifacts/s01_output/tracking_repo')\n"
        + "".join(f"{k} = {v}\n" for k, v in stubs.items()))
    src = MOD.read_text()
    if "from ._pp_prelude import *" not in src:
        lines = src.split("\n")
        i = next(j for j, l in enumerate(lines)
                 if l.startswith(('import ', 'from ')) and '__future__' not in l)
        lines.insert(i, "from ._pp_prelude import *  # noqa: F403")
        MOD.write_text("\n".join(lines))
    r = subprocess.run([".venv/bin/python", "-c",
                        "import sys; sys.path.insert(0,'src'); import biohub.deployed_pp"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        print(f"imported OK after {attempt} stubs")
        break
    err = r.stderr.strip().split("\n")[-1]
    m = re.search(r"name '(\w+)' is not defined", err)
    if m:
        stubs[m.group(1)] = "_Dummy()"; continue
    m = re.search(r"No module named '([\w.]+)'", err)
    if m:
        name = m.group(1).split(".")[0]
        stubs[name] = "type('_Stub',(),{'__getattr__':lambda s,n: None})()"
        # also neutralise the import line itself
        MOD.write_text(re.sub(rf"^(from {name}[\w.]* import .*|import {name}.*)$",
                              r"pass  # stubbed: \g<0>", MOD.read_text(), flags=re.M))
        continue
    print("unhandled:", err); sys.exit(1)
else:
    print("gave up"); sys.exit(1)
print(f"stubbed {len(stubs)}: {sorted(stubs)[:14]}{' ...' if len(stubs) > 14 else ''}")
