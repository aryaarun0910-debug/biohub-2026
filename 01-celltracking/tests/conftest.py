"""Put the grouped scripts/ subdirs on sys.path so tests can `import <script>` by bare
name after the 2026-08-16 scripts/{core,metric,d1}/ reorg. Auto-loaded by pytest."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _sub in ("scripts", "scripts/core", "scripts/metric", "scripts/d1", "scripts/win_bet"):
    _p = str(_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
