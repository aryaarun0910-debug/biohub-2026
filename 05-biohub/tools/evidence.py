"""Verify frozen evidence before interpreting it; ingestion time is not observation time."""
import hashlib
import json
from pathlib import Path


def verify(root):
    root = Path(root).resolve()
    manifest = {}
    for line in (root / 'index.jsonl').read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        path = (root / row['file']).resolve()
        if not path.is_relative_to(root / 'raw') or path in manifest:
            raise ValueError(f'Unsafe or duplicate evidence path: {row["file"]}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError(f'Evidence digest mismatch: {row["file"]}')
        if not row.get('fetched_at'):
            raise ValueError(f'Missing acquisition timestamp: {row["file"]}')
        manifest[path] = row
    files = {p.resolve() for p in (root / 'raw').rglob('*') if p.is_file()}
    if files != set(manifest):
        raise ValueError('Evidence index does not exactly cover raw files')
    return manifest
