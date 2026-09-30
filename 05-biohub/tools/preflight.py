#!/usr/bin/env python3
"""Syntax plus a bounded, explicit CPU smoke-check contract for standalone scripts.

The script must support --preflight and print one JSON line with
{"biohub_check":"preflight", "device":"cpu", "ok":true} after completing its smoke test.
This is the script's attestation, not proof of its dependencies or remote environment.
"""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys


def main(script):
    path = Path(script).resolve(strict=True)
    ast.parse(path.read_text(), filename=str(path))
    result = subprocess.run([sys.executable, str(path), '--preflight'],
                            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'BIOHUB_PREFLIGHT': '1'},
                            capture_output=True, text=True, timeout=300)
    if result.returncode:
        raise ValueError(f'CPU smoke check failed: {result.stderr[-1000:]}')
    for line in result.stdout.splitlines():
        try:
            report = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(report, dict) and report.get('biohub_check') == 'preflight' and report.get('device') == 'cpu' and report.get('ok') is True:
            print('PASS: script reports successful CPU smoke check')
            return 0
    raise ValueError('No explicit successful CPU preflight report; exit status alone is insufficient')


if __name__ == '__main__':
    try:
        if len(sys.argv) != 2:
            raise ValueError('usage: preflight.py script.py')
        sys.exit(main(sys.argv[1]))
    except (OSError, ValueError, SyntaxError, subprocess.TimeoutExpired) as exc:
        sys.exit(f'BLOCKED: {exc}')
