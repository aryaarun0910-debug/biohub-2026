"""Safely inventory external PyTorch state dictionaries without executing pickle code."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def inspect_checkpoint(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected mapping, got {type(payload).__name__}")
    state = payload.get("model_state_dict", payload.get("state_dict", payload))
    if not isinstance(state, dict) or not state:
        raise TypeError(f"{path}: no state dictionary found")
    tensors = {str(key): value for key, value in state.items() if isinstance(value, torch.Tensor)}
    if not tensors:
        raise TypeError(f"{path}: state dictionary has no tensors")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest,
        "top_level_keys": sorted(str(key) for key in payload),
        "tensor_count": len(tensors),
        "state_elements": int(sum(value.numel() for value in tensors.values())),
        "tensors": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in sorted(tensors.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = {path.name: inspect_checkpoint(path) for path in args.checkpoints}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"CHECKPOINT_INVENTORY files={len(result)} out={args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
