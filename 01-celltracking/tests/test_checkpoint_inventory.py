from __future__ import annotations

import hashlib

import torch

from scripts.win_bet.checkpoint_inventory import inspect_checkpoint


def test_weights_only_checkpoint_inventory(tmp_path):
    path = tmp_path / "model.pt"
    torch.save({"state_dict": {"head.weight": torch.ones(1, 4, 1, 1, 1),
                               "head.bias": torch.zeros(1)}}, path)
    got = inspect_checkpoint(path)
    assert got["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert got["tensor_count"] == 2
    assert got["state_elements"] == 5
    assert got["tensors"]["head.weight"]["shape"] == [1, 4, 1, 1, 1]
