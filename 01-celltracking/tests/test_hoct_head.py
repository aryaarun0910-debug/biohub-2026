"""Contracts for the reconstructed HOCT association head (PKT-0029 STEP 0).

Software contracts only. Whether HOCT chooses better parents is an experiment result.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.win_bet.hoct_head import (
    D_MODEL,
    N_HEADS,
    NODE_EXTRAS,
    PAIR_EXTRAS,
    RELATION_IN,
    HoctAssociationHead,
    load_checkpoint,
)

CKPT_DIR = Path("C:/temp/arch_inventory/hoct/weights")
FOLDS = [
    CKPT_DIR / "fold0_hoct_hard_negative_9294.pt",
    CKPT_DIR / "fold1_hoct_hard_negative_fold1_9294.pt",
]


def test_reconstruction_matches_the_published_shapes():
    """The dims the checkpoint DOES determine. Getting any of these wrong makes the strict
    load fail, which is the point of stating them as constants rather than magic numbers."""
    model = HoctAssociationHead()
    assert model.node_projection.in_features == 35        # 32 UNet channels + 3 extras
    assert model.node_projection.out_features == D_MODEL
    assert model.frame_embedding.num_embeddings == 2      # window_size 2
    assert len(model.edge_blocks) == 3
    assert model.edge_projection[0].in_features == 196    # 96 src + 96 tgt + 4 pair extras
    # The relation bias emits one scalar PER ATTENTION HEAD - that is how head count is known.
    assert model.edge_blocks[0].relation_bias[-1].out_features == N_HEADS
    assert model.edge_blocks[0].relation_bias[0].in_features == RELATION_IN


def test_the_undetermined_contract_is_stated_not_guessed():
    """PKT-0029 falsifier (b): a wrong feature ordering loads cleanly and scores near chance.
    The module must not silently invent one, so it exposes the slot counts and no defaults."""
    assert NODE_EXTRAS == 3
    assert PAIR_EXTRAS == 4
    assert RELATION_IN == 13
    # No forward() that would require inventing a feature layout.
    assert not hasattr(HoctAssociationHead, "forward") or \
        HoctAssociationHead.forward is torch.nn.Module.forward


@pytest.mark.parametrize("path", FOLDS, ids=lambda p: p.name)
def test_published_checkpoints_load_strict(path):
    """Both folds must load with strict=True: every published tensor has a home of the right
    shape. Necessary, and NOT sufficient - it says nothing about the feature contract."""
    if not path.exists():
        pytest.skip(f"checkpoint not present: {path}")
    model, meta = load_checkpoint(path)
    assert sum(p.numel() for p in model.parameters()) == 447376
    assert meta.get("method") == "hoct_hard_negative_v1"


def test_fork_head_exists_but_is_not_wired_into_any_scoring_path():
    """FACT-0347: its published division recall is zero, on 110 positive pairs (FACT-0362).
    It is reconstructed so the strict load covers the whole checkpoint, and deliberately unused."""
    model = HoctAssociationHead()
    assert model.fork_head[0].in_features == 197
    source = Path("scripts/win_bet/hoct_head.py").read_text(encoding="utf-8")
    assert "fork_head(" not in source, "the fork head must not be invoked anywhere"
