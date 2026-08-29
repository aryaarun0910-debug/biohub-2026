r"""Reconstruction of the published HOCT association head, and its strict-load gate.

WHY A RECONSTRUCTION AT ALL
---------------------------
``FACT-0361`` established that HOCT's base trunk is architecturally identical to ours - 136 tensors,
2,077,996 parameters, same shapes - so its association head is a head swap rather than a port, and
``FACT-0368``/``FACT-0376`` put the campaign's remaining headroom in exactly that: choosing the right
parent. But the publisher shipped CHECKPOINTS AND A CONFIG, NO MODEL CODE, and searching their
Kaggle account and the platform for a StableDet/HOCT kernel returns nothing. The architecture
therefore has to be rebuilt from the state dict.

WHAT THE TENSORS DETERMINE, AND WHAT THEY DO NOT
-----------------------------------------------
Determined exactly, and asserted by the strict load below:

    node_projection   Linear(35 -> 96)          35 = 32 UNet channels + 3 extras
    frame_embedding   Embedding(2, 96)          window_size 2
    node_encoder      2 x TransformerEncoderLayer(d_model 96, ffn 192)
    edge_projection   Linear(196 -> 96) + LayerNorm    196 = 96 src + 96 tgt + 4 pair extras
    edge_blocks       3 x { pre-norm attention with q/k/v/out, a relation bias
                            MLP(13 -> 48 -> 4), and MLP(96 -> 192 -> 96) }
                      the relation bias emits 4 values per pair, i.e. FOUR ATTENTION HEADS
    edge_head         LayerNorm(96) + Linear(96 -> 1)

NOT determined by the tensors, and this is the load-bearing risk (``PKT-0029`` falsifier b):
WHICH 3 node extras, WHICH 4 pair extras, and WHICH 13 relation features, in WHAT ORDER. A wrong
choice loads cleanly under ``strict=True`` and then scores near chance - a silent failure. So this
module deliberately takes those features as an explicit, named contract supplied by the caller and
refuses to invent a default. Pinning the contract is an EXPERIMENT (rank candidate contracts by
held-out parent-choice accuracy), not a code decision, and it is not made here.

THE AUXILIARY HEADS ARE LOADED AND NOT USED
-------------------------------------------
``quiet_head``, ``division_head`` and ``fork_head`` are reconstructed so the strict load covers the
whole checkpoint, but ``PKT-0029`` forbids using the fork head: its published division recall is
zero (``FACT-0347``) on 110 positive training pairs (``FACT-0362``), and an association gain must
never be read as division recovery (``FACT-0371``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

D_MODEL = 96
N_HEADS = 4
NODE_IN = 35
UNET_CHANNELS = 32
NODE_EXTRAS = NODE_IN - UNET_CHANNELS      # 3
EDGE_PROJ_IN = 196
PAIR_EXTRAS = EDGE_PROJ_IN - 2 * D_MODEL   # 4
RELATION_IN = 13
FORK_IN = 197


class EdgeBlock(nn.Module):
    """Pre-norm attention over candidate pairs with a learned relation bias.

    The bias MLP maps ``RELATION_IN`` per-pair geometric features to one scalar PER HEAD, which is
    how the block injects geometry into attention rather than into the token features.
    """

    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.q = nn.Linear(D_MODEL, D_MODEL)
        self.k = nn.Linear(D_MODEL, D_MODEL)
        self.v = nn.Linear(D_MODEL, D_MODEL)
        self.out = nn.Linear(D_MODEL, D_MODEL)
        self.relation_bias = nn.Sequential(
            nn.Linear(RELATION_IN, 48), nn.GELU(), nn.Linear(48, N_HEADS),
        )
        self.mlp = nn.Sequential(
            nn.Linear(D_MODEL, 192), nn.GELU(), nn.Dropout(0.0), nn.Linear(192, D_MODEL),
        )


class HoctAssociationHead(nn.Module):
    """The published HOCT association head, rebuilt to match its state dict exactly."""

    def __init__(self) -> None:
        super().__init__()
        self.node_projection = nn.Linear(NODE_IN, D_MODEL)
        self.frame_embedding = nn.Embedding(2, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=192, batch_first=True,
        )
        self.node_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.edge_projection = nn.Sequential(
            nn.Linear(EDGE_PROJ_IN, D_MODEL), nn.GELU(), nn.LayerNorm(D_MODEL),
        )
        self.edge_blocks = nn.ModuleList([EdgeBlock() for _ in range(3)])
        self.edge_head = nn.Sequential(nn.LayerNorm(D_MODEL), nn.Linear(D_MODEL, 1))
        self.quiet_head = nn.Sequential(nn.LayerNorm(D_MODEL), nn.Linear(D_MODEL, 1))
        self.division_head = nn.Sequential(
            nn.LayerNorm(3 * D_MODEL), nn.Linear(3 * D_MODEL, D_MODEL),
            nn.GELU(), nn.Dropout(0.0), nn.Linear(D_MODEL, 1),
        )
        self.fork_head = nn.Sequential(
            nn.Linear(FORK_IN, D_MODEL), nn.GELU(), nn.Dropout(0.0), nn.Linear(D_MODEL, 1),
        )


def load_checkpoint(path: Path) -> tuple[HoctAssociationHead, dict]:
    """Build the module and load the published weights with strict=True.

    Strict is the point: it proves every tensor in the checkpoint has a home of exactly the right
    shape. It does NOT prove the feature contract is right - see the module docstring.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["model"] if "model" in payload else payload
    model = HoctAssociationHead()
    model.load_state_dict(state, strict=True)
    model.eval()
    meta = {k: v for k, v in payload.items() if k not in {"model", "optimizer"}} \
        if isinstance(payload, dict) else {}
    return model, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    rows = []
    for path in args.checkpoints:
        entry: dict = {"checkpoint": str(path)}
        try:
            model, meta = load_checkpoint(path)
            entry["strict_load"] = True
            entry["parameters"] = int(sum(p.numel() for p in model.parameters()))
            entry["method"] = str(meta.get("method"))
            entry["gate_um"] = meta.get("gate_um")
            entry["edge_neighbors"] = meta.get("edge_neighbors")
            entry["feature_dim"] = meta.get("feature_dim")
        except Exception as exc:  # noqa: BLE001 - the failure IS the result here
            entry["strict_load"] = False
            entry["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(entry)
        status = "OK  " if entry["strict_load"] else "FAIL"
        print(f"  {status} {Path(entry['checkpoint']).name}"
              + (f"  params={entry['parameters']:,}  method={entry['method']}"
                 if entry["strict_load"] else f"\n       {entry['error'][:300]}"))

    passed = all(r["strict_load"] for r in rows)
    result = {
        "schema_version": 1,
        "all_strict_loads_passed": passed,
        "checkpoints": rows,
        "undetermined_contract": {
            "node_extras": NODE_EXTRAS,
            "pair_extras": PAIR_EXTRAS,
            "relation_features": RELATION_IN,
            "note": (
                "Shapes are fixed; WHICH features occupy these slots, and in what order, is NOT "
                "determined by the checkpoint and is not guessed here. A wrong contract loads "
                "cleanly and scores near chance. Pinning it is PKT-0029 STEP 1."
            ),
        },
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nHOCT_HEAD_RECONSTRUCTION strict_load_all={passed} "
          f"({sum(r['strict_load'] for r in rows)}/{len(rows)})")
    print(f"  undetermined: {NODE_EXTRAS} node extras, {PAIR_EXTRAS} pair extras, "
          f"{RELATION_IN} relation features - PKT-0029 STEP 1 decides these, not this module")
    if not passed:
        raise SystemExit("strict load FAILED - the reconstruction does not match the checkpoint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
