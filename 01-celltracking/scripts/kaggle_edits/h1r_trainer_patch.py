r"""H1-R trainer patch -- exact-string fixes to the vendored training script.

Applies the retrain-recipe fixes (retrain_recipes_2026-08-17: A1/A3/A5/B4 + C2 hook) to a COPY
of ``train_unet_transformer.py`` (in a kernel: ``REPO_DIR/scripts/``; locally: a scratch copy --
never the vendored original in-place under ``vendor/``).

Every patch asserts an exact occurrence count, mirroring the kaggle_factory edit discipline:
a silently-skipped patch is an error, not a warning.

WHAT IT PATCHES (all locations verified against the vendored file 2026-08-18)
-----------------------------------------------------------------------------
P1  AMP (F5, trainer :64):  `F.binary_cross_entropy` raises under autocast, so autocast is
    applied INSIDE `TemporalUNet3D.forward` (monkeypatch appended to the module) with the
    output cast back to float32. This is deliberate: `nn.DataParallel` runs replica forwards
    in worker threads and `torch.autocast` is thread-local, so an autocast context around
    `model.encode(...)` would silently NOT autocast the 2-GPU path. Patching the UNet forward
    autocasts in every replica thread; everything downstream (detect head, transformer, both
    losses) stays fp32-safe. A `GradScaler` is added to `train_epoch`.
P2  Precision-aware checkpoint selection (F2, trainer :1180): `evaluate` additionally returns
    node_precision (matched detections / all detections); `train` selects on
    `test_acc * detection_F1` (env `H1R_SELECT=acc_f1`, default) or the legacy
    `test_acc * test_recall` (`H1R_SELECT=acc_recall`).
P3  Real division upweight (F6, trainer :68-70): `weight[div_rows] = _H1R_DIV_WEIGHT`
    (env `H1R_DIV_WEIGHT`, default 3.0; Trackastra uses 11).
P4  Sparse-annotation ignore mask (recipe C2) in `compute_detection_loss`: zero the negative
    weight in a ball of radius `H1R_IGNORE_RADIUS_VOX` voxels around confident non-GT peaks
    (likely unannotated true nuclei). Default 0 = off; enable ONLY for the competition
    fine-tune stage, never for dense Zebrahub labels.
P5  Cosine LR decay over epochs (recipe B4), env `H1R_COSINE` (default on).
P6  Always-save `edge_predictor_last.pth` each epoch for cross-session resume.

Usage (local test / kernel cell):
  .venv\Scripts\python.exe scripts\kaggle_edits\h1r_trainer_patch.py --trainer <path-to-copy>
  # or, in a notebook cell where this file's content was embedded:
  apply_h1r_trainer_patch(REPO_DIR / "scripts" / "train_unet_transformer.py")
"""
from __future__ import annotations

from pathlib import Path

_KNOBS = '''from itertools import cycle as _cycle

import os as _os

# --- H1-R patch knobs (scripts/kaggle_edits/h1r_trainer_patch.py) -------------------
_H1R_AMP = _os.environ.get("H1R_AMP", "1") == "1"
_H1R_DIV_WEIGHT = float(_os.environ.get("H1R_DIV_WEIGHT", "3.0"))
_H1R_SELECT = _os.environ.get("H1R_SELECT", "acc_f1")
_H1R_COSINE = _os.environ.get("H1R_COSINE", "1") == "1"
_H1R_IGNORE_RADIUS_VOX = int(_os.environ.get("H1R_IGNORE_RADIUS_VOX", "0"))


def _h1r_enable_amp_unet() -> None:
    """Autocast fp16 inside TemporalUNet3D.forward (thread-local => DataParallel-safe).

    Output is cast back to float32 so the detect head, transformer and both losses --
    including F.binary_cross_entropy at compute_loss, which raises under autocast --
    keep their fp32 semantics unchanged.
    """
    if getattr(TemporalUNet3D, "_h1r_amp", False):
        return
    _orig = TemporalUNet3D.forward

    def _fwd(self, x, _orig=_orig):
        if getattr(self, "_h1r_amp_enabled", _H1R_AMP) and torch.cuda.is_available():
            with torch.autocast("cuda", dtype=torch.float16):
                out = _orig(self, x)
            return out.float()
        return _orig(self, x)

    TemporalUNet3D.forward = _fwd
    TemporalUNet3D._h1r_amp = True
    TemporalUNet3D._h1r_amp_enabled = _H1R_AMP


_h1r_enable_amp_unet()
'''

PATCHES: list[tuple[str, str, int]] = [
    # --- knobs + P1 AMP-in-UNet (inserted right after the last vendored import) ------
    (
        "from itertools import cycle as _cycle\n",
        _KNOBS,
        1,
    ),
    # --- P3: real division upweight (F6 no-op fix; trainer :68-70) -------------------
    (
        "    div_rows = target.sum(dim=1) > 1\n"
        "    weight = torch.ones_like(loss)\n"
        "    weight[div_rows] = 1.0\n",
        "    div_rows = target.sum(dim=1) > 1\n"
        "    weight = torch.ones_like(loss)\n"
        "    weight[div_rows] = _H1R_DIV_WEIGHT\n",
        1,
    ),
    # --- P4: sparse-annotation ignore mask (recipe C2; off unless env set) -----------
    (
        "    weight = torch.where(target == 1.0, w_pos, w_neg)\n"
        "\n"
        "    return F.binary_cross_entropy_with_logits(\n",
        "    weight = torch.where(target == 1.0, w_pos, w_neg)\n"
        "\n"
        "    if _H1R_IGNORE_RADIUS_VOX > 0:\n"
        "        # Recipe C2: confident non-GT peaks are likely unannotated true nuclei\n"
        "        # (2.8% of instances labelled) -- ignore a ball around them instead of\n"
        "        # teaching them as background. Enable for the sparse competition\n"
        "        # fine-tune stage only.\n"
        "        with torch.no_grad():\n"
        "            _r = _H1R_IGNORE_RADIUS_VOX\n"
        "            _k = 2 * _r + 1\n"
        "            _lg = logits.unsqueeze(1)\n"
        "            _pooled = F.max_pool3d(_lg, _k, stride=1, padding=_r)\n"
        "            _peaks = (_lg == _pooled) & (_lg > 0.0)\n"
        "            _near_gt = F.max_pool3d(target.unsqueeze(1), _k, stride=1, padding=_r) > 0\n"
        "            _ignore = (_peaks & ~_near_gt).float()\n"
        "            _ball = F.max_pool3d(_ignore, _k, stride=1, padding=_r)[:, 0] > 0\n"
        "        weight = torch.where(_ball & (target != 1.0),\n"
        "                             torch.zeros_like(weight), weight)\n"
        "\n"
        "    return F.binary_cross_entropy_with_logits(\n",
        1,
    ),
    # --- P1: GradScaler in train_epoch ----------------------------------------------
    (
        "    model.train()\n"
        "    total_edge_loss = 0.0\n",
        "    model.train()\n"
        "    scaler = torch.amp.GradScaler(\"cuda\", enabled=_H1R_AMP and torch.cuda.is_available())\n"
        "    total_edge_loss = 0.0\n",
        1,
    ),
    (
        "        optimizer.zero_grad()\n"
        "        loss.backward()\n"
        "        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n"
        "        optimizer.step()\n",
        "        optimizer.zero_grad()\n"
        "        scaler.scale(loss).backward()\n"
        "        scaler.unscale_(optimizer)\n"
        "        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n"
        "        scaler.step(optimizer)\n"
        "        scaler.update()\n",
        1,
    ),
    # --- P2: node precision out of evaluate ------------------------------------------
    (
        "    Returns (avg_loss, accuracy, node_recall).\n",
        "    Returns (avg_loss, accuracy, node_recall, node_precision).\n",
        1,
    ),
    (
        "    total_loss, correct, total, n_pairs = 0.0, 0, 0, 0\n"
        "    gt_matched, gt_total = 0, 0\n",
        "    total_loss, correct, total, n_pairs = 0.0, 0, 0, 0\n"
        "    gt_matched, gt_total = 0, 0\n"
        "    det_matched, det_total = 0, 0\n",
        1,
    ),
    (
        "                gt_total += n_gt\n"
        "                gt_matched += n_matched\n",
        "                gt_total += n_gt\n"
        "                gt_matched += n_matched\n"
        "                det_total += int(matches[b].shape[0])\n"
        "                det_matched += int(n_matched)\n",
        1,
    ),
    (
        "    node_recall = gt_matched / max(gt_total, 1)\n"
        "    return total_loss / max(n_pairs, 1), correct / max(total, 1), node_recall\n",
        "    node_recall = gt_matched / max(gt_total, 1)\n"
        "    node_precision = det_matched / max(det_total, 1)\n"
        "    return (total_loss / max(n_pairs, 1), correct / max(total, 1),\n"
        "            node_recall, node_precision)\n",
        1,
    ),
    # --- P2: precision-aware checkpoint selection (F2 fix; trainer :1177-1180) -------
    (
        "        test_loss, test_acc, test_recall = evaluate(model, test_loader, device, pool_kernel_um=pool_kernel_um)\n"
        "        test_time = time.monotonic() - t0\n"
        "\n"
        "        score = test_acc * test_recall\n",
        "        test_loss, test_acc, test_recall, test_precision = evaluate(\n"
        "            model, test_loader, device, pool_kernel_um=pool_kernel_um)\n"
        "        test_time = time.monotonic() - t0\n"
        "\n"
        "        if _H1R_SELECT == \"acc_recall\":\n"
        "            score = test_acc * test_recall\n"
        "        else:\n"
        "            _det_f1 = (2.0 * test_precision * test_recall\n"
        "                       / max(test_precision + test_recall, 1e-9))\n"
        "            score = test_acc * _det_f1\n",
        1,
    ),
    (
        "            f\"test_loss={test_loss:.4f} | acc={test_acc:.4f} | recall={test_recall:.4f} | best={best_score:.4f} {marker} | \"\n",
        "            f\"test_loss={test_loss:.4f} | acc={test_acc:.4f} | recall={test_recall:.4f} | \"\n"
        "            f\"prec={test_precision:.4f} | best={best_score:.4f} {marker} | \"\n",
        1,
    ),
    # --- P5: cosine LR over epochs (recipe B4) ---------------------------------------
    (
        "    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)\n"
        "    print(f\"Starting training for {n_epochs} epochs (batch_size={batch_size})...\", flush=True)\n",
        "    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)\n"
        "    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(n_epochs, 1))\n"
        "                 if _H1R_COSINE else None)\n"
        "    print(f\"Starting training for {n_epochs} epochs (batch_size={batch_size})...\", flush=True)\n",
        1,
    ),
    # --- P6: always-save last checkpoint (session-chain resume) + LR step ------------
    (
        "        marker = \"*\" if is_best else \" \"\n",
        "        torch.save(\n"
        "            {k.replace(\"unet.module.\", \"unet.\", 1): v for k, v in model.state_dict().items()},\n"
        "            output_dir / \"edge_predictor_last.pth\",\n"
        "        )\n"
        "        if scheduler is not None:\n"
        "            scheduler.step()\n"
        "        marker = \"*\" if is_best else \" \"\n",
        1,
    ),
]


def apply_h1r_trainer_patch(trainer_path: Path | str) -> None:
    """Apply all patches to *trainer_path* in place; assert exact counts; compile-check."""
    trainer_path = Path(trainer_path)
    src = trainer_path.read_text(encoding="utf-8")
    if "_H1R_AMP" in src:
        print(f"h1r_trainer_patch: {trainer_path} already patched -- skipping")
        return
    for i, (old, new, expect) in enumerate(PATCHES):
        n = src.count(old)
        assert n == expect, (
            f"h1r_trainer_patch: patch {i} matched {n} times (expected {expect}) "
            f"in {trainer_path}"
        )
        src = src.replace(old, new, expect)
    compile(src, str(trainer_path), "exec")
    trainer_path.write_text(src, encoding="utf-8")
    print(f"h1r_trainer_patch: {len(PATCHES)} patches applied to {trainer_path}")


# NOTE: notebook cells execute with __name__ == "__main__", so the CLI below is guarded
# by the H1R_KERNEL env var (set in the kernel's env cell) to keep cell-embedding inert.
if __name__ == "__main__":
    import os as _os

    if _os.environ.get("H1R_KERNEL") != "1":
        import argparse

        _ap = argparse.ArgumentParser()
        _ap.add_argument("--trainer", required=True,
                         help="path to a COPY of train_unet_transformer.py (patched in place)")
        _a = _ap.parse_args()
        apply_h1r_trainer_patch(_a.trainer)
