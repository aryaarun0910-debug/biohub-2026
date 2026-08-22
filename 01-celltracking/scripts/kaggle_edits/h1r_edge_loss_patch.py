r"""H1-R EDGE-LOSS patch -- structural fixes to the association objective.

Layers ON TOP of ``scripts/kaggle_edits/h1r_trainer_patch.py`` (apply that first). Same
discipline: exact-string replacements against a COPY of the vendored
``train_unet_transformer.py``, every patch asserting an exact occurrence count, then a
``compile()`` check. Never edits the vendored original under ``vendor/``.

WHY (see research/06-knowledge-system/internal-reports/edge_loss_structure_2026-08-18.md)
-----------------------------------------------------------------------------------------
The vendored ``compute_loss`` (trainer :55-72) normalises the edge logits with a bare
column softmax ``softmax(logits, dim=0)``. Every column therefore sums to EXACTLY 1: the
objective asserts that every node at t+1 has a parent among the detections at t. There is
no "this node has no parent" outlet, so (a) track births / unmatched FP detections are
supervised toward an arbitrary unannotated source, and (b) the emitted probability is a
pure *share* whose scale depends on the number of candidate sources in the frame -- which
differs by ~10x between the training crops and the deployment frames, and which the
deployed linker consumes as an absolute quantity
(``src/biotrack/wrapper.py:340``: ``cost = motion + 0.05*raw - 0.75*prob``).

WHAT IT PATCHES (locations verified against the h1r-patched trainer, 2026-08-18)
-----------------------------------------------------------------------------------------
E0  Knobs, inserted after the h1r_trainer_patch knob block.
E1  **Background / no-parent term** (trainer :63). Replaces
    ``softmax(logits, dim=0)`` with Trackastra's parental softmax
    ``p_ij = exp(a_ij) / (1 + sum_i' exp(a_i'j))`` (arXiv:2405.15700 sec. 3.2), implemented
    by concatenating a constant zero logit row before the softmax and stripping it after.
    Env ``H1R_BG_TERM`` (default 1). **Inference must match** -- see
    ``apply_h1r_edge_loss_inference_patch`` below.
E2  **Auxiliary sigmoid BCE** at lambda = 1e-2 (Trackastra ``L = L_BCE(A, Phi(A^), W) +
    lambda * L_BCE(A, sigma(A^), W)``). Gives every edge an ABSOLUTE score that does not
    depend on the column normalisation. Env ``H1R_AUX_SIGMOID`` (default 0.01).
E3  **Real per-row weights.** ``H1R_DIV_WEIGHT`` (already introduced by
    h1r_trainer_patch P3, default 3.0) is kept as the dividing-row multiplier; a new
    ``H1R_CONT_WEIGHT`` (default 1.0 == today's behaviour) multiplies non-dividing
    *annotated* rows so the two can be set independently, and ``H1R_DIV_WEIGHT=0`` cleanly
    MASKS division rows for the Ultrack-derived Zebrahub stage.
E4  **Focal exponent knob** ``H1R_FOCAL_GAMMA`` (default 2.0 == today). gamma=0 disables the
    focal modulation. Also clamps the softmax probabilities before
    ``F.binary_cross_entropy`` (which returns inf at exactly 0/1).
E5  **Association-aware checkpoint selection.** ``_evaluate_pair`` additionally reports
    top-1 parent accuracy over supervised columns (argmax_i p[:, j] lands on the true
    parent). ``H1R_SELECT=link_f1`` selects on ``link_top1 * detection_F1`` instead of
    ``test_acc * ...``; ``test_acc`` thresholds a column softmax at 0.5 and is near-constant
    at deployment node counts, so it carries almost no association signal.

Usage
-----
  .venv\Scripts\python.exe scripts\kaggle_edits\h1r_trainer_patch.py   --trainer <copy>
  .venv\Scripts\python.exe scripts\kaggle_edits\h1r_edge_loss_patch.py --trainer <copy>
  # semantics check, no files touched:
  .venv\Scripts\python.exe scripts\kaggle_edits\h1r_edge_loss_patch.py --self-test
  # inference-side companion (a COPY of predict_unet_transformer.py):
  .venv\Scripts\python.exe scripts\kaggle_edits\h1r_edge_loss_patch.py --predict <copy>
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------------------
# E0 -- knobs
# --------------------------------------------------------------------------------------
_KNOBS_ANCHOR = '_H1R_IGNORE_RADIUS_VOX = int(_os.environ.get("H1R_IGNORE_RADIUS_VOX", "0"))\n'

_KNOBS = _KNOBS_ANCHOR + '''
# --- H1-R edge-loss knobs (scripts/kaggle_edits/h1r_edge_loss_patch.py) -------------
# Background ("no parent") term in the parental softmax: p_ij = e^a_ij / (1 + sum_i' e^a_i'j).
_H1R_BG_TERM = _os.environ.get("H1R_BG_TERM", "1") == "1"
# Auxiliary plain-sigmoid BCE weight (Trackastra lambda = 1e-2). 0 disables.
_H1R_AUX_SIGMOID = float(_os.environ.get("H1R_AUX_SIGMOID", "0.01"))
# Focal modulation exponent on (1 - p_t). 0 disables the focal term.
_H1R_FOCAL_GAMMA = float(_os.environ.get("H1R_FOCAL_GAMMA", "2.0"))
# Row weight for annotated NON-dividing (continuation) sources. 1.0 == vendored behaviour.
_H1R_CONT_WEIGHT = float(_os.environ.get("H1R_CONT_WEIGHT", "1.0"))
# Withhold division supervision by dropping the DAUGHTER COLUMNS from the mask. A zero row
# weight is NOT enough: under a column softmax the dividing row's logits still normalise
# every column it appears in. Only column removal actually zeroes the gradient.
_H1R_DIV_MASK_COLS = _os.environ.get("H1R_DIV_MASK_COLS", "0") == "1"
'''

# --------------------------------------------------------------------------------------
# E1-E4 -- the loss body (matches the text AFTER h1r_trainer_patch P3)
# --------------------------------------------------------------------------------------
_LOSS_OLD = '''    probs = torch.softmax(logits, dim=0)  # dim=0 intentional: divisions allowed, merges aren't
    bce = F.binary_cross_entropy(probs, target, reduction="none")
    p_t = probs * target + (1 - probs) * (1 - target)
    loss = ((1 - p_t) ** 2) * bce

    div_rows = target.sum(dim=1) > 1
    weight = torch.ones_like(loss)
    weight[div_rows] = _H1R_DIV_WEIGHT

    return (loss * weight)[mask].mean()
'''

_LOSS_NEW = '''    # E1: parental softmax over candidate parents. dim=0 is intentional (divisions
    # allowed, merges aren't). With _H1R_BG_TERM a constant zero logit is appended as an
    # extra "no parent" row, giving Trackastra's
    #     p_ij = exp(a_ij) / (1 + sum_i' exp(a_i'j))
    # so a column is free to sum to LESS than 1 -- the explicit track-birth outlet the
    # bare softmax does not have, and the calibration anchor (logit 0 == "as likely as no
    # parent") that makes p comparable across frames of very different node counts.
    if _H1R_BG_TERM:
        _bg = torch.zeros(1, logits.shape[1], device=logits.device, dtype=logits.dtype)
        probs = torch.softmax(torch.cat([logits, _bg], dim=0), dim=0)[:-1]
    else:
        probs = torch.softmax(logits, dim=0)

    # E4: F.binary_cross_entropy is inf at exactly 0/1; the clamp is a no-op elsewhere.
    probs = probs.clamp(1e-7, 1.0 - 1e-7)
    bce = F.binary_cross_entropy(probs, target, reduction="none")
    p_t = probs * target + (1 - probs) * (1 - target)
    if _H1R_FOCAL_GAMMA > 0.0:
        loss = ((1 - p_t) ** _H1R_FOCAL_GAMMA) * bce
    else:
        loss = bce

    # E2: auxiliary plain-sigmoid BCE (Trackastra lambda = 1e-2). Unlike the softmax term
    # this is an ABSOLUTE per-edge score, independent of how many candidates share the
    # column -- which is what the deployed linker's learned-prob bonus actually wants.
    if _H1R_AUX_SIGMOID > 0.0:
        loss = loss + _H1R_AUX_SIGMOID * F.binary_cross_entropy_with_logits(
            logits, target, reduction="none",
        )

    # E3: real per-row weights. active_rows are the annotated sources; dividing rows take
    # _H1R_DIV_WEIGHT (0 masks them entirely, for Ultrack-derived division labels).
    div_rows = target.sum(dim=1) > 1
    weight = torch.ones_like(loss)
    weight[active_rows] = _H1R_CONT_WEIGHT
    weight[div_rows] = _H1R_DIV_WEIGHT

    if _H1R_DIV_MASK_COLS and bool(div_rows.any()):
        # TRAP: weight[div_rows] = 0 does NOT withhold division supervision. The softmax
        # is per COLUMN, so a dividing row's logits keep normalising both daughter columns
        # and the gradient there stays non-zero. Removing the daughter COLUMNS from the
        # mask is what actually zeroes it (a column's logits affect only that column).
        div_cols = target[div_rows].sum(dim=0) > 0
        mask = mask & ~div_cols.unsqueeze(0)
        if not mask.any():
            return logits.sum() * 0.0

    return (loss * weight)[mask].mean()
'''

# --------------------------------------------------------------------------------------
# E5 -- association-aware evaluation + checkpoint selection
# --------------------------------------------------------------------------------------
_EVAL_PAIR_OLD = '''    """Per-pair evaluation. Returns (loss, correct, total)."""
    active_rows = target.sum(dim=1) > 0
    active_cols = target.sum(dim=0) > 0
    if not active_rows.any():
        return 0.0, 0, 0

    loss = compute_loss(logits, target).item()
    probs = torch.softmax(logits, dim=0)
    preds = (probs > 0.5).float()

    mask = active_rows.unsqueeze(1) | active_cols.unsqueeze(0)
    correct = (preds[mask] == target[mask]).sum().item()
    total = mask.sum().item()

    return loss, correct, total
'''

_EVAL_PAIR_NEW = '''    """Per-pair evaluation. Returns (loss, correct, total, link_correct, link_total).

    ``correct/total`` is the vendored cell-wise accuracy at a 0.5 threshold on the column
    softmax. It is near-degenerate at deployment node counts (a column of N candidates
    rarely puts >0.5 anywhere), so E5 additionally reports TOP-1 PARENT accuracy over the
    supervised columns -- argmax_i p[:, j] landing on a true parent -- which is the
    quantity ``adj_edge_jaccard`` is actually a function of.
    """
    active_rows = target.sum(dim=1) > 0
    active_cols = target.sum(dim=0) > 0
    if not active_rows.any():
        return 0.0, 0, 0, 0, 0

    loss = compute_loss(logits, target).item()
    if _H1R_BG_TERM:
        _bg = torch.zeros(1, logits.shape[1], device=logits.device, dtype=logits.dtype)
        probs = torch.softmax(torch.cat([logits, _bg], dim=0), dim=0)[:-1]
    else:
        probs = torch.softmax(logits, dim=0)
    preds = (probs > 0.5).float()

    mask = active_rows.unsqueeze(1) | active_cols.unsqueeze(0)
    correct = (preds[mask] == target[mask]).sum().item()
    total = mask.sum().item()

    link_total = int(active_cols.sum().item())
    if link_total > 0:
        top1 = probs[:, active_cols].argmax(dim=0)
        link_correct = int(
            target[:, active_cols].gather(0, top1.unsqueeze(0)).sum().item()
        )
    else:
        link_correct = 0

    return loss, correct, total, link_correct, link_total
'''

PATCHES: list[tuple[str, str, int]] = [
    # --- E0: knobs -------------------------------------------------------------------
    (_KNOBS_ANCHOR, _KNOBS, 1),
    # --- E1-E4: the loss body --------------------------------------------------------
    (_LOSS_OLD, _LOSS_NEW, 1),
    # --- E5a: _evaluate_pair reports link top-1 --------------------------------------
    (_EVAL_PAIR_OLD, _EVAL_PAIR_NEW, 1),
    # --- E5b: evaluate() accumulators ------------------------------------------------
    (
        "    total_loss, correct, total, n_pairs = 0.0, 0, 0, 0\n"
        "    gt_matched, gt_total = 0, 0\n"
        "    det_matched, det_total = 0, 0\n",
        "    total_loss, correct, total, n_pairs = 0.0, 0, 0, 0\n"
        "    gt_matched, gt_total = 0, 0\n"
        "    det_matched, det_total = 0, 0\n"
        "    link_correct, link_total = 0, 0\n",
        1,
    ),
    (
        "                pair_loss, pair_correct, pair_total = _evaluate_pair(\n"
        "                    pair_logits[b, :ns_b, :nt_b], pair_target[b, :ns_b, :nt_b],\n"
        "                )\n"
        "                total_loss += pair_loss\n"
        "                correct += pair_correct\n"
        "                total += pair_total\n",
        "                (pair_loss, pair_correct, pair_total,\n"
        "                 pair_link_correct, pair_link_total) = _evaluate_pair(\n"
        "                    pair_logits[b, :ns_b, :nt_b], pair_target[b, :ns_b, :nt_b],\n"
        "                )\n"
        "                total_loss += pair_loss\n"
        "                correct += pair_correct\n"
        "                total += pair_total\n"
        "                link_correct += pair_link_correct\n"
        "                link_total += pair_link_total\n",
        1,
    ),
    (
        "    Returns (avg_loss, accuracy, node_recall, node_precision).\n",
        "    Returns (avg_loss, accuracy, node_recall, node_precision, link_top1).\n",
        1,
    ),
    (
        "    node_recall = gt_matched / max(gt_total, 1)\n"
        "    node_precision = det_matched / max(det_total, 1)\n"
        "    return (total_loss / max(n_pairs, 1), correct / max(total, 1),\n"
        "            node_recall, node_precision)\n",
        "    node_recall = gt_matched / max(gt_total, 1)\n"
        "    node_precision = det_matched / max(det_total, 1)\n"
        "    link_top1 = link_correct / max(link_total, 1)\n"
        "    return (total_loss / max(n_pairs, 1), correct / max(total, 1),\n"
        "            node_recall, node_precision, link_top1)\n",
        1,
    ),
    # --- E5c: selection ---------------------------------------------------------------
    (
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
        "        test_loss, test_acc, test_recall, test_precision, test_link = evaluate(\n"
        "            model, test_loader, device, pool_kernel_um=pool_kernel_um)\n"
        "        test_time = time.monotonic() - t0\n"
        "\n"
        "        _det_f1 = (2.0 * test_precision * test_recall\n"
        "                   / max(test_precision + test_recall, 1e-9))\n"
        "        if _H1R_SELECT == \"acc_recall\":\n"
        "            score = test_acc * test_recall\n"
        "        elif _H1R_SELECT == \"link_f1\":\n"
        "            # E5: association-aware. test_acc thresholds a column softmax at 0.5\n"
        "            # and is near-constant at deployment node counts.\n"
        "            score = test_link * _det_f1\n"
        "        else:\n"
        "            score = test_acc * _det_f1\n",
        1,
    ),
    (
        "            f\"prec={test_precision:.4f} | best={best_score:.4f} {marker} | \"\n",
        "            f\"prec={test_precision:.4f} | link1={test_link:.4f} | \"\n"
        "            f\"best={best_score:.4f} {marker} | \"\n",
        1,
    ),
]


# --------------------------------------------------------------------------------------
# Inference companion: the normalisation used at predict time MUST match training.
# --------------------------------------------------------------------------------------
_PREDICT_OLD = '''            raw = edge_logits_pair[0]
            if cfg.edge_activation == "softmax":
                probs = torch.softmax(raw, dim=0).cpu().numpy()
            else:
                probs = torch.sigmoid(raw).cpu().numpy()
'''

_PREDICT_NEW = '''            raw = edge_logits_pair[0]
            # H1-R edge-loss patch: the exported edge_prob must use the SAME
            # normalisation the head was trained under. H1R_EDGE_PROB selects:
            #   softmax     -- vendored bare column softmax (sums to 1 per column)
            #   softmax_bg  -- parental softmax with the "no parent" background term
            #   sigmoid     -- the auxiliary absolute per-edge score (needs H1R_AUX_SIGMOID>0)
            _h1r_edge_prob = os.environ.get("H1R_EDGE_PROB", "softmax")
            if _h1r_edge_prob == "softmax_bg":
                _bg = torch.zeros(1, raw.shape[1], device=raw.device, dtype=raw.dtype)
                probs = torch.softmax(torch.cat([raw, _bg], dim=0), dim=0)[:-1].cpu().numpy()
            elif _h1r_edge_prob == "sigmoid":
                probs = torch.sigmoid(raw).cpu().numpy()
            elif cfg.edge_activation == "softmax":
                probs = torch.softmax(raw, dim=0).cpu().numpy()
            else:
                probs = torch.sigmoid(raw).cpu().numpy()
'''


def apply_h1r_edge_loss_patch(trainer_path: Path | str) -> None:
    """Apply E0-E5 to *trainer_path* in place; assert exact counts; compile-check.

    ``scripts/kaggle_edits/h1r_trainer_patch.py`` MUST have been applied first -- the
    anchors below are its post-patch text.
    """
    trainer_path = Path(trainer_path)
    src = trainer_path.read_text(encoding="utf-8")
    if "_H1R_AMP" not in src:
        raise RuntimeError(
            f"h1r_edge_loss_patch: {trainer_path} has not been through "
            "h1r_trainer_patch.py -- apply that first."
        )
    if "_H1R_BG_TERM" in src:
        print(f"h1r_edge_loss_patch: {trainer_path} already patched -- skipping")
        return
    for i, (old, new, expect) in enumerate(PATCHES):
        n = src.count(old)
        assert n == expect, (
            f"h1r_edge_loss_patch: patch {i} matched {n} times (expected {expect}) "
            f"in {trainer_path}"
        )
        src = src.replace(old, new, expect)
    compile(src, str(trainer_path), "exec")
    trainer_path.write_text(src, encoding="utf-8")
    print(f"h1r_edge_loss_patch: {len(PATCHES)} patches applied to {trainer_path}")


def apply_h1r_edge_loss_inference_patch(predict_path: Path | str) -> None:
    """Match the training-time normalisation in a COPY of predict_unet_transformer.py."""
    predict_path = Path(predict_path)
    src = predict_path.read_text(encoding="utf-8")
    if "H1R_EDGE_PROB" in src:
        print(f"h1r_edge_loss_patch: {predict_path} already patched -- skipping")
        return
    n = src.count(_PREDICT_OLD)
    assert n == 1, f"h1r_edge_loss_patch: predict patch matched {n} times (expected 1)"
    src = src.replace(_PREDICT_OLD, _PREDICT_NEW, 1)
    compile(src, str(predict_path), "exec")
    predict_path.write_text(src, encoding="utf-8")
    print(f"h1r_edge_loss_patch: inference patch applied to {predict_path}")


# --------------------------------------------------------------------------------------
# Falsification tests -- semantics, not plumbing. Each maps to a claim in
# research/06-knowledge-system/internal-reports/edge_loss_structure_2026-08-18.md.
# --------------------------------------------------------------------------------------
def self_test() -> None:
    """Numeric checks of every claimed patch semantic. No files touched."""
    import torch
    import torch.nn.functional as F

    ok = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        ok.append(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")

    def probs_bare(a):
        return torch.softmax(a, dim=0)

    def probs_bg(a):
        bg = torch.zeros(1, a.shape[1], dtype=a.dtype)
        return torch.softmax(torch.cat([a, bg], dim=0), dim=0)[:-1]

    torch.manual_seed(0)

    # T1 -- the bare softmax forces every column to sum to EXACTLY 1: a node at t+1 with
    #       no true parent (track birth, or an unmatched FP detection) cannot be expressed.
    #       The background term gives that column an outlet.
    born = torch.full((64, 4), -12.0)  # model is confident nothing here is a parent
    check("T1a bare column softmax sums to 1 no matter how negative the logits",
          bool(torch.allclose(probs_bare(born).sum(0), torch.ones(4), atol=1e-5)),
          f"col sum {float(probs_bare(born).sum(0)[0]):.6f}")
    check("T1b background term lets a no-parent column collapse to ~0",
          bool((probs_bg(born).sum(0) < 1e-3).all()),
          f"col sum {float(probs_bg(born).sum(0)[0]):.3e}")

    # T2 -- the bare softmax is SHIFT-INVARIANT down a column, so the objective gives the
    #       model no gradient on the absolute logit level; the level it converges to is
    #       uncontrolled, and the emitted probability is a pure share of however many
    #       candidates the frame happens to hold. The background term pins the level.
    tgt2 = torch.zeros(32, 4)
    tgt2[0, :] = 1.0
    base = torch.randn(32, 4)

    def col_loss(a, fn):
        p = fn(a).clamp(1e-7, 1 - 1e-7)
        return float(F.binary_cross_entropy(p, tgt2, reduction="mean"))

    # Shift DOWN: that is the direction the background term can see (once every logit is
    # far above 0 the constant background is negligible and the two agree again).
    d_bare = max(
        abs(col_loss(base, probs_bare) - col_loss(base + s, probs_bare))
        for s in (5.0, -6.0)
    )
    d_bg = max(
        abs(col_loss(base, probs_bg) - col_loss(base + s, probs_bg))
        for s in (5.0, -6.0)
    )
    check("T2a bare softmax loss is invariant to a logit shift (no level signal at all)",
          d_bare < 1e-5, f"max delta {d_bare:.3e}")
    check("T2b background term makes the loss sensitive to the logit level",
          d_bg > 1e-2, f"max delta {d_bg:.4f}")

    # T2c -- consequence: candidate-count invariance of the top-1 probability becomes
    #        LEARNABLE (not automatic). With non-parents pushed far down it holds; the
    #        background term is what makes "far down" mean anything.
    def top_prob(n, fn, off):
        z = torch.full((n, 1), off)
        z[0, 0] = 3.0
        return float(fn(z)[0, 0])

    check("T2c background term + separated logits => count-invariant top-1 prob",
          abs(top_prob(10, probs_bg, -14.0) - top_prob(2000, probs_bg, -14.0)) < 1e-3,
          f"{top_prob(10, probs_bg, -14.0):.4f} vs {top_prob(2000, probs_bg, -14.0):.4f}")

    # T3 -- division upweight is REAL (the vendored weight[div_rows] = 1.0 is a no-op).
    tgt = torch.zeros(4, 4)
    tgt[0, 0] = tgt[0, 1] = 1.0   # dividing row
    tgt[1, 2] = 1.0               # continuation row
    lg = torch.randn(4, 4, requires_grad=True)

    def loss_with(div_w, cont_w=1.0, gamma=2.0, aux=0.0, bg=True, mask_div_cols=False):
        ar = tgt.sum(1) > 0
        ac = tgt.sum(0) > 0
        m = ar.unsqueeze(1) | ac.unsqueeze(0)
        p = (probs_bg(lg) if bg else probs_bare(lg)).clamp(1e-7, 1 - 1e-7)
        bce = F.binary_cross_entropy(p, tgt, reduction="none")
        pt = p * tgt + (1 - p) * (1 - tgt)
        L = ((1 - pt) ** gamma) * bce if gamma > 0 else bce
        if aux > 0:
            L = L + aux * F.binary_cross_entropy_with_logits(lg, tgt, reduction="none")
        dr = tgt.sum(1) > 1
        w = torch.ones_like(L)
        w[ar] = cont_w
        w[dr] = div_w
        if mask_div_cols and bool(dr.any()):
            m = m & ~(tgt[dr].sum(0) > 0).unsqueeze(0)
        return (L * w)[m].mean()

    g1 = torch.autograd.grad(loss_with(1.0), lg, retain_graph=True)[0][0].abs().sum()
    g10 = torch.autograd.grad(loss_with(10.0), lg, retain_graph=True)[0][0].abs().sum()
    check("T3a division row gradient scales with H1R_DIV_WEIGHT",
          float(g10) > 9.0 * float(g1), f"|g|(w=1)={float(g1):.5f} |g|(w=10)={float(g10):.5f}")

    # T3b -- THE TRAP. Zeroing the row weight does NOT withhold division supervision:
    # the dividing row still normalises the daughter columns, so the gradient there is
    # non-zero. This falsifies the "set loss weight 0 on dividing rows" masking recipe.
    dcols = (tgt[tgt.sum(1) > 1].sum(0) > 0)
    g0 = torch.autograd.grad(loss_with(0.0), lg, retain_graph=True)[0][:, dcols].abs().sum()
    check("T3b row-weight 0 leaves NON-zero gradient on the daughter columns (the trap)",
          float(g0) > 1e-6, f"|g| on daughter cols = {float(g0):.3e}")

    # T3c -- removing the daughter COLUMNS from the mask is what actually zeroes it.
    g0c = torch.autograd.grad(
        loss_with(0.0, mask_div_cols=True), lg, retain_graph=True,
    )[0][:, dcols].abs().sum()
    check("T3c H1R_DIV_MASK_COLS=1 zeroes the daughter-column gradient exactly",
          float(g0c) < 1e-12, f"|g| on daughter cols = {float(g0c):.3e}")

    # T4 -- the auxiliary sigmoid term is scale-free: it is unchanged when the number of
    #       candidate sources changes, while the softmax term is not.
    def aux_only(n):
        z = torch.full((n, 1), -2.0)
        z[0, 0] = 3.0
        t = torch.zeros(n, 1)
        t[0, 0] = 1.0
        return float(F.binary_cross_entropy_with_logits(z, t, reduction="none")[0, 0])

    check("T4 auxiliary sigmoid term is candidate-count invariant",
          abs(aux_only(10) - aux_only(2000)) < 1e-6)

    # T5 -- focal gamma=2 makes the negative population's contribution vanish (~p^3),
    #       i.e. the vendored loss is almost entirely positive-driven.
    p = torch.tensor(0.01)
    neg_focal = float((p ** 2) * (-(1 - p).log()))
    neg_plain = float(-(1 - p).log())
    check("T5 focal gamma=2 suppresses small-p negatives ~100x",
          neg_plain / max(neg_focal, 1e-30) > 50.0,
          f"plain {neg_plain:.3e} vs focal {neg_focal:.3e}")

    # T6 -- top-1 parent accuracy is informative where a 0.5 threshold on a column
    #       softmax is not (the E5 selection claim).
    n = 400
    z = torch.full((n, 6), -1.0)
    for j in range(6):
        z[j, j] = 4.0
    t6 = torch.zeros(n, 6)
    for j in range(6):
        t6[j, j] = 1.0
    pb = probs_bare(z)
    check("T6a 0.5-threshold accuracy is degenerate at n=400 candidates",
          int((pb > 0.5).sum()) == 0, f"cells over 0.5: {int((pb > 0.5).sum())}")
    top1 = pb.argmax(0)
    acc = float(t6.gather(0, top1.unsqueeze(0)).mean())
    check("T6b top-1 parent accuracy resolves the same logits", acc == 1.0, f"top1 acc {acc}")

    print(f"\n{sum(ok)}/{len(ok)} checks passed")
    if not all(ok):
        raise SystemExit(1)


if __name__ == "__main__":
    import os as _os

    if _os.environ.get("H1R_KERNEL") != "1":
        import argparse

        _ap = argparse.ArgumentParser()
        _ap.add_argument("--trainer", help="path to a COPY of train_unet_transformer.py")
        _ap.add_argument("--predict", help="path to a COPY of predict_unet_transformer.py")
        _ap.add_argument("--self-test", action="store_true",
                         help="run the numeric semantics checks and exit")
        _a = _ap.parse_args()
        if _a.self_test:
            self_test()
        if _a.trainer:
            apply_h1r_edge_loss_patch(_a.trainer)
        if _a.predict:
            apply_h1r_edge_loss_inference_patch(_a.predict)
        if not (_a.self_test or _a.trainer or _a.predict):
            _ap.error("nothing to do: pass --self-test, --trainer and/or --predict")
