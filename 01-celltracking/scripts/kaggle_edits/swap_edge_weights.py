# =====================================================================================
# POINT THE PREDICTOR AT A CLEAN RETRAINED EDGE CHECKPOINT
#
# WHY. The deployed edge predictor's own split_manifest.json reads
# `method: unet_transformer_ALLTRAIN_seed314159_v1`, `train: 199` (every labelled crop) with a
# `test` list that is a strict SUBSET of train -- it memorised every movie we can score locally.
# `leevvin/biohub-movie-heldout-edge-predictor-v1` (CC0) is a 195-movie retrain holding out
# exactly the four public twins. Host-verified locally: a bare state_dict of 136 tensors that
# loads into our exact `UNetNodeTransformer` with strict=True.
#
# WHY IT SHOULD TRANSFER. The edge head's column softmax > 0.5 defines every candidate edge, so
# replacing it is a CANDIDATE-SET change -- the one lever class with demonstrated LB transfer
# (2026-08-19: detection-surface local -0.0091 -> LB -0.0320, 3.5x amplified; division 0.000;
# edge-permutation 0.000).
#
# TWO FAILURES THIS EDIT IS WRITTEN AROUND (both hit on real kernel runs, v1 and v2):
#   1. `/kaggle/input/<slug>/edge_predictor_best.pth` did NOT exist -- the mount layout is not
#      reliably the dataset slug, and an rglob for that exact filename returned nothing. So this
#      version searches broadly for *.pth, disambiguates by exact byte size, and DUMPS the input
#      tree first so a miss is diagnosable from the log without another GPU run.
#   2. `REPO_DIR/weights/.../edge_predictor_best.pth` is on a READ-ONLY filesystem, so overwriting
#      it raises OSError(30). So this version does not copy anything: it rebinds the `--weights`
#      argument already sitting in `predict_cmd`. That also survives `predict_cmd` having been
#      built earlier in the same cell, which rebinding WEIGHTS_RELATIVE would not.
# =====================================================================================
from pathlib import Path as _SwPath

_SW_EXPECT_BYTES = 8355927          # host-verified size of the leevvin checkpoint
_sw_root = _SwPath("/kaggle/input")

print("edge-weight swap: /kaggle/input tree (depth 2):", flush=True)
for _d in sorted(_sw_root.iterdir()) if _sw_root.exists() else []:
    print(f"    {_d.name}/", flush=True)
    try:
        for _c in sorted(_d.iterdir())[:12]:
            _sz = f" ({_c.stat().st_size:,} B)" if _c.is_file() else "/"
            print(f"        {_c.name}{_sz}", flush=True)
    except Exception as _e:
        print(f"        <unreadable: {_e}>", flush=True)

_sw_pth = sorted(_sw_root.rglob("*.pth")) if _sw_root.exists() else []
print(f"edge-weight swap: {len(_sw_pth)} .pth files under /kaggle/input", flush=True)
for _p in _sw_pth:
    print(f"    {_p}  ({_p.stat().st_size:,} B)", flush=True)

_sw_hits = [_p for _p in _sw_pth if _p.stat().st_size == _SW_EXPECT_BYTES]
if not _sw_hits:
    raise RuntimeError(
        f"edge-weight swap: no .pth of exactly {_SW_EXPECT_BYTES:,} bytes under /kaggle/input. "
        f"Saw: {[(str(_p), _p.stat().st_size) for _p in _sw_pth]}")
_sw_src = _sw_hits[0]

# Rebind the --weights argument in the already-constructed predict command.
if "--weights" not in predict_cmd:
    raise RuntimeError(f"edge-weight swap: '--weights' not in predict_cmd: {predict_cmd}")
_sw_i = predict_cmd.index("--weights") + 1
_sw_old = predict_cmd[_sw_i]
predict_cmd[_sw_i] = str(_sw_src)
print(f"EDGE WEIGHTS REBOUND:\n    was: {_sw_old}\n    now: {predict_cmd[_sw_i]}", flush=True)
print("  source: leevvin/biohub-movie-heldout-edge-predictor-v1 "
      "(195-movie held-out retrain, CC0, strict=True 136/136)", flush=True)
