"""Generate scripts/kaggle_edits/d1_inject.py from the audit block.

The audit block must land in predict_unet_transformer.py's namespace, not the notebook's.
Embedding it as a literal inside a notebook cell is unsafe -- it contains triple-quoted
docstrings -- so it is base64'd here and written to a sibling module at kernel runtime.

Three injections, each with an asserted-unique anchor. The whole patch is idempotent.
Every payload goes through !r so no escape is ever hand-written into the template.
"""
import base64
import hashlib
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "scripts" / "kaggle_edits" / "d1_response_audit.py"
OUT = ROOT / "scripts" / "kaggle_edits" / "d1_inject.py"

raw = SRC.read_bytes()
b64 = base64.b64encode(raw).decode()
sha = hashlib.sha256(raw).hexdigest()
blob = "\n".join('    "%s"' % b64[i:i + 76] for i in range(0, len(b64), 76))

# NOTE: `def predict_video(` is preceded by an @torch.no_grad() DECORATOR. Anchoring on
# the bare def and inserting before it would split the decorator from its function and
# produce a SyntaxError. Caught by the local injection simulation, not on GPU.
A1 = "@torch.no_grad()\ndef predict_video("
A2 = (
    "        for f_idx, t in enumerate(frame_indices):\n"
    "            if t not in seen_frames:\n"
    "                arr = _detect_cells_pooled(\n"
    "                    det_logits[f_idx][0], t, cfg.det_threshold, pool_k,\n"
    "                )"
)
A3 = 'if __name__ == "__main__":'

IMPORTS = (
    "from _d1_audit_block import _d1_audit_frame, _d1_flush  # noqa: E402\n"
    "import os as _d1_os  # noqa: E402\n\n\n"
)

CALL = (
    "\n                _d1_audit_frame(\n"
    "                    ds_path.stem, ds_path.parent, t, det_logits[f_idx],\n"
    "                    unet_out[0, f_idx], cfg.det_threshold, pool_k,\n"
    "                    voxel_size, downsample,\n"
    "                )"
)

WRAP = (
    "_d1_orig_main = main\n\n\n"
    "def main():\n"
    "    try:\n"
    "        _d1_orig_main()\n"
    "    finally:\n"
    "        _d1_flush(\n"
    '            fold=_d1_os.environ.get("BIOHUB_D1_FOLD"),\n'
    '            ckpt_hash=_d1_os.environ.get("BIOHUB_D1_CKPT_SHA"),\n'
    '            expected_crops=int(_d1_os.environ.get("BIOHUB_D1_EXPECTED_CROPS", "0"))\n'
    "            or None,\n"
    "        )\n\n\n"
)

TEMPLATE = '''# --- D1 + D1-F INJECTION -------------------------------------------------------------
# Writes the audit block as a sibling module of predict_unet_transformer.py, then applies
# THREE injections, each with an asserted-unique anchor. Aborts on 0 or >1 matches.
# Idempotent: re-running is a no-op once `_d1_audit_frame` is present.
#
# Audit block sha256: {sha}
import base64 as _d1i_b64
import hashlib as _d1i_hashlib

_D1_BLOCK_B64 = (
{blob}
)
_d1_block_bytes = _d1i_b64.b64decode(_D1_BLOCK_B64)
_d1_block_sha = _d1i_hashlib.sha256(_d1_block_bytes).hexdigest()
if _d1_block_sha != {sha!r}:
    raise RuntimeError("D1 block hash drift: " + _d1_block_sha)

_d1_mod = _ps.parent / "_d1_audit_block.py"
_d1_mod.write_bytes(_d1_block_bytes)
print("D1: wrote", _d1_mod, _d1_block_sha[:16], flush=True)

_s = _ps.read_text()
if "_d1_audit_frame" in _s:
    print("D1: injection already present; no-op", flush=True)
else:
    _d1_anchors = {{"A1": {a1!r}, "A2": {a2!r}, "A3": {a3!r}}}
    for _nm, _a in _d1_anchors.items():
        _n = _s.count(_a)
        if _n != 1:
            raise RuntimeError(
                "D1 anchor " + _nm + " matched " + str(_n) + " times, expected exactly 1"
            )
    _s = _s.replace(_d1_anchors["A1"], {imports!r} + _d1_anchors["A1"], 1)
    _s = _s.replace(_d1_anchors["A2"], _d1_anchors["A2"] + {call!r}, 1)
    _s = _s.replace(_d1_anchors["A3"], {wrap!r} + _d1_anchors["A3"], 1)
    compile(_s, str(_ps), "exec")
    _ps.write_text(_s)
    print("D1: three injections applied and compiled", flush=True)
'''

OUT.write_text(
    TEMPLATE.format(sha=sha, blob=blob, a1=A1, a2=A2, a3=A3,
                    imports=IMPORTS, call=CALL, wrap=WRAP),
    encoding="utf-8",
)
print("wrote %s  (audit sha256 %s)" % (OUT, sha[:16]))
