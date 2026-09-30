"""Read a torch checkpoint's KEY SET and tensor shapes WITHOUT materialising its tensors.

WHY THIS EXISTS RATHER THAN `torch.load`
----------------------------------------
Two reasons, and the first is the scientific one.

FACT-0425: FOCUS-3D's own loader calls ``load_state_dict(..., strict=False)`` and the two prints
that would report the missing and unexpected key counts are COMMENTED OUT in the released source.
A checkpoint whose keys do not match therefore loads SILENTLY and a 411.6M-parameter, 300-query
segmenter runs on RANDOM WEIGHTS. That looks exactly like "the foreign model does not transfer".
So the key set has to be established BEFORE anything is loaded through the publisher's path, and
it has to be established by something that cannot itself be fooled by a silent fallback.

The second reason is operational. The nuclei checkpoint is 4.47 GB and this box has 16 GB with a
CPU experiment already running; ``torch.load`` would materialise every storage and page the
machine. This reads the zip's own directory and the pickle's opcode stream, so peak memory is a
few megabytes regardless of checkpoint size.

WHAT IT DOES NOT DO
-------------------
It never unpickles a real object and never executes the archived graph. Every class the pickle
names is resolved to an INERT STUB that records the name and does nothing - the method PKT-0047
used to take HOCT's parameter count without calling ``torch.jit.load``. Consequently it cannot
be tricked into running publisher code, and it also cannot validate numerics: it establishes
NAMES, DTYPES, SHAPES and COUNTS, which is exactly what a strict-match guard needs.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickle
import re
import zipfile
from pathlib import Path
from typing import Any

HEARTBEAT_OK = "CKPT_KEYS_OK"
HEARTBEAT_REFUSED = "CKPT_KEYS_REFUSED"

#: torch stores every tensor's element type in the storage class name.
_DTYPE_BYTES = {
    "FloatStorage": 4, "DoubleStorage": 8, "HalfStorage": 2, "BFloat16Storage": 2,
    "LongStorage": 8, "IntStorage": 4, "ShortStorage": 2, "CharStorage": 1,
    "ByteStorage": 1, "BoolStorage": 1,
}


class _Inst:
    """An inert instance. It must carry a real ``__dict__`` and accept ``__setstate__``.

    A pickle's BUILD opcode does ``inst.__dict__.update(state)`` or calls ``__setstate__``, so a
    stub that returns a bare dict raises ``AttributeError`` and the read dies for a reason that
    has nothing to do with the checkpoint. Nothing here executes publisher code: the class is
    never imported, only its NAME is recorded.
    """

    def __init__(self, qualname: str, args: int = 0):
        self.__qualname = qualname
        self.__args = args

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["__state__"] = state

    def __reduce__(self):                                        # pragma: no cover - not re-pickled
        return (_Inst, (self.__qualname,))

    def as_mapping(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_Inst__")}


class _TensorRef(_Inst):
    """A tensor's SHAPE and storage description, with no storage behind it.

    torch emits ``_rebuild_tensor_v2`` and then a BUILD carrying the tensor's backward hooks, so
    this must tolerate ``__setstate__`` like any other instance - returning a bare dict from the
    rebuild raises ``AttributeError: 'dict' object has no attribute '__dict__'`` and the read
    dies for a reason that has nothing to do with the checkpoint.
    """

    def __init__(self, storage, offset, size, stride):
        super().__init__("torch.Tensor")
        self.storage = storage
        self.size = tuple(size)
        self.stride = tuple(stride)
        self.offset = offset


class _Stub:
    """Records what the pickle asked for and constructs an inert instance."""

    def __init__(self, module: str, name: str):
        self.__module = module
        self.__name = name

    def __call__(self, *a, **k):
        return _Inst(f"{self.__module}.{self.__name}", len(a))

    def __repr__(self) -> str:                                   # pragma: no cover - debugging aid
        return f"<stub {self.__module}.{self.__name}>"


class _InertUnpickler(pickle.Unpickler):
    """Resolves every global to a stub and every storage reference to a description.

    ``persistent_load`` is where a torch checkpoint asks for its actual bytes. Returning a
    DESCRIPTION instead of a storage is what keeps peak memory flat: the object graph is rebuilt
    with placeholders and no tensor data is ever read out of the zip.
    """

    def __init__(self, fh, seen: list[str]):
        super().__init__(fh)
        self._seen = seen

    def find_class(self, module: str, name: str):
        self._seen.append(f"{module}.{name}")
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor",
                                                 "_rebuild_parameter"):
            def rebuild(*a):
                if name == "_rebuild_parameter":
                    return a[0]                     # (tensor, requires_grad, backward_hooks)
                storage, offset, size, stride = a[0], a[1], a[2], a[3]
                return _TensorRef(storage, offset, size, stride)
            return rebuild
        if module == "collections" and name == "OrderedDict":
            from collections import OrderedDict
            return OrderedDict
        return _Stub(module, name)

    def persistent_load(self, pid: Any):
        if isinstance(pid, tuple) and pid and pid[0] == "storage":
            kind = getattr(pid[1], "_Stub__name", None) or str(pid[1])
            return {"__storage__": True, "dtype": kind,
                    "key": pid[2] if len(pid) > 2 else None,
                    "numel": pid[4] if len(pid) > 4 else None}
        return {"__persistent__": repr(pid)}


def _walk_tensors(obj: Any, prefix: str = "", out: dict | None = None) -> dict:
    out = {} if out is None else out
    if isinstance(obj, _TensorRef):
        st = obj.storage if isinstance(obj.storage, dict) else {}
        out[prefix] = {"shape": list(obj.size), "storage_dtype": str(st.get("dtype", "?")),
                       "numel": int(_prod(obj.size))}
        return out
    if isinstance(obj, _Inst):
        return _walk_tensors(obj.as_mapping(), prefix, out)
    if isinstance(obj, dict):
        for k, v in obj.items():
            ks = str(k)
            if ks.startswith("__"):
                continue
            _walk_tensors(v, f"{prefix}.{ks}" if prefix else ks, out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _walk_tensors(v, f"{prefix}[{i}]", out)
    return out


def _prod(shape) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def inspect(path: Path, *, expect_sha256: str | None = None) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    measured = _sha256(path)
    if expect_sha256 and measured != expect_sha256:
        raise RuntimeError(
            f"REFUSED: {path.name} hashes to {measured}, expected {expect_sha256}. Identity is "
            f"the hash, never the filename or the byte size - and note that on a Xet-backed "
            f"HuggingFace repository the `x-linked-etag` header is NOT the sha256, so the "
            f"authoritative digest is the API tree's lfs.oid"
        )
    if not zipfile.is_zipfile(path):
        raise RuntimeError(
            f"REFUSED: {path.name} is not a zip archive, so it is a legacy torch pickle. This "
            f"reader only handles the zipfile format, which is the one that permits reading the "
            f"structure without reading the data"
        )
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        pkl = [n for n in names if n.endswith("data.pkl")]
        if not pkl:
            raise RuntimeError(f"REFUSED: no data.pkl in {path.name}; entries: {names[:8]}")
        raw = z.read(pkl[0])
        storage_entries = [n for n in names if re.search(r"/data/\d+$", n)]
        storage_bytes = sum(zi.file_size for zi in z.infolist()
                            if re.search(r"/data/\d+$", zi.filename))
    seen: list[str] = []
    obj = _InertUnpickler(io.BytesIO(raw), seen).load()

    # A training checkpoint usually wraps the weights; find the state dict wherever it lives.
    if isinstance(obj, _Inst):
        obj = obj.as_mapping()
    wrapper_keys = sorted(str(k) for k in obj.keys()) if isinstance(obj, dict) else []
    tensors = _walk_tensors(obj)
    numel = sum(t["numel"] for t in tensors.values())
    prefixes: dict[str, int] = {}
    for k in tensors:
        head = k.split(".")[0]
        prefixes[head] = prefixes.get(head, 0) + 1
    return {
        "file": str(path), "bytes": path.stat().st_size, "sha256": measured,
        "zip_entries": len(names), "storage_entries": len(storage_entries),
        "storage_bytes": storage_bytes,
        "blob_to_storage_ratio": round(path.stat().st_size / max(storage_bytes, 1), 4),
        "wrapper_top_level_keys": wrapper_keys,
        "n_tensors": len(tensors), "n_elements": numel,
        "top_level_prefixes": dict(sorted(prefixes.items(), key=lambda kv: -kv[1])),
        "first_10_keys": sorted(tensors)[:10],
        "last_10_keys": sorted(tensors)[-10:],
        "classes_the_pickle_named": sorted(set(seen)),
        "executed_the_archived_graph": False,
        "materialised_any_tensor": False,
        "heartbeat": HEARTBEAT_OK,
    }


def strict_match(ckpt_keys: list[str], model_keys: list[str]) -> dict:
    """The FACT-0425 guard, made explicit: what would ``strict=False`` have swallowed?"""
    c, m = set(ckpt_keys), set(model_keys)
    missing, unexpected = sorted(m - c), sorted(c - m)
    return {
        "n_checkpoint_keys": len(c), "n_model_keys": len(m),
        "n_matched": len(c & m), "n_missing": len(missing), "n_unexpected": len(unexpected),
        "missing_first_10": missing[:10], "unexpected_first_10": unexpected[:10],
        "strict_would_pass": not missing and not unexpected,
        "why_this_matters": (
            "FACT-0425: the publisher loads with strict=False and its two reporting prints are "
            "COMMENTED OUT, so a mismatch runs the model on RANDOM WEIGHTS silently. On a "
            "300-query segmenter that is indistinguishable from a model that does not transfer."
        ),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--expect-sha256")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    try:
        rep = inspect(args.checkpoint, expect_sha256=args.expect_sha256)
    except Exception as exc:                       # recorded loudly, never swallowed
        print(f"{HEARTBEAT_REFUSED} {type(exc).__name__}: {exc}", flush=True)
        return 2
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"  {args.checkpoint.name}  {rep['bytes']:,} B  sha256={rep['sha256'][:16]}...")
    print(f"  wrapper keys      : {rep['wrapper_top_level_keys']}")
    print(f"  tensors           : {rep['n_tensors']:,}  elements {rep['n_elements']:,}")
    print(f"  storage bytes     : {rep['storage_bytes']:,}  "
          f"blob/storage {rep['blob_to_storage_ratio']}")
    print(f"  top prefixes      : {list(rep['top_level_prefixes'].items())[:6]}")
    print(f"  first key         : {rep['first_10_keys'][0] if rep['first_10_keys'] else '-'}")
    print(rep["heartbeat"], flush=True)              # the LAST line; its absence is the alarm
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
