#!/usr/bin/env python
"""Gate 1 replay worker, CONTRACT 2: re-run ONLY the deployed PRIMARY head, from a cache.

WHAT THIS IS. The second half of the passive-instrumentation design of PKT-0029 Gate 1. The first
half (scripts/kaggle_edits/assoc_feature_tap.py) injects a tap INSIDE the deployed `predict_video`
and records the exact tensors it hands to `model.predict_edges`, the PRE-FUSION primary logits and
probabilities that call produced, and the POST-FUSION deployed probabilities computed further
down. This script starts in a FRESH PROCESS, loads only those files, and re-runs the head.

WHAT CHANGED FROM CONTRACT 1, AND WHY. Contract 1 compared this replay against the DEPLOYED
probabilities, which are computed after the bidirectional harmonic
(predict_unet_transformer.py:602-659) and the secondary logit blend (:660-756). A primary-only
replay cannot equal a fused quantity and should not be asked to. P36 / EXP-0043 therefore returned
an INVALID verdict, not a negative one (FACT-0403).

    THE PARITY TARGET IS NOW `primary_*_preblend`.  The deployed post-fusion probability is kept
    and reported, but it is OBSERVATIONAL EVIDENCE for reconstructing deployed behaviour - it is
    never a pass condition.

WHICH BAND MAY BE RE-DERIVED, AND WHICH MAY NOT. Bands A and B were SELECTED on the deployed
post-fusion surface. Re-deriving their membership on the pre-blend surface would silently change
what those artifacts mean, so their membership is reported and never gated. Band P was selected on
the pre-blend surface, so its membership IS the gate's membership condition.

WHAT IT DELIBERATELY DOES NOT DO. It never opens a zarr, never reads an image, never normalises,
never downsamples, never runs the UNet trunk and never extracts a detection peak. Five of the six
defects this gate produced were reimplementations of exactly those steps (FACT-0387, FACT-0397,
FACT-0399, FACT-0400, plus the per-frame feature-cache defect FACT-0402). There is nothing left
here to reimplement, so there is nothing left of that class to get wrong.

WHY A SEPARATE PROCESS. Attempt 1 wrote features into a dict and read them back in the same
process, so it exercised no cache at all. Here the operating system enforces "from the cache
alone": this interpreter cannot see the writer's memory.

WHY IT STILL NEEDS THE DEPLOYED MODULE. Only to rebuild the head: `load_model` reconstructs
UNetNodeTransformer from the same config and weights, and `predict_edges` is the deployed method.
It resolves via PYTHONPATH, which the LAUNCHER must set to scripts + src - cwd is NOT on sys.path
for a script invocation, which is the FACT-0399 defect verbatim.

FAIL CLOSED. A missing cache, a contract-1 cache, an empty band, a node-ordering disagreement, an
unreadable file or any exception leaves all_passed False.

THE SOFTMAX BLIND SPOT IS CLOSED. Contract 1 compared probabilities only, and a source-axis
softmax is invariant to a per-target constant logit offset, so an error of that exact shape was
mathematically invisible. Contract 2 records the pre-fusion LOGITS per band row and compares them
too, which is the fibre that invariance hides.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

BANDS = ("a", "b", "p")
GATED_MEMBERSHIP_BAND = "p"
REQUIRED_CONTRACT = 2


def _import_deployed():
    """Import the deployed predictor, naming the launcher defect if it is not importable."""
    try:
        import predict_unet_transformer as AFP  # noqa: PLC0415
    except ImportError:
        print(
            "AFT_REPLAY_IMPORT_FAILED predict_unet_transformer is not importable. "
            "sys.path[0] is this SCRIPT's directory, not the working directory, so cwd=REPO_DIR "
            "alone never makes this work. The launcher must pass "
            'env={**os.environ, "PYTHONPATH": "scripts" + os.pathsep + "src"} - "src" alone is '
            "not enough because predict_unet_transformer lives in scripts/ (FACT-0399). "
            f"sys.path[0]={sys.path[0]!r}",
            flush=True,
        )
        raise
    return AFP


def _tensor(arr, device, dtype=torch.float32):
    return torch.from_numpy(np.ascontiguousarray(arr)).to(device=device, dtype=dtype)


def _slice(cache, pair, role):
    ptr = int(cache["pair_src_ptr" if role == 0 else "pair_tgt_ptr"][pair])
    n = int(cache["pair_src_n" if role == 0 else "pair_tgt_n"][pair])
    return slice(ptr, ptr + n)


def _activate(logits, activation):
    """The DEPLOYED activation over the SOURCE axis - predict_unet_transformer.py:758-762."""
    if activation == "softmax":
        return torch.softmax(logits, dim=0)
    return torch.sigmoid(logits)


def _bidirectional_harmonic(forward, reverse_native, weight):
    """The deployed bidirectional harmonic, copied verbatim from the deployed source.

    predict_unet_transformer.py:612-650. Copied rather than paraphrased because this is the one
    fusion stage a PRIMARY-ONLY cache can reproduce: it re-runs the SAME model on the SAME cached
    tensors with the source and target roles swapped. The secondary blend cannot be reproduced
    here and is never attempted - the cache does not carry a second model's features, and it must
    not, because those features are not what the primary head is asked to consume.
    """
    reverse_logits_pair = reverse_native.transpose(1, 2)
    forward_center = forward.mean(dim=1, keepdim=True)
    forward_scale = forward.float().std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
    reverse_center = reverse_logits_pair.mean(dim=1, keepdim=True)
    reverse_scale = reverse_logits_pair.float().std(
        dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
    reverse_scale_ratio = (forward_scale / reverse_scale).clamp(0.5, 2.0)
    reverse_scale_ratio = reverse_scale_ratio.to(reverse_logits_pair.dtype)
    reverse_aligned = (
        (reverse_logits_pair - reverse_center) * reverse_scale_ratio + forward_center
    )
    forward_prob = torch.softmax(forward.float(), dim=1).clamp_min(1e-8)
    reverse_prob = torch.softmax(reverse_aligned.float(), dim=1).clamp_min(1e-8)
    harmonic_prob = 1.0 / ((1.0 - weight) / forward_prob + weight / reverse_prob)
    harmonic_prob = harmonic_prob / harmonic_prob.sum(dim=1, keepdim=True).clamp_min(1e-8)
    harmonic_logits = torch.log(harmonic_prob.clamp_min(1e-8))
    harmonic_center = harmonic_logits.mean(dim=1, keepdim=True)
    harmonic_scale = harmonic_logits.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
    harmonic_scale_ratio = (forward_scale / harmonic_scale).clamp(0.5, 2.0)
    return (
        (harmonic_logits - harmonic_center) * harmonic_scale_ratio + forward_center
    ).to(reverse_aligned.dtype)


def _check_contract(cache, crop):
    """Refuse a cache written under a contract this worker cannot interpret.

    A contract-1 cache has one column called `band_a_prob` and no record of which fusion stages
    ran. Reading it here would silently resurrect the comparison that made P36 invalid, so it is
    rejected by name rather than by KeyError.
    """
    if "contract_version" not in cache.files:
        raise ValueError(
            f"{crop}: this cache has no contract_version, so it was written by contract 1, whose "
            "single unqualified `band_a_prob` column holds the POST-FUSION deployed probability. "
            "Comparing a primary-only replay against it is the defect FACT-0403 records. "
            "Re-run the tap at contract 2."
        )
    got = int(cache["contract_version"])
    if got != REQUIRED_CONTRACT:
        raise ValueError(
            f"{crop}: cache contract_version {got} != worker contract {REQUIRED_CONTRACT}"
        )


def _check_node_ordering(cache, pair, role, failures):
    """The cached role block must be exactly the deployed frame slice, in the deployed order.

    `coords_so_far` numbers nodes by frame in detection order, and the pre-ILP export and the ECB
    sidecars both address nodes by that number. If the cache reorders within a frame, every id in
    every downstream artifact silently means a different cell.
    """
    sl = _slice(cache, pair, role)
    t = int(cache["pair_t_src" if role == 0 else "pair_t_tgt"][pair])
    frames = cache["frames"].tolist()
    if t not in frames:
        failures.append({"pair": int(pair), "role": int(role), "frame": t,
                         "why": "frame absent from the deployed coord_offset"})
        return
    k = frames.index(t)
    expected = np.arange(int(cache["starts"][k]), int(cache["ends"][k]), dtype=np.int64)
    got = cache["role_gid"][sl]
    if got.shape != expected.shape or not np.array_equal(got, expected):
        failures.append({
            "pair": int(pair), "role": int(role), "frame": t,
            "why": "node ids are not the deployed frame slice in the deployed order",
            "expected_n": int(expected.shape[0]), "cached_n": int(got.shape[0]),
        })


def _pos_features(AFP, cache, pair, role):
    """Positional features: the recorded ones when present, else the DEPLOYED function.

    Re-derivation is not a reimplementation - it calls `extract_pos_features` itself, on the
    window-relative coordinates and window shape the deployed code passed it, both recorded
    verbatim. When both are available they are cross-checked, which costs nothing and catches a
    torn window_shape record.
    """
    sl = _slice(cache, pair, role)
    rel = cache["role_coord_rel"][sl].astype(np.int16)
    shape = tuple(int(v) for v in cache["pair_window_shape"][pair])
    derived = AFP.extract_pos_features(rel, shape)
    if "role_pos" in cache.files:
        recorded = cache["role_pos"][sl]
        return recorded, float(np.abs(recorded - derived).max()) if recorded.size else 0.0
    return derived, 0.0


def _read_fusion(cache):
    return {
        "stages": str(cache["fusion_stages"]),
        "bidirectional_weight": float(cache["fusion_bidirectional_weight"]),
        "secondary_enabled": bool(cache["fusion_secondary_enabled"]),
        "secondary_edge_weight": float(cache["fusion_secondary_edge_weight"]),
        "secondary_link_mode": str(cache["fusion_secondary_link_mode"]),
        "secondary_mix_temperature": float(cache["fusion_secondary_mix_temperature"]),
        "reproducible_from_primary_cache": bool(
            cache["fusion_reproducible_from_primary_cache"]),
    }


def replay_crop(AFP, model, cache, device, tol, logit_tol, reconstruct):
    """Re-run the PRIMARY head for every recorded pair and compare against the primary record."""
    dtype = getattr(torch, str(cache["feat_dtype"]), torch.float32)
    activation = str(cache["edge_activation"])
    threshold = float(cache["edge_threshold"])
    n_pairs = int(cache["pair_f_idx"].shape[0])
    fusion = _read_fusion(cache)

    integrity: list[dict] = []
    if int(cache["coords"].shape[0]) != int(cache["node_count"]):
        integrity.append({"why": "coords length does not equal the deployed node count",
                          "coords": int(cache["coords"].shape[0]),
                          "node_count": int(cache["node_count"])})
    if not bool(cache["role_mask"].all()):
        integrity.append({"why": "a cached mask slot is False; the deployed site builds "
                                 "torch.ones and never pads",
                          "false_slots": int((~cache["role_mask"]).sum())})
    # THE DEPLOYED COORDINATE RESCALE, EXERCISED ON THE REAL DOWNSAMPLE RATHER THAN ASSUMED.
    # The tap flushes BETWEEN the coordinate concatenate and the rescale
    # (predict_unet_transformer.py:798-802), so `coords` holds the DOWNSAMPLED grid the model
    # indexed while `role_coord_scaled` holds `coords[:, 1:] * downsample` - the tensor the head
    # was actually fed, built from the same `ds_arr` at :336-337. Defect 7 (FACT-0405) survived
    # because a fixture used downsample (1,1,1), where that multiplication is the identity. This
    # runs on the deployed (1,4,4) against real data, so the step under test cannot be deleted by
    # the fixture that is supposed to check it.
    ds_vec = np.asarray(cache["downsample"]).astype(np.float32)
    gid_all = cache["role_gid"]
    n_coords = int(cache["coords"].shape[0])
    if gid_all.size and (int(gid_all.min()) < 0 or int(gid_all.max()) >= n_coords):
        integrity.append({"why": "a role node id is outside the deployed coord table",
                          "role_gid_max": int(gid_all.max()), "coords": n_coords})
    else:
        expect_scaled = cache["coords"][gid_all][:, 1:].astype(np.float32) * ds_vec
        got_scaled = cache["role_coord_scaled"].astype(np.float32)
        if got_scaled.shape != expect_scaled.shape or not np.array_equal(got_scaled,
                                                                        expect_scaled):
            bad = (int(np.argmax(np.abs(got_scaled - expect_scaled).max(axis=1)))
                   if got_scaled.shape == expect_scaled.shape and got_scaled.size else -1)
            integrity.append({
                "why": "role_coord_scaled is not coords[:, 1:] * downsample - the deployed "
                       "rescale at predict_unet_transformer.py:798-802 (FACT-0405 defect 7)",
                "downsample": [int(d) for d in ds_vec],
                "role_slots": int(got_scaled.shape[0]), "first_bad_slot": bad,
            })
    for name in BANDS:
        declared = str(cache[f"band_{name}_selected_on"])
        expected = "primary_probs_preblend" if name == "p" else "deployed_probs_postblend"
        if declared != expected:
            integrity.append({"band": name, "why": "the band declares the wrong selection "
                                                   "surface", "declared": declared,
                              "expected": expected})

    band = {
        name: {
            "pair": cache[f"band_{name}_pair"].astype(np.int64),
            "i": cache[f"band_{name}_i"].astype(np.int64),
            "j": cache[f"band_{name}_j"].astype(np.int64),
            "src": cache[f"band_{name}_source_id"].astype(np.int64),
            "tgt": cache[f"band_{name}_target_id"].astype(np.int64),
            "logit_pre": cache[f"band_{name}_primary_logit_preblend"].astype(np.float64),
            "prob_pre": cache[f"band_{name}_primary_prob_preblend"].astype(np.float64),
            "prob_post": cache[f"band_{name}_deployed_prob_postblend"].astype(np.float64),
        }
        for name in BANDS
    }
    tally = {
        name: {"recorded": 0, "checked": 0, "prob_delta": 0.0, "logit_delta": 0.0,
               "postblend_gap": 0.0, "missing": 0, "extra": 0, "reproduced": 0}
        for name in BANDS
    }
    # The contract-1 verdict, computed alongside so the correction is measured, not asserted.
    v1 = {"max_abs_prob_delta": 0.0, "band_a_missing": 0, "band_a_extra": 0, "checked": 0}
    recon = {"max_abs_delta": 0.0, "band_a_missing": 0, "band_a_extra": 0, "pairs": 0}
    id_failures: list[dict] = []
    pos_delta = 0.0
    # WHAT THE GATE ACTUALLY TOUCHED. Every comparison is per BAND ROW, so a role node that
    # appears in no band row of its pair is never compared. Reporting the fraction that was
    # touched keeps the scope of a pass measurable instead of assumed - "non-vacuous" was the
    # word that let FACT-0394 through, and a fraction cannot be read that loosely.
    covered_slots: set[tuple[int, int, int]] = set()
    total_slots = 0
    for pair in range(n_pairs):
        for role in (0, 1):
            _check_node_ordering(cache, pair, role, integrity)
        s, t = _slice(cache, pair, 0), _slice(cache, pair, 1)
        pos_s, ds_ = _pos_features(AFP, cache, pair, 0)
        pos_t, dt_ = _pos_features(AFP, cache, pair, 1)
        pos_delta = max(pos_delta, ds_, dt_)
        n_src = s.stop - s.start
        n_tgt = t.stop - t.start
        total_slots += n_src + n_tgt
        if n_src == 0 or n_tgt == 0:
            continue
        feat_s = _tensor(cache["role_feat"][s], device, dtype).unsqueeze(0)
        feat_t = _tensor(cache["role_feat"][t], device, dtype).unsqueeze(0)
        coord_s = _tensor(cache["role_coord_scaled"][s], device).unsqueeze(0)
        coord_t = _tensor(cache["role_coord_scaled"][t], device).unsqueeze(0)
        ps = _tensor(pos_s, device).unsqueeze(0)
        pt = _tensor(pos_t, device).unsqueeze(0)
        mask_s = torch.from_numpy(cache["role_mask"][s]).to(device).unsqueeze(0)
        mask_t = torch.from_numpy(cache["role_mask"][t]).to(device).unsqueeze(0)
        with torch.no_grad():
            primary_logits = model.predict_edges(
                feat_s, feat_t, coord_s, coord_t, ps, pt, mask_s, mask_t,
            )
        primary_logit_np = primary_logits[0].to("cpu", torch.float64).numpy()
        primary_prob_np = _activate(
            primary_logits[0].to(torch.float32), activation,
        ).to("cpu", torch.float64).numpy()
        gid_s = cache["role_gid"][s]
        gid_t = cache["role_gid"][t]

        # BAND P MEMBERSHIP - the ONLY membership the gate re-derives, because band P is the only
        # band selected on the surface this replay computes. Both directions are checked: nothing
        # recorded may be absent from the replay, and nothing the replay puts above the threshold
        # may be unrecorded.
        rep_keys = {(int(a), int(b)) for a, b in np.argwhere(primary_prob_np > threshold)}
        sel = band[GATED_MEMBERSHIP_BAND]["pair"] == pair
        rec_keys = {(int(a), int(b)) for a, b in zip(band[GATED_MEMBERSHIP_BAND]["i"][sel],
                                                    band[GATED_MEMBERSHIP_BAND]["j"][sel])}
        tally[GATED_MEMBERSHIP_BAND]["reproduced"] += len(rep_keys)
        tally[GATED_MEMBERSHIP_BAND]["missing"] += len(rec_keys - rep_keys)
        tally[GATED_MEMBERSHIP_BAND]["extra"] += len(rep_keys - rec_keys)

        # BAND A MEMBERSHIP, OBSERVATIONAL ONLY. Band A was selected on the post-fusion surface,
        # so this difference measures the FUSION, not the cache. It is exactly the number
        # contract 1 gated on, and it is reported here so the correction can be seen rather than
        # believed.
        sel_a = band["a"]["pair"] == pair
        rec_a = {(int(x), int(y)) for x, y in zip(band["a"]["i"][sel_a], band["a"]["j"][sel_a])}
        v1["band_a_missing"] += len(rec_a - rep_keys)
        v1["band_a_extra"] += len(rep_keys - rec_a)

        for name in BANDS:
            rows = band[name]["pair"] == pair
            sel_i, sel_j = band[name]["i"][rows], band[name]["j"][rows]
            sel_src, sel_tgt = band[name]["src"][rows], band[name]["tgt"][rows]
            tally[name]["recorded"] += int(sel_i.size)
            if sel_i.size == 0:
                continue
            inside = (sel_i < n_src) & (sel_j < n_tgt)
            if not bool(inside.all()):
                tally[name]["missing"] += int((~inside).sum())
                id_failures.append({"pair": int(pair), "band": name,
                                    "why": "a recorded local index is outside the pair's block"})
            si, sj = sel_i[inside], sel_j[inside]
            # The recorded GLOBAL ids must be what the cached node ordering says they are. This
            # ties the band artifacts to the node numbering the pre-ILP export and the ECB
            # sidecars address, so an id in either artifact cannot silently mean a different cell.
            if not (np.array_equal(gid_s[si], sel_src[inside])
                    and np.array_equal(gid_t[sj], sel_tgt[inside])):
                id_failures.append({"pair": int(pair), "band": name,
                                    "why": "recorded global ids disagree with the cached node "
                                           "ordering"})
            tally[name]["checked"] += int(si.size)
            if si.size == 0:
                continue
            covered_slots.update((pair, 0, int(v)) for v in np.unique(si))
            covered_slots.update((pair, 1, int(v)) for v in np.unique(sj))
            # THE PARITY COMPARISON. Pre-fusion probability AND pre-fusion logit.
            tally[name]["prob_delta"] = max(
                tally[name]["prob_delta"],
                float(np.abs(primary_prob_np[si, sj]
                             - band[name]["prob_pre"][rows][inside]).max()),
            )
            tally[name]["logit_delta"] = max(
                tally[name]["logit_delta"],
                float(np.abs(primary_logit_np[si, sj]
                             - band[name]["logit_pre"][rows][inside]).max()),
            )
            # OBSERVATIONAL. The distance between the primary surface and the deployed one. This
            # is what contract 1 called the parity delta.
            gap = float(np.abs(primary_prob_np[si, sj]
                               - band[name]["prob_post"][rows][inside]).max())
            tally[name]["postblend_gap"] = max(tally[name]["postblend_gap"], gap)
            if name in ("a", "b"):
                v1["max_abs_prob_delta"] = max(v1["max_abs_prob_delta"], gap)
                v1["checked"] += int(si.size)

        # DEPLOYED RECONSTRUCTION, OBSERVATIONAL. When the only fusion stage that ran is the
        # bidirectional harmonic, the deployed probability IS reproducible from this cache,
        # because that stage re-runs the same primary model with the roles swapped.
        if reconstruct and fusion["reproducible_from_primary_cache"]:
            fused = primary_logits
            if fusion["bidirectional_weight"] > 0.0:
                with torch.no_grad():
                    reverse = model.predict_edges(
                        feat_t, feat_s, coord_t, coord_s, pt, ps, mask_t, mask_s,
                    )
                fused = _bidirectional_harmonic(
                    primary_logits, reverse, fusion["bidirectional_weight"])
            recon_prob = _activate(
                fused[0].to(torch.float32), activation,
            ).to("cpu", torch.float64).numpy()
            recon["pairs"] += 1
            for name in ("a", "b"):
                rows = band[name]["pair"] == pair
                si, sj = band[name]["i"][rows], band[name]["j"][rows]
                inside = (si < n_src) & (sj < n_tgt)
                si, sj = si[inside], sj[inside]
                if si.size:
                    recon["max_abs_delta"] = max(
                        recon["max_abs_delta"],
                        float(np.abs(recon_prob[si, sj]
                                     - band[name]["prob_post"][rows][inside]).max()),
                    )
            recon_keys = {(int(a), int(b)) for a, b in np.argwhere(recon_prob > threshold)}
            recon["band_a_missing"] += len(rec_a - recon_keys)
            recon["band_a_extra"] += len(recon_keys - rec_a)

    integrity.extend(id_failures)
    result = {
        "pairs": n_pairs,
        "nodes": int(cache["node_count"]),
        "fusion": fusion,
        "integrity_failures": integrity[:20],
        "integrity_failure_count": len(integrity),
        "pos_feature_max_abs_delta": pos_delta,
        "role_slot_coverage": {
            "compared": len(covered_slots),
            "total": total_slots,
            "fraction": (len(covered_slots) / total_slots) if total_slots else 0.0,
            "what": "role-node slots that appear in at least one compared band row. Slots "
                    "outside this set were persisted but never compared, so a pass says "
                    "nothing about them.",
        },
    }

    # NON-VACUITY, ALL THREE BANDS. A crop whose band compared nothing satisfies every condition
    # on that band trivially. FACT-0394 is what that costs: the repository's own green happy-path
    # test ran a crop whose band A compared ZERO pairs and asserted a pass. The floor is PER CROP.
    # BAND B is not optional either: FACT-0382 measured that ALL 691 fold-0 contested errors have
    # their true parent at or below the deployed threshold, so a band-A-only pass is not Gate 1.
    for name in BANDS:
        row = {
            "selected_on": str(cache[f"band_{name}_selected_on"]),
            "recorded": tally[name]["recorded"],
            "checked": tally[name]["checked"],
            "max_abs_primary_prob_delta": tally[name]["prob_delta"],
            "max_abs_primary_logit_delta": tally[name]["logit_delta"],
            "max_abs_deployed_postblend_gap": tally[name]["postblend_gap"],
        }
        if name == GATED_MEMBERSHIP_BAND:
            row.update({"reproduced": tally[name]["reproduced"],
                        "missing": tally[name]["missing"],
                        "extra": tally[name]["extra"]})
        result[f"band_{name}"] = row

    result["contract_v1_verdict"] = {
        "what": "the comparison contract 1 gated on: this primary-only replay against the "
                "DEPLOYED post-fusion probability, plus band-A membership re-derived on the "
                "pre-fusion surface",
        "checked": v1["checked"],
        "max_abs_prob_delta": v1["max_abs_prob_delta"],
        "band_a_missing": v1["band_a_missing"],
        "band_a_extra": v1["band_a_extra"],
        "would_have_passed": bool(
            v1["max_abs_prob_delta"] <= tol
            and v1["band_a_missing"] == 0 and v1["band_a_extra"] == 0
        ),
    }
    result["deployed_reconstruction"] = {
        "attempted": bool(reconstruct and fusion["reproducible_from_primary_cache"]),
        "why_not": None if fusion["reproducible_from_primary_cache"] else
                   "a secondary model's logits were blended in; a primary-only cache cannot "
                   "carry them and must not pretend to",
        "pairs": recon["pairs"],
        "max_abs_delta": recon["max_abs_delta"],
        "band_a_missing": recon["band_a_missing"],
        "band_a_extra": recon["band_a_extra"],
    }

    parity_ok = all(
        result[f"band_{name}"]["max_abs_primary_prob_delta"] <= tol
        and result[f"band_{name}"]["max_abs_primary_logit_delta"] <= logit_tol
        and result[f"band_{name}"]["checked"] > 0
        for name in BANDS
    )
    result["passed"] = bool(
        not integrity
        and parity_ok
        and result[f"band_{GATED_MEMBERSHIP_BAND}"]["missing"] == 0
        and result[f"band_{GATED_MEMBERSHIP_BAND}"]["extra"] == 0
        and pos_delta <= 1e-6
    )
    return result


def verify(args) -> int:
    AFP = _import_deployed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, window, downsample = AFP.load_model(Path(args.weights), device)
    cache_dir = Path(args.cache_dir)
    names = args.crops or sorted(p.stem for p in cache_dir.glob("*.npz"))
    report = {
        "gate": "feature_tap_parity",
        "contract_version": REQUIRED_CONTRACT,
        "comparison_target": "primary_logits_preblend and primary_probs_preblend, against a "
                             "replay of the PRIMARY head from the cached features",
        "observational_only": [
            "deployed_probs_postblend",
            "band_a and band_b membership (both were selected on deployed_probs_postblend)",
            "deployed_reconstruction",
            "contract_v1_verdict",
        ],
        "device": device.type, "window": int(window),
        "downsample": [int(d) for d in downsample],
        "tolerance": args.tol, "logit_tolerance": args.logit_tol,
        "crops": [], "all_passed": False,
    }
    if not names:
        # A gate that can quietly compare nothing looks exactly like a gate that passed
        # (FACT-0387 compared zero crops and had to be diagnosed from a log).
        report["error"] = f"no cache files under {cache_dir} - the tap wrote nothing"
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"AFT_REPLAY_FAILED {report['error']}", flush=True)
        return 1

    for crop in names:
        path = cache_dir / f"{crop}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"cache missing for {crop} - the tap did not write it")
        with np.load(path, allow_pickle=False) as cache:
            _check_contract(cache, crop)
            if int(cache["window"]) != int(window):
                raise ValueError(
                    f"{crop}: cache window {int(cache['window'])} != model window {int(window)}"
                )
            row = replay_crop(AFP, model, cache, device, args.tol, args.logit_tol,
                              not args.no_reconstruct_deployed)
        row["crop"] = crop
        report["crops"].append(row)
        print(
            f"AFT_REPLAY crop={crop} pairs={row['pairs']} nodes={row['nodes']} "
            f"fusion={row['fusion']['stages']} "
            + " ".join(
                f"{n.upper()}[rec={row['band_' + n]['recorded']} "
                f"checked={row['band_' + n]['checked']} "
                f"dp={row['band_' + n]['max_abs_primary_prob_delta']:.3e} "
                f"dl={row['band_' + n]['max_abs_primary_logit_delta']:.3e} "
                f"gap={row['band_' + n]['max_abs_deployed_postblend_gap']:.3e}]"
                for n in BANDS
            )
            + f" P_miss={row['band_p']['missing']} P_extra={row['band_p']['extra']} "
            f"integrity_failures={row['integrity_failure_count']} "
            f"pos_d={row['pos_feature_max_abs_delta']:.3e} "
            f"slot_cov={row['role_slot_coverage']['compared']}/"
            f"{row['role_slot_coverage']['total']} "
            + (f"recon_d={row['deployed_reconstruction']['max_abs_delta']:.3e} "
               if row["deployed_reconstruction"]["attempted"] else "recon_d=NA ")
            + f"v1_would_have_passed={row['contract_v1_verdict']['would_have_passed']} "
            f"passed={row['passed']}",
            flush=True,
        )

    # THE CROP COUNT IS AN EXTERNAL CONDITION AND MUST BE STATED BY THE CALLER. `all_passed` is
    # an ALL over the crops that happened to exist, so a run that captured ONE crop - or a run
    # whose second crop raised before the flush - satisfies it exactly as well as a run that
    # captured both. PKT-0036 measured that: a one-crop run passed. The expectation therefore
    # comes in from the spec (BIOHUB_AFT_EXPECT_CROPS) and is enforced here, in the same
    # fail-closed report, instead of being left to a human reading the receipt afterwards.
    expected = int(args.expect_crops or 0)
    report["crop_count"] = {
        "captured": len(report["crops"]), "expected": expected or None,
        "asserted": bool(expected),
        "why_external": "all_passed is an ALL over the crops that exist; it cannot see a crop "
                        "that never ran",
    }
    count_ok = (not expected) or len(report["crops"]) == expected
    if not count_ok:
        report["crop_count"]["error"] = (
            f"expected {expected} crops, captured {len(report['crops'])}"
        )
        print(f"AFT_CROP_COUNT_FAILED {report['crop_count']['error']}", flush=True)
    else:
        print(f"AFT_CROP_COUNT_OK captured={len(report['crops'])} expected="
              f"{expected or 'unasserted'}", flush=True)
    report["all_passed"] = bool(report["crops"]) and count_ok and all(
        c["passed"] for c in report["crops"])
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["all_passed"] else 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True,
                    help="the SAME weights the deployed run used; config.json sits beside it")
    ap.add_argument("--cache-dir", required=True, help="directory of per-crop tap .npz files")
    ap.add_argument("--out", required=True, help="where the gate report JSON is written")
    ap.add_argument("--crops", nargs="*", default=None)
    ap.add_argument("--expect-crops", type=int, default=0,
                    help="how many crops this run MUST have captured. 0 leaves it unasserted, "
                         "which is what let a one-crop run pass all_passed.")
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="max |replay primary probability - recorded primary probability|")
    ap.add_argument("--logit-tol", type=float, default=1e-3,
                    help="max |replay primary logit - recorded primary logit|. Probabilities "
                         "alone cannot see a per-target constant logit offset; this can.")
    ap.add_argument("--no-reconstruct-deployed", action="store_true",
                    help="skip the observational reconstruction of the deployed probability "
                         "(it costs one extra reverse head call per pair)")
    args = ap.parse_args(argv)
    return verify(args)


if __name__ == "__main__":
    raise SystemExit(main())
