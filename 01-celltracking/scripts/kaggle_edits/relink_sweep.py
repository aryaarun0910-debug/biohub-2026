# =====================================================================================
# RELINK DIVISION PENALTY SWEEP — buy the whole curve from ONE GPU session
#
# WHY A SWEEP AND NOT AN ARM
# On 2026-08-25 two submission slots were spent on two points of the detection curve
# (p16 at 0.90, p17 at 0.94). One offline CPU pass over the detpeak export then reproduced
# the entire curve and predicted both scores. The lesson is explicit: BUY THE CURVE, NOT
# THE POINT. The relink penalty has exactly the same shape — a single scalar whose optimum
# is unknown — so a per-value submission would repeat the mistake.
#
# WHY IT IS CHEAP
# The expensive work (UNet inference, edge prediction, the ILP solve) is already done and
# cached as per-crop `.geff` files. Only `filter_output_graph` is re-run per penalty, and
# that is post-processing. So N penalties cost roughly N x (post-processing), not
# N x (whole pipeline).
#
# WHERE IT MUST RUN
# BEFORE the LOEO export cell's cleanup. That cleanup deletes every /kaggle/working entry
# not in `_LOEO_KEEP`, and `tracking_repo/` — which holds the geffs — is inside
# /kaggle/working. A cell appended after it would find no geffs at all.
#
# WHAT IT EMITS
# One gzipped submission-schema CSV per penalty, scoreable offline against `data/train`
# with `scripts/core/score_oof.py`. **Zero submission slots.** The penalty `-1` is the
# DISABLED sentinel and reproduces the champion exactly, so it is the in-run control:
# if its node/edge counts do not match the primary LOEO export, the sweep is not measuring
# what it claims to.
# =====================================================================================
_sweep_raw = os.environ.get("BIOHUB_RELINK_DIVISION_SWEEP", "").strip()
if _sweep_raw:
    import gzip as _sw_gzip

    _sw_pens = [float(_x) for _x in _sweep_raw.split(",") if _x.strip()]
    print(f"relink sweep: {len(_sw_pens)} penalties {_sw_pens}", flush=True)
    _sw_keep: list[str] = []
    _sw_summary: list[dict] = []

    for _sw_pen in _sw_pens:
        # motion_relink_edges reads RELINK_DIVISION_PENALTY as a module global at call
        # time, so rebinding it here is what varies the arm.
        globals()["RELINK_DIVISION_PENALTY"] = _sw_pen
        _sw_tag = ("off" if _sw_pen < 0 else f"{_sw_pen:g}").replace(".", "p").replace("-", "m")
        _sw_out = Path(f"/kaggle/working/sweep_pen_{_sw_tag}.csv.gz")
        _sw_rid = _sw_nodes = _sw_edges = _sw_forks = 0

        with _sw_gzip.open(_sw_out, "wt", newline="", compresslevel=6) as _sw_f:
            _sw_w = csv.DictWriter(_sw_f, fieldnames=CSV_COLUMNS)
            _sw_w.writeheader()
            for _sw_gp in geffs:
                _sw_ds = _sw_gp.stem
                _sw_g = graph_from_geff(_sw_gp)

                _sw_nbid: dict[int, dict] = {}
                for _sw_r in _sw_g.node_attrs().iter_rows(named=True):
                    _sw_nid = int(_sw_r["node_id"])
                    _sw_nbid[_sw_nid] = {
                        "node_id": _sw_nid, "t": int(_sw_r["t"]),
                        "z": float(_sw_r["z"]), "y": float(_sw_r["y"]), "x": float(_sw_r["x"]),
                    }

                _sw_raw: list[dict] = []
                for _sw_r in _sw_g.edge_attrs().iter_rows(named=True):
                    _sw_ep = _sw_r.get("edge_prob") if hasattr(_sw_r, "get") else None
                    _sw_raw.append({
                        "source_id": int(_sw_r["source_id"]),
                        "target_id": int(_sw_r["target_id"]),
                        "edge_prob": None if _sw_ep is None else float(_sw_ep),
                    })

                _sw_nbid, _sw_e, _sw_st = filter_output_graph(
                    _sw_nbid, _sw_raw, dataset=_sw_ds,
                    deepcenter_bundle=DEEPCENTER_VETO_DETECTOR,
                )
                if not _sw_nbid:
                    raise AssertionError(f"{_sw_ds}: sweep post-processing removed every node")

                for _sw_nid in sorted(_sw_nbid):
                    _sw_n = _sw_nbid[_sw_nid]
                    _sw_w.writerow({
                        "id": _sw_rid, "dataset": _sw_ds, "row_type": "node",
                        "node_id": int(_sw_n["node_id"]), "t": int(_sw_n["t"]),
                        "z": max(0, int(round(float(_sw_n["z"])))),
                        "y": max(0, int(round(float(_sw_n["y"])))),
                        "x": max(0, int(round(float(_sw_n["x"])))),
                        "source_id": -1, "target_id": -1,
                    })
                    _sw_rid += 1
                    _sw_nodes += 1

                _sw_outdeg: dict[int, int] = {}
                for _sw_edge in _sw_e:
                    _sw_s = int(_sw_edge["source_id"])
                    _sw_t = int(_sw_edge["target_id"])
                    _sw_w.writerow({
                        "id": _sw_rid, "dataset": _sw_ds, "row_type": "edge",
                        "node_id": -1, "t": -1, "z": -1, "y": -1, "x": -1,
                        "source_id": _sw_s, "target_id": _sw_t,
                    })
                    _sw_rid += 1
                    _sw_edges += 1
                    _sw_outdeg[_sw_s] = _sw_outdeg.get(_sw_s, 0) + 1
                _sw_forks += sum(1 for _v in _sw_outdeg.values() if _v == 2)

        print(f"  penalty {_sw_pen:>7}: nodes {_sw_nodes:,}  edges {_sw_edges:,}  "
              f"forks {_sw_forks:,}  -> {_sw_out.name} "
              f"({_sw_out.stat().st_size / 1e6:.1f} MB)", flush=True)
        _sw_keep.append(_sw_out.name)
        _sw_summary.append({"penalty": _sw_pen, "nodes": _sw_nodes,
                            "edges": _sw_edges, "forks": _sw_forks,
                            "file": _sw_out.name})

    # restore the declared value so anything downstream sees the arm it was configured with
    globals()["RELINK_DIVISION_PENALTY"] = float(
        os.environ.get("BIOHUB_RELINK_DIVISION_PENALTY", "-1")
    )
    Path("/kaggle/working/relink_sweep_manifest.json").write_text(
        json.dumps({
            "penalties": _sw_pens,
            "control": "penalty -1 is the disabled sentinel and must match the primary "
                       "LOEO export exactly; if it does not, the sweep is invalid",
            "scoring": "score each sweep_pen_*.csv.gz offline vs data/train with "
                       "scripts/core/score_oof.py - zero submission slots",
            "summary": _sw_summary,
        }, indent=2)
    )
    _sw_keep.append("relink_sweep_manifest.json")
    globals()["_SWEEP_KEEP"] = _sw_keep
    print(f"relink sweep complete: {len(_sw_keep)} artifacts kept", flush=True)
else:
    globals()["_SWEEP_KEEP"] = []
