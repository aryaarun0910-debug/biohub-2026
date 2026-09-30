"""Re-price EVERY post-processing knob on the 4 films the board actually scores.

scripts/100 and 101 re-priced the chain and the safe_div gates and found nothing
-- but both ran on the 8 VALIDATOR films. scripts/102-103 then showed those films
disagree with the scored films: s08 is +0.00748 there and +0.00000 here, s09 is
+0.00487 there and -0.00086 here.

So the "parametric search is closed" conclusion was established on the wrong set.
Re-establish it on the right one.

One contrast is newly interesting here and was never worth asking before. On the
scored films the division ledger is 0 TP / 6 FP / 3 FN, so divJ = 0 in EVERY arm.
safe_div therefore buys no division credit at all on these films while still
adding edges and competing for orphans. Turning it off costs 0.019 on the board
historically (OUTPUT_SAFE_DIVISIONS=0 scored 0.906 against ~0.925) -- but that
was measured with motion relink ON, on a different topology, which is precisely
the setup section 6 says does not transfer.

Baseline here is s05: relink off, gap2 before safe_div, linefit 0.8 -- the
configuration that is actually submitted and that scripts/103 puts at +0.02707
on these films.

Parallel over (config x film) on 16 workers; one 4-film config is ~3 s serial.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import csv
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_SRC = (ROOT / "scripts/91_other_stages.py").read_text().split("\nALL = []")[0]
exec(_SRC)

TEST_PRED = ROOT / ("artifacts/s05_output/tracking_repo/predictions/unknown/"
                    "unet_transformer/split_0")
SCORED = ["44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"]
TDATA = {s: (load_pred(TEST_PRED / f"{s}.geff"), load_gt(s)) for s in SCORED}

S0 = dict(gc_um=5.0, gc_reuse=3.2, gc_on=True, g2_on=True, g2_after=False,
          g2_total=10.2, g2_step=4.4, sd_on=True, prune=True, st_on=True,
          st_len=6, st_forks=True, lf_w=0.8, lf_win=2)


def chain(G, P, c):
    if c["gc_on"]:
        G, _ = gap_close(G, max_um=c["gc_um"], reuse_um=c["gc_reuse"], allow_synth=True)
    if c["g2_on"] and not c["g2_after"]:
        G, _ = gap2(G, max_total=c["g2_total"], max_step=c["g2_step"])
    if c["sd_on"]:
        Q = dict(P); Q["t"] = G["t"]; Q["zyx"] = G["zyx"]; Q["edges"] = G["edges"]
        G = dict(G, edges=safe_div(Q)[0])
    if c["g2_on"] and c["g2_after"]:
        G, _ = gap2(G, max_total=c["g2_total"], max_step=c["g2_step"])
    if c["prune"]:
        G, _ = prune_isolated(G)
    if c["st_on"]:
        G, _ = short_track(G, c["st_len"], c["st_forks"])
    if c["lf_w"] > 0:
        G, _ = linefit(G, w=c["lf_w"], window=c["lf_win"])
    return G


def score_one(job):
    label, over = job
    c = {**S0, **over}
    rows = []
    for stem, (P, GT) in TDATA.items():
        G = chain(G_of(P), P, c)
        rows.append(M2.score(G["t"], G["zyx"], G["edges"],
                             GT["t"], GT["zyx"], GT["edges"], GT["n_est"]))
    r = M2.aggregate(rows)
    r["label"] = label
    r["ratio"] = sum(x["n_pred"] for x in rows) / sum(x["n_est"] for x in rows)
    r["per_film"] = [(s, x["adj"] + 0.1 * x.get("divJ", 0.0), x["weight"])
                     for s, x in zip(SCORED, rows)]
    return r


JOBS = [("s05 BASE", {})]
JOBS += [("STAGE OFF: safe_div", {"sd_on": False}),
         ("STAGE OFF: gap2", {"g2_on": False}),
         ("STAGE OFF: gap_close", {"gc_on": False}),
         ("STAGE OFF: short_track", {"st_on": False}),
         ("STAGE OFF: prune", {"prune": False}),
         ("STAGE OFF: linefit", {"lf_w": 0.0}),
         ("ORDER: gap2 after safe_div (s08)", {"g2_after": True})]
JOBS += [(f"gc_um={v}", {"gc_um": v}) for v in (3.0, 4.0, 6.0, 8.0, 10.0, 12.0)]
JOBS += [(f"g2_total={v}", {"g2_total": v}) for v in (8.0, 12.0, 14.0)]
JOBS += [(f"g2_step={v}", {"g2_step": v}) for v in (4.0, 5.0, 6.0)]
JOBS += [(f"st_len={v}", {"st_len": v}) for v in (4, 8, 9, 12)]
JOBS += [("st_forks=0", {"st_forks": False})]
JOBS += [(f"lf_w={v}", {"lf_w": v}) for v in (0.2, 0.3, 0.4, 0.5, 0.6, 1.0)]
JOBS += [(f"lf_win={v}", {"lf_win": v}) for v in (1, 3, 4)]
JOBS += [("s08+lf0.3 (= s09)", {"g2_after": True, "lf_w": 0.3})]


def main():
    print(__doc__.split("Parallel over")[0].rstrip())
    print("=" * 112)
    with ProcessPoolExecutor(max_workers=16) as ex:
        res = list(ex.map(score_one, JOBS, chunksize=1))
    base = res[0]
    EVENT = 0.1 / max(base["dtp"] + base["dfp"] + base["dfn"], 1)
    print(f"s05 BASE on the 4 SCORED films: proxy {base['proxy']:.5f}  "
          f"J {base['J']:.5f}  divJ {base['divJ']:.4f}  "
          f"{base['dtp']}/{base['dfp']}/{base['dfn']}  ratio {base['ratio']:.4f}")
    print(f"one division event at this ledger = {EVENT:.5f}\n")
    print(f"{'knob':<36}{'dproxy':>10}{'dJ':>10}{'dratio':>10}{'div':>9}  verdict")
    for r in sorted(res[1:], key=lambda x: -x["proxy"]):
        dp = r["proxy"] - base["proxy"]
        dj = r["J"] - base["J"]
        dr = r["ratio"] - base["ratio"]
        led = f"{r['dtp']}/{r['dfp']}/{r['dfn']}"
        if dp <= 1e-5:
            v = "no gain"
        elif abs(dr) > 1e-9 and dj <= 1e-6:
            v = "NODE-COUNT EXPLOIT (J flat/down, ratio moved)"
        elif dp < EVENT:
            v = f"gain below one division event"
        else:
            v = "*** CANDIDATE ***"
        print(f"{r['label']:<36}{dp:>+10.5f}{dj:>+10.5f}{dr:>+10.5f}{led:>9}  {v}")

    print(f"\n--- per-film for the top 5 (weights: "
          + ", ".join(f"{s}={w}" for s, _, w in base['per_film']) + ") ---")
    for r in sorted(res[1:], key=lambda x: -x["proxy"])[:5]:
        b = {s: p for s, p, _ in base["per_film"]}
        d = "  ".join(f"{s.split('_')[1][:6]}:{p - b[s]:+.5f}" for s, p, _ in r["per_film"])
        print(f"{r['label']:<36}{d}")

    out = ROOT / "artifacts/reprice_on_scored.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "proxy", "dproxy", "J", "ratio", "divJ", "dtp", "dfp", "dfn"])
        for r in res:
            w.writerow([r["label"], f"{r['proxy']:.6f}",
                        f"{r['proxy'] - base['proxy']:+.6f}", f"{r['J']:.6f}",
                        f"{r['ratio']:.4f}", f"{r['divJ']:.4f}",
                        r["dtp"], r["dfp"], r["dfn"]])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
