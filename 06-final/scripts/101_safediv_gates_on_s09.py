"""Re-price the safe_div GATES on the s09 chain. Division axis.

scripts/100 re-priced gap_close / gap2 / prune / short_track on this chain and
found nothing. It did not touch the safe_div gates, and those are the axis that
matters: the offline-vs-board ledger in section 7 is 3 for 3 on divisions and
0 for 3 on edges.

Re-pricing them is not redundant with section 4's "gates are at a local
optimum", because safe_div's INPUT has changed. Deployed it saw
raw + relink + gap_close + gap2; on the s09 chain it sees raw + gap_close. The
orphan pool it draws daughters from is a different set, and the orphan pool is
the whole mechanism.

Section 5: an autopsy of all 12 divisions found EXACTLY ONE blocked by a gate,
and it misses by 0.35 um. So the steps here are deliberately fine near the
deployed values -- a coarse sweep would step straight over it. Deployed gates:

    parent_max 9.0   sister_max 14.0   child_max 10.0
    tau 0.6          diverge 2.25      frame_cap 0.0076   glob_cap 0.00375

Section 4's warning stands and is what the FP column is for: loosening tau or
diverge floods false positives and divJ FALLS even as TP rises. A row only
counts if divJ goes UP.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_DOC = __doc__
_src = (ROOT / "scripts/91_other_stages.py").read_text()
exec(_src.split("\nALL = []")[0])

S = dict(gc_um=5.0, gc_reuse=3.2, g2_total=10.2, g2_step=4.4,
         st_len=6, st_forks=True, lf_w=0.3, lf_win=2)
GATES = dict(parent_max=9.0, sister_max=14.0, child_max=10.0, tau=0.6,
             diverge=2.25, frame_cap=0.0076, glob_cap=0.00375)


def chain(G, P, s, **gate_over):
    g = {**GATES, **gate_over}
    st = {}
    G, d = gap_close(G, max_um=S["gc_um"], reuse_um=S["gc_reuse"], allow_synth=True)
    st.update(d)
    Q = dict(P); Q["t"] = G["t"]; Q["zyx"] = G["zyx"]; Q["edges"] = G["edges"]
    e, added = safe_div(Q, **g)
    G = dict(G, edges=e); st["sd_added"] = len(added)
    G, d = gap2(G, max_total=S["g2_total"], max_step=S["g2_step"]); st.update(d)
    G, d = prune_isolated(G); st.update(d)
    G, d = short_track(G, S["st_len"], S["st_forks"]); st.update(d)
    G, d = linefit(G, w=S["lf_w"], window=S["lf_win"]); st.update(d)
    return G, st


print(_DOC.split("Section 4's warning")[0].rstrip())
print("=" * 118)
print(HDR)
BASE = run(lambda G, P, s: chain(G, P, s), "s09 BASE (deployed gates)")
show(BASE)
print()

CASES = []
# fine steps up from the deployed value -- the gate-blocked division misses by 0.35um
CASES += [("parent_max", v) for v in (9.25, 9.5, 10.0, 11.0, 12.0, 8.5)]
CASES += [("sister_max", v) for v in (14.5, 15.0, 16.0, 13.0)]
CASES += [("child_max", v) for v in (10.5, 11.0, 12.0, 9.0)]
CASES += [("tau", v) for v in (0.65, 0.7, 0.8, 1.0, 0.5)]
CASES += [("diverge", v) for v in (2.0, 1.75, 1.5, 2.5, 3.0)]
CASES += [("frame_cap", v) for v in (0.0100, 0.0150)]
CASES += [("glob_cap", v) for v in (0.0050, 0.0075)]

rows = []
cur = None
for k, v in CASES:
    if k != cur:
        print(f"--- {k} (deployed {GATES[k]}) ---")
        cur = k
    r = run(lambda G, P, s, a=k, b=v: chain(G, P, s, **{a: b}), f"  {k}={v}")
    show(r, BASE)
    rows.append((k, v, r))
print()

print("=" * 118)
print("SCREEN: a gate change only counts if divJ RISES. Section 4: loosening")
print("tau/diverge raises TP while divJ FALLS, because FP rises faster.")
print(f"{'gate':<22}{'dproxy':>10}{'ddivJ':>9}{'TP/FP/FN':>11}{'dratio':>9}  verdict")
b = f"{BASE['dtp']}/{BASE['dfp']}/{BASE['dfn']}"
print(f"{'(deployed)':<22}{0.0:>+10.5f}{0.0:>+9.4f}{b:>11}{0.0:>+9.5f}  base")
for k, v, r in sorted(rows, key=lambda x: -x[2]["divJ"]):
    dp = r["proxy"] - BASE["proxy"]
    dd = r["divJ"] - BASE["divJ"]
    dr = r["ratio"] - BASE["ratio"]
    led = f"{r['dtp']}/{r['dfp']}/{r['dfn']}"
    if dd > 1e-9 and dp > 0:
        verdict = "CANDIDATE -- divJ up and proxy up"
    elif dd > 1e-9:
        verdict = "divJ up but proxy down"
    elif r["dtp"] > BASE["dtp"] and dd <= 1e-9:
        verdict = "TP up but divJ flat/down -- FP flood, the section 4 trap"
    elif dd < -1e-9:
        verdict = "divJ down"
    else:
        verdict = "no division change"
    print(f"{f'{k}={v}':<22}{dp:>+10.5f}{dd:>+9.4f}{led:>11}{dr:>+9.5f}  {verdict}")
