"""Re-price the remaining post-processing knobs on the CURRENT best chain (s09).

Everything in section 4 of docs/HANDOFF.md was priced on either the RAW ILP
graph or the RELINKED deployed one. The chain has moved twice since: relink off
(s05), gap2 after safe_div (s08), linefit 0.8 -> 0.3 (s09). Section 6 is the
record of what happens when a price measured on one topology is assumed to hold
on another -- so re-price rather than inherit.

Chain under test (s09):
    gap_close -> safe_div -> gap2 -> prune_isolated -> short_track -> linefit(0.3, 2)

One knob moves per row, everything else at its s09 value.

ABORT_RULES guards, enforced here rather than remembered:
  * `ratio` = n_pred/n_est is printed for every row and any row that moves it is
    flagged. The metric multiplier is UNCAPPED, so deleting nodes pays forever;
    a proxy gain that arrives with a moving ratio and flat J is the exploit, not
    a tracking gain. short_track is the known offender -- its multiplier gain is
    monotone to L=40.
  * One division event is 0.1/(dtp+dfp+dfn) of proxy. At 5/2/7 that is 0.00714.
    Nothing finer than that is reported as meaningful.
  * BIOHUB_GAP_CLOSE_UM is in the notebook's _EXPECTED_NUMERIC drift guard
    (asserts 5.0). If a gap_close row wins, shipping it needs the guard updated
    in the SAME edit or the kernel aborts at cell 3 -- make_env_variant.py will
    refuse it, by design.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_DOC = __doc__
_src = (ROOT / "scripts/91_other_stages.py").read_text()
exec(_src.split("\nALL = []")[0])

# s09 configuration
S = dict(gc_um=5.0, gc_reuse=3.2, gc_synth=True,
         g2_total=10.2, g2_step=4.4, g2_ctx=True, g2_frac=0.0045, g2_abs=180,
         prune=True, st_len=6, st_forks=True, lf_w=0.3, lf_win=2)


def SD(G, P):
    Q = dict(P); Q["t"] = G["t"]; Q["zyx"] = G["zyx"]; Q["edges"] = G["edges"]
    e, a = safe_div(Q)
    return dict(G, edges=e), len(a)


def chain(G, P, s, **over):
    c = {**S, **over}
    st = {}
    G, d = gap_close(G, max_um=c["gc_um"], reuse_um=c["gc_reuse"],
                     allow_synth=c["gc_synth"]); st.update(d)
    G, _ = SD(G, P)
    G, d = gap2(G, max_total=c["g2_total"], max_step=c["g2_step"],
                require_context=c["g2_ctx"], frac_cap=c["g2_frac"],
                abs_cap=c["g2_abs"]); st.update(d)
    if c["prune"]:
        G, d = prune_isolated(G); st.update(d)
    G, d = short_track(G, c["st_len"], c["st_forks"]); st.update(d)
    if c["lf_w"] > 0:
        G, d = linefit(G, w=c["lf_w"], window=c["lf_win"]); st.update(d)
    return G, st


print(_DOC.split("ABORT_RULES")[0].rstrip())
print("=" * 118)
print(HDR)
BASE = run(lambda G, P, s: chain(G, P, s), "s09 BASE")
show(BASE)
EVENT = 0.1 / max(BASE["dtp"] + BASE["dfp"] + BASE["dfn"], 1)
print(f"one division event at this ledger = {EVENT:.5f} proxy\n")

CASES = [
    ("gap_close um", [("gc_um", v) for v in (3.0, 4.0, 6.0, 8.0, 10.0, 12.0)]),
    ("gap_close synth off", [("gc_synth", False)]),
    ("gap_close reuse um", [("gc_reuse", v) for v in (2.8, 4.0)]),
    ("gap2 total um", [("g2_total", v) for v in (8.0, 12.0, 14.0)]),
    ("gap2 step um", [("g2_step", v) for v in (4.0, 5.0, 6.0)]),
    ("gap2 context gate off", [("g2_ctx", False)]),
    ("gap2 caps", [("g2_frac", 0.009), ("g2_abs", 360)]),
    ("prune isolated off", [("prune", False)]),
    ("short_track L", [("st_len", v) for v in (4, 8, 9)]),
    ("short_track keep_forks off", [("st_forks", False)]),
]
rows = []
for title, variants in CASES:
    print(f"--- {title} ---")
    for k, v in variants:
        r = run(lambda G, P, s, a=k, b=v: chain(G, P, s, **{a: b}), f"  {k}={v}")
        show(r, BASE)
        rows.append((k, v, r))
    print()

print("=" * 118)
print("SCREEN: proxy gain, is it real, and does it clear one division event?")
print(f"{'knob':<28}{'dproxy':>10}{'dJ':>10}{'dratio':>10}  verdict")
for k, v, r in sorted(rows, key=lambda x: -x[2]["proxy"]):
    dp = r["proxy"] - BASE["proxy"]
    dj = r["J"] - BASE["J"]
    dr = r["ratio"] - BASE["ratio"]
    if dp <= 0:
        verdict = "no gain"
    elif abs(dr) > 1e-9 and dj <= 1e-6:
        verdict = "NODE-COUNT EXPLOIT -- proxy up, J flat, ratio moved"
    elif dp < EVENT:
        verdict = f"below one division event ({EVENT:.5f})"
    else:
        verdict = "CANDIDATE"
    print(f"{f'{k}={v}':<28}{dp:>+10.5f}{dj:>+10.5f}{dr:>+10.5f}  {verdict}")
