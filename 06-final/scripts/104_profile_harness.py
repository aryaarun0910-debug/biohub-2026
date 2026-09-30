"""Where does harness time actually go, and what is worth parallelising?"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
t0 = time.time()
exec((ROOT / "scripts/91_other_stages.py").read_text().split("\nALL = []")[0])
t_load = time.time() - t0
print(f"import + load 8 films : {t_load:6.2f}s")

def SD(G, P):
    Q = dict(P); Q["t"] = G["t"]; Q["zyx"] = G["zyx"]; Q["edges"] = G["edges"]
    return dict(G, edges=safe_div(Q)[0])

STAGES = ["gap_close", "safe_div", "gap2", "prune", "short_track", "linefit"]

def chain_timed(G, P):
    ts = []
    a = time.time(); G, _ = gap_close(G, max_um=5.0, reuse_um=3.2, allow_synth=True); ts.append(time.time()-a)
    a = time.time(); G = SD(G, P);                                                     ts.append(time.time()-a)
    a = time.time(); G, _ = gap2(G, max_total=10.2, max_step=4.4);                     ts.append(time.time()-a)
    a = time.time(); G, _ = prune_isolated(G);                                         ts.append(time.time()-a)
    a = time.time(); G, _ = short_track(G, 6, True);                                   ts.append(time.time()-a)
    a = time.time(); G, _ = linefit(G, w=0.3, window=2);                               ts.append(time.time()-a)
    return G, ts

agg = [0.0]*6; tot_score = 0.0
print(f"\n{'film':<18}{'nodes':>8}{'chain s':>9}{'score s':>9}")
for stem, (P, GT) in DATA.items():
    a = time.time(); G, ts = chain_timed(G_of(P), P); b = time.time()
    M2.score(G["t"], G["zyx"], G["edges"], GT["t"], GT["zyx"], GT["edges"], GT["n_est"])
    c = time.time()
    agg = [x+y for x, y in zip(agg, ts)]; tot_score += c-b
    print(f"{stem:<18}{len(P['t']):>8}{b-a:>9.3f}{c-b:>9.3f}")

tot_chain = sum(agg)
print(f"\n{'stage':<14}{'seconds':>9}{'% of chain':>12}")
for n, v in sorted(zip(STAGES, agg), key=lambda x: -x[1]):
    print(f"{n:<14}{v:>9.3f}{100*v/tot_chain:>11.1f}%")
print(f"{'--score--':<14}{tot_score:>9.3f}")

unit = tot_chain + tot_score
print(f"\none full 8-film config : {unit:.2f}s")
for n in (24, 28, 100, 1000):
    print(f"  {n:>4} configs serial : {n*unit:>7.0f}s"
          f"   on 16 workers ~{n*unit/16 + t_load:>5.0f}s (incl warmup)")
