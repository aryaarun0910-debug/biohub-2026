"""H1-M2 temporal-profile extraction for ZSNS004 (Kaggle, internet on, CPU).

Streams Zebrahub level-0 OME-Zarr (CC BY 4.0, Lange et al., Cell 2024,
doi:10.1016/j.cell.2024.09.047) and writes one row per (event, dt) with the frozen
h1m mother-centred node statistics, dt in [-4, +10] around the tracker's fork frame.
Nothing is redistributed; only derived statistics leave the kernel.
"""
import glob, os, shutil, subprocess, sys, time

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "zarr", "numcodecs"], check=True)

CAND = sorted(glob.glob("/kaggle/input/*"))
print("INPUT MOUNTS:", CAND, flush=True)
IN = None
for d in CAND:
    if os.path.exists(os.path.join(d, "h1m2_profile.py")):
        IN = d
        break
if IN is None:
    hits = glob.glob("/kaggle/input/**/h1m2_profile.py", recursive=True)
    print("DEEP HITS:", hits, flush=True)
    IN = os.path.dirname(hits[0])
print("USING", IN, os.listdir(IN), flush=True)

WORK = "/kaggle/working"
for f in ("h1i_node_appearance.py", "h1m_features.py", "h1m2_profile.py"):
    shutil.copy(os.path.join(IN, f), os.path.join(WORK, f))
sys.path.insert(0, WORK)

t0 = time.time()
sys.argv = ["h1m2_profile.py",
            "--embryo", "ZSNS004",
            "--forks", os.path.join(IN, "realigned_forks.parquet"),
            "--tracks", os.path.join(IN, "ZSNS004.parquet"),
            "--frame-lo", "285", "--frame-hi", "318",
            "--dt-lo", "-4", "--dt-hi", "10",
            "--z-slabs", "0", "--n-neg", "3000",
            "--out", os.path.join(WORK, "prof_ZSNS004.parquet")]
import h1m2_profile
h1m2_profile.main()
print("TOTAL_SECONDS", round(time.time() - t0, 1))
for f in ("h1i_node_appearance.py", "h1m_features.py", "h1m2_profile.py"):
    os.remove(os.path.join(WORK, f))
