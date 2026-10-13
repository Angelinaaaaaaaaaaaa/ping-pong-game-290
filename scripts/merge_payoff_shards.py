"""Merge sharded payoff matrices into one."""
import argparse, glob
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("shards", nargs="+")
ap.add_argument("--out", default="logs/AD/payoff.npz")
ap.add_argument("--fill", default=None,
                help="npz of cells replayed separately, e.g. all-truncated pairs")
a = ap.parse_args()

files = sorted(f for pat in a.shards for f in glob.glob(pat))
A = None
trunc = 0
for f in files:
    z = np.load(f)
    if A is None:
        A = np.full_like(z["A"], np.nan)
        np.fill_diagonal(A, 0.5)
    m = ~np.isnan(z["A"]) & ~np.eye(A.shape[0], dtype=bool)
    A[m] = z["A"][m]
    trunc += int(z["trunc_total"])
    rallies = int(z["rallies"])
if a.fill:
    z = np.load(a.fill)
    for (i, j), v in zip(z["keys"], z["vals"]):
        A[i, j] = v
        A[j, i] = 1.0 - v
    print(f"filled {len(z['vals'])} replayed pairs from {a.fill}")
missing = int(np.isnan(A).sum())
print(f"merged {len(files)} shards, {missing} cells still missing, {trunc} truncated rallies")
if missing:
    raise SystemExit("incomplete: some cells were never played")
np.savez(a.out, A=A, rallies=rallies, trunc_total=trunc)
print(f"wrote {a.out}")
