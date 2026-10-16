"""Extract one feature family over all official windows.

    python -m proxies.extract --family spectral [--limit 64]

Writes $FEATURES/<family>.npz with F (n_windows, n_features) float32 and
names; rows are in official.npz order, so every family aligns with the meta.
"""

import argparse
import os
import time

import numpy as np

from . import data, features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(features.FAMILIES))
    ap.add_argument("--data", default=os.environ.get("OFFICIAL"))
    ap.add_argument("--out", default=os.environ.get("FEATURES"))
    ap.add_argument("--n-jobs", type=int, default=int(os.environ.get("N_JOBS", 4)))
    ap.add_argument("--limit", type=int, help="first N windows only (smoke test)")
    a = ap.parse_args()

    w = data.load(a.data)
    if a.limit:
        w.X, w.P, w.meta = w.X[:a.limit], w.P[:a.limit], w.meta.iloc[:a.limit]
    t0 = time.time()
    F, names = features.get(a.family)(w, n_jobs=a.n_jobs)
    if F.shape != (len(w.X), len(names)):
        raise RuntimeError(f"{a.family}: F {F.shape} vs {len(names)} names")
    if len(set(names)) != len(names):
        raise RuntimeError(f"{a.family}: duplicate feature names")
    bad = ~np.isfinite(F)
    print(f"[extract] {a.family}: {F.shape} in {time.time() - t0:.0f} s, "
          f"non-finite {bad.mean():.2%}")
    suffix = f"_limit{a.limit}" if a.limit else ""
    out = os.path.join(a.out, f"{a.family}{suffix}.npz")
    np.savez_compressed(out, F=F, names=np.array(names))
    print(f"[extract] -> {out}")


if __name__ == "__main__":
    main()
