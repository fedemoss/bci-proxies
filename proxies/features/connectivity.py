"""Single-trial functional connectivity per band.

Four measures from the band-limited analytic signal z = A e^{i phi}, each over
the samples of one window:

    PLV       |<e^{i(phi_a - phi_b)}>|                      phase locking
    wPLI      |<Im S_ab>| / <|Im S_ab|>,  S_ab = z_a z_b*   lag-only phase
    AEC       corr(A_a, A_b)                                 envelope coupling
    AEC-orth  corr(A_a, |Im(z_b z_a*)| / A_a), symmetrised   leakage-corrected

wPLI and AEC-orth discard zero-lag coupling, i.e. volume conduction (Vinck
2011; Hipp 2012). Two families:

    conn        region-pair values (9 regions -> 36 pairs), each channel's
                node strength (mean over its 40 links) and weighted clustering
                coefficient, the global mean, mean clustering and the spread
                of node strengths
    conn_pairs  every channel pair only (820 per band and measure: 22 960)
"""

import numpy as np
from joblib import Parallel, delayed

from ..regions import CHANNELS, REGION_NAMES
from .common import BANDS, analytic, pair_names, to_regions

METRICS = ["plv", "wpli", "aec", "aecorth"]


def _corr(a, b):
    a = a - a.mean(-1, keepdims=True); b = b - b.mean(-1, keepdims=True)
    return (a * b).sum(-1) / np.sqrt((a * a).sum(-1) * (b * b).sum(-1) + 1e-20)


def matrices(z):
    """z (n, C, T) complex -> dict metric -> (n, C, C)."""
    T = z.shape[-1]
    A = np.abs(z) + 1e-12
    u = z / A
    out = {"plv": np.abs(np.einsum("nct,ndt->ncd", u, u.conj())) / T}
    im = np.einsum("nct,ndt->ncdt", z, z.conj()).imag          # Im(z_a z_b*)
    out["wpli"] = np.abs(im.mean(-1)) / (np.abs(im).mean(-1) + 1e-20)
    out["aec"] = _corr(A[:, :, None], A[:, None, :])
    # orthogonalised envelope of b w.r.t. a: |Im(z_b z_a*)| / A_a = |im[a,b]| / A_a
    orth = _corr(A[:, :, None], np.abs(im) / A[:, :, None])
    out["aecorth"] = 0.5 * (orth + orth.transpose(0, 2, 1))
    return out


def clustering(W):
    """Weighted clustering coefficient per node (Onnela 2005) of a dense
    non-negative matrix: (W^(1/3))^3 diagonal / (k (k - 1)), weights / max."""
    W = np.abs(W).astype(np.float64)
    idx = np.arange(W.shape[1]); W = W.copy(); W[:, idx, idx] = 0
    W /= W.max((1, 2), keepdims=True) + 1e-20
    R = np.cbrt(W)
    k = W.shape[1] - 1
    return np.einsum("nij,njk,nki->ni", R, R, R) / (k * (k - 1))


def _chunk(Xc, sfreq, pairs):
    zs = {b: analytic(Xc, sfreq, *r).astype(np.complex64) for b, r in BANDS.items()}
    ci, cj = np.triu_indices(len(CHANNELS), 1)
    ri, rj = np.triu_indices(len(REGION_NAMES), 1)
    cols = []
    for b, z in zs.items():
        Mc = matrices(z)
        Mr = None if pairs else matrices(to_regions(z))
        for m in METRICS:
            if pairs:                 # conn_pairs holds the pairs only
                cols.append(Mc[m][:, ci, cj])
                continue
            C = Mc[m].copy()
            idx = np.arange(C.shape[1]); C[:, idx, idx] = np.nan
            cols.append(Mr[m][:, ri, rj])                      # region pairs
            cols.append(np.nanmean(C, -1))                     # node strength
            cols.append(Mc[m][:, ci, cj].mean(-1, keepdims=True))  # global
            clu = clustering(Mc[m])
            cols += [clu, clu.mean(-1, keepdims=True),
                     np.nanstd(np.nanmean(C, -1), -1, keepdims=True)]  # strength heterogeneity
    return np.concatenate(cols, axis=1)


def _names(pairs):
    rp, _ = pair_names(REGION_NAMES)
    cp, _ = pair_names(CHANNELS)
    names = []
    for b in BANDS:
        for m in METRICS:
            if pairs:
                names += [f"conn_pairs:{m}:{b}:ch:{p}" for p in cp]
                continue
            names += [f"conn:{m}:{b}:reg:{p}" for p in rp]
            names += [f"conn:{m}_strength:{b}:ch:{c}" for c in CHANNELS]
            names += [f"conn:{m}_global:{b}:all:all"]
            names += [f"conn:{m}_clustering:{b}:ch:{c}" for c in CHANNELS]
            names += [f"conn:{m}_clustering_global:{b}:all:all",
                      f"conn:{m}_strength_sd:{b}:all:all"]
    return names


def extract(w, n_jobs=1, pairs=False, chunk=16):
    X = w.X.astype(np.float64)
    parts = Parallel(n_jobs=n_jobs)(
        delayed(_chunk)(X[i:i + chunk], w.sfreq, pairs) for i in range(0, len(X), chunk))
    return np.concatenate(parts).astype(np.float32), _names(pairs)


def extract_pairs(w, n_jobs=1):
    return extract(w, n_jobs, pairs=True)
