"""Weighted symbolic mutual information (King et al. 2013; Sitt et al. 2014).

Each signal becomes a sequence of ordinal patterns (order 3: 6 symbols) at
delay tau. For channel pair (a, b), from the joint symbol distribution p(x, y):

    wSMI = 1/log(6) sum_{x,y} w(x,y) p(x,y) log[p(x,y) / (p(x) p(y))]

with w = 0 for identical and opposite-sign symbol pairs, which discounts the
zero-lag coupling that volume conduction produces. tau = 1, 2, 4 samples
(~40, 20, 10 Hz at 120 Hz). Region pairs, channel node strength and global mean.
"""

from itertools import permutations

import numpy as np

from ..regions import CHANNELS, REGION_NAMES
from .common import pair_names, to_regions

PATTERNS = list(permutations(range(3)))
TAUS = [1, 2, 4]
_W = np.ones((6, 6))
for i, p in enumerate(PATTERNS):
    for j, q in enumerate(PATTERNS):
        # sign-flipped signal: argsort(-x) = argsort(x) reversed
        if p == q or p == q[::-1]:
            _W[i, j] = 0.0


def symbols(X, tau):
    """X (n, C, T) -> (n, C, T') int symbols 0..5."""
    T = X.shape[-1] - 2 * tau
    emb = np.stack([X[..., i * tau:i * tau + T] for i in range(3)], -1)
    rank = np.argsort(emb, -1)
    code = rank[..., 0] * 9 + rank[..., 1] * 3 + rank[..., 2]
    lut = np.full(27, -1)
    for i, p in enumerate(PATTERNS):
        lut[p[0] * 9 + p[1] * 3 + p[2]] = i
    return lut[code]


def wsmi(S):
    """S (n, C, T) symbols -> (n, C, C) wSMI."""
    oh = np.eye(6, dtype=np.float32)[S]                      # (n, C, T, 6)
    T = S.shape[-1]
    pxy = np.einsum("ncta,ndtb->ncdab", oh, oh) / T
    px = oh.mean(2)
    ratio = pxy / (px[:, :, None, :, None] * px[:, None, :, None, :] + 1e-20)
    return (_W * pxy * np.log(ratio + 1e-20)).sum((-1, -2)) / np.log(6)


def extract(w, n_jobs=1, chunk=64):
    X = w.X.astype(np.float64)
    ci, cj = np.triu_indices(len(CHANNELS), 1)
    ri, rj = np.triu_indices(len(REGION_NAMES), 1)
    cols, names = [], []
    rp, _ = pair_names(REGION_NAMES)
    for tau in TAUS:
        reg, strength, glob = [], [], []
        for i in range(0, len(X), chunk):
            Mc = wsmi(symbols(X[i:i + chunk], tau))
            Mr = wsmi(symbols(to_regions(X[i:i + chunk]), tau))
            reg.append(Mr[:, ri, rj])
            strength.append((Mc.sum(-1) - np.diagonal(Mc, axis1=1, axis2=2)) / (len(CHANNELS) - 1))
            glob.append(Mc[:, ci, cj].mean(-1, keepdims=True))
        cols += [np.concatenate(reg), np.concatenate(strength), np.concatenate(glob)]
        names += [f"wsmi:wsmi:tau{tau}:reg:{p}" for p in rp]
        names += [f"wsmi:wsmi_strength:tau{tau}:ch:{c}" for c in CHANNELS]
        names += [f"wsmi:wsmi_global:tau{tau}:all:all"]
    return np.concatenate(cols, 1).astype(np.float32), names
