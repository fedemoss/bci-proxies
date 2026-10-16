"""Higher-order interactions between brain regions (Gaussian-copula estimates).

For a set S of region signals within one window, after copula-normalising each
signal (rank -> standard normal), the Gaussian entropy is

    H(S) = 1/2 log det(2 pi e Sigma_S)

and, for n variables X = (X_1..X_n),

    TC   = sum_i H(X_i) - H(X)                 total correlation
    DTC  = H(X) - sum_i H(X_i | X_-i)          dual total correlation
    O    = TC - DTC     (> 0 redundancy-dominated, < 0 synergy-dominated)
    S    = TC + DTC

(Rosas et al. 2019). Computed per band on the region envelopes ("env") and on
the band-limited region signals ("sig"): O-information of every region
triplet (84) and the four whole-system quantities over all 9 regions. Samples
within a window are autocorrelated, so these are descriptive statistics of
the window, not unbiased information estimates.
"""

from itertools import combinations

import numpy as np
from scipy.special import ndtri

from ..regions import REGION_NAMES
from .common import BANDS, analytic, to_regions

TRIPLETS = [list(t) for t in combinations(range(len(REGION_NAMES)), 3)]


def copnorm(x):
    """Rank-transform the last axis to standard normal."""
    r = np.argsort(np.argsort(x, -1), -1)
    return ndtri((r + 1) / (x.shape[-1] + 1))


def _logdet(S):
    return np.linalg.slogdet(S)[1]


def _h(S):
    """Gaussian entropy (nats) of covariance S (..., k, k)."""
    k = S.shape[-1]
    return 0.5 * (_logdet(S) + k * np.log(2 * np.pi * np.e))


def measures(Sig):
    """Sig (n, k, k) -> TC, DTC, O, S, each (n,)."""
    k = Sig.shape[-1]
    H = _h(Sig)
    Hi = 0.5 * np.log(2 * np.pi * np.e * np.diagonal(Sig, axis1=-2, axis2=-1)).sum(-1)
    Hmi = sum(_h(np.delete(np.delete(Sig, i, -1), i, -2)) for i in range(k))
    tc = Hi - H
    dtc = H - (k * H - Hmi)          # sum_i H(X_i|X_-i) = sum_i [H - H(X_-i)]
    return tc, dtc, tc - dtc, tc + dtc


def extract(w, n_jobs=1):
    X = w.X.astype(np.float64)
    cols, names = [], []
    for b, r in BANDS.items():
        z = to_regions(analytic(X, w.sfreq, *r))
        for kind, s in (("env", np.abs(z)), ("sig", z.real)):
            g = copnorm(s)
            Sig = np.einsum("nit,njt->nij", g, g) / g.shape[-1]
            for nm, v in zip(["tc", "dtc", "oinfo", "sinfo"], measures(Sig)):
                cols.append(v[:, None]); names.append(f"hoi:{nm}_{kind}:{b}:reg:all")
            for t in TRIPLETS:
                cols.append(measures(Sig[:, t][:, :, t])[2][:, None])
                names.append(f"hoi:oinfo_{kind}:{b}:reg:"
                             + "|".join(REGION_NAMES[i] for i in t))
    return np.concatenate(cols, 1).astype(np.float32), names
