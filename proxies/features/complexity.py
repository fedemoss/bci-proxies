"""Entropy and complexity per channel (antropy).

Permutation entropy at three delays (a cheap multiscale view), spectral, SVD
and sample entropy, Lempel-Ziv complexity of the median-binarised signal,
Higuchi / Katz / Petrosian fractal dimensions and the DFA exponent. Also
permutation entropy of the alpha- and beta-band signals: complexity of the
rhythm that ERD/S suppresses, rather than of the broadband mix.
"""

import antropy as ant
import numpy as np
from joblib import Parallel, delayed

from ..regions import CHANNELS
from .common import BANDS, bandpass

MEASURES = {
    "perm_d1": lambda x, fs: ant.perm_entropy(x, order=3, delay=1, normalize=True),
    "perm_d2": lambda x, fs: ant.perm_entropy(x, order=3, delay=2, normalize=True),
    "perm_d4": lambda x, fs: ant.perm_entropy(x, order=3, delay=4, normalize=True),
    "spectral_ent": lambda x, fs: ant.spectral_entropy(x, sf=fs, method="welch", nperseg=int(fs), normalize=True),
    "svd_ent": lambda x, fs: ant.svd_entropy(x, order=3, delay=1, normalize=True),
    "sample_ent": lambda x, fs: ant.sample_entropy(x, order=2),
    "lziv": lambda x, fs: ant.lziv_complexity((x > np.median(x)).astype(int), normalize=True),
    "higuchi_fd": lambda x, fs: ant.higuchi_fd(x, kmax=10),
    "katz_fd": lambda x, fs: ant.katz_fd(x),
    "petrosian_fd": lambda x, fs: ant.petrosian_fd(x),
    "dfa": lambda x, fs: ant.detrended_fluctuation(x),
}
BAND_PE = ["alpha_lo", "alpha_hi", "beta_lo", "beta_hi"]


def _one(x, xb, fs):
    """x (C, T) broadband, xb (B, C, T) band signals -> flat feature vector."""
    out = [[MEASURES[m](x[c], fs) for c in range(len(x))] for m in MEASURES]
    out += [[ant.perm_entropy(xb[b, c], order=3, delay=1, normalize=True)
             for c in range(len(x))] for b in range(len(xb))]
    return np.asarray(out, dtype=np.float64).ravel()


def extract(w, n_jobs=1):
    X = w.X.astype(np.float64)
    Xb = np.stack([bandpass(X, w.sfreq, *BANDS[b]) for b in BAND_PE], axis=1)
    F = Parallel(n_jobs=n_jobs, batch_size=16)(
        delayed(_one)(X[i], Xb[i], w.sfreq) for i in range(len(X)))
    names = [f"complexity:{m}:broad:ch:{c}" for m in MEASURES for c in CHANNELS]
    names += [f"complexity:perm_d1:{b}:ch:{c}" for b in BAND_PE for c in CHANNELS]
    return np.stack(F).astype(np.float32), names


def _coarse(x, s):
    n = len(x) // s
    return x[:n * s].reshape(n, s).mean(1)


def _multi(x, fs):
    out = [[ant.sample_entropy(_coarse(x[c], s), order=2) for c in range(len(x))]
           for s in MSE_SCALES]
    out += [[ant.app_entropy(x[c], order=2) for c in range(len(x))]]
    out += [[ant.perm_entropy(_coarse(x[c], s), order=4, delay=1, normalize=True)
             for c in range(len(x))] for s in MSE_SCALES]
    return np.asarray(out, dtype=np.float64).ravel()


MSE_SCALES = [2, 3, 4, 5]


def extract_multiscale(w, n_jobs=1):
    """Multiscale sample entropy and order-4 permutation entropy on coarse-
    grained signals (Costa 2002) at scales 2-5, and approximate entropy."""
    X = w.X.astype(np.float64)
    F = Parallel(n_jobs=n_jobs, batch_size=16)(delayed(_multi)(X[i], w.sfreq) for i in range(len(X)))
    names = [f"complexity_ms:mse:s{s}:ch:{c}" for s in MSE_SCALES for c in CHANNELS]
    names += [f"complexity_ms:app_ent:broad:ch:{c}" for c in CHANNELS]
    names += [f"complexity_ms:perm4:s{s}:ch:{c}" for s in MSE_SCALES for c in CHANNELS]
    return np.stack(F).astype(np.float32), names
