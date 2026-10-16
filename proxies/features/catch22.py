"""catch22 (Lubba et al. 2019): the 22 hctsa features that best summarise
time-series classification across a large benchmark, chosen from >7 700.

Computed per channel on the broadband window and on each band-limited signal,
so every band's dynamics get the whole canonical set (22 x 8 x 41 = 7 216).
"""

import numpy as np
import pycatch22
from joblib import Parallel, delayed

from ..regions import CHANNELS
from .common import BANDS, bandpass

SIGNALS = ["broad"] + list(BANDS)


def _one(xs):
    """xs (S, C, T) -> (S * 22 * C,), ordered signal, feature, channel."""
    out = np.empty((xs.shape[0], 22, xs.shape[1]))
    for s in range(xs.shape[0]):
        for c in range(xs.shape[1]):
            out[s, :, c] = pycatch22.catch22_all(xs[s, c].tolist())["values"]
    return out.ravel()


def extract(w, n_jobs=1):
    X = w.X.astype(np.float64)
    Xs = np.stack([X] + [bandpass(X, w.sfreq, *BANDS[b]) for b in BANDS], axis=1)
    F = Parallel(n_jobs=n_jobs, batch_size=4)(delayed(_one)(Xs[i]) for i in range(len(X)))
    fnames = pycatch22.catch22_all(list(np.random.default_rng(0).standard_normal(100)))["names"]
    names = [f"catch22:{f}:{s}:ch:{c}" for s in SIGNALS for f in fnames for c in CHANNELS]
    return np.stack(F).astype(np.float32), names
