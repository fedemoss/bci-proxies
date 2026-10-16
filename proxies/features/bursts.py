"""Band-envelope dynamics and bursts per channel.

Beyond mean power: how the rhythm fluctuates. For every band the Hilbert
envelope A(t) gives log mean, coefficient of variation and skewness, and
bursts -- runs where A exceeds 1.5x its window median (a window-relative
threshold, so nothing outside the window is needed): rate (per s), mean
duration (s) and fraction of time in a burst. Transient beta bursts rather
than sustained power carry the sensorimotor signal in single trials
(Shin et al. 2017; Little et al. 2019).
"""

import numpy as np
from scipy.stats import skew

from ..regions import CHANNELS
from .common import BANDS, analytic

MEASURES = ["env_logmean", "env_cv", "env_skew", "burst_rate", "burst_dur", "burst_frac"]


def _bursts(A, sfreq, k=1.5):
    on = A > k * np.median(A, -1, keepdims=True)
    edges = np.diff(on.astype(np.int8), axis=-1)
    starts = (edges == 1).sum(-1) + on[..., 0]
    T = A.shape[-1] / sfreq
    frac = on.mean(-1)
    rate = starts / T
    dur = np.where(starts > 0, frac * T / np.maximum(starts, 1), 0.0)
    return rate, dur, frac


def extract(w, n_jobs=1):
    X = w.X.astype(np.float64)
    cols, names = [], []
    for b, r in BANDS.items():
        A = np.abs(analytic(X, w.sfreq, *r))
        m = A.mean(-1)
        rate, dur, frac = _bursts(A, w.sfreq)
        for nm, v in zip(MEASURES, [np.log(m + 1e-12), A.std(-1) / m, skew(A, axis=-1),
                                    rate, dur, frac]):
            cols.append(v); names += [f"bursts:{nm}:{b}:ch:{c}" for c in CHANNELS]
    return np.concatenate(cols, 1).astype(np.float32), names
