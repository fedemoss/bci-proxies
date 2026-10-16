"""Time-domain statistics per channel: amplitude, Hjorth parameters, shape."""

import numpy as np
from scipy.stats import kurtosis, skew

from ..regions import CHANNELS


def extract(w, n_jobs=1):
    X = w.X.astype(np.float64)
    d1 = np.diff(X, axis=-1)
    d2 = np.diff(d1, axis=-1)
    v0, v1, v2 = X.var(-1), d1.var(-1), d2.var(-1)
    mobility = np.sqrt(v1 / v0)
    feats = {
        "logvar": np.log(v0 + 1e-12),
        "linelength": np.abs(d1).mean(-1),
        "hjorth_mobility": mobility,
        "hjorth_complexity": np.sqrt(v2 / v1) / mobility,
        "skew": skew(X, axis=-1),
        "kurtosis": kurtosis(X, axis=-1),
        "zerocross": (np.diff(np.sign(X - X.mean(-1, keepdims=True)), axis=-1) != 0).mean(-1),
        "ptp": np.log(np.ptp(X, axis=-1) + 1e-12),
    }
    F = np.concatenate(list(feats.values()), axis=1)
    names = [f"temporal:{k}:broad:ch:{c}" for k in feats for c in CHANNELS]
    return F.astype(np.float32), names


def extract_bands(w, n_jobs=1):
    """Hjorth mobility/complexity, log variance, skew and kurtosis of every
    band-limited signal per channel."""
    from .common import BANDS, bandpass
    cols, names = [], []
    for b, r in BANDS.items():
        X = bandpass(w.X.astype(np.float64), w.sfreq, *r)
        d1 = np.diff(X, axis=-1); d2 = np.diff(d1, axis=-1)
        v0, v1, v2 = X.var(-1), d1.var(-1), d2.var(-1)
        mob = np.sqrt(v1 / v0)
        for nm, v in (("logvar", np.log(v0 + 1e-12)), ("hjorth_mobility", mob),
                      ("hjorth_complexity", np.sqrt(v2 / v1) / mob),
                      ("skew", skew(X, axis=-1)), ("kurtosis", kurtosis(X, axis=-1))):
            cols.append(v); names += [f"temporal_band:{nm}:{b}:ch:{c}" for c in CHANNELS]
    return np.concatenate(cols, 1).astype(np.float32), names
