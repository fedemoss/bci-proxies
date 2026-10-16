"""Discrete wavelet decomposition per channel (db4, 5 levels at 120 Hz).

Levels: D1 30-60, D2 15-30, D3 7.5-15, D4 3.75-7.5, D5 1.9-3.75 Hz, A5 < 1.9 Hz.
Per level: log energy, relative energy, coefficient std, and the Shannon
entropy of the normalised squared coefficients. Plus the wavelet entropy across levels.
"""

import numpy as np
import pywt

from ..regions import CHANNELS

LEVELS = ["A5", "D5", "D4", "D3", "D2", "D1"]


def extract(w, n_jobs=1):
    coeffs = pywt.wavedec(w.X.astype(np.float64), "db4", level=5, axis=-1)
    E = np.stack([(c ** 2).sum(-1) for c in coeffs], -1)      # (n, C, L)
    rel = E / E.sum(-1, keepdims=True)
    cols, names = [], []
    for l, (lev, c) in enumerate(zip(LEVELS, coeffs)):
        p = c ** 2 / ((c ** 2).sum(-1, keepdims=True) + 1e-20)
        ent = -(p * np.log(p + 1e-20)).sum(-1) / np.log(c.shape[-1])
        for nm, v in (("logenergy", np.log(E[..., l] + 1e-12)), ("relenergy", rel[..., l]),
                      ("std", c.std(-1)), ("coef_entropy", ent)):
            cols.append(v); names += [f"wavelet:{nm}:{lev}:ch:{ch}" for ch in CHANNELS]
    cols.append(-(rel * np.log(rel + 1e-20)).sum(-1))
    names += [f"wavelet:wentropy:all:ch:{ch}" for ch in CHANNELS]
    return np.concatenate(cols, 1).astype(np.float32), names
