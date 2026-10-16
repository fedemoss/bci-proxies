"""Riemannian covariance features, fold-independent (log-Euclidean).

Per band, the shrunk spatial covariance of the 9 region signals is mapped to
the tangent space at the identity, logm(C), and its upper triangle (diagonal
scaled by 1, off-diagonal by sqrt 2) is kept. A tangent space at the training
mean (pyriemann's TangentSpace) would depend on the fold, so it belongs in the
model, not in the feature table; this is the fold-free counterpart.
"""

import numpy as np

from ..regions import REGION_NAMES
from .common import BANDS, bandpass, pair_names, to_regions


def logm_spd(C):
    w, V = np.linalg.eigh(C)
    return (V * np.log(np.clip(w, 1e-12, None))[..., None, :]) @ V.swapaxes(-1, -2)


def extract(w, n_jobs=1, shrink=0.05):
    k = len(REGION_NAMES)
    iu = np.triu_indices(k)
    scale = np.where(iu[0] == iu[1], 1.0, np.sqrt(2))
    labels = REGION_NAMES + pair_names(REGION_NAMES)[0]
    order = np.r_[np.flatnonzero(iu[0] == iu[1]), np.flatnonzero(iu[0] != iu[1])]
    cols, names = [], []
    for b, r in BANDS.items():
        s = to_regions(bandpass(w.X.astype(np.float64), w.sfreq, *r))
        s -= s.mean(-1, keepdims=True)
        C = np.einsum("nit,njt->nij", s, s) / s.shape[-1]
        mu = np.trace(C, axis1=1, axis2=2)[:, None, None] / k
        C = (1 - shrink) * C + shrink * mu * np.eye(k)
        L = logm_spd(C)[:, iu[0], iu[1]] * scale
        cols.append(L[:, order])
        names += [f"cov:logeuclid:{b}:reg:{lab}" for lab in labels]
    return np.concatenate(cols, 1).astype(np.float32), names


def extract_channels(w, n_jobs=1, shrink=0.05):
    """The same log-Euclidean map on the full 41-channel covariance per band
    (861 per band): the input of every Riemannian BCI pipeline."""
    from ..regions import CHANNELS
    k = len(CHANNELS)
    iu = np.triu_indices(k)
    scale = np.where(iu[0] == iu[1], 1.0, np.sqrt(2))
    labels = [CHANNELS[i] if i == j else f"{CHANNELS[i]}-{CHANNELS[j]}" for i, j in zip(*iu)]
    cols, names = [], []
    for b, r in BANDS.items():
        s = bandpass(w.X.astype(np.float64), w.sfreq, *r)
        s -= s.mean(-1, keepdims=True)
        C = np.einsum("nit,njt->nij", s, s) / s.shape[-1]
        mu = np.trace(C, axis1=1, axis2=2)[:, None, None] / k
        C = (1 - shrink) * C + shrink * mu * np.eye(k)
        cols.append(logm_spd(C)[:, iu[0], iu[1]] * scale)
        names += [f"cov_ch:logeuclid:{b}:ch:{lab}" for lab in labels]
    return np.concatenate(cols, 1).astype(np.float32), names
