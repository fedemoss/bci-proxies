"""Cross-frequency coupling per channel.

Phase-amplitude coupling as the normalised mean vector length

    PAC = |<A_hi e^{i phi_lo}>| / <A_hi>

(Canolty 2006, normalised by mean amplitude so it is bounded and insensitive
to power), and amplitude-amplitude coupling corr(A_lo, A_hi).
"""

import numpy as np

from ..regions import CHANNELS
from .common import BANDS, analytic

PHASE = ["theta", "alpha_lo", "alpha_hi"]
AMP = ["beta_hi", "gamma"]


def extract(w, n_jobs=1):
    X = w.X.astype(np.float64)
    z = {b: analytic(X, w.sfreq, *BANDS[b]) for b in PHASE + AMP}
    cols, names = [], []
    for p in PHASE:
        ph = np.exp(1j * np.angle(z[p]))
        Ap = np.abs(z[p])
        for a in AMP:
            Aa = np.abs(z[a])
            cols.append(np.abs((Aa * ph).mean(-1)) / Aa.mean(-1))
            names += [f"cfc:pac:{p}>{a}:ch:{c}" for c in CHANNELS]
            x = Ap - Ap.mean(-1, keepdims=True); y = Aa - Aa.mean(-1, keepdims=True)
            cols.append((x * y).sum(-1) / np.sqrt((x * x).sum(-1) * (y * y).sum(-1)))
            names += [f"cfc:aac:{p}>{a}:ch:{c}" for c in CHANNELS]
    return np.concatenate(cols, 1).astype(np.float32), names
