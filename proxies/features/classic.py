"""Classic ERD/S baselines.

``classic_ref``: Pfurtscheller-style ERD/S, log band power of each 1 s patch of
the window minus that of the pre-cue second (bci-erds ``P``), per channel and
per region, for the four mu/beta bands of Friedrich (2012). The pre-cue second
is outside the official window, so a submission cannot compute this at test
time -- it is the privileged upper reference, not a deployable proxy. The
deployable classic proxy is mu/beta log power inside the window, which the
``spectral`` family already contains.
"""

import numpy as np

from ..regions import BAND_NAMES, CHANNELS, REGION_NAMES, membership


def extract(w, n_jobs=1):
    logP = np.log(w.P + 1e-12)                       # (n, C, B, 1 + S)
    E = logP[..., 1:] - logP[..., :1]                # (n, C, B, S)
    Er = np.log(np.einsum("rc,ncbs->nrbs", membership(), w.P) + 1e-12)
    Er = Er[..., 1:] - Er[..., :1]
    cols, names = [], []
    for lab, A, locs in (("ch", E, CHANNELS), ("reg", Er, REGION_NAMES)):
        for b, bn in enumerate(BAND_NAMES):
            for s in range(A.shape[-1]):
                cols.append(A[:, :, b, s]); names += [f"classic_ref:erd_p{s}:{bn}:{lab}:{l}" for l in locs]
            cols.append(A[:, :, b].mean(-1)); names += [f"classic_ref:erd_mean:{bn}:{lab}:{l}" for l in locs]
    return np.concatenate(cols, 1).astype(np.float32), names


def extract_inwin(w, n_jobs=1):
    """Deployable ERD/S: band power of each 1 s patch relative to the window's
    own first second (cue +1 to +2 s), and the log-power trend across the four
    patches, for all 7 bands per channel and region. No pre-cue data needed."""
    from scipy.signal import welch

    from .common import BANDS, to_regions
    n = int(w.sfreq)
    patches = np.stack([w.X[..., i * n:(i + 1) * n] for i in range(w.X.shape[-1] // n)], 2)
    f, S = welch(patches.astype(np.float64), fs=w.sfreq, nperseg=n // 2, axis=-1)  # (n, C, S, F)
    cols, names = [], []
    t = np.arange(patches.shape[2]) - (patches.shape[2] - 1) / 2
    for lab, A, locs in (("ch", S, CHANNELS), ("reg", to_regions(S), REGION_NAMES)):
        for b, (lo, hi) in BANDS.items():
            m = (f >= lo) & (f < hi)
            lp = np.log(A[..., m].sum(-1) + 1e-12)                 # (n, L, S)
            for s in range(1, lp.shape[-1]):
                cols.append(lp[..., s] - lp[..., 0])
                names += [f"erd_inwin:rel_p{s}:{b}:{lab}:{l}" for l in locs]
            cols.append(lp @ t / (t @ t))
            names += [f"erd_inwin:trend:{b}:{lab}:{l}" for l in locs]
    return np.concatenate(cols, 1).astype(np.float32), names
