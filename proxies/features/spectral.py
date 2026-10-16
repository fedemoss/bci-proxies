"""Spectral features: band power, ratios, shape of the spectrum, asymmetry.

Per channel and per region: absolute (log) and relative band power, the
classic ratios, spectral edge / centroid, individual alpha peak frequency and
the aperiodic (1/f) slope -- broadband and the 30-45 Hz slope that Gao et al.
(2017) relate to excitation/inhibition balance. Plus left-right asymmetry of
log band power for homologous regions.
"""

import numpy as np

from ..regions import CHANNELS, REGION_NAMES
from .common import BANDS, BROAD, HOMOLOGOUS, psd, to_regions

RATIOS = [("theta", "beta_lo"), ("theta", "alpha_hi"), ("alpha_hi", "beta_lo"),
          ("alpha_lo", "alpha_hi"), ("beta_lo", "beta_hi")]


def _slope(f, S, lo, hi):
    """OLS slope of log10 power on log10 frequency in [lo, hi]."""
    m = (f >= lo) & (f <= hi)
    x = np.log10(f[m]); x = x - x.mean()
    y = np.log10(S[..., m] + 1e-12)
    return (y - y.mean(-1, keepdims=True)) @ x / (x @ x)


def _block(f, S, labels, where):
    feats, names = [], []
    broad = (f >= BROAD[0]) & (f <= BROAD[1])
    total = S[..., broad].sum(-1)
    logp = {}
    for b, (lo, hi) in BANDS.items():
        m = (f >= lo) & (f < hi)
        p = S[..., m].sum(-1)
        logp[b] = np.log(p + 1e-12)
        feats += [logp[b], p / total]
        names += [f"logpow:{b}", f"relpow:{b}"]
    for a, b in RATIOS:
        feats.append(logp[a] - logp[b]); names.append(f"ratio:{a}/{b}")
    Sb, fb = S[..., broad], f[broad]
    cum = np.cumsum(Sb, -1) / Sb.sum(-1, keepdims=True)
    feats.append(fb[(cum < 0.9).sum(-1).clip(max=len(fb) - 1)]); names.append("edge90:broad")
    feats.append((Sb * fb).sum(-1) / Sb.sum(-1)); names.append("centroid:broad")
    am = (f >= 7) & (f <= 14)
    feats.append(f[am][S[..., am].argmax(-1)]); names.append("peakfreq:alpha")
    feats.append(_slope(f, S, 2, 45)); names.append("slope:2-45")
    feats.append(_slope(f, S, 30, 45)); names.append("slope:30-45")
    # each entry (n, L) -> columns measure:loc
    F = np.concatenate(feats, axis=1)
    cols = [f"spectral:{nm}:{where}:{lab}" for nm in names for lab in labels]
    return F, cols, logp


def extract(w, n_jobs=1):
    f, S = psd(w.X, w.sfreq)                         # (n, C, F)
    Fc, cc, _ = _block(f, S, CHANNELS, "ch")
    Fr, cr, logp_r = _block(f, to_regions(S), REGION_NAMES, "reg")
    asym, ca = [], []
    for b in BANDS:
        for L, R in HOMOLOGOUS:
            asym.append(logp_r[b][:, REGION_NAMES.index(L)] - logp_r[b][:, REGION_NAMES.index(R)])
            ca.append(f"spectral:asym:{b}:reg:{L}|{R}")
    F = np.concatenate([Fc, Fr, np.stack(asym, 1)], axis=1)
    return F.astype(np.float32), cc + cr + ca
