"""Shared signal helpers: bands, filtering, analytic signals, region pooling."""

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt, welch

from ..regions import CHANNELS, REGIONS, membership

# Broader than the ERD/S bands of bci-erds: delta..gamma. The window is
# low-passed at 75 Hz and sampled at 120 Hz, so gamma stops at 45 Hz to keep
# clear of the 50/60 Hz notches.
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 7.0),
    "alpha_lo": (7.0, 10.0),
    "alpha_hi": (10.0, 13.0),
    "beta_lo": (13.0, 20.0),
    "beta_hi": (20.0, 30.0),
    "gamma": (30.0, 45.0),
}
BROAD = (1.0, 45.0)

# Homologous left/right region pairs, for hemispheric asymmetry.
HOMOLOGOUS = [
    ("L_frontal", "R_frontal"),
    ("L_central", "R_central"),
    ("L_parieto_temporal", "R_parieto_temporal"),
    ("L_parieto_occipital", "R_parieto_occipital"),
]


def psd(X, sfreq, seg_s=1.0):
    """Welch PSD per window/channel: 1 s Hann segments, 75 % overlap."""
    n = int(round(seg_s * sfreq))
    f, S = welch(X, fs=sfreq, nperseg=n, noverlap=3 * n // 4, axis=-1)
    return f, S


def bandpass(X, sfreq, lo, hi, order=4):
    sos = butter(order, [lo, hi], btype="band", fs=sfreq, output="sos")
    return sosfiltfilt(sos, X, axis=-1)


def analytic(X, sfreq, lo, hi, trim_s=0.25):
    """Band-limited analytic signal, edges trimmed against filter transients."""
    z = hilbert(bandpass(X, sfreq, lo, hi), axis=-1)
    k = int(round(trim_s * sfreq))
    return z[..., k:-k] if k else z


def to_regions(X):
    """(n, C, ...) -> (n, R, ...): mean signal of each region's electrodes."""
    M = membership(CHANNELS, REGIONS, normalise=True)
    return np.einsum("rc,nc...->nr...", M, X)


def pair_names(labels):
    i, j = np.triu_indices(len(labels), 1)
    return [f"{labels[a]}-{labels[b]}" for a, b in zip(i, j)], (i, j)
