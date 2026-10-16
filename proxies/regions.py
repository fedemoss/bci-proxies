"""Electrodes, brain regions and ERD/S bands for PROTEUS (nm000290).

The regions follow Friedrich et al. (2012), who pooled electrodes into
left/right frontal, central, parieto-occipital and parieto-temporal clusters.
PROTEUS has a denser centro-parietal montage and no temporal or occipital
electrodes, so each cluster is re-populated with its nearest PROTEUS
electrodes, and one midline region is added: Friedrich drops the midline, but
it carries the vertex (Cz) and frontal-midline (Fz) activity that mental
arithmetic and word generation recruit.

Every electrode belongs to exactly one region, so the membership matrix

    M[r, c] = delta(rho(c), r)

is a partition of the 41 channels -- the Kronecker delta of the region pooling.
"""

import numpy as np

# Channel order exactly as in every *_channels.tsv of the release (verified
# identical across all 112 runs). Tensors are always (..., CHANNELS, ...).
CHANNELS = [
    "C2", "CP2", "P2", "Pz", "CPz", "Cz", "C1", "CP1", "P1", "P3", "PO7", "P5",
    "TP7", "CP5", "CP3", "C3", "PO8", "P6", "P4", "TP8", "CP6", "CP4", "C6",
    "C4", "FC6", "FC4", "FC2", "F6", "F4", "F2", "AF4", "AF3", "F5", "F3", "F1",
    "Fz", "FC5", "FC3", "FC1", "FCz", "C5",
]

REGIONS = {
    "L_frontal": ["AF3", "F5", "F3", "F1", "FC5", "FC3"],
    "R_frontal": ["AF4", "F6", "F4", "F2", "FC6", "FC4"],
    "L_central": ["FC1", "C5", "C3", "C1", "CP3", "CP1"],
    "R_central": ["FC2", "C6", "C4", "C2", "CP4", "CP2"],
    "L_parieto_temporal": ["TP7", "CP5", "P5"],
    "L_parieto_occipital": ["P3", "P1", "PO7"],
    "R_parieto_occipital": ["P2", "P4", "PO8"],
    "R_parieto_temporal": ["TP8", "CP6", "P6"],
    "midline": ["Fz", "FCz", "Cz", "CPz", "Pz"],
}
REGION_NAMES = list(REGIONS)

# Friedrich et al. (2012) bands: lower/upper alpha (mu), lower/upper beta.
BANDS = [(7.0, 10.0), (10.0, 13.0), (13.0, 20.0), (20.0, 30.0)]
BAND_NAMES = ["alpha_lo", "alpha_hi", "beta_lo", "beta_hi"]

CLASSES = ["mi", "sub", "word"]


def membership(channels=CHANNELS, regions=REGIONS, normalise=True):
    """``(R, C)`` region membership matrix, rows averaging when ``normalise``."""
    M = np.zeros((len(regions), len(channels)), dtype=np.float32)
    for r, members in enumerate(regions.values()):
        for ch in members:
            M[r, channels.index(ch)] = 1.0
    assigned = M.sum(0)
    if (assigned > 1).any():
        raise ValueError("an electrode sits in two regions")
    if normalise:
        M /= M.sum(1, keepdims=True)
    return M
