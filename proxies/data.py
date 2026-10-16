"""The official Track 02 windows (PROTEUS, NeuralBench loader) and their labels.

``official.npz`` is written by bci-erds (``erds.official``): 2655 cued windows,
41 EEG channels x 480 samples (4 s from cue + 1 s, 0.1-75 Hz, 120 Hz, uV),
the official 60/20/20 subject split, and per-window metadata. ``P`` holds the
pre-cue reference band power that only the ``classic_ref`` family uses -- it
lies outside the window, so the model never has it at test time.
"""

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .regions import CHANNELS, CLASSES

META = ["split", "subject", "session", "task", "run", "interface", "cue", "y"]


@dataclass
class Windows:
    X: np.ndarray        # (n, C, T) float32, uV
    P: np.ndarray        # (n, C, 4 bands, 1 ref + 4 patches) band power
    meta: pd.DataFrame   # one row per window, columns META + "cell"
    sfreq: float
    ch_names: list


def load(path=None):
    d = np.load(path or os.environ["OFFICIAL"], allow_pickle=True)
    ch_names = [str(c) for c in d["ch_names"]]
    if ch_names != CHANNELS:
        raise ValueError(f"channel order differs from regions.CHANNELS: {ch_names}")
    meta = pd.DataFrame({k: d[k] for k in META})
    meta["subject"] = meta.subject.astype(str)
    meta["session"] = meta.session.astype(str)
    # The official scoring unit: balanced accuracy per subject x session x
    # interface, averaged over cells.
    meta["cell"] = meta.subject + "/" + meta.session + "/" + meta.interface.astype(str)
    return Windows(d["X"].astype(np.float32), d["P"].astype(np.float32), meta,
                   float(d["sfreq"]), ch_names)


def cell_mean(meta, y_true, y_pred):
    """Official metric: mean over cells of the per-cell balanced accuracy."""
    from sklearn.metrics import balanced_accuracy_score
    per = {c: balanced_accuracy_score(y_true[i], y_pred[i])
           for c, i in meta.groupby("cell").indices.items()}
    return float(np.mean(list(per.values()))), per


__all__ = ["Windows", "load", "cell_mean", "CLASSES"]
