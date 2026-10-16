"""Feature families. Every family maps the official windows to a table:

    extract(windows, n_jobs) -> (F (n_windows, n_features) float32, names)

Names are ``family:measure:band:level:location`` (level = ch | reg | all), so
results can be grouped by family, measure, band or region afterwards.
"""

from importlib import import_module

FAMILIES = {
    # classic BCI features
    "spectral": ("spectral", "extract"),
    "erd_inwin": ("classic", "extract_inwin"),
    "bursts": ("bursts", "extract"),
    "cov": ("covariance", "extract"),
    "cov_ch": ("covariance", "extract_channels"),
    # generic time-series / complexity
    "temporal": ("temporal", "extract"),
    "temporal_band": ("temporal", "extract_bands"),
    "wavelet": ("wavelet", "extract"),
    "complexity": ("complexity", "extract"),
    "complexity_ms": ("complexity", "extract_multiscale"),
    "catch22": ("catch22", "extract"),
    # interactions
    "conn": ("connectivity", "extract"),
    "conn_pairs": ("connectivity", "extract_pairs"),
    "wsmi": ("wsmi", "extract"),
    "hoi": ("hoi", "extract"),
    "cfc": ("cfc", "extract"),
    # privileged baseline
    "classic_ref": ("classic", "extract"),
}
# Uses information outside the official window: excluded unless asked for.
PRIVILEGED = {"classic_ref"}
DEFAULT = [f for f in FAMILIES if f not in PRIVILEGED]


def get(name):
    mod, fn = FAMILIES[name]
    return getattr(import_module(f"{__name__}.{mod}"), fn)
