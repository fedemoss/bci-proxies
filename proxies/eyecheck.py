"""Are the frontal low-frequency proxies eye movements?

PROTEUS has no EOG and no Fp1/Fp2/AF7/AF8 (AF3/AF4 are the most anterior
electrodes), so the check is indirect:

1. Gradient. An ocular artifact is largest at the most anterior electrodes and
   fades towards the back of the head. Per feature type (delta / theta log
   power, in-window delta / theta trend, and upper-alpha log power as a
   sensorimotor control), the per-user Cohen's d of each class vs rest is
   averaged over users per electrode and per electrode row (AF ... PO).
2. Eye proxies. From the window low-passed below 4 Hz: vertical = mean of
   AF3/AF4 (log peak-to-peak and blink-like peaks), horizontal = F6 - F5 and
   AF4 - AF3 (log peak-to-peak). Per user they are scored like every feature;
   leave-one-user-out XGBoost on the 4 proxies alone says how much of the task
   they carry.
3. Overlap. Within each user, the correlation of the most general features
   (from a run's generality.csv) with each proxy, median over users.

The third check, removing every frontal feature, is
`python -m proxies.evaluate --exclude-frontal`.

    python -m proxies.eyecheck [--run all] [--top 50]

Writes $RESULTS/eyecheck/: gradient.csv, gradient_rows.csv, gradient.png,
proxies_users.csv, proxies_loo.csv, overlap.csv.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt
from scipy.stats import spearmanr
from sklearn.feature_selection import f_classif

from . import data
from .evaluate import _impute, fit_predict, load_features, unit_scores
from .regions import CHANNELS, CLASSES

ROWS = ["AF", "F", "FC", "C", "CP", "P", "PO"]
ROW_OF = {"TP": "CP"}
TYPES = {
    "logpow delta": "spectral:logpow:delta:ch:",
    "logpow theta": "spectral:logpow:theta:ch:",
    "trend delta": "erd_inwin:trend:delta:ch:",
    "trend theta": "erd_inwin:trend:theta:ch:",
    "logpow alpha_hi (control)": "spectral:logpow:alpha_hi:ch:",
}
PROXIES = ["veog_logptp", "blinks", "heog_F_logptp", "heog_AF_logptp"]


def row(ch):
    p = ch.rstrip("0123456789z")
    return ROW_OF.get(p, p)


def gradient(F, names, y, unit_of, units):
    col = {n: i for i, n in enumerate(names)}
    rows = []
    for t, prefix in TYPES.items():
        idx = [col[prefix + ch] for ch in CHANNELS if prefix + ch in col]
        chans = [ch for ch in CHANNELS if prefix + ch in col]
        d = np.stack([unit_scores(F[unit_of == u][:, idx], y[unit_of == u])[1]
                      for u in units])                                # (U, K, ch)
        for c, cls in enumerate(CLASSES):
            sign = np.sign(np.nanmean(d[:, c], 0))
            for j, ch in enumerate(chans):
                rows.append({"type": t, "class": cls, "channel": ch, "row": row(ch),
                             "d_mean": np.nanmean(d[:, c, j]),
                             "absd_mean": np.nanmean(np.abs(d[:, c, j])),
                             "sign_consistency": np.mean(np.sign(d[:, c, j]) == sign[j])})
    g = pd.DataFrame(rows)
    g["row_idx"] = g.row.map(ROWS.index)
    by_row = g.groupby(["type", "class", "row", "row_idx"], as_index=False)[
        ["d_mean", "absd_mean"]].mean().sort_values(["type", "class", "row_idx"])
    rho = g.groupby(["type", "class"]).apply(
        lambda x: spearmanr(x.row_idx, x.absd_mean)[0], include_groups=False)
    peak = g.loc[g.groupby(["type", "class"]).absd_mean.idxmax()].set_index(["type", "class"])
    return g, by_row, rho, peak.channel


def plot_gradient(by_row, out):
    fig, axes = plt.subplots(1, len(TYPES), figsize=(4 * len(TYPES), 3.6), sharey=True)
    for ax, t in zip(axes, TYPES):
        for cls in CLASSES:
            r = by_row[(by_row.type == t) & (by_row["class"] == cls)]
            ax.plot(r.row_idx, r.d_mean, marker="o", label=cls)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(range(len(ROWS)), ROWS)
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("front  →  back")
    axes[0].set_ylabel("Cohen's d, class vs rest (mean over users)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def eye_proxies(X, sfreq):
    """(n, 4) per window, from the < 4 Hz signal of the most frontal channels."""
    sos = butter(4, 4.0, "low", fs=sfreq, output="sos")
    ch = {c: i for i, c in enumerate(CHANNELS)}
    lp = sosfiltfilt(sos, X[:, [ch["AF3"], ch["AF4"], ch["F5"], ch["F6"]]], axis=-1)
    v = lp[:, :2].mean(1)
    blinks = [len(find_peaks(s - np.median(s), prominence=40, width=(0.1 * sfreq, 0.6 * sfreq))[0])
              for s in v]
    ptp = lambda s: np.log(np.ptp(s, -1) + 1e-6)
    return np.stack([ptp(v), np.array(blinks, float), ptp(lp[:, 3] - lp[:, 2]),
                     ptp(lp[:, 1] - lp[:, 0])], 1).astype(np.float32)


def proxy_scores(E, y, unit_of, units):
    rows = []
    for u in units:
        i = unit_of == u
        fstat, d = unit_scores(E[i], y[i])
        with np.errstate(all="ignore"):
            p = f_classif(_impute(E[i]), y[i])[1]
        for j, name in enumerate(PROXIES):
            rows.append({"user": u, "proxy": name, "F": fstat[j], "p": p[j],
                         **{f"{c}_d": d[k, j] for k, c in enumerate(CLASSES)}})
    return pd.DataFrame(rows)


def proxy_loo(E, y, meta, unit_of, units, seed, n_jobs):
    rows = []
    for u in units:
        te, tr = np.flatnonzero(unit_of == u), np.flatnonzero(unit_of != u)
        pred, _ = fit_predict(E, y, tr, te, np.arange(E.shape[1]), seed, n_jobs)
        rows.append({"user": u, "bacc": data.cell_mean(meta.iloc[te], y[te], pred)[0]})
    return pd.DataFrame(rows)


def overlap(gen, E, feat_dir, y, unit_of, units, top):
    keys = ["F_top100"] + [f"{c}_top100" for c in CLASSES]
    pick = pd.concat([gen.sort_values([k, "F_rank_median"], ascending=[False, True])
                      .head(top if k == "F_top100" else 10).assign(ranked_by=k) for k in keys])
    pick = pick.drop_duplicates("feature")
    F, names = load_features(sorted(pick.family.unique()), feat_dir)
    col = {n: i for i, n in enumerate(names)}
    X = F[:, [col[f] for f in pick.feature]]
    r = np.zeros((len(units), X.shape[1], E.shape[1]))
    for k, u in enumerate(units):
        i = unit_of == u
        a, b = _impute(X[i]), E[i]
        a = (a - a.mean(0)) / (a.std(0) + 1e-12)
        b = (b - b.mean(0)) / (b.std(0) + 1e-12)
        r[k] = a.T @ b / i.sum()
    med = np.median(r, 0)
    out = pick[["ranked_by", "feature", "family", *keys]].reset_index(drop=True)
    for j, p in enumerate(PROXIES):
        out[f"r_{p}"] = med[:, j]
    out["max_abs_r"] = np.abs(med).max(1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.environ.get("RESULTS"))
    ap.add_argument("--features", default=os.environ.get("FEATURES"))
    ap.add_argument("--data", default=os.environ.get("OFFICIAL"))
    ap.add_argument("--run", default="all")
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=int(os.environ.get("N_JOBS", 4)))
    a = ap.parse_args()
    out = os.path.join(a.results, "eyecheck")
    os.makedirs(out, exist_ok=True)

    w = data.load(a.data)
    y = w.meta.y.to_numpy()
    unit_of = (w.meta.subject + "/" + w.meta.session).to_numpy()
    units = sorted(set(unit_of))

    # 1. gradient
    F, names = load_features(["spectral", "erd_inwin"], a.features)
    g, by_row, rho, peak = gradient(F, names, y, unit_of, units)
    g.to_csv(os.path.join(out, "gradient.csv"), index=False)
    by_row.to_csv(os.path.join(out, "gradient_rows.csv"), index=False)
    plot_gradient(by_row, os.path.join(out, "gradient.png"))
    print("[eye] |d| vs front-to-back row (Spearman rho < 0: largest at the front), "
          "and the electrode with the largest |d|")
    print(pd.DataFrame({"rho": rho, "peak": peak}).round(2).to_string())
    print(by_row.pivot_table(index=["type", "class"], columns="row", values="d_mean")[ROWS]
          .round(2).to_string())

    # 2. eye proxies
    E = eye_proxies(w.X, w.sfreq)
    ps = proxy_scores(E, y, unit_of, units)
    ps.to_csv(os.path.join(out, "proxies_users.csv"), index=False)
    summ = ps.groupby("proxy").agg(
        users_p05=("p", lambda p: int((p < 0.05).sum())),
        **{f"{c}_d": (f"{c}_d", "mean") for c in CLASSES},
        **{f"{c}_same_sign": (f"{c}_d", lambda d: max((d > 0).mean(), (d < 0).mean()))
           for c in CLASSES})
    print("\n[eye] proxies: users (of 14) where the proxy differs by task (p < 0.05), "
          "mean d per class and share of users with the majority sign")
    print(summ.loc[PROXIES].round(2).to_string())
    print(f"[eye] blinks per window: median {np.median(E[:, 1]):.0f}, "
          f"windows with >= 1: {(E[:, 1] > 0).mean():.0%}")
    loo = proxy_loo(E, y, w.meta, unit_of, units, a.seed, a.n_jobs)
    loo.to_csv(os.path.join(out, "proxies_loo.csv"), index=False)
    print(f"[eye] leave-one-user-out XGB on the 4 proxies alone: "
          f"{loo.bacc.mean():.3f} (chance {1 / len(CLASSES):.3f})")

    # 3. overlap with the most general features
    gen = pd.read_csv(os.path.join(a.results, a.run, "generality.csv"))
    ov = overlap(gen, E, a.features, y, unit_of, units, a.top)
    ov.to_csv(os.path.join(out, "overlap.csv"), index=False)
    print("\n[eye] median within-user r with the proxies")
    for k, g_ in ov.groupby("ranked_by", sort=False):
        print(f"  {k}: {len(g_)} features, median max|r| {g_.max_abs_r.median():.2f}, "
              f"share with max|r| > 0.5: {(g_.max_abs_r > 0.5).mean():.0%}")
    print(ov.head(15)[["feature", "max_abs_r", *[f"r_{p}" for p in PROXIES]]]
          .round(2).to_string(index=False))
    print(f"[eye] -> {out}")


if __name__ == "__main__":
    main()
