"""Read the evaluation results back: which runs win, which features are general,
and how many of the top general features are really distinct.

    python -m proxies.analyze [--run all] [--top 20] [--n-cluster 300]

Writes $RESULTS/analysis/:
  summary.csv            every run: mean / sd of all, general, pooled; paired
                         general - pooled per user (Wilcoxon)
  loo_users.png          per-user scores of the joint runs (session / subject)
  generality_family.png  per family, share of features in the top 100 of >= n users
  generality_family.csv  the same as a table, plus each family's best feature
  top_general.csv        top features by F generality and per class, with d
  top_general_d.png      per-user Cohen's d of the top general features per class
  clusters.csv           the n most general features grouped at |r| > 0.9
                         (hierarchical, complete linkage), one row per feature
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import wilcoxon

from .evaluate import load_features
from .regions import CLASSES

MODELS = ["all", "general", "pooled"]
LEVELS = [2, 4, 7, 10]


def summary(results):
    rows = []
    for f in sorted(glob.glob(os.path.join(results, "*", "metrics.json"))):
        run = os.path.basename(os.path.dirname(f))
        m = json.load(open(f))
        u = pd.read_csv(os.path.join(os.path.dirname(f), "loo_users.csv"), index_col=0)
        diff = u.general - u.pooled
        p = wilcoxon(diff).pvalue if (diff != 0).any() else np.nan
        rows.append({"run": run, "n_features": m["n_features"],
                     "strict": m["strict_subject"],
                     **{f"{k}_mean": u[k].mean() for k in MODELS},
                     **{f"{k}_sd": u[k].std(ddof=1) for k in MODELS},
                     "general_minus_pooled": diff.mean(), "wilcoxon_p": p,
                     "general_wins": int((diff > 0).sum())})
    return pd.DataFrame(rows).sort_values("all_mean", ascending=False)


def plot_users(results, runs, out):
    fig, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 4), sharey=True)
    for ax, run in zip(np.atleast_1d(axes), runs):
        u = pd.read_csv(os.path.join(results, run, "loo_users.csv"), index_col=0)
        x = np.arange(len(u))
        for i, k in enumerate(MODELS):
            ax.bar(x + (i - 1) * 0.27, u[k], 0.27, label=f"{k} ({u[k].mean():.3f})")
        ax.axhline(1 / len(CLASSES), color="k", ls=":", lw=1)
        ax.set_xticks(x, u.index, rotation=60)
        ax.set_title(run)
        ax.legend(fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("balanced accuracy (held-out user)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def family_table(gen, n_users):
    rows = []
    for fam, g in gen.groupby("family"):
        best = g.sort_values(["F_top100", "F_rank_median"], ascending=[False, True]).iloc[0]
        rows.append({"family": fam, "n": len(g),
                     **{f"share_ge{n}": (g.F_top100 >= n).mean() for n in LEVELS},
                     "max_users": int(g.F_top100.max()), "best": best.feature})
    t = pd.DataFrame(rows).sort_values("max_users", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(t))
    for i, n in enumerate(LEVELS):
        ax.barh(y + (i - 1.5) * 0.2, t[f"share_ge{n}"], 0.2, label=f">= {n}/{n_users} users")
    ax.set_yticks(y, [f"{f} ({n})" for f, n in zip(t.family, t.n)])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("share of the family's features in a user's top 100")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return t, fig


def top_features(gen, top):
    keys = ["F_top100"] + [f"{c}_top100" for c in CLASSES]
    parts = []
    for k in keys:
        t = gen.sort_values([k, "F_rank_median"], ascending=[False, True]).head(top)
        parts.append(t.assign(ranked_by=k)[["ranked_by", "feature", "family", *keys,
                                             "F_rank_median",
                                             *[f"{c}_d_mean" for c in CLASSES],
                                             *[f"{c}_sign_consistency" for c in CLASSES]]])
    return pd.concat(parts)


def plot_d(scores, gen, top, out):
    names = scores["names"].astype(str)
    idx = {n: i for i, n in enumerate(names)}
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(7 * len(CLASSES), 0.3 * top + 2))
    for c, (cls, ax) in enumerate(zip(CLASSES, axes)):
        feats = gen.sort_values([f"{cls}_top100", "F_rank_median"],
                                ascending=[False, True]).feature.head(top)
        d = scores["d"][:, c, [idx[f] for f in feats]].T          # (top, users)
        im = ax.imshow(d, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
        ax.set_yticks(range(len(feats)), feats, fontsize=6)
        ax.set_xticks(range(d.shape[1]), scores["units"], rotation=60, fontsize=7)
        ax.set_title(f"{cls} vs rest: Cohen's d per user")
    fig.colorbar(im, ax=axes, shrink=0.6)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def clusters(gen, feat_dir, n, r):
    """Group the n most general features: complete linkage on 1 - |r|, so every
    pair inside a cluster has |r| > r."""
    top = gen.sort_values(["F_top100", "F_rank_median"], ascending=[False, True]).head(n)
    F, names = load_features(sorted(top.family.unique()), feat_dir)
    col = {nm: i for i, nm in enumerate(names)}
    X = F[:, [col[f] for f in top.feature]]
    X = np.where(np.isfinite(X), X, np.nanmedian(X, 0))
    C = np.abs(np.corrcoef(X, rowvar=False))
    dist = squareform(1 - np.nan_to_num(C, nan=0.0), checks=False).clip(0)
    lab = fcluster(linkage(dist, "complete"), 1 - r, "distance")
    out = top[["feature", "family", "F_top100", "F_rank_median"]].assign(cluster=lab)
    # clusters numbered by their most general member
    order = {c: i + 1 for i, c in enumerate(dict.fromkeys(lab))}
    out["cluster"] = out.cluster.map(order)
    out["representative"] = ~out.cluster.duplicated()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.environ.get("RESULTS"))
    ap.add_argument("--features", default=os.environ.get("FEATURES"))
    ap.add_argument("--run", default="all", help="run whose generality.csv is analysed")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--n-cluster", type=int, default=300)
    ap.add_argument("--r", type=float, default=0.9)
    a = ap.parse_args()
    out = os.path.join(a.results, "analysis")
    os.makedirs(out, exist_ok=True)

    s = summary(a.results)
    s.to_csv(os.path.join(out, "summary.csv"), index=False)
    print(s[["run", "n_features", *[f"{k}_mean" for k in MODELS],
             "general_minus_pooled", "wilcoxon_p"]].round(3).to_string(index=False))
    plot_users(a.results, ["all", "all_strict"], os.path.join(out, "loo_users.png"))

    run = os.path.join(a.results, a.run)
    gen = pd.read_csv(os.path.join(run, "generality.csv"))
    scores = np.load(os.path.join(run, "unit_scores.npz"))
    t, fig = family_table(gen, len(scores["units"]))
    t.to_csv(os.path.join(out, "generality_family.csv"), index=False)
    fig.savefig(os.path.join(out, "generality_family.png"), dpi=150)
    print(t.round(4).to_string(index=False))

    top_features(gen, a.top).to_csv(os.path.join(out, "top_general.csv"), index=False)
    plot_d(scores, gen, a.top, os.path.join(out, "top_general_d.png"))

    cl = clusters(gen, a.features, a.n_cluster, a.r)
    cl.to_csv(os.path.join(out, "clusters.csv"), index=False)
    print(f"[analyze] top {a.n_cluster} general features -> {cl.cluster.nunique()} "
          f"clusters at |r| > {a.r}")
    print(cl[cl.representative].head(a.top).to_string(index=False))
    print(f"[analyze] -> {out}")


if __name__ == "__main__":
    main()
