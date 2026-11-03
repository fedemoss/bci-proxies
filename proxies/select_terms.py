"""Choose the feature targets of an auxiliary loss (L_feat) without looking at
the test subjects.

    python -m proxies.select_terms [--run all_nofrontal] [--exclude-subjects P04 P09]

The official split tests on P04 and P09, so generality is recounted over the
other users only, from the run's per-user scores (unit_scores.npz: they are
within-user, so dropping users needs no re-extraction). Then, on those users:

1. the --n-candidates most general features (top-100 count, ties by median rank);
2. drop those that track the eye proxies (median within-user max|r| above
   --max-blink-r; proxies.eyecheck);
3. group at |r| > --r (complete linkage), one representative per group;
4. collect the representatives into blocks = family:measure:band:level, each
   a candidate loss term with its head type (ch / pair / reg / all);
5. leave-one-user-out XGBoost on the representatives alone, a check that they
   carry the task for a new user.

Writes $RESULTS/terms/: candidates.csv, blocks.csv, loo.csv.
"""

import argparse
import os

import numpy as np
import pandas as pd

from . import data
from .analyze import clusters
from .evaluate import _impute, fit_predict, generality, load_features
from .eyecheck import PROXIES, eye_proxies
from .regions import CLASSES


def blink_r(X, E, unit_of, units):
    """(f, proxies) median over users of the within-user correlation."""
    r = np.zeros((len(units), X.shape[1], E.shape[1]))
    for k, u in enumerate(units):
        i = unit_of == u
        a, b = _impute(X[i]), E[i]
        a = (a - a.mean(0)) / (a.std(0) + 1e-12)
        b = (b - b.mean(0)) / (b.std(0) + 1e-12)
        r[k] = a.T @ b / i.sum()
    return np.median(r, 0)


def block_of(name):
    p = name.split(":")
    level = p[-2] if len(p) >= 5 else "all"
    if level == "ch" and "-" in p[-1]:
        level = "pair"
    return ":".join(p[:3]), level


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.environ.get("RESULTS"))
    ap.add_argument("--features", default=os.environ.get("FEATURES"))
    ap.add_argument("--data", default=os.environ.get("OFFICIAL"))
    ap.add_argument("--run", default="all_nofrontal")
    ap.add_argument("--exclude-subjects", nargs="*", default=["P04", "P09"])
    ap.add_argument("--n-candidates", type=int, default=500)
    ap.add_argument("--max-blink-r", type=float, default=0.3)
    ap.add_argument("--r", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=int(os.environ.get("N_JOBS", 4)))
    a = ap.parse_args()
    out = os.path.join(a.results, "terms")
    os.makedirs(out, exist_ok=True)

    s = np.load(os.path.join(a.results, a.run, "unit_scores.npz"))
    units_all = [str(u) for u in s["units"]]
    keep = [i for i, u in enumerate(units_all) if u.split("/")[0] not in a.exclude_subjects]
    units = [units_all[i] for i in keep]
    names = s["names"].astype(str)
    print(f"[terms] {a.run}: {len(names)} features, generality over {len(units)} users "
          f"(excluded {a.exclude_subjects})")

    g = generality(s["fstat"], s["d"], np.array(keep), [100])
    gen = pd.DataFrame({"feature": names, "family": [n.split(":")[0] for n in names], **g})
    gen = gen.sort_values(["F_top100", "F_rank_median"], ascending=[False, True])
    cand = gen.head(a.n_candidates).reset_index(drop=True)

    w = data.load(a.data)
    y = w.meta.y.to_numpy()
    unit_of = (w.meta.subject + "/" + w.meta.session).to_numpy()
    mask = np.isin(unit_of, units)
    F, fn = load_features(sorted(cand.family.unique()), a.features)
    col = {n: i for i, n in enumerate(fn)}
    X = F[:, [col[f] for f in cand.feature]]
    E = eye_proxies(w.X, w.sfreq)
    br = blink_r(X[mask], E[mask], unit_of[mask], units)
    for j, p in enumerate(PROXIES):
        cand[f"r_{p}"] = br[:, j]
    cand["blink_r"] = np.abs(br).max(1)
    clean = cand[cand.blink_r <= a.max_blink_r]
    print(f"[terms] top {len(cand)}: {len(clean)} pass max|r| <= {a.max_blink_r} with the eye proxies")

    cl = clusters(clean, a.features, len(clean), a.r)
    cand = cand.merge(cl[["feature", "cluster", "representative"]], on="feature", how="left")
    cand["representative"] = cand.representative.fillna(False).astype(bool)
    blocks = cand.feature.map(block_of)
    cand["block"], cand["level"] = [b[0] for b in blocks], [b[1] for b in blocks]
    cand.to_csv(os.path.join(out, "candidates.csv"), index=False)

    rep = cand[cand.representative]
    agg = {"n_rep": ("feature", "size"), "max_users": ("F_top100", "max"),
           "mean_users": ("F_top100", "mean"), "blink_r": ("blink_r", "max"),
           **{f"{c}_users": (f"{c}_top100", "max") for c in CLASSES},
           **{f"{c}_d": (f"{c}_d_mean", "mean") for c in CLASSES},
           "locations": ("feature", lambda f: " ".join(sorted({n.split(':')[-1] for n in f})))}
    bl = rep.groupby(["block", "level"]).agg(**agg).reset_index() \
            .sort_values(["max_users", "n_rep"], ascending=False)
    bl.to_csv(os.path.join(out, "blocks.csv"), index=False)
    print(f"[terms] {len(rep)} representatives in {len(bl)} blocks")
    with pd.option_context("display.width", 250, "display.max_colwidth", 70):
        print(bl.round(2).to_string(index=False))

    # does the representative set carry the task for a new user?
    cols = np.array([col[f] for f in rep.feature])
    Xm, ym, meta = F[mask][:, cols], y[mask], w.meta[mask].reset_index(drop=True)
    um = unit_of[mask]
    rows = []
    for u in units:
        te, tr = np.flatnonzero(um == u), np.flatnonzero(um != u)
        pred, _ = fit_predict(Xm, ym, tr, te, np.arange(len(cols)), a.seed, a.n_jobs)
        rows.append({"user": u, "bacc": data.cell_mean(meta.iloc[te], ym[te], pred)[0]})
    loo = pd.DataFrame(rows)
    loo.to_csv(os.path.join(out, "loo.csv"), index=False)
    print(f"[terms] leave-one-user-out XGB on the {len(cols)} representatives "
          f"({len(units)} users): {loo.bacc.mean():.3f}")
    print(f"[terms] -> {out}")


if __name__ == "__main__":
    main()
