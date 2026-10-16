"""Which features generalise across users? Every subject x session is a "user"
(14 units: P04 and P10 have three sessions each, the others one).

    python -m proxies.evaluate --out $RESULTS/all
    python -m proxies.evaluate --families spectral erd_inwin --out $RESULTS/classic
    python -m proxies.evaluate --strict-subject --out $RESULTS/all_strict

1. Best features per user (model-free, that user's trials only): per feature,
   the 3-class ANOVA F and, per class, Cohen's d of class k vs rest. Each user
   ranks every feature.
2. Generality = in how many users a feature is in the top k (k = 50, 100,
   500). Per class, the d must also have the majority sign. A proxy that
   works for one user but not the others scores 1/14.
3. Does generality transfer? Leave-one-user-out XGBoost, three models per fold:
     all      every feature, F-test filter (top 2000) on the training users
     general  the N most general features, counted on the training users only
     pooled   the N best by pooled F on the training users (the usual
              filter: tests whether "general" beats "strong on average")
   scored with the official metric (balanced accuracy per session x
   interface, averaged) on the held-out user. The SHAP of the `all` model on
   the held-out user gives a model-based top-k count as a second view.

With sessions as users, a fold that holds out P04 session 1 still trains on
P04 sessions 2-3. --strict-subject drops the held-out user's other sessions
from the training fold and from the generality count: the cross-subject
version of the same question.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import f_classif

from . import data, features
from .regions import CLASSES

TOPK = [50, 100, 500]


def load_features(families, root):
    Fs, names = [], []
    for fam in families:
        d = np.load(os.path.join(root, f"{fam}.npz"))
        Fs.append(d["F"]); names += [str(n) for n in d["names"]]
    return np.concatenate(Fs, 1), np.array(names)


def _impute(F):
    med = np.nanmedian(F, 0)
    return np.where(np.isfinite(F), F, np.nan_to_num(med))


def unit_scores(F, y):
    """One user's trials -> (F-stat (f,), Cohen's d (K, f))."""
    Fi = _impute(F)
    with np.errstate(all="ignore"):
        fstat, _ = f_classif(Fi, y)
    d = []
    for k in range(len(CLASSES)):
        a, b = Fi[y == k], Fi[y != k]
        sp = np.sqrt(0.5 * (a.var(0) + b.var(0)))
        d.append((a.mean(0) - b.mean(0)) / np.where(sp > 0, sp, np.nan))
    return np.nan_to_num(fstat, nan=0.0), np.asarray(d)


def ranks(score):
    """Rank along the last axis, 0 = best (highest score)."""
    return (-score).argsort(-1).argsort(-1)


def generality(fstat, d, users, top):
    """Counts over ``users`` (row indices): top-k by F, and per class top-k by
    |d| with the majority sign."""
    rF = ranks(fstat[users])                                  # (U, f)
    out = {f"F_top{k}": (rF < k).sum(0) for k in top}
    out["F_rank_median"] = np.median(rF, 0)
    for c, cls in enumerate(CLASSES):
        dc = d[users, c]                                      # (U, f)
        sign = np.sign(np.nansum(np.sign(dc), 0))
        rd = ranks(np.nan_to_num(np.abs(dc)))
        agree = np.sign(dc) == sign
        for k in top:
            out[f"{cls}_top{k}"] = ((rd < k) & agree).sum(0)
        out[f"{cls}_d_mean"] = np.nanmean(dc, 0)
        out[f"{cls}_sign_consistency"] = agree.mean(0)
    return out


def fit_predict(F, y, tr, te, cols, seed, n_jobs, shap=False):
    clf = xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.3, min_child_weight=5, reg_lambda=1.0,
        tree_method="hist", n_jobs=n_jobs, random_state=seed)
    clf.fit(F[tr][:, cols], y[tr])
    Xte = F[te][:, cols]
    pred = clf.predict(Xte)
    if not shap:
        return pred, None
    contrib = clf.get_booster().predict(xgb.DMatrix(Xte), pred_contribs=True)  # (n, K, f+1)
    imp = np.zeros((len(CLASSES), F.shape[1]), np.float32)
    imp[:, cols] = np.abs(contrib[..., :-1]).mean(0)
    return pred, imp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=features.DEFAULT)
    ap.add_argument("--strict-subject", action="store_true")
    ap.add_argument("--k-keep", type=int, default=2000, help="F-filter size of the `all` model")
    ap.add_argument("--n-general", type=int, default=100, help="features in `general`/`pooled`")
    ap.add_argument("--gen-k", type=int, default=100, help="top-k defining generality in the fold")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", default=os.environ.get("OFFICIAL"))
    ap.add_argument("--features", default=os.environ.get("FEATURES"))
    ap.add_argument("--n-jobs", type=int, default=int(os.environ.get("N_JOBS", 4)))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    w = data.load(a.data)
    meta, y = w.meta, w.meta.y.to_numpy()
    meta["unit"] = meta.subject + "/" + meta.session
    F, names = load_features(a.families, a.features)
    ok = (np.isfinite(F).mean(0) > 0.8) & (np.nanstd(F, 0) > 0)
    F, names = F[:, ok], names[ok]
    units = sorted(meta.unit.unique())
    unit_of = meta.unit.to_numpy()
    subj_of = dict(zip(meta.unit, meta.subject))
    print(f"[eval] {F.shape[1]} features ({(~ok).sum()} dropped) from {len(a.families)} "
          f"families, {len(units)} users")

    # 1. per-user scores
    fstat = np.zeros((len(units), F.shape[1]), np.float32)
    d = np.zeros((len(units), len(CLASSES), F.shape[1]), np.float32)
    for u, unit in enumerate(units):
        i = unit_of == unit
        fstat[u], d[u] = unit_scores(F[i], y[i])

    # 2. generality over all users
    gen = pd.DataFrame({"feature": names, "family": [n.split(":")[0] for n in names],
                        **generality(fstat, d, np.arange(len(units)), TOPK)})
    gen["F_mean"] = fstat.mean(0)

    # 3. leave-one-user-out
    rows, shap_top = [], np.zeros((len(CLASSES), F.shape[1]), int)
    for u, unit in enumerate(units):
        te = np.flatnonzero(unit_of == unit)
        keep = np.array([v != unit and not (a.strict_subject and subj_of[v] == subj_of[unit])
                         for v in units])
        tr = np.flatnonzero(np.isin(unit_of, np.array(units)[keep]))
        others = np.flatnonzero(keep)

        Fi = _impute(F[tr])
        with np.errstate(all="ignore"):
            pooled = np.nan_to_num(f_classif(Fi, y[tr])[0], nan=0.0)
        g = generality(fstat, d, others, [a.gen_k])
        # most general first, ties broken by the median within-user rank
        order = np.lexsort((g["F_rank_median"], -g[f"F_top{a.gen_k}"]))
        models = {
            "all": np.argsort(-pooled)[:a.k_keep],
            "general": order[:a.n_general],
            "pooled": np.argsort(-pooled)[:a.n_general],
        }
        res = {}
        for m, cols in models.items():
            pred, imp = fit_predict(F, y, tr, te, cols, a.seed, a.n_jobs, shap=(m == "all"))
            score, cells = data.cell_mean(meta.iloc[te], y[te], pred)
            res[m] = score
            rows += [{"user": unit, "model": m, "cell": c, "bacc": v} for c, v in cells.items()]
            if imp is not None:
                shap_top += ranks(imp) < a.gen_k
        print(f"[eval] {unit}: " + "  ".join(f"{m} {s:.3f}" for m, s in res.items()))

    for c, cls in enumerate(CLASSES):
        gen[f"{cls}_shap_top{a.gen_k}"] = shap_top[c]
    gen.sort_values([f"F_top{TOPK[1]}", "F_rank_median"], ascending=[False, True]) \
       .to_csv(os.path.join(a.out, "generality.csv"), index=False)
    np.savez_compressed(os.path.join(a.out, "unit_scores.npz"), fstat=fstat, d=d,
                        names=names, units=np.array(units))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(a.out, "loo_cells.csv"), index=False)
    per_user = res.groupby(["model", "user"]).bacc.mean().unstack(0)
    per_user.to_csv(os.path.join(a.out, "loo_users.csv"))
    summary = {**vars(a), "n_features": int(F.shape[1]), "users": units,
               "mean": per_user.mean().to_dict(), "sd": per_user.std(ddof=1).to_dict()}
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(per_user.mean().round(3).to_string())
    print(f"[eval] -> {a.out}")


if __name__ == "__main__":
    main()
