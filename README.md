# bci-proxies

The search for EEG features ("proxies") that tell the three PROTEUS BCI tasks
apart across subjects and sessions. Many hand-crafted feature families are
ranked within every user (subject × session). A feature counts as general when
it is in the top k for many users; leave-one-user-out XGBoost then tests whether
those features transfer.
The design is in [notes/PLAN.md](notes/PLAN.md).

Same windows and metric as the Track 02 submission (via bci-erds
`official.npz`), but each subject × session is a user (no official split); run on the UdeSA cluster, never locally.

## Layout

```
proxies/
  data.py            official windows + official metric (cell-mean balanced acc.)
  regions.py         41 channels, 9 regions (from bci-erds)
  features/          one module per family; registry in __init__.py
  extract.py         python -m proxies.extract --family <name>
  evaluate.py        per-user feature ranks, top-k generality, leave-one-user-out XGB
run/                 sbatch wrappers + submit scripts
site/pinky.sh        UdeSA cluster profile
```

Outputs under `$WORK` (cluster: `/share/data1/mossf/data/bci-proxies`):
`features/<family>.npz`, `output/results/<tag>/{generality.csv, loo_users.csv,
unit_scores.npz, metrics.json}`.

## Run (cluster)

```bash
rsync -a --exclude work ~/projects/phd/bci-proxies/ udesa-cluster:~/projects/bci-proxies/
ssh udesa-cluster
cd ~/projects/bci-proxies
SITE=pinky bash setup.sh                 # login node: env + link official.npz
LIMIT=64 bash run/submit_extract.sh      # smoke test
bash run/submit_extract.sh               # all families (CPU jobs)
bash run/submit_eval.sh                  # joint + per-family leave-one-user-out
```
