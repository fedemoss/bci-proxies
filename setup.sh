#!/usr/bin/env bash
# One-time setup, LOGIN node (needs network):
#
#   SITE=pinky bash setup.sh
#
# Creates the `proxies` env (CPU only: numpy/scipy/antropy/xgboost) and links
# the official Track 02 windows produced by bci-erds.
set -euo pipefail

PROXIES_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${SITE:-}" ] && [ -f "$PROXIES_HOME/site/$SITE.sh" ]; then
    # shellcheck disable=SC1090
    source "$PROXIES_HOME/site/$SITE.sh"
else
    # shellcheck disable=SC1091
    source "$PROXIES_HOME/env.sh"
fi

_conda_base="${CONDA_BASE:-$(conda info --base)}"
# shellcheck disable=SC1091
source "$_conda_base/etc/profile.d/conda.sh"
ENV_NAME="${CONDA_ENV:-proxies}"
if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
    conda create -y -q -n "$ENV_NAME" python=3.11
fi
conda activate "$ENV_NAME"
case "$(command -v python)" in
    "$CONDA_PREFIX"/*) ;;
    *) echo "ERROR: python is $(command -v python), not inside $CONDA_PREFIX"; exit 1 ;;
esac
python -m pip install -q -r "$PROXIES_HOME/requirements.txt"

if [ ! -f "$ERDS_OFFICIAL" ]; then
    echo "ERROR: $ERDS_OFFICIAL missing — run bci-erds run/prepare.sbatch first"; exit 1
fi
ln -sfn "$ERDS_OFFICIAL" "$OFFICIAL"

python - <<'PY'
import importlib.metadata as m
for p in ("numpy", "scipy", "antropy", "xgboost", "scikit-learn"):
    print(p, m.version(p))
PY
echo "Next:  bash run/submit_extract.sh"
