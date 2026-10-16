# UdeSA cluster (pinky). Source this INSTEAD of env.sh:
#   source ~/projects/bci-proxies/site/pinky.sh
#
# Site policy: only code in $HOME (nearly full), everything else on
# /share/data1. Login node = network (downloads, installs) only; all compute
# goes through sbatch. This project is CPU only: no --gres needed.
export WORK="${WORK:-/share/data1/mossf/data/bci-proxies}"
export ERDS_OFFICIAL="${ERDS_OFFICIAL:-/share/data1/mossf/data/bci-erds/input/official.npz}"
export CONDA_BASE="/share/data1/mossf/miniconda3"
export CONDA_ENV="${CONDA_ENV:-proxies}"

export SCRATCH_CACHE="${SCRATCH_CACHE:-/share/data1/mossf/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SCRATCH_CACHE/pip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH_CACHE}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$SCRATCH_CACHE/numba}"
mkdir -p "$PIP_CACHE_DIR" "$NUMBA_CACHE_DIR"

# `module load` can return non-zero here; under a caller's `set -e` that would
# silently kill the script, so never let it fail.
if command -v module >/dev/null 2>&1; then
    module load python/3.11.3 >/dev/null 2>&1 || true
fi

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
