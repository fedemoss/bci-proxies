# Source this before anything else, on a laptop or a cluster:
#   source ~/projects/bci-proxies/env.sh
#
# Code lives with this file; everything generated lives under $WORK:
# input/ (staged windows), features/ (one file per feature family), output/.
# On a cluster set WORK to a scratch filesystem first -- see site/.
PROXIES_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROXIES_HOME

export WORK="${WORK:-$PROXIES_HOME/work}"
export INPUT="$WORK/input"
export FEATURES="$WORK/features"
export OUTPUT="$WORK/output"
export RESULTS="$OUTPUT/results"
export LOGS="$OUTPUT/logs"

# The windows are the official Track 02 ones, produced by bci-erds
# (erds.official: the competition's own loader, split and preprocessing).
# setup.sh links that file here instead of re-running the loader.
export OFFICIAL="$INPUT/official.npz"
export ERDS_OFFICIAL="${ERDS_OFFICIAL:-$PROXIES_HOME/../bci-erds/work/cluster/input/official.npz}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"   # parallelism is joblib's
export N_JOBS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROXIES_HOME${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$INPUT" "$FEATURES" "$RESULTS" "$LOGS"

_conda_base="${CONDA_BASE:-$(conda info --base 2>/dev/null)}"
if [ -n "$_conda_base" ] && [ -f "$_conda_base/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$_conda_base/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV:-proxies}" 2>/dev/null \
        || echo "[env] conda env '${CONDA_ENV:-proxies}' not found — run setup.sh first"
fi
