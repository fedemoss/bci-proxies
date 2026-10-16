#!/bin/bash
#SBATCH -N 1
#SBATCH -t 0-12:00:00
#SBATCH --cpus-per-task 16
#
# CPU only: one feature family over all official windows.
#   sbatch -J px_spectral -o $LOGS/%x_%j.out run/extract.sbatch spectral [--limit 64]
set -uo pipefail
FAM="${1:?family}"; shift
source "${PROXIES_HOME:-$HOME/projects/bci-proxies}/site/${SITE:-pinky}.sh"
echo "[job] $FAM on $(hostname) with $N_JOBS cpus, $(date)"
cd "$PROXIES_HOME"
python -m proxies.extract --family "$FAM" "$@" && echo "EXTRACT DONE $FAM"
