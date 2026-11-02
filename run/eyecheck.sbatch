#!/bin/bash
#SBATCH -N 1
#SBATCH -t 0-02:00:00
#SBATCH --cpus-per-task 8
#
# CPU only: are the frontal low-frequency proxies eye movements?
#   sbatch -J px_eyecheck -o $LOGS/%x_%j.out run/eyecheck.sbatch [--run all]
set -uo pipefail
source "${PROXIES_HOME:-$HOME/projects/bci-proxies}/site/${SITE:-pinky}.sh"
echo "[job] eyecheck $* on $(hostname), $(date)"
cd "$PROXIES_HOME"
python -m proxies.eyecheck "$@" && echo "EYECHECK DONE"
