#!/bin/bash
#SBATCH -N 1
#SBATCH -t 0-01:00:00
#SBATCH --cpus-per-task 4
#
# CPU only: summary tables and figures from the evaluation results.
#   sbatch -J px_analyze -o $LOGS/%x_%j.out run/analyze.sbatch [--run all]
set -uo pipefail
source "${PROXIES_HOME:-$HOME/projects/bci-proxies}/site/${SITE:-pinky}.sh"
echo "[job] analyze $* on $(hostname), $(date)"
cd "$PROXIES_HOME"
python -m proxies.analyze "$@" && echo "ANALYZE DONE"
