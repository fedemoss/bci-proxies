#!/bin/bash
#SBATCH -N 1
#SBATCH -t 0-02:00:00
#SBATCH --cpus-per-task 8
#
# CPU only: choose the L_feat targets without the test subjects.
#   sbatch -J px_terms -o $LOGS/%x_%j.out run/select_terms.sbatch [--max-blink-r 0.3]
set -uo pipefail
source "${PROXIES_HOME:-$HOME/projects/bci-proxies}/site/${SITE:-pinky}.sh"
echo "[job] select_terms $* on $(hostname), $(date)"
cd "$PROXIES_HOME"
python -m proxies.select_terms "$@" && echo "TERMS DONE"
