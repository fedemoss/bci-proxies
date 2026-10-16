#!/bin/bash
#SBATCH -N 1
#SBATCH -t 0-12:00:00
#SBATCH --cpus-per-task 16
#
# Leave-one-user-out XGBoost + generality. Everything after the tag goes to
# proxies.evaluate:
#   sbatch -J px_eval_all -o $LOGS/%x_%j.out run/evaluate.sbatch all
#   sbatch -J px_eval_spec -o $LOGS/%x_%j.out run/evaluate.sbatch spectral --families spectral
set -uo pipefail
TAG="${1:?tag}"; shift
source "${PROXIES_HOME:-$HOME/projects/bci-proxies}/site/${SITE:-pinky}.sh"
echo "[job] $TAG | $* on $(hostname), $(date)"
cd "$PROXIES_HOME"
python -m proxies.evaluate --out "$RESULTS/$TAG" "$@" && echo "EVAL DONE $TAG"
