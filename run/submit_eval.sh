#!/usr/bin/env bash
# Leave-one-user-out (user = subject x session) evaluation grid:
#   all deployable families together, sessions as users and strict cross-subject;
#   each family alone; the privileged classic_ref baseline.
#   DEPEND=afterok:<id>:<id> bash run/submit_eval.sh   # start after extraction
set -euo pipefail
source "${PROXIES_HOME:-$HOME/projects/bci-proxies}/site/${SITE:-pinky}.sh"
sub() { sbatch ${DEPEND:+--dependency=$DEPEND --kill-on-invalid-dep=yes} -J "px_eval_$1" -o "$LOGS/%x_%j.out" "$PROXIES_HOME/run/evaluate.sbatch" "$@"; }
sub all
sub all_strict --strict-subject
for f in $(python -c "from proxies.features import FAMILIES; print(*FAMILIES)"); do
    sub "$f" --families "$f"
done
