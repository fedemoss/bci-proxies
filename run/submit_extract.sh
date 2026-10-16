#!/usr/bin/env bash
# Submit one extraction job per family (they are independent).
#   bash run/submit_extract.sh                 # every family
#   bash run/submit_extract.sh spectral hoi    # a subset
#   LIMIT=64 bash run/submit_extract.sh        # smoke test on 64 windows
set -euo pipefail
source "${PROXIES_HOME:-$HOME/projects/bci-proxies}/site/${SITE:-pinky}.sh"
FAMS=("$@")
[ ${#FAMS[@]} -eq 0 ] && FAMS=($(python -c "from proxies.features import FAMILIES; print(*FAMILIES)"))
for f in "${FAMS[@]}"; do
    sbatch -J "px_$f" -o "$LOGS/%x_%j.out" "$PROXIES_HOME/run/extract.sbatch" "$f" \
        ${LIMIT:+--limit $LIMIT}
done
