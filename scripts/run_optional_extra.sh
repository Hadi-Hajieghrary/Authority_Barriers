#!/usr/bin/env bash
# After the main corridor run: the sensitivity subset with a gentler transport nominal (10 states).
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
L=results/logs; P="$L/optional.progress"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
until [ -f results/optional/e7_corridor/summary.md ]; do sleep 60; done
echo "[$(date '+%F %T')] e7_corridor_gentle: start" >> "$P"
if python -m authority_barriers.experiments.e7_corridor --gentle --n 10 --workers 2 > "$L/e7_corridor_gentle.log" 2>&1; then echo "[$(date '+%F %T')] e7_corridor_gentle: ok" >> "$P"; else echo "[$(date '+%F %T')] e7_corridor_gentle: FAILED" >> "$P"; fi
