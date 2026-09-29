#!/usr/bin/env bash
# Optional track, second launch: corridor (with the 1200 s wall cap) -> gentler-nominal subset -> Monte Carlo. Resumable.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
L=results/logs; mkdir -p "$L"; P="$L/optional.progress"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run() {
  local name=$1; shift
  echo "[$(date '+%F %T')] $name: start" >> "$P"
  if "$@" > "$L/$name.log" 2>&1; then echo "[$(date '+%F %T')] $name: ok" >> "$P"
  else echo "[$(date '+%F %T')] $name: FAILED (see $L/$name.log)" >> "$P"; echo "$name FAILED $(date '+%F %T')" >> "$L/queue_errors.log"; return 1; fi
}
echo "[$(date '+%F %T')] ===== run_optional2.sh started (pid $$) =====" >> "$P"
run e7_corridor python -m authority_barriers.experiments.e7_corridor --n 50 --workers 4
run e7_corridor_gentle python -m authority_barriers.experiments.e7_corridor --gentle --n 10 --workers 4
run e7_montecarlo python -m authority_barriers.experiments.e7_montecarlo --n 500 --workers 6
touch "$L/optional_done.flag"
echo "[$(date '+%F %T')] ===== run_optional2.sh finished =====" >> "$P"
