#!/usr/bin/env bash
# Optional track (P7-P8) runs; every step resumes from its caches. Waits for E6 (backup comparison) to finish so that
# the box is not oversubscribed while the 6-D kernel runs on 8 threads. Re-run after an interruption.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
L=results/logs; mkdir -p "$L"; P="$L/optional.progress"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run() {
  local name=$1; shift
  echo "[$(date '+%F %T')] $name: start" >> "$P"
  if "$@" > "$L/$name.log" 2>&1; then echo "[$(date '+%F %T')] $name: ok" >> "$P"
  else echo "[$(date '+%F %T')] $name: FAILED (see $L/$name.log)" >> "$P"; echo "$name FAILED $(date '+%F %T')" >> "$L/queue_errors.log"; return 1; fi
}
until [ -f results/optional/e6_backup/summary.md ]; do sleep 60; done
echo "[$(date '+%F %T')] ===== run_optional.sh started (pid $$) =====" >> "$P"
W="--workers ${WORKERS:-4}"
run e7_gusts_2dbar python -m authority_barriers.experiments.e5_robust --scale 2 --actuator perfect --h-offset 0.8 --layer-abs 0.5 $W
run e7_stress python -m authority_barriers.experiments.e7_stress --variant all $W
run e7_corridor python -m authority_barriers.experiments.e7_corridor --n 50 $W
run e7_montecarlo python -m authority_barriers.experiments.e7_montecarlo --n 500 --workers ${WORKERS_MC:-6}
touch "$L/optional_done.flag"
echo "[$(date '+%F %T')] ===== run_optional.sh finished =====" >> "$P"
