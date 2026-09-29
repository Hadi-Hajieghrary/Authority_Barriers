#!/usr/bin/env bash
# Chain D, third launch: after chain D2 has finished, complete E1-attitude from the trial cache (19 trials were
# missing when the step was stopped: three of them had stalled for hours, C-12; they now run under the 2400 s
# wall-time cap and are reported as truncated if the cap fires), then rebuild the core figures and tables.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
L=results/logs; P="$L/finish_core.progress"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run() {
  local name=$1; shift
  echo "[$(date '+%F %T')] $name: start" >> "$P"
  if "$@" > "$L/$name.log" 2>&1; then echo "[$(date '+%F %T')] $name: ok" >> "$P"
  else echo "[$(date '+%F %T')] $name: FAILED (see $L/$name.log)" >> "$P"; echo "$name FAILED $(date '+%F %T')" >> "$L/queue_errors.log"; return 1; fi
}
until [ -f "$L/chainD_done.flag" ]; do sleep 60; done
echo "[$(date '+%F %T')] ===== run_finish_core_D3.sh started (pid $$) =====" >> "$P"
run e1_attitude python -m authority_barriers.experiments.e1_thm5 --actuator attitude --workers ${WORKERS:-4}
run figures_core python -m authority_barriers.experiments.make_figures --core
touch "$L/chainD3_done.flag"
echo "[$(date '+%F %T')] ===== run_finish_core_D3.sh finished =====" >> "$P"
