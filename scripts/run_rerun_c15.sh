#!/usr/bin/env bash
# C-15 (2026-09-26): the transport/hover nominal lacked the cable feed-forward m_i P_i a; every experiment that used it is
# re-run from the corrected nominal (the trial cache re-runs logs recorded without `nominal_version`): E1 (both filters, both
# actuator models, N = 3 replication), the corridor and its gentle subset, the Monte Carlo. Resumable: re-run after an interruption.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
L=results/logs; mkdir -p "$L"; P="$L/rerun_c15.progress"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run() {
  local name=$1; shift
  echo "[$(date '+%F %T')] $name: start" >> "$P"
  if "$@" > "$L/$name.log" 2>&1; then echo "[$(date '+%F %T')] $name: ok" >> "$P"
  else echo "[$(date '+%F %T')] $name: FAILED (see $L/$name.log)" >> "$P"; echo "$name FAILED $(date '+%F %T')" >> "$L/queue_errors.log"; return 1; fi
}
echo "[$(date '+%F %T')] ===== run_rerun_c15.sh started (pid $$) =====" >> "$P"
run e1_perfect python -m authority_barriers.experiments.e1_thm5 --actuator perfect --workers 8
run e1_perfect_N3 python -m authority_barriers.experiments.e1_thm5 --actuator perfect --params A_N3 --n 30 --workers 8
run e7_corridor python -m authority_barriers.experiments.e7_corridor --n 50 --workers 8
run e7_corridor_gentle python -m authority_barriers.experiments.e7_corridor --gentle --n 10 --workers 8
run e7_montecarlo python -m authority_barriers.experiments.e7_montecarlo --n 500 --workers 8
touch "$L/rerun_c15_main_done.flag"
run e1_attitude python -m authority_barriers.experiments.e1_thm5 --actuator attitude --workers 8
touch "$L/rerun_c15_done.flag"
echo "[$(date '+%F %T')] ===== run_rerun_c15.sh finished =====" >> "$P"
