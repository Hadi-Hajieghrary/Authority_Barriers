#!/usr/bin/env bash
# Finish the necessary track (P4-P6): every step is resumable (trial / job caches, kernel checkpoint), so after
# an interruption simply re-run this script. Progress: results/logs/finish_core.progress; per-step logs alongside.
#   chain K: 4-D kernel on the refined grid (4 threads) -> full 3-D collocation (6 workers)
#   chain D: Drake trials (4 workers): E1 attitude, E2 attitude A/C, E5 (2 actuators x 2 layers), E1 N = 3 replication
# E4 runs first, alone (its runtime benchmark M5.1 needs a quiet machine); the core figures are rebuilt at the end.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
L=results/logs; mkdir -p "$L"
export PYTHONUNBUFFERED=1
P="$L/finish_core.progress"
run() {  # run <name> <cmd...>: log to $L/<name>.log, record the outcome
  local name=$1; shift
  echo "[$(date '+%F %T')] $name: start" >> "$P"
  if "$@" > "$L/$name.log" 2>&1; then
    echo "[$(date '+%F %T')] $name: ok" >> "$P"
  else
    echo "[$(date '+%F %T')] $name: FAILED (see $L/$name.log)" >> "$P"; echo "$name FAILED $(date '+%F %T')" >> "$L/queue_errors.log"; return 1
  fi
}
echo "[$(date '+%F %T')] ===== run_finish_core.sh started (pid $$) =====" >> "$P"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
[ -f results/core/e4/summary.md ] || run e4_prop17 python -m authority_barriers.experiments.e4_prop17 --n-eval 1000 --logs "results/core/e2_perfect_max_off0/*.npz"
(
  [ -f results/core/e3_kernel/e3_kernel_121x31_h10.json ] || \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 run e3_kernel_121 python -m authority_barriers.experiments.e3_kernel --grid 121 --zn 31 --h-max 10 --threads 4
  run e3_collocation_full python -m authority_barriers.experiments.e3_collocation --configs A,B,C --workers 6
  touch "$L/chainK_done.flag"
) &
(
  W="--workers 4"
  run e1_attitude python -m authority_barriers.experiments.e1_thm5 --actuator attitude $W
  run e2_attitude_A python -m authority_barriers.experiments.e2_thm12i --actuator attitude --rows maximizers --h-offset 0 $W
  run e2_attitude_C python -m authority_barriers.experiments.e2_thm12i --actuator attitude --rows maximizers --h-offset 0 --layer-abs 1.0 $W
  run e5_perfect python -m authority_barriers.experiments.e5_robust --scale 1 --actuator perfect $W
  run e5_perfect_layer1 python -m authority_barriers.experiments.e5_robust --scale 1 --actuator perfect --layer-abs 1.0 $W
  run e5_attitude python -m authority_barriers.experiments.e5_robust --scale 1 --actuator attitude $W
  run e5_attitude_layer1 python -m authority_barriers.experiments.e5_robust --scale 1 --actuator attitude --layer-abs 1.0 $W
  run e1_perfect_N3 python -m authority_barriers.experiments.e1_thm5 --actuator perfect --params A_N3 --n 30 $W
  touch "$L/chainD_done.flag"
) &
wait
run figures_core python -m authority_barriers.experiments.make_figures --core
echo "[$(date '+%F %T')] ===== run_finish_core.sh finished =====" >> "$P"
touch "$L/finish_core_done.flag"
