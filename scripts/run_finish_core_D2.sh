#!/usr/bin/env bash
# Chain D, second launch (C-12): the Drake steps with the attitude-mode extras trimmed after measuring
# 30-45 s of wall time per simulated second for attitude-mode trials in which the filter is active
# (radau3 crawls through the inner loop's command-jump transients, C-10; RK3 is 3x faster but differs
# by ~1 mm in h, the scale of the reported contact depths, so the integrator is kept).
#   E1 attitude: full protocol (100 states x 3 gain pairs, both filters), resumed from the trial cache
#   E2 attitude A: 100 states, t_final 10 s (perfect-mode contacts all occur within the first seconds)
#   E2 attitude C: first 30 of the 100 states of the sampler, t_final 10 s
#   E5 perfect: full (50 + 50, 15 s); E5 attitude: 20 states, boundary layer only, 10 s
#   E1 N = 3 replication (D-9): 30 states x 3 gain pairs, perfect actuator
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
L=results/logs; mkdir -p "$L"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
P="$L/finish_core.progress"
run() {
  local name=$1; shift
  echo "[$(date '+%F %T')] $name: start" >> "$P"
  if "$@" > "$L/$name.log" 2>&1; then echo "[$(date '+%F %T')] $name: ok" >> "$P"
  else echo "[$(date '+%F %T')] $name: FAILED (see $L/$name.log)" >> "$P"; echo "$name FAILED $(date '+%F %T')" >> "$L/queue_errors.log"; return 1; fi
}
echo "[$(date '+%F %T')] ===== run_finish_core_D2.sh started (pid $$) =====" >> "$P"
W="--workers ${WORKERS:-4}"
run e1_attitude python -m authority_barriers.experiments.e1_thm5 --actuator attitude $W
run e2_attitude_A python -m authority_barriers.experiments.e2_thm12i --actuator attitude --rows maximizers --h-offset 0 --t-final 10 $W
run e2_attitude_C python -m authority_barriers.experiments.e2_thm12i --actuator attitude --rows maximizers --h-offset 0 --layer-abs 1.0 --n 30 --t-final 10 $W
run e5_perfect python -m authority_barriers.experiments.e5_robust --scale 1 --actuator perfect $W
run e5_perfect_layer1 python -m authority_barriers.experiments.e5_robust --scale 1 --actuator perfect --layer-abs 1.0 $W
run e5_attitude python -m authority_barriers.experiments.e5_robust --scale 1 --actuator attitude --n 20 --t-final 10 $W
run e1_perfect_N3 python -m authority_barriers.experiments.e1_thm5 --actuator perfect --params A_N3 --n 30 $W
touch "$L/chainD_done.flag"
echo "[$(date '+%F %T')] ===== run_finish_core_D2.sh finished =====" >> "$P"
