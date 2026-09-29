#!/usr/bin/env bash
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
until [ -f results/logs/p3_perfect_done.flag ]; do sleep 60; done
python -m authority_barriers.experiments.e1_thm5 --actuator attitude --workers 4 > results/logs/e1_attitude.log 2>&1 || echo "e1_attitude FAILED" >> results/logs/queue_errors.log
python -m authority_barriers.experiments.e2_thm12i --actuator attitude --rows maximizers --h-offset 0 --workers 4 > results/logs/e2_attitude_A.log 2>&1 || echo "e2_attitude_A FAILED" >> results/logs/queue_errors.log
python -m authority_barriers.experiments.e2_thm12i --actuator attitude --rows maximizers --h-offset 0 --layer-abs 1.0 --workers 4 > results/logs/e2_attitude_C.log 2>&1 || echo "e2_attitude_C FAILED" >> results/logs/queue_errors.log
python -m authority_barriers.experiments.e5_robust --scale 1 --workers 4 > results/logs/e5.log 2>&1 || echo "e5 FAILED" >> results/logs/queue_errors.log
python -m authority_barriers.experiments.e4_prop17 --n-eval 1000 --logs "results/core/e2_perfect_max_off0/*.npz" > results/logs/e4.log 2>&1 || echo "e4 FAILED" >> results/logs/queue_errors.log
echo done > results/logs/p3p5_done.flag
