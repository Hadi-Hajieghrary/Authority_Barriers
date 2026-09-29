#!/usr/bin/env bash
# P3 necessary-track runs, perfect actuator (paper's actuator model); 4 workers to leave CPU for the kernel run
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
mkdir -p results/logs
python -m authority_barriers.experiments.e1_thm5 --actuator perfect --workers 4 > results/logs/e1_perfect.log 2>&1
python -m authority_barriers.experiments.e2_thm12i --actuator perfect --rows maximizers --h-offset 0 --workers 4 > results/logs/e2_perfect_A.log 2>&1
python -m authority_barriers.experiments.e2_thm12i --actuator perfect --rows local_maxima --h-offset 0 --workers 4 > results/logs/e2_perfect_B.log 2>&1
python -m authority_barriers.experiments.e2_thm12i --actuator perfect --rows local_maxima --h-offset auto --workers 4 > results/logs/e2_perfect_C.log 2>&1
echo "P3 perfect-actuator runs done" > results/logs/p3_perfect_done.flag
