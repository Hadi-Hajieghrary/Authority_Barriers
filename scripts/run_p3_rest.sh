#!/usr/bin/env bash
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
python -m authority_barriers.experiments.e2_thm12i --actuator perfect --rows maximizers --h-offset 0 --layer-abs 1.0 --workers 4 > results/logs/e2_perfect_C.log 2>&1 || echo "e2_perfect_C FAILED" >> results/logs/queue_errors.log
echo done > results/logs/p3_perfect_done.flag
