#!/usr/bin/env bash
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
until [ -f results/logs/p3_perfect_done.flag ]; do sleep 60; done
python -m authority_barriers.experiments.e3_kernel --grid 121 --zn 31 --h-max 10 --threads 4 > results/logs_e3_kernel_121.log 2>&1 || echo "kernel121 FAILED" >> results/logs/queue_errors.log
echo done > results/logs/kernel121_done.flag
