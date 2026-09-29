#!/usr/bin/env bash
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
until [ -f results/core/e3_kernel/e3_kernel_61.json ]; do sleep 60; done
python -m authority_barriers.experiments.e3_kernel --grid 81 --threads 4 > results/logs_e3_kernel_81.log 2>&1 || echo "kernel81 FAILED" >> results/logs/queue_errors.log
echo done > results/logs/kernel81_done.flag
