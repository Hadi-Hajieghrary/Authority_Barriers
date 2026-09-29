#!/usr/bin/env bash
# full 3-D collocation search after the Drake chain has released the CPU and the smoke run has finished
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
until [ -f results/logs/p3_perfect_done.flag ] && [ -f results/core/e3_collocation_summary.md ]; do sleep 60; done
mv results/core/e3_collocation_summary.md results/core/e3_collocation_smoke_summary.md 2>/dev/null || true
mv results/core/e3_collocation.json results/core/e3_collocation_smoke.json 2>/dev/null || true
python -m authority_barriers.experiments.e3_collocation --configs A,B,C > results/logs/e3_collocation_full.log 2>&1 || echo "collocation_full FAILED" >> results/logs/queue_errors.log
echo done > results/logs/collocation_done.flag
