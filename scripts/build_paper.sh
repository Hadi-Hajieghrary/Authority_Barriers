#!/usr/bin/env bash
# Build the paper: compile IEEE_ACC2027/main.tex into IEEE_ACC2027/build/main.pdf. Where a results directory exists
# (results/ or $SIM_RESULTS_DIR, written by the experiments), the figure files of IEEE_ACC2027/figures/ are first
# verified against it; with --collect they are first written from it.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [ -d "${SIM_RESULTS_DIR:-results}" ]; then
  if [ "${1:-}" = "--collect" ]; then
    python IEEE_ACC2027/collect_figures.py
  fi
  python IEEE_ACC2027/collect_figures.py --check
elif [ "${1:-}" = "--collect" ]; then
  echo "no results directory: run the experiments first (python -m authority_barriers.reproduce)" >&2
  exit 1
fi
cd IEEE_ACC2027
latexmk
echo "IEEE_ACC2027/build/main.pdf"
