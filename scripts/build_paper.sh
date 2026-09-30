#!/usr/bin/env bash
# Build the paper: compile IEEE_ACC2027/main.tex into IEEE_ACC2027/build/main.pdf.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
cd IEEE_ACC2027
latexmk
echo "IEEE_ACC2027/build/main.pdf"
