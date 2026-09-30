# `scripts/` — build script and launch sequences

Every script changes to the repository root first, so it can be started from any directory.

## Build script

| Script | What it does | Output |
|---|---|---|
| `build_paper.sh` | compiles the paper with `latexmk` | `IEEE_ACC2027/build/main.pdf` |

## Launch sequences (`run_*.sh`)

The experiments behind the paper were started by these scripts. Each one starts a sequence of experiment drivers
with their arguments and worker counts, writes one log per step to `results/logs/` and, where several chains shared
the machine, waits for the completion flag of an earlier chain. They write into `results/`; a driver skips every
trial, collocation job or kernel state that is already on disk, so a script resumes after an interruption and does
not repeat finished work.

To run the experiments, use `python -m authority_barriers.reproduce`, which runs the same drivers in one sequence.
The scripts show how the runs were scheduled on one machine.

| Script | Phase | What it runs |
|---|---|---|
| `run_p3_perfect.sh`, `run_p3_rest.sh` | P3 | E1 and E2 with the perfect actuator: rows at the maximizers and at all local maxima from the boundary layer, rows at the maximizers with H(x0) in [0, 1] m, and a variant with a calibrated state margin |
| `run_queue_kernel.sh`, `run_queue_kernel121.sh` | P4 | the 4-D kernel on the grids 81 and 121 × 31 (h ≤ 10 m), 4 threads, after the grid 61 and the P3 chain. No script contains the grids 61 and 121 × 41; they are run with `--grid 61` and `--grid 121 --zn 41` (`authority_barriers/experiments/README.md`, Running) |
| `run_queue_drake.sh` | P3, P5 | E1 and E2 with the attitude loop, E5 and E4, after the P3 chain |
| `run_queue_collocation.sh` | P4 | the three-dimensional collocation map, after the Drake chain |
| `run_finish_core.sh` | P4 to P6 | E4 alone (its timing needs a quiet machine), then the finest kernel and the collocation in one chain and the Drake steps in another, then the core figures |
| `run_finish_core_D2.sh`, `run_finish_core_D3.sh` | P3, P5 | the Drake chain with the wall-clock cap of 2400 s per trial (C-12), E1 with the attitude loop completed from the cache, the core figures rebuilt |
| `run_optional.sh`, `run_optional2.sh`, `run_optional_extra.sh` | P7, P8 | gusts at 2 d̄, the stress list, the corridor (wall-clock cap of 1200 s) and its gentler-nominal subset, the Monte Carlo study |
| `run_rerun_c15.sh` | P8 | E1 (perfect actuator, N = 3, attitude loop), the corridor with its subset and the Monte Carlo study with the transport nominal of compromise C-15 |

Phases, decisions (D-) and compromises (C-) are those of the plan of the simulation study;
`authority_barriers/README.md`, Sec. 4, explains the terms.
