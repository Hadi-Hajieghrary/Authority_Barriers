"""One-command reproduction (M6.2 / M9.2).

  python -m authority_barriers.reproduce --core --results <dir>   # necessary track: E0, E1, E2, E3 (kernel + collocation), E4, E5, core figures
  python -m authority_barriers.reproduce --core --smoke           # reduced trial counts and grids; writes to results_smoke
  python -m authority_barriers.reproduce --all --results <dir>    # adds the optional track (E6, E3b, E7 stress list, corridor, Monte Carlo) and its figures
  python -m authority_barriers.reproduce --all --overwrite        # the same in the default directory, over the results of an earlier run

The results of an earlier run are never rewritten unless --overwrite is given: without --results, --smoke or
--overwrite the run stops if the results directory already holds summaries.

Every step is a subprocess with its own log under <results>/logs/; a failing step stops the run. Every driver
caches finished trials / (config, v) pairs / kernel integration states, so re-running after an interruption
resumes where it stopped. With the [viz] extra installed, a full run ends with the rendered figures: the trial-level
figures, the fixed-camera sequences and the figures of the paper (IEEE_ACC2027/collect_figures.py then takes the files
the paper includes).
Requirements: the venv with pydrake 1.51.1 and `pip install -e .`.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--kernel-threads", type=int, default=4)
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--skip-kernel", action="store_true")
    ap.add_argument("--results", default=None, help="results directory (default results; results_smoke with --smoke)")
    ap.add_argument("--overwrite", action="store_true", help="rewrite the results that the default results directory holds")
    args = ap.parse_args()
    if not (args.core or args.all):
        ap.error("choose --core or --all")
    results = Path(args.results).resolve() if args.results else (ROOT / ("results_smoke" if args.smoke else "results"))
    if args.results is None and not args.smoke and not args.overwrite and any(results.glob("*/*/summary.md")):
        ap.error(f"{results} holds results of an earlier run. Reproduce into another directory with --results <dir>, "
                 "or rewrite the recorded summaries and figures in place with --overwrite "
                 "(finished trials are taken from the cache in both cases).")
    env = {**os.environ, "SIM_RESULTS_DIR": str(results), "PYTHONUNBUFFERED": "1"}
    logs = results / "logs"

    def step(name: str, cmd: list[str]) -> None:
        logs.mkdir(parents=True, exist_ok=True)
        log = logs / f"{name}.log"
        t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] {name}: {' '.join(['python' if c == sys.executable else c for c in cmd])}", flush=True)
        with open(log, "w") as fh:
            r = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, text=True, env=env)
        print(f"    -> {'ok' if r.returncode == 0 else 'FAILED'} in {time.time() - t0:.0f} s (log: {log.relative_to(results)})", flush=True)
        if r.returncode != 0:
            print(Path(log).read_text()[-3000:])
            sys.exit(r.returncode)

    py = [sys.executable, "-m"]
    sm = ["--smoke"] if args.smoke else []
    W = ["--workers", str(args.workers)]
    if not args.skip_tests and (ROOT / "tests").is_dir():
        step("tests", [sys.executable, "-m", "pytest", "-q", "tests", "-x"])
    step("e0_worked_example", py + ["authority_barriers.experiments.e0_worked_example"])
    step("params", py + ["authority_barriers.theory.params", "A", "A1", "A2", "A_N3"])
    # E1 (Thm. 5) and E2 (Thm. 12(i)) in both actuator modes (D-19); E2 variants A (paper's filter, boundary layer),
    # C (state margin: H0 uniform in [0, 1] m, D-20 revised) and, for the paper's actuator model only, the reverted
    # D-21 comparison B (rows at all local maxima)
    for act in ("perfect", "attitude"):
        step(f"e1_{act}", py + ["authority_barriers.experiments.e1_thm5", "--actuator", act] + sm + W)
        step(f"e2_{act}_A", py + ["authority_barriers.experiments.e2_thm12i", "--actuator", act, "--rows", "maximizers", "--h-offset", "0"] + sm + W)
        step(f"e2_{act}_C", py + ["authority_barriers.experiments.e2_thm12i", "--actuator", act, "--rows", "maximizers", "--h-offset", "0", "--layer-abs", "1.0"] + sm + W)
        if act == "perfect":
            step("e2_perfect_B", py + ["authority_barriers.experiments.e2_thm12i", "--actuator", act, "--rows", "local_maxima", "--h-offset", "0"] + sm + W)
    step("e1_perfect_N3", py + ["authority_barriers.experiments.e1_thm5", "--actuator", "perfect", "--params", "A_N3", "--n", "30"] + sm + W)   # D-9 replication
    if not args.skip_kernel:
        kt = ["--threads", str(args.kernel_threads)]
        if args.smoke:
            step("e3_kernel_31", py + ["authority_barriers.experiments.e3_kernel", "--grid", "31"] + kt)
            step("e3_collocation", py + ["authority_barriers.experiments.e3_collocation", "--configs", "A", "--speeds", "1", "--knots", "11", "--steps", "1", "--time-limit", "120"])
        else:
            for g in (["--grid", "61"], ["--grid", "81"], ["--grid", "121", "--zn", "31", "--h-max", "10"], ["--grid", "121", "--zn", "41", "--h-max", "10"]):
                step("e3_kernel_" + g[1] + (f"x{g[3]}_h10" if g[1] == "121" else ""), py + ["authority_barriers.experiments.e3_kernel"] + g + kt)
            step("e3_kernel_crosscheck_hj", py + ["authority_barriers.viability.crosscheck_hj", "--validate", "--grid", "61", "--threads", str(args.kernel_threads)])   # C-7
            step("e3_collocation", py + ["authority_barriers.experiments.e3_collocation", "--configs", "A,B,C"] + W)
    e2_logs = results / "core" / "e2_perfect_max_off0" / "*.npz"                 # relative to the repository where possible
    step("e4_prop17", py + ["authority_barriers.experiments.e4_prop17", "--n-eval", "200" if args.smoke else "1000",
                            "--logs", str(e2_logs.relative_to(ROOT) if e2_logs.is_relative_to(ROOT) else e2_logs)])
    for act in ("perfect", "attitude"):
        step(f"e5_{act}", py + ["authority_barriers.experiments.e5_robust", "--scale", "1", "--actuator", act] + sm + W)
        step(f"e5_{act}_layer1", py + ["authority_barriers.experiments.e5_robust", "--scale", "1", "--actuator", act, "--layer-abs", "1.0"] + sm + W)
    step("figures_core", py + ["authority_barriers.experiments.make_figures", "--core"])
    if args.all:                                  # optional track (P7-P9, §13 of the plan)
        step("e6_backup_compare", py + ["authority_barriers.experiments.e6_backup_compare", "--n", "2" if args.smoke else "30", "--workers", "3"])
        step("e6_distributed", py + ["authority_barriers.experiments.e6_distributed", "--n", "2" if args.smoke else "10"] + W)
        if not args.skip_kernel:
            step("e3b_kernel6d", py + ["authority_barriers.experiments.e3b_kernel6d", "--shape"] + (["9", "9", "5", "5", "5", "5"] if args.smoke else ["25", "25", "13", "13", "13", "13"]) + kt)
        step("e7_gusts_2dbar", py + ["authority_barriers.experiments.e5_robust", "--scale", "2", "--actuator", "perfect", "--h-offset", "0.8", "--layer-abs", "0.5"] + sm + W)
        step("e7_stress", py + ["authority_barriers.experiments.e7_stress", "--variant", "all"] + sm + W)
        step("e7_corridor", py + ["authority_barriers.experiments.e7_corridor"] + sm + W)
        step("e7_corridor_gentle", py + ["authority_barriers.experiments.e7_corridor", "--gentle", "--n", "10"] + sm + W)   # D-7 sensitivity subset
        step("e7_montecarlo", py + ["authority_barriers.experiments.e7_montecarlo"] + sm + W)
        step("figures_optional", py + ["authority_barriers.experiments.make_figures_optional"])
    if importlib.util.find_spec("playwright") is not None and not args.smoke:   # Meshcat renders of the headline trials need the [viz] extra
        step("figures_trials", py + ["authority_barriers.experiments.trial_figures"])
        step("figures_sequences", py + ["authority_barriers.experiments.sequence_figures"])
        step("figures_paper", py + ["authority_barriers.experiments.paper_figures"])
    print("done")


if __name__ == "__main__":
    main()
