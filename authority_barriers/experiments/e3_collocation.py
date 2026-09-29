"""E3, three dimensions (Table I row "Fig. collocation map"; Sec. VI Q2; Thm. 12(ii); M4.4).

For the N = 3 team (set A constants, m_L scaled) and three fixed cable configurations at rest
    A: all cables leaning toward the wall, z = (-0.15, -0.15, -0.15), w = (0.12, 0, -0.12);
    B: mixed, z = (-0.15, 0, 0.15), w = (0.1, -0.1, 0);
    C: all saturated, z = z_bar, w = 0,
and approach speeds v in {0.5, 1, ..., 4} m/s, the direct-collocation program of kernel.collocation3d
searches for a trajectory that respects C and U(x) at every knot and ends in X_RF. Per (config, v):
D = D(v, z, 0) (nominal data) and D_rel (relaxed data, Prop. 15) are computed; the search is first run
from h0 = D, where the braking maneuver is a feasible witness (a failure there is a solver problem and
is reported as such, never as evidence); then h0 is bisected in [0, D] for the smallest h_c(v) from
which a feasible trajectory was found (last feasible solution as warm start; a failed step is retried
from a fresh guess). Reported: h_c, D_rel, D, the M4.4 verdict D_rel - 1 cm <= h_c <= D + 1e-3, and the
independent re-simulation (verify_trajectory) of the best trajectory, which for h0 < D_rel - 1 cm would
be a candidate refutation of Thm. 12(ii). The optimizer under-approximates the kernel, so h_c is an
upper bound on the kernel boundary (a lower bound on the kernel).

    python -m authority_barriers.experiments.e3_collocation [--smoke] [--configs A,B,C] [--workers W] [--knots 41]
Writes results/core/e3_collocation.json and results/core/e3_collocation_summary.md.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from authority_barriers.theory import stopping as SD
from authority_barriers.theory.authority import BarrierData
from authority_barriers.theory.params import Params, load_set
from authority_barriers.theory.state import State
from authority_barriers.experiments.bench_library import with_N
from authority_barriers.viability.collocation3d import CollocationResult, find_trajectory, verify_trajectory

RESULTS = Path(os.environ.get("SIM_RESULTS_DIR", Path(__file__).resolve().parents[2] / "results")).resolve() / "core"
SPEEDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
M44_TOL_LOW, M44_TOL_HIGH = 0.01, 1e-3
M44_VERIFY_TOL = 1e-3          # verify_trajectory's tolerance on the re-simulated path violation (normalized units)


def team() -> Params:
    return with_N(load_set("A"), 3)


def configuration(name: str, p: Params):
    """(z, w) of the three fixed configurations (all at rest)."""
    if name == "A":
        return np.array([-0.15, -0.15, -0.15]), np.array([0.12, 0.0, -0.12])
    if name == "B":
        return np.array([-0.15, 0.0, 0.15]), np.array([0.1, -0.1, 0.0])
    if name == "C":
        return np.full(3, p.z_bar), np.zeros(3)
    raise ValueError(f"unknown configuration {name!r}")


def _solve_record(res: CollocationResult, attempt: str) -> dict:
    return {"h0": res.h0, "attempt": attempt, "success": res.success, "status": res.solver_status, "solver": res.solver,
            "info": res.solver_info, "solve_time": res.solve_time, "duration": res.duration, "H_end": res.H_end,
            "prog_violation": res.prog_violation, "path_violation": res.constraint_violation,
            "terminal_violation": res.terminal_violation, "tie_fraction": res.tie_fraction,
            "clamp_fraction": res.clamp_fraction, "n_ties_end": res.n_ties_end}


def run_pair(job: dict) -> dict:
    """Witness check at h0 = D, then bisection of h0 in [0, D]; returns one table row."""
    p = team()
    nominal, relaxed = BarrierData.nominal(p), BarrierData.relaxed(p)
    cfg, v = job["config"], float(job["v"])
    z, w = configuration(cfg, p)
    zero = np.zeros(p.N)
    D = float(SD.D_of(v, z, zero, nominal))
    D_rel = float(SD.D_of(v, z, zero, relaxed))
    kw = dict(knots=job["knots"], dt_min=job["dt_min"], dt_max=job["dt_max"], terminal="xrf", data=nominal,
              verbose=job.get("verbose", False), time_limit=job.get("time_limit"), cost=job.get("cost", "reg"),
              major_iterations=job.get("major_iterations", 500))
    x0_of = lambda h0: State.from_swing(p, h0, v, z, w, zero, zero)
    solves: list[dict] = []
    t_start = time.time()

    # 1. the witness: from H(x0) = 0 the maneuver replay is feasible, so a failure is a solver problem
    res_w = find_trajectory(p, x0_of(D), guess="plan", **kw)
    solves.append(_solve_record(res_w, "witness/plan"))
    if not res_w.success:
        res_w2 = find_trajectory(p, x0_of(D), guess="line", **kw)
        solves.append(_solve_record(res_w2, "witness/line"))
        if res_w2.success:
            res_w = res_w2
    witness_ok = bool(res_w.success)
    best = res_w if witness_ok else None

    # 2. bisection on h0 in [0, D]: lo infeasible (h0 = 0 with v > 0 hits the wall), hi feasible
    lo, hi = 0.0, D
    for _ in range(job["steps"]):
        mid = 0.5 * (lo + hi)
        first = "warm" if best is not None else "plan"
        res = find_trajectory(p, x0_of(mid), guess="plan", warm_start=best if first == "warm" else None, **kw)
        solves.append(_solve_record(res, f"bisect/{first}"))
        if not res.success:
            second = "plan" if first == "warm" else "line"
            res2 = find_trajectory(p, x0_of(mid), guess=second, **kw)
            solves.append(_solve_record(res2, f"bisect/{second}"))
            if res2.success:
                res = res2
        if res.success:
            hi, best = mid, res
        else:
            lo = mid
    h_c = hi

    # 3. independent verification of the best trajectory (a candidate refutation if h0 < D_rel - 1 cm)
    ver = verify_trajectory(p, best, nominal) if best is not None else None
    n_fail = sum(not s["success"] for s in solves)
    fail_status = sorted({s["status"] for s in solves if not s["success"]})
    row = {"config": cfg, "v": v, "D": D, "D_rel": D_rel, "h_c": h_c, "h_infeasible_max": lo, "major_iterations": job.get("major_iterations", 500),
           "h_c_over_D": h_c / D if D > 0 else np.nan, "witness_ok": witness_ok,
           "witness_status": solves[0]["status"], "M4_4": bool(D_rel - M44_TOL_LOW <= h_c <= D + M44_TOL_HIGH),
           "candidate_h0": (best.h0 if best is not None and best.h0 < D_rel - M44_TOL_LOW else None),
           "refutation_candidate": bool(ver["refutation_candidate"]) if ver else False,
           "verify_passes": bool(ver["passes"]) if ver else None,
           "verify_max_violation": ver["max_violation"] if ver else None,
           "verify_min_h_plan": ver["min_h_plan"] if ver else None,
           "verify_H_end_sim": ver["H_end_sim"] if ver else None,
           "n_solves": len(solves), "n_failures": n_fail, "failure_statuses": fail_status,
           "solver": solves[0]["solver"], "solve_times": [s["solve_time"] for s in solves],
           "mean_solve_time": float(np.mean([s["solve_time"] for s in solves])),
           "max_solve_time": float(np.max([s["solve_time"] for s in solves])),
           "tie_fraction_mean": float(np.mean([s["tie_fraction"] for s in solves])),
           "n_ties_end_best": best.n_ties_end if best is not None else None,
           "best_duration": best.duration if best is not None else None,
           "solves": solves, "verification": ver, "wall_time": time.time() - t_start}
    if best is not None:
        row["best_trajectory"] = {"times": best.times.tolist(), "x": best.x_traj.tolist(), "u": best.u_traj.tolist()}
    print(f"[{cfg} v={v:.1f}] D_rel={D_rel:.3f} h_c={h_c:.3f} D={D:.3f} witness={'ok' if witness_ok else 'FAIL'} "
          f"solves={len(solves)} failures={n_fail} mean solve {row['mean_solve_time']:.1f} s "
          f"M4.4={'PASS' if row['M4_4'] else 'FAIL'} candidate={row['refutation_candidate']}", flush=True)
    return row


def table(rows: list[dict]) -> str:
    cols = ["config", "v [m/s]", "D_rel [m]", "h_c [m]", "D [m]", "h_c/D", "witness", "M4.4", "solves", "fails",
            "mean solve [s]", "ties (end)", "verify"]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        vd = r.get("verification") or {}
        mv = vd.get("max_violation")
        if r["verify_passes"] is None:
            ver = "-"
        elif r["verify_passes"]:
            ver = "pass"
        elif mv is not None and mv == mv and mv > M44_VERIFY_TOL:            # re-simulation of the NLP solution off by more than the tolerance
            ver = f"resim {100 * mv:.1f} %" + (" + replay timeout" if vd.get("timeout") else "")
        else:
            ver = "replay timeout" if vd.get("timeout") else "FAIL"
        if r["refutation_candidate"]:
            ver += " CANDIDATE"
        out.append(f"| {r['config']} | {r['v']:.1f} | {r['D_rel']:.3f} | {r['h_c']:.3f} | {r['D']:.3f} | {r['h_c_over_D']:.2f} | "
                   f"{'ok' if r['witness_ok'] else 'FAIL'} | {'PASS' if r['M4_4'] else 'FAIL'} | {r['n_solves']} | {r['n_failures']} | "
                   f"{r['mean_solve_time']:.1f} | {r['n_ties_end_best']} | {ver} |")
    return "\n".join(out)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parents[2],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="configuration A only, v in {1, 3}, 3 bisection steps, 31 knots")
    ap.add_argument("--configs", default="A,B,C")
    ap.add_argument("--speeds", default=None, help="comma-separated speeds [m/s] (default 0.5, 1, ..., 4)")
    ap.add_argument("--workers", type=int, default=1, help="processes over (config, v) pairs (spawn)")
    ap.add_argument("--knots", type=int, default=41)
    ap.add_argument("--steps", type=int, default=6, help="bisection steps")
    ap.add_argument("--dt-min", type=float, default=0.01)
    ap.add_argument("--dt-max", type=float, default=0.1)
    ap.add_argument("--time-limit", type=float, default=900.0, help="SNOPT wall-time limit per NLP [s] (not enforced by the bundled SNOPT: solves end at the major-iteration limit)")
    ap.add_argument("--major-iterations", type=int, default=500, help="SNOPT major-iteration limit per NLP (the effective per-solve bound; a smaller limit is conservative: h_c can only rise)")
    ap.add_argument("--cost", default="reg", choices=["reg", "time", "time+reg", "none"])
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=None, help="output stem (default results/core/e3_collocation)")
    ap.add_argument("--cache-dir", default=None, help="per-(config, v) job cache (default results/core/e3_collocation_jobs): a pair already solved with the same settings is loaded, not re-solved")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args(argv)
    configs = args.configs.split(",")
    speeds = [float(s) for s in args.speeds.split(",")] if args.speeds else list(SPEEDS)
    knots, steps = args.knots, args.steps
    if args.smoke:
        configs, speeds, steps, knots = ["A"], [1.0, 3.0], 3, 31
    jobs = [{"config": c, "v": v, "knots": knots, "steps": steps, "dt_min": args.dt_min, "dt_max": args.dt_max,
             "time_limit": args.time_limit, "cost": args.cost, "verbose": args.verbose, "major_iterations": args.major_iterations} for c in configs for v in speeds]
    t0 = time.time()
    cache = None if args.no_cache else (Path(args.cache_dir) if args.cache_dir else RESULTS / "e3_collocation_jobs")

    def key(config, v):
        mi = "" if args.major_iterations == 500 else f"_mi{args.major_iterations}"
        return f"{config}_v{float(v):g}_k{knots}_s{steps}_dt{args.dt_min:g}-{args.dt_max:g}_{args.cost}{mi}.json"

    rows, todo = [], []
    for j in jobs:
        f = cache / key(j["config"], j["v"]) if cache is not None else None
        if f is not None and f.exists():
            try:
                rows.append(json.loads(f.read_text()))
                print(f"cache: loaded {f.name}", flush=True)
                continue
            except Exception as exc:
                print(f"cache: could not load {f}: {exc!r}; re-solving", flush=True)
        todo.append(j)

    def done(row):
        if cache is not None:                      # save each pair the moment it completes (resumable)
            cache.mkdir(parents=True, exist_ok=True)
            f = cache / key(row["config"], row["v"])
            tmp = f.with_name(f.name + ".tmp")
            tmp.write_text(json.dumps(row, default=_json_default))
            os.replace(tmp, f)
        rows.append(row)

    print(f"collocation: {len(rows)} of {len(jobs)} (config, v) pairs cached; solving {len(todo)}", flush=True)
    if args.workers > 1 and len(todo) > 1:
        with get_context("spawn").Pool(min(args.workers, len(todo))) as pool:
            for row in pool.imap_unordered(run_pair, todo, chunksize=1):
                done(row)
    else:
        for j in todo:
            done(run_pair(j))
    rows.sort(key=lambda r: (r["config"], r["v"]))
    p = team()
    solvers = sorted({r["solver"] for r in rows})
    n_solves = sum(r["n_solves"] for r in rows)
    n_fail = sum(r["n_failures"] for r in rows)
    tab = table(rows)
    md = (f"# E3 collocation map (Thm. 12(ii); M4.4) -- {'smoke' if args.smoke else 'full'} run\n\n"
          f"N = 3 team ({p.name}: m_L = {p.m_L} kg, T_bar = {p.T_bar[0]} N, T_min = {p.T_min} N, f_max = {p.f_max[0]} N, "
          f"a_max = {p.a_max} m/s^2, z_bar = {p.z_bar:.4f}, w_bar = {p.w_bar:.4f}, nu = {p.nu}, nu_w = {p.nu_w}, "
          f"omega_bar = {p.omega_bar} rad/s, alpha_sat = {p.alpha_sat:.3f}, alpha_sat_rel = {p.alpha_sat_rel:.2f} m/s^2). "
          f"Direct collocation (pydrake 1.51.1, {'/'.join(solvers)}), {knots} knots, equal steps in "
          f"[{args.dt_min}, {args.dt_max}] s, terminal set H >= 1e-4 m and swing states in V (a subset of X_RF), "
          f"cost {args.cost}, {steps} bisection steps on h0 in [0, D] (resolution D/{2 ** steps}), SNOPT major-iteration limit {args.major_iterations}. "
          f"h_c is an upper bound on the kernel boundary (the optimizer under-approximates the kernel); "
          f"a solver failure is never a refutation.\n\n{tab}\n\n"
          f"NLPs solved: {n_solves}, failures: {n_fail} ({n_fail / max(n_solves, 1):.0%}); "
          f"witness failures (h0 = D): {sum(not r['witness_ok'] for r in rows)}; "
          f"M4.4 rows passing: {sum(r['M4_4'] for r in rows)}/{len(rows)}; "
          f"refutation candidates: {sum(r['refutation_candidate'] for r in rows)}; "
          f"tie fraction of H evaluations (mean over NLPs): {np.mean([r['tie_fraction_mean'] for r in rows]):.1%}; "
          f"wall {time.time() - t0:.0f} s, workers {args.workers}, commit {_git_commit()}.\n")
    stem = Path(args.out) if args.out else RESULTS / "e3_collocation"
    stem.parent.mkdir(parents=True, exist_ok=True)
    Path(str(stem) + "_summary.md").write_text(md)
    Path(str(stem) + ".json").write_text(json.dumps(
        {"rows": rows, "params": p.to_dict(), "settings": vars(args), "solvers": solvers, "n_solves": n_solves,
         "n_failures": n_fail, "wall_time": time.time() - t0, "commit": _git_commit()}, indent=1, default=_json_default))
    print(md)
    return rows


if __name__ == "__main__":
    main()
