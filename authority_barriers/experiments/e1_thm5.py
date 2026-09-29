"""E1 (Q1, Table I row 1): loss of feasibility of the HOCBF filter from x0 in D (Thm. 5), three
gain pairs, plus the proposed filter from the same states (M3.1, M3.2).

Per trial (GT-A): the onset t_inf is the first tick at which the HOCBF program is infeasible AND
mu < 0, persisting for two consecutive ticks; PASS iff t_inf <= tau - dt, h(t_inf) > 0, q in
Sigma_y at t_inf, and ||qdot_i|| <= omega_bar on [0, t_inf]; also the proof bound mu <= alpha2(psi1)
on [0, tau]. A trial with the HOCBF feasible through tau under the hypothesis refutes Thm. 5.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from authority_barriers.theory import hocbf as HB
from authority_barriers.theory.authority import BarrierData
from authority_barriers.theory.filters import ProposedFilter
from authority_barriers.theory.params import load_set
from authority_barriers.theory.sampling import sample_D
from authority_barriers.simulator.harness import TrialConfig
from authority_barriers.experiments.common import in_XRF, out_dir, run_many_cached, sampled_allowance, save_json, summary_table

GAIN_PAIRS = [(1.0, 20.0), (0.5, 10.0), (2.0, 40.0)]


def gtag(k1, k2):
    """File-name tag of a gain pair without decimal points (e.g. k1_20, k0p5_10)."""
    return f"k{k1:g}_{k2:g}".replace(".", "p")


def verdict_hocbf(log, p, a1, a2, tol=1e-6):
    d = log.d
    t, h, mu, psi1, feas = d("t"), d("h"), d("mu"), d("psi1"), d("feasible")
    in_sig, max_om = d("in_Sigma_y"), d("max_omega")
    x0 = log.meta["x0"]
    from authority_barriers.theory.state import State
    st0 = State(np.array(x0["x_L"]), np.array(x0["v_L"]), np.array(x0["q"]), np.array(x0["qd"]))
    tau, t_psi = HB.tau(st0, p), HB.t_psi(st0.h(p), st0.v(p), a1)
    inf = (feas < 0.5) & (mu < 0)
    onset = None
    for k in range(t.size - 1):
        if inf[k] and inf[k + 1]:
            onset = k
            break
    upto = t <= tau + 1e-12
    bound_ok = bool(np.all(mu[upto] <= a2(psi1[upto]) + tol))
    hyp_ok = bool(np.all(max_om[: (onset + 1) if onset is not None else np.sum(upto)] <= p.omega_bar + 1e-6))
    res = {"tau": tau, "t_psi": t_psi, "onset": None if onset is None else float(t[onset]),
           "h_at_onset": None if onset is None else float(h[onset]),
           "in_Sigma_y_at_onset": None if onset is None else bool(in_sig[onset] > 0.5),
           "proof_bound_ok": bound_ok, "hypothesis_ok": hyp_ok, "min_h": float(np.min(h)),
           "termination": log.termination, "wall_time": log.meta["wall_time"]}
    res["pass"] = bool(onset is not None and t[onset] <= tau - p.dt_filter + 1e-12 and h[onset] > 0
                       and in_sig[onset] > 0.5 and hyp_ok)
    res["refutes"] = bool(onset is None and hyp_ok and np.all(feas[upto] > 0.5))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--params", default="A")
    ap.add_argument("--t-final", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-proposed", action="store_true")
    ap.add_argument("--actuator", default="perfect", choices=["perfect", "attitude"])
    ap.add_argument("--wall-cap", type=float, default=None, help="wall-clock seconds per trial (default: 2400 for the attitude loop, none otherwise); "
                    "a trial stopped by the cap is reported as truncated and counted neither as a pass nor as a violation (C-12)")
    args = ap.parse_args()
    if args.wall_cap is None and args.actuator == "attitude":
        args.wall_cap = 2400.0
    if args.smoke:
        args.n, args.t_final = 4, 3.0
    p = load_set(args.params)
    data = BarrierData.nominal(p)
    out = out_dir("core", f"e1_{args.actuator}" + ("" if args.params == "A" else f"_{args.params}"))   # D-9: N = 3 replication in its own folder
    rows, all_results = [], {}
    t_start = time.time()
    for (k1, k2) in GAIN_PAIRS:
        a1, a2 = HB.LinearClassK(k1), HB.LinearClassK(k2)
        rng = np.random.default_rng(int(1000 * k1 + k2))
        x0s = sample_D(rng, p, args.n, a1, a2, v_range=(2.0, 4.0), dt=p.dt_filter)
        cfgs = [TrialConfig(params=args.params, filter="hocbf", gains=(k1, k2), nominal="velocity",
                            v_cmd=float(x0.v(p)), t_final=args.t_final, seed=i, log_mu=True, actuator=args.actuator, wall_cap=args.wall_cap, label=f"e1_{gtag(k1, k2)}_{i}")
                for i, x0 in enumerate(x0s)]
        logs, _ = run_many_cached(cfgs, x0s, out, args.workers)
        results = [verdict_hocbf(L, p, a1, a2) for L in logs]
        n_pass = sum(r["pass"] for r in results)
        n_ref = sum(r["refutes"] for r in results)
        n_hyp = sum(not r["hypothesis_ok"] for r in results)
        n_bound = sum(r["proof_bound_ok"] for r in results)
        n_trunc_h = sum(r["termination"] == "wall_cap" for r in results)
        row = {"gains": f"({k1}, {k2})", "trials": len(results), "PASS": n_pass, "refutations": n_ref, "HOCBF truncated (wall cap)": n_trunc_h,
               "hypothesis violated": n_hyp, "proof bound holds": n_bound,
               "median onset [s]": f"{np.median([r['onset'] for r in results if r['onset'] is not None]):.3f}" if n_pass else "-",
               "median tau [s]": f"{np.median([r['tau'] for r in results]):.3f}"}
        if not args.skip_proposed:
            cfgs2 = [TrialConfig(params=args.params, filter="proposed", nominal="velocity", v_cmd=float(x0.v(p)),
                                 t_final=args.t_final, seed=i, actuator=args.actuator, wall_cap=args.wall_cap, label=f"e1p_{gtag(k1, k2)}_{i}") for i, x0 in enumerate(x0s)]
            logs2, _ = run_many_cached(cfgs2, x0s, out, args.workers)
            n_in, n_safe_in, n_allow_in, n_safe_out, n_trunc = 0, 0, 0, 0, 0
            for L, x0, cfg in zip(logs2, x0s, cfgs2):
                inrf, H0 = in_XRF(x0, p, data)
                if L.termination == "wall_cap":                   # truncated by the wall-time cap: no verdict (C-12)
                    n_trunc += 1
                    n_in += inrf
                    continue
                safe = float(np.min(L.d("h"))) >= 0.0 and L.termination != "wall_contact"
                _, _, within = sampled_allowance(L, p)
                n_in += inrf
                n_safe_in += (inrf and safe)
                n_allow_in += (inrf and (not safe) and within)
                n_safe_out += ((not inrf) and safe)
            row.update({"x0 in X_RF": n_in, "proposed safe | in X_RF": f"{n_safe_in}/{n_in}",
                        "contact within sampled allowance | in X_RF": n_allow_in,
                        "proposed safe | not in X_RF": f"{n_safe_out}/{len(x0s) - n_in}", "truncated (wall cap)": n_trunc})
        rows.append(row)
        all_results[f"{k1},{k2}"] = results
        print(summary_table([row], list(row.keys())))
    table = summary_table(rows, list(rows[0].keys()))
    (out / "summary.md").write_text(f"# E1 (Thm. 5) — {'smoke' if args.smoke else 'full'} run\n\n{table}\n\n"
                                    f"Constants: set {args.params} (N = {p.N}, m_L = {p.m_L} kg); actuator {args.actuator}; dt = {p.dt_filter}; wall {time.time()-t_start:.0f} s\n")
    save_json(out / "results.json", {"rows": rows, "per_trial": all_results, "params": p.to_dict()})
    print(table)


if __name__ == "__main__":
    main()
