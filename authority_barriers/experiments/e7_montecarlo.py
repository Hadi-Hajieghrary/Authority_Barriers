"""E7 Monte Carlo (optional track; Sec. VI: "a Monte Carlo campaign of 500 trials over all of these reports the violation
rate, the number of infeasible instances, the minimum margins, and the computation time"; M8.6; D-24, D-28).
Per trial (seed 1000 + k) the stress factors are drawn: gust scale U[0, 2] x d_bar (wind unmodeled by the nominal filter),
f_max U[20, 44] N (plant and filter consistent; Assumptions 7/13/14 recorded with min_margin 0), true m_L = U[0.8, 1.2] x nominal (filter nominal; a true mass above the nominal violates the model hypothesis, D-26),
cable stiffness 5e4 or 5e3 N/m (p = 0.3), initial swing rates x U[0.5, 1.5], H(x0) U[0.8, 1.5] m, v0 U[1, 4] m/s; velocity
transport at v0 toward the wall, 10 s, paper's actuator model. Baselines: unfiltered nominal, HOCBF on h (gains (1, 20)),
HOCBF + per-quadrotor cable barriers, proposed filter. The numerically integrated backup filter is not run in closed loop
(D-24 revised: ~90 s per evaluation at 200 Hz); its equivalence to the proposed filter and its cost are E6's result.

  python -m authority_barriers.experiments.e7_montecarlo [--n 500] [--workers 6] [--baselines none,hocbf,cable_cbf,proposed]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from authority_barriers.theory.authority import BarrierData
from authority_barriers.theory.params import all_ok, check_assumptions, load_set
from authority_barriers.theory.sampling import sample_XRF_layer
from authority_barriers.theory.state import State
from authority_barriers.simulator.harness import TrialConfig, apply_overrides
from authority_barriers.experiments.common import in_XRF, out_dir, run_many_cached, save_json, summary_table

BASELINES = {"none": "unfiltered nominal", "hocbf": "HOCBF on h, gains (1, 20)", "cable_cbf": "HOCBF + per-quadrotor cable barriers", "proposed": "proposed filter (17)"}


def draw(seed: int, p, data):
    rng = np.random.default_rng(seed)
    f = {"gust": float(rng.uniform(0.0, 2.0)), "f_max": float(rng.uniform(20.0, 44.0)), "m_L_true": float(p.m_L * rng.uniform(0.8, 1.2)),
         "k_factor": float(0.1 if rng.uniform() < 0.3 else 1.0), "qd_factor": float(rng.uniform(0.5, 1.5)),
         "H0": float(rng.uniform(0.8, 1.5)), "v0": float(rng.uniform(1.0, 4.0))}
    st = sample_XRF_layer(rng, p, 1, data, v_range=(f["v0"], f["v0"] + 1e-9), delta_frac=0.0, h_offset=f["H0"], delta_abs=1e-6)[0]
    st = State(st.x_L, st.v_L, st.q, st.qd * f["qd_factor"])
    q = apply_overrides(p, TrialConfig(param_overrides={"f_max": f["f_max"]}))
    f["assumptions_hold"] = all_ok(check_assumptions(q), min_margin=0.0)   # Assumptions 7/13/14 hold (no design margin required, as in the stress list)
    f["mass_ok"] = bool(f["m_L_true"] <= p.m_L + 1e-9)                       # the assumed mass is an upper bound of the true one (D-26: a heavier payload voids the barrier)
    f["outside_Xop"] = bool(np.any(np.sqrt(st.omega2()) > p.omega_bar))
    f["in_XRF"] = bool(in_XRF(st, p, data)[0])                                # after the swing-rate scaling: the hypothesis of Thm. 12(i)
    return st, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--baselines", default="none,hocbf,cable_cbf,proposed")
    ap.add_argument("--t-final", type=float, default=10.0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n, args.t_final = 3, 3.0
    p = load_set("A"); data = BarrierData.nominal(p)
    out = out_dir("optional", "e7_montecarlo")
    seeds = [1000 + k for k in range(args.n)]
    draws = [draw(s, p, data) for s in seeds]
    x0s = [st for st, _ in draws]; factors = [f for _, f in draws]
    t0 = time.time()
    results = {}
    for name in args.baselines.split(","):
        cfgs = [TrialConfig(filter=name, nominal="velocity", v_cmd=f["v0"], t_final=args.t_final, seed=s, actuator="perfect", wind_scale=f["gust"],
                            param_overrides={"f_max": f["f_max"]}, m_L_true=f["m_L_true"], cable_k_factor=f["k_factor"], gains=(1.0, 20.0),
                            wall_cap=1200.0, label=f"e7mc_{name}_{s}") for s, f in zip(seeds, factors)]
        logs, _ = run_many_cached(cfgs, x0s, out, args.workers)
        per = []
        for L, f, s in zip(logs, factors, seeds):
            feas = L.d("feasible")
            per.append({"seed": s, **f, "contact": L.termination == "wall_contact", "truncated": L.termination == "wall_cap", "min_h": float(L.d("h").min()), "min_H": float(np.nanmin(L.d("H"))),
                        "infeasible_ticks": int(np.sum(feas < 0.5)) if name != "none" else 0, "solve_time_median_ms": float(1e3 * np.median(L.d("solve_time"))),
                        "t_end": float(L.t_diag[-1]), "slack_ticks": int(np.sum(np.any(L.cable[:p.N] <= 0.0, axis=0)))})
        results[name] = per
    rows = []
    for name, per in results.items():
        n = len(per)
        within = [r for r in per if r["assumptions_hold"] and r["in_XRF"] and r["gust"] <= 1.0 and r["k_factor"] == 1.0 and r["mass_ok"]]
        rows.append({"baseline": BASELINES[name], "trials": n, "wall contacts": sum(r["contact"] for r in per), "contact rate": f"{np.mean([r['contact'] for r in per]):.2f}",
                     "truncated (1200 s cap)": sum(r["truncated"] for r in per),
                     "contacts within the hypotheses (Ass. 7, x0 in X_RF, gusts <= d_bar, stiff cables, true m_L <= nominal)": f"{sum(r['contact'] for r in within)}/{len(within)}",
                     "trials with infeasible ticks": sum(r["infeasible_ticks"] > 0 for r in per), "infeasible ticks": sum(r["infeasible_ticks"] for r in per),
                     "min h [m]": f"{min(r['min_h'] for r in per):.3f}", "median min H [m]": f"{np.median([r['min_H'] for r in per]):.3f}",
                     "solve time median [ms]": f"{np.median([r['solve_time_median_ms'] for r in per]):.2f}"})
    table = summary_table(rows, list(rows[0].keys()))
    # the proposed filter's contacts by stress factor
    prop = results.get("proposed", [])
    def rate(sel):
        return f"{sum(r['contact'] for r in sel)}/{len(sel)}" if sel else "-"
    breakdown = ""
    if prop:
        bd = [{"factor": "f_max < Assumption-7 threshold", "contacts/trials": rate([r for r in prop if not r["assumptions_hold"]])},
              {"factor": "f_max with the assumptions holding", "contacts/trials": rate([r for r in prop if r["assumptions_hold"]])},
              {"factor": "initial swing rate above omega_bar", "contacts/trials": rate([r for r in prop if r["outside_Xop"]])},
              {"factor": "x0 outside X_RF after the swing-rate scaling (Thm. 12(i) does not apply)", "contacts/trials": rate([r for r in prop if not r["in_XRF"]])},
              {"factor": "x0 in X_RF", "contacts/trials": rate([r for r in prop if r["in_XRF"]])},
              {"factor": "gusts above d_bar", "contacts/trials": rate([r for r in prop if r["gust"] > 1.0])},
              {"factor": "gusts at or below d_bar", "contacts/trials": rate([r for r in prop if r["gust"] <= 1.0])},
              {"factor": "soft cables (k/10)", "contacts/trials": rate([r for r in prop if r["k_factor"] < 1.0])},
              {"factor": "true mass above the nominal (the nominal filter; the D-26 rule not applied)", "contacts/trials": rate([r for r in prop if not r["mass_ok"]])},
              {"factor": "true mass at or below the nominal", "contacts/trials": rate([r for r in prop if r["mass_ok"]])},
              {"factor": "every hypothesis satisfied (Ass. 7, x0 in X_RF, gusts <= d_bar, stiff cables, true m_L <= nominal)", "contacts/trials": rate([r for r in prop if r["assumptions_hold"] and r["in_XRF"] and r["gust"] <= 1.0 and r["k_factor"] == 1.0 and r["mass_ok"]])}]
        breakdown = "\n\nProposed filter, contacts by stress factor (factors overlap):\n\n" + summary_table(bd, ["factor", "contacts/trials"])
    md = (f"# E7 Monte Carlo (Sec. VI limits; M8.6; D-24, D-28) — {args.n} trials x {len(results)} baselines, paper's actuator model, velocity transport at v0 toward the wall, "
          f"{args.t_final:g} s{' (smoke)' if args.smoke else ''}; wall {time.time() - t0:.0f} s with {args.workers} workers\n\n{table}{breakdown}\n\n"
          "The numerically integrated backup filter is not run in closed loop (D-24 revised): E6 measured ~90 s per evaluation against 4 ms for the closed form, "
          "with commands within 1 % of f_max of the proposed filter's on X_RF, so its closed-loop behaviour is that of the proposed filter at a cost that excludes it from a 200 Hz loop.\n")
    (out / ("summary_smoke.md" if args.smoke else "summary.md")).write_text(md)
    save_json(out / ("results_smoke.json" if args.smoke else "results.json"), {"rows": rows, "per_baseline": results, "params": p.to_dict()})
    print(md)


if __name__ == "__main__":
    main()
