"""E6, distributed implementation (optional track; Sec. VI Q3; M7.2; D-27): Prop. 17(b) with a one-step-lagged broadcast
of a_hat against the centralized filter from the same states on the full-order plant (paper's actuator model, adversarial
nominal): |Delta H(t)|, the a_hat mismatch (charged to the d_bar budget), contacts and infeasible ticks of both.

  python -m authority_barriers.experiments.e6_distributed [--n 10] [--workers 2]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from authority_barriers.theory.authority import BarrierData
from authority_barriers.theory.params import load_set
from authority_barriers.theory.sampling import sample_XRF_layer
from authority_barriers.simulator.harness import TrialConfig
from authority_barriers.experiments.common import out_dir, run_many_cached, save_json, summary_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--t-final", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    p = load_set("A"); data = BarrierData.nominal(p)
    out = out_dir("optional", "e6_distributed")
    rng = np.random.default_rng(args.seed)
    x0s = sample_XRF_layer(rng, p, args.n, data, v_range=(1.0, 4.0), delta_frac=0.0, h_offset=0.8, delta_abs=0.5)   # H0 in [0.8, 1.3] m (D-23)
    t0 = time.time()
    per = []
    logs = {}
    for variant in ("proposed", "distributed"):
        cfgs = [TrialConfig(filter=variant, nominal="adversarial", t_final=args.t_final, seed=i, actuator="perfect", label=f"e6d_{variant}_{i}")
                for i in range(len(x0s))]
        logs[variant], _ = run_many_cached(cfgs, x0s, out, args.workers)
    for i in range(len(x0s)):
        Lc, Ld = logs["proposed"][i], logs["distributed"][i]
        n = min(Lc.t_diag.size, Ld.t_diag.size)
        dH = np.abs(Lc.d("H")[:n] - Ld.d("H")[:n])
        aerr = Ld.d("a_hat_err"); lam = Ld.d("lambda")
        per.append({"trial": i, "H0": float(Lc.d("H")[0]), "v0": float(Lc.d("v")[0]),
                    "central": {"min_h": float(Lc.d("h").min()), "min_H": float(Lc.d("H").min()), "contact": Lc.termination == "wall_contact",
                                "infeasible_ticks": int(np.sum(Lc.d("feasible") < 0.5)), "t_end": float(Lc.t_diag[-1])},
                    "distributed": {"min_h": float(Ld.d("h").min()), "min_H": float(Ld.d("H").min()), "contact": Ld.termination == "wall_contact",
                                    "infeasible_ticks": int(np.sum(Ld.d("feasible") < 0.5)), "t_end": float(Ld.t_diag[-1]),
                                    "a_hat_err_median": float(np.nanmedian(aerr)), "a_hat_err_max": float(np.nanmax(aerr)),
                                    "a_hat_err_force_max_N": float(np.max(p.m_arr) * np.nanmax(aerr)), "lambda_median": float(np.nanmedian(lam))},
                    "dH_max": float(dH.max()), "dH_median": float(np.median(dH)), "common_ticks": int(n)})
    budget = 0.05
    row = {"trials": len(per), "central contacts": sum(r["central"]["contact"] for r in per), "distributed contacts": sum(r["distributed"]["contact"] for r in per),
           "central infeasible ticks": sum(r["central"]["infeasible_ticks"] for r in per), "distributed infeasible ticks": sum(r["distributed"]["infeasible_ticks"] for r in per),
           "max |Delta H| [m]": f"{max(r['dH_max'] for r in per):.3f}", "median |Delta H| [m]": f"{np.median([r['dH_median'] for r in per]):.3f}",
           "a_hat mismatch median [m/s^2]": f"{np.median([r['distributed']['a_hat_err_median'] for r in per]):.3f}",
           "a_hat mismatch max, force-equivalent m_i |Delta a| [N]": f"{max(r['distributed']['a_hat_err_force_max_N'] for r in per):.2f}",
           "budget for the a_hat lag (D-16) [N]": budget, "wall [s]": f"{time.time() - t0:.0f}"}
    table = summary_table([row], list(row.keys()))
    within = max(r["distributed"]["a_hat_err_force_max_N"] for r in per) <= budget
    verdict = "PASS" if within and row["distributed contacts"] == row["central contacts"] else "FAIL (the one-step-lagged broadcast is not a commitment: the mismatch exceeds the budget; see the entry)"
    md = (f"# E6, distributed variant (Prop. 17(b), one-step-lagged a_hat; M7.2; D-27) — {verdict}\n\n{table}\n\n"
          f"Per trial (central vs distributed): " + "; ".join(f"{r['trial']}: H0 {r['H0']:.2f} m, min h {r['central']['min_h']:.3f}/{r['distributed']['min_h']:.3f} m, "
          f"contact {int(r['central']['contact'])}/{int(r['distributed']['contact'])}, infeasible ticks {r['central']['infeasible_ticks']}/{r['distributed']['infeasible_ticks']}, "
          f"a_hat mismatch median {r['distributed']['a_hat_err_median']:.2f} m/s^2, lambda median {r['distributed']['lambda_median']:.2f}" for r in per) + "\n")
    (out / "summary.md").write_text(md)
    save_json(out / "results.json", {"row": row, "per_trial": per, "verdict": verdict, "params": p.to_dict()})
    print(md)


if __name__ == "__main__":
    main()
