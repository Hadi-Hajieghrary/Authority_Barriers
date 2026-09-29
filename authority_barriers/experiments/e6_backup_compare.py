"""E6 (optional track; Sec. VI Q3; M7.1): the numerically integrated backup-CBF filter against the proposed filter
on X_RF states: |D_num - D| / D, the difference of the commanded thrusts ||u_backup - u_proposed|| against
1 % f_max (both from the same adversarial nominal), the wall-time ratio of one filter evaluation, and the ratio
of integrator policy calls to breakpoints of the closed form (the practicality claim of Sec. VI: "faster by the
ratio of the number of integration steps to the number of breakpoints").

  python -m authority_barriers.experiments.e6_backup_compare [--n 100] [--workers 3]
"""
from __future__ import annotations

import argparse
import time
from multiprocessing import get_context

import numpy as np

from authority_barriers.theory import stopping as SD
from authority_barriers.theory.authority import BarrierData
from authority_barriers.theory.filters import BackupIntegratedFilter, ProposedFilter
from authority_barriers.theory.params import load_set
from authority_barriers.theory.sampling import sample_XRF_layer
from authority_barriers.theory.state import State
from authority_barriers.simulator.nominal import AdversarialNominal
from authority_barriers.experiments.common import out_dir, save_json, summary_table


def one(job):
    x0, layer = job
    p = load_set("A"); data = BarrierData.nominal(p)
    st = State(*(np.array(x0[k]) for k in ("x_L", "v_L", "q", "qd")))
    prop = ProposedFilter(p, data)
    back = BackupIntegratedFilter(p, data)
    u_nom = AdversarialNominal(p)(0.0, st)
    cf = SD.evaluate(st.v(p), st.z(p), st.zd(p), data)
    t0 = time.perf_counter(); rp = prop.solve(st, u_nom); tp = time.perf_counter() - t0
    t0 = time.perf_counter(); rb = back.solve(st, u_nom); tb = time.perf_counter() - t0        # includes the one numerical evaluation
    num = back.last_res
    du = float(np.max(np.linalg.norm(rb.u - rp.u, axis=1)))
    gz_err = float(np.max(np.abs(num.grad_zeta[0] - cf.grad_zeta[0]) / np.maximum(np.abs(cf.grad_zeta[0]), 1e-3)))
    gw_err = float(np.max(np.abs(num.grad_omega[0] - cf.grad_omega[0]) / np.maximum(np.abs(cf.grad_omega[0]), 1e-3)))
    return {"layer": layer, "v0": float(st.v(p)), "H0": float(st.h(p) - cf.D), "D": cf.D, "D_num": num.D, "rel_dD": abs(num.D - cf.D) / max(cf.D, 1e-6),
            "t_star": float(cf.t_star[0]), "t_star_num": float(num.t_star[0]), "grad_v_num": num.grad_v, "grad_v": float(cf.t_star[0]),
            "grad_zeta_rel_err": gz_err, "grad_omega_rel_err": gw_err, "n_breakpoints": int(cf.n_breakpoints),
            "policy_calls": int(num.n_policy_calls), "integrations": int(num.n_integrations), "timeouts": int(num.timeouts),
            "eval_time_num": num.eval_time, "t_proposed": tp, "t_backup": tb, "du_max": du, "du_frac_fmax": du / p.f_max[0],
            "feasible_proposed": bool(rp.feasible), "feasible_backup": bool(rb.feasible), "hdot_proposed": float(rp.hdot[0]) if rp.hdot.size else np.nan,
            "hdot_backup": float(rb.hdot[0]) if rb.hdot.size else np.nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="states per layer (boundary layer of X_RF and H0 in [0, 1] m); one evaluation costs ~90 s")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=6)
    args = ap.parse_args()
    p = load_set("A"); data = BarrierData.nominal(p)
    rng = np.random.default_rng(args.seed)
    states = [(s, "boundary layer [0, 0.05 D]") for s in sample_XRF_layer(rng, p, args.n, data, v_range=(1.0, 4.0))]
    states += [(s, "H0 in [0, 1] m") for s in sample_XRF_layer(rng, p, args.n, data, v_range=(1.0, 4.0), delta_frac=0.0, delta_abs=1.0)]
    jobs = [({"x_L": s.x_L.tolist(), "v_L": s.v_L.tolist(), "q": s.q.tolist(), "qd": s.qd.tolist()}, layer) for s, layer in states]
    t0 = time.time()
    if args.workers > 1:
        with get_context("spawn").Pool(args.workers) as pool:
            rows = pool.map(one, jobs, chunksize=2)
    else:
        rows = [one(j) for j in jobs]
    out = out_dir("optional", "e6_backup")
    rel = np.array([r["rel_dD"] for r in rows]); du = np.array([r["du_frac_fmax"] for r in rows])
    ratio_t = np.array([r["t_backup"] / r["t_proposed"] for r in rows]); ratio_c = np.array([r["policy_calls"] / max(r["n_breakpoints"], 1) for r in rows])
    gz = np.array([r["grad_zeta_rel_err"] for r in rows]); gw = np.array([r["grad_omega_rel_err"] for r in rows])
    table = summary_table([
        {"metric": "|D_num - D| / D", "value": f"max {rel.max():.1e}, median {np.median(rel):.1e}", "target": "<= 1e-3", "verdict": "PASS" if rel.max() <= 1e-3 else "FAIL"},
        {"metric": "gradient rel. error (zeta, omega), FD delta 1e-4", "value": f"max {gz.max():.1e} / {gw.max():.1e}", "target": "report", "verdict": "-"},
        {"metric": "||u_backup - u_proposed|| / f_max (same nominal)", "value": f"max {du.max():.2e}, median {np.median(du):.2e}; within 1 %: {int(np.sum(du <= 0.01))}/{len(rows)}", "target": "<= 1 % f_max", "verdict": "PASS" if du.max() <= 0.01 else f"{int(np.sum(du > 0.01))} states above"},
        {"metric": "wall time per filter evaluation, backup / proposed", "value": f"median {np.median(ratio_t):.0f}x (backup {1e3*np.median([r['t_backup'] for r in rows]):.0f} ms, proposed {1e3*np.median([r['t_proposed'] for r in rows]):.1f} ms)", "target": "report", "verdict": "-"},
        {"metric": "integrator policy calls per evaluation / breakpoints of the closed form", "value": f"median {np.median(ratio_c):.0f} ({np.median([r['policy_calls'] for r in rows]):.0f} calls over {np.median([r['integrations'] for r in rows]):.0f} integrations vs {np.median([r['n_breakpoints'] for r in rows]):.0f} breakpoints)", "target": "the predicted speed ratio", "verdict": "-"},
        {"metric": "integration timeouts (20 s cap) / evaluations", "value": f"{sum(r['timeouts'] for r in rows)} / {len(rows)}", "target": "report", "verdict": "-"},
        {"metric": "feasible (proposed, backup)", "value": f"{sum(r['feasible_proposed'] for r in rows)}, {sum(r['feasible_backup'] for r in rows)} of {len(rows)}", "target": "equal", "verdict": "PASS" if sum(r['feasible_proposed'] for r in rows) == sum(r['feasible_backup'] for r in rows) else "differ"},
    ], ["metric", "value", "target", "verdict"])
    md = (f"# E6 (Sec. VI Q3; M7.1) — backup-CBF filter (numerical integration of the maneuver, RK45 rtol 1e-8, central differences) vs the proposed filter\n\n"
          f"{2 * args.n} X_RF states (set A, N = 4; {args.n} in the boundary layer, {args.n} with H0 in [0, 1] m), adversarial nominal, one filter evaluation each; "
          f"wall {time.time() - t0:.0f} s with {args.workers} workers.\n\n{table}\n")
    (out / "summary.md").write_text(md)
    save_json(out / "results.json", {"rows": rows, "n": len(rows), "params": p.to_dict()})
    print(md)


if __name__ == "__main__":
    main()
