"""E4 (Table I row 4, Prop. 17): runtime of D and of the filter vs N (M5.1) and the
distributed-vs-central barrier derivative (M5.2), algebraically at random states and along the
E2 closed-loop logs with the logged commands."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from authority_barriers.theory import stopping as SD
from authority_barriers.theory.authority import BarrierData
from authority_barriers.theory.filters import ProposedFilter, decomposition_terms
from authority_barriers.theory.params import load_set
from authority_barriers.theory.sampling import sample_XRF_layer
from authority_barriers.theory.socp import tangent_basis
from authority_barriers.theory.taut_model import rhs_from_thrust, unpack
from authority_barriers.simulator.recorder import TrialLog
from authority_barriers.experiments.bench_library import bench
from authority_barriers.experiments.common import out_dir, save_json, summary_table


def decomposition_along_logs(p, data, logs, max_ticks=400):
    """At every tick: coupled Hdot_j(x, u) vs const + sum_i local_i with a_hat = a(T) (exact identity)
    and with a_hat = a_meas from the cable forces (the broadcast quantity of Prop. 17(b))."""
    filt = ProposedFilter(p, data, swing_mode="none")
    err_exact, err_meas = [], []
    for L in logs:
        idx = np.linspace(0, L.t_diag.size - 1, min(max_ticks, L.t_diag.size)).astype(int)
        for k in idx:
            st = unpack(L.taut[:, k], p.N)
            u = L.cmd[:, k].reshape(p.N, 3)
            out = rhs_from_thrust(st, u, p)
            T = out["T"]
            c = np.concatenate([tangent_basis(st.q[i]).T @ out["u_perp"][i] for i in range(p.N)])
            x = np.concatenate([T, c])
            res = SD.evaluate(st.v(p), st.z(p), st.zd(p), data)
            if not np.isfinite(res.D):
                continue
            cp, sw, _, H, rows, _, _ = filt.build(st, u, res=res)
            coupled = rows[0][0] @ x + rows[0][1]
            for a_hat, store in ((out["a"], err_exact), (L.cable[2 * p.N:2 * p.N + 3, k], err_meas)):
                const, locals_, _ = decomposition_terms(p, st, res, a_hat, j=0)
                total = const + sum(A @ np.concatenate([[T[i]], c[2 * i:2 * i + 2]]) + b for i, (A, b) in enumerate(locals_))
                store.append(abs(total - coupled))
    return np.array(err_exact), np.array(err_meas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=1000)
    ap.add_argument("--logs", default="results/core/e2_perfect_lmax_off0/*.npz")
    ap.add_argument("--max-logs", type=int, default=20)
    args = ap.parse_args()
    p = load_set("A")
    data = BarrierData.nominal(p)
    out = out_dir("core", "e4")
    # --- M5.1 runtime vs N
    rows = bench(args.n_eval)
    Ns = np.array([r[0] for r in rows]); tD = np.array([r[1] for r in rows]); tF = np.array([r[3] for r in rows])
    slope_D = np.polyfit(np.log(Ns), np.log(tD), 1)[0]
    slope_F = np.polyfit(np.log(Ns), np.log(tF), 1)[0]
    slope_NlogN = np.polyfit(np.log(Ns), np.log(Ns * np.log(Ns)), 1)[0]
    # --- M5.2 algebraic identity at random states
    rng = np.random.default_rng(0)
    filt = ProposedFilter(p, data, swing_mode="none")
    alg = []
    for st in sample_XRF_layer(rng, p, 300, data):
        res = SD.evaluate(st.v(p), st.z(p), st.zd(p), data)
        x = rng.normal(size=3 * p.N); x[:p.N] = np.abs(x[:p.N]) + p.T_min
        cp, sw, _, H, rws, _, _ = filt.build(st, np.zeros((p.N, 3)), res=res)
        const, locals_, _ = decomposition_terms(p, st, res, (x[:p.N] @ st.q) / p.m_L, j=0)
        total = const + sum(A @ np.concatenate([[x[i]], x[p.N + 2 * i:p.N + 2 * i + 2]]) + b for i, (A, b) in enumerate(locals_))
        alg.append(abs(total - (rws[0][0] @ x + rws[0][1])))
    alg = np.array(alg)
    # --- M5.2 along E2 logs
    files = sorted(glob.glob(args.logs))[: args.max_logs]
    logs = [TrialLog.load(Path(f).with_suffix("")) for f in files]
    e_exact, e_meas = decomposition_along_logs(p, data, logs) if logs else (np.zeros(0), np.zeros(0))
    table = summary_table([
        {"metric": "D + gradients runtime median [ms] N=2..8", "value": ", ".join(f"{1e3*t:.2f}" for t in tD), "target": "N=8 <= 1 ms", "verdict": "PASS" if tD[-1] <= 1e-3 else "FAIL"},
        {"metric": "filter build+solve median [ms] N=2..8", "value": ", ".join(f"{1e3*t:.2f}" for t in tF), "target": "N=4 <= 3 ms", "verdict": "PASS" if tF[Ns == 4][0] <= 3e-3 else "FAIL"},
        {"metric": "log-log slope of D runtime vs N (N log N reference)", "value": f"{slope_D:.2f} ({slope_NlogN:.2f})", "target": "<= 1.3", "verdict": "PASS" if slope_D <= 1.3 else "FAIL"},
        {"metric": "log-log slope of filter runtime vs N", "value": f"{slope_F:.2f}", "target": "report", "verdict": "-"},
        {"metric": "|sum local - coupled| algebraic, 300 states", "value": f"max {alg.max():.1e}", "target": "<= 1e-10", "verdict": "PASS" if alg.max() <= 1e-10 else "FAIL"},
        {"metric": f"|sum local - coupled| along {len(logs)} E2 logs, a_hat = a(T)", "value": f"max {e_exact.max():.1e}" if e_exact.size else "no logs", "target": "<= 1e-8", "verdict": ("PASS" if e_exact.max() <= 1e-8 else "FAIL") if e_exact.size else "-"},
        {"metric": "mismatch with a_hat = a_meas (broadcast from cable forces) [m/s^2]", "value": f"max {e_meas.max():.2e}" if e_meas.size else "no logs", "target": "report (charged to d_bar)", "verdict": "-"},
    ], ["metric", "value", "target", "verdict"])
    (out / "summary.md").write_text(f"# E4 (Prop. 17)\n\n{table}\n")
    save_json(out / "results.json", {"N": Ns, "t_D": tD, "t_filter": tF, "slope_D": slope_D, "slope_filter": slope_F,
                                     "alg_max": alg.max(), "logs_exact_max": e_exact.max() if e_exact.size else None,
                                     "logs_meas_max": e_meas.max() if e_meas.size else None, "n_logs": len(logs)})
    print(table)


if __name__ == "__main__":
    main()
