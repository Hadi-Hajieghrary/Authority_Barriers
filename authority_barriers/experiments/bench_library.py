"""M1.7: runtime of the closed-form barrier (D + gradients) and of the filter build + solve."""
from __future__ import annotations

import math
import time

import numpy as np

from authority_barriers.theory import stopping as SD
from authority_barriers.theory.authority import BarrierData
from authority_barriers.theory.filters import ProposedFilter
from authority_barriers.theory.maneuver import plan
from authority_barriers.theory.params import Params, load_set
from authority_barriers.theory.sampling import sample_V, sample_XRF_layer


def with_N(p: Params, N: int) -> Params:
    """Set A with a different team size (per-cable constants unchanged, m_L scaled with N)."""
    return Params(**{**p.to_dict(), "name": f"A_N{N}", "N": N, "m_L": p.m_L * N / p.N,
                     "m": [p.m[0]] * N, "l": [p.l[0]] * N, "f_max": [p.f_max[0]] * N,
                     "T_bar": [p.T_bar[0]] * N, "rho": [p.rho[0]] * N, "rho_fb": [p.rho_fb[0]] * N})


def bench(n_eval: int = 1000, seed: int = 0):
    base = load_set("A")
    rng = np.random.default_rng(seed)
    print(f"{'N':>2s} {'D+grad median [ms]':>20s} {'D+grad p95 [ms]':>16s} {'filter median [ms]':>19s} {'filter p95 [ms]':>16s} {'breakpoints':>12s}")
    rows = []
    for N in range(2, 9):
        p = with_N(base, N)
        data = BarrierData.nominal(p)
        states = [(rng.uniform(-1, 4), *sample_V(rng, N, data.nu, data.zeta_bar)) for _ in range(n_eval)]
        t = np.zeros(n_eval); nb = np.zeros(n_eval)
        for k, (v, z, zd) in enumerate(states):
            t0 = time.perf_counter(); r = SD.evaluate(v, z, zd, data); t[k] = time.perf_counter() - t0
            nb[k] = r.n_breakpoints
        filt = ProposedFilter(p, data)
        sts = sample_XRF_layer(rng, p, min(200, n_eval), data)
        tf = np.zeros(len(sts))
        for k, st in enumerate(sts):
            u_nom = plan(st, p, data).u
            t0 = time.perf_counter(); res = filt.solve(st, u_nom); tf[k] = time.perf_counter() - t0
            assert res.feasible, res.status
        print(f"{N:2d} {1e3*np.median(t):20.3f} {1e3*np.percentile(t,95):16.3f} {1e3*np.median(tf):19.3f} {1e3*np.percentile(tf,95):16.3f} {np.max(nb):12.0f}")
        rows.append((N, np.median(t), np.percentile(t, 95), np.median(tf), np.percentile(tf, 95)))
    return rows


if __name__ == "__main__":
    import sys
    rows = bench(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
    d8 = [r for r in rows if r[0] == 8][0][1]
    f4 = [r for r in rows if r[0] == 4][0][3]
    ok = d8 <= 1e-3 and f4 <= 3e-3
    print(f"M1.7: D+grad N=8 median {1e3*d8:.3f} ms (target <= 1 ms); filter N=4 median {1e3*f4:.3f} ms (target <= 3 ms) -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
