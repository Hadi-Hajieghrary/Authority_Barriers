"""E0: reproduce the worked example of Sec. I / III (matrix row M0.1).

Two 1 kg quadrotors, 1 kg payload, 1 m cables, both cables 30 deg toward the wall, T_min = 1 N,
h0 = 1 m, v0 = 0.85 m/s, alpha1(h) = h, alpha2(psi) = 20 psi, omega_bar = 2 rad/s.
Expected: psi1 = 0.15 m/s, beta = -2.15 m/s^2, alpha_ex = -1 m/s^2 (SOCP), mu = 1.15 m/s^2,
t_psi = 0.15/0.85 = 0.1765 s (paper rounds to 0.18), tau = 0.5/2 = 0.25 s, and x0 in D.
"""
from __future__ import annotations

import math
import sys

import numpy as np

from authority_barriers.theory import hocbf as H
from authority_barriers.theory.params import check_assumptions, format_margins, paper_example
from authority_barriers.theory.state import State


def main() -> int:
    p = paper_example()
    a1, a2 = H.LinearClassK(1.0), H.LinearClassK(20.0)
    z = np.array([-math.sin(math.radians(30.0))] * 2)
    st = State.from_swing(p, h=1.0, v=0.85, z=z, w=[0.0, 0.0], zd=[0.0, 0.0], wd=[0.0, 0.0])
    h0, v0 = st.h(p), st.v(p)
    alpha_ex, res = H.alpha_exact(p, st)
    vals = {
        "psi1 [m/s]": H.psi1(h0, v0, a1),
        "beta [m/s^2]": H.beta(st, p, a1, a2),
        "alpha_ex [m/s^2]": alpha_ex,
        "mu [m/s^2]": alpha_ex - H.beta(st, p, a1, a2),
        "t_psi [s]": H.t_psi(h0, v0, a1),
        "tau [s]": H.tau(st, p),
    }
    expected = {"psi1 [m/s]": 0.15, "beta [m/s^2]": -2.15, "alpha_ex [m/s^2]": -1.0,
                "mu [m/s^2]": 1.15, "t_psi [s]": 0.15 / 0.85, "tau [s]": 0.25}
    ok = True
    print(f"{'quantity':18s} {'computed':>12s} {'expected':>12s} {'|delta|':>10s}  verdict")
    for k, e in expected.items():
        d = abs(vals[k] - e)
        flag = d <= 1e-6
        ok &= flag
        print(f"{k:18s} {vals[k]:12.6f} {e:12.6f} {d:10.2e}  {'PASS' if flag else 'FAIL'}")
    print(f"SOCP status: {res.get_solution_result()}, solver: {res.get_solver_id().name()}")
    inD, conds = H.in_D(st, p, a1, a2)
    ok &= inD
    print(f"x0 in D: {inD}  " + ", ".join(f"{k}: {v}" for k, v in conds.items()))
    print("\nAssumption checker on the example constants (for information only; Assumptions 7/13/14 are not claimed here):")
    print(format_margins(check_assumptions(p)))
    print(f"\nM0.1 {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
