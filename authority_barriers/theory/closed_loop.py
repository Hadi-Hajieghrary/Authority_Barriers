"""Sampled (zero-order-hold) closed loop of a filter on the exact taut-cable model: the theory's own
model with the paper's sampled filter, used to attribute full-order violations (M3.4) to the inner
loop / cable compliance versus the sampled-data effect itself."""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .params import Params
from .state import State
from .taut_model import f_ambient, pack, unpack, rhs_from_thrust


def run_sampled(x0: State, filt, nominal, p: Params, t_f: float, dt: float | None = None, rtol=1e-9, atol=1e-11,
                stop_on_wall: bool = True):
    """Returns dict of arrays per tick: t, h, v, H, D, feasible, u (N,3), T (N,), state (packed)."""
    dt = p.dt_filter if dt is None else dt
    x = pack(x0).copy()
    n = int(round(t_f / dt))
    rec = {k: [] for k in ("t", "h", "v", "H", "D", "feasible", "u", "T", "x", "b", "hdot0")}
    termination = "horizon"
    for k in range(n + 1):
        t = k * dt
        st = unpack(x, p.N)
        res = filt.solve(st, nominal(t, st))
        out = rhs_from_thrust(st, res.u, p)
        rec["t"].append(t); rec["h"].append(st.h(p)); rec["v"].append(st.v(p))
        rec["H"].append(getattr(res, "H", np.nan)); rec["D"].append(getattr(res, "D", np.nan))
        rec["feasible"].append(res.feasible); rec["u"].append(res.u.copy()); rec["T"].append(out["T"]); rec["x"].append(x.copy())
        rec["b"].append(res.b)
        rec["hdot0"].append(float(res.hdot[0]) if getattr(res, "hdot", np.zeros(0)).size else np.nan)
        if stop_on_wall and st.h(p) < 0:
            termination = "wall_contact"; break
        if np.any(out["T"] < 0):
            termination = "slack"; break
        if k == n:
            break
        u_flat = res.u.reshape(-1)
        sol = solve_ivp(lambda tt, xx: f_ambient(xx, u_flat, p), (0.0, dt), x, method="DOP853", rtol=rtol, atol=atol)
        x = sol.y[:, -1]
        st_n = unpack(x, p.N)
        q = st_n.q / np.linalg.norm(st_n.q, axis=1)[:, None]                      # numerical hygiene
        qd = st_n.qd - np.einsum("ij,ij->i", st_n.qd, q)[:, None] * q
        x = pack(State(st_n.x_L, st_n.v_L, q, qd))
    return {k: np.array(v) for k, v in rec.items()} | {"termination": termination}
