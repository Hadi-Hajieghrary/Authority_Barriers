"""Sampled (zero-order-hold) closed loop of a filter on the exact taut-cable model: the theory's own
model with the paper's sampled filter, used to attribute full-order violations (M3.4) to the inner
loop / cable compliance versus the sampled-data effect itself."""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from . import altitude as ALT
from . import profile as PF
from . import stopping as SD
from .params import Params
from .state import State
from .taut_model import f_ambient, pack, unpack, rhs_from_thrust

BETWEEN = ("h_between", "H_between", "margin_z_between", "margin_w_between", "H_up_between", "H_down_between")


def between_samples(xs: np.ndarray, p: Params, data, altitude: bool) -> dict:
    """Minima over the states xs (rows, packed) of h, H, the swing margins min(h+, h-) of z and w, and the altitude
    barriers (NaN without them): the quantities of the certificate between two updates of the filter."""
    vals = {k: np.inf for k in BETWEEN}
    for x in xs:
        st = unpack(x, p.N)
        z, zd, w, wd = st.z(p), st.zd(p), st.w(p), st.wd(p)
        mn = SD.build(st.v(p), z, zd, data)
        vals["h_between"] = min(vals["h_between"], st.h(p))
        vals["H_between"] = min(vals["H_between"], st.h(p) - SD.D_of(st.v(p), z, zd, data))
        lo, hi = PF.bounds_V(zd, data.nu, data.zeta_bar); vals["margin_z_between"] = min(vals["margin_z_between"], float(np.min(np.minimum(z - lo, hi - z))))
        lo, hi = PF.bounds_V(wd, p.nu_w, p.w_bar); vals["margin_w_between"] = min(vals["margin_w_between"], float(np.min(np.minimum(w - lo, hi - w))))
        if altitude:
            a = ALT.evaluate(st, p, mn)
            vals["H_up_between"] = min(vals["H_up_between"], a.H_up); vals["H_down_between"] = min(vals["H_down_between"], a.H_down)
    if not altitude:
        vals["H_up_between"] = vals["H_down_between"] = np.nan
    return vals


def run_sampled(x0: State, filt, nominal, p: Params, t_f: float, dt: float | None = None, rtol=1e-9, atol=1e-11,
                stop_on_wall: bool = True, substeps: int = 0):
    """Returns dict of arrays per tick: t, h, v, H, D, feasible, u (N,3), T (N,), state (packed), b, hdot0, the
    altitude barriers H_up, H_down (NaN for a filter without them), the solve time, the largest violations of the
    tension floor and of the thrust limit by the applied command (T_viol = max_i (T_min - T_i), f_viol =
    max_i (||u_i|| - f_max,i)), and, with substeps > 0, the minima of h, H, the swing margins and the altitude
    barriers over `substeps` states inside each hold (keys *_between; NaN without substeps)."""
    dt = p.dt_filter if dt is None else dt
    x = pack(x0).copy()
    n = int(round(t_f / dt))
    rec = {k: [] for k in ("t", "h", "v", "H", "D", "feasible", "u", "T", "x", "b", "hdot0", "H_up", "H_down", "solve_time", "T_viol", "f_viol") + BETWEEN}
    data, altitude = getattr(filt, "data", None), bool(getattr(filt, "altitude", False))
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
        rec["H_up"].append(getattr(res, "H_up", np.nan)); rec["H_down"].append(getattr(res, "H_down", np.nan))
        rec["solve_time"].append(getattr(res, "solve_time", np.nan))
        rec["hdot0"].append(float(res.hdot[0]) if getattr(res, "hdot", np.zeros(0)).size else np.nan)
        rec["T_viol"].append(float(np.max(p.T_min - out["T"]))); rec["f_viol"].append(float(np.max(np.linalg.norm(res.u, axis=1) - p.f_max_arr)))
        if stop_on_wall and st.h(p) < 0:
            termination = "wall_contact"; break
        if np.any(out["T"] < 0):
            termination = "slack"; break
        if k == n:
            break
        u_flat = res.u.reshape(-1)
        sol = solve_ivp(lambda tt, xx: f_ambient(xx, u_flat, p), (0.0, dt), x, method="DOP853", rtol=rtol, atol=atol,
                        dense_output=substeps > 0)
        x = sol.y[:, -1]
        if substeps > 0 and data is not None:
            xs = sol.sol(np.linspace(0.0, dt, substeps + 2)[1:-1]).T
            for key, val in between_samples(xs, p, data, altitude).items():
                rec[key].append(val)
        else:
            for key in BETWEEN:
                rec[key].append(np.nan)
        st_n = unpack(x, p.N)
        q = st_n.q / np.linalg.norm(st_n.q, axis=1)[:, None]                      # numerical hygiene
        qd = st_n.qd - np.einsum("ij,ij->i", st_n.qd, q)[:, None] * q
        x = pack(State(st_n.x_L, st_n.v_L, q, qd))
    return {k: np.array(v) for k, v in rec.items()} | {"termination": termination}
