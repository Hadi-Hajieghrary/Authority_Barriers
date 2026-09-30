"""The braking maneuver pi^sharp(x) of Sec. IV-C, i.e. the map of (C3) instantiated for the
cable system in Prop. 15: tensions T_i = T_bar_i if z_i > 0 else the floor of the barrier data
(T_min, or the hover floor T_h), the initial acceleration of the swing for every z_i, a transverse
law that keeps (w_i, wdot_i) in V(nu_w, w_bar), and the tangent-plane thrust that produces both
through the Gram matrix (eq. gram). With the certified barrier data the maneuver has a second phase:
once every swing is about to arrive and the payload recedes from the wall (altitude.stop_time), one uniform
tension holds the altitude (vertical acceleration -c_dn while the payload climbs, +c_up while it descends)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import profile as PF
from .authority import BarrierData
from .params import Params
from .state import State


def w_law(w, wd, nu_w: float, w_bar: float):
    """Saturated PD toward w = 0 with |wddot| <= nu_w: k_p = nu_w / w_bar (saturates at |w| = w_bar
    at rest), k_d = 2 sqrt(k_p). On the boundary of V(nu_w, w_bar) it decelerates at least at
    the rate the tangency condition (17d) requires (checked numerically in the tests)."""
    kp = nu_w / w_bar
    kd = 2.0 * np.sqrt(kp)
    return np.clip(-kp * np.asarray(w, float) - kd * np.asarray(wd, float), -nu_w, nu_w)


def swing_thrust(st: State, p: Params, a: np.ndarray, zdd_d, wdd_d) -> np.ndarray:
    """Perpendicular thrust u_i_perp that produces the prescribed zddot_i and wddot_i given the
    payload specific force a (eq. cable projected on y and r, inverted through eq. gram)."""
    y, r = p.y_vec, p.r_vec
    z, w = st.z(p), st.w(p)
    om2 = st.omega2()
    q = st.q
    Pa = a[None, :] - (q @ a)[:, None] * q                     # P_i a
    eta_z = p.m_arr * p.l_arr * (zdd_d + om2 * z) + p.m_arr * (Pa @ y)
    eta_w = p.m_arr * p.l_arr * (wdd_d + om2 * w) + p.m_arr * (Pa @ r)
    det = (1.0 - z ** 2) * (1.0 - w ** 2) - (z * w) ** 2       # = 1 - z^2 - w^2 = varrho^2
    c1 = ((1.0 - w ** 2) * eta_z + z * w * eta_w) / det
    c2 = (z * w * eta_z + (1.0 - z ** 2) * eta_w) / det
    Py = y[None, :] - z[:, None] * q
    Pr = r[None, :] - w[:, None] * q
    return c1[:, None] * Py + c2[:, None] * Pr


@dataclass
class Plan:
    T: np.ndarray        # tensions (N,)
    u_perp: np.ndarray   # (N, 3)
    u: np.ndarray        # thrust vectors (N, 3)
    a: np.ndarray        # payload specific force (3,)
    b: float             # braking deceleration y^T xddot_L
    zdd: np.ndarray      # prescribed zddot_i
    wdd: np.ndarray      # prescribed wddot_i
    s: np.ndarray        # parallel thrust components


def swing_accel(z, zd, nu: float, zbar: float, band: float = 2e-3, wn: float = 60.0, nu_dec: float | None = None):
    """Initial acceleration of the swing (profile.accel0) with a boundary
    layer: within `band` seconds of arrival at (zbar, 0) a saturated PD (natural frequency wn)
    holds the corner instead of switching sign, which removes chatter in any feedback
    implementation; the output stays in [-nu, nu], so it is an admissible prescription of (C3)."""
    z = np.asarray(z, float); zd = np.asarray(zd, float)
    _, t1, t2, _, _, _ = PF.pieces(z, zd, nu, zbar, nu_dec)
    bang = PF.accel0(z, zd, nu, zbar, nu_dec=nu_dec)
    pd = np.clip(-wn ** 2 * (z - zbar) - 2.0 * wn * zd, -nu, nu)
    return np.where(t2 <= band, pd, bang)


def holding(st: State, p: Params, data: BarrierData) -> bool:
    """Second phase of the maneuver with altitude barriers: the stopping time is zero, that is, every swing is within
    hold_lead seconds of its arrival and the payload recedes from the wall (altitude.stop_time)."""
    if data.kind != "certified":
        return False
    from . import altitude as ALT
    from . import stopping as SD
    return ALT.stop_time(SD.build(st.v(p), st.z(p), st.zd(p), data), p.hold_lead).t_s <= 0.0


def hold_tension(st: State, p: Params, k: float = 50.0) -> np.ndarray:
    """Uniform tension of the altitude hold: vertical acceleration -c_dn while the payload climbs and +c_up
    while it descends, with a linear layer of width c/k around zero vertical speed."""
    c = float(np.clip(-k * st.v_L[2], -p.c_dn, p.c_up))
    return np.full(p.N, p.m_L * (p.g + c) / float(np.sum(st.q[:, 2])))


def plan(st: State, p: Params, data: BarrierData | None = None) -> Plan:
    data = data or BarrierData.nominal(p)
    z, w, zd, wd = st.z(p), st.w(p), st.zd(p), st.wd(p)
    T = hold_tension(st, p) if holding(st, p, data) else np.where(z > 0, np.asarray(data.T_bar, float), data.T_min)
    a = (T @ st.q) / p.m_L                                       # no disturbance in the plan
    zdd = swing_accel(z, zd, data.nu, data.zeta_bar, nu_dec=data.nu_dec)
    wdd = w_law(w, wd, p.nu_w, p.w_bar)
    u_perp = swing_thrust(st, p, a, zdd, wdd)
    s = T + p.m_arr * (st.q @ a) - p.m_arr * p.l_arr * st.omega2()   # eq. tension inverted
    u = s[:, None] * st.q + u_perp
    b = float(p.y_vec @ (a - p.g * np.array([0.0, 0.0, 1.0])))
    return Plan(T, u_perp, u, a, b, zdd, wdd, s)


def admissible(pl: Plan, p: Params, tol: float = 1e-9) -> dict:
    """Checks of U(x) for a plan: thrust norms, tension floor, D-4 cone."""
    norms = np.linalg.norm(pl.u, axis=1)
    return {
        "thrust": bool(np.all(norms <= p.f_max_arr + tol)),
        "tension": bool(np.all(pl.T >= p.T_min - tol)),
        "a_max": bool(np.linalg.norm(pl.T @ np.zeros((p.N, 3)) + pl.a * p.m_L) <= p.m_L * p.a_max + tol),
        "max_norm_ratio": float(np.max(norms / p.f_max_arr)),
    }
