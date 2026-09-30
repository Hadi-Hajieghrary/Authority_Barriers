"""Altitude barriers of the braking maneuver (sets with a hover floor).

With tensions in [T_h, T_bar] and cables in the cone, the vertical acceleration of the payload during the braking
phase lies in [c_low, c_high] with c_low >= 0 (params.Params). The braking phase ends at the stopping time

    t_s(x) = max{ t_a + v~(t_a) / alpha_sat,  t_a - hold_lead,  0 },      t_a = max_i t2_i,

when every swing is within hold_lead seconds of its arrival and the payload recedes from the wall; t_s decreases at
unit rate along the maneuver. Afterwards the altitude hold decelerates a climb at c_dn and a descent at c_up. Hence, with chi the
altitude and U = chidot + c_high t_s, L = chidot + c_low t_s,

    climb <= Dz_up   = ( chidot t_s + c_high t_s^2 / 2 + U^2 / (2 c_dn) )_+          if U > 0, else 0,
    drop  <= Dz_down = chidot^2 / (2 c_low)                                           if chidot < 0 <= L,
                       -chidot t_s - c_low t_s^2 / 2 + L^2 / (2 c_up)                 if L < 0,     else 0,

and H_up = (alt_max - chi) - Dz_up, H_down = (chi - alt_min) - Dz_down do not decrease along the maneuver.
t_s is the largest of N + 1 smooth candidate times (stop_time); `evaluate` returns the barriers of every candidate and
`rows` the affine rows Hdot >= -kappa_alt H of the filter, one per candidate and one-sided gradient, so that the
rows remain valid where the largest candidate changes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import stopping as SD
from .params import Params
from .state import State

TIE = 1e-9


@dataclass
class StopTime:
    t_s: float
    cand: list           # [(tau, d/dv, d/dzeta (N,), d/domega (N,))]: t_s is the largest tau


def stop_time(mn: SD.Maneuver, lead: float = 0.1, tol: float = 0.0) -> StopTime:
    """End t_s of the braking phase as the largest of the candidate times

        tau_v = t_a + v~(t_a) / alpha_sat   (t_a = max_i t2_i),      tau_i = t2_i - lead,

    cut at zero: the altitude hold begins `lead` seconds before the last swing arrives, once the payload recedes.
    Every candidate is smooth in (v, zeta, omega) where it is positive and decreases at unit rate along the maneuver,
    so one barrier row per candidate handles the points where the largest one changes; the lead bounds the
    gradient of t2_i, which grows without bound at the arrival. A candidate within `tol` below zero is returned
    with the value zero and its gradient, and the candidate of the hold (zero gradient) is added when no candidate
    exceeds tol: the rows of a sampled filter then cover both sides of the cut at zero."""
    N, data = mn.N, mn.data
    t_a = float(np.max(mn.t2)) if N else 0.0
    v_a = float(mn.v_tilde(t_a))
    cand = []
    tau_v = t_a + v_a / data.alpha_sat
    if tau_v > -tol:                                          # alpha~(t_a) = alpha_sat removes the derivative of t_a
        k = int(mn.interval_of(t_a)); s = t_a - mn.tau[k]
        G1z = mn.G1z[k] + mn.gz[k, :, 0] * s + mn.gz[k, :, 1] * s ** 2 / 2.0 + mn.gz[k, :, 2] * s ** 3 / 3.0
        G1w = mn.G1w[k] + mn.gw[k, :, 0] * s + mn.gw[k, :, 1] * s ** 2 / 2.0 + mn.gw[k, :, 2] * s ** 3 / 3.0
        cand.append((max(tau_v, 0.0), 1.0 / data.alpha_sat, -G1z / data.alpha_sat, -G1w / data.alpha_sat))
    for i in np.where(mn.t2 > lead - tol)[0]:
        gz, gw = np.zeros(N), np.zeros(N)
        if mn.t1[i] == mn.t2[i]:                              # on or above the deceleration curve: t2 = 2 (zeta_bar - zeta) / omega
            gz[i], gw[i] = -2.0 / mn.omega[i], -mn.t2[i] / mn.omega[i]
        else:
            gz[i], gw[i] = -1.0 / mn.r[i], -(1.0 - mn.omega[i] / mn.r[i]) / data.nu
        cand.append((max(float(mn.t2[i]) - lead, 0.0), 0.0, gz, gw))
    if not cand or max(c[0] for c in cand) <= tol:            # the altitude hold
        cand.append((0.0, 0.0, np.zeros(N), np.zeros(N)))
    return StopTime(max(c[0] for c in cand), cand)


def climb(chid: float, tau: float, p: Params):
    """Dz_up and its one-sided partial derivatives [(d/dchidot, d/dtau)]."""
    U = chid + p.c_high * tau
    val = chid * tau + 0.5 * p.c_high * tau ** 2 + U ** 2 / (2.0 * p.c_dn)
    zero, grow = (0.0, 0.0), (tau + U / p.c_dn, U * (1.0 + p.c_high / p.c_dn))
    if U <= 0.0 or val < -TIE:
        return 0.0, [zero]
    return max(val, 0.0), ([grow] if val > TIE else [zero, grow])


def drop(chid: float, tau: float, p: Params):
    """Dz_down and its one-sided partial derivatives [(d/dchidot, d/dtau)]."""
    if chid >= 0.0:
        return 0.0, ([(0.0, 0.0)] if chid > TIE or p.c_low > 0 else [(0.0, 0.0), (-tau, 0.0)])
    L = chid + p.c_low * tau
    early = (chid / p.c_low, 0.0) if p.c_low > 0 else None
    late = (-tau + L / p.c_up, -L * (1.0 - p.c_low / p.c_up))
    if L >= 0.0:
        return chid ** 2 / (2.0 * p.c_low), ([early] if L > TIE else [early, late])
    return -chid * tau - 0.5 * p.c_low * tau ** 2 + L ** 2 / (2.0 * p.c_up), [late]


@dataclass
class AltitudeResult:
    H_up: float          # smallest over the candidate times, which is the value at t_s
    H_down: float
    Dz_up: float
    Dz_down: float
    t_s: float
    up: list            # [(H, d/dchidot, d/dv, d/dzeta, d/domega)]: barrier and gradient of its bound, per candidate time and one-sided gradient
    down: list


def evaluate(st: State, p: Params, mn: SD.Maneuver, tol: float = 0.0) -> AltitudeResult:
    chi, chid = float(st.x_L[2]), float(st.v_L[2])
    ts = stop_time(mn, p.hold_lead, tol)
    up, down = [], []
    for tau, gv, gz, gw in ts.cand:
        Du, gu = climb(chid, tau, p)
        Dd, gd = drop(chid, tau, p)
        up += [((p.alt_max - chi) - Du, a, b * gv, b * gz, b * gw) for a, b in gu]
        down += [((chi - p.alt_min) - Dd, a, b * gv, b * gz, b * gw) for a, b in gd]
    Du, Dd = climb(chid, ts.t_s, p)[0], drop(chid, ts.t_s, p)[0]
    return AltitudeResult((p.alt_max - chi) - Du, (chi - p.alt_min) - Dd, Du, Dd, ts.t_s, up, down)


def rows(st: State, p: Params, res: AltitudeResult, Az: np.ndarray, bz: np.ndarray):
    """Affine rows (A, b, H) with Hdot = A x + b in the variables x = (T, c) of the cone program, for the barriers of
    every candidate time and each of their one-sided gradients. Az, bz: zddot = Az x + bz (filters.swing_affine)."""
    N = p.N
    z, zd, chid = st.z(p), st.zd(p), float(st.v_L[2])
    cz = np.zeros(3 * N); cz[:N] = st.q[:, 2] / p.m_L                     # chiddot = cz x - g
    bT = np.zeros(3 * N); bT[:N] = z / p.m_L                              # vdot = -(bT x + offset)
    offset = -p.g * float(np.array([0.0, 0.0, 1.0]) @ p.y_vec)
    out = []
    for sign, grads in ((-1.0, res.up), (+1.0, res.down)):
        for H, g_chid, g_v, g_z, g_w in grads:
            A = -g_chid * cz + g_v * bT - g_w @ Az
            b = sign * chid + g_chid * p.g + g_v * offset - float(g_z @ zd) - float(g_w @ bz)
            out.append((A, b, H))
    return out
