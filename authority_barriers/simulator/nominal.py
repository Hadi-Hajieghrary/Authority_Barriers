"""Nominal (unfiltered) controllers, evaluated inside the 200 Hz filter tick as u_nom = f(t, state)."""
from __future__ import annotations

import numpy as np

from authority_barriers.theory.params import E3, Params
from authority_barriers.theory.state import State


def allocate_tensions(p: Params, st: State, a_d: np.ndarray, w_vertical: float = 3.0) -> np.ndarray:
    """Bounded least-squares tensions, sum_i T_i q_i ~ m_L a_d with T_i in [T_min, m_L a_max / N] (the D-4 cone is then
    satisfied for any directions), the vertical component weighted w_vertical times the horizontal ones so that the
    payload's weight is carried before horizontal acceleration is served (an unbounded solve clipped afterwards can
    drop the vertical force by a third, C-15)."""
    from scipy.optimize import lsq_linear
    W = np.array([1.0, 1.0, w_vertical])
    lo, hi = p.T_min, p.m_L * p.a_max / p.N
    r = lsq_linear(st.q.T * W[:, None], p.m_L * a_d * W, bounds=(lo, hi), method="bvls")
    return np.clip(r.x, lo, hi)


def thrust_from(p: Params, st: State, T: np.ndarray, u_perp: np.ndarray) -> np.ndarray:
    a = (T @ st.q) / p.m_L
    s = T + p.m_arr * (st.q @ a) - p.m_arr * p.l_arr * st.omega2()
    return s[:, None] * st.q + u_perp


class VelocityTransportNominal:
    """Payload velocity command with altitude hold: the transport controller of Sec. VI (B7 of the plan), a geometric
    controller for the taut-cable team. Each tick: (i) the desired payload specific force a_d = k_v (v_cmd - v_L) + g e3
    + altitude PD, with its horizontal part limited to the lean the operational set allows (|a_h| <= a_z tan(asin lean_limit))
    and its vertical part kept inside what the tension caps can carry; (ii) a formation reference around a_d's direction,
    the cables placed on an ellipse (radii `spread` along y and r, half the swing box by default; offsets sum to zero, so
    the vehicles surround the payload) unless a fixed `q_ref` is given; (iii) bounded least-squares tensions for m_L a_d on the current
    directions; (iv) perpendicular thrusts u_i^perp = m_i P_i a + m_i l_i (k_q (q_ref,i - q_i)_perp - k_swing qdot_i):
    the feed-forward eq. cable needs (Lemma 1(b): qddot_i = u_i^perp/(m_i l_i) - P_i a/l_i - |qdot_i|^2 q_i) plus a PD on
    the cable direction. Without the feed-forward a tilted cable accelerates away from vertical and the team collapses
    (C-15)."""

    def __init__(self, p: Params, v_cmd: np.ndarray, z_ref: float = 15.0, k_v: float = 1.0, k_z: float = 1.0,
                 k_dz: float = 1.5, k_swing: float = 2.0, q_ref: np.ndarray | None = None, k_q: float = 2.0,
                 lean_limit: float | None = None, spread: tuple | None = None):
        """lean_limit: largest lean (sine of the cable angle) of the formation's centre along the commanded velocity
        (default 0.5 z_bar: half the box toward the wall); spread: (radius along y, radius along r) of the ellipse on
        which the N cables are placed around the centre (default (0.5 z_bar, 0.5 w_bar)); the vehicles then stay inside
        the swing box with a margin (0.5 + 0.5 cos 45 deg = 0.85 of the box along the velocity)."""
        self.p = p
        self.v_cmd = np.asarray(v_cmd, float)
        self.z_ref, self.k_v, self.k_z, self.k_dz, self.k_swing = z_ref, k_v, k_z, k_dz, k_swing
        self.q_ref, self.k_q = q_ref, k_q
        self.lean_limit = 0.5 * p.z_bar if lean_limit is None else float(lean_limit)
        ry, rr = (0.5 * p.z_bar, 0.5 * p.w_bar) if spread is None else spread
        if p.N > 1:                                                                # the vehicles around the payload, no one on the velocity axis
            phi = 2.0 * np.pi * np.arange(p.N) / p.N + np.pi / 4.0
            self.offsets = np.cos(phi)[:, None] * ry * p.y_vec[None, :] + np.sin(phi)[:, None] * rr * p.r_vec[None, :]
        else:
            self.offsets = np.zeros((1, 3))

    def desired_force(self, st: State) -> np.ndarray:
        p = self.p
        a_d = self.k_v * (self.v_cmd - st.v_L) + p.g * E3
        a_d = a_d + (self.k_z * (self.z_ref - st.x_L[2]) - self.k_dz * st.v_L[2]) * E3
        a_z = float(np.clip(a_d[2], 0.5 * p.g, 0.9 * p.a_max))
        a_h = a_d - a_d[2] * E3
        h_max = self.lean_limit / np.sqrt(1.0 - self.lean_limit ** 2) * a_z
        nh = float(np.linalg.norm(a_h))
        if nh > h_max:
            a_h = a_h * (h_max / nh)
        return a_h + a_z * E3

    def formation(self, a_d: np.ndarray) -> np.ndarray:
        if self.q_ref is not None:
            return self.q_ref
        d = a_d / np.linalg.norm(a_d)
        q_ref = d[None, :] + self.offsets
        return q_ref / np.linalg.norm(q_ref, axis=1)[:, None]

    def __call__(self, t: float, st: State) -> np.ndarray:
        p = self.p
        a_d = self.desired_force(st)
        q_ref = self.formation(a_d)
        T = allocate_tensions(p, st, a_d)
        a = (T @ st.q) / p.m_L                                                   # specific force the allocated tensions produce
        u_perp = p.m_arr[:, None] * (a[None, :] - (st.q @ a)[:, None] * st.q)    # m_i P_i a (eq. cable feed-forward)
        u_perp += -self.k_swing * (p.m_arr * p.l_arr)[:, None] * st.qd           # damp the swing
        if self.k_q > 0:
            err = q_ref - st.q
            err -= np.einsum("ij,ij->i", err, st.q)[:, None] * st.q
            u_perp += self.k_q * (p.m_arr * p.l_arr)[:, None] * err              # formation hold
        u = thrust_from(p, st, T, u_perp)
        n = np.linalg.norm(u, axis=1)
        over = n > p.f_max_arr
        u[over] *= (p.f_max_arr[over] / n[over])[:, None]
        return u


class AdversarialNominal:
    """Full thrust toward the wall, tilted up so the team also pulls the payload along (E2)."""

    def __init__(self, p: Params, tilt: float = np.radians(40.0)):
        self.p = p
        d = np.cos(tilt) * p.n_vec + np.sin(tilt) * E3
        self.u = np.outer(p.f_max_arr, d / np.linalg.norm(d))

    def __call__(self, t: float, st: State) -> np.ndarray:
        return self.u.copy()


class HoverNominal:
    """Hold the payload where it is (tests)."""

    def __init__(self, p: Params):
        self.inner = VelocityTransportNominal(p, np.zeros(3))

    def __call__(self, t: float, st: State) -> np.ndarray:
        self.inner.z_ref = st.x_L[2] if t == 0.0 else self.inner.z_ref
        return self.inner(t, st)
