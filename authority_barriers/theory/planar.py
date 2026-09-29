"""Planar reductions of the taut-cable model (Lemma 1) used by the kernel computations (plan K1, K3).

All cables move in the vertical plane spanned by the braking direction y and e3 (w_i = 0,
wdot_i = 0, no disturbance). Cable i has q_i = z_i y + varrho_i e3, varrho_i = sqrt(1 - z_i^2),
in-plane tangent tvec_i = varrho_i y - z_i e3, and ||qdot_i||^2 = zdot_i^2 / (1 - z_i^2).
Inputs are the tension T_i and eta_i := y . u_i^perp, so that u_i^perp = (eta_i / varrho_i) tvec_i
and ||u_i^perp|| = |eta_i| / varrho_i. (Plan K1 writes the same set with the tangential
magnitude eta_K1 = eta / varrho; the two conventions differ by this rescaling only.)

One cable (set A1), state (h, v, z, zd):  a = T q / m_L so P_1 a = 0 and
    hdot = -v,  vdot = -T z / m_L,  zddot = eta / (m l) - zd^2 z / (1 - z^2),
    s = T (1 + m/m_L) - m l zd^2 / (1 - z^2)                       (eq. tension),
    U(z, zd) = { T_min <= T <= m_L a_max,  s^2 + eta^2/(1 - z^2) <= f_max^2 }   (convex).
Two cables (set A2), state (h, v, z1, zd1, z2, zd2), inputs (T1, T2, eta1, eta2):
    a = (T1 q1 + T2 q2) / m_L,  zddot_i = eta_i/(m_i l_i) - (y . P_i a)/l_i - ||qdot_i||^2 z_i
(eq. cable projected on y), y . P_i a = y . a - z_i (q_i . a).
The thrust set is the ball (theta_max = pi, D-5); |z| <= z_bar < 1 keeps varrho away from 0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .params import E3, Params


@dataclass(frozen=True, eq=False)
class PlanarParams:
    """Constants of the planar reductions. Per-cable arrays keep the order of the Params set;
    the one-cable functions use cable 0, the two-cable functions cables 0 and 1."""
    N: int
    m_L: float
    m: np.ndarray
    l: np.ndarray
    f_max: np.ndarray
    T_min: float
    a_max: float
    z_bar: float
    omega_bar: float
    g: float

    @staticmethod
    def from_params(p: Params) -> "PlanarParams":
        if p.theta_max < math.pi - 1e-9:
            raise ValueError("planar input sets assume the ball thrust set (theta_max = pi, D-5)")
        return PlanarParams(N=p.N, m_L=float(p.m_L), m=p.m_arr.copy(), l=p.l_arr.copy(),
                            f_max=p.f_max_arr.copy(), T_min=float(p.T_min), a_max=float(p.a_max),
                            z_bar=float(p.z_bar), omega_bar=float(p.omega_bar), g=float(p.g))

    @property
    def T_max(self) -> float:
        """D-4 cap on one cable's tension: ||T q|| <= m_L a_max."""
        return self.m_L * self.a_max

    @property
    def zd_max(self) -> float:
        """Largest |zd| in the operating set: zd^2 <= omega_bar^2 (1 - z^2) at z = 0."""
        return self.omega_bar


def _stack(parts) -> np.ndarray:
    return np.stack(np.broadcast_arrays(*parts), axis=-1)


def lift_u_perp(z, eta, y: np.ndarray) -> np.ndarray:
    """Ambient perpendicular thrust of the planar input: u^perp = (eta / varrho) tvec = eta (y - (z/varrho) e3),
    shape (..., 3); y is the braking direction of the Params set."""
    z = np.asarray(z); eta = np.asarray(eta)
    varrho = np.sqrt(1.0 - z * z)
    return eta[..., None] * (y - (z / varrho)[..., None] * E3)


# ---------------------------------------------------------------- one cable (K1)
def planar1_parallel_thrust(z, zd, T, pp: PlanarParams):
    """s = T (1 + m/m_L) - m l zd^2 / (1 - z^2)  (eq. tension with a = T q/m_L)."""
    m, l = pp.m[0], pp.l[0]
    return T * (1.0 + m / pp.m_L) - m * l * zd * zd / (1.0 - z * z)


def planar1_rhs(x4, u2, pp: PlanarParams) -> np.ndarray:
    """(h, v, z, zd), (T, eta) -> (hdot, vdot, zd, zddot); broadcasts over leading axes; dtype-agnostic.
    hdot = -v, vdot = -T z/m_L (eq. load), zddot = eta/(m l) - zd^2 z/(1 - z^2) (eq. cable, P_1 a = 0)."""
    x4 = np.asarray(x4); u2 = np.asarray(u2)
    v, z, zd = x4[..., 1], x4[..., 2], x4[..., 3]
    T, eta = u2[..., 0], u2[..., 1]
    m, l = pp.m[0], pp.l[0]
    om2 = zd * zd / (1.0 - z * z)
    return _stack([-v, -T * z / pp.m_L, zd, eta / (m * l) - om2 * z])


def planar1_constraints(z, zd, T, eta, pp: PlanarParams) -> np.ndarray:
    """Constraint functions of U(z, zd) in newtons, feasible iff all <= 0, shape (..., 3):
    [T_min - T, T - m_L a_max, ||u|| - f_max] with ||u||^2 = s^2 + eta^2/(1 - z^2)."""
    z, zd, T, eta = (np.asarray(a) for a in (z, zd, T, eta))
    s = planar1_parallel_thrust(z, zd, T, pp)
    unorm = np.sqrt(s * s + eta * eta / (1.0 - z * z))
    return _stack([pp.T_min - T, T - pp.T_max, unorm - pp.f_max[0]])


def planar1_input_feasible(z, zd, T, eta, pp: PlanarParams, tol: float = 1e-12):
    """(T, eta) in U(z, zd) up to tol [N]; boolean, broadcast shape."""
    return np.all(planar1_constraints(z, zd, T, eta, pp) <= tol, axis=-1)


def _ellipse(z, zd, pp: PlanarParams):
    """U's thrust constraint as the ellipse ((T - Tc)/aT)^2 + (eta/aE)^2 <= 1:
    Tc = m l zd^2 / ((1 - z^2) k), aT = f_max / k, aE = f_max varrho, k = 1 + m/m_L."""
    m, l, f = pp.m[0], pp.l[0], pp.f_max[0]
    k = 1.0 + m / pp.m_L
    c = m * l * zd * zd / (1.0 - z * z)
    return c / k, f / k, f * np.sqrt(1.0 - z * z)


def planar1_input_nonempty(z, zd, pp: PlanarParams):
    """U(z, zd) != empty iff the strip [T_min, m_L a_max] meets the ellipse's T-range."""
    z, zd = np.broadcast_arrays(np.asarray(z, float), np.asarray(zd, float))
    Tc, aT, _ = _ellipse(z, zd, pp)
    return (pp.T_min <= pp.T_max) & (pp.T_min <= Tc + aT) & (pp.T_max >= Tc - aT)


def planar1_input_boundary(z, zd, pp: PlanarParams, K: int = 128, angles=None) -> np.ndarray:
    """K support points of the convex set U(z, zd) in the (T, eta) plane, shape (..., K, 2).

    Point k maximizes cos(th_k) T + sin(th_k) eta over U, th_k = 2 pi k / K (or the given
    `angles`, increasing in [0, 2 pi)), so the points lie on the boundary in counterclockwise
    order and max_k <p, b_k> is an inner approximation of the support function sigma_U(p), exact
    along the K directions (plan K2). The maximizer over the ellipse is closed-form; when its T
    leaves [T_min, m_L a_max] the maximizer sits on that vertical edge at the extreme eta (a
    corner) or, for a direction exactly along +-T, at eta = 0. Since U is much wider in eta than
    in T, uniformly spaced directions place many points on the corners (T_min, +-E): correct for
    the support function, but pass `angles` to concentrate points on the arcs if a polygon with
    distinct vertices is wanted. Nodes where U is empty (see planar1_input_nonempty) get NaN.
    """
    z, zd = np.broadcast_arrays(np.asarray(z, float), np.asarray(zd, float))
    Tc, aT, aE = _ellipse(z, zd, pp)
    th = 2.0 * np.pi * np.arange(K) / K if angles is None else np.asarray(angles, float).reshape(-1)
    dT, dE = np.cos(th), np.sin(th)
    gT = dT * aT                                    # (K,)
    gE = dE * aE[..., None]                         # (..., K)
    gn = np.sqrt(gT * gT + gE * gE)
    T_e = Tc[..., None] + aT * gT / gn              # ellipse maximizer
    eta_e = aE[..., None] * gE / gn
    below, above = T_e < pp.T_min, T_e > pp.T_max
    T_b = np.where(below, pp.T_min, np.where(above, pp.T_max, T_e))
    E = aE[..., None] * np.sqrt(np.clip(1.0 - ((T_b - Tc[..., None]) / aT) ** 2, 0.0, None))
    eta_b = np.where(below | above, np.sign(dE) * E, eta_e)
    out = np.stack([T_b, eta_b], axis=-1)
    empty = ~planar1_input_nonempty(z, zd, pp)
    if np.any(empty):
        out = np.where(empty[..., None, None], np.nan, out)
    return out


def planar1_max_speeds(z, zd, pp: PlanarParams) -> tuple[np.ndarray, np.ndarray]:
    """(max_U |vdot|, max_U |zddot|) at (z, zd), for the CFL condition of the kernel solver:
    |vdot| <= T_top |z| / m_L with T_top the largest tension in U; |zddot| <= eta_top/(m l) + zd^2 |z|/(1 - z^2)
    with eta_top the largest |eta| in U (at the admissible T closest to the ellipse center)."""
    z, zd = np.broadcast_arrays(np.asarray(z, float), np.asarray(zd, float))
    m, l = pp.m[0], pp.l[0]
    Tc, aT, aE = _ellipse(z, zd, pp)
    T_top = np.minimum(pp.T_max, Tc + aT)
    T_c = np.clip(Tc, pp.T_min, pp.T_max)
    eta_top = aE * np.sqrt(np.clip(1.0 - ((T_c - Tc) / aT) ** 2, 0.0, None))
    vdot_max = T_top * np.abs(z) / pp.m_L
    zdd_max = eta_top / (m * l) + zd * zd * np.abs(z) / (1.0 - z * z)
    return vdot_max, zdd_max


# ---------------------------------------------------------------- two cables (K3)
def _planar2_geometry(x6, u4, pp: PlanarParams):
    x6 = np.asarray(x6); u4 = np.asarray(u4)
    z1, zd1, z2, zd2 = x6[..., 2], x6[..., 3], x6[..., 4], x6[..., 5]
    T1, T2 = u4[..., 0], u4[..., 1]
    r1, r2 = np.sqrt(1.0 - z1 * z1), np.sqrt(1.0 - z2 * z2)
    q12 = z1 * z2 + r1 * r2                       # q_1 . q_2
    ya = (T1 * z1 + T2 * z2) / pp.m_L             # y . a
    q1a = (T1 + T2 * q12) / pp.m_L                # q_1 . a
    q2a = (T1 * q12 + T2) / pp.m_L                # q_2 . a
    om2_1 = zd1 * zd1 / (1.0 - z1 * z1)
    om2_2 = zd2 * zd2 / (1.0 - z2 * z2)
    return z1, zd1, z2, zd2, T1, T2, q12, ya, q1a, q2a, om2_1, om2_2


def planar2_rhs(x6, u4, pp: PlanarParams) -> np.ndarray:
    """(h, v, z1, zd1, z2, zd2), (T1, T2, eta1, eta2) -> state derivative; broadcasts; dtype-agnostic.
    vdot = -(T1 z1 + T2 z2)/m_L; zddot_i = eta_i/(m_i l_i) - (y . a - z_i q_i . a)/l_i - zd_i^2 z_i/(1 - z_i^2)
    (eq. cable projected on y with a = (T1 q1 + T2 q2)/m_L)."""
    x6 = np.asarray(x6); u4 = np.asarray(u4)
    v = x6[..., 1]
    eta1, eta2 = u4[..., 2], u4[..., 3]
    z1, zd1, z2, zd2, T1, T2, q12, ya, q1a, q2a, om2_1, om2_2 = _planar2_geometry(x6, u4, pp)
    m, l = pp.m, pp.l
    zdd1 = eta1 / (m[0] * l[0]) - (ya - z1 * q1a) / l[0] - om2_1 * z1
    zdd2 = eta2 / (m[1] * l[1]) - (ya - z2 * q2a) / l[1] - om2_2 * z2
    return _stack([-v, -ya, zd1, zdd1, zd2, zdd2])


def planar2_parallel_thrust(x6, u4, pp: PlanarParams) -> np.ndarray:
    """s_i = T_i + m_i q_i . a - m_i l_i ||qdot_i||^2 (eq. tension), shape (..., 2)."""
    z1, zd1, z2, zd2, T1, T2, q12, ya, q1a, q2a, om2_1, om2_2 = _planar2_geometry(x6, u4, pp)
    m, l = pp.m, pp.l
    return _stack([T1 + m[0] * q1a - m[0] * l[0] * om2_1, T2 + m[1] * q2a - m[1] * l[1] * om2_2])


def planar2_constraints(x6, u4, pp: PlanarParams) -> np.ndarray:
    """Constraint functions in newtons, feasible iff all <= 0, shape (..., 5):
    [T_min - T1, T_min - T2, ||u_1|| - f_max1, ||u_2|| - f_max2, ||T1 q1 + T2 q2|| - m_L a_max] (D-4)."""
    x6 = np.asarray(x6); u4 = np.asarray(u4)
    z1, zd1, z2, zd2, T1, T2, q12, *_ = _planar2_geometry(x6, u4, pp)
    eta1, eta2 = u4[..., 2], u4[..., 3]
    s = planar2_parallel_thrust(x6, u4, pp)
    u1 = np.sqrt(s[..., 0] ** 2 + eta1 * eta1 / (1.0 - z1 * z1))
    u2 = np.sqrt(s[..., 1] ** 2 + eta2 * eta2 / (1.0 - z2 * z2))
    sumTq = np.sqrt(T1 * T1 + T2 * T2 + 2.0 * T1 * T2 * q12)
    return _stack([pp.T_min - T1, pp.T_min - T2, u1 - pp.f_max[0], u2 - pp.f_max[1], sumTq - pp.T_max])


def planar2_input_feasible(x6, u4, pp: PlanarParams, tol: float = 1e-12):
    return np.all(planar2_constraints(x6, u4, pp) <= tol, axis=-1)
