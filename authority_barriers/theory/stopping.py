"""Stopping distance D(x) of the braking maneuver (eq. D), its maximizing times T*(x), and the
exact partial derivatives of E(t; x) (Prop. 17(a)).

Per cable, the time-optimal swing zeta~_i is quadratic on [0, t1_i] and [t1_i, t2_i] and constant
afterwards, and phi_i is linear on each side of zero. Merging the at most 4N breakpoints (switch,
arrival, zero crossings) gives intervals on which alpha~ is a quadratic, v~ = v - int alpha~ a
cubic and E = int v~ a quartic. The maximizers of E over t >= 0 are t = 0 or roots of v~.
Gradients: dE/dv = t, dE/dzeta_i = -iint phi_i'(zeta~_i) d_zeta zeta~_i / m_L, and the same with
d_omega; the breakpoints are held fixed because alpha~ is continuous there (App. A(ii), App. C).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import profile as PF
from .authority import BarrierData

TIE_REL = 1e-9


def _polyval(c, x):
    """Ascending-coefficient polynomial evaluation, c (..., d+1) broadcast against x (...)."""
    c = np.asarray(c, float)
    x = np.asarray(x, float)
    out = np.array(np.broadcast_to(c[..., -1], np.broadcast(c[..., -1], x).shape), dtype=float)
    for j in range(c.shape[-1] - 2, -1, -1):
        out = out * x + c[..., j]
    return out


def real_roots_cubic(c: np.ndarray) -> np.ndarray:
    """Real roots of c0 + c1 x + c2 x^2 + c3 x^3 = 0 for each row of c (K, 4); returns (K, 3)
    with NaN padding. Degenerate leading coefficients fall back to quadratic / linear formulas;
    two Newton steps polish every root."""
    c = np.atleast_2d(np.asarray(c, float))
    K = c.shape[0]
    roots = np.full((K, 3), np.nan)
    scale = np.max(np.abs(c), axis=1)
    scale[scale == 0] = 1.0
    c0, c1, c2, c3 = (c[:, j] / scale for j in range(4))
    tol = 1e-13
    cubic = np.abs(c3) > tol
    quad = (~cubic) & (np.abs(c2) > tol)
    lin = (~cubic) & (~quad) & (np.abs(c1) > tol)
    roots[lin, 0] = -c0[lin] / c1[lin]
    disc = c1 ** 2 - 4.0 * c2 * c0
    mq = quad & (disc >= 0)
    sq = np.sqrt(np.where(mq, disc, 0.0))
    roots[mq, 0] = (-c1[mq] - sq[mq]) / (2.0 * c2[mq])
    roots[mq, 1] = (-c1[mq] + sq[mq]) / (2.0 * c2[mq])
    idx = np.where(cubic)[0]
    if idx.size:
        A = c2[idx] / c3[idx]
        B = c1[idx] / c3[idx]
        C = c0[idx] / c3[idx]
        p = B - A ** 2 / 3.0
        q = 2.0 * A ** 3 / 27.0 - A * B / 3.0 + C
        shift = -A / 3.0
        d3 = -(4.0 * p ** 3 + 27.0 * q ** 2)
        three = d3 > 0
        if np.any(three):
            pp, qq = p[three], q[three]
            mag = 2.0 * np.sqrt(-pp / 3.0)
            arg = np.clip((3.0 * qq / (2.0 * pp)) * np.sqrt(-3.0 / pp), -1.0, 1.0)
            theta = np.arccos(arg) / 3.0
            for k in range(3):
                roots[idx[three], k] = mag * np.cos(theta - 2.0 * np.pi * k / 3.0) + shift[three]
        one = ~three
        if np.any(one):
            pp, qq = p[one], q[one]
            sq1 = np.sqrt(np.maximum(qq ** 2 / 4.0 + pp ** 3 / 27.0, 0.0))
            t = np.cbrt(-qq / 2.0 + sq1) + np.cbrt(-qq / 2.0 - sq1)
            roots[idx[one], 0] = t + shift[one]
    for _ in range(2):
        x = roots
        f = c0[:, None] + c1[:, None] * x + c2[:, None] * x ** 2 + c3[:, None] * x ** 3
        df = c1[:, None] + 2.0 * c2[:, None] * x + 3.0 * c3[:, None] * x ** 2
        upd = np.where(np.abs(df) > 1e-14, f / df, 0.0)
        roots = np.where(np.isnan(x), np.nan, x - upd)
    return roots


@dataclass
class Maneuver:
    """Piecewise-polynomial description of the maneuver from the state (v, zeta, omega)."""
    data: BarrierData
    v: float
    zeta: np.ndarray
    omega: np.ndarray
    r: np.ndarray
    t1: np.ndarray
    t2: np.ndarray
    tau: np.ndarray      # (K,) interval starts, tau[0] = 0
    delta: np.ndarray    # (K,) interval lengths, inf for the last
    slope: np.ndarray    # (K, N) phi_i' on each interval
    a: np.ndarray        # (K, 3) local coefficients of alpha~
    vt: np.ndarray       # (K, 4) local coefficients of v~
    E: np.ndarray        # (K, 5) local coefficients of E
    gz: np.ndarray       # (K, N, 2) local linear coefficients of d alpha~/d zeta_i
    gw: np.ndarray       # (K, N, 2) same for omega_i
    G1z: np.ndarray      # (K, N) int_0^{tau_k} of gz
    G2z: np.ndarray      # (K, N) iint up to tau_k
    G1w: np.ndarray
    G2w: np.ndarray
    in_V: bool
    clamped: bool

    @property
    def N(self) -> int:
        return self.zeta.size

    @property
    def n_breakpoints(self) -> int:
        return self.tau.size - 1

    def interval_of(self, t):
        return np.clip(np.searchsorted(self.tau, np.asarray(t, float), side="right") - 1, 0, self.tau.size - 1)

    def alpha_tilde(self, t):
        k = self.interval_of(t)
        return _polyval(self.a[k], np.asarray(t, float) - self.tau[k])

    def v_tilde(self, t):
        k = self.interval_of(t)
        return _polyval(self.vt[k], np.asarray(t, float) - self.tau[k])

    def E_of(self, t):
        k = self.interval_of(t)
        return _polyval(self.E[k], np.asarray(t, float) - self.tau[k])

    def zeta_tilde(self, t):
        """(N, len(t)) swings of all cables."""
        t = np.atleast_1d(np.asarray(t, float))
        return np.stack([PF.zeta_tilde(t, self.zeta[i], self.omega[i], self.data.nu, self.data.zeta_bar)[0]
                         for i in range(self.N)]) if self.N else np.zeros((0, t.size))

    def grad_E(self, t):
        """(dE/dv, dE/dzeta (N,), dE/domega (N,)) at a single time t (Prop. 17(a))."""
        t = float(t)
        k = int(self.interval_of(t))
        s = t - self.tau[k]
        gz = -(self.G2z[k] + self.G1z[k] * s + self.gz[k, :, 0] * s ** 2 / 2.0 + self.gz[k, :, 1] * s ** 3 / 6.0)
        gw = -(self.G2w[k] + self.G1w[k] * s + self.gw[k, :, 0] * s ** 2 / 2.0 + self.gw[k, :, 1] * s ** 3 / 6.0)
        return t, gz, gw

    def stationary_times(self) -> np.ndarray:
        """All t >= 0 with v~(t) = 0 (roots of the cubic pieces inside their intervals)."""
        R = real_roots_cubic(self.vt)                      # local roots (K, 3)
        lo = -1e-10
        hi = np.where(np.isfinite(self.delta), self.delta + 1e-10, np.inf)[:, None]
        ok = (~np.isnan(R)) & (R >= lo) & (R <= hi)
        times = (self.tau[:, None] + np.clip(R, 0.0, None))[ok]
        return np.unique(np.round(times, 12))

    def running_max(self, t: float) -> float:
        """max_{s >= t} E(s) (eq. runmax): the maximum over s = t and the stationary times >= t."""
        cand = np.concatenate([[t], self.stationary_times()])
        cand = cand[cand >= t - 1e-12]
        return float(np.max(self.E_of(cand)))


def build(v: float, zeta, omega, data: BarrierData) -> Maneuver:
    zeta = np.atleast_1d(np.asarray(zeta, float)).copy()
    omega = np.atleast_1d(np.asarray(omega, float)).copy()
    N = zeta.size
    nu, zb, mL = data.nu, data.zeta_bar, data.m_L
    in_V = bool(np.all(PF.in_V(zeta, omega, nu, zb, tol=1e-12))) if N else True
    clamped = False
    if not in_V:
        zeta, omega, clamped = PF.clamp_to_V(zeta, omega, nu, zb)
    r, t1, t2 = PF.switching(zeta, omega, nu, zb)
    zc = PF.zero_crossings_batch(zeta, omega, nu, zb) if N else np.zeros((0, 3))
    bps = np.concatenate([[0.0], t1[t1 > 0], t2[t2 > t1], zc[~np.isnan(zc)]])
    tau = np.unique(np.round(bps, 13))
    tau = tau[tau >= 0.0]
    K = tau.size
    delta = np.append(np.diff(tau), np.inf)
    tm = np.where(np.isfinite(delta), tau + delta / 2.0, tau + 1.0)
    T_ = tau[:, None]
    M_ = tm[:, None]
    T_bar = np.asarray(data.T_bar, float)
    if N:
        pc = PF.piece_index(M_, t1[None, :], t2[None, :])                     # (K, N)
        val, d1, d2 = PF.eval_piece(pc, T_, zeta[None, :], omega[None, :], nu, zb, t2[None, :])
        valm, _, _ = PF.eval_piece(pc, M_, zeta[None, :], omega[None, :], nu, zb, t2[None, :])
        slope = np.where(valm > 0, T_bar[None, :], data.T_min) * np.ones((K, N))
        a0 = (slope * val).sum(1) / mL + data.offset
        a1 = (slope * d1).sum(1) / mL
        a2 = (slope * d2).sum(1) / (2.0 * mL)
        rs = np.where(r > 0, r, 1.0)[None, :]
        s0 = t2[None, :] - T_
        fac = 1.0 - omega[None, :] / rs
        gz0 = np.where(pc == 0, 1.0, np.where(pc == 1, nu * s0 / rs, 0.0))
        gz1 = np.where(pc == 1, -nu / rs, 0.0) * np.ones((K, N))
        gw0 = np.where(pc == 0, T_ * np.ones((K, N)), np.where(pc == 1, s0 * fac, 0.0))
        gw1 = np.where(pc == 0, 1.0, np.where(pc == 1, -fac, 0.0)) * np.ones((K, N))
        sc = slope / mL
        gz = np.stack([gz0 * sc, gz1 * sc], -1)
        gw = np.stack([gw0 * sc, gw1 * sc], -1)
    else:
        slope = np.zeros((K, 0))
        a0 = np.full(K, data.offset); a1 = np.zeros(K); a2 = np.zeros(K)
        gz = np.zeros((K, 0, 2)); gw = np.zeros((K, 0, 2))
    a = np.stack([a0, a1, a2], 1)
    # sweep over the intervals, vectorized: v~ and E at the interval starts are cumulative sums
    Dk = delta[:-1]
    I1 = a0[:-1] * Dk + a1[:-1] * Dk ** 2 / 2.0 + a2[:-1] * Dk ** 3 / 3.0
    vk = float(v) - np.concatenate([[0.0], np.cumsum(I1)])
    J = vk[:-1] * Dk - (a0[:-1] * Dk ** 2 / 2.0 + a1[:-1] * Dk ** 3 / 6.0 + a2[:-1] * Dk ** 4 / 12.0)
    Ek = np.concatenate([[0.0], np.cumsum(J)])
    vt = np.stack([vk, -a0, -a1 / 2.0, -a2 / 3.0], 1)
    E = np.stack([Ek, vk, -a0 / 2.0, -a1 / 6.0, -a2 / 12.0], 1)

    def cum(g):
        g0, g1 = g[:-1, :, 0], g[:-1, :, 1]
        D_ = Dk[:, None]
        G1 = np.vstack([np.zeros((1, N)), np.cumsum(g0 * D_ + g1 * D_ ** 2 / 2.0, axis=0)])
        G2 = np.vstack([np.zeros((1, N)), np.cumsum(G1[:-1] * D_ + g0 * D_ ** 2 / 2.0 + g1 * D_ ** 3 / 6.0, axis=0)])
        return G1, G2

    G1z, G2z = cum(gz)
    G1w, G2w = cum(gw)
    return Maneuver(data, float(v), zeta, omega, r, t1, t2, tau, delta, slope, a, vt, E, gz, gw,
                    G1z, G2z, G1w, G2w, in_V, clamped)


@dataclass
class StoppingResult:
    D: float
    t_star: np.ndarray        # all maximizing times (ties within TIE_REL), ascending
    grad_v: np.ndarray        # (J,) = t_star
    grad_zeta: np.ndarray     # (J, N)
    grad_omega: np.ndarray    # (J, N)
    in_V: bool
    clamped: bool
    n_breakpoints: int
    maneuver: Maneuver
    t_lmax: np.ndarray = None         # all local maximizers of E (global ones first), for the sampled filter
    grad_v_lmax: np.ndarray = None
    grad_zeta_lmax: np.ndarray = None
    grad_omega_lmax: np.ndarray = None

    @property
    def t_min(self) -> float:
        return float(self.t_star[0])

    @property
    def unique(self) -> bool:
        return self.t_star.size == 1


def evaluate(v: float, zeta, omega, data: BarrierData) -> StoppingResult:
    """D(x) = max_{t >= 0} E(t; x) with all maximizers and the gradient of E at each (eq. D, Prop. 17)."""
    mn = build(v, zeta, omega, data)
    if data.alpha_sat <= 0:
        return StoppingResult(np.inf, np.array([np.inf]), np.array([np.inf]),
                              np.full((1, mn.N), np.nan), np.full((1, mn.N), np.nan),
                              mn.in_V, mn.clamped, mn.n_breakpoints, mn)
    roots = mn.stationary_times()
    cand = np.concatenate([[0.0], roots])
    vals = mn.E_of(cand)
    D = float(np.max(vals))
    ties = cand[vals >= D - TIE_REL * max(1.0, abs(D))]
    ties = np.unique(np.round(ties, 12))
    grads = [mn.grad_E(t) for t in ties]
    # local maximizers of E: stationary times where E'' = -alpha~ < 0, plus t = 0 when v <= 0 (boundary maximum)
    lm = roots[mn.alpha_tilde(roots) > 0] if roots.size else roots
    if mn.v <= 0.0:
        lm = np.concatenate([[0.0], lm])
    lm = np.unique(np.round(np.concatenate([ties, lm]), 12))
    order = np.argsort(-mn.E_of(lm), kind="stable")           # global maximizers first
    lm = lm[order]
    glm = [mn.grad_E(t) for t in lm]
    return StoppingResult(D, ties, np.array([g[0] for g in grads]),
                          np.array([g[1] for g in grads]).reshape(len(ties), mn.N),
                          np.array([g[2] for g in grads]).reshape(len(ties), mn.N),
                          mn.in_V, mn.clamped, mn.n_breakpoints, mn,
                          lm, np.array([g[0] for g in glm]),
                          np.array([g[1] for g in glm]).reshape(len(lm), mn.N),
                          np.array([g[2] for g in glm]).reshape(len(lm), mn.N))


def D_of(v: float, zeta, omega, data: BarrierData) -> float:
    return evaluate(v, zeta, omega, data).D


def H_of(h: float, v: float, zeta, omega, data: BarrierData) -> float:
    """Authority barrier H = h - D (Sec. IV-C)."""
    return h - D_of(v, zeta, omega, data)
