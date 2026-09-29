"""Samplers for the experiments: admissible swing sets, X_op, the boundary layer of X_RF, and
the set D of Thm. 5 (constructive, Sec. VI Q1)."""
from __future__ import annotations

import numpy as np

from . import hocbf as H
from . import profile as PF
from . import stopping as SD
from .authority import BarrierData
from .params import Params
from .socp import ConeProgram
from .state import State
from pydrake.solvers import ClarabelSolver


def sample_V(rng, n: int, nu: float, zbar: float, frac: float = 1.0):
    """(zeta, omega) uniformly in the admissible swing set V(nu, zbar) (scaled by frac)."""
    om_max = 2.0 * np.sqrt(nu * zbar) * frac
    omega = rng.uniform(-om_max, om_max, n)
    lo, hi = PF.bounds_V(omega, nu, zbar)
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * frac      # shrink toward the centre, never outside
    zeta = rng.uniform(mid - half, mid + half)
    return zeta, omega


def sample_tangent_rates(rng, q: np.ndarray, speed) -> np.ndarray:
    """Tangent vectors qdot_i with |qdot_i| = speed_i."""
    d = rng.normal(size=q.shape)
    d -= (np.einsum("ij,ij->i", d, q))[:, None] * q
    d /= np.linalg.norm(d, axis=1)[:, None]
    return d * np.asarray(speed, float)[:, None]


def sample_Xop(rng, p: Params, n: int, h_range=(1.0, 20.0), v_range=(-1.0, 4.0), strict: float = 0.95):
    """Random states in X_op (boxes and swing-rate bound), cables anywhere in the cone."""
    out = []
    for _ in range(n):
        z = rng.uniform(-p.z_bar, p.z_bar, p.N) * strict
        w = rng.uniform(-p.w_bar, p.w_bar, p.N) * strict
        q = State.from_swing(p, 1.0, 0.0, z, w, np.zeros(p.N), np.zeros(p.N)).q
        qd = sample_tangent_rates(rng, q, rng.uniform(0, p.omega_bar * strict, p.N))
        st = State.from_swing(p, rng.uniform(*h_range), rng.uniform(*v_range), z, w, qd @ p.y_vec, qd @ p.r_vec)
        out.append(st)
    return out


def sample_XRF_layer(rng, p: Params, n: int, data: BarrierData | None = None, v_range=(1.0, 4.0),
                     delta_frac: float = 0.05, frac: float = 0.95, h_offset: float = 0.0, delta_abs: float | None = None):
    """States in the boundary layer of X_RF: swing states in V (z) and V_w (w), h = D + h_offset + delta,
    delta ~ U[0, delta_frac D] (E2; h_offset = delta/kappa for the margin-enforced filter)."""
    data = data or BarrierData.nominal(p)
    out = []
    while len(out) < n:
        z, zd = sample_V(rng, p.N, data.nu, data.zeta_bar, frac)
        w, wd = sample_V(rng, p.N, p.nu_w, p.w_bar, frac)
        v = rng.uniform(*v_range)
        D = SD.D_of(v, z, zd, data)
        if not np.isfinite(D):
            continue
        h = D + h_offset + (rng.uniform(0.0, delta_abs) if delta_abs else rng.uniform(0.0, delta_frac * max(D, 1e-3)))
        st = State.from_swing(p, h, v, z, w, zd, wd)
        if np.all(np.sqrt(st.omega2()) <= p.omega_bar):
            out.append(st)
    return out


def sample_D(rng, p: Params, n: int, a1, a2, v_range=(2.0, 4.0), dt: float = 0.005, margin_ticks: int = 3,
             z_frac=(0.6, 0.95), max_tries: int = 200000, require_feasible: bool = True):
    """Constructive sampler for the open set D of Thm. 5: all cables well inside Sigma_y
    (z_i in -[z_frac] z_bar), v0 in v_range, h0 chosen so that 0 < t_psi < tau - margin_ticks dt;
    accepted iff U(x0) is nonempty and (require_feasible) the HOCBF program is feasible at x0
    (mu(x0) > 0), so that the trial shows feasibility being lost rather than absent."""
    out, tries = [], 0
    solver = ClarabelSolver()
    while len(out) < n and tries < max_tries:
        tries += 1
        z = -rng.uniform(z_frac[0], z_frac[1], p.N) * p.z_bar
        w = rng.uniform(-0.9, 0.9, p.N) * p.w_bar
        q = State.from_swing(p, 1.0, 0.0, z, w, np.zeros(p.N), np.zeros(p.N)).q
        qd = sample_tangent_rates(rng, q, rng.uniform(0, 0.5 * p.omega_bar, p.N))
        v0 = rng.uniform(*v_range)
        tau = float(np.min(-z)) / p.omega_bar
        t_max = tau - margin_ticks * dt
        if t_max <= 1e-3:
            continue
        h0 = a1.inv(v0) + v0 * rng.uniform(0.02 * t_max, t_max)
        st = State.from_swing(p, h0, v0, z, w, qd @ p.y_vec, qd @ p.r_vec)
        ok, conds = H.in_D(st, p, a1, a2)
        if not ok:
            continue
        cp = ConeProgram(p, st)
        if not solver.Solve(cp.prog).is_success():
            continue
        if require_feasible and H.mu(p, st, a1, a2) <= 1e-6:
            continue
        out.append(st)
    return out
