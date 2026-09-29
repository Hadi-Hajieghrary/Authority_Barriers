"""Brute-force stopping distance on a dense time grid (test oracle for stopping.py only)."""
from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_simpson

from . import profile as PF
from .authority import BarrierData, phi


def D_oracle(v: float, zeta, omega, data: BarrierData, dt: float = 1e-4):
    """E on a uniform grid with cumulative Simpson integration; returns (D, t_star)."""
    zeta = np.atleast_1d(np.asarray(zeta, float))
    omega = np.atleast_1d(np.asarray(omega, float))
    N = zeta.size
    nu, zb = data.nu, data.zeta_bar
    _, _, t2 = PF.switching(zeta, omega, nu, zb) if N else (None, None, np.zeros(1))
    t2max = float(np.max(t2)) if N else 0.0
    a_sat = data.alpha_sat
    if a_sat <= 0:
        return np.inf, np.inf
    # after t2max the deceleration is a_sat; bound the speed reached at t2max from above
    T_bar = np.asarray(data.T_bar, float)
    a_low = -(float(np.sum(np.maximum(T_bar, data.T_min))) * zb / data.m_L) + min(0.0, data.offset)
    v_bound = max(0.0, v) + abs(a_low) * t2max
    t_end = t2max + v_bound / a_sat + 0.5
    n = int(np.ceil(t_end / dt))
    n += n % 2  # even number of intervals for Simpson
    t = np.linspace(0.0, n * dt, n + 1)
    if N:
        Z = np.stack([PF.zeta_tilde(t, zeta[i], omega[i], nu, zb)[0] for i in range(N)])
        alpha = phi(Z, T_bar[:, None], data.T_min).sum(0) / data.m_L + data.offset
    else:
        alpha = np.full(t.size, data.offset)
    vt = v - cumulative_simpson(alpha, dx=dt, initial=0.0)
    E = cumulative_simpson(vt, dx=dt, initial=0.0)
    j = int(np.argmax(E))
    return float(E[j]), float(t[j])
