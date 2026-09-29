"""Time-optimal swing of one coordinate (Sec. IV-C, eq. profile) and its partial derivatives
(App. A(ii)). All functions broadcast over numpy arrays of (zeta, omega)."""
from __future__ import annotations

import numpy as np


def switching(zeta, omega, nu: float, zbar: float):
    """r = sqrt(omega^2/2 + nu (zbar - zeta)) (largest speed), t1 = (r - omega)/nu (switch),
    t2 = t1 + r/nu (arrival at (zbar, 0)). On V, r >= omega so t1 >= 0; tiny negatives are clamped."""
    zeta = np.asarray(zeta, float)
    omega = np.asarray(omega, float)
    r = np.sqrt(np.maximum(0.5 * omega ** 2 + nu * (zbar - zeta), 0.0))
    t1 = np.maximum((r - omega) / nu, 0.0)
    t2 = t1 + r / nu
    return r, t1, t2


def bounds_V(omega, nu: float, zbar: float):
    """Lower and upper bound on zeta in the admissible swing set V(nu, zbar) (eq. box)."""
    omega = np.asarray(omega, float)
    lo = -zbar + np.maximum(-omega, 0.0) ** 2 / (2 * nu)
    hi = zbar - np.maximum(omega, 0.0) ** 2 / (2 * nu)
    return lo, hi


def in_V(zeta, omega, nu: float, zbar: float, tol: float = 0.0):
    lo, hi = bounds_V(omega, nu, zbar)
    zeta = np.asarray(zeta, float)
    return (zeta >= lo - tol) & (zeta <= hi + tol)


def clamp_to_V(zeta, omega, nu: float, zbar: float):
    """Project (zeta, omega) into V: first |omega| <= 2 sqrt(nu zbar), then zeta into [lo, hi].
    Returns (zeta_c, omega_c, changed). Used only for sampled-data states just outside V (D-12)."""
    zeta = np.array(zeta, float, copy=True)
    omega = np.array(omega, float, copy=True)
    om_max = 2.0 * np.sqrt(nu * zbar)
    omega_c = np.clip(omega, -om_max, om_max)
    lo, hi = bounds_V(omega_c, nu, zbar)
    zeta_c = np.clip(zeta, lo, hi)
    changed = bool(np.any(zeta_c != zeta) or np.any(omega_c != omega))
    return zeta_c, omega_c, changed


def piece_index(t, t1, t2):
    """0 on [0, t1), 1 on [t1, t2), 2 on [t2, inf); broadcasts t against (t1, t2)."""
    t = np.asarray(t, float)
    return np.where(t < t1, 0, np.where(t < t2, 1, 2))


def eval_piece(piece, t, zeta, omega, nu: float, zbar: float, t2):
    """Value, first and second time derivative of the selected piece's polynomial at t
    (the polynomial is evaluated even outside its interval, which is what the merged-interval
    construction of stopping.py needs at interval starts)."""
    t = np.asarray(t, float)
    one = np.ones(np.broadcast(t, zeta, omega, t2).shape)
    val0 = zeta + omega * t + 0.5 * nu * t ** 2
    d0 = omega + nu * t
    s = t2 - t
    val1 = zbar - 0.5 * nu * s ** 2
    d1 = nu * s
    val = np.where(piece == 0, val0, np.where(piece == 1, val1, zbar * one))
    d = np.where(piece == 0, d0, np.where(piece == 1, d1, 0.0 * one))
    dd = np.where(piece == 0, nu * one, np.where(piece == 1, -nu * one, 0.0 * one))
    return val, d, dd


def zeta_tilde(t, zeta, omega, nu: float, zbar: float):
    """(zeta~(t), d/dt zeta~(t), d^2/dt^2 zeta~(t)) of eq. profile."""
    _, t1, t2 = switching(zeta, omega, nu, zbar)
    pc = piece_index(t, t1, t2)
    return eval_piece(pc, t, zeta, omega, nu, zbar, t2)


def d_zeta_tilde(t, zeta, omega, nu: float, zbar: float):
    """Partial derivatives of zeta~(t; zeta, omega) with respect to zeta and omega (App. A(ii)):
    d/dzeta = 1 | nu (t2 - t)/r | 0 and d/domega = t | (t2 - t)(1 - omega/r) | 0 on the three pieces."""
    r, t1, t2 = switching(zeta, omega, nu, zbar)
    t = np.asarray(t, float)
    pc = piece_index(t, t1, t2)
    rs = np.where(r > 0, r, 1.0)
    dz = np.where(pc == 0, 1.0, np.where(pc == 1, nu * (t2 - t) / rs, 0.0))
    dw = np.where(pc == 0, t, np.where(pc == 1, (t2 - t) * (1.0 - omega / rs), 0.0))
    return dz, dw


def accel0(zeta, omega, nu: float, zbar: float, tol: float = 1e-12):
    """Initial acceleration of the time-optimal swing: +nu before the switch, -nu on the
    switching curve, 0 at rest at zbar."""
    _, t1, t2 = switching(zeta, omega, nu, zbar)
    return np.where(t1 > tol, nu, np.where(t2 > tol, -nu, 0.0))


def zero_crossings_batch(zeta, omega, nu: float, zbar: float) -> np.ndarray:
    """(N, 3) zero-crossing times of the swings in (0, t2): up to two on the first piece
    (columns 0, 1) and one on the second (column 2); NaN where absent."""
    zeta = np.atleast_1d(np.asarray(zeta, float))
    omega = np.atleast_1d(np.asarray(omega, float))
    _, t1, t2 = switching(zeta, omega, nu, zbar)
    out = np.full((zeta.size, 3), np.nan)
    disc = omega ** 2 - 2.0 * nu * zeta
    ok = (t1 > 0) & (disc >= 0)
    sq = np.sqrt(np.where(ok, disc, 0.0))
    r1 = (-omega - sq) / nu
    r2 = (-omega + sq) / nu
    m1 = ok & (r1 > 0) & (r1 < t1)
    m2 = ok & (r2 > 0) & (r2 < t1)
    out[m1, 0] = r1[m1]
    out[m2, 1] = r2[m2]
    if zbar > 0:
        r3 = t2 - np.sqrt(2.0 * zbar / nu)
        m3 = (t2 > t1) & (r3 >= t1) & (r3 < t2) & (r3 > 0)
        out[m3, 2] = r3[m3]
    return out


def zero_crossings(zeta: float, omega: float, nu: float, zbar: float) -> np.ndarray:
    """Sorted zero-crossing times of one swing (scalar inputs)."""
    z = zero_crossings_batch(zeta, omega, nu, zbar)[0]
    return np.sort(z[~np.isnan(z)])
