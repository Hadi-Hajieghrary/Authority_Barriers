"""Swing of one coordinate to zeta_bar (Sec. IV-C, eq. profile) and its partial derivatives
(App. A(ii)). All functions broadcast over numpy arrays of (zeta, omega).

Without `nu_dec` the swing is time-optimal: acceleration nu, then deceleration nu. With
nu_dec < nu the swing accelerates at nu and decelerates at nu_dec, so that it arrives along the
curve omega^2 = 2 nu_dec (zeta_bar - zeta), which lies inside V(nu, zeta_bar); from a state on or
above that curve it decelerates at the constant rate omega^2 / (2 (zeta_bar - zeta)), which lies
between nu_dec and nu on V and also ends at (zeta_bar, 0)."""
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


def pieces(zeta, omega, nu: float, zbar: float, nu_dec: float | None = None):
    """(r, t1, t2, acc1, kz, kw) of the swing: peak speed r, end t1 of the first piece, arrival t2,
    acceleration acc1 of the first piece, and the halves kz = (1/2) d acc1/d zeta, kw = (1/2) d acc1/d omega
    of its derivatives. Below the deceleration curve acc1 = nu, the second piece decelerates at nu_dec and
    kz = kw = 0. On or above the curve the first piece is the whole swing (t1 = t2 = 2 (zbar - zeta)/omega)
    with acc1 = -omega^2 / (2 (zbar - zeta)), kz = -1/t2^2, kw = -1/t2."""
    zeta = np.asarray(zeta, float)
    omega = np.asarray(omega, float)
    if nu_dec is None or nu_dec >= nu:
        r, t1, t2 = switching(zeta, omega, nu, zbar)
        one = np.ones(np.broadcast(zeta, omega).shape)
        return r, t1, t2, nu * one, 0.0 * one, 0.0 * one
    gap = np.maximum(zbar - zeta, 0.0)
    arc = (omega > 0) & (omega ** 2 >= 2.0 * nu_dec * gap) & (gap > 0)
    r_b = np.sqrt(np.maximum((2.0 * nu * nu_dec * gap + nu_dec * omega ** 2) / (nu + nu_dec), 0.0))
    t1_b = np.maximum((r_b - omega) / nu, 0.0)
    t2_b = t1_b + r_b / nu_dec
    om = np.where(arc, omega, 1.0)
    t2_a = np.where(arc, 2.0 * gap / om, 1.0)
    acc_a = -om ** 2 / (2.0 * np.where(arc, gap, 1.0))
    r = np.where(arc, omega, r_b)
    t1 = np.where(arc, t2_a, t1_b)
    t2 = np.where(arc, t2_a, t2_b)
    return (r, t1, t2, np.where(arc, acc_a, nu), np.where(arc, -1.0 / t2_a ** 2, 0.0), np.where(arc, -1.0 / t2_a, 0.0))


def piece_index(t, t1, t2):
    """0 on [0, t1), 1 on [t1, t2), 2 on [t2, inf); broadcasts t against (t1, t2)."""
    t = np.asarray(t, float)
    return np.where(t < t1, 0, np.where(t < t2, 1, 2))


def eval_piece(piece, t, zeta, omega, nu: float, zbar: float, t2, nu_dec: float | None = None, acc1=None):
    """Value, first and second time derivative of the selected piece's polynomial at t
    (the polynomial is evaluated even outside its interval, which is what the merged-interval
    construction of stopping.py needs at interval starts). acc1: acceleration of the first piece
    (default nu); nu_dec: deceleration of the second piece (default nu)."""
    t = np.asarray(t, float)
    one = np.ones(np.broadcast(t, zeta, omega, t2).shape)
    a1 = nu if acc1 is None else acc1
    nd = nu if nu_dec is None else nu_dec
    val0 = zeta + omega * t + 0.5 * a1 * t ** 2
    d0 = omega + a1 * t
    s = t2 - t
    val1 = zbar - 0.5 * nd * s ** 2
    d1 = nd * s
    val = np.where(piece == 0, val0, np.where(piece == 1, val1, zbar * one))
    d = np.where(piece == 0, d0, np.where(piece == 1, d1, 0.0 * one))
    dd = np.where(piece == 0, a1 * one, np.where(piece == 1, -nd * one, 0.0 * one))
    return val, d, dd


def zeta_tilde(t, zeta, omega, nu: float, zbar: float, nu_dec: float | None = None):
    """(zeta~(t), d/dt zeta~(t), d^2/dt^2 zeta~(t)) of eq. profile."""
    if nu_dec is None:
        _, t1, t2 = switching(zeta, omega, nu, zbar)
        return eval_piece(piece_index(t, t1, t2), t, zeta, omega, nu, zbar, t2)
    _, t1, t2, acc1, _, _ = pieces(zeta, omega, nu, zbar, nu_dec)
    return eval_piece(piece_index(t, t1, t2), t, zeta, omega, nu, zbar, t2, nu_dec, acc1)


def d_zeta_tilde(t, zeta, omega, nu: float, zbar: float, nu_dec: float | None = None):
    """Partial derivatives of zeta~(t; zeta, omega) with respect to zeta and omega (App. A(ii)):
    d/dzeta = 1 + kz t^2 | nu_dec (t2 - t)/r | 0 and d/domega = t + kw t^2 | nu_dec (t2 - t)(1 - omega/r)/nu | 0
    on the three pieces (kz = kw = 0 and nu_dec = nu for the time-optimal swing)."""
    r, t1, t2, _, kz, kw = pieces(zeta, omega, nu, zbar, nu_dec)
    nd = nu if nu_dec is None else nu_dec
    t = np.asarray(t, float)
    pc = piece_index(t, t1, t2)
    rs = np.where(r > 0, r, 1.0)
    dz = np.where(pc == 0, 1.0 + kz * t ** 2, np.where(pc == 1, nd * (t2 - t) / rs, 0.0))
    dw = np.where(pc == 0, t + kw * t ** 2, np.where(pc == 1, nd * (t2 - t) * (1.0 - omega / rs) / nu, 0.0))
    return dz, dw


def accel0(zeta, omega, nu: float, zbar: float, tol: float = 1e-12, nu_dec: float | None = None):
    """Initial acceleration of the swing: that of the first piece before its end, the deceleration
    of the second piece on the deceleration curve, 0 at rest at zbar."""
    _, t1, t2, acc1, _, _ = pieces(zeta, omega, nu, zbar, nu_dec)
    nd = nu if nu_dec is None else nu_dec
    return np.where(t1 > tol, acc1, np.where(t2 > tol, -nd, 0.0))


def zero_crossings_batch(zeta, omega, nu: float, zbar: float, nu_dec: float | None = None) -> np.ndarray:
    """(N, 3) zero-crossing times of the swings in (0, t2): up to two on the first piece
    (columns 0, 1) and one on the second (column 2); NaN where absent."""
    zeta = np.atleast_1d(np.asarray(zeta, float))
    omega = np.atleast_1d(np.asarray(omega, float))
    _, t1, t2, acc1, _, _ = pieces(zeta, omega, nu, zbar, nu_dec)
    nd = nu if nu_dec is None else nu_dec
    out = np.full((zeta.size, 3), np.nan)
    disc = omega ** 2 - 2.0 * acc1 * zeta
    ok = (t1 > 0) & (disc >= 0)
    sq = np.sqrt(np.where(ok, disc, 0.0))
    r1 = (-omega - sq) / acc1
    r2 = (-omega + sq) / acc1
    m1 = ok & (r1 > 0) & (r1 < t1)
    m2 = ok & (r2 > 0) & (r2 < t1)
    out[m1, 0] = r1[m1]
    out[m2, 1] = r2[m2]
    if zbar > 0:
        r3 = t2 - np.sqrt(2.0 * zbar / nd)
        m3 = (t2 > t1) & (r3 >= t1) & (r3 < t2) & (r3 > 0)
        out[m3, 2] = r3[m3]
    return out


def zero_crossings(zeta: float, omega: float, nu: float, zbar: float, nu_dec: float | None = None) -> np.ndarray:
    """Sorted zero-crossing times of one swing (scalar inputs)."""
    z = zero_crossings_batch(zeta, omega, nu, zbar, nu_dec)[0]
    return np.sort(z[~np.isnan(z)])
