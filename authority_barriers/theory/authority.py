"""Directional authority (Def. 8, Lemma 9) and the barrier data (nominal, relaxed, robust)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import E3, Params


def phi(z, T_bar, T_min: float):
    """phi_i(z) = T_bar_i z_+ - T_min z_- : best contribution of a cable with z_i = z (Def. 8)."""
    z = np.asarray(z, float)
    return np.where(z > 0, T_bar * z, T_min * z)


def phi_slope(z, T_bar, T_min: float):
    """Slope of phi_i: T_bar_i for z > 0, T_min for z <= 0 (the value at 0 is immaterial)."""
    z = np.asarray(z, float)
    return np.where(z > 0, T_bar, T_min) * np.ones_like(z)


@dataclass(frozen=True)
class BarrierData:
    """Data (T_bar, T_min, nu, zeta_bar, m_L, offset) with which alpha_y and D are evaluated.
    offset = -g e3^T y is zero for a wall with horizontal normal. T_min is the floor of the tensions of
    the maneuver (the hover floor T_h in the data with altitude barriers); nu_dec, if given, is the
    deceleration of the swing (profile.pieces), nu its acceleration and the bound of the set V."""
    T_bar: np.ndarray
    T_min: float
    nu: float
    zeta_bar: float
    m_L: float
    offset: float = 0.0
    kind: str = "nominal"
    nu_dec: float | None = None

    @property
    def N(self) -> int:
        return int(np.asarray(self.T_bar).size)

    @property
    def alpha_sat(self) -> float:
        """Authority with every coordinate at zeta_bar (Sec. IV-C); must be positive for D < inf."""
        return float(np.sum(self.T_bar) * self.zeta_bar / self.m_L + self.offset)

    def alpha(self, zeta) -> float:
        """alpha_y(q) = (1/m_L) sum_i phi_i(zeta_i) + offset (eq. authority)."""
        return float(np.sum(phi(zeta, self.T_bar, self.T_min)) / self.m_L + self.offset)

    @staticmethod
    def nominal(p: Params) -> "BarrierData":
        return BarrierData(np.asarray(p.T_bar, float), p.T_min, p.nu, p.z_bar, p.m_L,
                           -p.g * float(E3 @ p.y_vec), "nominal")

    @staticmethod
    def certified(p: Params) -> "BarrierData":
        """Data of the maneuver with altitude barriers: the floor of its tensions is the hover floor T_h
        (N T_h cos(theta_q) >= m_L g) and its swing decelerates at nu_dec < nu. Needs a set with these constants."""
        if p.T_h is None or p.nu_dec is None:
            raise ValueError(f"set {p.name} has no hover floor T_h or no swing deceleration nu_dec")
        return BarrierData(np.asarray(p.T_bar, float), float(p.T_h), p.nu, p.z_bar, p.m_L,
                           -p.g * float(E3 @ p.y_vec), "certified", float(p.nu_dec))

    @staticmethod
    def relaxed(p: Params) -> "BarrierData":
        """Optimistic data of Prop. 15: T_bar_rel = f_max + m a_max + m l omega_bar^2, nu_rel."""
        return BarrierData(np.asarray(p.T_bar_rel, float), p.T_min, p.nu_rel, p.z_bar, p.m_L,
                           -p.g * float(E3 @ p.y_vec), "relaxed")

    @staticmethod
    def robust(p: Params, eps=0.0) -> "BarrierData":
        """Rem. 18 substitutions: nu - d_bar_i/(m_i l_i) - d_bar_L/(m_L l_i), zeta_bar - eps,
        T_bar_i - d_bar_i, and the deceleration offset -d_bar_L/m_L."""
        nu_rob = float(np.min(p.nu - p.d_bar_i / (p.m_arr * p.l_arr) - p.d_bar_L / (p.m_L * p.l_arr)))
        if nu_rob <= 0:
            raise ValueError("robust swing acceleration is not positive for these disturbance budgets")
        eps_max = float(np.max(np.asarray(eps, float)))
        return BarrierData(np.asarray(p.T_bar, float) - p.d_bar_i, p.T_min, nu_rob, p.z_bar - eps_max, p.m_L,
                           -p.g * float(E3 @ p.y_vec) - p.d_bar_L / p.m_L, "robust")


def alpha_y(zeta, data: BarrierData) -> float:
    return data.alpha(zeta)


def support_function_bruteforce(zeta, data: BarrierData) -> float:
    """max over the tension box of (1/m_L) sum_i T_i zeta_i + offset, by enumerating the
    2^N vertices (test oracle for Lemma 9)."""
    zeta = np.asarray(zeta, float)
    N = zeta.size
    best = -np.inf
    for mask in range(2 ** N):
        T = np.where([(mask >> i) & 1 for i in range(N)], data.T_bar, data.T_min)
        best = max(best, float(T @ zeta) / data.m_L + data.offset)
    return best
