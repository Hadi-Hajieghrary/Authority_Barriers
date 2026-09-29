"""Taut-cable state x = (x_L, xdot_L, q, qdot) and the wall coordinates of Sec. II-C."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import E3, Params


@dataclass
class State:
    x_L: np.ndarray   # payload position (3,)
    v_L: np.ndarray   # payload velocity (3,)
    q: np.ndarray     # cable unit directions, payload -> quadrotor (N, 3)
    qd: np.ndarray    # their rates, tangent to the sphere (N, 3)

    @staticmethod
    def from_swing(p: Params, h, v, z, w, zd, wd, x_perp=0.0, altitude=15.0, v_perp=0.0, v_z=0.0) -> "State":
        """State from wall coordinates: distance h, approach speed v (positive toward the wall),
        and per-cable swing coordinates; q_i = z_i y + w_i r + varrho_i e3 (Sec. II-C)."""
        y, r, n = p.y_vec, p.r_vec, p.n_vec
        z, w, zd, wd = (np.asarray(a, float) for a in (z, w, zd, wd))
        varrho = np.sqrt(1.0 - z ** 2 - w ** 2)
        q = z[:, None] * y + w[:, None] * r + varrho[:, None] * E3
        varrho_d = -(z * zd + w * wd) / varrho
        qd = zd[:, None] * y + wd[:, None] * r + varrho_d[:, None] * E3
        x_L = (p.d0 - h) * n + x_perp * r + altitude * E3
        v_L = v * n + v_perp * r + v_z * E3
        return State(x_L, v_L, q, qd)

    def h(self, p: Params) -> float: return float(p.d0 - p.n_vec @ self.x_L)
    def v(self, p: Params) -> float: return float(p.n_vec @ self.v_L)
    def z(self, p: Params) -> np.ndarray: return self.q @ p.y_vec
    def w(self, p: Params) -> np.ndarray: return self.q @ p.r_vec
    def zd(self, p: Params) -> np.ndarray: return self.qd @ p.y_vec
    def wd(self, p: Params) -> np.ndarray: return self.qd @ p.r_vec
    def omega2(self) -> np.ndarray: return np.einsum("ij,ij->i", self.qd, self.qd)

    def quad_positions(self, p: Params) -> np.ndarray: return self.x_L + p.l_arr[:, None] * self.q
    def quad_velocities(self, p: Params) -> np.ndarray: return self.v_L + p.l_arr[:, None] * self.qd
