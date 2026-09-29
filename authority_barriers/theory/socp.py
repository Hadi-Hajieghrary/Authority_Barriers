"""Shared second-order-cone scaffold of Sec. V.

Decision variables: tensions T in R^N and tangent thrust coordinates c in R^{2N}, so that the
thrust vector of quadrotor i is u_i = s_i(T) q_i + B_i c_i, affine in x = (T, c), with
s_i(T) = T_i + (m_i/m_L) sum_j (q_i . q_j) T_j - m_i l_i |qdot_i|^2   (eq. tension with a = sum T_j q_j / m_L).
Rows: ||u_i|| <= f_max,i (Lorentz), T_i >= T_min, ||sum_j T_j q_j|| <= m_L a_max (D-4),
and, if theta_max < pi/2, the tilt cone e3^T u_i >= cos(theta_max) ||u_i||.
"""
from __future__ import annotations

import math

import numpy as np
from pydrake.solvers import MathematicalProgram

from .params import E3, Params
from .state import State


def tangent_basis(q: np.ndarray) -> np.ndarray:
    """Orthonormal (3, 2) basis of the plane perpendicular to the unit vector q (deterministic)."""
    e = E3 if abs(q @ E3) < 0.9 else np.array([1.0, 0.0, 0.0])
    b1 = e - (q @ e) * q
    b1 /= np.linalg.norm(b1)
    b2 = np.cross(q, b1)
    return np.stack([b1, b2], axis=1)


class ConeProgram:
    def __init__(self, p: Params, st: State, a_max_constraint: bool = True):
        N = p.N
        self.p, self.st = p, st
        self.prog = MathematicalProgram()
        self.T = self.prog.NewContinuousVariables(N, "T")
        self.c = self.prog.NewContinuousVariables(2 * N, "c")
        self.x = np.concatenate([self.T, self.c])
        q = st.q
        om2 = st.omega2()
        self.S = np.eye(N) + (p.m_arr[:, None] / p.m_L) * (q @ q.T)     # s = S T + s0
        self.s0 = -p.m_arr * p.l_arr * om2
        self.B = np.stack([tangent_basis(q[i]) for i in range(N)])         # (N, 3, 2)
        self.A = np.zeros((N, 3, 3 * N))                                    # u_i = A_i x + b_i
        self.b = np.zeros((N, 3))
        for i in range(N):
            self.A[i, :, :N] = np.outer(q[i], self.S[i])
            self.A[i, :, N + 2 * i:N + 2 * i + 2] = self.B[i]
            self.b[i] = q[i] * self.s0[i]
        self.prog.AddBoundingBoxConstraint(p.T_min, np.inf, self.T)
        for i in range(N):
            A4 = np.vstack([np.zeros((1, 3 * N)), self.A[i]])
            b4 = np.concatenate([[p.f_max_arr[i]], self.b[i]])
            self.prog.AddLorentzConeConstraint(A4, b4, self.x)
            if p.theta_max < math.pi - 1e-9:
                if p.theta_max >= math.pi / 2:
                    raise ValueError("tilt limits in [pi/2, pi) are not convex; use theta_max < pi/2 or pi")
                cs = math.cos(p.theta_max)
                A4t = np.vstack([(E3 @ self.A[i]) / cs, self.A[i]])
                b4t = np.concatenate([[(E3 @ self.b[i]) / cs], self.b[i]])
                self.prog.AddLorentzConeConstraint(A4t, b4t, self.x)
        if a_max_constraint:
            Aa = np.zeros((4, 3 * N)); Aa[1:, :N] = q.T
            ba = np.array([p.m_L * p.a_max, 0.0, 0.0, 0.0])
            self.prog.AddLorentzConeConstraint(Aa, ba, self.x)

    # braking deceleration b = y^T xddot_L = (1/m_L) sum_j T_j z_j - g e3^T y : coefficients on T
    def b_coeffs(self) -> tuple[np.ndarray, float]:
        p = self.p
        return self.st.z(p) / p.m_L, -p.g * float(E3 @ p.y_vec)

    def thrusts(self, xval: np.ndarray) -> np.ndarray:
        return np.einsum("ijk,k->ij", self.A, xval) + self.b

    def tensions_from_x(self, xval: np.ndarray) -> np.ndarray:
        return xval[: self.p.N]
