"""HOCBF quantities of Sec. III: class-K functions, psi_1, beta, the exact directional
authority (eq. exact) as a Clarabel SOCP, the feasibility margin mu, tau, t_psi, and the
membership tests for Sigma_y, X_op, C_HO and the set D of Thm. 5."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydrake.solvers import ClarabelSolver

from .params import E3, Params
from .socp import ConeProgram
from .state import State


@dataclass(frozen=True)
class LinearClassK:
    k: float
    def __call__(self, s): return self.k * s
    def deriv(self, s): return self.k
    def inv(self, s): return s / self.k


def psi1(h: float, v: float, a1) -> float:
    return -v + a1(h)


def beta(st: State, p: Params, a1, a2) -> float:
    """Right-hand side of the HOCBF condition b >= beta(x) (eq. hocbf)."""
    h, v = st.h(p), st.v(p)
    return a1.deriv(h) * v - a2(psi1(h, v, a1))


def t_psi(h0: float, v0: float, a1) -> float:
    return (h0 - a1.inv(v0)) / v0


def tau(st: State, p: Params) -> float:
    """Minimum time to leave Sigma_y: min_i zeta_i / omega_bar with zeta_i = q_i^T n = -z_i (Thm. 5, step 2)."""
    return float(np.min(-st.z(p))) / p.omega_bar


def alpha_exact(p: Params, st: State, solver=None):
    """Exact directional authority alpha^ex_y(x) = max { y^T xddot_L : u in U(x) } (eq. exact).
    Returns (value, MathematicalProgramResult); value = -inf when U(x) is empty."""
    cp = ConeProgram(p, st)
    cT, c0 = cp.b_coeffs()
    cp.prog.AddLinearCost(-cT, 0.0, cp.T)
    res = (solver or ClarabelSolver()).Solve(cp.prog)
    if not res.is_success():
        return -np.inf, res
    return float(cT @ res.GetSolution(cp.T) + c0), res


def mu(p: Params, st: State, a1, a2) -> float:
    """Feasibility margin of the HOCBF filter (Prop. 3): K_HO(x) nonempty iff mu >= 0."""
    return alpha_exact(p, st)[0] - beta(st, p, a1, a2)


def in_Sigma_y(st: State, p: Params, strict: bool = False) -> bool:
    z = st.z(p)
    return bool(np.all(z < 0) if strict else np.all(z <= 0))


def in_Xop(st: State, p: Params, strict: bool = False) -> bool:
    z, w, om = np.abs(st.z(p)), np.abs(st.w(p)), np.sqrt(st.omega2())
    if strict:
        return bool(np.all(z < p.z_bar) and np.all(w < p.w_bar) and np.all(om < p.omega_bar))
    return bool(np.all(z <= p.z_bar) and np.all(w <= p.w_bar) and np.all(om <= p.omega_bar))


def in_Cho(st: State, p: Params, a1) -> bool:
    h, v = st.h(p), st.v(p)
    return h >= 0 and psi1(h, v, a1) >= 0


def in_D(st: State, p: Params, a1, a2) -> tuple[bool, dict]:
    """Membership in the open set D of Thm. 5 (all inequalities strict)."""
    h0, v0 = st.h(p), st.v(p)
    conds = {
        "h0 > 0": h0 > 0,
        "0 < v0 < alpha1(h0)": 0 < v0 < a1(h0),
        "q in int Sigma_y": in_Sigma_y(st, p, strict=True),
        "strict X_op": in_Xop(st, p, strict=True),
    }
    if conds["0 < v0 < alpha1(h0)"] and conds["q in int Sigma_y"]:
        conds["t_psi < tau"] = t_psi(h0, v0, a1) < tau(st, p)
    else:
        conds["t_psi < tau"] = False
    return all(conds.values()), conds
