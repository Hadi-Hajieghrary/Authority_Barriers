"""Drake LeafSystem of the taut-cable model (Lemma 1, tension path: eq. load -> eq. tension ->
eq. cable) with the Gram parametrization of the tangent-plane thrust (eq. gram), scalar-convertible
so that pydrake.planning.DirectCollocation can differentiate it (P4.S4, E3 collocation).

Continuous state (6 + 6N) in the taut_model layout x = [x_L, xdot_L, q (3N), qdot (3N)].
One input port of size 3N ordered [T_1, ..., T_N, c_11, c_12, ..., c_N1, c_N2] with
    u_i^perp = c_i1 P_i y + c_i2 P_i r,   P_i y = y - z_i q_i,   P_i r = r - w_i q_i,
z_i = q_i . y, w_i = q_i . r (App. C, eq. gram; smooth in q). The thrust vector is
u_i = s_i q_i + u_i^perp with s_i = T_i + m_i q_i . a - m_i l_i ||qdot_i||^2, a = sum_j T_j q_j / m_L
(taut_model.rhs_from_tensions). Every function here is dtype-agnostic (float arrays and object
arrays of pydrake AutoDiffXd): only +, -, *, /, reductions and matmul, no np.linalg.
"""
from __future__ import annotations

import numpy as np
from pydrake.systems.framework import LeafSystem_
from pydrake.systems.scalar_conversion import TemplateSystem

from authority_barriers.theory import taut_model as tm
from authority_barriers.theory.params import Params
from authority_barriers.theory.state import State


# ---------------------------------------------------------------- input layout
def n_input(N: int) -> int:
    """Size 3N of the input port."""
    return 3 * N


def split_input(x_in, N: int):
    """Input vector (3N,) -> tensions T (N,) and Gram coefficients c (N, 2)."""
    x_in = np.asarray(x_in).reshape(-1)
    if x_in.shape[0] != 3 * N:
        raise ValueError(f"input has length {x_in.shape[0]}, expected {3 * N} for N = {N}")
    return x_in[:N], x_in[N:].reshape(N, 2)


def join_input(T, c) -> np.ndarray:
    """(T (N,), c (N, 2)) -> input vector (3N,)."""
    return np.concatenate([np.asarray(T).reshape(-1), np.asarray(c).reshape(-1)])


def tangent_basis(q: np.ndarray, p: Params):
    """P_i y = y - z_i q_i and P_i r = r - w_i q_i for every cable, (N, 3) each (dtype-agnostic)."""
    y, r = p.y_vec, p.r_vec
    z = q @ y
    w = q @ r
    return y[None, :] - z[:, None] * q, r[None, :] - w[:, None] * q


def perp_from_coeffs(st: State, p: Params, c) -> np.ndarray:
    """u_i^perp = c_i1 P_i y + c_i2 P_i r, (N, 3); tangent to q_i whenever ||q_i|| = 1."""
    Py, Pr = tangent_basis(st.q, p)
    c = np.asarray(c).reshape(p.N, 2)
    return c[:, 0:1] * Py + c[:, 1:2] * Pr


def coeffs_from_perp(st: State, p: Params, u_perp) -> np.ndarray:
    """Invert eq. gram per cable: (y . u_perp, r . u_perp) = G_i (c_i1, c_i2) with
    G_i = [[1 - z^2, -z w], [-z w, 1 - w^2]], det G_i = 1 - z^2 - w^2 = varrho_i^2 > 0 on X_op.
    Returns c (N, 2); exact when u_perp is tangent to q_i."""
    y, r = p.y_vec, p.r_vec
    u_perp = np.asarray(u_perp).reshape(p.N, 3)
    z, w = st.q @ y, st.q @ r
    eta_z, eta_w = u_perp @ y, u_perp @ r
    det = 1.0 - z ** 2 - w ** 2
    c1 = ((1.0 - w ** 2) * eta_z + z * w * eta_w) / det
    c2 = (z * w * eta_z + (1.0 - z ** 2) * eta_w) / det
    return np.stack([c1, c2], axis=1)


# ---------------------------------------------------------------- dynamics and conversions
def dynamics(x, x_in, p: Params) -> np.ndarray:
    """Time derivative of the packed state under the input [T, c] (dtype-agnostic)."""
    st = tm.unpack(x, p.N)
    T, c = split_input(x_in, p.N)
    out = tm.rhs_from_tensions(st, T, perp_from_coeffs(st, p, c), p)
    return tm._pack(st.v_L, out["xdd"], st.qd, out["qdd"])


def thrust_from_input(st: State, p: Params, x_in) -> np.ndarray:
    """Thrust vectors u (N, 3) that the input [T, c] commands at the state st (eq. tension inverted)."""
    T, c = split_input(x_in, p.N)
    return tm.rhs_from_tensions(st, T, perp_from_coeffs(st, p, c), p)["u"]


def input_from_tensions(st: State, p: Params, T, u_perp) -> np.ndarray:
    """(T, u_perp) -> input vector [T, c] (3N,) by the 2x2 Gram solve per cable."""
    return join_input(T, coeffs_from_perp(st, p, u_perp))


def input_from_plan(st: State, p: Params, plan) -> np.ndarray:
    """maneuver.Plan (T, u_perp) -> input vector [T, c] (3N,)."""
    return input_from_tensions(st, p, plan.T, plan.u_perp)


def input_from_thrust(st: State, p: Params, u) -> np.ndarray:
    """Thrust vectors u (N, 3) -> input vector [T, c] through eqs. closure and tension
    (taut_model.rhs_from_thrust); thrust_from_input inverts it."""
    out = tm.rhs_from_thrust(st, np.asarray(u, float).reshape(p.N, 3), p)
    return input_from_tensions(st, p, out["T"], out["u_perp"])


# ---------------------------------------------------------------- the Drake system
@TemplateSystem.define("TautCableSystem_")
def TautCableSystem_(T):
    class Impl(LeafSystem_[T]):
        """Taut-cable team as a continuous-time LeafSystem; input [T, c], state output port."""

        def _construct(self, p: Params, converter=None):
            LeafSystem_[T].__init__(self, converter)
            self.p = p
            self.N = p.N
            self.DeclareVectorInputPort("tension_and_gram_coeffs", n_input(p.N))
            state_index = self.DeclareContinuousState(tm.n_state(p.N))
            self.DeclareStateOutputPort("state", state_index)

        def _construct_copy(self, other, converter=None):
            Impl._construct(self, other.p, converter=converter)

        def DoCalcTimeDerivatives(self, context, derivatives):
            x = context.get_continuous_state_vector().CopyToVector()
            x_in = self.get_input_port(0).Eval(context)
            derivatives.get_mutable_vector().SetFromVector(dynamics(x, x_in, self.p))

    return Impl


TautCableSystem = TautCableSystem_[None]
