"""Geometric attitude controller for the multi-quadrotor Drake plant (plan P2.S4, item B4).

Discrete-time (1 kHz) SO(3) tracking law of Lee, Leok and McClamroch (CDC 2010), simplified
with Omega_d = 0.  For each quadrotor i the commanded world-frame thrust vector u_i (from the
safety filter, zero-order held) defines the desired body z-axis b3_d = u_i/|u_i|; the desired
heading is fixed at ``yaw_ref``.  The controller publishes ``[f_i, tau_i]`` for
``actuation.ThrustTorqueApplier`` and the tracking error d_att,i = |f_i R_i e3 - u_i| that
Remark 18 charges to the disturbance budget.

Everything is computed inside a periodic discrete update and stored in discrete state; both
output ports are declared from that state, so the controller is not direct-feedthrough and
closing the loop through the plant creates no algebraic loop.  An initialization event makes the
first command available at t = 0 (periodic updates with offset 0 do not fire in Initialize()).
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import RigidBody
from pydrake.systems.framework import LeafSystem

from authority_barriers.simulator.actuation import floating_state_indices, quaternions_to_matrices


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cross product of (N, 3) arrays (np.cross is slow for tiny arrays)."""
    c = np.empty_like(a)
    c[:, 0] = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
    c[:, 1] = a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]
    c[:, 2] = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return c


class GeometricAttitudeController(LeafSystem):
    """Discrete geometric attitude controller (Lee-Leok-McClamroch, Omega_d = 0) for N quadrotors.

    Input ports
        ``plant_state``: plant state output (num_positions + num_velocities).
        ``thrust_cmd``: 3N vector of commanded world-frame thrust vectors u_i.
    Output ports (from discrete state, no direct feedthrough)
        ``actuation``: 4N vector, per quad ``[f_i, tau_i (3, body frame)]``.
        ``tracking_error``: N vector, d_att,i = |f_i R_i e3 - u_i| at the last update.

    Law (per quad, R = R_WB, Omega = body angular velocity = R^T w_WB_W):
        b3_d = u/|u| (current body z-axis and f = 0 if |u| < 1e-9),
        b1_ref = [cos yaw_ref, sin yaw_ref, 0], b2_d = normalize(b3_d x b1_ref), b1_d = b2_d x b3_d,
        R_d = [b1_d b2_d b3_d],  e_R = 1/2 vee(R_d^T R - R^T R_d),  e_Omega = Omega,
        tau = -K_R e_R - K_Omega e_Omega + Omega x (J Omega),  K_R = omega_n^2 J,  K_Omega = 2 zeta omega_n J,
        f = clip(u . R e3, 0, f_max).
    With these gains the linearised error dynamics are J (theta_dd + 2 zeta omega_n theta_d + omega_n^2 theta) = 0.

    Parameters
        plant: finalized MultibodyPlant containing ``quad_bodies`` as free bodies.
        J: (3, 3) inertia shared by all quads or (N, 3, 3) per-quad inertias (body frame).
        f_max: scalar or (N,) thrust limits.  dt: update period.  omega_n, zeta: loop natural
        frequency and damping.  yaw_ref: fixed heading reference (rad).
    """

    def __init__(self, plant: MultibodyPlant, quad_bodies: Sequence[RigidBody], J, f_max,
                 dt: float = 1e-3, omega_n: float = 100.0, zeta: float = 0.9, yaw_ref: float = 0.0):
        super().__init__()
        self.N = N = len(quad_bodies)
        self.dt, self.omega_n, self.zeta, self.yaw_ref = float(dt), float(omega_n), float(zeta), float(yaw_ref)
        J = np.asarray(J, dtype=float)
        if J.shape == (3, 3):
            J = np.broadcast_to(J, (N, 3, 3))
        if J.shape != (N, 3, 3):
            raise ValueError(f"J must have shape (3, 3) or ({N}, 3, 3), got {J.shape}")
        self.J = np.array(J)
        self.K_R = self.omega_n ** 2 * self.J
        self.K_Omega = 2.0 * self.zeta * self.omega_n * self.J
        self.f_max = np.array(np.broadcast_to(np.asarray(f_max, dtype=float), (N,)))
        b1 = np.array([np.cos(self.yaw_ref), np.sin(self.yaw_ref), 0.0])
        self._b1_ref = b1
        # b3_d x b1_ref = b3_d @ S with S = [b1_ref]_x (skew-symmetric), one matmul instead of a cross.
        self._S_b1 = np.array([[0.0, -b1[2], b1[1]], [b1[2], 0.0, -b1[0]], [-b1[1], b1[0], 0.0]])
        # 1/2 vee(M - M^T)_i = -1/2 eps_ijk M_jk: one einsum with the Levi-Civita tensor.
        eps = np.zeros((3, 3, 3))
        eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1.0
        eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1.0
        self._half_vee = -0.5 * eps
        self._q_idx, self._w_idx = floating_state_indices(plant, quad_bodies)
        nx = plant.num_positions() + plant.num_velocities()

        self._state_port = self.DeclareVectorInputPort("plant_state", nx)
        self._cmd_port = self.DeclareVectorInputPort("thrust_cmd", 3 * N)
        self._act_index = self.DeclareDiscreteState(4 * N)
        self._err_index = self.DeclareDiscreteState(N)
        self.DeclareStateOutputPort("actuation", self._act_index)
        self.DeclareStateOutputPort("tracking_error", self._err_index)
        self.DeclareInitializationDiscreteUpdateEvent(self._update)
        self.DeclarePeriodicDiscreteUpdateEvent(self.dt, 0.0, self._update)
        # Forced event: lets callers step the law by hand (CalcForcedDiscreteVariableUpdate) in tests.
        self.DeclareForcedDiscreteUpdateEvent(self._update)

    # ------------------------------------------------------------------ control law
    def compute(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Control law on raw arrays: plant state ``x`` and 3N command -> ((N, 4) [f, tau], (N,) d_att)."""
        u = np.asarray(u, dtype=float).reshape(self.N, 3)
        x = np.asarray(x, dtype=float)
        R = quaternions_to_matrices(x[self._q_idx])                     # (N, 3, 3) R_WB
        Rt = R.transpose(0, 2, 1)
        Omega = (Rt @ x[self._w_idx][:, :, None])[:, :, 0]              # body angular velocity R^T w
        b3 = R[:, :, 2]                                                 # current body z-axis (world)

        unorm = np.sqrt((u * u).sum(1))
        small = unorm < 1e-9
        b3_d = np.where(small[:, None], b3, u / np.where(small, 1.0, unorm)[:, None])

        b2_d = b3_d @ self._S_b1                                        # b3_d x b1_ref
        n2 = np.sqrt((b2_d * b2_d).sum(1))
        if n2.min() < 1e-6:                                             # thrust along the heading axis
            degenerate = n2 < 1e-6
            b2_d[degenerate] = R[degenerate][:, :, 1]
            n2 = np.sqrt((b2_d * b2_d).sum(1))
        b2_d /= n2[:, None]
        R_d = np.empty_like(R)                                          # columns b1_d, b2_d, b3_d
        R_d[:, :, 0] = _cross(b2_d, b3_d)
        R_d[:, :, 1] = b2_d
        R_d[:, :, 2] = b3_d

        M = R_d.transpose(0, 2, 1) @ R                                  # R_d^T R
        e_R = np.einsum("ijk,njk->ni", self._half_vee, M)               # 1/2 vee(M - M^T)
        J_Omega = (self.J @ Omega[:, :, None])[:, :, 0]
        tau = (-(self.K_R @ e_R[:, :, None])[:, :, 0]
               - (self.K_Omega @ Omega[:, :, None])[:, :, 0]
               + _cross(Omega, J_Omega))

        f = np.minimum(np.maximum((u * b3).sum(1), 0.0), self.f_max)
        f[small] = 0.0
        err = f[:, None] * b3 - u
        d_att = np.sqrt((err * err).sum(1))
        act = np.empty((self.N, 4))
        act[:, 0] = f
        act[:, 1:] = tau
        return act, d_att

    # ------------------------------------------------------------------ Drake plumbing
    def _update(self, context, discrete_state):
        act, d_att = self.compute(self._state_port.Eval(context), self._cmd_port.Eval(context))
        discrete_state.set_value(int(self._act_index), act.ravel())
        discrete_state.set_value(int(self._err_index), d_att)
