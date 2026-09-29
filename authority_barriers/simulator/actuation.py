"""Thrust/torque actuation for the multi-quadrotor Drake plant (plan P2.S3, item B3).

``ThrustTorqueApplier`` turns per-quadrotor commands ``[f_i, tau_i]`` (thrust magnitude along
the body z-axis and a body-frame torque) into ``ExternallyAppliedSpatialForce``s at the centres
of mass, expressed in the world frame as ``MultibodyPlant.get_applied_spatial_force_input_port()``
expects.  The thrust magnitude is clipped to the actuator limit ``[0, f_max_i]``.

The module also holds the small state-indexing / quaternion helpers shared with ``attitude.py``.
The plant's floating-base layout (verified on pydrake 1.51.1) is, per free body,
``q = [qw, qx, qy, qz, x, y, z]`` from ``body.floating_positions_start()`` and
``v = [w_WB_W (3), v_WB_W (3)]`` from ``body.floating_velocities_start_in_v()`` inside the
velocity block; the angular velocity is expressed in the WORLD frame.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from pydrake.common.eigen_geometry import Quaternion
from pydrake.common.value import AbstractValue
from pydrake.math import RotationMatrix
from pydrake.multibody.math import SpatialForce
from pydrake.multibody.plant import ExternallyAppliedSpatialForce, MultibodyPlant
from pydrake.multibody.tree import RigidBody
from pydrake.systems.framework import LeafSystem


def quaternion_to_rotation(qw: float, qx: float, qy: float, qz: float) -> RotationMatrix:
    """Rotation matrix R_WB from Drake's (w, x, y, z) floating-body quaternion.

    The quaternion is normalised first: during continuous integration the plant's quaternion
    drifts slightly from unit norm (the plant itself normalises when it builds R).
    """
    q = np.array([qw, qx, qy, qz], dtype=float)
    return RotationMatrix(Quaternion(q / np.linalg.norm(q)))


def quaternions_to_matrices(q: np.ndarray) -> np.ndarray:
    """Vectorised (N, 4) array of (w, x, y, z) quaternions -> (N, 3, 3) rotation matrices R_WB.

    Any nonzero norm is accepted (the result is the rotation of the normalised quaternion):
    R = ((w^2 - |v|^2) I + 2 v v^T + 2 w [v]_x) / |q|^2.
    """
    q = np.asarray(q, dtype=float).reshape(-1, 4)
    w, v = q[:, 0], q[:, 1:]
    vv = v[:, :, None] * v[:, None, :]                       # v v^T
    vx = v[:, _SKEW_INDEX] * _SKEW_SIGN                      # [v]_x
    diag = w * w - (vv[:, 0, 0] + vv[:, 1, 1] + vv[:, 2, 2])
    R = 2.0 * (vv + w[:, None, None] * vx)
    R[:, _DIAG, _DIAG] += diag[:, None]
    R /= (q * q).sum(1)[:, None, None]
    return R


_SKEW_INDEX = np.array([[0, 2, 1], [2, 1, 0], [1, 0, 2]])   # [v]_x = v[_SKEW_INDEX] * _SKEW_SIGN
_SKEW_SIGN = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
_DIAG = np.arange(3)


def floating_state_indices(plant: MultibodyPlant, bodies: Sequence[RigidBody]) -> Tuple[np.ndarray, np.ndarray]:
    """Index arrays into the plant state vector x = [q; v] for a list of free (quaternion) bodies.

    Returns ``(q_idx, w_idx)``: ``x[q_idx]`` is the (N, 4) array of quaternions (w, x, y, z) and
    ``x[w_idx]`` the (N, 3) array of world-frame angular velocities w_WB_W.
    """
    if not plant.is_finalized():
        raise ValueError("plant must be finalized before its floating-base state layout is known")
    for b in bodies:
        if not (b.is_floating_base_body() and b.has_quaternion_dofs()):
            raise ValueError(f"body '{b.name()}' is not a free body with quaternion coordinates")
    nq = plant.num_positions()
    q_idx = np.array([b.floating_positions_start() + np.arange(4) for b in bodies], dtype=int).reshape(-1, 4)
    w_idx = np.array([nq + b.floating_velocities_start_in_v() + np.arange(3) for b in bodies], dtype=int).reshape(-1, 3)
    return q_idx, w_idx


class ThrustTorqueApplier(LeafSystem):
    """Applies f_i R_i e3 (world) at each quadrotor's CoM plus the world-frame torque R_i tau_i.

    Input ports
        ``plant_state``: the plant's state output (num_positions + num_velocities).
        ``actuation``: 4N vector, per quad ``[f_i, tau_i (3, body frame)]``.
    Output port
        ``spatial_forces``: ``List[ExternallyAppliedSpatialForce]`` for
        ``plant.get_applied_spatial_force_input_port()`` (directly or through an
        ``ExternallyAppliedSpatialForceMultiplexer``).

    The output is direct-feedthrough from both inputs (it is evaluated at every derivative
    evaluation), which is loop-free because the plant's state port and the attitude controller's
    state-only outputs feed it.
    """

    def __init__(self, plant: MultibodyPlant, quad_bodies: Sequence[RigidBody], f_max):
        super().__init__()
        self.N = N = len(quad_bodies)
        self.f_max = np.array(np.broadcast_to(np.asarray(f_max, dtype=float), (N,)))
        self._q_idx, _ = floating_state_indices(plant, quad_bodies)
        nx = plant.num_positions() + plant.num_velocities()
        self._forces = []
        for b in quad_bodies:
            F = ExternallyAppliedSpatialForce()
            F.body_index = b.index()
            F.p_BoBq_B = np.zeros(3)  # the body origin is the CoM
            self._forces.append(F)
        self._state_port = self.DeclareVectorInputPort("plant_state", nx)
        self._cmd_port = self.DeclareVectorInputPort("actuation", 4 * N)
        self.DeclareAbstractOutputPort(
            "spatial_forces",
            lambda: AbstractValue.Make([ExternallyAppliedSpatialForce() for _ in range(N)]),
            self._calc_forces,
            prerequisites_of_calc={self.all_input_ports_ticket()},
        )

    def wrenches(self, x: np.ndarray, actuation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """World-frame (forces (N,3), torques (N,3)) for plant state ``x`` and 4N command vector."""
        cmd = np.asarray(actuation, dtype=float).reshape(self.N, 4)
        R = quaternions_to_matrices(np.asarray(x)[self._q_idx])
        f = np.clip(cmd[:, 0], 0.0, self.f_max)
        F_W = f[:, None] * R[:, :, 2]
        tau_W = np.einsum("nij,nj->ni", R, cmd[:, 1:])
        return F_W, tau_W

    def _calc_forces(self, context, output):
        F_W, tau_W = self.wrenches(self._state_port.Eval(context), self._cmd_port.Eval(context))
        for i, F in enumerate(self._forces):
            F.F_Bq_W = SpatialForce(tau_W[i], F_W[i])
        output.set_value(self._forces)
