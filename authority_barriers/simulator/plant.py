"""Full-order plant of Sec. VI: N free rigid quadrotors and a rigid, point-like payload in a
continuous MultibodyPlant (time_step = 0). Cables, thrust, torque and wind are applied through
the plant's applied-spatial-force input port by separate systems."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pydrake.geometry import Box, Cylinder, Rgba, Sphere
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.math import SpatialVelocity
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.multibody.tree import SpatialInertia, UnitInertia

from authority_barriers.theory.params import E3, Params
from authority_barriers.theory.state import State

J_QUAD = np.diag([0.02, 0.02, 0.04])     # D-17 [kg m^2]
PAYLOAD_RADIUS = 0.05


@dataclass
class PlantInfo:
    plant: MultibodyPlant
    scene_graph: object
    payload: object
    quads: list
    nq: int
    nv: int
    q_start: dict = field(default_factory=dict)   # body name -> index of the quaternion in q
    v_start: dict = field(default_factory=dict)   # body name -> index of the angular velocity in v
    J: np.ndarray = field(default_factory=lambda: J_QUAD.copy())

    def pos(self, x: np.ndarray, name: str) -> np.ndarray:
        s = self.q_start[name]
        return x[s + 4:s + 7]

    def quat(self, x: np.ndarray, name: str) -> np.ndarray:
        s = self.q_start[name]
        return x[s:s + 4]

    def lin_vel(self, x: np.ndarray, name: str) -> np.ndarray:
        s = self.nq + self.v_start[name]
        return x[s + 3:s + 6]

    def ang_vel_world(self, x: np.ndarray, name: str) -> np.ndarray:
        s = self.nq + self.v_start[name]
        return x[s:s + 3]


QUAD_COLORS = [(0.16, 0.47, 0.84), (0.92, 0.41, 0.20), (0.11, 0.69, 0.48), (0.93, 0.63, 0.0), (0.44, 0.44, 0.43), (0.56, 0.27, 0.68)]


def build_plant(p: Params, builder, visualize: bool = False, J: np.ndarray | None = None, detail: bool = False,
                wall: bool = False) -> PlantInfo:
    """detail: quadrotor bodies rendered as a cross of arms with four rotor discs (one color per vehicle) instead of a
    plain box; wall: the constraint plane n^T x = d0 rendered as a large translucent slab on the world body."""
    J = J_QUAD if J is None else np.asarray(J, float)
    if visualize:
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    else:
        plant, scene_graph = builder.AddSystem(MultibodyPlant(time_step=0.0)), None
    payload = plant.AddRigidBody("payload", SpatialInertia(p.m_L, np.zeros(3), UnitInertia.SolidSphere(PAYLOAD_RADIUS)))
    quads = []
    for i in range(p.N):
        m = p.m_arr[i]
        G = UnitInertia(J[0, 0] / m, J[1, 1] / m, J[2, 2] / m)
        quads.append(plant.AddRigidBody(f"quad{i}", SpatialInertia(m, np.zeros(3), G)))
    if visualize:
        plant.RegisterVisualGeometry(payload, RigidTransform(), Sphere(PAYLOAD_RADIUS * 1.6), "payload_vis", np.array([0.1, 0.1, 0.1, 1.0]))
        for i, b in enumerate(quads):
            if not detail:
                plant.RegisterVisualGeometry(b, RigidTransform(), Box(0.3, 0.3, 0.05), f"quad{i}_vis", np.array([0.2, 0.4, 0.8, 1.0]))
                continue
            c = QUAD_COLORS[i % len(QUAD_COLORS)]
            body_rgba, rotor_rgba = np.array([*c, 1.0]), np.array([*c, 0.55])
            plant.RegisterVisualGeometry(b, RigidTransform(), Box(0.08, 0.08, 0.04), f"quad{i}_hub", body_rgba)
            for j, ang in enumerate((np.pi / 4, 3 * np.pi / 4)):                    # two arms as a cross
                R = RotationMatrix.MakeZRotation(ang)
                plant.RegisterVisualGeometry(b, RigidTransform(R, [0, 0, 0]), Box(0.3, 0.02, 0.015), f"quad{i}_arm{j}", body_rgba)
            for j, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):        # rotor discs at the arm tips
                pos = [sx * 0.15 * np.cos(np.pi / 4), sy * 0.15 * np.sin(np.pi / 4), 0.02]
                plant.RegisterVisualGeometry(b, RigidTransform(pos), Cylinder(0.06, 0.006), f"quad{i}_rotor{j}", rotor_rgba)
    if visualize and wall:
        n = p.n_vec
        R = RotationMatrix.MakeFromOneVector(n, 0)                                  # box x-axis along the wall normal
        X = RigidTransform(R, p.d0 * n + 15.0 * E3 + 0.05 * n)
        plant.RegisterVisualGeometry(plant.world_body(), X, Box(0.1, 40.0, 40.0), "wall", np.array([0.55, 0.55, 0.52, 0.45]))
    plant.Finalize()
    info = PlantInfo(plant, scene_graph, payload, quads, plant.num_positions(), plant.num_velocities(), J=J)
    for b in [payload] + quads:
        info.q_start[b.name()] = b.floating_positions_start()
        info.v_start[b.name()] = b.floating_velocities_start_in_v()
    return info


def _set_velocity(plant, context, body, V):
    try:
        plant.SetFreeBodySpatialVelocity(context, body, V)
    except TypeError:
        plant.SetFreeBodySpatialVelocity(body, V, context)


def set_initial_state(info: PlantInfo, plant_context, st: State, p: Params, attitudes=None,
                      prestretch: np.ndarray | None = None, k: float | None = None) -> None:
    """Place the payload at x_L and quadrotor i at x_L + (l_i + e_i) q_i with velocities consistent
    with taut cables (pdot_i = xdot_L + (l_i + e_i) qdot_i); attitudes default to identity, angular
    rates zero. `prestretch` gives the initial tensions T_i whose spring stretch e_i = T_i / k is
    applied so that the spring-damper cables start at their taut-model tension instead of zero."""
    plant = info.plant
    plant.SetFreeBodyPose(plant_context, info.payload, RigidTransform(st.x_L))
    _set_velocity(plant, plant_context, info.payload, SpatialVelocity(np.zeros(3), st.v_L))
    e = np.zeros(p.N) if prestretch is None or k is None else np.asarray(prestretch, float) / k
    L = p.l_arr + e
    P = st.x_L + L[:, None] * st.q
    V = st.v_L + L[:, None] * st.qd
    for i, b in enumerate(info.quads):
        R = RotationMatrix() if attitudes is None else RotationMatrix(attitudes[i])
        plant.SetFreeBodyPose(plant_context, b, RigidTransform(R, P[i]))
        _set_velocity(plant, plant_context, b, SpatialVelocity(np.zeros(3), V[i]))


def read_state(info: PlantInfo, x: np.ndarray, p: Params) -> State:
    """Taut-cable state from the plant state vector: q_i = (p_i - x_L)/|p_i - x_L|,
    qdot_i = P_i (pdot_i - xdot_L)/|p_i - x_L| (the tangent part; radial motion is cable stretch)."""
    x_L = info.pos(x, "payload")
    v_L = info.lin_vel(x, "payload")
    q = np.zeros((p.N, 3)); qd = np.zeros((p.N, 3))
    for i, b in enumerate(info.quads):
        d = info.pos(x, b.name()) - x_L
        L = np.linalg.norm(d)
        q[i] = d / L
        rel = info.lin_vel(x, b.name()) - v_L
        qd[i] = (rel - (rel @ q[i]) * q[i]) / L
    return State(x_L.copy(), v_L.copy(), q, qd)


def cable_lengths(info: PlantInfo, x: np.ndarray) -> np.ndarray:
    x_L = info.pos(x, "payload")
    return np.array([np.linalg.norm(info.pos(x, b.name()) - x_L) for b in info.quads])
