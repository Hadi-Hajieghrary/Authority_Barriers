"""Unilateral spring-damper cables (Sec. VI): T_i = max(0, k e_i + c edot_i) when the cable is
stretched (e_i = |p_i - x_L| - l_i > 0) and zero when slack; force -T_i q_i on quadrotor i and
+T_i q_i on the payload. Vectorized over cables: this output is evaluated at every derivative
evaluation of the implicit integrator."""
from __future__ import annotations

import numpy as np
from pydrake.common.value import AbstractValue
from pydrake.multibody.math import SpatialForce
from pydrake.multibody.plant import ExternallyAppliedSpatialForce
from pydrake.systems.framework import LeafSystem

from authority_barriers.theory.params import Params
from .plant import PlantInfo


def cable_constants(p: Params, stretch_frac: float = 1e-3, zeta: float = 0.4):
    """Stiffness so that the stretch under f_max is <= stretch_frac * l, and damping ratio zeta
    against the reduced mass m_i m_L/(m_i + m_L) (D-17)."""
    k = float(np.max(p.f_max_arr / (stretch_frac * p.l_arr)))
    k = float(np.ceil(k / 1e4) * 1e4)
    m_eff = p.m_arr * p.m_L / (p.m_arr + p.m_L)
    c = float(np.mean(2.0 * zeta * np.sqrt(k * m_eff)))
    return k, c


class CableForces(LeafSystem):
    def __init__(self, info: PlantInfo, p: Params, k: float | None = None, c: float | None = None):
        super().__init__()
        self.info, self.p = info, p
        if k is None or c is None:
            k0, c0 = cable_constants(p)
            k = k0 if k is None else k
            c = c0 if c is None else c
        self.k, self.c = float(k), float(c)
        self.N = p.N
        self.l = p.l_arr.copy()
        self.qpos = np.array([info.q_start[b.name()] + 4 for b in info.quads])
        self.qvel = np.array([info.nq + info.v_start[b.name()] + 3 for b in info.quads])
        self.ppos = info.q_start["payload"] + 4
        self.pvel = info.nq + info.v_start["payload"] + 3
        self.payload_index = info.payload.index()
        self.quad_indices = [b.index() for b in info.quads]
        self.DeclareVectorInputPort("plant_state", info.nq + info.nv)
        self.DeclareAbstractOutputPort("spatial_forces", lambda: AbstractValue.Make([ExternallyAppliedSpatialForce()]),
                                       self._calc_forces)
        # [T_i (N), stretch_i (N), a_meas (3)]: tensions, cable stretch, payload specific force from the cable forces
        self.DeclareVectorOutputPort("cable_data", 2 * self.N + 3, self._calc_data)

    def _kin(self, x):
        idx = self.qpos[:, None] + np.arange(3)[None, :]
        P = x[idx]
        V = x[self.qvel[:, None] + np.arange(3)[None, :]]
        xL = x[self.ppos:self.ppos + 3]
        vL = x[self.pvel:self.pvel + 3]
        d = P - xL[None, :]
        L = np.sqrt(np.einsum("ij,ij->i", d, d))
        qhat = d / L[:, None]
        e = L - self.l
        edot = np.einsum("ij,ij->i", qhat, V - vL[None, :])
        T = np.where(e > 0, np.maximum(0.0, self.k * e + self.c * edot), 0.0)
        return qhat, e, T

    def tensions(self, x):
        return self._kin(x)

    def _calc_forces(self, context, output):
        x = self.get_input_port(0).Eval(context)
        qhat, _, T = self._kin(x)
        forces = []
        F = T[:, None] * qhat
        for i in range(self.N):
            f = ExternallyAppliedSpatialForce()
            f.body_index = self.quad_indices[i]
            f.p_BoBq_B = np.zeros(3)
            f.F_Bq_W = SpatialForce(np.zeros(3), -F[i])
            forces.append(f)
        fp = ExternallyAppliedSpatialForce()
        fp.body_index = self.payload_index
        fp.p_BoBq_B = np.zeros(3)
        fp.F_Bq_W = SpatialForce(np.zeros(3), F.sum(0))
        forces.append(fp)
        output.set_value(forces)

    def _calc_data(self, context, output):
        x = self.get_input_port(0).Eval(context)
        qhat, e, T = self._kin(x)
        a_meas = (T @ qhat) / self.p.m_L
        output.SetFromVector(np.concatenate([T, e, a_meas]))
