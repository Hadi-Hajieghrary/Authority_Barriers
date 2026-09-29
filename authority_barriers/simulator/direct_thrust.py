"""Perfect actuator: applies commanded thrust VECTORS directly as world-frame forces at the
quadrotor CoMs (no attitude dynamics). Used only for the plant-fidelity test (M2.1) and as a
diagnostic; the campaign runs through the attitude loop."""
from __future__ import annotations

import numpy as np
from pydrake.common.value import AbstractValue
from pydrake.multibody.math import SpatialForce
from pydrake.multibody.plant import ExternallyAppliedSpatialForce
from pydrake.systems.framework import LeafSystem

from .plant import PlantInfo


class DirectThrustApplier(LeafSystem):
    def __init__(self, info: PlantInfo, f_max):
        super().__init__()
        self.bodies = [b.index() for b in info.quads]
        self.f_max = np.asarray(f_max, float)
        self.N = len(self.bodies)
        self.DeclareVectorInputPort("thrust_cmd", 3 * self.N)
        self.DeclareAbstractOutputPort("spatial_forces", lambda: AbstractValue.Make([ExternallyAppliedSpatialForce()]), self._calc)

    def _calc(self, context, output):
        u = self.get_input_port(0).Eval(context).reshape(self.N, 3)
        n = np.linalg.norm(u, axis=1)
        over = n > self.f_max
        u = u.copy()
        u[over] *= (self.f_max[over] / n[over])[:, None]
        out = []
        for bi, f in zip(self.bodies, u):
            e = ExternallyAppliedSpatialForce(); e.body_index = bi; e.p_BoBq_B = np.zeros(3)
            e.F_Bq_W = SpatialForce(np.zeros(3), f)
            out.append(e)
        output.set_value(out)
