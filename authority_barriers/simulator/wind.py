"""Bounded wind (D-16): first-order filtered Gaussian gusts, clipped in norm at d_bar (or a
multiple of it), sampled at 100 Hz and held, applied to the payload (d_L) and to each quadrotor
(d_i) through the applied-spatial-force port."""
from __future__ import annotations

import numpy as np
from pydrake.common.value import AbstractValue
from pydrake.multibody.math import SpatialForce
from pydrake.multibody.plant import ExternallyAppliedSpatialForce
from pydrake.systems.framework import LeafSystem

from authority_barriers.theory.params import Params
from .plant import PlantInfo


class WindForces(LeafSystem):
    def __init__(self, info: PlantInfo, p: Params, seed: int = 0, scale: float = 1.0, tau_w: float = 0.5,
                 dt: float = 0.01, sigma_frac: float = 1.0):
        """scale = 1 clips at d_bar (Rem. 18 budget), scale = 2 at 2 d_bar (E7, deliberate violation).
        The driving noise has standard deviation sigma_frac * d_bar so that the clip is active often."""
        super().__init__()
        self.info, self.p = info, p
        self.bounds = np.concatenate([[p.d_bar_L * scale], np.full(p.N, p.d_bar_i * scale)])
        self.sigma = sigma_frac * self.bounds
        self.alpha = np.exp(-dt / tau_w)
        self.rng = np.random.default_rng(seed)
        self.n = p.N + 1
        self.bodies = [info.payload.index()] + [b.index() for b in info.quads]
        self.state_index = self.DeclareDiscreteState(3 * self.n)
        self.DeclarePeriodicDiscreteUpdateEvent(dt, 0.0, self._update)
        self.DeclareAbstractOutputPort("spatial_forces", lambda: AbstractValue.Make([ExternallyAppliedSpatialForce()]),
                                       self._calc, prerequisites_of_calc={self.all_state_ticket()})

    def _update(self, context, discrete_state):
        x = context.get_discrete_state(self.state_index).get_value().reshape(self.n, 3)
        noise = self.rng.normal(size=(self.n, 3)) * (self.sigma[:, None] * np.sqrt(1.0 - self.alpha ** 2))
        x = self.alpha * x + noise
        nrm = np.linalg.norm(x, axis=1)
        over = nrm > self.bounds
        x[over] *= (self.bounds[over] / nrm[over])[:, None]
        discrete_state.get_mutable_vector(self.state_index).SetFromVector(x.reshape(-1))

    def _calc(self, context, output):
        x = context.get_discrete_state(self.state_index).get_value().reshape(self.n, 3)
        forces = []
        for k, bi in enumerate(self.bodies):
            f = ExternallyAppliedSpatialForce()
            f.body_index = bi
            f.p_BoBq_B = np.zeros(3)
            f.F_Bq_W = SpatialForce(np.zeros(3), x[k])
            forces.append(f)
        output.set_value(forces)
