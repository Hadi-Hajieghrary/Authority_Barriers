"""StateProjector: plant state vector -> packed taut-cable state (Sec. II-C coordinates), so that
the filter and the recorder see the same quantities the theory is written in."""
from __future__ import annotations

import numpy as np
from pydrake.systems.framework import LeafSystem

from authority_barriers.theory.params import Params
from authority_barriers.theory.taut_model import n_state, pack
from .plant import PlantInfo, read_state


class StateProjector(LeafSystem):
    def __init__(self, info: PlantInfo, p: Params):
        super().__init__()
        self.info, self.p = info, p
        self.DeclareVectorInputPort("plant_state", info.nq + info.nv)
        self.DeclareVectorOutputPort("taut_state", n_state(p.N), self._calc)

    def _calc(self, context, output):
        x = self.get_input_port(0).Eval(context)
        output.SetFromVector(pack(read_state(self.info, x, self.p)))
