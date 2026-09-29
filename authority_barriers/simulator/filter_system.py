"""FilterSystem: the 200 Hz sampled safety filter (Sec. V, Rem. 19) as a discrete LeafSystem.
Reads the projected taut-cable state and the cable data, evaluates the nominal controller and
the filter, stores the thrust commands (zero-order hold) and a diagnostics vector in discrete
state, and exposes both from state so that no algebraic loop is formed."""
from __future__ import annotations

import numpy as np
from pydrake.systems.framework import LeafSystem

from authority_barriers.theory import hocbf as HB
from authority_barriers.theory import profile as PF
from authority_barriers.theory import stopping as SD
from authority_barriers.theory.authority import BarrierData
from authority_barriers.theory.filters import HOCBFFilter, ProposedFilter
from authority_barriers.theory.params import Params
from authority_barriers.theory.taut_model import n_state, unpack

DIAG_FIELDS = ["t", "h", "v", "H", "D", "D_rel", "D_rob", "feasible", "relaxed", "slack", "solve_time", "b",
               "du_norm", "utilization", "minT_cable", "max_omega", "n_rows", "in_V", "clamped", "t_star",
               "beta", "mu", "psi1", "in_Sigma_y", "min_z", "a_meas_norm", "alpha_y", "H_rob", "hdot0",
               "unom_norm", "lambda", "a_hat_err"]          # appended 2026-09-25 (optional track); older logs have 29 fields


class FilterSystem(LeafSystem):
    def __init__(self, p: Params, filt, nominal, dt: float | None = None, log_mu: bool = False, a1=None, a2=None,
                 rel_data: BarrierData | None = None, rob_data: BarrierData | None = None):
        super().__init__()
        self.p, self.filt, self.nominal = p, filt, nominal
        self.dt = p.dt_filter if dt is None else dt
        self.log_mu, self.a1, self.a2 = log_mu, a1, a2
        self.data = getattr(filt, "data", None) or BarrierData.nominal(p)      # baselines without barrier data (unfiltered, HOCBF)
        self.rel = rel_data or BarrierData.relaxed(p)
        try:
            self.rob = rob_data or BarrierData.robust(p, eps=0.0)
        except ValueError:
            self.rob = None
        N = p.N
        self.DeclareVectorInputPort("taut_state", n_state(N))
        self.DeclareVectorInputPort("cable_data", 2 * N + 3)
        self.cmd_index = self.DeclareDiscreteState(3 * N)
        self.diag_index = self.DeclareDiscreteState(len(DIAG_FIELDS))
        self.state_index = self.DeclareDiscreteState(n_state(N))
        self.DeclarePeriodicDiscreteUpdateEvent(self.dt, 0.0, self._update)
        self.DeclareInitializationDiscreteUpdateEvent(self._update)
        self.DeclareStateOutputPort("thrust_cmd", self.cmd_index)
        self.DeclareStateOutputPort("diag", self.diag_index)
        self.DeclareStateOutputPort("taut_at_tick", self.state_index)     # the state the command was computed for

    def _update(self, context, discrete_state):
        p = self.p
        t = context.get_time()
        xs = self.get_input_port(0).Eval(context)
        st = unpack(xs, p.N)
        cab = self.get_input_port(1).Eval(context)
        T_cable, a_meas = cab[:p.N], cab[2 * p.N:2 * p.N + 3]
        u_nom = self.nominal(t, st)
        res = self.filt.solve(st, u_nom)
        h, v, z, zd = st.h(p), st.v(p), st.z(p), st.zd(p)
        D = res.D if np.isfinite(getattr(res, "D", np.nan)) else SD.D_of(v, z, zd, self.data)
        D_rel = SD.D_of(v, z, zd, self.rel)
        D_rob = SD.D_of(v, z, zd, self.rob) if self.rob is not None else np.nan
        beta = mu = psi1 = np.nan
        if self.a1 is not None:
            beta = HB.beta(st, p, self.a1, self.a2)
            psi1 = HB.psi1(h, v, self.a1)
            if self.log_mu:
                mu = HB.alpha_exact(p, st)[0] - beta
        diag = np.array([
            t, h, v, h - D, D, D_rel, D_rob, float(res.feasible), float(res.relaxed), res.slack, res.solve_time, res.b,
            float(np.linalg.norm(res.u - u_nom)), float(np.max(np.linalg.norm(res.u, axis=1) / p.f_max_arr)),
            float(np.min(T_cable)), float(np.max(np.sqrt(st.omega2()))), float(res.rows.get("n_barrier_rows", 0)),
            float(res.in_V), float(res.clamped), float(res.t_star[0]) if getattr(res, "t_star", np.zeros(0)).size else np.nan,
            beta, mu, psi1, float(np.all(z <= 0)), float(np.min(z)), float(np.linalg.norm(a_meas)),
            self.data.alpha(z), h - D_rob,
            float(res.hdot[0]) if getattr(res, "hdot", np.zeros(0)).size else np.nan,
            float(np.linalg.norm(u_nom)), float(res.rows.get("lambda", np.nan)), float(res.rows.get("a_hat_err", np.nan)),
        ])
        discrete_state.get_mutable_vector(self.cmd_index).SetFromVector(res.u.reshape(-1))
        discrete_state.get_mutable_vector(self.diag_index).SetFromVector(diag)
        discrete_state.get_mutable_vector(self.state_index).SetFromVector(np.asarray(xs, float))
