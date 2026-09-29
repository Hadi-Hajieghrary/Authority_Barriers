"""Trial harness: build the full-order diagram (plant, cables, thrust/torque, wind, attitude loop at
1 kHz, projector, 200 Hz filter, loggers), run one trial with a termination monitor, run many in
parallel. Integrator: radau3 at accuracy 1e-6, max step 1e-3 (D-17) unless configured otherwise."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from multiprocessing import get_context

import numpy as np
from pydrake.multibody.plant import ExternallyAppliedSpatialForceMultiplexer
from pydrake.systems.analysis import ApplySimulatorConfig, Simulator, SimulatorConfig
from pydrake.systems.framework import DiagramBuilder, EventStatus
from pydrake.systems.primitives import LogVectorOutput

from authority_barriers.theory import hocbf as HB
from authority_barriers.theory.authority import BarrierData
from authority_barriers.theory.filters import (BackupIntegratedFilter, CableCBFBaseline, DistributedProposedFilter, HOCBFFilter, MultiWallFilter,
                                   ProposedFilter)
from authority_barriers.theory.params import E3, Params, load_set
from authority_barriers.theory.state import State
from authority_barriers.theory.taut_model import rhs_from_thrust, unpack
from .cables import CableForces, cable_constants
from .filter_system import DIAG_FIELDS, FilterSystem
from .nominal import AdversarialNominal, HoverNominal, VelocityTransportNominal
from .plant import build_plant, read_state, set_initial_state
from .projector import StateProjector
from .recorder import TrialLog
from .wind import WindForces


@dataclass
class TrialConfig:
    params: str = "A"
    filter: str = "proposed"          # proposed | hocbf | none | distributed (D-27) | cable_cbf (E7 baseline) | backup (E6/E7)
    robust: bool = False              # proposed filter with D_rob
    swing_mode: str = "sampled"
    hdot_margin: float = 0.0          # sampled-data margin delta on the barrier rows (Rem. 19); 0 = raw filter (17)
    rows: str = "maximizers"          # barrier rows: maximizers (paper's literal (17c); D-21's local_maxima reverted after E2) | local_maxima
    gains: tuple = (1.0, 20.0)        # HOCBF (k1, k2)
    hocbf_swing_rate_row: bool = True
    nominal: str = "velocity"         # velocity | adversarial | hover
    v_cmd: float = 3.0
    t_final: float = 10.0
    seed: int = 0
    wind_scale: float = 0.0           # 0 = no wind; 1 = d_bar; 2 = 2 d_bar
    integrator: str = "radau3"
    accuracy: float = 1e-6
    max_step: float = 1e-3
    m_L_true: float | None = None     # plant payload mass if different from the nominal
    cable_k_factor: float = 1.0
    omega_n_att: float = 300.0        # practical limit of a 1 kHz loop (D-8 revised); 100 as a sensitivity case
    actuator: str = "attitude"        # attitude (geometric loop at 1 kHz, Sec. VI) | perfect (thrust vector applied directly, Sec. II model)
    log_mu: bool = False
    terminate_on_wall: bool = True
    wall_cap: float | None = None     # wall-clock seconds per trial; the monitor stops the trial (termination "wall_cap"), reported as truncated
    param_overrides: dict | None = None   # constants overriding the named set for the filter and the plant (E7: f_max sweep, mass rule)
    corridor_half_width: float | None = None   # E7 corridor: two walls with normals +/- r at +/- half width (MultiWallFilter); nominal "corridor"
    qd_scale: float = 1.0             # E7: initial swing rates scaled after sampling (1.5 = above omega_bar, outside X_op)
    nominal_k_swing: float = 2.0      # transport nominal: swing-rate damping gain (B7)
    nominal_k_q: float = 2.0          # transport nominal: formation-hold gain (corridor nominal; 0 = no hold)
    nominal_version: int = 3          # 3 = transport/hover nominal with the cable feed-forward m_i P_i a and the 2-D formation (C-15, 2026-09-26); older logs are re-run
    label: str = ""

    def to_dict(self):
        return asdict(self)


def attitudes_from_thrust(u: np.ndarray, yaw: float = 0.0) -> np.ndarray:
    """Rotation matrices whose third column is the commanded thrust direction (same construction as
    the attitude controller's R_d), identity for a zero command."""
    out = []
    b1_ref = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    for ui in np.asarray(u, float):
        n = np.linalg.norm(ui)
        if n < 1e-9:
            out.append(np.eye(3)); continue
        b3 = ui / n
        b2 = np.cross(b3, b1_ref); b2 /= np.linalg.norm(b2)
        b1 = np.cross(b2, b3)
        out.append(np.stack([b1, b2, b3], axis=1))
    return np.array(out)


def corridor_params(p: Params, half_width: float) -> list[Params]:
    """The two walls of a corridor of the given half width: normals +/- r_vec of the base set, planes r . x = +/- half width."""
    r = p.r_vec
    d = p.to_dict()
    return [Params(**{**d, "n": tuple((+r).tolist()), "d0": half_width}), Params(**{**d, "n": tuple((-r).tolist()), "d0": half_width})]


def apply_overrides(p: Params, cfg: TrialConfig) -> Params:
    if not cfg.param_overrides:
        return p
    d = p.to_dict()
    ov = dict(cfg.param_overrides)
    for k in ("f_max", "T_bar", "rho", "rho_fb", "m", "l"):                 # per-cable constants given as scalars
        if k in ov and np.isscalar(ov[k]):
            ov[k] = [float(ov[k])] * p.N
    return Params(**{**d, **ov})


def make_filter(p: Params, cfg: TrialConfig):
    if cfg.corridor_half_width is not None:
        return MultiWallFilter(corridor_params(p, cfg.corridor_half_width), swing_mode=cfg.swing_mode)
    if cfg.filter == "cable_cbf":
        a1, a2 = HB.LinearClassK(cfg.gains[0]), HB.LinearClassK(cfg.gains[1])
        return CableCBFBaseline(p, a1, a2, swing_rate_row=True)
    if cfg.filter == "backup":
        data = BarrierData.robust(p) if cfg.robust else BarrierData.nominal(p)
        return BackupIntegratedFilter(p, data, swing_mode=cfg.swing_mode, hdot_margin=cfg.hdot_margin)
    if cfg.filter == "proposed":
        data = BarrierData.robust(p) if cfg.robust else BarrierData.nominal(p)
        return ProposedFilter(p, data, swing_mode=cfg.swing_mode, hdot_margin=cfg.hdot_margin, rows=cfg.rows)
    if cfg.filter == "distributed":
        data = BarrierData.robust(p) if cfg.robust else BarrierData.nominal(p)
        return DistributedProposedFilter(p, data, swing_mode=cfg.swing_mode, hdot_margin=cfg.hdot_margin)
    if cfg.filter == "hocbf":
        a1, a2 = HB.LinearClassK(cfg.gains[0]), HB.LinearClassK(cfg.gains[1])
        return HOCBFFilter(p, a1, a2, swing_rate_row=cfg.hocbf_swing_rate_row)
    if cfg.filter == "none":
        return PassThrough()
    raise ValueError(cfg.filter)


class PassThrough:
    """Unfiltered baseline: the nominal command goes straight to the attitude loop."""
    data = None

    def solve(self, st, u_nom):
        from authority_barriers.theory.filters import FilterResult
        return FilterResult(np.asarray(u_nom, float), np.full(len(u_nom), np.nan), np.zeros(0), True, False, 0.0,
                            "passthrough", 0.0, np.nan)


def make_nominal(p: Params, cfg: TrialConfig, x0: State):
    if cfg.nominal == "velocity":                                            # transport toward the wall: lean along n at most 0.5 z_bar, formation ellipse (0.5 z_bar, 0.5 w_bar)
        return VelocityTransportNominal(p, cfg.v_cmd * p.n_vec, z_ref=float(x0.x_L[2]), k_swing=cfg.nominal_k_swing, k_q=cfg.nominal_k_q)
    if cfg.nominal == "adversarial":
        return AdversarialNominal(p)
    if cfg.nominal == "hover":
        return HoverNominal(p)
    if cfg.nominal == "corridor":                                            # transport along the corridor axis (the base set's wall normal)
        # the corridor axis is the walls' w-direction (limit w_bar) and the lateral direction their z-direction (limit z_bar):
        # lean along the axis at most 0.5 w_bar, formation ellipse of radii (0.5 w_bar along the axis, 0.5 z_bar laterally) (B7 of the plan)
        return VelocityTransportNominal(p, cfg.v_cmd * p.n_vec, z_ref=float(x0.x_L[2]), k_q=cfg.nominal_k_q, k_swing=cfg.nominal_k_swing,
                                        lean_limit=0.5 * p.w_bar, spread=(0.5 * p.w_bar, 0.5 * p.z_bar))
    raise ValueError(cfg.nominal)


def build_diagram(p: Params, cfg: TrialConfig, x0: State):
    from .actuation import ThrustTorqueApplier
    from .attitude import GeometricAttitudeController
    from .direct_thrust import DirectThrustApplier
    from pydrake.systems.primitives import ConstantVectorSource

    p_plant = p if cfg.m_L_true is None else Params(**{**p.to_dict(), "m_L": cfg.m_L_true})
    builder = DiagramBuilder()
    info = build_plant(p_plant, builder)
    k, c = cable_constants(p)
    cables = builder.AddSystem(CableForces(info, p_plant, k=k * cfg.cable_k_factor, c=c * np.sqrt(cfg.cable_k_factor)))
    if cfg.actuator == "attitude":
        applier = builder.AddSystem(ThrustTorqueApplier(info.plant, info.quads, p.f_max_arr))
        att = builder.AddSystem(GeometricAttitudeController(info.plant, info.quads, info.J, p.f_max_arr, dt=p.dt_attitude,
                                                            omega_n=cfg.omega_n_att))
    elif cfg.actuator == "perfect":
        applier = builder.AddSystem(DirectThrustApplier(info, p.f_max_arr))
        att = None
    else:
        raise ValueError(cfg.actuator)
    proj = builder.AddSystem(StateProjector(info, p))
    filt = make_filter(p, cfg)
    a1 = a2 = None
    if cfg.filter == "hocbf" or cfg.log_mu:
        a1, a2 = HB.LinearClassK(cfg.gains[0]), HB.LinearClassK(cfg.gains[1])
    p_log = corridor_params(p, cfg.corridor_half_width)[0] if cfg.corridor_half_width is not None else p   # distances logged against the first corridor wall
    fs = builder.AddSystem(FilterSystem(p_log, filt, make_nominal(p, cfg, x0), log_mu=cfg.log_mu, a1=a1, a2=a2))
    inputs = [cables, applier] + ([builder.AddSystem(WindForces(info, p, seed=cfg.seed, scale=cfg.wind_scale))]
                                  if cfg.wind_scale > 0 else [])
    mux = builder.AddSystem(ExternallyAppliedSpatialForceMultiplexer(len(inputs)))
    for k_, s in enumerate(inputs):
        builder.Connect(s.GetOutputPort("spatial_forces"), mux.get_input_port(k_))
    builder.Connect(mux.get_output_port(0), info.plant.get_applied_spatial_force_input_port())
    xport = info.plant.get_state_output_port()
    builder.Connect(xport, cables.get_input_port(0))
    builder.Connect(xport, proj.get_input_port(0))
    builder.Connect(proj.get_output_port(0), fs.GetInputPort("taut_state"))
    builder.Connect(cables.GetOutputPort("cable_data"), fs.GetInputPort("cable_data"))
    if att is not None:
        builder.Connect(xport, applier.GetInputPort("plant_state"))
        builder.Connect(xport, att.GetInputPort("plant_state"))
        builder.Connect(fs.GetOutputPort("thrust_cmd"), att.GetInputPort("thrust_cmd"))
        builder.Connect(att.GetOutputPort("actuation"), applier.GetInputPort("actuation"))
        att_port = att.GetOutputPort("tracking_error")
    else:
        builder.Connect(fs.GetOutputPort("thrust_cmd"), applier.GetInputPort("thrust_cmd"))
        att_port = builder.AddSystem(ConstantVectorSource(np.zeros(p.N))).get_output_port(0)
    logs = {
        "x": LogVectorOutput(xport, builder, p.dt_attitude),
        "diag": LogVectorOutput(fs.GetOutputPort("diag"), builder, p.dt_filter),
        "cable": LogVectorOutput(cables.GetOutputPort("cable_data"), builder, p.dt_filter),
        "cmd": LogVectorOutput(fs.GetOutputPort("thrust_cmd"), builder, p.dt_filter),
        "att": LogVectorOutput(att_port, builder, p.dt_filter),
        "taut": LogVectorOutput(fs.GetOutputPort("taut_at_tick"), builder, p.dt_filter),
    }
    diagram = builder.Build()
    systems = dict(info=info, cables=cables, applier=applier, att=att, proj=proj, fs=fs, filt=filt, logs=logs, k=k, c=c)
    return diagram, systems


def run_trial(cfg: TrialConfig, x0: State, p: Params | None = None) -> TrialLog:
    p = apply_overrides(p or load_set(cfg.params), cfg)
    if cfg.qd_scale != 1.0:
        x0 = State(x0.x_L.copy(), x0.v_L.copy(), x0.q.copy(), cfg.qd_scale * x0.qd)
    diagram, S = build_diagram(p, cfg, x0)
    sim = Simulator(diagram)
    ApplySimulatorConfig(SimulatorConfig(integration_scheme=cfg.integrator, accuracy=cfg.accuracy,
                                         max_step_size=cfg.max_step, use_error_control=True), sim)
    ctx = sim.get_mutable_context()
    info = S["info"]
    # consistent initialization (the theory's state has no cable stretch and no attitude): pre-stretch the
    # springs to the tensions implied by the filter's first command and align each thrust axis with it
    u0 = S["fs"].filt.solve(x0, make_nominal(p, cfg, x0)(0.0, x0)).u
    T0 = rhs_from_thrust(x0, u0, p)["T"]
    set_initial_state(info, info.plant.GetMyMutableContextFromRoot(ctx), x0, p, attitudes=attitudes_from_thrust(u0),
                      prestretch=np.clip(T0, 0.0, None), k=S["k"] * cfg.cable_k_factor)
    reason = {"why": "horizon"}
    t_wall0 = time.perf_counter()

    walls = corridor_params(p, cfg.corridor_half_width) if cfg.corridor_half_width is not None else [p]

    def monitor(root):
        x = info.plant.get_state_output_port().Eval(info.plant.GetMyContextFromRoot(root))
        xL = info.pos(x, "payload")
        h = min(pw.d0 - pw.n_vec @ xL for pw in walls)
        if cfg.terminate_on_wall and h < 0.0:
            reason["why"] = "wall_contact"
            return EventStatus.ReachedTermination(diagram, "wall contact")
        if cfg.wall_cap is not None and time.perf_counter() - t_wall0 > cfg.wall_cap:
            reason["why"] = "wall_cap"
            return EventStatus.ReachedTermination(diagram, "wall-time cap")
        if not np.all(np.isfinite(x)) or np.linalg.norm(xL) > 1e4:
            reason["why"] = "divergence"
            return EventStatus.ReachedTermination(diagram, "divergence")
        return EventStatus.Succeeded()

    sim.set_monitor(monitor)
    sim.Initialize()
    t0 = time.perf_counter()
    sim.AdvanceTo(cfg.t_final)
    wall = time.perf_counter() - t0
    L = S["logs"]
    lx, ld, lc, lu, la, lt = (L[k_].FindLog(ctx) for k_ in ("x", "diag", "cable", "cmd", "att", "taut"))
    meta = {"config": cfg.to_dict(), "params": p.to_dict(), "derived": {"alpha_sat": p.alpha_sat, "nu_rel": p.nu_rel,
            "T_bar_rel": p.T_bar_rel.tolist()}, "cable_k": S["k"], "cable_c": S["c"] * np.sqrt(cfg.cable_k_factor),
            "integrator": {"scheme": cfg.integrator, "accuracy": cfg.accuracy, "max_step": cfg.max_step},
            "filter_dt": p.dt_filter, "attitude_dt": p.dt_attitude, "solver": "Clarabel", "kappa_H": p.kappa_H,
            "gains": list(cfg.gains), "wall_time": wall, "x0": {"x_L": x0.x_L.tolist(), "v_L": x0.v_L.tolist(),
            "q": x0.q.tolist(), "qd": x0.qd.tolist()}}
    D = ld.data().copy()
    t_tick = D[DIAG_FIELDS.index("t")]                       # the tick each logged command/diag/state belongs to
    keep = np.concatenate([[True], np.diff(t_tick) > 0])     # drop the duplicate publish of the initialization tick
    return TrialLog(lx.sample_times().copy(), lx.data().copy(), t_tick[keep], D[:, keep],
                    lc.data()[:, keep].copy(), lu.data()[:, keep].copy(), la.data()[:, keep].copy(), lt.data()[:, keep].copy(),
                    meta, reason["why"])


def _worker(args):
    cfg_d, x0_d, params = args
    cfg = TrialConfig(**cfg_d)
    x0 = State(np.array(x0_d["x_L"]), np.array(x0_d["v_L"]), np.array(x0_d["q"]), np.array(x0_d["qd"]))
    log = run_trial(cfg, x0, load_set(params))
    return log


def run_many(cfgs, x0s, n_workers: int = 8):
    """Run trials in parallel processes (spawn; Meshcat is never started in workers)."""
    jobs = [(cfg.to_dict(), {"x_L": x0.x_L.tolist(), "v_L": x0.v_L.tolist(), "q": x0.q.tolist(), "qd": x0.qd.tolist()},
             cfg.params) for cfg, x0 in zip(cfgs, x0s)]
    if n_workers <= 1:
        return [_worker(j) for j in jobs]
    with get_context("spawn").Pool(n_workers) as pool:
        return pool.map(_worker, jobs)
