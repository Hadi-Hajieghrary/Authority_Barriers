"""3-D direct-collocation viability search of E3 (Sec. VI Q2, "In three dimensions"; Thm. 12; M4.4).

From a fixed initial state x0 of the N-quadrotor taut-cable team, search for an input trajectory that
respects the safe set C = {h >= 0} cap X_op and the admissible inputs U(x) (thrust norms, tension floor,
D-4 cone) at every knot and ends in X_RF = {H = h - D >= 0, swing states in V(nu, z_bar) x V(nu_w, w_bar)},
from where the braking maneuver is safe for all future time (Thm. 12(i)). A trajectory from a state with
h0 < D_rel(x0) - 1 cm that passes `verify_trajectory` refutes Thm. 12(ii); a solver failure proves
nothing (the optimizer under-approximates the kernel).

Transcription: pydrake.planning.DirectCollocation on kernel.taut_system.TautCableSystem (inputs
[T, c] of eq. gram), equal time intervals in [dt_min, dt_max], initial state fixed. Nonlinear knot
constraints are dtype-agnostic Python callables (PyFunctionConstraint) evaluated by Drake with
AutoDiffXd; linear ones are LinearConstraint/BoundingBoxConstraint rows. The terminal constraint
H(x_T) >= 0 is a custom constraint whose AutoDiffXd derivative is the analytic gradient of D
(stopping.evaluate, Prop. 17) at the smallest maximizing time, a valid element of the Clarke
gradient of the nonsmooth D (Danskin); ties are counted.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
from pydrake.autodiffutils import AutoDiffXd, ExtractGradient, ExtractValue
from pydrake.planning import DirectCollocation
from pydrake.solvers import (BoundingBoxConstraint, IpoptSolver, LinearConstraint, PyFunctionConstraint,
                             SnoptSolver, SolverOptions)
from pydrake.trajectories import PiecewisePolynomial

from authority_barriers.theory import profile as PF
from authority_barriers.theory import stopping as SD
from authority_barriers.theory import taut_model as tm
from authority_barriers.theory.authority import BarrierData
from authority_barriers.theory.maneuver import plan
from authority_barriers.theory.params import Params
from authority_barriers.theory.state import State
from .taut_system import TautCableSystem, input_from_plan, perp_from_coeffs, split_input, thrust_from_input

INF = float("inf")


# ---------------------------------------------------------------- dtype-agnostic helpers
def _values(x) -> np.ndarray:
    """Float values of a float array or an object array of AutoDiffXd, flattened."""
    x = np.asarray(x)
    if x.dtype == object:
        return ExtractValue(x).reshape(-1)
    return np.asarray(x, float).reshape(-1)


def _pos(x):
    """Positive part max(x, 0) elementwise; keeps AutoDiffXd entries (with zero derivative where x <= 0)."""
    x = np.asarray(x)
    return np.where(_values(x) > 0, x, 0.0 * x)


def _sqnorm_rows(a):
    return (a * a).sum(axis=-1)


def _wall(st: State, p: Params):
    """(h, v) = (d0 - n . x_L, n . xdot_L), dtype-agnostic."""
    n = p.n_vec
    return p.d0 - n @ st.x_L, n @ st.v_L


# ---------------------------------------------------------------- knot constraint functions
def path_values(xu, p: Params) -> np.ndarray:
    """Normalized nonlinear path constraints at one knot, every row <= 1:
    [||qdot_i||^2 / omega_bar^2 (N), ||u_i||^2 / f_max_i^2 (N), ||sum_j T_j q_j||^2 / (m_L a_max)^2 (1)]."""
    N = p.N
    nx = tm.n_state(N)
    xu = np.asarray(xu).reshape(-1)
    st = tm.unpack(xu[:nx], N)
    T, c = split_input(xu[nx:], N)
    out = tm.rhs_from_tensions(st, T, perp_from_coeffs(st, p, c), p)
    om2 = _sqnorm_rows(st.qd) / p.omega_bar ** 2
    u2 = _sqnorm_rows(out["u"]) / p.f_max_arr ** 2
    Tq = T @ st.q
    cone = (Tq * Tq).sum() / (p.m_L * p.a_max) ** 2
    return np.concatenate([om2, u2, np.array([cone], dtype=om2.dtype)])


def path_bounds(p: Params):
    m = 2 * p.N + 1
    return np.full(m, -INF), np.ones(m)


def swing_set_values(x, p: Params, data: BarrierData) -> np.ndarray:
    """Membership in V(nu, z_bar) x V(nu_w, w_bar) as 4N smooth rows (eq. box, two per coordinate):
    z_i + zd_i,+^2/(2 nu) <= z_bar ; z_i - zd_i,-^2/(2 nu) >= -z_bar ; and the same for (w_i, wd_i)."""
    N = p.N
    st = tm.unpack(x, N)
    y, r = p.y_vec, p.r_vec
    z, zd, w, wd = st.q @ y, st.qd @ y, st.q @ r, st.qd @ r
    nu, nu_w = data.nu, p.nu_w
    return np.concatenate([z + _pos(zd) ** 2 / (2.0 * nu), z - _pos(-zd) ** 2 / (2.0 * nu),
                           w + _pos(wd) ** 2 / (2.0 * nu_w), w - _pos(-wd) ** 2 / (2.0 * nu_w)])


def swing_set_bounds(p: Params, data: BarrierData):
    N = p.N
    zb, wb = data.zeta_bar, p.w_bar
    lb = np.concatenate([np.full(N, -INF), np.full(N, -zb), np.full(N, -INF), np.full(N, -wb)])
    ub = np.concatenate([np.full(N, zb), np.full(N, INF), np.full(N, wb), np.full(N, INF)])
    return lb, ub


class TerminalHConstraint:
    """H(x) = h - D(v, z, zd) as a Python constraint function with an analytic AutoDiffXd derivative.

    Float evaluation returns H. AutoDiffXd evaluation extracts the values, evaluates D and the gradient
    of E(t; x) at the smallest maximizing time t* (stopping.evaluate; Danskin: at a unique maximizer this
    is the gradient of D, at ties it is one element of the Clarke gradient), and chains it through
    the incoming gradients. Counts calls, ties and clamped (outside-V) evaluations."""

    def __init__(self, p: Params, data: BarrierData):
        self.p, self.data = p, data
        self.nx = tm.n_state(p.N)
        self.n_calls = 0
        self.n_ad_calls = 0
        self.n_ties = 0
        self.n_clamped = 0

    def evaluate(self, x: np.ndarray):
        """(H, dH/dx (nx,), StoppingResult) at a float state vector. Outside V, stopping.evaluate
        clamps (z_i, zd_i) into V (profile.clamp_to_V) before evaluating D; the gradient is chained
        through that clamp (zero for a clipped coordinate, and the dependence of the clipped z on zd
        through the boundary of V), so the constraint function is consistent at every iterate."""
        p, N = self.p, self.p.N
        data = self.data
        st = tm.unpack(np.asarray(x, float), N)
        y, n = p.y_vec, p.n_vec
        h, v = st.h(p), st.v(p)
        z, zd = st.q @ y, st.qd @ y
        res = SD.evaluate(v, z, zd, data)
        H = h - res.D
        gz, gw = res.grad_zeta[0].copy(), res.grad_omega[0].copy()
        if res.clamped:
            nu, zb = data.nu, data.zeta_bar
            om_max = 2.0 * np.sqrt(nu * zb)
            zc, wc, _ = PF.clamp_to_V(z, zd, nu, zb)
            lo, hi = PF.bounds_V(wc, nu, zb)
            dw_dw = np.where(np.abs(zd) < om_max, 1.0, 0.0)
            at_hi, at_lo = z > hi, z < lo
            dz_dz = np.where(at_hi | at_lo, 0.0, 1.0)
            dz_dw = np.where(at_hi, -np.maximum(wc, 0.0) / nu, np.where(at_lo, -np.maximum(-wc, 0.0) / nu, 0.0))
            gz, gw = gz * dz_dz, (gz * dz_dw + gw) * dw_dw
        g = np.zeros(self.nx)
        g[0:3] = -n
        g[3:6] = -res.grad_v[0] * n
        for i in range(N):
            g[6 + 3 * i:9 + 3 * i] = -gz[i] * y
            g[6 + 3 * N + 3 * i:9 + 3 * N + 3 * i] = -gw[i] * y
        return H, g, res

    def __call__(self, x):
        x = np.asarray(x).reshape(-1)
        self.n_calls += 1
        if x.dtype == object:
            self.n_ad_calls += 1
            H, g, res = self.evaluate(ExtractValue(x).reshape(-1))
            self.n_ties += int(res.t_star.size > 1)
            self.n_clamped += int(res.clamped)
            J = ExtractGradient(x)
            return np.array([AutoDiffXd(H, g @ J)], dtype=object)
        H, _, res = self.evaluate(x)
        self.n_ties += int(res.t_star.size > 1)
        self.n_clamped += int(res.clamped)
        return np.array([H])

    @property
    def tie_fraction(self) -> float:
        return self.n_ties / self.n_calls if self.n_calls else 0.0


def saturated_rest_value(x, p: Params, data: BarrierData) -> np.ndarray:
    """h - v^2 / (2 alpha_sat) (>= 0 together with v >= 0 gives h >= v_+^2/(2 alpha_sat))."""
    st = tm.unpack(x, p.N)
    h, v = _wall(st, p)
    return np.array([h - v * v / (2.0 * data.alpha_sat)])


# ---------------------------------------------------------------- native-unit violations
def path_violations(p: Params, X: np.ndarray, U: Optional[np.ndarray] = None,
                    T_of: Optional[np.ndarray] = None, thrust: Optional[np.ndarray] = None) -> dict:
    """Max violation (native units, >= 0) over the rows of X (K, nx) of: h >= 0 [m], |z_i| <= z_bar,
    |w_i| <= w_bar, ||qdot_i|| <= omega_bar [rad/s], T_i >= T_min [N], ||u_i|| <= f_max_i [N],
    ||sum T_j q_j|| <= m_L a_max [N]. Inputs either as collocation inputs U (K, 3N) or as
    (tensions T_of (K, N), thrust vectors (K, N, 3))."""
    N = p.N
    X = np.atleast_2d(X)
    out = {k: 0.0 for k in ("h", "z", "w", "omega", "tension", "thrust", "cone")}
    for k in range(X.shape[0]):
        st = tm.unpack(X[k], N)
        out["h"] = max(out["h"], -st.h(p))
        out["z"] = max(out["z"], float(np.max(np.abs(st.z(p)) - p.z_bar)))
        out["w"] = max(out["w"], float(np.max(np.abs(st.w(p)) - p.w_bar)))
        out["omega"] = max(out["omega"], float(np.max(np.sqrt(st.omega2()) - p.omega_bar)))
        if U is not None:
            T, c = split_input(U[k], N)
            u = thrust_from_input(st, p, U[k])
        elif T_of is not None:
            T, u = T_of[k], thrust[k]
        else:
            continue
        out["tension"] = max(out["tension"], float(np.max(p.T_min - T)))
        out["thrust"] = max(out["thrust"], float(np.max(np.linalg.norm(u, axis=1) - p.f_max_arr)))
        out["cone"] = max(out["cone"], float(np.linalg.norm(T @ st.q) - p.m_L * p.a_max))
    out = {k: max(v, 0.0) for k, v in out.items()}
    out["max"] = max(out.values())
    return out


def terminal_membership(st: State, p: Params, data: BarrierData, tol: float = 1e-3) -> dict:
    """H and swing-set membership of a state (strict and within tol)."""
    z, zd, w, wd = st.z(p), st.zd(p), st.w(p), st.wd(p)
    H = st.h(p) - SD.D_of(st.v(p), z, zd, data)
    inV = bool(np.all(PF.in_V(z, zd, data.nu, data.zeta_bar)) and np.all(PF.in_V(w, wd, p.nu_w, p.w_bar)))
    inV_tol = bool(np.all(PF.in_V(z, zd, data.nu, data.zeta_bar, tol)) and np.all(PF.in_V(w, wd, p.nu_w, p.w_bar, tol)))
    return {"H": float(H), "in_V": inV, "in_V_tol": inV_tol, "in_XRF": bool(inV and H >= 0.0),
            "in_XRF_tol": bool(inV_tol and H >= -tol)}


# ---------------------------------------------------------------- initial guesses
@dataclass
class Guess:
    times: np.ndarray   # (knots,)
    X: np.ndarray       # (knots, nx)
    U: np.ndarray       # (knots, 3N)
    kind: str = ""


def plan_guess(p: Params, x0: State, data: BarrierData, knots: int, horizon: float) -> Guess:
    """Replay of the braking maneuver pi^sharp on the taut ODE from x0 (terminate_on_slack=False),
    sampled at the knots; inputs from input_from_plan. A feasible witness whenever H(x0) >= 0."""
    policy = lambda t, st: plan(st, p, data).u
    dt_out = horizon / (knots - 1)
    tr = tm.integrate(x0, policy, horizon, p, dt_out=dt_out, rtol=1e-9, atol=1e-11, terminate_on_slack=False)
    X = tr.X[:knots]
    if X.shape[0] < knots:                                  # guard against a short output (never expected)
        X = np.vstack([X, np.repeat(X[-1:], knots - X.shape[0], axis=0)])
    U = np.array([input_from_plan(tm.unpack(X[k], p.N), p, plan(tm.unpack(X[k], p.N), p, data)) for k in range(knots)])
    return Guess(np.linspace(0.0, horizon, knots), X, U, "plan")


def line_guess(p: Params, x0: State, knots: int, horizon: float) -> Guess:
    """Straight line in the swing coordinates to a saturated-rest state (z = z_bar, w = 0, rates 0,
    v = 0, h = h0 / 2), tensions T_bar, no tangent-plane thrust."""
    N = p.N
    h0, v0 = x0.h(p), x0.v(p)
    z0, w0, zd0, wd0 = x0.z(p), x0.w(p), x0.zd(p), x0.wd(p)
    x_perp = float(p.r_vec @ x0.x_L)
    alt = float(x0.x_L[2])
    s = np.linspace(0.0, 1.0, knots)
    X = np.empty((knots, tm.n_state(N)))
    for k in range(knots):
        st = State.from_swing(p, (1 - s[k]) * h0 + s[k] * 0.5 * h0, (1 - s[k]) * v0,
                              (1 - s[k]) * z0 + s[k] * p.z_bar, (1 - s[k]) * w0, (1 - s[k]) * zd0, (1 - s[k]) * wd0,
                              x_perp=x_perp, altitude=alt)
        X[k] = tm.pack(st)
    U = np.tile(np.concatenate([p.T_bar_arr, np.zeros(2 * N)]), (knots, 1))
    return Guess(np.linspace(0.0, horizon, knots), X, U, "line")


def shift_guess(g: Guess, p: Params, x0: State) -> Guess:
    """Translate a previous trajectory along the wall normal so that it starts at h(x0)."""
    X = np.array(g.X, float, copy=True)
    h_old = p.d0 - p.n_vec @ X[0, 0:3]
    X[:, 0:3] -= p.n_vec[None, :] * (x0.h(p) - h_old)
    X[0] = tm.pack(x0)
    return Guess(np.array(g.times, float), X, np.array(g.U, float, copy=True), g.kind + "+shift")


# ---------------------------------------------------------------- result
@dataclass
class CollocationResult:
    success: bool
    solver_status: str
    x_traj: np.ndarray            # (knots, nx)
    u_traj: np.ndarray            # (knots, 3N)
    times: np.ndarray             # (knots,)
    H_end: float
    terminal_state: State
    constraint_violation: float   # max native-unit violation of the knot constraints (path_violations)
    solve_time: float
    solver: str = ""
    solver_info: int = -1
    prog_violation: float = np.nan   # max violation over all program constraints (program units, incl. defects)
    terminal_violation: float = np.nan
    h0: float = np.nan
    D0: float = np.nan
    knots: int = 0
    terminal: str = ""
    guess: str = ""
    duration: float = np.nan
    tie_fraction: float = 0.0     # fraction of H-constraint evaluations at tied maximizing times
    clamp_fraction: float = 0.0   # fraction evaluated outside V (clamped)
    n_ties_end: int = 0           # number of maximizing times of D at the returned terminal state
    violations: dict = field(default_factory=dict)

    def as_guess(self) -> Guess:
        return Guess(self.times, self.x_traj, self.u_traj, "warm")

    def to_dict(self, arrays: bool = True) -> dict:
        d = {k: v for k, v in asdict(self).items() if k not in ("terminal_state", "x_traj", "u_traj", "times")}
        if arrays:
            d["x_traj"], d["u_traj"], d["times"] = self.x_traj.tolist(), self.u_traj.tolist(), self.times.tolist()
        return d


# ---------------------------------------------------------------- solvers
def _snopt_ok() -> bool:
    s = SnoptSolver()
    return bool(s.available() and s.enabled())


def _make_solver(name: str, major_iterations: int, time_limit: Optional[float], verbose: bool, num_vars: int,
                 print_file: Optional[str]):
    if name == "snopt":
        solver = SnoptSolver()
        opts = SolverOptions()
        sid = solver.solver_id()
        opts.SetOption(sid, "Major feasibility tolerance", 1e-6)
        opts.SetOption(sid, "Major optimality tolerance", 1e-4)
        opts.SetOption(sid, "Major iterations limit", int(major_iterations))
        opts.SetOption(sid, "Iterations limit", int(max(200000, 400 * major_iterations)))
        opts.SetOption(sid, "Superbasics limit", int(num_vars + 1))
        if time_limit is not None:
            opts.SetOption(sid, "Time limit", float(time_limit))
        if verbose and print_file:
            opts.SetOption(sid, "Print file", print_file)
        return solver, opts
    solver = IpoptSolver()
    opts = SolverOptions()
    sid = solver.solver_id()
    opts.SetOption(sid, "max_iter", int(major_iterations))
    opts.SetOption(sid, "tol", 1e-6)
    opts.SetOption(sid, "constr_viol_tol", 1e-6)
    opts.SetOption(sid, "acceptable_tol", 1e-4)
    opts.SetOption(sid, "print_level", 5 if verbose else 0)
    if time_limit is not None:
        opts.SetOption(sid, "max_wall_time", float(time_limit))
    return solver, opts


def _is_license_problem(err: Exception) -> bool:
    msg = str(err).lower()
    return any(s in msg for s in ("license", "licence", "not available", "unavailable", "not enabled"))


# ---------------------------------------------------------------- program
class _Program:
    """Holds the DirectCollocation program and the objects it aliases (system, context, constraints)."""

    def __init__(self, p: Params, x0: State, knots: int, dt_min: float, dt_max: float, terminal: str,
                 data: BarrierData, cost: str, H_margin: float = 1e-4, sat_tol: float = 1e-4, w_time: float = 1e-2,
                 w_reg: float = 1e-3):
        N = p.N
        nx, nu = tm.n_state(N), 3 * N
        self.p, self.data, self.knots, self.terminal, self.H_margin = p, data, knots, terminal, H_margin
        self.system = TautCableSystem(p)
        self.context = self.system.CreateDefaultContext()
        self.dircol = dc = DirectCollocation(self.system, self.context, num_time_samples=knots,
                                             minimum_time_step=dt_min, maximum_time_step=dt_max)
        self.prog = prog = dc.prog()
        dc.AddEqualTimeIntervalsConstraints()
        x, u = dc.state(), dc.input()
        n, y, r = p.n_vec, p.y_vec, p.r_vec
        # initial state fixed (||q_i|| = 1 and q_i . qdot_i = 0 hold there and are invariants of the ODE)
        x0v = tm.pack(x0).astype(float)
        prog.AddBoundingBoxConstraint(x0v, x0v, dc.initial_state())
        # ---- linear knot constraints: h >= 0, |z_i| <= z_bar, |w_i| <= w_bar, T_i >= T_min
        dc.AddConstraintToAllKnotPoints(LinearConstraint(n.reshape(1, 3), [-INF], [p.d0]), x[0:3])
        for i in range(N):
            qi = x[6 + 3 * i:9 + 3 * i]
            dc.AddConstraintToAllKnotPoints(LinearConstraint(y.reshape(1, 3), [-p.z_bar], [p.z_bar]), qi)
            dc.AddConstraintToAllKnotPoints(LinearConstraint(r.reshape(1, 3), [-p.w_bar], [p.w_bar]), qi)
        # tension floor, plus bounds implied by the cone and the thrust limit (they do not restrict the
        # feasible set: sum_j T_j <= m_L a_max / cos theta_q and ||c_i|| <= f_max_i / cos theta_q on X_op)
        T_hi = p.m_L * p.a_max / math.cos(p.theta_q)
        c_hi = np.repeat(p.f_max_arr / math.cos(p.theta_q), 2)
        lb_u = np.concatenate([np.full(N, p.T_min), -c_hi])
        ub_u = np.concatenate([np.full(N, T_hi), c_hi])
        dc.AddConstraintToAllKnotPoints(BoundingBoxConstraint(lb_u, ub_u), u)
        # loose state bounds (unit cable directions, rate bound), also implied
        lb_x = np.concatenate([np.full(6, -INF), np.full(3 * N, -1.0), np.full(3 * N, -p.omega_bar)])
        ub_x = -lb_x
        dc.AddConstraintToAllKnotPoints(BoundingBoxConstraint(lb_x, ub_x), x)
        # ---- nonlinear knot constraints: ||qdot_i||^2, ||u_i||^2, D-4 cone (normalized, <= 1)
        lb, ub = path_bounds(p)
        self.path_con = PyFunctionConstraint(nx + nu, lambda xu: path_values(xu, p), lb, ub, "path")
        dc.AddConstraintToAllKnotPoints(self.path_con, np.concatenate([x, u]))
        # ---- terminal set
        xf = dc.final_state()
        self.H_con = TerminalHConstraint(p, data)
        if terminal == "xrf":
            prog.AddConstraint(self.H_con, [H_margin], [INF], xf, "H_end")
            lbV, ubV = swing_set_bounds(p, data)
            prog.AddConstraint(lambda xv: swing_set_values(xv, p, data), lbV, ubV, xf, "V_end")
        elif terminal == "saturated_rest":
            A = np.zeros((4 * N, nx))
            lbs, ubs = np.zeros(4 * N), np.zeros(4 * N)
            for i in range(N):
                A[4 * i, 6 + 3 * i:9 + 3 * i] = y;       lbs[4 * i], ubs[4 * i] = p.z_bar - sat_tol, p.z_bar + sat_tol
                A[4 * i + 1, 6 + 3 * N + 3 * i:9 + 3 * N + 3 * i] = y; lbs[4 * i + 1], ubs[4 * i + 1] = -sat_tol, sat_tol
                A[4 * i + 2, 6 + 3 * i:9 + 3 * i] = r;   lbs[4 * i + 2], ubs[4 * i + 2] = -sat_tol, sat_tol
                A[4 * i + 3, 6 + 3 * N + 3 * i:9 + 3 * N + 3 * i] = r; lbs[4 * i + 3], ubs[4 * i + 3] = -sat_tol, sat_tol
            prog.AddConstraint(LinearConstraint(A, lbs, ubs), xf)
            prog.AddConstraint(LinearConstraint(n.reshape(1, 3), [0.0], [INF]), xf[3:6])        # v(T) >= 0
            prog.AddConstraint(lambda xv: saturated_rest_value(xv, p, data), [0.0], [INF], xf, "h_sat")
        else:
            raise ValueError(f"unknown terminal mode {terminal!r}")
        # ---- cost
        if cost in ("time", "time+reg"):
            dc.AddFinalCost(w_time * dc.time())
        if cost in ("reg", "time+reg"):
            c = u[N:]
            dc.AddRunningCost(w_reg * sum(ci * ci for ci in c))
        elif cost not in ("none", "time"):
            raise ValueError(f"unknown cost {cost!r}")

    def set_guess(self, g: Guess) -> None:
        self.dircol.SetInitialTrajectory(PiecewisePolynomial.FirstOrderHold(g.times, g.U.T),
                                         PiecewisePolynomial.FirstOrderHold(g.times, g.X.T))

    def max_violation(self, result) -> float:
        """Largest violation of any program constraint (bounds, defects, path, terminal) at the solution."""
        worst = 0.0
        for b in self.prog.GetAllConstraints():
            val = np.asarray(result.EvalBinding(b), float).reshape(-1)
            lo = np.asarray(b.evaluator().lower_bound(), float).reshape(-1)
            hi = np.asarray(b.evaluator().upper_bound(), float).reshape(-1)
            worst = max(worst, float(np.max(np.maximum(np.maximum(lo - val, val - hi), 0.0))))
        return worst

    def terminal_violation(self, xT: np.ndarray) -> float:
        p, data = self.p, self.data
        if self.terminal == "xrf":
            H = float(self.H_con(xT)[0])
            vals = swing_set_values(xT, p, data)
            lb, ub = swing_set_bounds(p, data)
            return max(0.0, self.H_margin - H, float(np.max(np.maximum(lb - vals, vals - ub))))
        st = tm.unpack(xT, p.N)
        y, r = p.y_vec, p.r_vec
        dev = np.concatenate([np.abs(st.q @ y - p.z_bar), np.abs(st.qd @ y), np.abs(st.q @ r), np.abs(st.qd @ r)])
        return max(0.0, float(np.max(dev)), -st.v(p), -float(saturated_rest_value(xT, p, data)[0]))


# ---------------------------------------------------------------- main entry points
def find_trajectory(p: Params, x0: State, knots: int = 41, dt_min: float = 0.01, dt_max: float = 0.1,
                    terminal: str = "xrf", data: Optional[BarrierData] = None, guess: str = "plan",
                    verbose: bool = False, warm_start=None, cost: str = "reg", solver: str = "auto",
                    major_iterations: int = 500, time_limit: Optional[float] = None, feas_tol: float = 2e-5,
                    H_margin: float = 1e-4, print_file: Optional[str] = None) -> CollocationResult:
    """Direct-collocation search for a trajectory from x0 into the terminal set.

    terminal: "xrf" (H(x_T) >= H_margin and swing states in V, a subset of the set of Thm. 12(i); the
    margin, 0.1 mm by default, keeps the endpoint inside X_RF despite the solver's feasibility tolerance)
    or "saturated_rest" (z_i = z_bar, zd_i = 0, w_i = wd_i = 0 within 1e-4, v >= 0, h >= v^2/(2 alpha_sat)).
    guess: "plan" (maneuver replay over knots*dt_max/2 s) or "line"; `warm_start` (a CollocationResult
    or Guess) overrides it after a shift along the wall normal to h(x0).
    cost: "reg" (1e-3 int ||c||^2, default), "time", "time+reg", "none" -- feasibility is what matters.
    solver: "auto" (SNOPT, IPOPT if SNOPT is unavailable or reports a license problem), "snopt", "ipopt".
    success := the returned solution violates no program constraint by more than feas_tol
    (program units; path rows are normalized to <= 1), whatever the solver's own verdict."""
    data = data or BarrierData.nominal(p)
    P = _Program(p, x0, knots, dt_min, dt_max, terminal, data, cost, H_margin=H_margin)
    horizon = knots * dt_max / 2.0
    if warm_start is not None:
        g = warm_start.as_guess() if isinstance(warm_start, CollocationResult) else warm_start
        g = shift_guess(g, p, x0)
    elif guess == "plan":
        g = plan_guess(p, x0, data, knots, horizon)
    elif guess == "line":
        g = line_guess(p, x0, knots, horizon)
    else:
        raise ValueError(f"unknown guess {guess!r}")
    P.set_guess(g)

    name = solver
    if name == "auto":
        name = "snopt" if _snopt_ok() else "ipopt"
    sol, opts = _make_solver(name, major_iterations, time_limit, verbose, P.prog.num_vars(), print_file)
    t0 = time.perf_counter()
    try:
        result = sol.Solve(P.prog, None, opts)
    except RuntimeError as err:
        if name == "snopt" and solver == "auto" and _is_license_problem(err):
            if verbose:
                print(f"SNOPT unavailable ({err}); falling back to IPOPT")
            name = "ipopt"
            sol, opts = _make_solver(name, major_iterations, time_limit, verbose, P.prog.num_vars(), print_file)
            t0 = time.perf_counter()
            result = sol.Solve(P.prog, None, opts)
        else:
            raise
    solve_time = time.perf_counter() - t0

    details = result.get_solver_details()
    info = int(getattr(details, "info", getattr(details, "status", -1)))
    status = f"{result.get_solution_result().name if hasattr(result.get_solution_result(), 'name') else result.get_solution_result()}"
    status = f"{status} ({name} info {info})"
    X = np.asarray(P.dircol.GetStateSamples(result), float).T
    U = np.asarray(P.dircol.GetInputSamples(result), float).T
    times = np.asarray(P.dircol.GetSampleTimes(result), float).reshape(-1)
    prog_viol = P.max_violation(result)
    term_viol = P.terminal_violation(X[-1])
    viol = path_violations(p, X, U)
    stT = tm.unpack(X[-1], p.N)
    res_end = SD.evaluate(stT.v(p), stT.z(p), stT.zd(p), data)
    H_end = float(stT.h(p) - res_end.D)
    success = bool(prog_viol <= feas_tol)
    if verbose:
        print(f"[{name}] {status}: success={success} prog_viol={prog_viol:.2e} path_viol={viol['max']:.2e} "
              f"term_viol={term_viol:.2e} H_end={H_end:.4f} T={times[-1]:.3f}s solve={solve_time:.1f}s "
              f"ties={P.H_con.tie_fraction:.0%} clamped={P.H_con.n_clamped}/{P.H_con.n_calls}")
    return CollocationResult(
        success=success, solver_status=status, x_traj=X, u_traj=U, times=times, H_end=H_end, terminal_state=stT,
        constraint_violation=float(viol["max"]), solve_time=float(solve_time), solver=name, solver_info=info,
        prog_violation=float(prog_viol), terminal_violation=float(term_viol), h0=float(x0.h(p)),
        D0=float(SD.D_of(x0.v(p), x0.z(p), x0.zd(p), data)), knots=knots, terminal=terminal, guess=g.kind,
        duration=float(times[-1]), tie_fraction=float(P.H_con.tie_fraction),
        clamp_fraction=float(P.H_con.n_clamped / P.H_con.n_calls) if P.H_con.n_calls else 0.0,
        n_ties_end=int(res_end.t_star.size), violations=viol)


def _capped_integrate(st, policy, horizon, p, wall_cap: float, **kw):
    """taut_model.integrate_capped without the call count (kept for the verification code below)."""
    tr, timed_out, _ = tm.integrate_capped(st, policy, horizon, p, wall_cap, **kw)
    return tr, timed_out


def verify_trajectory(p: Params, res: CollocationResult, data: Optional[BarrierData] = None, tol: float = 1e-3,
                      dt_out: float = 1e-3, plan_extra: float = 3.0, wall_cap: float = 600.0) -> dict:
    """Independent check of a collocation solution (K4): re-simulate the taut ODE (taut_model.integrate,
    DOP853) from x0 under the piecewise-linear interpolation of the returned inputs [T, c] over the
    returned times, report the max native-unit violation of the tension floor, thrust norms, D-4 cone,
    X_op and h >= 0 along the re-simulation, membership of the final state in X_RF (H >= 0 and swing
    sets), and the minimum h under the braking maneuver run from that final state for
    max_i t2_i + v_+/alpha_sat + plan_extra seconds (Thm. 12(i) check). `passes` requires violations <= tol,
    the endpoint in X_RF within tol, and min h >= 0 under the plan; `refutation_candidate` adds
    h0 < D_rel(x0) - 1 cm (the outer bound of Thm. 12(ii))."""
    data = data or BarrierData.nominal(p)
    N = p.N
    st0 = tm.unpack(res.x_traj[0], N)
    t_knots, U = np.asarray(res.times, float), np.asarray(res.u_traj, float)

    def u_of_t(t):
        return np.array([np.interp(t, t_knots, U[:, j]) for j in range(U.shape[1])])

    policy = lambda t, st: thrust_from_input(st, p, u_of_t(t))
    # each integration is capped at wall_cap seconds; a timed-out check is inconclusive, never a pass
    tr, timed_out = _capped_integrate(st0, policy, float(t_knots[-1]), p, wall_cap, dt_out=dt_out, rtol=1e-9, atol=1e-11,
                                      terminate_on_slack=False)
    if timed_out:
        return {"h0": float(st0.h(p)), "v0": float(st0.v(p)), "D0": float(SD.D_of(st0.v(p), st0.z(p), st0.zd(p), data)),
                "D0_rel": float(SD.D_of(st0.v(p), st0.z(p), st0.zd(p), BarrierData.relaxed(p))), "violations": None,
                "max_violation": float("nan"), "min_h_sim": float("nan"), "endpoint_gap": float("nan"), "H_end_col": float(res.H_end),
                "min_h_plan": float("nan"), "passes": False, "refutation_candidate": False, "timeout": "re-simulation", "wall_cap": wall_cap}
    viol = path_violations(p, tr.X, T_of=tr.T, thrust=tr.U)
    stT = tr.state_at(tr.t.size - 1)
    mem = terminal_membership(stT, p, data, tol)
    gap = float(np.max(np.abs(tr.X[-1] - res.x_traj[-1])))
    # braking maneuver from the re-simulated final state
    z, zd = stT.z(p), stT.zd(p)
    _, _, t2 = PF.switching(z, zd, data.nu, data.zeta_bar)
    horizon = float(np.max(t2)) + max(stT.v(p), 0.0) / data.alpha_sat + plan_extra
    tr2, plan_timeout = _capped_integrate(stT, lambda t, st: plan(st, p, data).u, horizon, p, wall_cap, dt_out=1e-2,
                                          rtol=1e-9, atol=1e-11, terminate_on_slack=True)
    if plan_timeout:
        min_h_plan, plan_slack, plan_early = float("nan"), False, False
    else:
        h_plan = np.array([tr2.state_at(k).h(p) for k in range(tr2.t.size)])
        min_h_plan, plan_slack, plan_early = float(np.min(h_plan)), tr2.slack_time is not None, bool(tr2.terminated)
    rel = BarrierData.relaxed(p)
    D_rel0 = float(SD.D_of(st0.v(p), st0.z(p), st0.zd(p), rel))
    D0 = float(SD.D_of(st0.v(p), st0.z(p), st0.zd(p), data))
    passes = bool(viol["max"] <= tol and mem["in_XRF_tol"] and (not plan_timeout) and min_h_plan >= 0.0 and not plan_slack)
    return {"h0": float(st0.h(p)), "v0": float(st0.v(p)), "D0": D0, "D0_rel": D_rel0, "violations": viol,
            "max_violation": float(viol["max"]), "min_h_sim": float(min(tr.state_at(k).h(p) for k in range(tr.t.size))),
            "endpoint_gap": gap, "drift": tr.drift(), "H_end_sim": mem["H"], "H_end_col": float(res.H_end),
            "in_V": mem["in_V"], "in_XRF": mem["in_XRF"], "in_XRF_tol": mem["in_XRF_tol"],
            "plan_horizon": horizon, "min_h_plan": min_h_plan, "plan_slack": plan_slack,
            "plan_terminated_early": plan_early, "passes": passes, "timeout": "plan replay" if plan_timeout else None, "wall_cap": wall_cap,
            "refutation_candidate": bool(passes and st0.h(p) < D_rel0 - 0.01)}
