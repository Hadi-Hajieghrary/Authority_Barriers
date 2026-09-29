"""Taut-cable point-mass model of Sec. II (eqs. load, veh) in the structured form of Lemma 1:
eq. tension (a), eq. cable (b), eq. closure (c). Pure numpy; every right-hand side is
dtype-agnostic (only +, -, *, /, np.sqrt and reductions; no np.linalg) so the same code runs
on float arrays and on object arrays of pydrake AutoDiffXd (collocation, Prop. 17 gradients).

State layout (pack/unpack): x = [x_L (3), v_L (3), q (3N, row-major (N, 3)), qdot (3N)].
Cable directions q_i point from the payload to quadrotor i, p_i = x_L + l_i q_i; the specific
force of the payload is a := xddot_L + g e3 (Sec. II-B). The sphere identities ||q_i|| = 1 and
q_i . qdot_i = 0 are invariants of the ambient ODE (Lemma 1(b): q_i . qddot_i = -||qdot_i||^2);
`integrate` renormalizes the *returned samples* only as numerical hygiene and `drift` measures
the raw violation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp

from .params import E3, Params
from .state import State

Policy = Callable[[float, State], np.ndarray]           # policy(t, st) -> thrusts u (N, 3)
DistL = Callable[[float, State], np.ndarray]            # d_L_fn(t, st) -> (3,)
DistQ = Callable[[float, State], np.ndarray]            # d_fn(t, st) -> (N, 3)


# ---------------------------------------------------------------- state packing
def n_state(N: int) -> int:
    """Dimension 6 + 6N of the packed state."""
    return 6 + 6 * N


def pack(st: State) -> np.ndarray:
    """State -> packed vector [x_L, v_L, q.ravel(), qd.ravel()] (dtype of the fields)."""
    return _pack(st.x_L, st.v_L, st.q, st.qd)


def _pack(x_L, v_L, q, qd) -> np.ndarray:
    return np.concatenate([np.asarray(x_L).reshape(-1), np.asarray(v_L).reshape(-1),
                           np.asarray(q).reshape(-1), np.asarray(qd).reshape(-1)])


def unpack(x: np.ndarray, N: int) -> State:
    """Packed vector (6 + 6N,) -> State with q, qd of shape (N, 3) (views when possible)."""
    x = np.asarray(x).reshape(-1)
    if x.shape[0] != n_state(N):
        raise ValueError(f"packed state has length {x.shape[0]}, expected {n_state(N)} for N = {N}")
    return State(x[0:3], x[3:6], x[6:6 + 3 * N].reshape(N, 3), x[6 + 3 * N:6 + 6 * N].reshape(N, 3))


# ---------------------------------------------------------------- dtype-agnostic helpers
def _rowdot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise inner products of two (N, 3) arrays -> (N,); works on object arrays."""
    return (a * b).sum(axis=-1)


def solve3(M: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve the 3x3 system M x = b by the adjugate (cofactor) formula x = adj(M) b / det(M).

    Uses only +, -, *, / on the entries, so it is dtype-agnostic (float, AutoDiffXd, ...).
    Intended for the well-conditioned closure matrix M_a = m_L I + sum_i m_i q_i q_i^T of
    Lemma 1(c), whose condition number is at most 1 + sum_i m_i / m_L.
    """
    m00, m01, m02 = M[0, 0], M[0, 1], M[0, 2]
    m10, m11, m12 = M[1, 0], M[1, 1], M[1, 2]
    m20, m21, m22 = M[2, 0], M[2, 1], M[2, 2]
    # cofactors
    c00 = m11 * m22 - m12 * m21
    c01 = -(m10 * m22 - m12 * m20)
    c02 = m10 * m21 - m11 * m20
    c10 = -(m01 * m22 - m02 * m21)
    c11 = m00 * m22 - m02 * m20
    c12 = -(m00 * m21 - m01 * m20)
    c20 = m01 * m12 - m02 * m11
    c21 = -(m00 * m12 - m02 * m10)
    c22 = m00 * m11 - m01 * m10
    det = m00 * c00 + m01 * c01 + m02 * c02
    b0, b1, b2 = b[0], b[1], b[2]
    # adj(M) = C^T
    x0 = (c00 * b0 + c10 * b1 + c20 * b2) / det
    x1 = (c01 * b0 + c11 * b1 + c21 * b2) / det
    x2 = (c02 * b0 + c12 * b1 + c22 * b2) / det
    return np.array([x0, x1, x2])


def _disturbances(N: int, d_L, d):
    d_L = np.zeros(3) if d_L is None else np.asarray(d_L).reshape(3)
    d = np.zeros((N, 3)) if d is None else np.asarray(d).reshape(N, 3)
    return d_L, d


def _cable_accel(q, om2, u_perp, a, d, m, l):
    """eq. cable: qddot_i = u_i^perp/(m_i l_i) - P_i a / l_i - ||qdot_i||^2 q_i + P_i d_i/(m_i l_i)."""
    qa = q @ a                                    # (N,)  q_i . a
    P_a = a[None, :] - qa[:, None] * q            # P_i a
    P_d = d - _rowdot(q, d)[:, None] * q          # P_i d_i
    ml = (m * l)[:, None]
    return u_perp / ml - P_a / l[:, None] - om2[:, None] * q + P_d / ml


# ---------------------------------------------------------------- right-hand sides (Lemma 1)
def rhs_from_thrust(st: State, u: np.ndarray, p: Params, d_L=None, d=None) -> dict:
    """Accelerations from the thrust vectors u (N, 3): Lemma 1 with the closure solve.

    s_i = q_i . u_i, u_i^perp = P_i u_i; a solves eq. closure
        M_a a = sum_i s_i q_i + sum_i m_i l_i ||qdot_i||^2 q_i + sum_i (q_i . d_i) q_i + d_L,
        M_a = m_L I + sum_i m_i q_i q_i^T   (3x3 SPD, adjugate solve);
    then eq. tension T_i = s_i - m_i q_i . a + m_i l_i ||qdot_i||^2 + q_i . d_i, eq. cable for
    qddot_i, and xddot_L = a - g e3. Returns dict(a, T, qdd, xdd, s, u_perp). Dtype-agnostic.
    """
    q, qd = st.q, st.qd
    N = q.shape[0]
    m, l = p.m_arr, p.l_arr
    u = np.asarray(u).reshape(N, 3)
    d_L, d = _disturbances(N, d_L, d)
    s = _rowdot(q, u)                                       # parallel components
    u_perp = u - s[:, None] * q                             # P_i u_i
    om2 = _rowdot(qd, qd)                                   # ||qdot_i||^2
    qdi = _rowdot(q, d)                                     # q_i . d_i
    M_a = p.m_L * np.eye(3) + (m[:, None] * q).T @ q        # m_L I + sum m_i q_i q_i^T
    rhs = (s + m * l * om2 + qdi) @ q + d_L                 # sum_i (...) q_i + d_L
    a = solve3(M_a, rhs)                                    # eq. closure
    T = s - m * (q @ a) + m * l * om2 + qdi                 # eq. tension
    qdd = _cable_accel(q, om2, u_perp, a, d, m, l)          # eq. cable
    xdd = a - p.g * E3
    return dict(a=a, T=T, qdd=qdd, xdd=xdd, s=s, u_perp=u_perp)


def rhs_from_tensions(st: State, T: np.ndarray, u_perp: np.ndarray, p: Params, d_L=None, d=None) -> dict:
    """Accelerations from the tensions T (N,) and perpendicular thrusts u_perp (N, 3), which must be
    tangent (q_i . u_i^perp = 0; not projected here).

    a = (sum_j T_j q_j + d_L) / m_L is explicit (eq. load); eq. tension inverted for the parallel
    component s_i = T_i + m_i q_i . a - m_i l_i ||qdot_i||^2 - q_i . d_i; u_i = s_i q_i + u_i^perp;
    qddot_i by eq. cable; xddot_L = a - g e3. Returns dict(a, T, qdd, xdd, s, u). Dtype-agnostic
    (this is the path used with AutoDiffXd in the collocation program).
    """
    q, qd = st.q, st.qd
    N = q.shape[0]
    m, l = p.m_arr, p.l_arr
    T = np.asarray(T).reshape(N)
    u_perp = np.asarray(u_perp).reshape(N, 3)
    d_L, d = _disturbances(N, d_L, d)
    om2 = _rowdot(qd, qd)
    qdi = _rowdot(q, d)
    a = (T @ q + d_L) / p.m_L                               # eq. load with a = xdd_L + g e3
    s = T + m * (q @ a) - m * l * om2 - qdi                 # eq. tension solved for s_i
    u = s[:, None] * q + u_perp
    qdd = _cable_accel(q, om2, u_perp, a, d, m, l)          # eq. cable
    xdd = a - p.g * E3
    return dict(a=a, T=T, qdd=qdd, xdd=xdd, s=s, u=u)


def f_ambient(x: np.ndarray, u_flat: np.ndarray, p: Params, d_L=None, d=None) -> np.ndarray:
    """Time derivative of the packed state under thrusts u (flat (3N,) or (N, 3)); scipy-style
    signature. Uses rhs_from_thrust (eqs. closure, tension, cable). Dtype-agnostic."""
    st = unpack(x, p.N)
    out = rhs_from_thrust(st, np.asarray(u_flat).reshape(p.N, 3), p, d_L, d)
    return _pack(st.v_L, out["xdd"], st.qd, out["qdd"])


# ---------------------------------------------------------------- sphere invariants
def drift(X: np.ndarray) -> tuple[float, float]:
    """Sphere-invariant violation of raw samples X (K, 6 + 6N) or (6 + 6N,):
    (max_k,i | ||q_i|| - 1 |, max_k,i | q_i . qdot_i |)."""
    X = np.atleast_2d(np.asarray(X, float))
    N = (X.shape[1] - 6) // 6
    q = X[:, 6:6 + 3 * N].reshape(-1, N, 3)
    qd = X[:, 6 + 3 * N:].reshape(-1, N, 3)
    nq = np.sqrt((q * q).sum(-1))
    return float(np.max(np.abs(nq - 1.0))), float(np.max(np.abs((q * qd).sum(-1))))


def project_samples(X: np.ndarray) -> np.ndarray:
    """Numerical hygiene for returned samples: q_i <- q_i/||q_i||, qdot_i <- qdot_i - (q_i . qdot_i) q_i."""
    X = np.array(np.atleast_2d(X), float, copy=True)
    N = (X.shape[1] - 6) // 6
    q = X[:, 6:6 + 3 * N].reshape(-1, N, 3)
    qd = X[:, 6 + 3 * N:].reshape(-1, N, 3)
    q /= np.sqrt((q * q).sum(-1))[..., None]
    qd -= (q * qd).sum(-1)[..., None] * q
    X[:, 6:6 + 3 * N] = q.reshape(X.shape[0], -1)
    X[:, 6 + 3 * N:] = qd.reshape(X.shape[0], -1)
    return X


# ---------------------------------------------------------------- integration
@dataclass
class Trajectory:
    """Sampled solution of the taut ODE. X are the returned samples (projected iff requested),
    X_raw the raw integrator output, T (K, N) the tensions of eq. tension recomputed at the
    samples, U (K, N, 3) the thrusts the policy applied there."""
    t: np.ndarray
    X: np.ndarray
    T: np.ndarray
    U: np.ndarray
    X_raw: np.ndarray
    N: int
    slack_time: Optional[float]              # first time any T_i < 0 (event or sample), None if never
    floor_violation_time: Optional[float]    # first time any T_i < T_min, None if never
    terminated: bool                         # stopped by the slack event before t_f
    solver_message: str = ""

    def state_at(self, k: int) -> State:
        return unpack(self.X[k], self.N)

    def drift(self) -> tuple[float, float]:
        """Sphere-invariant drift of the raw integrator output (see `drift`)."""
        return drift(self.X_raw)

    @property
    def min_tension(self) -> float:
        return float(np.min(self.T))


def integrate(st0: State, policy: Policy, t_f: float, p: Params, dt_out: float = 1e-3,
              rtol: float = 1e-10, atol: float = 1e-10, project: bool = True,
              d_L_fn: Optional[DistL] = None, d_fn: Optional[DistQ] = None,
              terminate_on_slack: bool = True, method: str = "DOP853", t0: float = 0.0) -> Trajectory:
    """Integrate the ambient taut ODE (Lemma 1 via rhs_from_thrust) from st0 over [t0, t0 + t_f]
    with scipy.integrate.solve_ivp under `policy(t, st) -> u (N, 3)` and optional disturbances
    `d_L_fn(t, st) -> (3,)`, `d_fn(t, st) -> (N, 3)`.

    Samples every dt_out. A solve_ivp event on min_i T_i(t, x) (eq. tension) locates the first
    slack instant and, if terminate_on_slack, stops there (the event state is appended as the
    last sample). A second, non-terminal event locates the first crossing of the floor T_min.
    The policy is also evaluated inside the event functions, so it should be a pure function of
    (t, state). project=True renormalizes q_i and removes the radial part of qdot_i in the
    returned samples X only; X_raw and `Trajectory.drift()` keep the raw output.
    """
    N = p.N
    x0 = pack(st0).astype(float)
    T_min = p.T_min

    def evaluate(t, x):
        st = unpack(x, N)
        u = np.asarray(policy(t, st), float).reshape(N, 3)
        d_L = None if d_L_fn is None else d_L_fn(t, st)
        d = None if d_fn is None else d_fn(t, st)
        out = rhs_from_thrust(st, u, p, d_L, d)
        return out, u, st

    def fun(t, x):
        out, _, st = evaluate(t, x)
        return _pack(st.v_L, out["xdd"], st.qd, out["qdd"])

    def ev_slack(t, x):
        return float(np.min(evaluate(t, x)[0]["T"]))

    def ev_floor(t, x):
        return float(np.min(evaluate(t, x)[0]["T"])) - T_min

    ev_slack.terminal = bool(terminate_on_slack)
    ev_slack.direction = -1
    ev_floor.terminal = False
    ev_floor.direction = -1

    # slack at the initial state: the model is invalid from the start
    T0 = evaluate(t0, x0)[0]["T"]
    if terminate_on_slack and np.any(T0 < 0):
        t = np.array([t0]); Y = x0[None, :]
        return _finish(t, Y, N, evaluate, project, T_min, slack_time=float(t0), terminated=True,
                       floor_events=np.array([]), msg="slack at t0")

    n_out = int(np.floor(t_f / dt_out + 1e-9)) + 1
    t_eval = t0 + dt_out * np.arange(n_out)
    t_eval[-1] = min(t_eval[-1], t0 + t_f)
    if t_eval[-1] < t0 + t_f - 1e-12 * max(1.0, abs(t_f)):
        t_eval = np.append(t_eval, t0 + t_f)
    sol = solve_ivp(fun, (t0, t0 + t_f), x0, method=method, t_eval=t_eval, rtol=rtol, atol=atol,
                    events=[ev_slack, ev_floor])
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    t, Y = sol.t, sol.y.T
    slack_events = sol.t_events[0]
    terminated = bool(sol.status == 1)
    if terminated:                                    # append the event state as the last sample
        te, ye = sol.t_events[0][0], sol.y_events[0][0]
        if t.size == 0 or te > t[-1] + 1e-15:
            t = np.append(t, te); Y = np.vstack([Y, ye[None, :]])
    slack_time = float(slack_events[0]) if slack_events.size else None
    return _finish(t, Y, N, evaluate, project, T_min, slack_time, terminated, sol.t_events[1], sol.message)


def _finish(t, Y, N, evaluate, project, T_min, slack_time, terminated, floor_events, msg) -> Trajectory:
    K = t.shape[0]
    T = np.empty((K, N)); U = np.empty((K, N, 3))
    for k in range(K):
        out, u, _ = evaluate(t[k], Y[k])
        T[k] = out["T"]; U[k] = u
    # sample-based detections complement the event locations (a violation present at t0 has no crossing)
    slack_k = np.flatnonzero(np.any(T < 0.0, axis=1))
    floor_k = np.flatnonzero(np.any(T < T_min, axis=1))
    cands = [] if slack_time is None else [slack_time]
    if slack_k.size:
        cands.append(float(t[slack_k[0]]))
    slack_time = min(cands) if cands else None
    cands = [float(x) for x in np.asarray(floor_events).reshape(-1)]
    if floor_k.size:
        cands.append(float(t[floor_k[0]]))
    if slack_time is not None:
        cands.append(slack_time)
    floor_time = min(cands) if cands else None
    X = project_samples(Y) if project else Y
    return Trajectory(t=t, X=X, T=T, U=U, X_raw=Y, N=N, slack_time=slack_time,
                      floor_violation_time=floor_time, terminated=terminated, solver_message=str(msg))


class IntegrationTimeout(Exception):
    pass


def integrate_capped(st, policy, horizon, p, wall_cap: float, **kw):
    """`integrate` under a wall-clock cap enforced from inside the policy callback (the braking maneuver's
    bang-bang swing law can chatter and stall an error-controlled integrator); returns (trajectory or None,
    timed_out, n_policy_calls)."""
    import time as _time
    t0 = _time.perf_counter()
    calls = [0]

    def pol(t, st_):
        calls[0] += 1
        if _time.perf_counter() - t0 > wall_cap:
            raise IntegrationTimeout()
        return policy(t, st_)

    try:
        return integrate(st, pol, horizon, p, **kw), False, calls[0]
    except IntegrationTimeout:
        return None, True, calls[0]
