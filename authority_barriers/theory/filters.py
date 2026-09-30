"""Safety filters of Sec. V as second-order cone programs on the shared scaffold (socp.ConeProgram).

ProposedFilter: eq. (17) with one barrier row per maximizing time (Prop. 17) and the swing
constraints either as the exact tangency rows (17d) (mode "tangency", active only on the
boundary of V) or, for the sampled 200 Hz implementation, as Remark 19's rule that every swing
state must remain in its admissible swing set until the next tick (mode "sampled"): with
s = zetadot + dt zetaddot and c = zeta_bar - zeta - zetadot dt - dt^2 zetaddot / 2 (both affine in
the decision variables), the condition (s)_+^2 <= 2 nu c is convex and is written with an
auxiliary sigma >= max(s, 0) and the rotated cone c * 2nu >= sigma^2; the lower boundary is
symmetric. As dt -> 0 the rule reduces to (17d).
HOCBFFilter: the high-order CBF baseline of Sec. III with the row b >= beta(x) and, optionally,
a class-K swing-rate row so that the hypothesis of Thm. 5 holds by construction.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from pydrake.solvers import ClarabelSolver, MathematicalProgram, SolutionResult

from . import altitude as ALT
from . import profile as PF
from . import stopping as SD
from .authority import BarrierData
from .maneuver import plan as plan_map
from .params import E3, Params
from .taut_model import rhs_from_thrust
from .socp import ConeProgram
from .state import State


@dataclass
class SwingAffine:
    """Affine maps x = (T, c) -> zddot_i, wddot_i, d/dt |qdot_i|^2 for every cable (Lemma 1(b))."""
    Az: np.ndarray   # (N, 3N)
    bz: np.ndarray   # (N,)
    Aw: np.ndarray
    bw: np.ndarray
    Ar: np.ndarray   # swing-rate derivative 2 qdot^T qddot
    br: np.ndarray


def swing_affine(cp: ConeProgram, p: Params, st: State) -> SwingAffine:
    N = p.N
    q, qd = st.q, st.qd
    y, r = p.y_vec, p.r_vec
    om2 = st.omega2()
    z, w = st.z(p), st.w(p)
    Q = q.T                                                  # a = Q T / m_L
    Az = np.zeros((N, 3 * N)); Aw = np.zeros((N, 3 * N)); Ar = np.zeros((N, 3 * N))
    for i in range(N):
        Pi = np.eye(3) - np.outer(q[i], q[i])
        Az[i, :N] = -(y @ Pi @ Q) / (p.l_arr[i] * p.m_L)
        Az[i, N + 2 * i:N + 2 * i + 2] = (y @ cp.B[i]) / (p.m_arr[i] * p.l_arr[i])
        Aw[i, :N] = -(r @ Pi @ Q) / (p.l_arr[i] * p.m_L)
        Aw[i, N + 2 * i:N + 2 * i + 2] = (r @ cp.B[i]) / (p.m_arr[i] * p.l_arr[i])
        Ar[i, :N] = -2.0 * (qd[i] @ Q) / (p.l_arr[i] * p.m_L)          # qd_i^T P_i = qd_i^T
        Ar[i, N + 2 * i:N + 2 * i + 2] = 2.0 * (qd[i] @ cp.B[i]) / (p.m_arr[i] * p.l_arr[i])
    return SwingAffine(Az, -om2 * z, Aw, -om2 * w, Ar, np.zeros(N))


@dataclass
class FilterResult:
    u: np.ndarray
    T: np.ndarray
    c: np.ndarray
    feasible: bool
    relaxed: bool
    slack: float
    status: str
    solve_time: float
    b: float
    H: float = np.nan
    D: float = np.nan
    t_star: np.ndarray = field(default_factory=lambda: np.zeros(0))
    hdot: np.ndarray = field(default_factory=lambda: np.zeros(0))
    in_V: bool = True
    clamped: bool = False
    rows: dict = field(default_factory=dict)
    H_up: float = np.nan        # altitude barriers (filters with altitude=True)
    H_down: float = np.nan


def _sampled_swing_rows(cp: ConeProgram, A_acc: np.ndarray, b_acc: float, zeta: float, zetad: float,
                        nu: float, zbar: float, dt: float):
    """Remark 19 one-step membership in V(nu, zbar) for one coordinate (two rotated cones)."""
    prog, x = cp.prog, cp.x
    n = x.size
    for sign in (+1.0, -1.0):
        # sign=+1: upper boundary; sign=-1: lower boundary (zeta -> -zeta, zetad -> -zetad)
        s_A = sign * dt * A_acc
        s_b = sign * (zetad + dt * b_acc)
        c_A = -sign * (dt ** 2 / 2.0) * A_acc
        c_b = zbar - sign * (zeta + zetad * dt + (dt ** 2 / 2.0) * b_acc)
        sigma = prog.NewContinuousVariables(1, "sigma")
        prog.AddBoundingBoxConstraint(0.0, np.inf, sigma)
        prog.AddLinearConstraint(np.concatenate([-s_A, [1.0]])[None, :], [s_b], [np.inf],
                                 np.concatenate([x, sigma]))                     # sigma >= s
        A = np.zeros((3, n + 1)); b = np.zeros(3)
        A[0, :n] = c_A; b[0] = c_b
        b[1] = 2.0 * nu
        A[2, n] = 1.0
        prog.AddRotatedLorentzConeConstraint(A, b, np.concatenate([x, sigma]))    # c * 2nu >= sigma^2


def _tangency_rows(cp: ConeProgram, A_acc, b_acc, zeta, zetad, nu, zbar, tol=1e-9, vtol=1e-9):
    """Exact (17d) rows, imposed only on the boundary of V (within tol)."""
    prog, x = cp.prog, cp.x
    lo, hi = PF.bounds_V(zetad, nu, zbar)
    active = {}
    if zeta >= hi - tol:                       # upper boundary or corner
        ub = -nu if zetad > vtol else 0.0
        prog.AddLinearConstraint(A_acc[None, :], [-np.inf], [ub - b_acc], x)
        active["upper"] = ub
    if zeta <= lo + tol:
        lb = nu if zetad < -vtol else 0.0
        prog.AddLinearConstraint(A_acc[None, :], [lb - b_acc], [np.inf], x)
        active["lower"] = lb
    return active


class ProposedFilter:
    """Filter (17): minimize sum |u_i - u_i^nom|^2 subject to U(x) (with the D-4 cone), the barrier
    rows Hdot_j >= -kappa_H H for every maximizing time, and the swing constraints."""

    def __init__(self, p: Params, data: BarrierData | None = None, kappa_H: float | None = None,
                 swing_mode: str = "sampled", dt: float | None = None, relax_on_infeasible: bool = True,
                 solver=None, hdot_margin: float = 0.0, rows: str = "maximizers", altitude: bool = False,
                 accel_bound: float | None = None):
        """altitude: add the rows of the altitude barriers H_up, H_down (theory/altitude.py); needs a set with a
        hover floor and the certified barrier data, which is then the default.
        accel_bound: factor c >= 1 of the rows |zddot_i| <= c nu and |wddot_i| <= c nu_w. The maneuver uses swing
        accelerations up to nu and nu_w (C3), so it satisfies the rows; they bound the change of the swing rates
        within one sampling period. c > 1 leaves room for the one-step rows of a swing state on the boundary of V.
        hdot_margin: additional slack delta on the barrier rows, Hdot_j >= -gamma(H) + delta (Rem. 19).
        rows: "maximizers" = the paper's filter (17c), one row per maximizing time of E (ties within
        1e-9); "local_maxima" = one row per local maximizer of E (D-21), which the braking maneuver
        also satisfies with equality (v~ = 0 there), so feasibility on X_RF is unchanged while a branch
        of E that overtakes the current maximum between two ticks is already constrained."""
        self.p = p
        self.altitude = bool(altitude)
        self.accel_bound = None if accel_bound is None else float(accel_bound)
        self.data = data or (BarrierData.certified(p) if altitude else BarrierData.nominal(p))
        self.kappa_H = p.kappa_H if kappa_H is None else kappa_H
        self.margin = float(hdot_margin)
        self.rows_mode = rows
        self.swing_mode = swing_mode
        self.dt = p.dt_filter if dt is None else dt
        self.relax = relax_on_infeasible
        self.solver = solver or ClarabelSolver()

    # ---- program assembly -------------------------------------------------------------
    def barrier_rows(self, st: State, res: SD.StoppingResult, sw: SwingAffine):
        """(A_j, b_j) with Hdot_j(x) = A_j x + b_j for each maximizing time t_j (eq. Hdot)."""
        p = self.p
        v = st.v(p)
        z, zd = st.z(p), st.zd(p)
        rows = []
        bT = np.zeros(3 * p.N); bT[:p.N] = z / p.m_L
        if self.rows_mode == "local_maxima" and res.t_lmax is not None:
            times, GZ, GW = res.t_lmax, res.grad_zeta_lmax, res.grad_omega_lmax
        else:
            times, GZ, GW = res.t_star, res.grad_zeta, res.grad_omega
        for j, tj in enumerate(times):
            gz, gw = GZ[j], GW[j]
            A = tj * bT - gw @ sw.Az
            b = -v + tj * self.data.offset - float(gz @ zd) - float(gw @ sw.bz)
            rows.append((A, b))
        return rows

    def build(self, st: State, u_nom: np.ndarray, res: SD.StoppingResult | None = None, slack: bool = False):
        p = self.p
        z, zd, w, wd = st.z(p), st.zd(p), st.w(p), st.wd(p)
        if res is None:
            res = SD.evaluate(st.v(p), z, zd, self.data)
        cp = ConeProgram(p, st)
        sw = swing_affine(cp, p, st)
        H = st.h(p) - res.D
        rows = self.barrier_rows(st, res, sw)
        self._alt = ALT.evaluate(st, p, res.maneuver, tol=4.0 * self.dt) if self.altitude and np.isfinite(res.D) else None
        alt_rows = ALT.rows(st, p, self._alt, sw.Az, sw.bz) if self._alt is not None else []
        sl = None
        if slack:
            sl = cp.prog.NewContinuousVariables(1, "slack")
            cp.prog.AddBoundingBoxConstraint(0.0, np.inf, sl)
            cp.prog.AddQuadraticCost(1e6 * sl[0] ** 2)
        for A, b in rows:
            lb = -self.kappa_H * H - b + self.margin
            if slack:
                cp.prog.AddLinearConstraint(np.concatenate([A, [1.0]])[None, :], [lb], [np.inf], np.concatenate([cp.x, sl]))
            else:
                cp.prog.AddLinearConstraint(A[None, :], [lb], [np.inf], cp.x)
        for A, b, Ha in alt_rows:
            if not np.any(A):                                  # no input in this row: the barrier does not depend on the input here
                continue
            lb = -p.kappa_alt * Ha - b
            if slack:
                cp.prog.AddLinearConstraint(np.concatenate([A, [1.0]])[None, :], [lb], [np.inf], np.concatenate([cp.x, sl]))
            else:
                cp.prog.AddLinearConstraint(A[None, :], [lb], [np.inf], cp.x)
        if self.accel_bound is not None:
            cz_, cw_ = self.accel_bound * self.data.nu, self.accel_bound * p.nu_w
            cp.prog.AddLinearConstraint(sw.Az, -cz_ - sw.bz, cz_ - sw.bz, cp.x)
            cp.prog.AddLinearConstraint(sw.Aw, -cw_ - sw.bw, cw_ - sw.bw, cp.x)
        active = {}
        for i in range(p.N):
            if self.swing_mode == "sampled":
                _sampled_swing_rows(cp, sw.Az[i], sw.bz[i], z[i], zd[i], self.data.nu, self.data.zeta_bar, self.dt)
                _sampled_swing_rows(cp, sw.Aw[i], sw.bw[i], w[i], wd[i], p.nu_w, p.w_bar, self.dt)
            elif self.swing_mode == "tangency":
                active[f"z{i}"] = _tangency_rows(cp, sw.Az[i], sw.bz[i], z[i], zd[i], self.data.nu, self.data.zeta_bar)
                active[f"w{i}"] = _tangency_rows(cp, sw.Aw[i], sw.bw[i], w[i], wd[i], p.nu_w, p.w_bar)
            elif self.swing_mode != "none":
                raise ValueError(self.swing_mode)
        A_all = cp.A.reshape(3 * p.N, 3 * p.N)
        b_all = cp.b.reshape(3 * p.N)
        cp.prog.Add2NormSquaredCost(A_all, np.asarray(u_nom, float).reshape(3 * p.N) - b_all, cp.x)
        return cp, sw, res, H, rows, sl, active

    def solve(self, st: State, u_nom: np.ndarray, warm_start: bool = True) -> FilterResult:
        p = self.p
        t0 = time.perf_counter()
        cp, sw, res, H, rows, _, active = self.build(st, u_nom)
        if warm_start:
            pl = plan_map(st, p, self.data)
            x0 = np.concatenate([pl.T, np.concatenate([cp.B[i].T @ pl.u_perp[i] for i in range(p.N)])])
            cp.prog.SetInitialGuess(cp.x, x0)
        r = self.solver.Solve(cp.prog)
        feasible, relaxed, slack, status = r.is_success(), False, 0.0, str(r.get_solution_result())
        if not feasible and self.relax:
            cp2, _, _, _, _, sl, _ = self.build(st, u_nom, res=res, slack=True)
            r2 = self.solver.Solve(cp2.prog)
            if r2.is_success():
                r, cp, relaxed, slack = r2, cp2, True, float(r2.GetSolution(sl)[0])
                status = f"relaxed({status})"
        if r.is_success():
            xv = r.GetSolution(cp.x)
            u = cp.thrusts(xv)
        else:
            pl = plan_map(st, p, self.data)
            u = pl.u
            xv = np.concatenate([pl.T, np.zeros(2 * p.N)])
            status = f"fallback_plan({status})"
        bT, b0 = cp.b_coeffs()
        hdot = np.array([A @ xv + b for A, b in rows])
        return FilterResult(u, xv[:p.N], xv[p.N:3 * p.N], feasible, relaxed, slack, status,
                            time.perf_counter() - t0, float(bT @ xv[:p.N] + b0), H, res.D, res.t_star, hdot,
                            res.in_V, res.clamped, {"active_tangency": active, "n_barrier_rows": len(rows)},
                            self._alt.H_up if self._alt is not None else np.nan, self._alt.H_down if self._alt is not None else np.nan)

    # ---- diagnostics -------------------------------------------------------------------
    def check_plan(self, st: State, tol: float = 1e-7) -> dict:
        """Does the braking maneuver satisfy every row of (17) at this state (Thm. 12(i))?
        Uses the exact tangency form of the swing constraints."""
        p = self.p
        res = SD.evaluate(st.v(p), st.z(p), st.zd(p), self.data)
        pl = plan_map(st, p, self.data)
        cp = ConeProgram(p, st)
        sw = swing_affine(cp, p, st)
        x = np.concatenate([pl.T, np.concatenate([cp.B[i].T @ pl.u_perp[i] for i in range(p.N)])])
        H = st.h(p) - res.D
        rows = self.barrier_rows(st, res, sw)
        hdot = np.array([A @ x + b for A, b in rows])
        out = {
            "H": H, "D": res.D, "hdot": hdot, "slack_barrier": float(np.min(hdot + self.kappa_H * H)),
            "barrier_ok": bool(np.all(hdot + self.kappa_H * H >= -tol)),
            "hdot_nonneg": bool(np.all(hdot >= -tol)),          # proof of Thm 12(i): Hdot(plan) = -max v~(t_j) >= 0
            "thrust_ok": bool(np.all(np.linalg.norm(cp.thrusts(x), axis=1) <= p.f_max_arr + tol)),
            "tension_ok": bool(np.all(pl.T >= p.T_min - tol)),
            "a_max_ok": bool(np.linalg.norm(pl.T @ st.q) <= p.m_L * p.a_max + tol),
            "zdd_matches": bool(np.allclose(sw.Az @ x + sw.bz, pl.zdd, atol=1e-8)),
            "wdd_matches": bool(np.allclose(sw.Aw @ x + sw.bw, pl.wdd, atol=1e-8)),
            "b_matches_alpha": abs((cp.b_coeffs()[0] @ pl.T + cp.b_coeffs()[1]) - self.data.alpha(st.z(p))) < 1e-9,
            "in_V_ok": bool(res.in_V and not res.clamped),
        }
        if self.altitude:
            alt = ALT.evaluate(st, p, res.maneuver)
            ar = np.array([A @ x + b + p.kappa_alt * Ha for A, b, Ha in ALT.rows(st, p, alt, sw.Az, sw.bz)])
            out.update({"H_up": alt.H_up, "H_down": alt.H_down, "slack_altitude": float(np.min(ar)), "altitude_ok": bool(np.all(ar >= -tol))})
        out["all_ok"] = all(v for k, v in out.items() if k.endswith("_ok") or k.endswith("matches") or k == "b_matches_alpha")
        return out

    def hdot_at(self, st: State, u: np.ndarray):
        """Hdot at the smallest maximizing time for the thrust vectors u applied at state st
        (eq. Hdot evaluated along a held input); returns (value, StoppingResult)."""
        p = self.p
        res = SD.evaluate(st.v(p), st.z(p), st.zd(p), self.data)
        if not np.isfinite(res.D):
            return np.nan, res
        cp = ConeProgram(p, st)
        sw = swing_affine(cp, p, st)
        out = rhs_from_thrust(st, np.asarray(u, float).reshape(p.N, 3), p)
        x = np.concatenate([out["T"], np.concatenate([cp.B[i].T @ out["u_perp"][i] for i in range(p.N)])])
        A, b = self.barrier_rows(st, res, sw)[0]
        return float(A @ x + b), res

    def probe_margin(self, st: State) -> float:
        """mu_H = max over U(x) of min_j (Hdot_j + kappa_H H): the filter is feasible iff mu_H >= 0
        (swing constraints excluded; used to classify solver failures)."""
        p = self.p
        res = SD.evaluate(st.v(p), st.z(p), st.zd(p), self.data)
        cp = ConeProgram(p, st)
        sw = swing_affine(cp, p, st)
        H = st.h(p) - res.D
        m = cp.prog.NewContinuousVariables(1, "m")
        for A, b in self.barrier_rows(st, res, sw):
            cp.prog.AddLinearConstraint(np.concatenate([A, [-1.0]])[None, :], [-b - self.kappa_H * H], [np.inf],
                                        np.concatenate([cp.x, m]))
        cp.prog.AddLinearCost(-m[0])
        r = self.solver.Solve(cp.prog)
        return float(r.GetSolution(m)[0]) if r.is_success() else -np.inf


class HOCBFFilter:
    """High-order CBF baseline (Sec. III): b(T) >= beta(x) over U(x); optional class-K
    swing-rate rows 2 qdot_i^T qddot_i <= -k_w (|qdot_i|^2 - omega_bar^2)."""

    def __init__(self, p: Params, a1, a2, swing_rate_row: bool = True, k_w: float = 5.0,
                 relax_on_infeasible: bool = True, solver=None):
        self.p, self.a1, self.a2 = p, a1, a2
        self.swing_rate_row, self.k_w, self.relax = swing_rate_row, k_w, relax_on_infeasible
        self.solver = solver or ClarabelSolver()

    def beta(self, st: State) -> float:
        h, v = st.h(self.p), st.v(self.p)
        return self.a1.deriv(h) * v - self.a2(-v + self.a1(h))

    def build(self, st: State, u_nom, slack: bool = False):
        p = self.p
        cp = ConeProgram(p, st)
        bT, b0 = cp.b_coeffs()
        beta = self.beta(st)
        A = np.zeros(3 * p.N); A[:p.N] = bT
        sl = None
        if slack:
            sl = cp.prog.NewContinuousVariables(1, "slack")
            cp.prog.AddBoundingBoxConstraint(0.0, np.inf, sl)
            cp.prog.AddQuadraticCost(1e6 * sl[0] ** 2)
            cp.prog.AddLinearConstraint(np.concatenate([A, [1.0]])[None, :], [beta - b0], [np.inf], np.concatenate([cp.x, sl]))
        else:
            cp.prog.AddLinearConstraint(A[None, :], [beta - b0], [np.inf], cp.x)
        if self.swing_rate_row:
            sw = swing_affine(cp, p, st)
            om2 = st.omega2()
            for i in range(p.N):
                cp.prog.AddLinearConstraint(sw.Ar[i][None, :], [-np.inf], [-self.k_w * (om2[i] - p.omega_bar ** 2) - sw.br[i]], cp.x)
        A_all = cp.A.reshape(3 * p.N, 3 * p.N)
        b_all = cp.b.reshape(3 * p.N)
        cp.prog.Add2NormSquaredCost(A_all, np.asarray(u_nom, float).reshape(3 * p.N) - b_all, cp.x)
        return cp, beta, sl

    def solve(self, st: State, u_nom) -> FilterResult:
        p = self.p
        t0 = time.perf_counter()
        cp, beta, _ = self.build(st, u_nom)
        r = self.solver.Solve(cp.prog)
        feasible, relaxed, slack, status = r.is_success(), False, 0.0, str(r.get_solution_result())
        if not feasible and self.relax:
            cp2, _, sl = self.build(st, u_nom, slack=True)
            r2 = self.solver.Solve(cp2.prog)
            if r2.is_success():
                r, cp, relaxed, slack = r2, cp2, True, float(r2.GetSolution(sl)[0])
                status = f"relaxed({status})"
        if r.is_success():
            xv = r.GetSolution(cp.x)
            u = cp.thrusts(xv)
        else:
            u = np.asarray(u_nom, float)
            xv = np.full(3 * p.N, np.nan)
            status = f"fallback_nominal({status})"
        bT, b0 = cp.b_coeffs()
        bval = float(bT @ xv[:p.N] + b0) if np.all(np.isfinite(xv)) else np.nan
        return FilterResult(u, xv[:p.N], xv[p.N:3 * p.N], feasible, relaxed, slack, status,
                            time.perf_counter() - t0, bval, rows={"beta": beta})


def decomposition_terms(p: Params, st: State, res: SD.StoppingResult, a_hat: np.ndarray, j: int = 0):
    """Prop. 17(b): with a common a_hat, Hdot_j = const + sum_i local_i(T_i, c_i). Returns
    (const, [(A_i (3,), b_i)]) where local_i = A_i . (T_i, c_i1, c_i2) + b_i, for tangent basis B_i."""
    y = p.y_vec
    tj = res.t_star[j]
    gz, gw = res.grad_zeta[j], res.grad_omega[j]
    z, zd = st.z(p), st.zd(p)
    om2 = st.omega2()
    B = [np.stack(_basis(st.q[i]), axis=1) for i in range(p.N)]
    const = -st.v(p) + tj * (-p.g * float(E3 @ y))
    locals_ = []
    for i in range(p.N):
        Pi = np.eye(3) - np.outer(st.q[i], st.q[i])
        zdd_const = -(y @ Pi @ a_hat) / p.l_arr[i] - om2[i] * z[i]
        A = np.concatenate([[tj * z[i] / p.m_L], -gw[i] * (y @ B[i]) / (p.m_arr[i] * p.l_arr[i])])
        b = -gz[i] * zd[i] - gw[i] * zdd_const
        locals_.append((A, b))
    return const, locals_, B


def _basis(q):
    from .socp import tangent_basis
    Bm = tangent_basis(q)
    return Bm[:, 0], Bm[:, 1]


@dataclass
class NumericStoppingResult:
    """What ProposedFilter.barrier_rows needs, obtained by integrating the maneuver (BackupIntegratedFilter)."""
    D: float
    t_star: np.ndarray
    grad_zeta: np.ndarray        # (1, N)
    grad_omega: np.ndarray       # (1, N)
    grad_v: float
    in_V: bool = True
    clamped: bool = False
    t_lmax: object = None
    n_breakpoints: int = 0
    n_integrations: int = 0
    n_policy_calls: int = 0
    timeouts: int = 0
    eval_time: float = 0.0


class BackupIntegratedFilter(ProposedFilter):
    """Backup-CBF baseline (Rem. 16, Sec. VI Q3): the same program as (17) but with the barrier data obtained
    numerically: D_num(x) = h(0) - min_t h(t) along the braking maneuver integrated on the taut-cable model
    (RK45, rtol 1e-8), the maximizing time from the integrated profile, and the gradients in (v, z_i, zdot_i) by
    central differences (2(1 + 2N) + 1 integrations per evaluation). Each integration is capped in wall time; a
    timed-out evaluation falls back to the closed form and is counted."""

    def __init__(self, p: Params, data: BarrierData | None = None, rtol: float = 1e-8, atol: float = 1e-10,
                 delta: float = 1e-4, wall_cap: float = 20.0, plan_extra: float = 0.5, **kw):
        super().__init__(p, data, **kw)
        self.rtol, self.atol, self.delta, self.wall_cap, self.plan_extra = rtol, atol, delta, wall_cap, plan_extra
        self.stats = {"evaluations": 0, "integrations": 0, "policy_calls": 0, "timeouts": 0, "fallbacks": 0, "eval_time": 0.0}

    def _D_num(self, st_base: State, v: float, z: np.ndarray, zd: np.ndarray):
        from .taut_model import integrate_capped
        p, data = self.p, self.data
        st = State.from_swing(p, 20.0, v, z, st_base.w(p), zd, st_base.wd(p), altitude=float(st_base.x_L[2]))
        _, _, t2 = PF.switching(z, zd, data.nu, data.zeta_bar)
        horizon = float(np.max(t2)) + max(v, 0.0) / data.alpha_sat + self.plan_extra
        policy = lambda t, s: plan_map(s, p, data).u
        tr, timed_out, calls = integrate_capped(st, policy, horizon, p, self.wall_cap, dt_out=1e-3, rtol=self.rtol, atol=self.atol,
                                                terminate_on_slack=False)
        if timed_out:
            return np.nan, np.nan, calls
        h = np.array([tr.state_at(k).h(p) for k in range(tr.t.size)])
        k = int(np.argmin(h))
        return float(h[0] - h[k]), float(tr.t[k]), calls

    def evaluate(self, st: State) -> NumericStoppingResult:
        p = self.p
        t0 = time.perf_counter()
        v, z, zd = st.v(p), st.z(p), st.zd(p)
        d = self.delta
        n_int, calls, timeouts = 0, 0, 0
        D0, ts, c = self._D_num(st, v, z, zd); n_int += 1; calls += c; timeouts += int(np.isnan(D0))
        Dp, _, c = self._D_num(st, v + d, z, zd); n_int += 1; calls += c; timeouts += int(np.isnan(Dp))
        Dm, _, c = self._D_num(st, v - d, z, zd); n_int += 1; calls += c; timeouts += int(np.isnan(Dm))
        gv = (Dp - Dm) / (2 * d)
        gz, gw = np.zeros(p.N), np.zeros(p.N)
        for i in range(p.N):
            for arr, g in ((z, gz), (zd, gw)):
                a = arr.copy(); a[i] += d; Dp, _, c = self._D_num(st, v, a if arr is z else z, a if arr is zd else zd); calls += c; timeouts += int(np.isnan(Dp))
                a = arr.copy(); a[i] -= d; Dm, _, c = self._D_num(st, v, a if arr is z else z, a if arr is zd else zd); calls += c; timeouts += int(np.isnan(Dm))
                n_int += 2
                g[i] = (Dp - Dm) / (2 * d)
        dt = time.perf_counter() - t0
        self.stats["evaluations"] += 1; self.stats["integrations"] += n_int; self.stats["policy_calls"] += calls
        self.stats["timeouts"] += timeouts; self.stats["eval_time"] += dt
        return NumericStoppingResult(D0, np.array([ts]), gz[None, :], gw[None, :], gv, n_integrations=n_int, n_policy_calls=calls,
                                     timeouts=timeouts, eval_time=dt)

    def build(self, st: State, u_nom: np.ndarray, res=None, slack: bool = False):
        if res is None:
            res = self.evaluate(st)
            self.last_res = res
            if not np.isfinite(res.D) or not np.all(np.isfinite(res.grad_zeta)) or not np.all(np.isfinite(res.grad_omega)):
                self.stats["fallbacks"] += 1
                res = SD.evaluate(st.v(self.p), st.z(self.p), st.zd(self.p), self.data)
        return super().build(st, u_nom, res=res, slack=slack)


class _Local:
    """A one-vehicle program exposing .prog and .x, so that the swing-row helpers apply unchanged."""

    def __init__(self, prog, x):
        self.prog, self.x = prog, x


class DistributedProposedFilter(ProposedFilter):
    """Distributed implementation of Prop. 17(b) (D-27): every vehicle solves its own program with the broadcast
    a_hat = the previous tick's commanded specific force a(T) (one-step lag), the coupled barrier requirement
    Hdot_0 >= -kappa_H H is split into N local rows local_i(T_i, c_i) >= theta_i whose right-hand sides sum to
    the requirement; theta_i = l_i^nom + lambda (m_i - l_i^nom) with l_i^nom the nominal command's local value,
    m_i the vehicle's largest achievable local value over its own constraints, and lambda in [0, 1] the single
    shared number (the one-dimensional consensus on the sum). Given a_hat the thrust of vehicle i is affine in its
    own variables (Lemma 1(d)), so the thrust norm and the sampled swing rows are local; the D-4 cone is replaced
    by the caps T_i <= m_L a_max / N."""

    def __init__(self, p: Params, data: BarrierData | None = None, **kw):
        super().__init__(p, data, **kw)
        self.a_hat = None
        self.last = {}

    def _local_program(self, st: State, i: int, a_hat: np.ndarray, u_nom_i: np.ndarray):
        p = self.p
        q, qd = st.q[i], st.qd[i]
        om2 = float(qd @ qd)
        y, r = p.y_vec, p.r_vec
        Bm = np.stack(_basis(q), axis=1)                                   # (3, 2)
        Pi = np.eye(3) - np.outer(q, q)
        s0 = p.m_arr[i] * float(q @ a_hat) - p.m_arr[i] * p.l_arr[i] * om2   # s_i = T_i + s0 given a_hat
        A_u = np.column_stack([q, Bm]); b_u = q * s0                        # u_i = A_u x + b_u, x = (T_i, c_i)
        prog = MathematicalProgram()
        x = prog.NewContinuousVariables(3, f"x{i}")
        prog.AddBoundingBoxConstraint([p.T_min, -np.inf, -np.inf], [p.m_L * p.a_max / p.N, np.inf, np.inf], x)
        prog.AddLorentzConeConstraint(np.vstack([np.zeros((1, 3)), A_u]), np.concatenate([[p.f_max_arr[i]], b_u]), x)
        loc = _Local(prog, x)
        # swing rows given a_hat: zddot_i = (y . B_i c_i)/(m_i l_i) - y . P_i a_hat / l_i - om2 z_i (and w likewise)
        for vec, zeta, zetad, nu, zbar in ((y, float(q @ y), float(qd @ y), self.data.nu, self.data.zeta_bar),
                                          (r, float(q @ r), float(qd @ r), p.nu_w, p.w_bar)):
            A_acc = np.concatenate([[0.0], (vec @ Bm) / (p.m_arr[i] * p.l_arr[i])])
            b_acc = -float(vec @ Pi @ a_hat) / p.l_arr[i] - om2 * zeta
            if self.swing_mode == "sampled":
                _sampled_swing_rows(loc, A_acc, b_acc, zeta, zetad, nu, zbar, self.dt)
        return prog, x, A_u, b_u

    def solve(self, st: State, u_nom: np.ndarray, warm_start: bool = True) -> FilterResult:
        p = self.p
        t0 = time.perf_counter()
        u_nom = np.asarray(u_nom, float).reshape(p.N, 3)
        res = SD.evaluate(st.v(p), st.z(p), st.zd(p), self.data)
        H = st.h(p) - res.D
        nom = rhs_from_thrust(st, u_nom, p)
        if self.a_hat is None:                                              # first tick: the nominal's own specific force
            self.a_hat = nom["a"].copy()
        a_hat = self.a_hat
        if not np.isfinite(res.D):
            pl = plan_map(st, p, self.data)
            self.a_hat = rhs_from_thrust(st, pl.u, p)["a"]
            return FilterResult(pl.u, pl.T, np.zeros(2 * p.N), False, False, 0.0, "fallback_plan(D undefined)", time.perf_counter() - t0, np.nan, H, res.D,
                                res.t_star, np.zeros(0), res.in_V, res.clamped, {"n_barrier_rows": 0})
        const, locals_, B = decomposition_terms(p, st, res, a_hat, j=0)
        R = -self.kappa_H * H - const + self.margin                          # sum_i local_i >= R
        progs, caps, lnom = [], [], []
        for i in range(p.N):
            prog, x, A_u, b_u = self._local_program(st, i, a_hat, u_nom[i])
            A_i, b_i = locals_[i]
            x_nom = np.concatenate([[nom["T"][i]], B[i].T @ nom["u_perp"][i]])
            lnom.append(float(A_i @ x_nom + b_i))
            # capacity: the largest local value over the vehicle's own constraints
            cap_prog, cap_x, _, _ = self._local_program(st, i, a_hat, u_nom[i])
            cap_prog.AddLinearCost(-A_i, 0.0, cap_x)
            rc = self.solver.Solve(cap_prog)
            caps.append(float(A_i @ rc.GetSolution(cap_x) + b_i) if rc.is_success() else -np.inf)
            progs.append((prog, x, A_u, b_u, A_i, b_i))
        caps = np.array(caps); lnom = np.array(lnom)
        spare = np.maximum(caps - lnom, 0.0)
        need = R - float(np.sum(lnom))
        feasible_alloc = bool(np.all(np.isfinite(caps)) and float(np.sum(caps)) >= R - 1e-9)
        lam = 0.0 if need <= 0.0 else (float(np.clip(need / max(float(np.sum(spare)), 1e-12), 0.0, 1.0)) if np.sum(spare) > 0 else 1.0)
        theta = lnom + lam * spare if feasible_alloc else caps                   # infeasible: everyone at capacity
        u = np.zeros((p.N, 3)); T = np.zeros(p.N); c = np.zeros(2 * p.N)
        feasible, statuses = feasible_alloc, []
        for i, (prog, x, A_u, b_u, A_i, b_i) in enumerate(progs):
            prog.AddLinearConstraint(A_i[None, :], [theta[i] - b_i - 1e-9], [np.inf], x)
            prog.Add2NormSquaredCost(A_u, u_nom[i] - b_u, x)
            r = self.solver.Solve(prog)
            if r.is_success():
                xv = r.GetSolution(x)
            else:                                                            # relax the local row (D-15): closest feasible command
                prog2, x2, A_u2, b_u2 = self._local_program(st, i, a_hat, u_nom[i])
                sl = prog2.NewContinuousVariables(1, "sl"); prog2.AddBoundingBoxConstraint(0.0, np.inf, sl)
                prog2.AddQuadraticCost(1e6 * sl[0] ** 2)
                prog2.AddLinearConstraint(np.concatenate([A_i, [1.0]])[None, :], [theta[i] - b_i], [np.inf], np.concatenate([x2, sl]))
                prog2.Add2NormSquaredCost(A_u2, u_nom[i] - b_u2, x2)
                r2 = self.solver.Solve(prog2)
                xv = r2.GetSolution(x2) if r2.is_success() else np.concatenate([[nom["T"][i]], B[i].T @ nom["u_perp"][i]])
                feasible = False
            statuses.append(str(r.get_solution_result()))
            u[i] = A_u @ xv + b_u; T[i] = xv[0]; c[2 * i:2 * i + 2] = xv[1:]
        out = rhs_from_thrust(st, u, p)                                         # the true specific force of the applied command
        a_true = out["a"]
        a_err = float(np.linalg.norm(a_true - a_hat))
        self.a_hat = a_true.copy()                                             # broadcast for the next tick (one-step lag)
        # the coupled row evaluated with the true coupling, for the log
        cp = ConeProgram(p, st)
        sw = swing_affine(cp, p, st)
        x_full = np.concatenate([out["T"], np.concatenate([cp.B[i].T @ out["u_perp"][i] for i in range(p.N)])])
        A0, b0 = self.barrier_rows(st, res, sw)[0]
        hdot = float(A0 @ x_full + b0)
        bT, bb = cp.b_coeffs()
        self.last = {"lambda": lam, "a_hat_err": a_err, "R": R, "caps": caps.tolist(), "lnom": lnom.tolist(), "theta": theta.tolist()}
        return FilterResult(u, out["T"], c, feasible, not feasible, 0.0, "distributed:" + ("ok" if feasible else "relaxed"),
                            time.perf_counter() - t0, float(bT @ out["T"] + bb), H, res.D, res.t_star, np.array([hdot]), res.in_V, res.clamped,
                            {"n_barrier_rows": 1, "lambda": lam, "a_hat_err": a_err, "a_hat_err_force": float(np.max(p.m_arr) * a_err)})


class CableCBFBaseline(HOCBFFilter):
    """E7 baseline: the HOCBF filter on h plus independent per-quadrotor cable barriers: second-order class-K rows
    on the box constraints z_bar -/+ z_i and w_bar -/+ w_i (relative degree two in the thrust) with the same gain pair
    (b_dd + (k1 + k2) b_d + k1 k2 b >= 0), and the swing-rate row of the HOCBF baseline."""

    def build(self, st: State, u_nom, slack: bool = False):
        cp, beta, sl = super().build(st, u_nom, slack)
        p = self.p
        sw = swing_affine(cp, p, st)
        k1, k2 = self.a1.deriv(0.0), self.a2.deriv(0.0)                    # linear gains (D-11)
        z, zd, w, wd = st.z(p), st.zd(p), st.w(p), st.wd(p)
        for i in range(p.N):
            for A_acc, b_acc, zeta, zetad, zbar in ((sw.Az[i], sw.bz[i], z[i], zd[i], self.data_zbar), (sw.Aw[i], sw.bw[i], w[i], wd[i], p.w_bar)):
                for sign in (+1.0, -1.0):                                  # b = zbar - sign * zeta >= 0
                    # b_dd = -sign zeta_dd = -sign (A_acc x + b_acc); row: -sign (A x + b_acc) + (k1 + k2)(-sign zetad) + k1 k2 (zbar - sign zeta) >= 0
                    rhs = -(-sign * b_acc - (k1 + k2) * sign * zetad + k1 * k2 * (zbar - sign * zeta))
                    cp.prog.AddLinearConstraint((-sign * A_acc)[None, :], [rhs], [np.inf], cp.x)
        return cp, beta, sl

    @property
    def data_zbar(self) -> float:
        return self.p.z_bar


class MultiWallFilter:
    """The proposed filter for several walls at once (E7 corridor): one cone program (thrust, tension, D-4 rows of the
    first wall's parameter set, identical for all walls) with the barrier rows and swing rows of every wall
    (each wall has its own approach direction y, hence its own z_i, D and H)."""

    def __init__(self, params: list[Params], kappa_H: float | None = None, swing_mode: str = "sampled", solver=None):
        self.walls = [ProposedFilter(pw, BarrierData.nominal(pw), kappa_H=kappa_H, swing_mode=swing_mode) for pw in params]
        self.p = params[0]
        self.data = self.walls[0].data
        self.solver = solver or ClarabelSolver()

    def build(self, st: State, u_nom, slack: bool = False):
        p = self.p
        cp = ConeProgram(p, st)
        sl = None
        if slack:
            sl = cp.prog.NewContinuousVariables(1, "slack")
            cp.prog.AddBoundingBoxConstraint(0.0, np.inf, sl)
            cp.prog.AddQuadraticCost(1e6 * sl[0] ** 2)
        info = []
        for wf in self.walls:
            pw = wf.p
            z, zd, w, wd = st.z(pw), st.zd(pw), st.w(pw), st.wd(pw)
            res = SD.evaluate(st.v(pw), z, zd, wf.data)
            H = st.h(pw) - res.D
            sw = swing_affine(cp, pw, st)
            rows = wf.barrier_rows(st, res, sw) if np.isfinite(res.D) else []
            for A, b in rows:
                lb = -wf.kappa_H * H - b
                if slack:
                    cp.prog.AddLinearConstraint(np.concatenate([A, [1.0]])[None, :], [lb], [np.inf], np.concatenate([cp.x, sl]))
                else:
                    cp.prog.AddLinearConstraint(A[None, :], [lb], [np.inf], cp.x)
            if wf.swing_mode == "sampled":
                for i in range(pw.N):
                    _sampled_swing_rows(cp, sw.Az[i], sw.bz[i], z[i], zd[i], wf.data.nu, wf.data.zeta_bar, wf.dt)
                    _sampled_swing_rows(cp, sw.Aw[i], sw.bw[i], w[i], wd[i], pw.nu_w, pw.w_bar, wf.dt)
            info.append({"res": res, "H": H, "h": st.h(pw), "rows": rows})
        A_all = cp.A.reshape(3 * p.N, 3 * p.N); b_all = cp.b.reshape(3 * p.N)
        cp.prog.Add2NormSquaredCost(A_all, np.asarray(u_nom, float).reshape(3 * p.N) - b_all, cp.x)
        return cp, info, sl

    def solve(self, st: State, u_nom, warm_start: bool = True) -> FilterResult:
        p = self.p
        t0 = time.perf_counter()
        cp, info, _ = self.build(st, u_nom)
        r = self.solver.Solve(cp.prog)
        feasible, relaxed, slack, status = r.is_success(), False, 0.0, str(r.get_solution_result())
        if not feasible:
            cp2, info2, sl = self.build(st, u_nom, slack=True)
            r2 = self.solver.Solve(cp2.prog)
            if r2.is_success():
                r, cp, relaxed, slack, status = r2, cp2, True, float(r2.GetSolution(sl)[0]), f"relaxed({status})"
        if r.is_success():
            xv = r.GetSolution(cp.x); u = cp.thrusts(xv)
        else:
            u = np.asarray(u_nom, float); xv = np.full(3 * p.N, np.nan); status = f"fallback_nominal({status})"
        k = int(np.argmin([w["H"] for w in info]))                               # the most critical wall
        res = info[k]["res"]
        hdot = np.array([A @ xv + b for A, b in info[k]["rows"]]) if np.all(np.isfinite(xv)) else np.zeros(0)
        bT, b0 = cp.b_coeffs()
        return FilterResult(u, xv[:p.N], xv[p.N:3 * p.N], feasible, relaxed, slack, status, time.perf_counter() - t0,
                            float(bT @ xv[:p.N] + b0) if np.all(np.isfinite(xv)) else np.nan, info[k]["H"], res.D, res.t_star, hdot, res.in_V, res.clamped,
                            {"n_barrier_rows": sum(len(w["rows"]) for w in info), "H_walls": [w["H"] for w in info], "h_walls": [w["h"] for w in info],
                             "critical_wall": k})
