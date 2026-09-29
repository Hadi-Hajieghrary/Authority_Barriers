"""C-7 cross-check of the level-set solver with an independent implementation (hj_reachability, JAX):
the same viability problem (same nodes, same initial level set, same exact support function of U(z, zd),
same per-node dissipation bounds) discretized by the toolbox's WENO5 upwinding, TVD-RK3 and Lax-Friedrichs
flux, solved backward in time with the `backwards_reachable_tube` Hamiltonian postprocessor (min(H, 0)),
which is the freeze form V_tau = min(0, max_u grad V . f) of hj4d. Compared: the kernel boundary h*(v, z, zd)
at the 200 probe states of E3 and the sign of V node by node.

  python -m authority_barriers.viability.crosscheck_hj --validate          # 2-D double integrator against the closed form
  python -m authority_barriers.viability.crosscheck_hj --grid 61 --threads 2
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np


def _setup_threads(n: int):
    os.environ.setdefault("XLA_FLAGS", f"--xla_cpu_multi_thread_eigen={'true' if n > 1 else 'false'} intra_op_parallelism_threads={n}")


def make_dynamics_4d(pp):
    import hj_reachability as hj
    import jax.numpy as jnp
    m, l, m_L, f = float(pp.m[0]), float(pp.l[0]), float(pp.m_L), float(pp.f_max[0])
    k = 1.0 + m / m_L
    T_min, T_max = float(pp.T_min), float(pp.T_max)

    def ellipse(z, zd):
        c = m * l * zd * zd / (1.0 - z * z)
        return c / k, f / k, f * jnp.sqrt(1.0 - z * z)

    class Planar1(hj.Dynamics):
        """h' = -v, v' = -T z / m_L, z' = zd, zd' = eta rho / (m l) - z zd^2 / (1 - z^2), (T, eta) in U(z, zd)."""

        def __init__(self):
            super().__init__("max", "min", hj.sets.Box(jnp.array([T_min, -f]), jnp.array([T_max, f])), hj.sets.Box(jnp.zeros(1), jnp.zeros(1)))

        def __call__(self, state, control, disturbance, time):
            h, v, z, zd = state
            T, eta = control
            rho = jnp.sqrt(1.0 - z * z)
            return jnp.array([-v, -T * z / m_L, zd, eta * rho / (m * l) - z * zd * zd / (1.0 - z * z)])

        def _maximizer(self, state, grad_value):
            """argmax over U(z, zd) of p . f: the support-function maximizer of hj4d.EllipseStripInputs."""
            h, v, z, zd = state
            p_v, p_zd = grad_value[1], grad_value[3]
            Tc, aT, aE = ellipse(z, zd)
            dT = -p_v * z / m_L
            dE = p_zd * jnp.sqrt(1.0 - z * z) / (m * l)
            gT, gE = dT * aT, dE * aE
            gn = jnp.maximum(jnp.sqrt(gT * gT + gE * gE), 1e-300)
            Tb = jnp.clip(Tc + gT * aT / gn, T_min, T_max)
            r = (Tb - Tc) / aT
            Eb = aE * jnp.sqrt(jnp.maximum(1.0 - r * r, 0.0)) * jnp.sign(dE)
            return jnp.array([Tb, Eb])

        def optimal_control_and_disturbance(self, state, time, grad_value):
            return self._maximizer(state, grad_value), jnp.zeros(1)

        def hamiltonian(self, state, time, value, grad_value):
            u = self._maximizer(state, grad_value)
            return grad_value @ self(state, u, jnp.zeros(1), time)

        def partial_max_magnitudes(self, state, time, value, grad_value_box):
            h, v, z, zd = state
            Tc, aT, aE = ellipse(z, zd)
            T_top = jnp.minimum(T_max, Tc + aT)
            T_c = jnp.clip(Tc, T_min, T_max)
            eta_top = aE * jnp.sqrt(jnp.clip(1.0 - ((T_c - Tc) / aT) ** 2, 0.0, None))
            return jnp.array([jnp.abs(v), T_top * jnp.abs(z) / m_L, jnp.abs(zd), eta_top / (m * l) + zd * zd * jnp.abs(z) / (1.0 - z * z)])

    return Planar1()


def make_dynamics_2d(a_brake: float):
    import hj_reachability as hj
    import jax.numpy as jnp

    class DoubleIntegrator(hj.Dynamics):
        def __init__(self):
            super().__init__("max", "min", hj.sets.Box(jnp.array([-a_brake]), jnp.array([a_brake])), hj.sets.Box(jnp.zeros(1), jnp.zeros(1)))

        def __call__(self, state, control, disturbance, time):
            return jnp.array([-state[1], -control[0]])

        def optimal_control_and_disturbance(self, state, time, grad_value):
            return jnp.array([-a_brake * jnp.sign(grad_value[1])]), jnp.zeros(1)

        def partial_max_magnitudes(self, state, time, value, grad_value_box):
            return jnp.array([jnp.abs(state[1]), a_brake])

    return DoubleIntegrator()


def hj_grid(grid):
    import hj_reachability as hj
    import jax.numpy as jnp
    lo = jnp.array([a[0] for a in grid.axes]); hi = jnp.array([a[-1] for a in grid.axes])
    g = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(lo, hi), grid.shape,
                                                                boundary_conditions=tuple(hj.boundary_conditions.extrapolate for _ in grid.axes))
    for a, c in zip(grid.axes, g.coordinate_vectors):
        assert np.allclose(np.asarray(c), a, atol=1e-9), "hj grid nodes differ from the hj4d grid"
    return g


def solve(dynamics, g, V0, tol=1e-4, window=0.5, t_max=8.0, clip=(-2.0, 0.5), verbose=True):
    import hj_reachability as hj
    import jax.numpy as jnp
    settings = hj.SolverSettings.with_accuracy("very_high", hamiltonian_postprocessor=hj.solver.backwards_reachable_tube,
                                               value_postprocessor=lambda t, v: jnp.clip(v, clip[0], clip[1]))
    V = jnp.asarray(V0)
    t, hist = 0.0, []
    t0 = time.perf_counter()
    converged = False
    while t > -t_max + 1e-12:
        Vn = hj.step(settings, dynamics, g, t, V, t - window, progress_bar=False)
        Vn.block_until_ready()
        change = float(jnp.max(jnp.abs(Vn - V)))
        t -= window
        V = Vn
        hist.append((round(-t, 3), change))
        if verbose:
            print(f"[hj] tau = {-t:.2f} s  max|V(tau) - V(tau - {window})| = {change:.2e}  elapsed {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
        if change < tol:
            converged = True
            break
    return np.asarray(V), {"converged": converged, "t_final": -t, "wall": time.perf_counter() - t0, "window_changes": hist,
                           "scheme": "WENO5 + TVD-RK3 + LF (hj_reachability 0.7.0), CFL 0.75"}


def validate_2d():
    from authority_barriers.viability.hj4d import Grid, KernelResult, boundary_h, initial_level_set
    import torch
    grid = Grid.double_integrator(shape=(61, 61), h_max=20.0, v_range=(-1.0, 4.0), h_pad=2.0)
    h, v = grid.axis_tensor(0), grid.axis_tensor(1)
    V0 = torch.minimum(h, torch.full_like(h, 0.5)).expand(grid.shape).contiguous().clamp_(-2.0, 0.5).numpy()
    V, info = solve(make_dynamics_2d(3.0), hj_grid(grid), V0, verbose=False)
    res = KernelResult(V, grid, -1, info["converged"], info["wall"], np.zeros(0))
    vv = np.linspace(0.2, 3.8, 20)
    err = boundary_h(res, vv) - vv ** 2 / 6.0
    print(f"2-D double integrator (hj_reachability): converged {info['converged']} at tau {info['t_final']:.1f} s, {info['wall']:.0f} s; "
          f"max |h* - v^2/(2a)| = {np.nanmax(np.abs(err)):.3f} m = {np.nanmax(np.abs(err)) / grid.dx[0]:.2f} cells (M4.1 target <= 1 cell)")
    return float(np.nanmax(np.abs(err)) / grid.dx[0])


def crosscheck_4d(grid_id: int, out: Path):
    from authority_barriers.theory import stopping as SD
    from authority_barriers.theory.authority import BarrierData
    from authority_barriers.theory.params import load_set
    from authority_barriers.theory.planar import PlanarParams
    from authority_barriers.experiments.e3_kernel import SHAPES, probe_states
    from authority_barriers.viability.hj4d import Grid, KernelResult, boundary_h, initial_level_set
    p = load_set("A1"); pp = PlanarParams.from_params(p)
    grid = Grid.planar4d(pp, shape=SHAPES[grid_id], h_max=20.0, v_range=(-1.0, 4.0))
    V0 = initial_level_set(grid, pp).numpy()
    print(f"[hj] grid {grid.shape} = {grid.size:,} nodes", flush=True)
    V, info = solve(make_dynamics_4d(pp), hj_grid(grid), V0)
    res = KernelResult(V, grid, -1, info["converged"], info["wall"], np.zeros(0))
    mine = KernelResult.load(out / f"e3_kernel_{grid_id}.npz")
    assert mine.V.shape == V.shape
    S = probe_states(p); speeds = np.linspace(0.5, 4.0, 10)
    nom, rel = BarrierData.nominal(p), BarrierData.relaxed(p)
    rows = []
    for z, zd in S:
        for v in speeds:
            hs_hj = float(boundary_h(res, np.array([v]), np.array([z]), np.array([zd]))[0])
            hs_me = float(boundary_h(mine, np.array([v]), np.array([z]), np.array([zd]))[0])
            rows.append({"z": float(z), "zd": float(zd), "v": float(v), "h_hj": hs_hj, "h_hj4d": hs_me,
                         "D": SD.D_of(v, np.array([z]), np.array([zd]), nom), "D_rel": SD.D_of(v, np.array([z]), np.array([zd]), rel)})
    d = np.array([r["h_hj"] - r["h_hj4d"] for r in rows]); ok = np.isfinite(d)
    cell = grid.dx[0]
    sign_diff = float(np.mean((V >= 0) != (mine.V >= 0)))
    verdict = {"n_probes": int(ok.sum()), "max_abs_dh": float(np.max(np.abs(d[ok]))), "median_abs_dh": float(np.median(np.abs(d[ok]))),
               "max_abs_dh_cells": float(np.max(np.abs(d[ok])) / cell), "hj_minus_hj4d_median": float(np.median(d[ok])),
               "within_one_cell": bool(np.all(np.abs(d[ok]) <= cell)), "sign_mismatch_fraction": sign_diff,
               "max_abs_dV": float(np.max(np.abs(V - mine.V))),
               "hj_lower_bound_holds": bool(np.all(np.array([r["h_hj"] for r in rows])[ok] >= np.array([r["D_rel"] for r in rows])[ok] - cell)),
               "hj_upper_bound_holds": bool(np.all(np.array([r["h_hj"] for r in rows])[ok] <= np.array([r["D"] for r in rows])[ok] + cell)),
               "cell_m": cell}
    print(json.dumps(verdict, indent=1))
    out.mkdir(parents=True, exist_ok=True)
    (out / f"crosscheck_hj_{grid_id}.json").write_text(json.dumps({"grid": grid_id, "shape": list(grid.shape), "info": info, "verdict": verdict, "rows": rows}, indent=1))
    np.savez_compressed(out / f"crosscheck_hj_{grid_id}.npz", V=V)
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--grid", type=int, default=None)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    _setup_threads(args.threads)
    if args.validate:
        validate_2d()
    if args.grid:
        out = Path(os.environ.get("SIM_RESULTS_DIR", Path(__file__).resolve().parents[2] / "results")).resolve() / "core" / "e3_kernel"
        crosscheck_4d(args.grid, out)


if __name__ == "__main__":
    main()
