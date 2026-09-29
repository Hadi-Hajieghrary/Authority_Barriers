"""M4.1: validation of the level-set solver on problems with closed-form viability kernels.

(a) double integrator hdot = -v, vdot = -b, b in [-a, a]: kernel {h >= v_+^2 / (2a)};
(b) planar one-cable system with frozen cables (eta = 0, zd = 0 slices): for z > 0 every input brakes,
    the largest braking is a_max z (T = m_L a_max), so the kernel in the slice is {h >= v_+^2 / (2 a_max z)};
    for z < 0 every input pushes toward the wall and the slice is empty for v >= 0.
"""
from __future__ import annotations

import time

import numpy as np

from authority_barriers.theory.params import load_set
from authority_barriers.theory.planar import PlanarParams
from .hj4d import Grid, boundary_h, kernel_2d_double_integrator, viability_kernel_4d


def validate_2d(shape=(101, 101), a_brake: float = 3.0, verbose: bool = False) -> dict:
    grid = Grid.double_integrator(shape=shape, h_max=20.0, v_range=(-1.0, 4.0), h_pad=2.0)
    t0 = time.perf_counter()
    res = kernel_2d_double_integrator(grid, a_brake, verbose=verbose)
    v = np.linspace(0.2, 3.8, 20)
    hstar = boundary_h(res, v)
    exact = np.maximum(v, 0.0) ** 2 / (2.0 * a_brake)
    err = hstar - exact
    return {"shape": shape, "dx_h": float(grid.dx[0]), "max_err_m": float(np.nanmax(np.abs(err))),
            "max_err_cells": float(np.nanmax(np.abs(err)) / grid.dx[0]), "mean_err_m": float(np.nanmean(err)),
            "converged": res.converged, "iterations": res.iterations, "wall": time.perf_counter() - t0,
            "hstar": hstar, "exact": exact}


def validate_4d_frozen(shape=(41, 41, 11, 11), verbose: bool = False) -> dict:
    p = load_set("A1")
    pp = PlanarParams.from_params(p)
    grid = Grid.planar4d(pp, shape=shape, h_max=20.0, v_range=(-1.0, 4.0))
    t0 = time.perf_counter()
    res = viability_kernel_4d(grid, pp, verbose=verbose, support="frozen")
    v = np.linspace(0.5, 3.5, 7)
    zs = np.linspace(0.05, pp.z_bar, 5)
    rows = []
    for z in zs:
        hstar = boundary_h(res, v, np.full_like(v, z), np.zeros_like(v))
        exact = v ** 2 / (2.0 * p.a_max * z)
        rows.append({"z": float(z), "max_err_m": float(np.nanmax(np.abs(hstar - exact))),
                     "max_err_cells": float(np.nanmax(np.abs(hstar - exact)) / grid.dx[0]), "hstar": hstar, "exact": exact})
    # z < 0 slices: every state with v >= 0 is lost
    lost = []
    for z in (-0.05, -0.15):
        hs = boundary_h(res, v, np.full_like(v, z), np.zeros_like(v))
        lost.append({"z": z, "fraction_nan_or_large": float(np.mean(~np.isfinite(hs) | (hs > 15.0)))})
    return {"shape": shape, "dx_h": float(grid.dx[0]), "rows": rows, "lost": lost, "converged": res.converged,
            "iterations": res.iterations, "wall": time.perf_counter() - t0}


def run_all(verbose: bool = False) -> dict:
    out = {"2d": [validate_2d((61, 61), verbose=verbose), validate_2d((101, 101), verbose=verbose)],
           "4d_frozen": validate_4d_frozen(verbose=verbose)}
    return out


if __name__ == "__main__":
    out = run_all()
    print("2-D double integrator (M4.1a):")
    for r in out["2d"]:
        print(f"  grid {r['shape']}: max |h* - exact| = {r['max_err_m']:.4f} m = {r['max_err_cells']:.2f} cells "
              f"(mean signed {r['mean_err_m']:+.4f} m), converged {r['converged']} in {r['iterations']} steps, {r['wall']:.1f} s")
    f = out["4d_frozen"]
    print(f"4-D frozen cables (M4.1b), grid {f['shape']}, dx_h {f['dx_h']:.3f} m, converged {f['converged']} in {f['iterations']} steps, {f['wall']:.1f} s:")
    for r in f["rows"]:
        print(f"  z = {r['z']:.3f}: max |h* - v^2/(2 a_max z)| = {r['max_err_m']:.4f} m = {r['max_err_cells']:.2f} cells")
    for r in f["lost"]:
        print(f"  z = {r['z']:.2f}: fraction of (v >= 0.5) probes lost = {r['fraction_nan_or_large']:.2f} (expected 1.0)")
