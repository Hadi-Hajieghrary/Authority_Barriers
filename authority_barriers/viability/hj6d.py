"""E3b (Sec. VI Q2, "for two cables, the six-dimensional state ... allows only a coarser grid, whose resolution
we report"): viability kernel of the planar two-cable system on the 6-D grid (h, v, z1, zd1, z2, zd2), set A2,
with the numerics of hj4d (WENO3 upwinding, local Lax-Friedrichs dissipation, TVD-RK2, freeze form
V_tau = min(0, max_u grad V . f), checkpoint/resume).

Dynamics (planar.planar2_rhs, a = (T1 q1 + T2 q2)/m_L): hdot = -v, vdot = -(T1 z1 + T2 z2)/m_L,
zddot_i = eta_i/(m_i l_i) - T_j (z_j - z_i q12)/(m_L l_i) - omega_i^2 z_i with q12 = q_1 . q_2 (the own tension
drops out of the swing coupling because P_i q_i = 0). The Hamiltonian is therefore linear in (T1, T2, eta1, eta2)
with state-dependent coefficients, and its maximum over the input set separates per cable once the set is
inner-approximated per cable (D-25, C-3): the thrust constraint ||u_i|| <= f_max is the ellipse
((T_i - Tc_i)/aT_i)^2 + (eta_i/aE_i)^2 <= 1 whose center is shifted by the other cable's tension through
s_i = k_i T_i + (m_i q12 / m_L) T_j - m_i l_i omega_i^2 (k_i = 1 + m_i/m_L); the intersection of the ellipses over
T_j in [T_min, T_cap] contains the ellipse centered at the mid-shift with the T semi-axis shrunk by half the shift
range, which is what is used; the D-4 cone ||T1 q1 + T2 q2|| <= m_L a_max is replaced by the caps
T_i <= T_cap = m_L a_max / 2 (exact for parallel cables). The kernel is under-approximated (h* biased upward).
"""
from __future__ import annotations

import math
import time
import warnings

import numpy as np
import torch

from ..theory.planar import PlanarParams
from .hj4d import (BYTES, DTYPE, Grid, KernelResult, _maybe_compile, cfl_time_step, estimate_resources, evolve,
                   one_sided_derivatives, peak_rss_gb)


def grid_planar6d(pp: PlanarParams, shape=(25, 25, 13, 13, 13, 13), h_max: float = 10.0, v_range=(-1.0, 4.0),
                  h_pad: float = 2.0) -> Grid:
    """(h, v, z1, zd1, z2, zd2): h on [-h_pad, h_max] (shape[0] nodes on [0, h_max] plus the pad), v in v_range,
    z_i in [-z_bar, z_bar], zd_i in [-zd_max, zd_max] with zd_max = omega_bar sqrt(1 - z_bar^2)."""
    zd_max = pp.omega_bar * math.sqrt(1.0 - pp.z_bar ** 2)
    h = Grid._h_axis(int(shape[0]), h_max, h_pad)
    v = np.linspace(v_range[0], v_range[1], int(shape[1]))
    z1 = np.linspace(-pp.z_bar, pp.z_bar, int(shape[2])); zd1 = np.linspace(-zd_max, zd_max, int(shape[3]))
    z2 = np.linspace(-pp.z_bar, pp.z_bar, int(shape[4])); zd2 = np.linspace(-zd_max, zd_max, int(shape[5]))
    return Grid((h, v, z1, zd1, z2, zd2), ("h", "v", "z1", "zd1", "z2", "zd2"))


class TwoCableInputs:
    """Per-cable inner approximation of U(x) (D-25) as ellipse-strip sets whose parameters are tensors over the
    swing axes, the support functions, and the per-node speed bounds for the CFL condition and the dissipation."""

    def __init__(self, grid: Grid, pp: PlanarParams):
        self.pp = pp
        z1, zd1, z2, zd2 = (grid.axis_tensor(i) for i in (2, 3, 4, 5))
        r1, r2 = torch.sqrt(1.0 - z1 * z1), torch.sqrt(1.0 - z2 * z2)
        q12 = z1 * z2 + r1 * r2                                    # (1,1,nz1,1,nz2,1)
        m, l, f, m_L = pp.m, pp.l, pp.f_max, pp.m_L
        self.T_min = float(pp.T_min)
        self.T_cap = float(pp.T_max / 2.0)                          # D-4 cone replaced by per-cable caps
        om1 = zd1 * zd1 / (1.0 - z1 * z1); om2 = zd2 * zd2 / (1.0 - z2 * z2)
        self.om = (om1, om2); self.z = (z1, z2); self.zd = (zd1, zd2); self.r = (r1, r2); self.q12 = q12
        self.k = [1.0 + m[i] / m_L for i in range(2)]
        self.Tc, self.aT, self.aE = [], [], []
        for i, (zi, omi, ri) in enumerate(((z1, om1, r1), (z2, om2, r2))):
            shift = m[i] * q12 / m_L                                 # coefficient of T_j in s_i
            delta = shift * (self.T_cap - self.T_min)                # range of the shift over T_j
            Tc = (m[i] * l[i] * omi - shift * 0.5 * (self.T_min + self.T_cap)) / self.k[i]
            aT = f[i] / self.k[i] - 0.5 * delta
            if float(aT.min()) <= 0.0:
                raise ValueError("the inner approximation of the thrust ellipse collapsed (aT <= 0)")
            self.Tc.append(Tc); self.aT.append(aT); self.aE.append(f[i] * ri)
        if not bool(torch.all(self.T_min <= self.Tc[0] + self.aT[0])) or not bool(torch.all(self.T_min <= self.Tc[1] + self.aT[1])):
            warnings.warn("the inner input set is empty at some nodes (fast swings); those nodes are lost states")

    def support(self, i: int, dT: torch.Tensor, dE: torch.Tensor) -> torch.Tensor:
        Tc, aT, aE = self.Tc[i], self.aT[i], self.aE[i]
        gT = dT * aT
        gE = dE * aE
        gn = torch.sqrt(gT * gT + gE * gE).clamp_(min=1e-300)
        Tb = (gT * (aT / gn)).add_(Tc).clamp_(self.T_min, self.T_cap)
        del gT, gE, gn
        rr = (Tb - Tc).div_(aT)
        Eb = torch.sqrt((1.0 - rr * rr).clamp_(min=0.0)).mul_(aE)
        del rr
        return Tb.mul_(dT).add_(Eb.mul_(dE.abs()))

    def max_speeds(self):
        """Per-node bounds on |vdot| and |zddot_i| over the inner input set."""
        pp = self.pp; m, l, m_L = pp.m, pp.l, pp.m_L
        z1, z2 = self.z; om1, om2 = self.om; q12 = self.q12
        vd = self.T_cap * (z1.abs() + z2.abs()) / m_L
        zdd1 = self.aE[0] / (m[0] * l[0]) + self.T_cap * (z2 - z1 * q12).abs() / (m_L * l[0]) + om1 * z1.abs()
        zdd2 = self.aE[1] / (m[1] * l[1]) + self.T_cap * (z1 - z2 * q12).abs() / (m_L * l[1]) + om2 * z2.abs()
        return vd, zdd1, zdd2


def initial_level_set_6d(grid: Grid, pp: PlanarParams, h_scale: float = 1.0, v_floor: float = 2.0, v_ceil: float = 0.5) -> torch.Tensor:
    h, v, z1, zd1, z2, zd2 = (grid.axis_tensor(i) for i in range(6))
    terms = [h / h_scale, (pp.z_bar - z1.abs()) / pp.z_bar, (pp.z_bar - z2.abs()) / pp.z_bar,
             (pp.omega_bar ** 2 * (1.0 - z1 * z1) - zd1 * zd1) / pp.omega_bar ** 2,
             (pp.omega_bar ** 2 * (1.0 - z2 * z2) - zd2 * zd2) / pp.omega_bar ** 2]
    V0 = terms[0].expand(grid.shape).contiguous()
    for t in terms[1:]:
        V0 = torch.minimum(V0, t.expand(grid.shape))
    return V0.clamp_(-v_floor, v_ceil)


def viability_kernel_6d(grid: Grid, pp: PlanarParams, cfl: float = 0.5, tol: float = 1e-4, t_max: float = 15.0,
                        verbose: bool = True, *, scheme: str = "weno3", window: float = 0.5, threads: int | None = None,
                        compile: bool | None = None, label: str = "hj6d", checkpoint=None, checkpoint_every: float = 600.0) -> KernelResult:
    if threads:
        torch.set_num_threads(int(threads))
    if grid.ndim != 6:
        raise ValueError("viability_kernel_6d needs a 6-D (h, v, z1, zd1, z2, zd2) grid")
    inputs = TwoCableInputs(grid, pp)
    pp_m, pp_l, m_L = pp.m, pp.l, float(pp.m_L)
    v = grid.axis_tensor(1)
    z1, z2 = inputs.z; zd1, zd2 = inputs.zd; om1, om2 = inputs.om; q12 = inputs.q12
    vd, zdd1, zdd2 = inputs.max_speeds()
    alphas = (v.abs(), vd, zd1.abs(), zdd1, zd2.abs(), zdd2)
    half = [0.5 * a for a in alphas]
    c_om1, c_om2 = om1 * z1, om2 * z2
    cT1_v, cT2_v = -z1 / m_L, -z2 / m_L                            # coefficients of T_i from vdot
    cT1_s, cT2_s = -(z1 - z2 * q12) / (m_L * pp_l[1]), -(z2 - z1 * q12) / (m_L * pp_l[0])   # from the other cable's swing
    inv1, inv2 = 1.0 / (pp_m[0] * pp_l[0]), 1.0 / (pp_m[1] * pp_l[1])
    dx = grid.dx
    clip = (-float(v_floor := 2.0), float(v_ceil := 0.5))
    dt = cfl_time_step(grid, alphas, cfl)

    def rhs(V):
        diss = None; pcs = []
        for ax in range(6):
            pm, pl = one_sided_derivatives(V, dx[ax], ax, scheme, clip)
            d = (pl - pm).mul_(half[ax])
            diss = d if diss is None else diss.add_(d)
            pcs.append(pm.add_(pl).mul_(0.5))
            del pl
        ph, pv, pz1, pzd1, pz2, pzd2 = pcs
        H = diss.sub_(ph * v).add_(pz1 * zd1).add_(pz2 * zd2).sub_(pzd1 * c_om1).sub_(pzd2 * c_om2)
        del ph, pz1, pz2
        H.add_(inputs.support(0, pv * cT1_v + pzd2 * cT1_s, pzd1 * inv1))
        H.add_(inputs.support(1, pv * cT2_v + pzd1 * cT2_s, pzd2 * inv2))
        return H.clamp_(max=0.0)

    V0 = initial_level_set_6d(grid, pp, 1.0, v_floor, v_ceil)
    res = estimate_resources(grid, n_arrays=26)
    if verbose:
        print(f"[{label}] grid {grid.shape} = {grid.size:,} nodes, {res['bytes_per_array'] / 1e6:.0f} MB per array, peak ~{res['peak_gb']:.2f} GB; "
              f"dt = {dt * 1e3:.3f} ms (CFL {cfl}), <= {math.ceil(t_max / dt)} steps to t_max = {t_max} s; scheme {scheme}; inner input set (D-25)", flush=True)
    do_compile = (grid.size >= 1_000_000) if compile is None else bool(compile)
    step_fn, compiled, t_compile = _maybe_compile(rhs, do_compile, V0, label, verbose)
    V, info = evolve(V0, step_fn, dt, t_max, tol, window, verbose, label, clip=clip, checkpoint=checkpoint, checkpoint_every=checkpoint_every)
    info.update({"scheme": scheme, "cfl": cfl, "tol": tol, "t_max": t_max, "window": window, "compiled": compiled, "compile_seconds": t_compile,
                 "threads": torch.get_num_threads(), "peak_rss_gb": peak_rss_gb(), "input_set": "per-cable inner approximation (D-25)",
                 "T_cap": inputs.T_cap, **res})
    return KernelResult(V.numpy(), grid, info["iterations"], info["converged"], info["wall_time"], info["history"], info["t_final"], dt,
                        info["window_changes"], info)
