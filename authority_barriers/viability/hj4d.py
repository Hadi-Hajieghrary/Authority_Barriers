"""Level-set computation of the viability kernel of the planar one-cable system (E3, Thm. 12(ii)).

State x = (h, v, z, zd) of plan K1 (`authority_barriers.theory.planar`):
    hdot = -v,  vdot = -T z / m_L,  zdot = zd,  zddot = eta / (m l) - zd^2 z / (1 - z^2),
inputs (T, eta) in the convex, state-dependent set U(z, zd) (ellipse of the thrust limit cut by the
tension strip T_min <= T <= m_L a_max). Constraint set
    C = { h >= 0,  |z| <= z_bar,  zd^2 <= omega_bar^2 (1 - z^2) }.

Level-set method ("freeze" form, Mitchell et al. 2005; ToolboxLS `termRestrictUpdate`):
    V(x, 0) = V0(x) = clip( min( h / h_scale, (z_bar - |z|) / z_bar, (omega_bar^2 (1 - z^2) - zd^2) / omega_bar^2 ),
                            -v_floor, v_ceil ),
    V_tau = min( 0, H(x, grad V) ),      H(x, p) = max_{u in U(z, zd)} p . f(x, u),
so V is nonincreasing in pseudo-time tau, {V(., tau) >= 0} is the set of states that can be kept in C
for tau seconds, and Viab(C) = {V_inf >= 0}. The sign convention is the forward-pseudo-time form of
"dV/dt + min(0, H) = 0" with t = -tau; one checks it on xdot = 1, V0 = -x, where V(x, tau) = -(x + tau)
requires V_tau = -1 = min(0, H) with H = p f = -1. The clipping does not change the kernel, because
V(x, tau) = max_u min_{t <= tau} V0(x(t)) commutes with the monotone maps min(., v_ceil) and
max(., -v_floor); it bounds V so that lost states converge to the floor instead of decreasing forever,
and it shortens the transient far from the wall, where V trades cable margin against braking.

Grid: the h axis is extended BELOW the wall by h_pad meters (at the requested spacing), because the
value of a lost state is the penetration depth (h - D)/h_scale that the optimal control still allows,
and this negative value must be represented on the grid: extrapolating V from h >= 0, where V is capped
by the cable terms, cannot create it (that failure mode makes the kernel far too large).

Hamiltonian:  H = -p_h v + p_z zd - p_zd z zd^2 / (1 - z^2) + sigma_U(z, zd)(-p_v z / m_L, p_zd / (m l)),
where sigma_U(d) = max_{u in U} d . u is the support function of U. Because U is an ellipse cut by a
vertical strip, sigma_U has a closed form (`EllipseStripInputs`), which the solver evaluates by default
("exact"). The K-point inner approximation of plan K2 (max over the K precomputed boundary points of
`planar1_input_boundary`, shape (nz, nzd, K, 2)) is available as `support="points"` and used for
cross-checks; on the 4-D grids it needs O(nodes x K) temporaries (25 GB of traffic per evaluation on
the finest grid), whereas the closed form costs a dozen elementwise passes and is exact.

Numerics: WENO3 (default) or ENO2 one-sided derivatives on every axis with two ghost cells by linear
extrapolation (clipped to [-v_floor, v_ceil]); local Lax-Friedrichs Hamiltonian
H(p_c) + sum_i alpha_i (p_i^+ - p_i^-)/2 with p_c = (p^- + p^+)/2 and the per-node dissipation
coefficients alpha_h = |v|, alpha_v = max_U |vdot|, alpha_z = |zd|, alpha_zd = max_U |zddot|
(`planar1_max_speeds`); the min(0, .) is applied to the whole LLF Hamiltonian, so V never increases
(ToolboxLS convention); TVD-RK2 (Heun) in pseudo-time with dt = cfl / max_nodes sum_i alpha_i / dx_i;
convergence when max |V(tau) - V(tau - window)| < tol (window = 0.5 s), or at tau = t_max. Everything
is vectorized in torch float64 on the CPU (optionally fused by torch.compile); no Python loop touches
grid nodes.

Sign of the numerical bias: LF dissipation, the inner approximation of U (if used) and the finite
horizon all shrink the computed kernel (raise the boundary h*), so a computed h* below D_rel cannot be
an artifact of them, while a computed h* slightly above D is expected on coarse grids. The
extrapolation boundary condition at the top of the v axis is the exception (mildly optimistic for
states at v = v_max that are still being accelerated toward the wall, z < 0); see `Grid.planar4d`.
"""
from __future__ import annotations

import itertools
import math
import os
import resource
import time
import warnings
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import torch

from ..theory.planar import (PlanarParams, _ellipse, planar1_input_boundary, planar1_input_nonempty,
                                planar1_max_speeds)

DTYPE = torch.float64
BYTES = 8
COMPILE_MIN_NODES = 1_000_000


# ---------------------------------------------------------------- grid
@dataclass(frozen=True, eq=False)
class Grid:
    """Regular Cartesian grid: `axes[i]` is the 1-D float64 node array of axis i (axis 0 is h)."""
    axes: tuple
    names: tuple = ()

    def __post_init__(self):
        axes = tuple(np.asarray(a, float) for a in self.axes)
        object.__setattr__(self, "axes", axes)
        if not self.names:
            object.__setattr__(self, "names", tuple(f"x{i}" for i in range(len(axes))))
        for a in axes:
            if a.ndim != 1 or a.size < 3 or not np.all(np.diff(a) > 0):
                raise ValueError("every axis needs at least 3 increasing nodes")
            if not np.allclose(np.diff(a), a[1] - a[0], rtol=1e-9, atol=1e-12):
                raise ValueError("axes must be uniformly spaced")

    @property
    def ndim(self) -> int:
        return len(self.axes)

    @property
    def shape(self) -> tuple:
        return tuple(int(a.size) for a in self.axes)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    def dx(self) -> tuple:
        return tuple(float(a[1] - a[0]) for a in self.axes)

    @property
    def bounds(self) -> tuple:
        return tuple((float(a[0]), float(a[-1])) for a in self.axes)

    @property
    def n_pad(self) -> int:
        """Number of h nodes below the wall (h < 0)."""
        return int(np.sum(self.axes[0] < -1e-12))

    def meshgrid(self, sparse: bool = True):
        """Coordinate arrays; sparse=True returns broadcastable views of shape (1, .., n_i, .., 1)."""
        return tuple(np.meshgrid(*self.axes, indexing="ij", sparse=sparse))

    def axis_tensor(self, i: int, dtype=DTYPE) -> torch.Tensor:
        """Axis i as a torch tensor broadcastable against a full grid array."""
        shape = [1] * self.ndim
        shape[i] = self.shape[i]
        return torch.as_tensor(self.axes[i], dtype=dtype).reshape(shape)

    def to_dict(self) -> dict:
        return {"names": list(self.names), "shape": list(self.shape), "dx": list(self.dx),
                "bounds": [list(b) for b in self.bounds], "n_pad": self.n_pad}

    @staticmethod
    def regular(bounds, shape, names=None) -> "Grid":
        axes = tuple(np.linspace(lo, hi, int(n)) for (lo, hi), n in zip(bounds, shape))
        return Grid(axes, tuple(names) if names else ())

    @staticmethod
    def _h_axis(nh: int, h_max: float, h_pad: float) -> np.ndarray:
        """nh nodes on [0, h_max] plus ceil(h_pad / dh) nodes below the wall at the same spacing."""
        dh = h_max / (nh - 1)
        n_pad = int(math.ceil(h_pad / dh - 1e-9)) if h_pad > 0 else 0
        return np.arange(-n_pad, nh) * dh

    @staticmethod
    def planar4d(pp: PlanarParams, shape=(61, 61, 21, 21), h_max: float = 20.0, v_range=(-1.0, 4.0),
                 h_pad: float = 2.0) -> "Grid":
        """(h, v, z, zd) grid of E3: h in [-h_pad, h_max] (shape[0] nodes on [0, h_max], plus the pad below
        the wall at the same spacing), v in v_range, z in [-z_bar, z_bar], zd in [-zd_max, zd_max] with
        zd_max = omega_bar sqrt(1 - z_bar^2) (the swing rate admissible at |z| = z_bar; at smaller |z| the
        constraint allows |zd| up to omega_bar, which is off the grid). Trajectories that leave the box
        through v = v_max or |zd| = zd_max are continued by the linear-extrapolation boundary condition,
        which is mildly optimistic there."""
        zd_max = pp.omega_bar * math.sqrt(1.0 - pp.z_bar ** 2)
        h = Grid._h_axis(int(shape[0]), h_max, h_pad)
        v = np.linspace(v_range[0], v_range[1], int(shape[1]))
        z = np.linspace(-pp.z_bar, pp.z_bar, int(shape[2]))
        zd = np.linspace(-zd_max, zd_max, int(shape[3]))
        return Grid((h, v, z, zd), ("h", "v", "z", "zd"))

    @staticmethod
    def double_integrator(shape=(101, 101), h_max: float = 20.0, v_range=(-1.0, 4.0), h_pad: float = 2.0) -> "Grid":
        h = Grid._h_axis(int(shape[0]), h_max, h_pad)
        v = np.linspace(v_range[0], v_range[1], int(shape[1]))
        return Grid((h, v), ("h", "v"))


def estimate_resources(grid: Grid, n_arrays: int = 22, ns_per_node_step: float | None = None) -> dict:
    """Memory of the solver (about `n_arrays` live full-grid float64 arrays at the peak of an RK2 step
    with WENO3: 4 central derivatives + dissipation + 3 stage buffers + snapshot + ~13 temporaries of
    the derivative routine) and, given a measured rate, the time per RK2 step."""
    per_array = grid.size * BYTES
    out = {"nodes": grid.size, "bytes_per_array": per_array, "peak_bytes": n_arrays * per_array,
           "peak_gb": n_arrays * per_array / 1e9, "n_arrays": n_arrays}
    if ns_per_node_step is not None:
        out["sec_per_step"] = ns_per_node_step * 1e-9 * grid.size
    return out


def peak_rss_gb() -> float:
    """Peak resident set size of this process so far [GB]."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e9


# ---------------------------------------------------------------- input sets
class InputSet:
    """Support function sigma_U(d_T, d_E) = max_{(T, eta) in U(z, zd)} d_T T + d_E eta on the 4-D grid,
    and the per-(z, zd) maxima of |vdot| and |zddot| for the CFL condition and the LF dissipation."""
    name = "abstract"

    def support(self, dT: torch.Tensor, dE: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def max_speeds(self):
        raise NotImplementedError


def _speeds_tensors(vd, zdd, nz, nzd):
    vd = np.array(np.broadcast_to(vd, (nz, nzd)), float)
    zdd = np.array(np.broadcast_to(zdd, (nz, nzd)), float)
    return torch.as_tensor(vd, dtype=DTYPE)[None, None], torch.as_tensor(zdd, dtype=DTYPE)[None, None]


class EllipseStripInputs(InputSet):
    """Exact support function of U(z, zd) = {T_min <= T <= T_max} cap {((T - Tc)/aT)^2 + (eta/aE)^2 <= 1}:
    phi(T) = d_T T + |d_E| aE sqrt(1 - ((T - Tc)/aT)^2) is concave on [Tc - aT, Tc + aT] with unconstrained
    maximizer T_e = Tc + aT^2 d_T / |(aT d_T, aE d_E)|, so the maximizer over the strip is the projection
    T_b of T_e onto [T_min, T_max] and sigma_U = d_T T_b + |d_E| aE sqrt(1 - ((T_b - Tc)/aT)^2)
    (the same case analysis as `planar1_input_boundary`, applied to an arbitrary direction)."""
    name = "exact"

    def __init__(self, grid: Grid, pp: PlanarParams):
        z, zd = grid.axes[2][:, None], grid.axes[3][None, :]
        if not np.all(planar1_input_nonempty(z, zd, pp)):
            raise ValueError("U(z, zd) is empty at some grid node; the solver assumes a nonempty input set")
        Tc, aT, aE = _ellipse(z, zd, pp)
        self.Tc = torch.as_tensor(np.array(np.broadcast_to(Tc, (z.size, zd.size))), dtype=DTYPE)[None, None]
        self.aT = float(aT)
        self.aE = torch.as_tensor(np.array(np.broadcast_to(aE, (z.size, zd.size))), dtype=DTYPE)[None, None]
        self.T_min, self.T_max = float(pp.T_min), float(pp.T_max)
        self._speeds = _speeds_tensors(*planar1_max_speeds(z, zd, pp), z.size, zd.size)

    def support(self, dT, dE):
        gT = dT * self.aT
        gE = dE * self.aE
        gn = torch.sqrt(gT * gT + gE * gE).clamp_(min=1e-300)
        Tb = (gT * (self.aT / gn)).add_(self.Tc).clamp_(self.T_min, self.T_max)
        del gT, gE, gn
        r = (Tb - self.Tc).div_(self.aT)
        Eb = torch.sqrt((1.0 - r * r).clamp_(min=0.0)).mul_(self.aE)
        del r
        return Tb.mul_(dT).add_(Eb.mul_(dE.abs()))

    def max_speeds(self):
        return self._speeds


class PolygonInputs(InputSet):
    """Plan-K2 inner approximation: sigma_U(d) ~ max_k d . b_k over the K boundary points b_k of
    `precompute_inputs` (exact along the K directions, error O(1/K^2) in between; always <= the exact
    value, so it shrinks the kernel slightly). Evaluated in chunks along the h axis to bound memory."""
    name = "points"

    def __init__(self, grid: Grid, pp: PlanarParams, K: int = 128, chunk_bytes: float = 2.5e8):
        pts = precompute_inputs(grid, pp, K)                       # (nz, nzd, K, 2)
        self.T = torch.as_tensor(np.ascontiguousarray(pts[..., 0]), dtype=DTYPE)[None, None]   # (1, 1, nz, nzd, K)
        self.eta = torch.as_tensor(np.ascontiguousarray(pts[..., 1]), dtype=DTYPE)[None, None]
        self.K = int(K)
        per_h = int(np.prod(grid.shape[1:])) * self.K * BYTES
        self.chunk = max(1, int(chunk_bytes // per_h))
        z, zd = grid.axes[2][:, None], grid.axes[3][None, :]
        self._speeds = _speeds_tensors(*planar1_max_speeds(z, zd, pp), z.size, zd.size)

    def support(self, dT, dE):
        dT, dE = torch.broadcast_tensors(dT, dE)
        out = torch.empty_like(dT)
        for a in range(0, dT.shape[0], self.chunk):
            b = min(a + self.chunk, dT.shape[0])
            vals = dT[a:b, ..., None] * self.T
            vals.add_(dE[a:b, ..., None] * self.eta)
            out[a:b] = vals.amax(dim=-1)
            del vals
        return out

    def max_speeds(self):
        return self._speeds


class FrozenTensionInputs(InputSet):
    """Validation input set: eta = 0 and T in [T_min, T_max] (cables cannot be swung). On the slice
    zd = 0 the cable is frozen and (h, v) is a double integrator with vdot in [-a_max z, -T_min z / m_L]."""
    name = "frozen"

    def __init__(self, grid: Grid, pp: PlanarParams):
        self.T_min, self.T_max = float(pp.T_min), float(pp.T_max)
        z, zd = grid.axes[2][:, None], grid.axes[3][None, :]
        vd = self.T_max * np.abs(z) / pp.m_L * np.ones_like(zd)
        zdd = zd * zd * np.abs(z) / (1.0 - z * z)
        self._speeds = _speeds_tensors(vd, zdd, z.size, zd.size)

    def support(self, dT, dE):
        return torch.maximum(dT * self.T_min, dT * self.T_max)

    def max_speeds(self):
        return self._speeds


def precompute_inputs(grid: Grid, pp: PlanarParams, K: int = 128) -> np.ndarray:
    """Boundary points of U(z, zd) at every (z, zd) node of `grid`, shape (nz, nzd, K, 2), in the K
    directions th_k = 2 pi k / K (see `planar1_input_boundary`)."""
    z, zd = grid.axes[2][:, None], grid.axes[3][None, :]
    return planar1_input_boundary(z, zd, pp, K=K)


def make_inputs(grid: Grid, pp: PlanarParams, support: str = "exact", K: int = 128) -> InputSet:
    if support == "exact":
        return EllipseStripInputs(grid, pp)
    if support == "points":
        return PolygonInputs(grid, pp, K)
    if support == "frozen":
        return FrozenTensionInputs(grid, pp)
    raise ValueError(f"unknown support mode {support!r}")


# ---------------------------------------------------------------- spatial derivatives
def _slicer(ndim: int, axis: int):
    def S(a, b):
        idx = [slice(None)] * ndim
        idx[axis] = slice(a, b)
        return tuple(idx)
    return S


def pad_linear(V: torch.Tensor, axis: int, width: int = 2, clip=None) -> torch.Tensor:
    """Ghost cells by linear extrapolation on both ends of `axis`, clipped to clip = (lo, hi) if given."""
    S = _slicer(V.ndim, axis)
    n = V.shape[axis]
    V0, V1, Vn1, Vn2 = V[S(0, 1)], V[S(1, 2)], V[S(n - 1, n)], V[S(n - 2, n - 1)]
    lo = [(k + 1) * V0 - k * V1 for k in range(width, 0, -1)]          # V_{-k} = V0 + k (V0 - V1)
    hi = [(k + 1) * Vn1 - k * Vn2 for k in range(1, width + 1)]
    if clip is not None:
        lo = [g.clamp_(clip[0], clip[1]) for g in lo]
        hi = [g.clamp_(clip[0], clip[1]) for g in hi]
    return torch.cat(lo + [V] + hi, dim=axis)


def one_sided_derivatives(V: torch.Tensor, dx: float, axis: int, scheme: str = "weno3", clip=None):
    """Left- and right-biased approximations (p^-, p^+) of dV/dx_axis at every node.

    weno3: third-order weighted ENO (Jiang & Peng 2000, Osher & Fedkiw Sec. 3.4) combining the
    upwind-biased second-order stencil (linear weight 1/3) and the central one (2/3) with smoothness
    indicators beta = (second difference)^2 and eps = 1e-6 max(first differences^2) + 1e-99.
    eno2: second-order ENO, the candidate with the smaller |second difference|.
    Boundaries: two ghost cells by linear extrapolation (`pad_linear`, clipped to `clip`)."""
    n = V.shape[axis]
    Vp = pad_linear(V, axis, 2, clip)
    S = _slicer(V.ndim, axis)
    D1 = Vp[S(1, n + 4)] - Vp[S(0, n + 3)]                     # D1[j] = Vp[j+1] - Vp[j], j = 0..n+2
    del Vp
    v1, v2, v3, v4 = D1[S(0, n)], D1[S(1, n + 1)], D1[S(2, n + 2)], D1[S(3, n + 3)]
    d21, d32, d43 = v2 - v1, v3 - v2, v4 - v3                  # second differences at i-1, i, i+1
    if scheme == "eno2":
        pm = torch.where(d21.abs() < d32.abs(), d21, d32).mul_(0.5).add_(v2)
        pp = torch.where(d43.abs() < d32.abs(), d43, d32).mul_(-0.5).add_(v3)
    elif scheme == "weno3":
        eps = torch.maximum(torch.maximum(v1.abs(), v2.abs()), torch.maximum(v3.abs(), v4.abs()))
        eps = eps.mul_(eps).mul_(1e-6).add_(1e-99)
        b32 = (d32 * d32).add_(eps)
        a1 = b32.mul_(b32).reciprocal_().mul_(2.0 / 3.0)        # central candidate weight (unnormalized)
        # left-biased: candidates q0 = v2 + d21/2 (smoothness from d21) and q1 = v2 + d32/2
        b = (d21 * d21).add_(eps)
        a0 = b.mul_(b).reciprocal_().mul_(1.0 / 3.0)
        w0 = a0.div_(a0 + a1)
        pm = (d21 * w0).add_(d32.mul(1.0 - w0)).mul_(0.5).add_(v2)      # w0 q0 + (1 - w0) q1
        del b, a0, w0
        # right-biased: candidates q0 = v3 - d43/2 (smoothness from d43) and q1 = v3 - d32/2
        b = (d43 * d43).add_(eps)
        a0 = b.mul_(b).reciprocal_().mul_(1.0 / 3.0)
        w0 = a0.div_(a0 + a1)
        pp = (d43 * w0).add_(d32.mul(1.0 - w0)).mul_(-0.5).add_(v3)
        del b, a0, w0, a1, b32, eps
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
    del d21, d32, d43, D1
    return pm.div_(dx), pp.div_(dx)


# ---------------------------------------------------------------- pseudo-time integration
def _fmt_t(s: float) -> str:
    return f"{s / 60:.1f} min" if s >= 90 else f"{s:.1f} s"


def _save_checkpoint(path: Path, V, Vsnap, t, t_snap, it, hist, win, wall, dt) -> None:
    """Atomic snapshot of the integration state (written to a temporary file, then renamed)."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, V=V.numpy(), Vsnap=Vsnap.numpy(), t=t, t_snap=t_snap, it=it, hist=np.asarray(hist, float),
                 win=np.asarray(win, float).reshape(-1, 2), wall=wall, dt=dt)
    os.replace(tmp, path)


def evolve(V: torch.Tensor, rhs, dt: float, t_max: float, tol: float, window: float = 0.5,
           verbose: bool = True, label: str = "", log_every: float = 15.0, clip=None,
           checkpoint: str | os.PathLike | None = None, checkpoint_every: float = 600.0):
    """TVD-RK2 (Heun) integration of V_tau = rhs(V) until max|V(tau) - V(tau - window)| < tol or tau = t_max.
    Returns (V, info) with the per-step history of max|V^{n+1} - V^n| and the windowed changes.
    checkpoint: file that receives the integration state every `checkpoint_every` seconds of wall time and
    from which an interrupted run resumes (same grid, same dt: the integration continues unchanged)."""
    t, it, converged = 0.0, 0, False
    hist, win = [], []
    V = V.clone()
    Vsnap, t_snap = V.clone(), 0.0
    wall_before = 0.0
    ckpt = Path(checkpoint) if checkpoint else None
    if ckpt is not None and ckpt.exists():
        try:
            d = np.load(ckpt, allow_pickle=False)
            if tuple(d["V"].shape) != tuple(V.shape) or abs(float(d["dt"]) - dt) > 1e-12 * max(1.0, dt):
                warnings.warn(f"checkpoint {ckpt} does not match this grid / time step; ignored")
            else:
                V = torch.from_numpy(d["V"]).to(V.dtype).clone()
                Vsnap = torch.from_numpy(d["Vsnap"]).to(V.dtype).clone()
                t, t_snap, it = float(d["t"]), float(d["t_snap"]), int(d["it"])
                hist = d["hist"].tolist()
                win = [(float(a), float(b)) for a, b in d["win"]]
                wall_before = float(d["wall"])
                if verbose:
                    print(f"[{label}] resumed from {ckpt.name}: step {it}, tau = {t:.3f} s, {_fmt_t(wall_before)} of wall time before", flush=True)
        except Exception as exc:
            warnings.warn(f"checkpoint {ckpt} unreadable ({exc!r}); starting from tau = 0")
    t0 = time.perf_counter()
    last_log = last_ckpt = t0
    n_max = int(math.ceil(t_max / dt - 1e-9))
    while t < t_max - 1e-12:
        dtn = min(dt, t_max - t)
        V1 = rhs(V).mul_(dtn).add_(V)                     # stage 1
        Vn = rhs(V1).mul_(dtn).add_(V1).add_(V).mul_(0.5)  # Heun average
        if clip is not None:
            Vn.clamp_(clip[0], clip[1])
        torch.sub(Vn, V, out=V1)
        dmax = float(V1.abs_().max())
        del V1
        V = Vn
        t += dtn
        it += 1
        hist.append(dmax)
        if t - t_snap >= window - 1e-9:
            change = float((V - Vsnap).abs_().max())
            win.append((t, change))
            if change < tol:
                converged = True
                if verbose:
                    print(f"[{label}] converged: max|V(t) - V(t - {window} s)| = {change:.2e} < {tol:g} at tau = {t:.3f} s "
                          f"after {it} steps, {_fmt_t(time.perf_counter() - t0)}", flush=True)
                break
            Vsnap.copy_(V)
            t_snap = t
        now = time.perf_counter()
        if ckpt is not None and now - last_ckpt >= checkpoint_every:
            _save_checkpoint(ckpt, V, Vsnap, t, t_snap, it, hist, win, wall_before + now - t0, dt)
            last_ckpt = now
        if verbose and (now - last_log >= log_every or it == 1):
            per = (now - t0) / it
            print(f"[{label}] step {it}/{n_max}  tau = {t:.3f} s  max|dV|/step = {dmax:.2e}  "
                  f"win = {win[-1][1] if win else float('nan'):.2e}  {per * 1e3:.0f} ms/step  "
                  f"elapsed {_fmt_t(now - t0)}  ETA <= {_fmt_t(per * (n_max - it))}", flush=True)
            last_log = now
    wall = wall_before + time.perf_counter() - t0
    if verbose and not converged:
        print(f"[{label}] horizon t_max = {t_max} s reached without convergence: last window change "
              f"{win[-1][1] if win else float('nan'):.2e} (tol {tol:g}); {it} steps, {_fmt_t(wall)}", flush=True)
    info = {"iterations": it, "converged": converged, "wall_time": wall, "t_final": t, "dt": dt,
            "history": np.asarray(hist), "window_changes": win, "resumed": wall_before > 0.0,
            "sec_per_step": wall / max(it, 1), "ns_per_node_step": wall / max(it, 1) / V.numel() * 1e9}
    return V, info


# ---------------------------------------------------------------- results
@dataclass
class KernelResult:
    V: np.ndarray
    grid: Grid
    iterations: int
    converged: bool
    wall_time: float
    history: np.ndarray
    t_final: float = float("nan")
    dt: float = float("nan")
    window_changes: list = field(default_factory=list)
    info: dict = field(default_factory=dict)

    @property
    def kernel(self) -> np.ndarray:
        return self.V >= 0.0

    def save(self, path) -> None:
        np.savez_compressed(path, V=self.V, **{f"axis_{n}": a for n, a in zip(self.grid.names, self.grid.axes)},
                            names=np.array(self.grid.names), history=self.history)

    @staticmethod
    def load(path) -> "KernelResult":
        d = np.load(path, allow_pickle=False)
        names = tuple(str(s) for s in d["names"])
        grid = Grid(tuple(d[f"axis_{n}"] for n in names), names)
        return KernelResult(d["V"], grid, -1, True, float("nan"), d["history"])


def _interp_lines(result: KernelResult, coords, start_axis: int):
    """Multilinear interpolation weights along the axes start_axis.. at broadcast query points;
    returns (idx_lo per axis, weights per axis, shape, Q)."""
    grid = result.grid
    qs = np.broadcast_arrays(*[np.asarray(c, float) for c in coords])
    shape = qs[0].shape
    Q = qs[0].size
    idx_lo, wts = [], []
    for k, q in enumerate(qs, start=start_axis):
        ax = grid.axes[k]
        x = np.clip(q.ravel(), ax[0], ax[-1])
        i = np.clip(np.searchsorted(ax, x, side="right") - 1, 0, ax.size - 2)
        idx_lo.append(i)
        wts.append((x - ax[i]) / (ax[i + 1] - ax[i]))
    return idx_lo, wts, shape, Q


def boundary_h(result: KernelResult, *coords):
    """h*(coords) = smallest h with V >= 0 along the h axis at the given values of the other coordinates
    (multilinear interpolation of V in those coordinates, then linear interpolation of the zero
    crossing in h). NaN where V < 0 for every h or V >= 0 for every h (no crossing); if V >= 0 at the
    lowest h node but not everywhere, that node's h is returned. `coords` broadcast against each other."""
    V, grid = result.V, result.grid
    if len(coords) != grid.ndim - 1:
        raise ValueError(f"expected {grid.ndim - 1} coordinates, got {len(coords)}")
    idx_lo, wts, shape, Q = _interp_lines(result, coords, 1)
    lines = np.zeros((grid.shape[0], Q))
    for corner in itertools.product((0, 1), repeat=grid.ndim - 1):
        w = np.ones(Q)
        idx = []
        for k, c in enumerate(corner):
            idx.append(idx_lo[k] + c)
            w = w * (wts[k] if c else 1.0 - wts[k])
        lines += w[None, :] * V[(slice(None), *idx)]
    nonneg = lines >= 0.0
    any_pos, all_pos = nonneg.any(0), nonneg.all(0)
    first = np.argmax(nonneg, axis=0)
    out = np.full(Q, np.nan)
    h = grid.axes[0]
    ok = any_pos & ~all_pos & (first >= 1)
    i, qi = first[ok], np.arange(Q)[ok]
    Vlo, Vhi = lines[i - 1, qi], lines[i, qi]
    out[ok] = h[i - 1] + (h[i] - h[i - 1]) * (-Vlo) / (Vhi - Vlo)
    out[any_pos & ~all_pos & (first == 0)] = h[0]
    return out.reshape(shape)


def interp_V(result: KernelResult, *coords):
    """Multilinear interpolation of V at points (h, v, ...); coordinates broadcast; outside the grid
    the point is clamped to the box."""
    V, grid = result.V, result.grid
    if len(coords) != grid.ndim:
        raise ValueError(f"expected {grid.ndim} coordinates, got {len(coords)}")
    idx_lo, wts, shape, Q = _interp_lines(result, coords, 0)
    out = np.zeros(Q)
    for corner in itertools.product((0, 1), repeat=grid.ndim):
        w = np.ones(Q)
        idx = []
        for k, c in enumerate(corner):
            idx.append(idx_lo[k] + c)
            w = w * (wts[k] if c else 1.0 - wts[k])
        out += w * V[tuple(idx)]
    return out.reshape(shape)


# ---------------------------------------------------------------- 4-D solver
def initial_level_set(grid: Grid, pp: PlanarParams, h_scale: float = 1.0, v_floor: float = 2.0,
                      v_ceil: float = 0.5) -> torch.Tensor:
    """V0 = clip(min(h / h_scale, (z_bar - |z|) / z_bar, (omega_bar^2 (1 - z^2) - zd^2) / omega_bar^2), -v_floor, v_ceil).
    Each term is nondimensional and positive inside C. h_scale = 1 m: within 1 m of the wall the h term
    is the smallest of the three unless a cable constraint is nearly active, so the zero level of V near
    the wall is the wall constraint and dV/dh ~ 1/m there, which keeps the zero-crossing interpolation
    in h well conditioned. The clipping does not move the zero level (module docstring)."""
    h, v, z, zd = (grid.axis_tensor(i) for i in range(4))
    t_h = h / h_scale
    t_z = (pp.z_bar - z.abs()) / pp.z_bar
    t_w = (pp.omega_bar ** 2 * (1.0 - z * z) - zd * zd) / pp.omega_bar ** 2
    shape = grid.shape
    V0 = torch.minimum(torch.minimum(t_h.expand(shape), t_z.expand(shape)), t_w.expand(shape)).contiguous()
    return V0.clamp_(-v_floor, v_ceil)


def cfl_time_step(grid: Grid, alphas, cfl: float) -> float:
    """dt = cfl / max_nodes sum_i alpha_i / dx_i (alphas broadcastable against the grid)."""
    tot = None
    for a, dx in zip(alphas, grid.dx):
        term = torch.as_tensor(a, dtype=DTYPE) / dx
        tot = term if tot is None else tot + term
    return cfl / float(tot.max())


def _maybe_compile(fn, enable: bool, V0: torch.Tensor, label: str, verbose: bool):
    """torch.compile `fn` and run it once on V0 (compilation happens at the first call); on any failure,
    fall back to the eager function. Returns (fn, compiled: bool, compile_seconds)."""
    if not enable:
        return fn, False, 0.0
    t0 = time.perf_counter()
    try:
        cfn = torch.compile(fn, dynamic=False)
        out_c = cfn(V0)
        out_e = fn(V0)
        err = float((out_c - out_e).abs().max())
        del out_c, out_e
        if not math.isfinite(err) or err > 1e-10:
            raise RuntimeError(f"compiled operator differs from eager by {err:.2e}")
        if verbose:
            print(f"[{label}] torch.compile: fused operator ready in {time.perf_counter() - t0:.1f} s "
                  f"(max deviation from eager {err:.1e})", flush=True)
        return cfn, True, time.perf_counter() - t0
    except Exception as exc:
        warnings.warn(f"torch.compile unavailable or inconsistent ({exc!r}); running eager")
        return fn, False, time.perf_counter() - t0


def viability_kernel_4d(grid: Grid, pp: PlanarParams, K: int = 128, cfl: float = 0.5, tol: float = 1e-4,
                        t_max: float = 15.0, verbose: bool = True, *, support: str = "exact",
                        scheme: str = "weno3", h_scale: float = 1.0, v_floor: float = 2.0, v_ceil: float = 0.5,
                        window: float = 0.5, compile: bool | None = None, threads: int | None = None,
                        label: str = "hj4d", checkpoint: str | os.PathLike | None = None,
                        checkpoint_every: float = 600.0) -> KernelResult:
    """Viability kernel of C for the planar one-cable system on `grid` (see the module docstring).

    support: "exact" (closed-form support function, default), "points" (max over the K precomputed
    boundary points of plan K2), "frozen" (eta = 0, validation). scheme: "weno3" | "eno2".
    v_floor / v_ceil: clipping of V (nondimensional; the grid's pad below the wall should cover
    v_floor * h_scale meters). compile: torch.compile the semi-discrete operator (None = only for
    grids with >= COMPILE_MIN_NODES nodes). checkpoint: see `evolve` (resumable long runs)."""
    if threads:
        torch.set_num_threads(int(threads))
    if grid.ndim != 4:
        raise ValueError("viability_kernel_4d needs a 4-D (h, v, z, zd) grid")
    if grid.n_pad * grid.dx[0] < v_floor * h_scale - 1e-9:
        warnings.warn(f"the h pad below the wall ({grid.n_pad * grid.dx[0]:.2f} m) is shorter than "
                      f"v_floor * h_scale = {v_floor * h_scale:.2f} m; deeply lost states rely on extrapolation")
    inputs = make_inputs(grid, pp, support, K)
    m, l, m_L = float(pp.m[0]), float(pp.l[0]), float(pp.m_L)
    v, z, zd = grid.axis_tensor(1), grid.axis_tensor(2), grid.axis_tensor(3)
    alpha_v, alpha_zd = inputs.max_speeds()                       # (1, 1, nz, nzd)
    alphas = (v.abs(), alpha_v, zd.abs(), alpha_zd)
    half_alphas = [0.5 * a for a in alphas]
    cor = (zd * zd * z / (1.0 - z * z))                           # z zd^2 / (1 - z^2), (1, 1, nz, nzd)
    neg_z_over_mL = -z / m_L
    inv_ml = 1.0 / (m * l)
    dx = grid.dx
    clip = (-float(v_floor), float(v_ceil))
    dt = cfl_time_step(grid, alphas, cfl)

    def rhs(V):
        diss = None
        pcs = []
        for ax in range(4):
            pm, pl = one_sided_derivatives(V, dx[ax], ax, scheme, clip)
            d = (pl - pm).mul_(half_alphas[ax])
            diss = d if diss is None else diss.add_(d)
            pcs.append(pm.add_(pl).mul_(0.5))
            del pl
        ph, pv, pz, pzd = pcs
        H = diss.sub_(ph * v).add_(pz * zd).sub_(pzd * cor)
        del ph, pz
        H.add_(inputs.support(pv * neg_z_over_mL, pzd * inv_ml))
        return H.clamp_(max=0.0)

    V0 = initial_level_set(grid, pp, h_scale, v_floor, v_ceil)
    res = estimate_resources(grid)
    if verbose:
        print(f"[{label}] grid {grid.shape} = {grid.size:,} nodes ({grid.n_pad} h nodes below the wall), "
              f"{res['bytes_per_array'] / 1e6:.0f} MB per array, peak ~{res['peak_gb']:.2f} GB; "
              f"dt = {dt * 1e3:.3f} ms (CFL {cfl}), <= {math.ceil(t_max / dt)} steps to t_max = {t_max} s; "
              f"support = {inputs.name}, scheme = {scheme}, clip = {clip}", flush=True)
    do_compile = (grid.size >= COMPILE_MIN_NODES) if compile is None else bool(compile)
    step_fn, compiled, t_compile = _maybe_compile(rhs, do_compile, V0, label, verbose)
    V, info = evolve(V0, step_fn, dt, t_max, tol, window, verbose, label, clip=clip, checkpoint=checkpoint,
                     checkpoint_every=checkpoint_every)
    info.update({"scheme": scheme, "support": inputs.name, "K": K if support == "points" else None,
                 "h_scale": h_scale, "v_floor": v_floor, "v_ceil": v_ceil, "cfl": cfl, "tol": tol, "t_max": t_max,
                 "window": window, "compiled": compiled, "compile_seconds": t_compile,
                 "threads": torch.get_num_threads(), "peak_rss_gb": peak_rss_gb(), **res})
    return KernelResult(V.numpy(), grid, info["iterations"], info["converged"], info["wall_time"], info["history"],
                        info["t_final"], dt, info["window_changes"], info)


# ---------------------------------------------------------------- 2-D validation problem
def kernel_2d_double_integrator(grid2: Grid, a_brake: float, a_push: float | None = None, cfl: float = 0.5,
                                tol: float = 1e-4, t_max: float = 15.0, verbose: bool = False, *,
                                scheme: str = "weno3", h_scale: float = 1.0, v_floor: float = 2.0,
                                v_ceil: float = 0.5, window: float = 0.5, label: str = "hj2d") -> KernelResult:
    """Same numerics on the (h, v) grid for hdot = -v, vdot = -b, b in [-a_push, a_brake] (a_push defaults
    to a_brake), constraint h >= 0. Exact kernel: h >= v_+^2 / (2 a_brake), independent of a_push."""
    if grid2.ndim != 2:
        raise ValueError("kernel_2d_double_integrator needs an (h, v) grid")
    a_push = a_brake if a_push is None else float(a_push)
    b_lo, b_hi = -a_push, float(a_brake)
    h, v = grid2.axis_tensor(0), grid2.axis_tensor(1)
    alphas = (v.abs(), torch.tensor(max(abs(b_lo), abs(b_hi)), dtype=DTYPE))
    half_alphas = [0.5 * a for a in alphas]
    dx = grid2.dx
    clip = (-float(v_floor), float(v_ceil))
    dt = cfl_time_step(grid2, alphas, cfl)

    def rhs(V):
        pm_h, pl_h = one_sided_derivatives(V, dx[0], 0, scheme, clip)
        pm_v, pl_v = one_sided_derivatives(V, dx[1], 1, scheme, clip)
        diss = (pl_h - pm_h).mul_(half_alphas[0]).add_((pl_v - pm_v).mul_(half_alphas[1]))
        ph = pm_h.add_(pl_h).mul_(0.5)
        pv = pm_v.add_(pl_v).mul_(0.5)
        H = diss.sub_(ph * v).add_(torch.maximum(pv * (-b_lo), pv * (-b_hi)))
        return H.clamp_(max=0.0)

    V0 = (h / h_scale).expand(grid2.shape).contiguous().clamp_(clip[0], clip[1])
    V, info = evolve(V0, rhs, dt, t_max, tol, window, verbose, label, clip=clip)
    info.update({"scheme": scheme, "h_scale": h_scale, "v_floor": v_floor, "v_ceil": v_ceil, "cfl": cfl,
                 "tol": tol, "t_max": t_max, "window": window, "a_brake": a_brake, "a_push": a_push,
                 **estimate_resources(grid2)})
    return KernelResult(V.numpy(), grid2, info["iterations"], info["converged"], info["wall_time"], info["history"],
                        info["t_final"], dt, info["window_changes"], info)
