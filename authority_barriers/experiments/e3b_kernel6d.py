"""E3b (optional track, Sec. VI Q2): coarse 6-D viability kernel of the planar two-cable system (set A2) against the
bounds h = D and h = D_rel at sampled swing states and speeds (M7.3; D-25, C-3).

  python -m authority_barriers.experiments.e3b_kernel6d --shape 25 25 13 13 13 13 --threads 8
  python -m authority_barriers.experiments.e3b_kernel6d --shape 9 9 5 5 5 5          # smoke
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from authority_barriers.theory import stopping as SD
from authority_barriers.theory.authority import BarrierData
from authority_barriers.theory.params import load_set
from authority_barriers.theory.planar import PlanarParams
from authority_barriers.theory.sampling import sample_V
from authority_barriers.experiments.common import out_dir, save_json, summary_table
from authority_barriers.viability.hj4d import boundary_h
from authority_barriers.viability.hj6d import grid_planar6d, viability_kernel_6d


def probes(p, n: int = 20, seed: int = 0):
    """n sampled swing states (z1, zd1, z2, zd2), each cable in V(nu, z_bar)."""
    rng = np.random.default_rng(seed)
    z1, zd1 = sample_V(rng, n, p.nu, p.z_bar, 0.9)
    z2, zd2 = sample_V(rng, n, p.nu, p.z_bar, 0.9)
    return np.column_stack([z1, zd1, z2, zd2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", type=int, nargs=6, default=[25, 25, 13, 13, 13, 13])
    ap.add_argument("--h-max", type=float, default=10.0)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--t-max", type=float, default=15.0)
    ap.add_argument("--speeds", type=int, default=6)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    p = load_set("A2"); pp = PlanarParams.from_params(p)
    out = out_dir("optional", "e3b_kernel6d")
    tag = args.tag or "x".join(str(s) for s in args.shape)
    grid = grid_planar6d(pp, shape=tuple(args.shape), h_max=args.h_max)
    ckpt = out / f"e3b_kernel6d_{tag}.ckpt.npz"
    t0 = time.perf_counter()
    res = viability_kernel_6d(grid, pp, verbose=True, threads=args.threads, t_max=args.t_max, checkpoint=ckpt)
    wall = res.info.get("wall_time", time.perf_counter() - t0)
    res.save(out / f"e3b_kernel6d_{tag}.npz")
    ckpt.unlink(missing_ok=True)
    nom, rel = BarrierData.nominal(p), BarrierData.relaxed(p)
    speeds = np.linspace(0.5, 4.0, args.speeds)
    rows = []
    for z1, zd1, z2, zd2 in probes(p):
        for v in speeds:
            hs = float(boundary_h(res, np.array([v]), np.array([z1]), np.array([zd1]), np.array([z2]), np.array([zd2]))[0])
            D = SD.D_of(v, np.array([z1, z2]), np.array([zd1, zd2]), nom); Dr = SD.D_of(v, np.array([z1, z2]), np.array([zd1, zd2]), rel)
            rows.append({"z1": float(z1), "zd1": float(zd1), "z2": float(z2), "zd2": float(zd2), "v": float(v), "h_star": hs, "D": D, "D_rel": Dr,
                         "frac_lower": (hs - Dr) / hs if hs > 0 else np.nan, "frac_upper": (D - hs) / hs if hs > 0 else np.nan})
    hs = np.array([r["h_star"] for r in rows]); D = np.array([r["D"] for r in rows]); Dr = np.array([r["D_rel"] for r in rows]); ok = np.isfinite(hs)
    tol = 2.0 * grid.dx[0]
    ver = {"n_probes": int(ok.sum()), "n_nan": int((~ok).sum()), "tol_m": tol, "lower_bound_holds": bool(np.all(hs[ok] >= Dr[ok] - tol)),
           "upper_bound_holds": bool(np.all(hs[ok] <= D[ok] + tol)), "min_hstar_minus_Drel": float(np.min(hs[ok] - Dr[ok])),
           "max_hstar_minus_D": float(np.max(hs[ok] - D[ok])), "probes_above_D_plus_tol": int(np.sum(hs[ok] > D[ok] + tol)),
           "median_frac_lower": float(np.nanmedian([r["frac_lower"] for r in rows])), "median_frac_upper": float(np.nanmedian([r["frac_upper"] for r in rows]))}
    info = {k: v for k, v in res.info.items() if isinstance(v, (int, float, str, bool)) or v is None}
    save_json(out / f"e3b_kernel6d_{tag}.json", {"shape": list(grid.shape), "h_max": args.h_max, "dx": [float(d) for d in grid.dx], "wall": wall,
                                                 "converged": res.converged, "iterations": res.iterations, "t_final": res.t_final, "dt": res.dt,
                                                 "info": info, "rows": rows, "verdict": ver, "params": p.to_dict()})
    tab = summary_table([{"z1": f"{r['z1']:+.3f}", "zd1": f"{r['zd1']:+.2f}", "z2": f"{r['z2']:+.3f}", "zd2": f"{r['zd2']:+.2f}", "v": f"{r['v']:.2f}",
                          "h* [m]": f"{r['h_star']:.3f}", "D [m]": f"{r['D']:.3f}", "D_rel [m]": f"{r['D_rel']:.3f}"} for r in rows if abs(r["v"] - speeds[len(speeds) // 2]) < 1e-9],
                        ["z1", "zd1", "z2", "zd2", "v", "h* [m]", "D [m]", "D_rel [m]"])
    md = (f"# E3b: coarse 6-D two-cable kernel (set A2; M7.3; D-25, C-3)\n\n"
          f"Grid {tuple(grid.shape)} (h in [0, {args.h_max}] m plus the pad below the wall): cells h {grid.dx[0]:.3f} m, v {grid.dx[1]:.3f} m/s, "
          f"z {grid.dx[2]:.4f}, zd {grid.dx[3]:.4f} s^-1; {grid.size:,} nodes; dt {res.dt * 1e3:.3f} ms; converged {res.converged} in {res.iterations} steps "
          f"(tau {res.t_final:.2f} s); wall {wall / 60:.0f} min on {info.get('threads')} threads; peak RSS {info.get('peak_rss_gb', 0):.1f} GB. "
          f"Input set: per-cable inner approximation (thrust ellipse shrunk by the other cable's tension range, caps T_i <= {info.get('T_cap'):.2f} N), "
          f"so the kernel is under-approximated and h* is biased upward (C-3), on top of the dissipation bias (C-7).\n\n"
          f"Probes: 20 sampled swing states x {args.speeds} speeds. Lower bound h* >= D_rel - 2 cells: {'holds' if ver['lower_bound_holds'] else 'VIOLATED'} "
          f"(min h* - D_rel = {ver['min_hstar_minus_Drel']:+.3f} m); upper h* <= D + 2 cells: {'holds' if ver['upper_bound_holds'] else 'exceeded'} "
          f"(max h* - D = {ver['max_hstar_minus_D']:+.3f} m at {ver['probes_above_D_plus_tol']} probes); median (h* - D_rel)/h* = {ver['median_frac_lower']:.2f}, "
          f"median (D - h*)/h* = {ver['median_frac_upper']:.2f}; NaN probes {ver['n_nan']}.\n\nProbes at v = {speeds[len(speeds) // 2]:.2f} m/s:\n\n{tab}\n")
    (out / f"e3b_kernel6d_{tag}_summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
