"""E3 (Table I row 3, Thm. 12(ii)): grid-converged 4-D viability kernel of the planar one-cable system
(set A1) compared with the two closed-form bounds h = D (Thm. 12(i)) and h = D_rel (Thm. 12(ii)) at
20 fixed cable states x 10 approach speeds (M4.2, M4.3, M4.5).

  python -m authority_barriers.experiments.e3_kernel --grid 61        # (61, 61, 21, 21)
  python -m authority_barriers.experiments.e3_kernel --grid 121       # (121, 121, 41, 41)
  python -m authority_barriers.experiments.e3_kernel --compare        # tabulate all saved grids and the change between them
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from authority_barriers.theory import profile as PF
from authority_barriers.theory import stopping as SD
from authority_barriers.theory.authority import BarrierData
from authority_barriers.theory.params import load_set
from authority_barriers.theory.planar import PlanarParams
from authority_barriers.experiments.common import out_dir, save_json, summary_table
from authority_barriers.viability.hj4d import Grid, KernelResult, boundary_h, viability_kernel_4d

SHAPES = {61: (61, 61, 21, 21), 81: (81, 81, 31, 31), 121: (121, 121, 41, 41), 31: (31, 31, 11, 11)}


def probe_states(p):
    """20 fixed (z1, zd1) inside V(nu, z_bar): 10 at rest spanning z, 5 swinging toward the wall,
    5 swinging away (zd = -/+ 0.3), each spanning the admissible z range of its zd."""
    zb, nu = p.z_bar, p.nu
    states = [(z, 0.0) for z in np.linspace(-0.9 * zb, 0.9 * zb, 10)]
    for zd in (0.3, -0.3):
        lo, hi = PF.bounds_V(zd, nu, zb)
        states += [(z, zd) for z in np.linspace(lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo), 5)]
    return np.array(states)


def compare(res: KernelResult, p, data_nom, data_rel, speeds):
    S = probe_states(p)
    rows = []
    for z, zd in S:
        for v in speeds:
            hs = float(boundary_h(res, np.array([v]), np.array([z]), np.array([zd]))[0])
            D = SD.D_of(v, np.array([z]), np.array([zd]), data_nom)
            Dr = SD.D_of(v, np.array([z]), np.array([zd]), data_rel)
            rows.append({"z": float(z), "zd": float(zd), "v": float(v), "h_star": hs, "D": D, "D_rel": Dr,
                         "frac_lower": (hs - Dr) / hs if hs > 0 else np.nan, "frac_upper": (D - hs) / hs if hs > 0 else np.nan})
    return rows


def verdicts(rows, tol):
    hs = np.array([r["h_star"] for r in rows]); D = np.array([r["D"] for r in rows]); Dr = np.array([r["D_rel"] for r in rows])
    ok = np.isfinite(hs)
    lower_ok = np.all(hs[ok] >= Dr[ok] - tol)          # Thm. 12(ii): a viable state has h >= D_rel
    upper_ok = np.all(hs[ok] <= D[ok] + tol)           # Thm. 12(i): X_RF is viable, so h* <= D
    refute = np.where(ok & (hs < Dr - tol))[0]
    return {"n_probes": int(ok.sum()), "n_nan": int((~ok).sum()), "lower_bound_holds": bool(lower_ok), "upper_bound_holds": bool(upper_ok),
            "max_violation_lower_m": float(np.max(Dr[ok] - hs[ok])), "max_violation_upper_m": float(np.max(hs[ok] - D[ok])),
            "refuting_probes": [int(i) for i in refute], "tol_m": tol}


def run(grid_id: int, threads: int | None, t_max: float, out: Path, zn: int | None = None, h_max: float = 20.0):
    p = load_set("A1")
    pp = PlanarParams.from_params(p)
    shape = SHAPES[grid_id] if zn is None else (grid_id, grid_id, zn, zn)
    tag = str(grid_id) if (zn is None and h_max == 20.0) else f"{grid_id}x{zn or SHAPES[grid_id][2]}_h{h_max:g}"
    grid = Grid.planar4d(pp, shape=shape, h_max=h_max, v_range=(-1.0, 4.0))
    ckpt = out / f"e3_kernel_{tag}.ckpt.npz"          # resumable: the integration state every 10 min of wall time
    t0 = time.perf_counter()
    res = viability_kernel_4d(grid, pp, verbose=True, threads=threads, t_max=t_max, checkpoint=ckpt)
    wall = res.info.get("wall_time", time.perf_counter() - t0)   # includes the wall time before a resume
    res.save(out / f"e3_kernel_{tag}.npz")
    ckpt.unlink(missing_ok=True)
    speeds = np.linspace(0.5, 4.0, 10)
    rows = compare(res, p, BarrierData.nominal(p), BarrierData.relaxed(p), speeds)
    tol = float(grid.dx[0])
    ver = verdicts(rows, tol)
    info = {k: v for k, v in res.info.items() if isinstance(v, (int, float, str, bool)) or v is None}
    save_json(out / f"e3_kernel_{tag}.json", {"grid": grid_id, "shape": list(shape), "h_max": h_max, "dx": [float(d) for d in grid.dx],
                                                  "wall": wall, "converged": res.converged, "iterations": res.iterations,
                                                  "t_final": res.t_final, "dt": res.dt, "info": info, "rows": rows, "verdict": ver,
                                                  "params": p.to_dict()})
    print(f"grid {shape} (h_max {h_max}): wall {wall/60:.1f} min, converged {res.converged} ({res.iterations} steps, t_final {res.t_final:.2f} s)")
    print(json.dumps(ver, indent=1))
    return rows, ver


def compare_saved(out: Path):
    files = list(out.glob("e3_kernel_*.json"))
    tabs_all = {f.stem.replace("e3_kernel_", ""): json.loads(f.read_text()) for f in files}
    saved = sorted(tabs_all, key=lambda k: (-tabs_all[k]["dx"][0], int(np.prod(tabs_all[k]["shape"]))))   # coarsest first (h cell, then nodes)
    tabs = {g: tabs_all[g] for g in saved}
    lines = [f"# E3 kernel (Thm. 12(ii)) — set A1, grids {saved}\n"]
    for g in saved:
        v = tabs[g]["verdict"]
        lines.append(f"- grid {tabs[g]['shape']} (h_max {tabs[g].get('h_max', 20.0)}): dx_h {tabs[g]['dx'][0]:.3f} m, converged {tabs[g]['converged']} in {tabs[g]['iterations']} steps, "
                     f"wall {tabs[g]['wall']/60:.1f} min; lower bound (h* >= D_rel - dx) {'holds' if v['lower_bound_holds'] else 'VIOLATED'} "
                     f"(max violation {v['max_violation_lower_m']:.3f} m); upper (h* <= D + dx) {'holds' if v['upper_bound_holds'] else 'violated'} "
                     f"(max excess {v['max_violation_upper_m']:.3f} m); NaN probes {v['n_nan']}")
    for ga, gb in zip(saved[:-1], saved[1:]):                 # M4.2 for every consecutive pair of the ladder
        a, b = tabs[ga], tabs[gb]
        ha = np.array([r["h_star"] for r in a["rows"]]); hb = np.array([r["h_star"] for r in b["rows"]])
        ok = np.isfinite(ha) & np.isfinite(hb)
        ch = np.abs(hb - ha)[ok]; rel = ch / np.maximum(hb[ok], 1e-9)
        cell = b["dx"][0]
        crit = np.maximum(0.02 * hb[ok], cell)
        conv = np.all(ch <= crit)
        v = np.array([r["v"] for r in b["rows"]])[ok]
        by_v = "; ".join(f"v <= {vmax:g}: {ch[v <= vmax].max():.3f} m" for vmax in (1.0, 2.0, 4.0) if np.any(v <= vmax))
        lines.append(f"- M4.2 convergence {ga} -> {gb}: max |dh*| {ch.max():.3f} m, max relative {rel.max():.3f}, probes meeting the criterion "
                     f"{int(np.sum(ch <= crit))}/{int(ok.sum())}; criterion max(2 % h*, one cell = {cell:.3f} m): {'PASS' if conv else 'FAIL'}; "
                     f"max change by speed ({by_v}); h* decreased at {int(np.sum((hb - ha)[ok] < 0))} probes")
    for f in sorted(out.glob("crosscheck_hj_*.json")):        # C-7 cross-check with hj_reachability (authority_barriers.viability.crosscheck_hj)
        c = json.loads(f.read_text()); v = c["verdict"]
        lines.append(f"- cross-check (hj_reachability, WENO5 + TVD-RK3, same nodes and initial level set) on grid {c['grid']}: "
                     f"|h*_hj - h*_hj4d| median {v['median_abs_dh']:.3f} m, max {v['max_abs_dh']:.3f} m = {v['max_abs_dh_cells']:.2f} cells "
                     f"({'within one cell' if v['within_one_cell'] else 'BEYOND one cell'}); median h*_hj - h*_hj4d {v['hj_minus_hj4d_median']:+.3f} m; "
                     f"sign of V differs at {100 * v['sign_mismatch_fraction']:.2f} % of the nodes; bounds for the hj kernel: lower "
                     f"{'holds' if v['hj_lower_bound_holds'] else 'VIOLATED'}, upper {'holds' if v['hj_upper_bound_holds'] else 'violated'}; "
                     f"converged {c['info']['converged']} at tau {c['info']['t_final']:.1f} s, wall {c['info']['wall'] / 60:.0f} min")
    fin = tabs[saved[-1]]["rows"]
    lines.append("\n| z | zd | v [m/s] | h* [m] | D [m] | D_rel [m] | (h*-D_rel)/h* | (D-h*)/h* |\n|---|---|---|---|---|---|---|---|")
    for r in fin:
        if abs(r["v"] - 2.0556) < 1e-3 or abs(r["v"] - 3.6111) < 1e-3 or abs(r["v"] - 0.5) < 1e-6:
            lines.append(f"| {r['z']:+.3f} | {r['zd']:+.2f} | {r['v']:.2f} | {r['h_star']:.3f} | {r['D']:.3f} | {r['D_rel']:.3f} | {r['frac_lower']:+.3f} | {r['frac_upper']:+.3f} |")
    txt = "\n".join(lines) + "\n"
    (out.parent / "e3_kernel_summary.md").write_text(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, choices=list(SHAPES), default=61)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--t-max", type=float, default=15.0)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--zn", type=int, default=None, help="nodes in z and zd (default from the grid ladder)")
    ap.add_argument("--h-max", type=float, default=20.0)
    args = ap.parse_args()
    out = out_dir("core", "e3_kernel")
    if not args.compare:
        run(args.grid, args.threads, args.t_max, out, zn=args.zn, h_max=args.h_max)
    compare_saved(out)


if __name__ == "__main__":
    main()
