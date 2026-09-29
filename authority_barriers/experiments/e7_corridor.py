"""E7 corridor (optional track; Sec. VI: "50 trials of nominal transport in a 20 m corridor with two walls from randomized
initial states in X_RF, where the filter should change the nominal thrust by at most 5 %"; M8.2; D-7):
two walls with opposite normals along the lateral axis at +-10 m (MultiWallFilter), velocity transport along the corridor
at 2-4 m/s, lateral speed U[-1, 1] m/s, cables random in the admissible swing sets of both walls, 15 s, paper's actuator
model. Reported: contacts with either wall, infeasible ticks, and ||u - u_nom|| / ||u_nom|| per tick (median, 95th
percentile and maximum over the trials) against the 5 % claim.

  python -m authority_barriers.experiments.e7_corridor [--n 50] [--workers 4]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from authority_barriers.theory.authority import BarrierData
from authority_barriers.theory.params import load_set
from authority_barriers.theory.sampling import sample_V
from authority_barriers.theory.state import State
from authority_barriers.theory.filters import MultiWallFilter
from authority_barriers.theory.taut_model import unpack
from authority_barriers.simulator.harness import TrialConfig, corridor_params, make_nominal
from authority_barriers.experiments.common import in_XRF, out_dir, run_many_cached, save_json, summary_table

HALF_WIDTH = 10.0


def sample_states(p, walls, n, seed):
    pA, pB = walls
    dA, dB = BarrierData.nominal(pA), BarrierData.nominal(pB)
    rng = np.random.default_rng(seed)
    axis_sign = float(np.sign(pA.r_vec @ p.n_vec)) or 1.0            # the corridor axis r_A is +/- the base set's wall normal
    out = []
    while len(out) < n:
        z, zd = sample_V(rng, p.N, dA.nu, dA.zeta_bar, 0.5)          # cables near vertical, as in nominal transport (inner half of V)
        w, wd = sample_V(rng, p.N, p.nu_w, p.w_bar, 0.5)
        v_lat = rng.uniform(-1.0, 1.0)                                  # toward wall A (positive) or B
        v_along = rng.uniform(2.0, 4.0) * rng.choice([-1.0, 1.0])
        h_A = rng.uniform(2.0, 2 * HALF_WIDTH - 2.0)
        st = State.from_swing(pA, h_A, v_lat, z, w, zd, wd, x_perp=0.0, altitude=15.0, v_perp=v_along)
        if in_XRF(st, pA, dA)[0] and in_XRF(st, pB, dB)[0]:
            out.append((st, v_along * axis_sign))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--t-final", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=75)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--gentle", action="store_true", help="sensitivity subset: a gentler transport nominal (k_swing 1.0, k_q 0.5) that keeps the swing states inside V")
    args = ap.parse_args()
    if args.smoke:
        args.n, args.t_final = 3, 3.0
    gains = dict(nominal_k_swing=1.0, nominal_k_q=0.5) if args.gentle else {}
    tag = "gentle_" if args.gentle else ""
    p = load_set("A")
    walls = corridor_params(p, HALF_WIDTH)
    pA, pB = walls
    pairs = sample_states(p, walls, args.n, args.seed)
    x0s = [s for s, _ in pairs]
    cfgs = [TrialConfig(filter="proposed", nominal="corridor", v_cmd=float(vc), t_final=args.t_final, seed=i, actuator="perfect",
                        corridor_half_width=HALF_WIDTH, wall_cap=1200.0, label=f"e7_corridor_{tag}{i}", **gains) for i, (_, vc) in enumerate(pairs)]
    out = out_dir("optional", "e7_corridor")
    t0 = time.time()
    logs, _ = run_many_cached(cfgs, x0s, out, args.workers)
    per = []
    barrier_only = MultiWallFilter(walls, swing_mode="none")             # the change attributable to the wall barriers alone
    for L, cfg in zip(logs, cfgs):
        hA = L.d("h")
        xL = L.taut[0:3, :]
        hB = pB.d0 - pB.n_vec @ xL
        rel = L.d("du_norm") / np.maximum(L.d("unom_norm"), 1e-9)
        nom = make_nominal(p, cfg, unpack(L.taut[:, 0], p.N))
        rel_b = []
        for k in range(0, L.t_diag.size, 25):                             # decomposition on every 25th tick: re-solve without the swing rows
            st = unpack(L.taut[:, k], p.N); u_nom = nom(float(L.t_diag[k]), st)
            rb = barrier_only.solve(st, u_nom)
            rel_b.append(np.linalg.norm(rb.u - u_nom) / max(np.linalg.norm(u_nom), 1e-9))
        rel_b = np.array(rel_b)
        per.append({"label": cfg.label, "v_along": cfg.v_cmd, "min_hA": float(hA.min()), "min_hB": float(hB.min()), "min_H": float(np.nanmin(L.d("H"))),
                    "contact": L.termination == "wall_contact", "truncated": L.termination == "wall_cap", "infeasible_ticks": int(np.sum(L.d("feasible") < 0.5)),
                    "closest_wall": "A" if hA.min() <= hB.min() else "B",
                    "rel_change_median": float(np.median(rel)), "rel_change_p95": float(np.quantile(rel, 0.95)), "rel_change_max": float(rel.max()),
                    "rel_change_barrier_only_median": float(np.median(rel_b)), "rel_change_barrier_only_max": float(rel_b.max()),
                    "ticks_above_5pct": int(np.sum(rel > 0.05)), "ticks": int(rel.size), "t_end": float(L.t_diag[-1])})
    n = len(per)
    med = np.array([r["rel_change_median"] for r in per]); p95 = np.array([r["rel_change_p95"] for r in per]); mx = np.array([r["rel_change_max"] for r in per])
    frac_above = sum(r["ticks_above_5pct"] for r in per) / max(sum(r["ticks"] for r in per), 1)
    medb = np.array([r["rel_change_barrier_only_median"] for r in per]); mxb = np.array([r["rel_change_barrier_only_max"] for r in per])
    row = {"trials": n, "contacts (either wall)": sum(r["contact"] for r in per), "truncated (1200 s cap, C-12)": sum(r["truncated"] for r in per),
           "min h over walls [m]": f"{min(min(r['min_hA'], r['min_hB']) for r in per):.3f}",
           "infeasible ticks": sum(r["infeasible_ticks"] for r in per), "||u - u_nom||/||u_nom||: median of trial medians": f"{np.median(med):.3f}",
           "median of trial 95th pct.": f"{np.median(p95):.3f}", "max over trials": f"{mx.max():.3f}", "fraction of ticks above 5 %": f"{frac_above:.3f}",
           "trials with median above 5 %": int(np.sum(med > 0.05)),
           "barrier rows alone (swing rows off): median of trial medians": f"{np.median(medb):.3f}", "barrier rows alone: max": f"{mxb.max():.3f}", "wall [s]": f"{time.time() - t0:.0f}"}
    table = summary_table([row], list(row.keys()))
    viol = row["contacts (either wall)"]
    ok5 = np.median(med) <= 0.05 and frac_above <= 0.05
    ok5b = np.median(medb) <= 0.05
    verdict = (f"{'no contacts' if viol == 0 else str(viol) + ' contacts'}; full filter thrust change {'within' if ok5 else 'ABOVE'} 5 %; "
               f"barrier rows alone {'within' if ok5b else 'ABOVE'} 5 %" + ("" if ok5 else " -> D-7: amend the practicality sentence (state which rows act)"))
    md = (f"# E7 corridor (Sec. VI limits; M8.2; D-7) — {verdict}\n\n20 m corridor (walls at lateral +-{HALF_WIDTH:g} m), velocity transport along the corridor at 2-4 m/s, "
          f"lateral speed U[-1, 1] m/s, {n} random states in X_RF of both walls, {args.t_final:g} s, paper's actuator model; MultiWallFilter (rows of both walls in one program).\n\n{table}\n\n"
          "Per trial: " + "; ".join(f"{r['label'][-2:].lstrip('_')}: v_along {r['v_along']:+.1f}, min h {min(r['min_hA'], r['min_hB']):.2f}, rel. change median {r['rel_change_median']:.3f} / max {r['rel_change_max']:.2f}, "
                                    f"infeasible {r['infeasible_ticks']}" for r in per) + "\n")
    stem = ("summary_smoke" if args.smoke else "summary") + ("_gentle" if args.gentle else "")
    md = md.replace("# E7 corridor", "# E7 corridor" + (" (gentler nominal: k_swing 1.0, k_q 0.5; sensitivity subset)" if args.gentle else ""))
    (out / f"{stem}.md").write_text(md)
    save_json(out / f"{stem.replace('summary', 'results')}.json", {"row": row, "per_trial": per, "verdict": verdict, "params": p.to_dict(), "half_width": HALF_WIDTH, "gains": gains})
    print(md)


if __name__ == "__main__":
    main()
