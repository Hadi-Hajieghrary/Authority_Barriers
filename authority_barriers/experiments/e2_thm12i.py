"""E2 (Table I row 2): invariance and feasibility of the proposed filter from the boundary layer
of X_RF under an adversarial nominal (Thm. 12(i); M3.3, M3.4), in two variants (D-20):
raw filter (17) at 200 Hz, and margin-enforced filter Hdot_j >= -gamma(H) + delta with delta the
sampled-data margin of Rem. 19 measured on the taut model (max per-tick deficit rate of H)."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from authority_barriers.theory.authority import BarrierData
from authority_barriers.theory.filters import ProposedFilter
from authority_barriers.theory.params import load_set
from authority_barriers.theory.sampling import sample_XRF_layer
from authority_barriers.theory.taut_model import rhs_from_thrust, unpack
from authority_barriers.simulator.harness import TrialConfig
from authority_barriers.simulator.nominal import AdversarialNominal
from authority_barriers.experiments.common import (calibrate_margin, classify_violation, out_dir, run_many_cached, sampled_allowance,
                                    save_json, summary_table, swing_states_in_V)


def model_tensions(L, p):
    """Tension implied by each logged command on the taut-cable model (the quantity the filter bounds)."""
    out = np.zeros((p.N, L.t_diag.size))
    for k in range(L.t_diag.size):
        st = unpack(L.taut[:, k], p.N)
        out[:, k] = rhs_from_thrust(st, L.cmd[:, k].reshape(p.N, 3), p)["T"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--params", default="A")
    ap.add_argument("--t-final", type=float, default=15.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--actuator", default="perfect", choices=["perfect", "attitude"])
    ap.add_argument("--wall-cap", type=float, default=None, help="wall-clock seconds per trial (default: 2400 for the attitude loop, none otherwise); "
                    "a trial stopped by the cap is reported as truncated and counted neither as a pass nor as a violation (C-12)")
    ap.add_argument("--margin", default="0", help="delta on the barrier rows [m/s] (kept for reference; 0 by default)")
    ap.add_argument("--rows", default="maximizers", choices=["local_maxima", "maximizers"])
    ap.add_argument("--h-offset", default="0", help="state margin H_min [m]: the sampler draws H0 in [H_min, H_min + 0.05 D]; "
                    "'auto' = largest accumulated deficit observed in the raw variants of this actuator (D-20 revised)")
    ap.add_argument("--layer-abs", type=float, default=None, help="if given, H0 ~ U[H_min, H_min + layer_abs] (absolute layer, in m) "
                    "instead of the relative layer 0.05 D; variant C measures the contact rate against H0")
    args = ap.parse_args()
    if args.wall_cap is None and args.actuator == "attitude":
        args.wall_cap = 2400.0
    if args.smoke:
        args.n, args.t_final = 4, 3.0
    p = load_set(args.params)
    data = BarrierData.nominal(p)
    calib = None
    delta = float(args.margin)
    if args.h_offset == "auto":
        import json, glob
        losses = []
        for f in glob.glob(str(out_dir("core", "") / f"e2_{args.actuator}_*_off0/results.json")):
            d = json.loads(Path(f).read_text())
            losses += [r["deficit_total"] for r in d["per_trial"]]
        if not losses:
            raise SystemExit("no raw-variant results to calibrate the state margin from; run --h-offset 0 first")
        h_offset = float(np.max(losses))
        calib = {"H_min": h_offset, "n_raw_trials": len(losses), "loss_quantiles": np.quantile(losses, [0.5, 0.9, 0.99, 1.0]).tolist()}
        print(f"state margin H_min = max accumulated deficit over {len(losses)} raw trials = {h_offset:.3f} m "
              f"(quantiles 50/90/99/100 %: {np.round(calib['loss_quantiles'], 3).tolist()})")
    else:
        h_offset = float(args.h_offset)
    tag = f"{args.actuator}_{'lmax' if args.rows == 'local_maxima' else 'max'}_off{'auto' if args.h_offset == 'auto' else args.h_offset}" + (f"_layer{args.layer_abs:g}" if args.layer_abs else "")
    out = out_dir("core", f"e2_{tag}")
    rng = np.random.default_rng(args.seed)
    x0s = sample_XRF_layer(rng, p, args.n, data, v_range=(1.0, 4.0), h_offset=h_offset,
                           delta_frac=0.05 if args.layer_abs is None else 0.0, delta_abs=args.layer_abs)
    cfgs = [TrialConfig(params=args.params, filter="proposed", nominal="adversarial", t_final=args.t_final, seed=i,
                        actuator=args.actuator, hdot_margin=delta, rows=args.rows, wall_cap=args.wall_cap, label=f"e2_{tag}_{i}") for i in range(len(x0s))]
    t0 = time.time()
    logs, _ = run_many_cached(cfgs, x0s, out, args.workers)
    tol_sw = 2.0 * p.nu_rel * p.dt_filter ** 2         # one-tick excursion of a swing state outside V (D-12)
    per = []
    for L, cfg in zip(logs, cfgs):
        inV = swing_states_in_V(L, p, data, tol=tol_sw)
        Tm = model_tensions(L, p)
        Tphys = L.cable[:p.N]
        m, rate, within = sampled_allowance(L, p)
        r = {"label": cfg.label, "H0": float(L.d("H")[0]), "h0": float(L.d("h")[0]), "v0": float(L.d("v")[0]),
             "min_h": float(np.min(L.d("h"))), "min_H": float(np.min(L.d("H"))), "v_end": float(L.d("v")[-1]),
             "t_end": float(L.d("t")[-1]),
             "all_feasible": bool(np.all(L.d("feasible") > 0.5)), "n_infeasible_ticks": int(np.sum(L.d("feasible") < 0.5)),
             "min_T_model": float(np.min(Tm)), "min_T_phys": float(np.min(Tphys)),
             "slack_ticks": int(np.sum(np.any(Tphys <= 0.0, axis=0))), "swing_in_V_all": bool(np.all(inV)),
             "deficit_total": float(m[-1]), "deficit_rate_max": rate, "within_allowance": within,
             "max_att_err": float(np.nanmax(L.att_err)), "median_att_err": float(np.nanmedian(L.att_err)),
             "termination": L.termination, "wall_time": L.meta["wall_time"]}
        r["pass"] = bool(r["min_h"] >= 0 and L.termination not in ("wall_contact", "wall_cap") and r["all_feasible"]
                         and r["min_T_model"] >= p.T_min - 0.05 and r["swing_in_V_all"])
        if not r["pass"]:
            if L.termination == "wall_cap":
                r["attribution"] = f"truncated by the wall-time cap at t = {r['t_end']:.2f} s (no verdict, C-12)"
            elif (r["min_h"] < 0 or L.termination == "wall_contact") and r["all_feasible"] and r["swing_in_V_all"] and within:
                r["attribution"] = "sampled-data case (contact depth within the accumulated deficit, Rem. 19)"
            elif not r["all_feasible"]:
                r["attribution"] = "infeasible ticks (see relaxed solutions)"
            elif not r["swing_in_V_all"]:
                r["attribution"] = "swing state left V beyond the one-tick tolerance"
            else:
                r["attribution"] = classify_violation(L, p, data, 0.0)
        per.append(r)
    n = len(per)
    row = {"variant": tag, "rows": args.rows, "H_min [m]": f"{h_offset:.3f}", "trials": n, "PASS": sum(r["pass"] for r in per),
           "min h >= 0": sum(r["min_h"] >= 0 and r["termination"] != "wall_contact" for r in per),
           "sampled-data cases": sum((not r["pass"]) and r.get("attribution", "").startswith("sampled-data") for r in per),
           "truncated (wall cap)": sum(r["termination"] == "wall_cap" for r in per),
           "feasible all ticks": sum(r["all_feasible"] for r in per),
           "model T >= T_min - 0.05": sum(r["min_T_model"] >= p.T_min - 0.05 for r in per),
           "swing in V (tol)": sum(r["swing_in_V_all"] for r in per),
           "trials with physical slack": sum(r["slack_ticks"] > 0 for r in per),
           "max deficit m(T) [m]": f"{max(r['deficit_total'] for r in per):.4f}",
           "max deficit rate [m/s]": f"{max(r['deficit_rate_max'] for r in per):.4f}"}
    table = summary_table([row], list(row.keys()))
    # contact rate against the initial margin (variant C: H_min = the largest H0 that still touched the wall)
    H0s = np.array([r["H0"] for r in per]); cont = np.array([r["termination"] == "wall_contact" or r["min_h"] < 0 for r in per])
    edges = np.quantile(H0s, np.linspace(0, 1, 6)) if not args.layer_abs else np.linspace(h_offset, h_offset + args.layer_abs, 6)
    bins = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (H0s >= a) & (H0s <= b)
        cs = [r["v_end"] for r, s_, c_ in zip(per, sel, cont) if s_ and c_]
        bins.append({"H0 range [m]": f"[{a:.3f}, {b:.3f}]", "trials": int(sel.sum()), "contacts": int(cont[sel].sum()),
                     "median contact speed [m/s]": f"{np.median(cs):.3f}" if cs else "-"})
    H_min_meas = float(np.max(H0s[cont])) if cont.any() else 0.0
    bin_table = summary_table(bins, list(bins[0].keys()))
    viol = [r for r in per if not r["pass"]]
    txt = (f"# E2 (Thm. 12(i)) — actuator {args.actuator}, rows {args.rows}, state margin H_min = {h_offset:.3f} m — "
           f"{'smoke' if args.smoke else 'full'} run\n\n{table}\n\nContact rate against H0 (largest H0 with a contact: {H_min_meas:.3f} m; "
           f"median contact speed over all contacts {np.median([r['v_end'] for r, c in zip(per, cont) if c]) if cont.any() else float('nan'):.3f} m/s):\n\n{bin_table}\n\nNon-PASS trials: {len(viol)}\n" +
           "\n".join(f"- {r['label']}: H0 {r['H0']:.3f}, min h {r['min_h']:.3f}, deficit {r['deficit_total']:.4f}, {r.get('attribution','')}" for r in viol) +
           f"\n\nattitude error: max {max(r['max_att_err'] for r in per):.3f} N, median {np.median([r['median_att_err'] for r in per]):.3f} N (C-10); "
           f"physical slack ticks total {sum(r['slack_ticks'] for r in per)}; swing tolerance {tol_sw:.1e}; wall {time.time()-t0:.0f} s\n")
    (out / "summary.md").write_text(txt)
    save_json(out / "results.json", {"row": row, "per_trial": per, "delta": delta, "h_offset": h_offset, "calibration": calib,
                                     "H_min_measured": H_min_meas, "H0_bins": bins,
                                     "actuator": args.actuator, "params": p.to_dict()})
    print(txt)


if __name__ == "__main__":
    main()
