"""E5 (Table I row 5, Rem. 18): the robust filter (D_rob, the substitutions of Rem. 18 in the barrier
data) under bounded wind at d_bar from states in X_RF^rob: min_t H_rob >= 0 (M5.3).

Two actuator modes (D-19): "perfect" (Sec. II model, thrust vectors applied directly): the wind is
then the only disturbance and lies within the budget (d_bar_L, d_bar_i) by construction, so this is the
test of Rem. 18; "attitude" (Sec. VI setup, geometric loop at 1 kHz): the inner loop's tracking error
adds to the wind and exceeds the budget in closed loop (C-10), reported with attribution.
Two initial layers, as in E2 (D-20 revised): the boundary layer of X_RF^rob, H_rob(x0) in
[0, 0.05 D_rob] (the pre-registered T7), and with --layer-abs L the layer H_rob(x0) uniform in
[0, L] m, which measures the violation rate against the initial margin (the sampled-data margin of
Rem. 19 is the largest H_rob(x0) that still lost H_rob >= 0). Rem. 18's PD tracking on rho_fb is not a
separate law here: the reserve rho_fb sits inside rho_i (params.derive_set) and the inner loop (or the
perfect actuator) realizes the command. With --scale 2 the gusts deliberately exceed the budget
(E7, optional track; no claim)."""
from __future__ import annotations

import argparse
import time

import numpy as np

from authority_barriers.theory.authority import BarrierData
from authority_barriers.theory.params import load_set
from authority_barriers.theory.sampling import sample_XRF_layer
from authority_barriers.simulator.harness import TrialConfig
from authority_barriers.experiments.common import out_dir, run_many_cached, sampled_deficit, save_json, summary_table
from authority_barriers.experiments.e2_thm12i import model_tensions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--t-final", type=float, default=15.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--nominal", default="adversarial")
    ap.add_argument("--actuator", default="perfect", choices=["perfect", "attitude"])
    ap.add_argument("--wall-cap", type=float, default=None, help="wall-clock seconds per trial (default: 2400 for the attitude loop, none otherwise); "
                    "a trial stopped by the cap is reported as truncated and counted neither as a pass nor as a violation (C-12)")
    ap.add_argument("--layer-abs", type=float, default=None, help="H_rob(x0) ~ U[0, layer_abs] m instead of the relative layer 0.05 D_rob")
    ap.add_argument("--h-offset", type=float, default=0.0, help="state margin added to H_rob(x0) (E7 gusts: 0.8 m, above the sampled-data margin, D-23)")
    ap.add_argument("--h-ref", type=float, default=0.8, help="state margin H_min measured in E2 (largest H0 with a contact), for the claim on {H_rob >= H_min}")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=21)
    args = ap.parse_args()
    if args.wall_cap is None and args.actuator == "attitude":
        args.wall_cap = 2400.0
    if args.smoke:
        args.n, args.t_final = 3, 3.0
    p = load_set("A")
    rob = BarrierData.robust(p)
    track = "core" if args.scale <= 1.0 else "optional"
    tag = f"e5_scale{args.scale:g}" + ("" if args.actuator == "perfect" else f"_{args.actuator}") + (f"_layer{args.layer_abs:g}" if args.layer_abs else "") + (f"_off{args.h_offset:g}" if args.h_offset else "")
    out = out_dir(track, tag)
    rng = np.random.default_rng(args.seed)
    x0s = sample_XRF_layer(rng, p, args.n, rob, v_range=(1.0, 4.0), h_offset=args.h_offset,   # x0 in X_RF^rob
                           delta_frac=0.05 if args.layer_abs is None else 0.0, delta_abs=args.layer_abs)
    cfgs = [TrialConfig(filter="proposed", robust=True, nominal=args.nominal, v_cmd=2.0, t_final=args.t_final,
                        seed=1000 + i, wind_scale=args.scale, actuator=args.actuator, wall_cap=args.wall_cap, label=f"e5_s{args.scale:g}_{i}")
            for i in range(len(x0s))]
    t0 = time.time()
    logs, _ = run_many_cached(cfgs, x0s, out, args.workers)
    per = []
    for L, cfg in zip(logs, cfgs):
        t, Hr, h, feas = L.d("t"), L.d("H_rob"), L.d("h"), L.d("feasible")
        _, m, rate = sampled_deficit(t, Hr, p.kappa_H)                       # sampled-data loss of H_rob (Rem. 19)
        within = bool(np.all(np.maximum(0.0, -Hr) <= m + 1e-9))
        Tm = model_tensions(L, p)
        att = float(np.nanmax(L.att_err)) if np.isfinite(L.att_err).any() else 0.0
        r = {"label": cfg.label, "H_rob0": float(Hr[0]), "H0": float(L.d("H")[0]), "h0": float(h[0]), "v0": float(L.d("v")[0]),
             "min_H_rob": float(np.min(Hr)), "min_H": float(np.min(L.d("H"))), "min_h": float(np.min(h)), "v_end": float(L.d("v")[-1]),
             "all_feasible": bool(np.all(feas > 0.5)), "n_infeasible_ticks": int(np.sum(feas < 0.5)),
             "min_T_model": float(np.min(Tm)), "min_T_phys": float(np.min(L.cable[:p.N])),
             "slack_ticks": int(np.sum(np.any(L.cable[:p.N] <= 0.0, axis=0))),
             "deficit_total": float(m[-1]), "deficit_rate_max": rate, "within_allowance": within,
             "max_att_err": att, "median_att_err": float(np.nanmedian(L.att_err)) if np.isfinite(L.att_err).any() else 0.0,
             "termination": L.termination, "wall_time": L.meta["wall_time"]}
        r["pass"] = bool(r["min_H_rob"] >= 0.0 and L.termination != "wall_cap")   # M5.3
        if not r["pass"]:
            if L.termination == "wall_cap":
                r["attribution"] = f"truncated by the wall-time cap at t = {float(t[-1]):.2f} s (no verdict, C-12)"
            elif att > 0.15:                                   # the disturbance hypothesis of Rem. 18 is violated first
                r["attribution"] = ("model gap (attitude tracking error beyond the 0.15 N budget of D-16, C-10)"
                                    + ("; loss also within the sampled-data deficit" if r["all_feasible"] and within else ""))
            elif r["all_feasible"] and within:
                r["attribution"] = "sampled-data case (loss of H_rob within the accumulated deficit, Rem. 19)"
            elif not r["all_feasible"]:
                r["attribution"] = "infeasible ticks (relaxed solutions applied, D-15)"
            else:
                r["attribution"] = "unattributed: ESCALATE (replay with the logged wind needed)"
        per.append(r)
    n = len(per)
    H0s = np.array([r["H_rob0"] for r in per]); viol = np.array([(not r["pass"]) and r["termination"] != "wall_cap" for r in per])
    n_above = int(np.sum(viol & (H0s >= args.h_ref)))
    row = {"variant": tag, "actuator": args.actuator, "scale": args.scale, "trials": n, "min_t H_rob >= 0 (M5.3)": int(np.sum(~viol)),
           "sampled-data cases": sum(r.get("attribution", "").startswith("sampled-data") for r in per),
           "model gap (attitude, C-10)": sum(r.get("attribution", "").startswith("model gap") for r in per),
           "truncated (wall cap)": sum(r["termination"] == "wall_cap" for r in per),
           "min h >= 0": sum(r["min_h"] >= 0 and r["termination"] != "wall_contact" for r in per),
           "feasible all ticks": sum(r["all_feasible"] for r in per),
           "model T >= T_min - 0.05": sum(r["min_T_model"] >= p.T_min - 0.05 for r in per),
           "trials with physical slack": sum(r["slack_ticks"] > 0 for r in per),
           "max attitude error [N]": f"{max(r['max_att_err'] for r in per):.3f}",
           f"violations with H_rob(x0) >= {args.h_ref:g} m": n_above, "wall [s]": f"{time.time()-t0:.0f}"}
    table = summary_table([row], list(row.keys()))
    edges = np.linspace(args.h_offset, args.h_offset + args.layer_abs, 6) if args.layer_abs else np.quantile(H0s, np.linspace(0, 1, 6))   # the absolute layer starts at h_offset
    bins = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (H0s >= a) & (H0s <= b)
        bins.append({"H_rob(x0) range [m]": f"[{a:.3f}, {b:.3f}]", "trials": int(sel.sum()), "violations": int(viol[sel].sum()),
                     "min H_rob [m]": f"{min(r['min_H_rob'] for r, s_ in zip(per, sel) if s_):.3f}" if sel.any() else "-"})
    H_min_meas = float(np.max(H0s[viol])) if viol.any() else 0.0
    if args.scale > 1.0:
        verdict = "no claim (2 d_bar)"
    elif not viol.any():
        verdict = "PASS"
    else:
        verdict = (f"FAIL on the literal M5.3 ({int(viol.sum())} of {n} lose H_rob >= 0: {row['sampled-data cases']} sampled-data cases, "
                   f"{row['model gap (attitude, C-10)']} inner-loop cases); "
                   f"on {{H_rob >= {args.h_ref:g} m}} (D-20): {'PASS' if n_above == 0 else f'{n_above} violations'}")
    bad = [r for r in per if not r["pass"]]
    txt = (f"# E5 (Rem. 18) — wind at {args.scale:g} d_bar, actuator {args.actuator}, "
           f"H_rob(x0) in {'[' + format(args.h_offset, 'g') + ', ' + format(args.h_offset + args.layer_abs, 'g') + ' m]' if args.layer_abs else '[' + format(args.h_offset, 'g') + ', ' + format(args.h_offset, 'g') + ' + 0.05 D_rob]'} — {verdict}\n\n{table}\n\n"
           f"Violation rate against the initial robust margin (largest H_rob(x0) with a violation: {H_min_meas:.3f} m):\n\n"
           f"{summary_table(bins, list(bins[0].keys()))}\n\nNon-PASS trials: {len(bad)}\n" +
           "\n".join(f"- {r['label']}: H_rob0 {r['H_rob0']:.3f}, min H_rob {r['min_H_rob']:.3f}, min h {r['min_h']:.3f}, deficit {r['deficit_total']:.4f}, "
                     f"{r.get('attribution', '')}" for r in bad) +
           f"\n\nd_bar_L = {p.d_bar_L} N, d_bar_i = {p.d_bar_i} N (wind clipped in norm at scale x d_bar, tau_w = 0.5 s), nu_rob = {rob.nu:.3f} s^-2, "
           f"alpha_sat_rob = {rob.alpha_sat:.3f} m/s^2; nominal {args.nominal}; dt = {p.dt_filter} s; {'smoke' if args.smoke else 'full'} run\n")
    (out / "summary.md").write_text(txt)
    save_json(out / "results.json", {"row": row, "per_trial": per, "verdict": verdict, "H_min_measured": H_min_meas, "bins": bins,
                                     "actuator": args.actuator, "layer_abs": args.layer_abs, "h_ref": args.h_ref, "params": p.to_dict()})
    print(txt)


if __name__ == "__main__":
    main()
