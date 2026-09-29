"""E7 stress list (optional track; Sec. VI "limits of the guarantee"; M8.1, M8.3, M8.4, M8.5; D-23, D-26):
    fmax   f_max reduced 44 -> 20 N in 2 N steps (plant and filter consistent), Assumption 7 checked per value
    mass   true m_L in {0.8, 1.2} kg with the nominal filter and with the filter designed for m_L = 1.2 kg (set A_mL12)
    soft   cable stiffness / 10
    omega  initial swing rates set to 1.5 omega_bar (outside X_op)
    gusts  2 d_bar: see e5_robust --scale 2 --h-offset 0.8 --layer-abs 0.5 (run by the queue)
Paper's actuator model, adversarial nominal, initial states with H(x0) in [0.8, 1.3] m (above the measured sampled-data
margin), 10 s. Every contact of the proposed filter is attributed to the stress factor by the M3.4 rules.

  python -m authority_barriers.experiments.e7_stress [--variant fmax|mass|soft|omega|all] [--workers 4] [--smoke]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from authority_barriers.theory.authority import BarrierData
from authority_barriers.theory.params import all_ok, check_assumptions, load_set
from authority_barriers.theory.sampling import sample_XRF_layer
from authority_barriers.theory.state import State
from authority_barriers.simulator.harness import TrialConfig, apply_overrides
from authority_barriers.experiments.common import out_dir, run_many_cached, sampled_allowance, save_json, summary_table


def states(p, n, seed, data=None):
    rng = np.random.default_rng(seed)
    return sample_XRF_layer(rng, p, n, data or BarrierData.nominal(p), v_range=(1.0, 4.0), delta_frac=0.0, h_offset=0.8, delta_abs=0.5)


def metrics(L, p, label_extra=None):
    h, H, feas = L.d("h"), L.d("H"), L.d("feasible")
    _, _, within = sampled_allowance(L, p)
    r = {"min_h": float(h.min()), "min_H": float(np.nanmin(H)), "contact": L.termination == "wall_contact", "termination": L.termination,
         "infeasible_ticks": int(np.sum(feas < 0.5)), "slack_ticks": int(np.sum(np.any(L.cable[:p.N] <= 0.0, axis=0))),
         "within_sampled_allowance": within, "t_end": float(L.t_diag[-1]), "max_omega": float(np.nanmax(L.d("max_omega")))}
    if label_extra:
        r.update(label_extra)
    return r


def run_variant(name, cfgs, x0s, p, workers, out):
    logs, _ = run_many_cached(cfgs, x0s, out, workers)
    return [metrics(L, p, {"label": c.label}) for L, c in zip(logs, cfgs)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="all")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--t-final", type=float, default=10.0)
    args = ap.parse_args()
    p = load_set("A"); data = BarrierData.nominal(p)
    out = out_dir("optional", "e7_stress")
    n_f, n_m, n_s, n_o = (2, 2, 2, 2) if args.smoke else (10, 20, 20, 20)
    tf = 3.0 if args.smoke else args.t_final
    base = dict(filter="proposed", nominal="adversarial", t_final=tf, actuator="perfect")
    t0 = time.time()
    results, tables = {}, []
    # ---- f_max sweep (M8.3)
    if args.variant in ("fmax", "all"):
        x0s = states(p, n_f, 71)
        rows = []
        for f in ([44, 30, 20] if args.smoke else range(44, 19, -2)):
            q = apply_overrides(p, TrialConfig(param_overrides={"f_max": float(f)}))
            mg = check_assumptions(q)
            failed = sorted({k.split("[")[0] for k, m in mg.items() if not m.ok})
            holds = not failed
            thin = holds and not all_ok(mg)                                    # holds, but with less than the 5 % margin of D-6
            cfgs = [TrialConfig(**base, param_overrides={"f_max": float(f)}, seed=i, label=f"e7_fmax{f:g}_{i}") for i in range(len(x0s))]
            per = run_variant("fmax", cfgs, x0s, p, args.workers, out)
            results[f"fmax_{f:g}"] = {"f_max": float(f), "assumptions_hold": holds, "margin_ok": all_ok(mg), "failed": failed, "per_trial": per}
            rows.append({"f_max [N]": f, "Assumptions 7, 13, 14": "hold" if all_ok(mg) else ("hold (margin < 5 %)" if thin else "FAIL: " + ", ".join(failed)),
                         "contacts": sum(r["contact"] for r in per), "of": len(per), "infeasible ticks": sum(r["infeasible_ticks"] for r in per),
                         "min h [m]": f"{min(r['min_h'] for r in per):.3f}", "contacts within sampled allowance": sum(r["contact"] and r["within_sampled_allowance"] for r in per),
                         "contacts not attributable to sampled data": sum(r["contact"] and not r["within_sampled_allowance"] for r in per)})
        first_fail = next((r["f_max [N]"] for r in rows if r["Assumptions 7, 13, 14"].startswith("FAIL")), None)
        first_contact = next((r["f_max [N]"] for r in rows if r["contacts"] > 0), None)
        bad = [r["f_max [N]"] for r in rows if r["contacts not attributable to sampled data"] > 0 and not r["Assumptions 7, 13, 14"].startswith("FAIL")]
        verdict = "PASS (no contact outside the sampled-data attribution while the assumptions hold)" if not bad else f"FAIL (contacts at f_max = {bad} N with the assumptions holding)"
        tables.append(f"## f_max sweep (M8.3) — {verdict}: Assumption 7 first fails at f_max = {first_fail} N; first contact (sampled-data case) at {first_contact} N\n\n" + summary_table(rows, list(rows[0].keys())))
    # ---- mass mismatch (M8.4)
    if args.variant in ("mass", "all"):
        rows = []
        # the rule set (D-26): every constant re-derived for m_L = 1.2 kg except the vehicles' f_max, which stays 44 N (Assumption 7 holds, 4.4 % margin)
        for rule, pset, ov in (("nominal filter", "A", None), ("filter designed for m_L = 1.2 kg (D-26)", "A_mL12", {"f_max": float(p.f_max[0])})):
            pf = apply_overrides(load_set(pset), TrialConfig(param_overrides=ov))
            x0s = states(pf, n_m, 72)
            for m_true in (0.8, 1.2):
                cfgs = [TrialConfig(**base, params=pset, param_overrides=ov, m_L_true=m_true, seed=i, label=f"e7_mass{m_true:g}_{pset}_{i}") for i in range(len(x0s))]
                per = run_variant("mass", cfgs, x0s, pf, args.workers, out)
                results[f"mass_{m_true:g}_{pset}"] = {"m_L_true": m_true, "filter_set": pset, "per_trial": per}
                rows.append({"filter": rule, "true m_L [kg]": m_true, "contacts": sum(r["contact"] for r in per), "of": len(per),
                             "infeasible ticks": sum(r["infeasible_ticks"] for r in per), "min h [m]": f"{min(r['min_h'] for r in per):.3f}",
                             "contacts within sampled allowance": sum(r["contact"] and r["within_sampled_allowance"] for r in per),
                             "contacts not attributable to sampled data": sum(r["contact"] and not r["within_sampled_allowance"] for r in per)})
        with_rule = [r for r in rows if "1.2 kg" in r["filter"]]
        n_rule = sum(r["contacts"] for r in with_rule); n_rule_bad = sum(r["contacts not attributable to sampled data"] for r in with_rule)
        verdict = ("PASS" if n_rule == 0 else (f"CONDITIONAL ({n_rule} contacts with the rule, all sampled-data cases; none attributable to the mass error)" if n_rule_bad == 0
                   else f"FAIL ({n_rule_bad} contacts with the rule not attributable to sampled data)"))
        tables.append(f"## true payload mass +-20 % (M8.4) — {verdict}\n\n" + summary_table(rows, list(rows[0].keys())))
    # ---- soft cables (M8.5)
    if args.variant in ("soft", "all"):
        x0s = states(p, n_s, 73)
        cfgs = [TrialConfig(**base, cable_k_factor=0.1, seed=i, label=f"e7_soft_{i}") for i in range(len(x0s))]
        per = run_variant("soft", cfgs, x0s, p, args.workers, out)
        results["soft"] = {"cable_k_factor": 0.1, "per_trial": per}
        rows = [{"variant": "cable stiffness k/10 (zeta unchanged)", "contacts": sum(r["contact"] for r in per), "of": len(per), "infeasible ticks": sum(r["infeasible_ticks"] for r in per),
                 "trials with physical slack": sum(r["slack_ticks"] > 0 for r in per), "min h [m]": f"{min(r['min_h'] for r in per):.3f}",
                 "contacts within sampled allowance": sum(r["contact"] and r["within_sampled_allowance"] for r in per)}]
        tables.append("## softened cables (M8.5, outside the taut-cable model when slack occurs)\n\n" + summary_table(rows, list(rows[0].keys())))
    # ---- swing rates above omega_bar (M8.5)
    if args.variant in ("omega", "all"):
        x0s = []
        for st in states(p, n_o, 74):
            n_ = np.linalg.norm(st.qd, axis=1, keepdims=True)
            x0s.append(State(st.x_L.copy(), st.v_L.copy(), st.q.copy(), st.qd * (1.5 * p.omega_bar / np.maximum(n_, 1e-9))))
        cfgs = [TrialConfig(**base, seed=i, label=f"e7_omega_{i}") for i in range(len(x0s))]
        per = run_variant("omega", cfgs, x0s, p, args.workers, out)
        results["omega"] = {"qd_norm": 1.5 * p.omega_bar, "per_trial": per}
        rows = [{"variant": "|qdot_i| = 1.5 omega_bar at t = 0 (outside X_op; D may be undefined)", "contacts": sum(r["contact"] for r in per), "of": len(per),
                 "infeasible ticks": sum(r["infeasible_ticks"] for r in per), "min h [m]": f"{min(r['min_h'] for r in per):.3f}",
                 "max |qdot| over trial": f"{max(r['max_omega'] for r in per):.2f}"}]
        tables.append("## swing rates above omega_bar (M8.5, outside the operational set)\n\n" + summary_table(rows, list(rows[0].keys())))
    md = (f"# E7 stress list (Sec. VI limits; D-23) — paper's actuator model, adversarial nominal, H(x0) in [0.8, 1.3] m, {tf:g} s"
          f"{' (smoke)' if args.smoke else ''}; wall {time.time() - t0:.0f} s\n\n" + "\n\n".join(tables) + "\n")
    (out / ("summary_smoke.md" if args.smoke else "summary.md")).write_text(md)
    save_json(out / ("results_smoke.json" if args.smoke else "results.json"), {"variants": results, "params": p.to_dict()})
    print(md)


if __name__ == "__main__":
    main()
