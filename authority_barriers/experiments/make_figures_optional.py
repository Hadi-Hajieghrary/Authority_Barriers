"""Optional-track figures (P9.S1), one plot per file with JSON sidecar and caption, from the results under
results/optional: E6 (backup filter: command difference and cost ratio), E6 distributed (|Delta H| and a_hat
mismatch), E3b (6-D kernel boundary against the bounds), E7 (f_max sweep with the Assumption 7 threshold, mass and
stiffness table sidecar, corridor thrust-change distribution, Monte Carlo violation rates and contact rate against
f_max), gusts at 2 d_bar (violation rate against H_rob(x0)).

  python -m authority_barriers.experiments.make_figures_optional
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from authority_barriers.theory.params import load_set
from authority_barriers.experiments.make_figures import C, COL_W, RES, constants, git_sha, load_json, save

OPT = RES / "optional"


def fig_e6_backup(out, p):
    d = load_json(OPT / "e6_backup" / "results.json")
    if d is None:
        return "E6 backup: missing"
    rows = d["rows"]
    du = np.array([r["du_frac_fmax"] for r in rows]) * 100.0
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    ax.hist(du, bins=20, color=C["blue"])
    ax.axvline(1.0, color=C["ink"], ls="-.", lw=0.8); ax.text(1.0, ax.get_ylim()[1] * 0.9, " 1 % f_max", fontsize=7)
    ax.set_xlabel(r"$\max_i\|u_i^{\rm backup}-u_i^{\rm proposed}\|$ [% of $f_{\max}$]"); ax.set_ylabel("states")
    fig.tight_layout()
    cap = (f"E6 (Sec. VI Q3, M7.1): difference between the commands of the numerically integrated backup-CBF filter (RK45, rtol 1e-8, central "
           f"differences) and of the proposed filter from the same {len(rows)} X_RF states and the same adversarial nominal, as a fraction of f_max. "
           "Expected: within 1 % (the two filters coincide up to integration error). Refuting: systematic differences.")
    save(fig, out, "fig_e6_backup_command_difference", cap, constants(p, {"n": len(rows), "du_max_pct": float(du.max()), "du_median_pct": float(np.median(du))}))
    ratio = np.array([r["t_backup"] / r["t_proposed"] for r in rows]); calls = np.array([r["policy_calls"] / max(r["n_breakpoints"], 1) for r in rows])
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    ax.scatter(calls, ratio, s=8, color=C["orange"])
    ax.set_xscale("log"); ax.set_yscale("log")
    lo = min(calls.min(), ratio.min()) * 0.5; hi = max(calls.max(), ratio.max()) * 2
    ax.plot([lo, hi], [lo, hi], color=C["gray"], lw=0.6, ls="--")
    ax.set_xlabel("integrator calls per evaluation / breakpoints"); ax.set_ylabel("wall time, backup / proposed")
    fig.tight_layout()
    cap = ("E6 (Sec. VI Q3): wall-time ratio of one backup-filter evaluation to one proposed-filter evaluation against the ratio of integrator "
           "policy calls to breakpoints of the closed form (dashed: equality). Expected: the proposed filter faster by the ratio of integration steps to breakpoints.")
    save(fig, out, "fig_e6_backup_cost_ratio", cap, constants(p, {"ratio_median": float(np.median(ratio)), "calls_ratio_median": float(np.median(calls))}))
    return "E6 backup figures written"


def fig_e6_distributed(out, p):
    d = load_json(OPT / "e6_distributed" / "results.json")
    if d is None:
        return "E6 distributed: missing"
    per = d["per_trial"]
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    x = np.arange(len(per))
    ax.bar(x - 0.2, [r["central"]["min_h"] for r in per], 0.4, color=C["blue"], label="centralized")
    ax.bar(x + 0.2, [r["distributed"]["min_h"] for r in per], 0.4, color=C["orange"], label="distributed (lagged $\\hat a$)")
    ax.axhline(0, color=C["ink"], lw=0.6); ax.set_xlabel("trial"); ax.set_ylabel("min h [m]"); ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    cap = ("E6 distributed (M7.2, D-27): minimum distance to the wall over 10 s under the adversarial nominal, centralized filter against the distributed "
           "implementation of Prop. 17(b) with a one-step-lagged broadcast of a_hat, from the same states with H(x0) in [0.8, 1.3] m. Expected: equal within the "
           "budget charged to the a_hat mismatch. Refuting: contacts of the distributed variant with the mismatch beyond budget (then the broadcast is not a commitment, Prop. 17(c)).")
    save(fig, out, "fig_e6_distributed_min_h", cap, constants(p, {"row": d["row"], "verdict": d["verdict"]}))
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    ax.bar(x, [r["distributed"]["a_hat_err_force_max_N"] for r in per], 0.6, color=C["orange"])
    ax.axhline(0.05, color=C["ink"], ls="-.", lw=0.8); ax.text(0, 0.05, " budget 0.05 N", fontsize=7, va="bottom")
    ax.set_yscale("log"); ax.set_xlabel("trial"); ax.set_ylabel(r"$\max_t\, m_i\|\hat a - a\|$ [N]")
    fig.tight_layout()
    cap = "E6 distributed (M7.2): largest force-equivalent mismatch between the broadcast a_hat (previous tick's commanded specific force) and the applied one, per trial, against the 0.05 N budget of D-16."
    save(fig, out, "fig_e6_distributed_ahat_mismatch", cap, constants(p, {"row": d["row"]}))
    return "E6 distributed figures written"


def fig_e3b(out):
    files = sorted(glob.glob(str(OPT / "e3b_kernel6d" / "e3b_kernel6d_*.json")))
    if not files:
        return "E3b: missing"
    d = load_json(files[-1]); rows = d["rows"]; p2 = load_set("A2")
    keys = sorted({(round(r["z1"], 3), round(r["zd1"], 2), round(r["z2"], 3), round(r["zd2"], 2)) for r in rows})
    written = 0
    for j, key in enumerate(keys[:6]):
        sel = sorted([r for r in rows if (round(r["z1"], 3), round(r["zd1"], 2), round(r["z2"], 3), round(r["zd2"], 2)) == key], key=lambda r: r["v"])
        v = np.array([r["v"] for r in sel]); hs = np.array([r["h_star"] for r in sel]); D = np.array([r["D"] for r in sel]); Dr = np.array([r["D_rel"] for r in sel])
        fig, ax = plt.subplots(figsize=(COL_W, 2.2))
        ax.plot(v, D, color=C["orange"], ls="--", label=r"$h=\mathsf{D}$ (inner bound)")
        ax.plot(v, hs, color=C["blue"], marker="o", ms=3, lw=1.0, label=r"$h^\star$ (6-D kernel, inner input set)")
        ax.plot(v, Dr, color=C["aqua"], ls=":", label=r"$h=\mathsf{D}^{\rm rel}$ (outer bound)")
        ax.set_yscale("log"); ax.set_xlabel("v [m/s]"); ax.set_ylabel("h [m]")
        ax.set_title(rf"$z=({key[0]:+.2f},{key[2]:+.2f})$, $\dot z=({key[1]:+.2f},{key[3]:+.2f})$", fontsize=8)
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, fontsize=6, handlelength=1.6); fig.tight_layout()
        cap = (f"E3b (Thm. 12(ii), Sec. VI Q2, two cables): kernel boundary h*(v) of the planar two-cable system (set A2) on the coarse 6-D grid {tuple(d['shape'])} "
               f"(cells: h {d['dx'][0]:.2f} m, v {d['dx'][1]:.2f} m/s, z {d['dx'][2]:.3f}, zd {d['dx'][3]:.3f} s^-1) at the swing state z = ({key[0]:+.3f}, {key[2]:+.3f}), "
               f"zd = ({key[1]:+.2f}, {key[3]:+.2f}) s^-1, with the two bounds. The input set is inner-approximated per cable (D-25), so the kernel is under-approximated "
               f"and h* is biased upward on top of the dissipation bias (C-3, C-7). Verdict over all probes: lower bound {'holds' if d['verdict']['lower_bound_holds'] else 'VIOLATED'}, "
               f"upper {'holds' if d['verdict']['upper_bound_holds'] else 'exceeded at ' + str(d['verdict']['probes_above_D_plus_tol']) + ' probes'} (tolerance 2 cells).")
        save(fig, out, f"fig_e3b_kernel6d_state{j}", cap, constants(p2, {"grid": d["shape"], "dx": d["dx"], "converged": d["converged"], "iterations": d["iterations"], "wall_s": d["wall"], "verdict": d["verdict"], "state": key}))
        written += 1
    return f"E3b figures written ({written})"


def fig_e7_fmax(out, p):
    d = load_json(OPT / "e7_stress" / "results.json")
    if d is None:
        return "E7 stress: missing"
    var = {k: v for k, v in d["variants"].items() if k.startswith("fmax_")}
    if not var:
        return "E7 f_max: missing"
    fs = sorted(v["f_max"] for v in var.values())
    rate = [np.mean([r["contact"] for r in var[f"fmax_{f:g}"]["per_trial"]]) for f in fs]
    hold = [var[f"fmax_{f:g}"]["assumptions_hold"] for f in fs]
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    ax.plot(fs, rate, marker="o", ms=3.5, color=C["blue"])
    thr = max([f for f, h in zip(fs, hold) if not h], default=None)
    if thr is not None:
        ax.axvspan(min(fs) - 1, thr + 1, color=C["yellow"], alpha=0.2, lw=0)
        ax.text(thr + 1, 0.95, "Assumption 7 fails", fontsize=7, ha="left", va="top")
    ax.set_ylim(-0.02, 1.05); ax.set_xlabel(r"$f_{\max}$ [N]"); ax.set_ylabel("wall-contact rate")
    fig.tight_layout()
    cap = ("E7 (M8.3): wall-contact rate of the proposed filter (paper's actuator model, adversarial nominal, H(x0) in [0.8, 1.3] m, 10 states per value) as f_max "
           "is reduced from 44 to 20 N with the plant and the filter consistent; shaded: the values at which Assumption 7 (or 13, 14) fails for set A. "
           "Expected: contacts only where the assumptions fail. Refuting: contacts while they hold.")
    save(fig, out, "fig_e7_fmax_sweep", cap, constants(p, {"f_max": fs, "contact_rate": rate, "assumptions_hold": hold}))
    return "E7 f_max figure written"


def fig_e7_corridor(out, p):
    d = load_json(OPT / "e7_corridor" / "results.json")
    if d is None:
        return "E7 corridor: missing"
    per = d["per_trial"]
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    med = np.array([r["rel_change_median"] for r in per]) * 100; p95 = np.array([r["rel_change_p95"] for r in per]) * 100
    ax.hist(med, bins=15, color=C["blue"], alpha=0.8, label="trial median")
    ax.hist(p95, bins=15, color=C["orange"], alpha=0.6, label="trial 95th percentile")
    ax.axvline(5.0, color=C["ink"], ls="-.", lw=0.8); ax.text(5.0, ax.get_ylim()[1] * 0.9, " 5 %", fontsize=7)
    ax.set_xlabel(r"$\|u-u^{\rm nom}\|/\|u^{\rm nom}\|$ per tick [%]"); ax.set_ylabel("trials"); ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    cap = (f"E7 corridor (M8.2, D-7): relative change of the nominal thrust by the two-wall filter during nominal transport along a 20 m corridor, {len(per)} random "
           "states in X_RF of both walls, 15 s; per-trial median and 95th percentile of the per-tick value. Expected: at most 5 %. Refuting: systematic changes "
           f"above 5 % (D-7: the practicality sentence is amended). Contacts: {d['row']['contacts (either wall)']}.")
    save(fig, out, "fig_e7_corridor_thrust_change", cap, constants(p, {"row": d["row"], "verdict": d["verdict"]}))
    return "E7 corridor figure written"


def fig_e7_mc(out, p):
    d = load_json(OPT / "e7_montecarlo" / "results.json")
    if d is None:
        return "E7 Monte Carlo: missing"
    rows = d["rows"]
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    names = [r["baseline"].split(" (")[0].split(",")[0] for r in rows]
    rates = [float(r["contact rate"]) for r in rows]
    ax.bar(np.arange(len(rows)), rates, color=[C["gray"], C["orange"], C["yellow"], C["blue"]][:len(rows)])
    ax.set_xticks(np.arange(len(rows))); ax.set_xticklabels(names, fontsize=6, rotation=15)
    for i, r in enumerate(rows):
        ax.text(i, rates[i] + 0.02, r["contacts within the hypotheses (Ass. 7, x0 in X_RF, gusts <= d_bar, stiff cables, true m_L <= nominal)"], ha="center", fontsize=6)
    ax.set_ylim(0, 1.15); ax.set_ylabel("wall-contact rate (500 trials)")
    fig.tight_layout()
    cap = ("E7 Monte Carlo (M8.6, D-28): wall-contact rate of the four baselines over 500 trials with randomly drawn stress factors (gusts up to 2 d_bar, f_max 20-44 N, "
           "true mass +-20 %, soft cables, initial swing rates up to 1.5 omega_bar; velocity transport at 1-4 m/s toward the wall, 10 s); the label above each bar counts the "
           "contacts among the trials in which every hypothesis of the theorems holds. Expected: the proposed filter without contacts within its hypotheses; the HOCBF "
           "filters with infeasible instances. The backup filter is not run in closed loop (D-24).")
    save(fig, out, "fig_e7_montecarlo_contact_rate", cap, constants(p, {"rows": rows}))
    prop = d["per_baseline"].get("proposed")
    if prop:
        f = np.array([r["f_max"] for r in prop]); c = np.array([r["contact"] for r in prop]); ok = np.array([r["assumptions_hold"] and r["in_XRF"] and r["gust"] <= 1.0 and r["k_factor"] == 1.0 and r["mass_ok"] for r in prop])
        edges = np.linspace(20, 44, 9); xc, rt, rt_ok, nn = [], [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            s = (f >= a) & (f <= b)
            if s.sum():
                xc.append(0.5 * (a + b)); rt.append(c[s].mean()); nn.append(int(s.sum())); rt_ok.append(c[s & ok].mean() if (s & ok).sum() else np.nan)
        fig, ax = plt.subplots(figsize=(COL_W, 2.1))
        ax.plot(xc, rt, marker="o", ms=3.5, color=C["blue"], label="all trials")
        ax.plot(xc, rt_ok, marker="s", ms=3.5, color=C["aqua"], label="hypotheses satisfied")
        for x, y, n in zip(xc, rt, nn):
            ax.text(x, y + 0.03, str(n), ha="center", fontsize=6, color=C["gray"])
        ax.set_ylim(-0.02, 1.1); ax.set_xlabel(r"$f_{\max}$ [N]"); ax.set_ylabel("contact rate, proposed filter"); ax.legend(frameon=False, fontsize=6)
        fig.tight_layout()
        cap = "E7 Monte Carlo (M8.6): contact rate of the proposed filter against the drawn f_max, all trials and the trials in which every hypothesis holds (numbers: trials per bin)."
        save(fig, out, "fig_e7_montecarlo_proposed_vs_fmax", cap, constants(p, {"bins": xc, "rate": rt, "rate_ok": rt_ok}))
    return "E7 Monte Carlo figures written"


def fig_e7_gusts(out, p):
    files = sorted(glob.glob(str(OPT / "e5_scale2*" / "results.json")))
    if not files:
        return "E7 gusts: missing"
    d = load_json(files[0]); per = d["per_trial"]
    H0 = np.array([r["H_rob0"] for r in per]); viol = np.array([not r["pass"] for r in per])
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    edges = np.linspace(H0.min(), H0.max() + 1e-6, 6); xc, rt, nn = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        s = (H0 >= a) & (H0 <= b)
        if s.sum():
            xc.append(0.5 * (a + b)); rt.append(viol[s].mean()); nn.append(int(s.sum()))
    ax.plot(xc, rt, marker="o", ms=3.5, color=C["orange"])
    for x, y, n in zip(xc, rt, nn):
        ax.text(x, y + 0.03, str(n), ha="center", fontsize=6, color=C["gray"])
    ax.set_ylim(-0.02, 1.1); ax.set_xlabel(r"$H^{\rm rob}(x_0)$ [m]"); ax.set_ylabel(r"rate of $\min_t H^{\rm rob}<0$")
    fig.tight_layout()
    cap = (f"E7 gusts at 2 d_bar (M8.1, no claim: Rem. 18's budget deliberately violated): rate at which the robust filter loses H_rob >= 0 against the initial robust "
           f"margin ({len(per)} states, paper's actuator model, adversarial nominal). Reported alongside E5 at d_bar for comparison.")
    save(fig, out, "fig_e7_gusts_2dbar", cap, constants(p, {"row": d["row"], "verdict": d["verdict"]}))
    return "E7 gusts figure written"


def main():
    out = RES / "optional" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    p = load_set("A")
    lines = ["# Optional-track figures index", "", f"commit {git_sha()}; one plot per file, JSON sidecar and caption per figure.", ""]
    for fn in (lambda: fig_e6_backup(out, p), lambda: fig_e6_distributed(out, p), lambda: fig_e3b(out), lambda: fig_e7_fmax(out, p),
               lambda: fig_e7_corridor(out, p), lambda: fig_e7_mc(out, p), lambda: fig_e7_gusts(out, p)):
        try:
            msg = fn()
        except Exception as e:
            msg = f"figure skipped: {type(e).__name__}: {e}"
        print(msg); lines.append(f"- {msg}")
    for f in sorted(OPT.glob("*/summary.md")):
        lines += ["", f"## {f.parent.name}", "", f.read_text()]
    for f in sorted(OPT.glob("e3b_kernel6d/*_summary.md")):
        lines += ["", f"## {f.stem}", "", f.read_text()]
    (out / "index.md").write_text("\n".join(lines) + "\n")
    print(f"index: {out / 'index.md'}")


if __name__ == "__main__":
    main()
