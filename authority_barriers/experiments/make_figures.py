"""Core figures and tables for the five Table I rows (P6.S1), each with a JSON sidecar holding the
constants that Sec. VI requires with every figure and a caption stating the expected and the
refuting observation. Colors: validated categorical slots (blue, orange, aqua, yellow), fixed order;
one axis per panel; thin marks; direct labels; reference lines in neutral gray.

  python -m authority_barriers.experiments.make_figures --core [--actuator perfect]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from authority_barriers.theory.params import load_set
from authority_barriers.simulator.recorder import TrialLog

RES = Path(os.environ.get("SIM_RESULTS_DIR", Path(__file__).resolve().parents[2] / "results")).resolve()
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "gray": "#6f6e6a", "ink": "#0b0b0b"}
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "lines.linewidth": 1.2, "grid.linewidth": 0.4, "grid.color": "#dddcd8", "pdf.fonttype": 42})
COL_W = 3.45   # IEEE column width [in]


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def constants(p, extra=None):
    d = {"set": p.name, "N": p.N, "m_L": p.m_L, "m_i": p.m[0], "l_i": p.l[0], "T_min": p.T_min, "T_bar_i": p.T_bar[0],
         "rho_i": p.rho[0], "rho_fb": p.rho_fb[0], "nu": p.nu, "nu_w": p.nu_w, "a_max": p.a_max, "theta_q_deg": np.degrees(p.theta_q),
         "z_bar": p.z_bar, "w_bar": p.w_bar, "omega_bar": p.omega_bar, "f_max_i": p.f_max[0], "theta_max": "pi (ball)",
         "gamma": f"{p.kappa_H} H", "filter_dt": p.dt_filter, "attitude_dt": p.dt_attitude, "solver": "Clarabel",
         "integrator": "radau3, accuracy 1e-6, max step 1e-3", "alpha_sat": p.alpha_sat, "git": git_sha()}
    if extra:
        d.update(extra)
    return d


def save(fig, out: Path, name: str, caption: str, sidecar: dict):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    (out / f"{name}.json").write_text(json.dumps({"caption": caption, **sidecar}, indent=1, default=str))
    (out / f"{name}_caption.txt").write_text(caption)


def load_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


# ------------------------------------------------------------------ F1: Thm 5 (E1) — one file per gain pair and panel
def fig_e1(out: Path, actuator: str, p):
    folder = RES / "core" / f"e1_{actuator}"
    res = load_json(folder / "results.json")
    if res is None:
        return "E1: results missing"
    written = []
    for g in res["per_trial"]:
        k1, k2 = (float(x) for x in g.split(","))
        per = res["per_trial"][g]
        onsets = np.array([r["onset"] if r["onset"] is not None else np.nan for r in per])
        taus = np.array([r["tau"] for r in per])
        ratio = onsets / taus
        ok = np.isfinite(ratio)
        idx = int(np.nanargmin(np.abs(ratio - np.nanmedian(ratio)))) if ok.any() else 0     # representative trial: median onset/tau
        tag = f"k{k1:g}_{k2:g}".replace(".", "p")
        logs = sorted(folder.glob(f"e1_{tag}_*.npz"), key=lambda f: int(f.stem.split("_")[-1]))
        n_pass = sum(x["pass"] for x in per); n_contact = sum(x.get("termination") == "wall_contact" for x in per)
        base = constants(p, {"gains": [k1, k2], "trials": len(per), "pass": n_pass, "hocbf_wall_contacts": n_contact, "actuator": actuator})
        # (a) mu(t) and the proof bound of one representative trial
        if idx < len(logs):
            L = TrialLog.load(str(logs[idx])[:-4])
            t, mu, psi1 = L.d("t"), L.d("mu"), L.d("psi1")
            m = t <= min(1.5 * taus[idx], t[-1])
            fig, ax = plt.subplots(figsize=(COL_W, 2.1))
            ax.plot(t[m], mu[m], color=C["blue"], label=r"$\mu(t)$")
            ax.plot(t[m], k2 * psi1[m], color=C["orange"], ls="--", label=r"$\alpha_2(\psi_1(t))$")
            ax.axhline(0, color=C["gray"], lw=0.6)
            ax.axvline(per[idx]["t_psi"], color=C["gray"], ls=":", lw=0.8)
            ax.axvline(taus[idx], color=C["ink"], ls="-.", lw=0.8)
            y0, y1 = ax.get_ylim()
            ax.text(per[idx]["t_psi"], y0 + 0.06 * (y1 - y0), r"$t_\psi$", ha="right", va="bottom", fontsize=7, color=C["gray"])
            ax.text(taus[idx], y0 + 0.06 * (y1 - y0), r"$\tau$", ha="left", va="bottom", fontsize=7)
            inf = (L.d("feasible") < 0.5) & m
            if inf.any():
                ax.fill_between(t[m], y0, y1, where=inf[m], color=C["yellow"], alpha=0.25, lw=0)
                ax.set_ylim(y0, y1)
            ax.set_xlabel("t [s]"); ax.set_ylabel(r"[m/s$^2$]")
            ax.legend(frameon=False, loc="upper right", handlelength=1.6)
            fig.tight_layout()
            cap = (f"E1 (Thm. 5), gain pair (k1, k2) = ({k1:g}, {k2:g}) s^-1, actuator {actuator}: feasibility margin mu(t) of the HOCBF filter and the "
                   "proof bound alpha_2(psi_1(t)) for the representative trial (median onset/tau), with the time t_psi after which the condition demands "
                   "braking and the minimum time tau for a cable to leave Sigma_y; shaded: the program is infeasible. Expected if Thm. 5 holds: "
                   "infeasibility before tau at h > 0. Refuting: feasibility through tau with |qdot_i| <= omega_bar.")
            save(fig, out, f"fig_e1_thm5_{actuator}_{tag}_mu", cap, {**base, "representative_trial": int(idx)}); written.append("mu")
        # (b) distribution of the onset relative to tau
        fig, ax2 = plt.subplots(figsize=(COL_W, 2.0))
        r = np.sort(ratio[ok])
        ax2.step(r, np.arange(1, r.size + 1) / r.size, where="post", color=C["blue"])
        ax2.axvline(1.0, color=C["ink"], ls="-.", lw=0.8)
        ax2.set_xlim(0, max(1.1, r.max() * 1.05 if r.size else 1.1))
        ax2.set_xlabel(r"onset $t_{\rm inf}/\tau$"); ax2.set_ylabel("fraction of trials")
        ax2.text(0.04, 0.92, f"{n_pass}/{len(per)} before $\\tau$", transform=ax2.transAxes, fontsize=7, va="top")
        ax2.text(0.04, 0.78, f"HOCBF wall contacts: {n_contact}", transform=ax2.transAxes, fontsize=7, va="top")
        fig.tight_layout()
        cap = (f"E1 (Thm. 5), gain pair (k1, k2) = ({k1:g}, {k2:g}) s^-1, actuator {actuator}: empirical distribution of the onset of infeasibility "
               f"relative to tau over {len(per)} initial states in D. Expected if Thm. 5 holds: every onset before tau (the curve reaches 1 left of the "
               "dash-dotted line). Refuting: a trial feasible through tau with |qdot_i| <= omega_bar. Only linear class-K gains are tested (C-8).")
        save(fig, out, f"fig_e1_thm5_{actuator}_{tag}_onset", cap, base); written.append("onset")
    return f"E1 figures written ({len(written)} files, actuator {actuator})"


# ------------------------------------------------------------------ F2: Thm 12(i) (E2) — contact rate and contact speed, one file each
def fig_e2(out: Path, actuator: str, p):
    variants = {"A: H0 in [0, 0.05 D]": f"e2_{actuator}_max_off0", "C: H0 in [0, 1 m]": f"e2_{actuator}_max_off0_layer1"}
    data = {k: load_json(RES / "core" / v / "results.json") for k, v in variants.items()}
    data = {k: v for k, v in data.items() if v is not None}
    if not data:
        return "E2: results missing"
    cols = [C["blue"], C["orange"]]
    H_min = None
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    fig2, ax2 = plt.subplots(figsize=(COL_W, 2.2))
    for (name, d), col in zip(data.items(), cols):
        per = d["per_trial"]
        H0 = np.array([r["H0"] for r in per])
        cont = np.array([r["termination"] == "wall_contact" or r["min_h"] < 0 for r in per])
        edges = np.linspace(0.0, max(H0.max(), 1e-3), 9) if "1 m" in name else np.quantile(H0, np.linspace(0, 1, 6))
        centers, rate, nn = [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            sel = (H0 >= a) & (H0 <= b)
            if sel.sum():
                centers.append(0.5 * (a + b)); rate.append(cont[sel].mean()); nn.append(int(sel.sum()))
        ax.plot(centers, rate, marker="o", ms=3.5, color=col, label=name)
        for x, y, n in zip(centers, rate, nn):
            ax.text(x, y + 0.03, str(n), ha="center", va="bottom", fontsize=6, color=C["gray"])
        folder = RES / "core" / variants[name]
        v_end = []
        for r, c in zip(per, cont):
            if not c:
                continue
            if "v_end" in r:
                v_end.append(r["v_end"])
            elif (folder / f"{r['label']}.npz").exists():
                v_end.append(float(TrialLog.load(folder / r["label"]).d("v")[-1]))
        vs = np.sort(np.array(v_end, float)); vs = vs[np.isfinite(vs)]
        if vs.size:
            ax2.step(np.concatenate([[0.0], vs]), np.arange(0, vs.size + 1) / vs.size, where="post", color=col, label=f"{name} ({vs.size} contacts)")
        if "1 m" in name and cont.any():
            H_min = float(H0[cont].max())
    if H_min is not None:
        ax.axvline(H_min, color=C["ink"], ls="-.", lw=0.8)
        ax.text(H_min, 0.5, r"$H_{\min}$", ha="left", va="center", fontsize=7)
    ax.set_ylim(-0.02, 1.15); ax.set_xlabel(r"$H(x_0)$ [m]"); ax.set_ylabel("wall-contact rate")
    ax.legend(frameon=False, loc="center right"); fig.tight_layout()
    ax2.set_xlabel("contact speed [m/s]"); ax2.set_ylabel("fraction of contacts"); ax2.legend(frameon=False, loc="lower right"); fig2.tight_layout()
    rows = [d["row"] for d in data.values()]
    cap = ("E2 (Thm. 12(i)), adversarial nominal (full thrust toward the wall), sampled filter (17) at 200 Hz on the full-order plant, actuator "
           f"{actuator}: fraction of trials that touch the wall against the initial margin H(x_0) (numbers = trials per bin); the dash-dotted line "
           "marks the largest H(x_0) that still touched the wall, the measured sampled-data margin of Rem. 19. Expected if Thm. 12(i) holds for the "
           "sampled implementation on {H >= H_min}: zero contacts right of the line. Refuting: a contact with H(x_0) >= H_min and the filter feasible at every tick.")
    save(fig, out, f"fig_e2_thm12i_{actuator}_contact_rate", cap, constants(p, {"rows": rows, "H_min_measured": H_min, "actuator": actuator}))
    cap2 = (f"E2 (Thm. 12(i)), actuator {actuator}: distribution of the payload speed at wall contact for the two initial layers. Expected: contacts, "
            "when they occur by the sampled-data mechanism, at low speed (the filter has almost stopped the payload). Refuting: high-speed contacts from X_RF.")
    save(fig2, out, f"fig_e2_thm12i_{actuator}_contact_speed", cap2, constants(p, {"rows": rows, "actuator": actuator}))
    return f"E2 figures written (actuator {actuator})"


# ------------------------------------------------------------------ F3: kernel (E3) — one file per probed cable state
def fig_kernel(out: Path):
    files = glob.glob(str(RES / "core" / "e3_kernel" / "e3_kernel_*.json"))
    if not files:
        return "E3 kernel: results missing"
    d = min((load_json(f) for f in files), key=lambda dd: (dd["dx"][0], -int(np.prod(dd["shape"]))))   # finest grid (h cell, then nodes)
    rows = d["rows"]
    p1 = load_set("A1")
    states = sorted({(round(r["z"], 4), round(r["zd"], 4)) for r in rows})
    rest = [s for s in states if abs(s[1]) < 1e-9]
    pick = [rest[0], rest[len(rest) // 2], rest[-1]] + [s for s in states if s[1] > 0][:1] + [s for s in states if s[1] < 0][:1]
    ver = d["verdict"]
    for j, (z, zd) in enumerate(pick):
        sel = sorted([r for r in rows if abs(r["z"] - z) < 1e-3 and abs(r["zd"] - zd) < 1e-3], key=lambda r: r["v"])
        v = np.array([r["v"] for r in sel]); hs = np.array([r["h_star"] for r in sel])
        D = np.array([r["D"] for r in sel]); Dr = np.array([r["D_rel"] for r in sel])
        fig, ax = plt.subplots(figsize=(COL_W, 2.2))
        ax.plot(v, D, color=C["orange"], ls="--", label=r"$h=\mathsf{D}$ (inner bound)")
        ax.plot(v, hs, color=C["blue"], marker="o", ms=3, lw=1.0, label=r"$h^\star$ (kernel boundary)")
        ax.plot(v, Dr, color=C["aqua"], ls=":", label=r"$h=\mathsf{D}^{\rm rel}$ (outer bound)")
        ax.set_yscale("log"); ax.set_xlabel("v [m/s]"); ax.set_ylabel("h [m]")
        ax.set_title(rf"$z_1={z:+.2f}$, $\dot z_1={zd:+.2f}$ s$^{{-1}}$", fontsize=8)
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, fontsize=6, handlelength=1.6); fig.tight_layout()
        cap = (f"E3 (Thm. 12(ii)), planar one-cable system (set A1): viability kernel boundary h*(v) from the level-set computation on the grid "
               f"{tuple(d['shape'])} (h cell {d['dx'][0]:.3f} m) at the fixed cable state z_1 = {z:+.3f}, dz_1/dt = {zd:+.2f} s^-1, with the two closed-form "
               "bounds. Expected: D_rel <= h* <= D at every speed (the sandwich of Thm. 12); numerical dissipation can only raise h* and the boundary "
               "converges at first order in the grid (C-7). Refuting: h* < D_rel beyond one cell. Over all 200 probes: lower bound "
               f"{'holds' if ver['lower_bound_holds'] else 'VIOLATED'} (min h* - D_rel {-ver['max_violation_lower_m']:+.3f} m), upper "
               f"{'holds' if ver['upper_bound_holds'] else 'violated'} (max h* - D {ver['max_violation_upper_m']:+.3f} m).")
        save(fig, out, f"fig_e3_kernel_state{j}", cap, constants(p1, {"z": z, "zd": zd, "grid": d["shape"], "dx": d["dx"], "converged": d["converged"],
                                                                     "iterations": d["iterations"], "wall_s": d["wall"], "verdict": ver}))
    return f"E3 kernel figures written ({len(pick)} cable states)"


# ------------------------------------------------------------------ F4: collocation (E3) — one file per configuration
def fig_collocation(out: Path):
    d = load_json(RES / "core" / "e3_collocation.json")
    if d is None:                                   # aggregate not written yet: the finished pairs of the job cache
        jobs = sorted(glob.glob(str(RES / "core" / "e3_collocation_jobs" / "*.json")))
        if not jobs:
            return "E3 collocation: results missing"
        rows = [load_json(f) for f in jobs]
        d = {"rows": rows, "partial": f"{len(rows)} of 24 (configuration, speed) pairs finished", "n_solves": sum(r["n_solves"] for r in rows),
             "n_failures": sum(r["n_failures"] for r in rows)}
    rows = d.get("rows") or []
    if not rows:
        return "E3 collocation: no rows"
    desc = {"A": "all cables leaning toward the wall", "B": "mixed cable configuration", "C": "all cables at z_bar (saturated authority)"}
    for cfg in sorted({r["config"] for r in rows}):
        sel = sorted([r for r in rows if r["config"] == cfg], key=lambda r: r["v"])
        v = np.array([r["v"] for r in sel]); hc = np.array([r["h_c"] for r in sel], float)
        D = np.array([r["D"] for r in sel]); Dr = np.array([r["D_rel"] for r in sel])
        fig, ax = plt.subplots(figsize=(COL_W, 2.2))
        ax.plot(v, D, color=C["orange"], ls="--", label=r"$h=\mathsf{D}$")
        ax.plot(v, hc, color=C["blue"], marker="s", ms=3, lw=1.0, label=r"$h_c$ (collocation)")
        ax.plot(v, Dr, color=C["aqua"], ls=":", label=r"$h=\mathsf{D}^{\rm rel}$")
        ax.set_yscale("log"); ax.set_xlabel("v [m/s]"); ax.set_ylabel("h [m]"); ax.set_title(f"configuration {cfg}: {desc.get(cfg, '')}", fontsize=8)
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, fontsize=6, handlelength=1.6); fig.tight_layout()
        cap = (f"E3 (Thm. 12(ii)), N = 3 in three dimensions, configuration {cfg} ({desc.get(cfg, '')}): smallest initial distance h_c(v) from which "
               "direct collocation (41 knots, SNOPT, 6 bisection steps) finds an admissible trajectory that ends in X_RF, with the two bounds. Expected: "
               "D_rel <= h_c <= D; the optimizer under-approximates the kernel, so h_c is an upper bound on its boundary. Refuting: a verified trajectory "
               "from h_0 < D_rel." + (f" Partial: {d['partial']}." if d.get("partial") else ""))
        save(fig, out, f"fig_e3_collocation_{cfg}", cap, constants(load_set("A_N3"), {"rows": [{k: r[k] for k in ("v", "D", "D_rel", "h_c", "h_c_over_D", "M4_4", "witness_ok", "verify_passes")} for r in sel],
                                                                                        "collocation_meta": {k: v for k, v in d.items() if k not in ("rows", "results")}}))
    return "E3 collocation figures written (one per configuration)"


# ------------------------------------------------------------------ F5: Rem 18 (E5)
def fig_e5(out: Path, p):
    """Violation rate of min_t H_rob >= 0 against the initial robust margin H_rob(x0) under wind at d_bar,
    for the absolute-layer variants (H_rob(x0) uniform in [0, 1] m), both actuator modes."""
    variants = {"perfect actuator": "e5_scale1_layer1", "attitude loop (C-10)": "e5_scale1_attitude_layer1"}
    data = {k: load_json(RES / "core" / v / "results.json") for k, v in variants.items()}
    data = {k: v for k, v in data.items() if v is not None}
    if not data:
        return "E5: results missing"
    fig, ax = plt.subplots(1, 1, figsize=(COL_W, 2.0))
    cols = [C["blue"], C["orange"]]
    H_ref = None
    for (name, d), col in zip(data.items(), cols):
        per = d["per_trial"]
        H0 = np.array([r["H_rob0"] for r in per]); viol = np.array([not r["pass"] for r in per])
        edges = np.linspace(0.0, max(H0.max(), 1e-3), 9)
        centers, rate, nn = [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            sel = (H0 >= a) & (H0 <= b)
            if sel.sum():
                centers.append(0.5 * (a + b)); rate.append(viol[sel].mean()); nn.append(int(sel.sum()))
        ax.plot(centers, rate, marker="o", ms=3.5, color=col, label=name)
        for x, y, n in zip(centers, rate, nn):
            ax.text(x, y + 0.03, str(n), ha="center", va="bottom", fontsize=6, color=C["gray"])
        H_ref = d.get("h_ref", H_ref)
    if H_ref:
        ax.axvline(H_ref, color=C["ink"], ls="-.", lw=0.8)
        ax.text(H_ref, 0.5, r"$H_{\min}$ (E2)", ha="left", va="center", fontsize=7)
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlabel(r"$H^{\rm rob}(x_0)$ [m]"); ax.set_ylabel(r"rate of $\min_t H^{\rm rob} < 0$")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    caption = ("E5 (Rem. 18): robust filter (D_rob) under bounded wind at d_bar (first-order gusts clipped in norm), adversarial nominal, "
               "from states of X_RF^rob; fraction of trials in which H_rob becomes negative, against the initial robust margin "
               "(numbers = trials per bin); the dash-dotted line is the sampled-data state margin measured in E2. Expected if Rem. 18 "
               "holds for the sampled implementation: no violation right of the line with the perfect actuator. Refuting: a violation "
               "there with the filter feasible at every tick and the disturbance within budget. The attitude-loop curve is the reported "
               "inner-loop gap (its tracking error exceeds the budget of D-16, C-10).")
    save(fig, out, "fig_e5_rem18", caption, constants(p, {"rows": {k: v["row"] for k, v in data.items()},
                                                          "H_min_measured": {k: v.get("H_min_measured") for k, v in data.items()}}))
    return "E5 figure written"


# ------------------------------------------------------------------ table sidecars (Table I rows 4, 5 and E1b)
def table_sidecars(out: Path, p):
    """The Table I artifacts that are tables, not figures, get the same JSON sidecar (constants, expected and
    refuting observation) as the figures: Table Prop 17 (E4), Table Rem 18 (E5), Table E1b (proposed filter
    from the E1 states)."""
    written = []
    e4 = load_json(RES / "core" / "e4" / "results.json")
    if e4 is not None:
        cap = ("Table Prop. 17 (E4): median runtime of D with its gradients and of the filter (build + Clarabel solve) against N = 2..8, "
               "with the log-log slopes; the distributed decomposition of Prop. 17(b) (sum of the local rows against the coupled row) "
               "algebraically at 300 states and along the E2 logs with the logged command. Expected: N=8 runtime <= 1 ms, slope <= 1.3, "
               "identity to 1e-10 (algebraic) and 1e-8 (logs). Refuting: a mismatch or superlinear growth.")
        (out / "table_e4_prop17.json").write_text(json.dumps({"caption": cap, **constants(p, {"results": e4, "table": (RES / "core" / "e4" / "summary.md").read_text()})}, indent=1, default=str))
        (out / "table_e4_prop17_caption.txt").write_text(cap); written.append("E4")
    rows = {}
    for v in ("e5_scale1", "e5_scale1_layer1", "e5_scale1_attitude", "e5_scale1_attitude_layer1"):
        d = load_json(RES / "core" / v / "results.json")
        if d is not None:
            rows[v] = {"row": d["row"], "verdict": d["verdict"], "H_min_measured": d.get("H_min_measured"), "table": (RES / "core" / v / "summary.md").read_text()}
    if rows:
        cap = ("Table Rem. 18 (E5): robust filter (D_rob) under wind clipped at d_bar (D-16), adversarial nominal, from states of X_RF^rob; "
               "min_t H_rob per trial, feasibility, model tension floor. Expected: min_t H_rob >= 0 in every trial (with the sampled-data "
               "state margin of Rem. 19 measured in E2, on {H_rob(x0) >= H_min}) with the paper's actuator model. Refuting: a loss of "
               "H_rob >= 0 with the filter feasible at every tick, the disturbance within budget and H_rob(x0) >= H_min. The attitude-loop "
               "rows report the inner-loop gap (C-10).")
        (out / "table_e5_rem18.json").write_text(json.dumps({"caption": cap, **constants(p, {"variants": rows})}, indent=1, default=str))
        (out / "table_e5_rem18_caption.txt").write_text(cap); written.append("E5")
    e1 = {act: load_json(RES / "core" / f"e1_{act}" / "results.json") for act in ("perfect", "attitude")}
    e1 = {k: v for k, v in e1.items() if v is not None}
    n3 = load_json(RES / "core" / "e1_perfect_A_N3" / "results.json")
    if e1:
        cap = ("Table E1b (Thm. 12(i)): the proposed filter from the same 300 initial states as E1 (per gain pair); trials whose "
               "state lies in X_RF must keep min h >= 0. Expected: 100 % of the X_RF states safe. Refuting: a wall contact from X_RF. "
               "States outside X_RF carry no claim (their outcome is reported).")
        (out / "table_e1b_thm12i.json").write_text(json.dumps({"caption": cap, **constants(p, {"rows": {k: v["rows"] for k, v in e1.items()},
                                                                                                 "N3_replication_rows": n3["rows"] if n3 else None})}, indent=1, default=str))
        (out / "table_e1b_thm12i_caption.txt").write_text(cap); written.append("E1b")
    return f"table sidecars written: {written}"


# ------------------------------------------------------------------ tables
ORDER = ["e1_perfect", "e1_attitude", "e1_perfect_A_N3", "e2_perfect_max_off0", "e2_perfect_max_off0_layer1", "e2_perfect_lmax_off0",
         "e2_attitude_max_off0", "e2_attitude_max_off0_layer1", "e3_kernel_summary.md", "e3_collocation_summary.md", "e4",
         "e5_scale1", "e5_scale1_layer1", "e5_scale1_attitude", "e5_scale1_attitude_layer1"]


def tables(out: Path):
    """Index of every summary table under results/core (the pre-declared ones first, missing ones flagged)."""
    core = RES / "core"
    found = {f.parent.name: f for f in core.glob("*/summary.md")} | {f.name: f for f in core.glob("*_summary.md")}
    names = ORDER + sorted(k for k in found if k not in ORDER)
    lines = ["# Core results index", "", f"commit {git_sha()}; one plot per file; every figure carries a JSON sidecar with the constants and a caption stating the expected and the refuting observation.", "",
             "Trial-level figures (snapshots of the team at the key instants and one time series per signal): [trials/index.md](trials/index.md) (`python -m authority_barriers.experiments.trial_figures`).", ""]
    for name in names:
        f = found.get(name)
        lines.append(f"## {name}\n")
        lines.append(f.read_text() if f is not None else "_missing_\n")
    (out / "index.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", action="store_true")
    ap.add_argument("--actuator", default="perfect,attitude", help="comma-separated actuator modes for the E1/E2 figures")
    args = ap.parse_args()
    out = RES / "core" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    p = load_set("A")
    fns = []
    for act in args.actuator.split(","):
        fns += [lambda act=act: fig_e1(out, act, p), lambda act=act: fig_e2(out, act, p)]
    fns += [lambda: fig_kernel_wrap(out), lambda: fig_collocation(out), lambda: fig_e5(out, p), lambda: table_sidecars(out, p)]
    for fn in fns:
        try:
            print(fn())
        except Exception as e:  # a missing input must not stop the other figures
            print(f"figure skipped: {type(e).__name__}: {e}")
    tables(out)
    print(f"index: {out / 'index.md'}")


def fig_kernel_wrap(out):
    return fig_kernel(out)


if __name__ == "__main__":
    main()
