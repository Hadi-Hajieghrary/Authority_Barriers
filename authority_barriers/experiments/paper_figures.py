"""Argument-carrying figures for the paper, built from the recorded logs and the theory library.
ONE PLOT OR ONE FRAME PER FILE: a figure of the paper is a group of files that the paper sets side by side or stacks.
  1 anatomy of the stopping distance   (1a swings, 1b authority, 1c braking profile)
  2 HOCBF vs authority barrier         (eight side-view sketches, 2a min z, 2b feasibility margin, 2c distance)
  3 storyboard of one maneuver         (six rendered frames, 3a distances, 3b swing, 3c tensions)
  4 the viability sandwich             (4a, 4b, 4c: configurations A, B, C)
  5 the sampled-data margin            (5a map, 5b mechanism)
  6 where the thrust goes              (6a thrust budget, 6b f_max sweep)
  7 D(nu)/h_c against nu               (one plot)
Production rules: vector PDF at the printed width, 8 pt fonts, direct labels, one palette; plots of one group share
their margins so that they align when stacked.
    python -m authority_barriers.experiments.paper_figures [--out results/core/figures/paper] [--no-render] [--only 1,3]
Outputs <out>/paper_fig<k><letter>_<name>.{pdf,png}, the frames as .png, and captions.md (one caption per group with its
files); IEEE_ACC2027/collect_figures.py takes the files the paper includes into IEEE_ACC2027/figures/."""
from __future__ import annotations

import argparse, glob, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from authority_barriers.theory import hocbf as HB, profile as PF, stopping as SD
from authority_barriers.theory.authority import BarrierData
from authority_barriers.theory.params import load_set
from authority_barriers.theory.state import State
from authority_barriers.theory.taut_model import rhs_from_thrust, unpack
from authority_barriers.simulator.recorder import TrialLog
from authority_barriers.experiments.common import RESULTS

COL, FULL = 3.5, 7.16
Q = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]           # quadrotors 1-4, fixed across the paper
CH, CD, CINK, CGRAY, CRED, CWALL = "#7b3fbf", "#0b0b0b", "#0b0b0b", "#8a8985", "#d7263d", "#b9b8b4"
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.0, "pdf.fonttype": 42, "figure.dpi": 150})
LINES = {"axes.linewidth": 0.8, "lines.linewidth": 1.0, "grid.color": "#b0b0b0", "grid.linewidth": 0.8}              # groups 1, 3, 4
LINES_TRIAL = {"axes.linewidth": 0.6, "lines.linewidth": 1.2, "grid.color": "#dddcd8", "grid.linewidth": 0.4}       # groups 2, 5, 6, 7: as in the trial figures
P = load_set("A"); DATA = BarrierData.nominal(P); N = P.N
CAPTIONS = {}                                                                 # group -> (files, caption)

def state_of(L: TrialLog, k: int) -> State:
    return unpack(L.taut[:, k], N)

def x0_of(L: TrialLog) -> State:
    x0 = L.meta["x0"]; return State(*(np.array(x0[key]) for key in ("x_L", "v_L", "q", "qd")))

def label_right(ax, x, y, text, color, dy=0.0):
    ax.annotate(text, (x, y), xytext=(3, dy), textcoords="offset points", color=color, fontsize=7, va="center", ha="left", annotation_clip=False)

def one(w: float, h: float, margins=None):
    """A figure with a single plot; `margins` (left, right, bottom, top) fixes the plot area so that plots of a group align."""
    fig, ax = plt.subplots(figsize=(w, h))
    if margins: fig.subplots_adjust(left=margins[0], right=margins[1], bottom=margins[2], top=margins[3])
    return fig, ax

def save(fig, out: Path, name: str, tight: bool = False):
    if tight: fig.tight_layout()
    fig.savefig(out / f"{name}.pdf"); fig.savefig(out / f"{name}.png", dpi=220); plt.close(fig)
    print("wrote", name, flush=True); return name

def group(key: str, files, caption: str):
    CAPTIONS[key] = (list(files), caption)

# ------------------------------------------------------------------ 1. anatomy of D
def fig1_anatomy(out: Path):
    L75 = TrialLog.load(RESULTS / "core" / "e2_perfect_max_off0_layer1" / "e2_perfect_max_off0_layer1_75")
    L72 = TrialLog.load(RESULTS / "core" / "e2_perfect_max_off0" / "e2_perfect_max_off0_72")
    H72 = L72.d("H"); k = int(np.argmax(H72[:-1] - H72[1:]))                  # the tick with the largest one-period loss of H
    cases = [(x0_of(L75), float(L75.d("h")[0]), "-", 1.0), (state_of(L72, k + 1), float(L72.d("h")[k + 1]), "--", 0.8)]
    M = (0.17, 0.86, 0.27, 0.95)
    (f0, a0), (f1, a1), (f2, a2) = one(COL, 1.55, M), one(COL, 1.45, M), one(COL, 1.55, M)
    aE = a2.twinx(); aE.spines["top"].set_visible(False)
    t_end = 0.0
    for st, h, ls, lw in cases:
        v, z, zd = st.v(P), st.z(P), st.zd(P)
        mn = SD.build(v, z, zd, DATA); res = SD.evaluate(v, z, zd, DATA)
        T = float(max(mn.t2.max(), res.t_star.max())) + 0.4; t_end = max(t_end, T)
        t = np.linspace(0.0, T, 600)
        zt = mn.zeta_tilde(t)                                                 # (N, len t)
        for i in range(N):
            a0.plot(t, zt[i], ls=ls, lw=lw, color=Q[i])
            if ls == "-":
                a0.plot(mn.t1[i], PF.zeta_tilde(mn.t1[i], z[i], zd[i], DATA.nu, DATA.zeta_bar)[0], "o", mfc="white", mec=Q[i], ms=4)
                a0.plot(mn.t2[i], PF.zeta_tilde(mn.t2[i], z[i], zd[i], DATA.nu, DATA.zeta_bar)[0], "o", color=Q[i], ms=4)
                for tc in PF.zero_crossings(z[i], zd[i], DATA.nu, DATA.zeta_bar):
                    a0.plot(tc, 0.0, "D", mfc="white", mec=Q[i], ms=4)
        al = mn.alpha_tilde(t)
        a1.plot(t, al, ls=ls, lw=lw, color=CINK)
        if ls == "-":
            a1.fill_between(t, al, 0.0, where=al < 0, color=CRED, alpha=0.18, lw=0)
        vt, E = mn.v_tilde(t), mn.E_of(t)
        a2.plot(t, vt, ls=ls, lw=lw, color=CGRAY); aE.plot(t, E, ls=ls, lw=lw, color=CH)
        for ts in res.t_star:
            aE.plot(ts, mn.E_of(ts), "v", color=CH, ms=5, mfc="white" if ls == "--" else CH)
        if ls == "-":
            aE.axhline(h, color=CD, lw=0.8, ls=":")
            ts = float(res.t_star[0])
            aE.annotate("", (ts, h), (ts, res.D), arrowprops=dict(arrowstyle="<->", color=CH, lw=0.8))
            aE.text(ts - 0.04, res.D + 0.65 * (h - res.D), "H = h − D ", color=CH, fontsize=7, ha="right", va="center")
            label_right(aE, T, h, "h", CD); aE.text(ts + 0.08, res.D, "D = E(t*)", color=CH, fontsize=7, va="top", ha="left")
            label_right(a2, T, vt[-1], "ṽ(t)", CGRAY)
            label_right(a1, T, al[-1], "α̃(t)", CINK)
            label_right(a0, T, DATA.zeta_bar, "z̃_i → z̄", CGRAY)
    a0.axhspan(-DATA.zeta_bar, DATA.zeta_bar, color=CGRAY, alpha=0.10, lw=0); a0.axhline(0, color=CGRAY, lw=0.5)
    a0.set_ylabel("swing $\\tilde z_i$"); a1.set_ylabel("$\\tilde\\alpha$ [m/s$^2$]"); a2.set_ylabel("$\\tilde v$ [m/s]"); aE.set_ylabel("E [m]", color=CH)
    asat = DATA.alpha(np.full(N, DATA.zeta_bar)); a1.axhline(asat, color=CGRAY, lw=0.5, ls=":"); a1.text(0.02, asat, " α_sat", fontsize=7, color=CGRAY, va="top")
    a1.text(0.02, a1.get_ylim()[0], " shaded: payload still accelerating toward the wall", fontsize=6.5, color=CRED, va="bottom")
    for ax in (a0, a1, a2): ax.set_xlabel("t [s]"); ax.set_xlim(0, t_end)
    files = [save(f0, out, "paper_fig1a_swings"), save(f1, out, "paper_fig1b_authority"), save(f2, out, "paper_fig1c_profile")]
    group("fig1_anatomy_of_D", files, "The stopping distance is the maximum of the braking profile E(t), whose maximizers are born and die with the cables' swings: (a) the time-optimal swings z̃_i (○ switch, ● arrival at z̄, ◇ zero crossing), (b) the braking authority α̃ = α_y(z̃) (negative while the cables lean toward the wall, shaded), (c) the predicted speed ṽ and distance E with t* and D = E(t*) (h dotted, the gap is H). Solid: the boundary-layer state of trial e2_perfect_max_off0_layer1_75 at t = 0; dashed: trial 72 one tick after its largest one-period loss of H, where a second maximum of E has appeared and overtaken the first.")

# ------------------------------------------------------------------ 2. HOCBF vs authority barrier
def sketch(ax, st: State, u: np.ndarray, p, title=None):
    n, e3 = p.n_vec, np.array([0, 0, 1.0])
    xh = lambda x: float(n @ x - p.d0); xz = lambda x: float(e3 @ x)
    xl = st.x_L; ax.plot([xh(xl)], [xz(xl)], "o", color=CINK, ms=5, zorder=5)
    for i in range(p.N):
        pi = xl + p.l_arr[i] * st.q[i]
        ax.plot([xh(xl), xh(pi)], [xz(xl), xz(pi)], color=CGRAY, lw=0.8)
        ax.plot([xh(pi)], [xz(pi)], "s", color=Q[i], ms=4)
        ui = u[i] / p.f_max_arr[i] * 0.6                                     # thrust at f_max drawn 0.6 m long
        ax.annotate("", (xh(pi) + n @ ui, xz(pi) + e3 @ ui), (xh(pi), xz(pi)), arrowprops=dict(arrowstyle="->", color=Q[i], lw=0.8))
    x0, x1 = xh(xl) - 1.4, xh(xl) + 1.4
    if 0.0 <= x1: ax.axvspan(0, x1, color=CWALL, hatch="////", lw=0, alpha=0.6)
    else: ax.text(x1, xz(xl) + 1.55, f"wall {-xh(xl):.1f} m →", fontsize=6, color=CGRAY, ha="right", va="top")
    ax.set_aspect("equal"); ax.set_xlim(x0, x1); ax.set_ylim(xz(xl) - 0.6, xz(xl) + 1.6); ax.axis("off")
    if title: ax.set_title(title, fontsize=7, pad=2)

def fig2_hocbf_vs_barrier(out: Path):
    from authority_barriers.experiments.trial_figures import _e1_pair
    hs, ps = _e1_pair(); Lh, Lp = TrialLog.load(hs), TrialLog.load(ps)
    a1 = HB.LinearClassK(0.5)
    x0 = x0_of(Lh); tpsi, tau = HB.t_psi(x0.h(P), x0.v(P), a1), HB.tau(x0, P)
    def idx(L, t): return int(np.searchsorted(L.t_diag, t))
    def stop_idx(L):
        v = L.d("v"); s = np.where(v <= 0)[0]; return int(s[0]) if s.size else L.t_diag.size - 1
    files = []
    for L, key, name in ((Lh, "hocbf", "HOCBF filter"), (Lp, "barrier", "authority-barrier filter")):
        ks = [0, min(idx(L, tpsi), L.t_diag.size - 1), min(idx(L, tau), L.t_diag.size - 1), stop_idx(L)]
        for col, (k, lab) in enumerate(zip(ks, ("t = 0", "t = t_ψ", "t = τ", "stop / contact"))):
            fig, ax = one(1.75, 1.6, (0.02, 0.98, 0.02, 0.84)); sketch(ax, state_of(L, k), L.cmd[:, k].reshape(N, 3), P, title=f"{name}\n{lab} ({L.t_diag[k]:.2f} s)")
            files.append(save(fig, out, f"paper_fig2_sketch_{key}_{col}"))
    t_max = max(Lh.t_diag[stop_idx(Lh)], Lp.t_diag[stop_idx(Lp)]) + 0.3
    M = (0.08, 0.90, 0.30, 0.93)
    (f0, b0), (f1, b1), (f2, b2) = one(FULL, 1.35, M), one(FULL, 1.35, M), one(FULL, 1.35, M)
    for L, ls, name in ((Lh, "-", "HOCBF"), (Lp, "--", "barrier")):
        m = L.t_diag <= t_max; t = L.t_diag[m]
        minz = np.array([state_of(L, k).z(P).min() for k in range(m.sum())])
        b0.plot(t, minz, ls=ls, color=CINK); label_right(b0, t[-1], minz[-1], name, CINK)
        b2.plot(t, L.d("h")[m], ls=ls, color=CINK); label_right(b2, t[-1], L.d("h")[m][-1], name, CINK)
    mh = Lh.t_diag <= t_max; th = Lh.t_diag[mh]
    b1.plot(th, Lh.d("mu")[mh], color=CINK); b1.plot(th, Lh.d("psi1")[mh], ls=":", color=CGRAY)
    label_right(b1, th[-1], Lh.d("mu")[mh][-1], "μ (HOCBF)", CINK); label_right(b1, th[-1], Lh.d("psi1")[mh][-1], "ψ₁", CGRAY)
    inf = Lh.d("feasible")[mh] < 0.5
    b1.fill_between(th, -100, 100, where=inf, color=CRED, alpha=0.12, lw=0); b1.set_yscale("symlog", linthresh=1.0); b1.set_ylim(-70, 3); b1.axhline(0, color=CGRAY, lw=0.5)
    b0.axhline(0, color=CGRAY, lw=0.5); b0.text(t_max * 0.55, 0.0, "boundary of Σ_y", fontsize=6.5, color=CGRAY, va="bottom")
    for ax in (b0, b1, b2):
        for tv in (tpsi, tau): ax.axvline(tv, color=CGRAY, lw=0.5, ls="--")
        ax.set_xlim(0, t_max); ax.set_xlabel("t [s]")
        ax.text(tpsi, ax.get_ylim()[1], "t_ψ ", fontsize=7, color=CGRAY, va="top", ha="right"); ax.text(tau, ax.get_ylim()[1], " τ", fontsize=7, color=CGRAY, va="top")
    b0.set_ylabel("min$_i$ z_i"); b1.set_ylabel("μ, ψ₁"); b2.set_ylabel("h [m]"); b2.axhline(0, color=CGRAY, lw=0.5)
    files += [save(f0, out, "paper_fig2a_min_z"), save(f1, out, "paper_fig2b_margin"), save(f2, out, "paper_fig2c_distance")]
    group("fig2_hocbf_vs_barrier", files, f"From one initial state in 𝒟 ∩ X_RF (E1, gains (0.5, 10), paper's actuator model) the HOCBF filter keeps every cable ahead of the payload (min_i z_i < 0) but its feasibility margin μ (K_HO(x) is nonempty iff μ ≥ 0) turns negative at the second tick and stays negative (shaded), ψ₁ crosses zero at t_ψ = {tpsi:.3f} s < τ = {tau:.3f} s, the earliest time a cable can leave Σ_y, and the relaxed loop hits the wall, while the authority-barrier filter starts the swing at t = 0, brings the first cable behind the payload before τ, and keeps h > 0. Sketches: side views (payload ●, cables, thrust arrows scaled to f_max, wall hatched) at t = 0, t_ψ, τ and the stop/contact; traces (a) min_i z_i, (b) μ and ψ₁, (c) h: solid HOCBF, dashed authority-barrier filter.")

# ------------------------------------------------------------------ 3. storyboard
def fig3_storyboard(out: Path, render: bool = True):
    L = TrialLog.load(RESULTS / "core" / "e2_perfect_max_off0_layer1" / "e2_perfect_max_off0_layer1_75")
    t, h, D, v = L.t_diag, L.d("h"), L.d("D"), L.d("v")
    du = L.d("du_norm"); z = np.array([state_of(L, k).z(P) for k in range(t.size)])
    k_eng = int(np.argmax(du > 1e-6)); k_beh = int(np.argmax(z.max(axis=1) > 0.0)); k_all = int(np.argmax(z.min(axis=1) > 0.0))
    dec = -np.gradient(v, t); k_stop = int(np.argmax(v <= 0.05)) if np.any(v <= 0.05) else t.size - 1
    k_peak = int(np.argmax(dec[:k_stop + 1])); k_rel = min(k_stop + int(0.5 / P.dt_filter), t.size - 1)
    moments = [(k_eng, "filter engages"), (k_beh, "first cable behind"), (k_all, "all cables behind"), (k_peak, "peak braking"), (k_stop, "stop"), (k_rel, "release")]
    frames = [f"paper_fig3_frame{j}" for j in range(len(moments))]
    if render:
        from authority_barriers.simulator.viz import SceneRenderer, crop_to_content, stamp, stamp_size
        from PIL import Image
        R = SceneRenderer(P, width=700, height=800); path = L.taut[0:3]
        R.set_wall_grid(state_of(L, k_stop).x_L, half_w=1.0, half_h=2.5)      # a narrow slab: the wall seen edge-on from the side
        side = P.r_vec + 0.12 * np.array([0, 0, 1.0])                          # side view at the vehicles' height, wall on the right
        for j, (k, lab) in enumerate(moments):
            st = state_of(L, k); focus = R.show(st=st, u=L.cmd[:, k].reshape(N, 3), v_L=st.v_L, path=path[:, :k + 1])
            pts = np.vstack([focus["xL"][None], focus["quads"], focus["quads"] + L.cmd[:, k].reshape(N, 3) / P.f_max_arr[:, None]])
            R.frame(pts, direction=side, pad=1.25)                               # the camera fits payload, vehicles and thrust rods at every instant
            R.screenshot(out / f"{frames[j]}.png", settle=1.2 if j == 0 else 0.6)
        R.close()
        crop_to_content([out / f"{f}.png" for f in frames], margin=0.03)
        labels = [f"{lab}, {t[k]:.2f} s" for k, lab in moments]
        size = min(stamp_size(Image.open(out / f"{f}.png").width, lab) for f, lab in zip(frames, labels))      # one font size for the six frames
        for f, lab in zip(frames, labels): stamp(out / f"{f}.png", lab, size=size); print("wrote", f, flush=True)
    win = t <= t[k_rel] + 0.05
    M = (0.08, 0.92, 0.28, 0.94)
    (f0, c0), (f1, c1), (f2, c2) = one(FULL, 1.5, M), one(FULL, 1.5, M), one(FULL, 1.5, M)
    c0.plot(t[win], h[win], color=CD); c0.plot(t[win], D[win], color=CD, ls="--"); c0.fill_between(t[win], D[win], h[win], color=CH, alpha=0.18, lw=0)
    label_right(c0, t[win][-1], h[win][-1], "h", CD); label_right(c0, t[win][-1], D[win][-1], "D", CD, dy=-6); c0.text(t[k_eng], 0.5 * (h[k_eng] + D[k_eng]), " H", color=CH, fontsize=7, va="center")
    c0.set_ylabel("h, D [m]")
    for i in range(N):
        c1.plot(t[win], z[win, i], color=Q[i]); label_right(c1, t[win][-1], z[win, i][-1], f"z{i+1}", Q[i], dy=(i - 1.5) * 5)
        c2.plot(t[win], L.cable[i][win], color=Q[i]); label_right(c2, t[win][-1], L.cable[i][win][-1], f"T{i+1}", Q[i], dy=(i - 1.5) * 5)
    c1.axhspan(-P.z_bar, P.z_bar, color=CGRAY, alpha=0.10, lw=0); c1.axhline(0, color=CGRAY, lw=0.5); c1.set_ylabel("z_i")
    c2.axhline(P.T_min, color=CGRAY, lw=0.5, ls=":"); c2.axhline(P.T_bar[0], color=CGRAY, lw=0.5, ls=":"); c2.set_ylabel("T_i [N]")
    c2.text(0.3, P.T_bar[0], "T̄", fontsize=6.5, color=CGRAY, va="bottom"); c2.text(2.7, P.T_min, "T_min", fontsize=6.5, color=CGRAY, va="top")
    c3 = c2.twinx(); c3.plot(t[win], L.d("utilization")[win], color=CGRAY, lw=0.6); c3.set_ylim(0, 1.05); c3.set_ylabel("max ‖u_i‖/f_max", color=CGRAY, fontsize=7); c3.spines["top"].set_visible(False)
    for ax in (c0, c1, c2):
        for k, _ in moments: ax.axvline(t[k], color=CGRAY, lw=0.4, ls="--")
        ax.set_xlim(t[win][0], t[win][-1]); ax.set_xlabel("t [s]")
    files = frames + [save(f0, out, "paper_fig3a_distances"), save(f1, out, "paper_fig3b_swing"), save(f2, out, "paper_fig3c_tensions")]
    group("fig3_storyboard", files, "One braking maneuver of the authority-barrier filter under the adversarial nominal (trial e2_perfect_max_off0_layer1_75, H(x₀) = 0.77 m, v₀ = 3.4 m/s): the filter engages, swings the cables behind the payload (z_i crosses zero), brakes at the tension cap T̄ on the cables behind and the floor T_min on those ahead, stops the payload with H riding zero, and releases. Frames: side views of the simulated scene at the six instants (dashed lines in the plots); (a) h and D with the barrier H shaded between them, (b) the swing coordinates z_i with the box ±z̄, (c) the cable tensions with T_min and T̄ and the thrust utilization (thin, right axis).")

# ------------------------------------------------------------------ 4. viability sandwich with trajectories
def fig4_sandwich(out: Path):
    from authority_barriers.experiments.e3_collocation import configuration, team
    p3 = team() if callable(team) else load_set("A_N3")
    dn, dr = BarrierData.nominal(p3), BarrierData.relaxed(p3)
    vv = np.linspace(0.05, 4.0, 120); files = []
    for cfg, letter in zip("ABC", "abc"):
        fig, ax = one(3.0 if cfg == "A" else 2.08, 2.7)
        z, w = configuration(cfg, p3); zd = np.zeros(p3.N)
        Dv = np.array([SD.D_of(v, z, zd, dn) for v in vv]); Dr = np.array([SD.D_of(v, z, zd, dr) for v in vv])
        ax.fill_between(vv, Dr, Dv, color=CH, alpha=0.12, lw=0); ax.plot(vv, Dv, ls="--", color=CD, lw=0.9); ax.plot(vv, Dr, ls=":", color=CD, lw=0.9)
        ax.text(3.95, Dv[-1], "D ", fontsize=7, ha="right", va="bottom"); ax.text(3.95, Dr[-1], "D_rel ", fontsize=7, ha="right", va="bottom")
        for f in sorted(glob.glob(str(RESULTS / "core" / "e3_collocation_jobs" / f"{cfg}_v*_k41_s6_dt0.01-0.1_reg.json"))):
            j = json.load(open(f)); v0 = float(j["v"]); hc = float(j["h_c"]); hi = float(j.get("h_infeasible_max") or 0.0)
            ax.errorbar([v0], [hc], yerr=[[max(hc - hi, 0.0)], [0.0]], fmt="o", color=CINK, ms=3, elinewidth=0.6, capsize=1.5)
            x = np.array(j["best_trajectory"]["x"]); xL, vL = x[:, 0:3], x[:, 3:6]
            ax.plot(vL @ p3.n_vec, p3.d0 - xL @ p3.n_vec, color=CINK, lw=0.5, alpha=0.7)
        if cfg == "A":
            for stem, name in (("e2_perfect_max_off0_layer1/e2_perfect_max_off0_layer1_75", "parks"), ("e2_perfect_max_off0/e2_perfect_max_off0_72", "touches the wall")):
                L = TrialLog.load(RESULTS / "core" / stem); hL, vL, HL = L.d("h"), L.d("v"), L.d("H")
                m = (vL > 0.02) & (hL > 1e-3); pts = np.column_stack([vL[m], hL[m]]).reshape(-1, 1, 2); segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                lc = LineCollection(segs, cmap="viridis", norm=plt.Normalize(-0.2, 0.8), lw=1.2); lc.set_array(HL[m][:-1]); ax.add_collection(lc)
                ax.text(*((0.15, 0.17) if name == "parks" else (0.55, 0.03)), f"filter: {name}", fontsize=6.5, color=CINK, va="center")
            cb = fig.colorbar(lc, ax=ax, pad=0.03, fraction=0.07); cb.set_label("H along the filter's trajectories [m]", fontsize=7)
        ax.set_yscale("log"); ax.set_xlim(0, 4.2); ax.set_ylim(5e-3, 12); ax.set_xlabel("v [m/s]"); ax.set_ylabel("h [m]")
        ax.set_title({"A": "A: all cables toward the wall", "B": "B: mixed", "C": "C: all cables at z̄"}[cfg], fontsize=7.5)
        files.append(save(fig, out, f"paper_fig4{letter}_sandwich_{cfg}", tight=True))
    group("fig4_sandwich", files, "The viability kernel boundary lies in the band between h = D_rel(v) (dotted, outer bound) and h = D(v) (dashed, inner bound), drawn for the initial cable state of each configuration (N = 3, set A_N3): the smallest initial distance h_c from which direct collocation found a verified trajectory into X_RF (●, with the bisection interval as error bar) and the trajectories themselves (thin curves from (v₀, h_c) to v = 0) hug the outer bound when all cables are at z̄ (C) and leave a factor 2–4 in A. (a) also shows two closed-loop E2 trajectories of the sampled filter (N = 4) colored by H along the curve: one that parks with H riding zero and one that touched the wall.")

# ------------------------------------------------------------------ 5. sampled-data margin as a map + mechanism
def fig5_margin_map(out: Path):
    (fl, axl), (fr, axr) = one(COL, 2.7), one(COL, 2.7)
    pts = []
    for folder in ("e2_perfect_max_off0", "e2_perfect_max_off0_layer1"):
        d = json.load(open(RESULTS / "core" / folder / "results.json"))
        for r in d["per_trial"]:
            L = TrialLog.load(RESULTS / "core" / folder / r["label"]); contact = L.termination == "wall_contact" or r["min_h"] < 0
            pts.append((r["v0"], r["H0"], contact, float(L.d("v")[-1]) if contact else 0.0))
    pts = np.array(pts, dtype=float)
    safe, con = pts[pts[:, 2] < 0.5], pts[pts[:, 2] > 0.5]
    axl.scatter(safe[:, 0], safe[:, 1], s=9, facecolors="white", edgecolors=CINK, lw=0.5, zorder=3)
    axl.scatter(con[:, 0], con[:, 1], s=6 + 60 * con[:, 3], facecolors=CRED, edgecolors="none", alpha=0.75, zorder=4)
    edges = np.linspace(1.0, 4.0, 7); step_x, step_y = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = con[(con[:, 0] >= a) & (con[:, 0] < b)]; hm = sel[:, 1].max() if sel.size else 0.0
        step_x += [a, b]; step_y += [hm, hm]
    axl.plot(step_x, step_y, color=CINK, lw=1.0, drawstyle="steps-post"); axl.text(3.45, step_y[-1] + 0.02, "empirical margin", fontsize=6.5, color=CINK, va="bottom", ha="right")
    axl.set_xlabel("v₀ [m/s]"); axl.set_ylabel("H(x₀) [m]"); axl.set_xlim(0.9, 4.1); axl.set_ylim(0, 1.05)
    axl.set_title("● contact (size ∝ contact speed)   ○ safe", fontsize=6.5, loc="left")
    # per tick of the contact trials of variant A: loss of H vs largest commanded swing acceleration
    d = json.load(open(RESULTS / "core" / "e2_perfect_max_off0" / "results.json")); xs, ys, yp = [], [], []
    for r in d["per_trial"]:
        L = TrialLog.load(RESULTS / "core" / "e2_perfect_max_off0" / r["label"])
        if not (L.termination == "wall_contact" or r["min_h"] < 0): continue
        H = L.d("H"); t = L.t_diag; dt = float(np.median(np.diff(t)))
        for k in range(0, t.size - 1, 4):
            st = state_of(L, k); u = L.cmd[:, k].reshape(N, 3); rhs = rhs_from_thrust(st, u, P)
            zdd = rhs["qdd"] @ P.y_vec; res = SD.evaluate(st.v(P), st.z(P), st.zd(P), DATA)
            if not np.isfinite(res.D) or res.t_star.size == 0: continue
            an = float(rhs["a"] @ P.n_vec)                                  # approach acceleration v̇
            dD = max(res.grad_v[j] * an + res.grad_zeta[j] @ st.zd(P) + res.grad_omega[j] @ zdd for j in range(res.t_star.size))
            pred = dt * (st.v(P) + dD)                                        # loss of H = -(ḣ - Ḋ) Δt, ḣ = -v
            xs.append(np.abs(zdd).max()); ys.append(H[k] - H[k + 1]); yp.append(pred)
    xs, ys, yp = map(np.array, (xs, ys, yp))
    axr.axvspan(0, DATA.nu, color=CGRAY, alpha=0.12, lw=0); axr.text(DATA.nu * 0.5, 0.97, "|z̈| ≤ ν", fontsize=6.5, color=CGRAY, ha="center", va="top", transform=axr.get_xaxis_transform())
    axr.scatter(xs, ys, s=4, color=CINK, alpha=0.35, lw=0, rasterized=True); axr.scatter(xs, yp, s=4, marker="x", color=CH, alpha=0.5, lw=0.5, rasterized=True)
    axr.set_xscale("symlog", linthresh=1.0); axr.set_yscale("symlog", linthresh=1e-3)
    axr.set_xlabel("largest commanded max$_i$ |z̈_i| [s$^{-2}$]"); axr.set_ylabel("one-period loss of H [m]")
    axr.set_title("● measured   × first-order prediction (v + ∂D·rates) Δt", fontsize=6.5, loc="left")
    files = [save(fl, out, "paper_fig5a_margin_map", tight=True), save(fr, out, "paper_fig5b_loss_mechanism", tight=True)]
    group("fig5_margin_map", files, "The sampled-data margin is a state margin that grows with speed, and its mechanism is the swing acceleration the filter admits. (a) every E2 trial of the paper's actuator model (variants A and C, 200 states) in the (v₀, H(x₀)) plane, contacts filled with the marker size proportional to the contact speed, safe trials hollow; the step is the largest H(x₀) that touched the wall per speed bin. (b) for every fourth tick of the 59 contact trials of variant A, the one-period loss of H against the largest swing acceleration |z̈_i| commanded that tick (from the commands through the cable equation), with the first-order prediction from the analytic gradient of E (×); the maneuver's own swing accelerations lie in the shaded band |z̈| ≤ ν.")

# ------------------------------------------------------------------ 6. where the thrust goes
def fig6_thrust(out: Path):
    (fl, axl), (fr, axr) = one(COL, 3.1), one(COL, 3.1)
    fm, m, l = P.f_max_arr[0], P.m_arr[0], P.l_arr[0]
    th = np.linspace(0, np.pi / 2, 200); axl.plot(fm * np.cos(th), fm * np.sin(th), color=CINK, lw=1.0); axl.plot([0, fm], [0, 0], color=CINK, lw=0.5)
    s_hi = P.T_bar[0] + m * P.a_max + m * l * P.omega_bar ** 2; reserve = m * P.a_max + m * l * P.omega_bar ** 2
    axl.add_patch(plt.Rectangle((P.T_min, 0), s_hi - P.T_min, P.rho[0], fill=True, color=CH, alpha=0.12, lw=0))
    axl.add_patch(plt.Rectangle((P.T_min, 0), s_hi - P.T_min, P.rho[0], fill=False, color=CH, lw=0.9))
    need_perp = m * (l * DATA.nu + P.a_max * np.sin(P.theta_q)); axl.add_patch(plt.Rectangle((P.T_min, 0), s_hi - P.T_min, need_perp, fill=False, color=CH, lw=0.9, ls="--"))
    L = TrialLog.load(RESULTS / "core" / "e2_perfect_max_off0_layer1" / "e2_perfect_max_off0_layer1_75")
    ss, pp = [], []
    for k in range(0, L.t_diag.size, 3):
        st = state_of(L, k); u = L.cmd[:, k].reshape(N, 3)
        for i in range(N):
            s = float(u[i] @ st.q[i]); ss.append(s); pp.append(float(np.linalg.norm(u[i] - s * st.q[i])))
    axl.scatter(ss, pp, s=3, color=CINK, alpha=0.25, lw=0, rasterized=True)
    axl.annotate("", (P.T_min, 36.5), (P.T_min + reserve, 36.5), arrowprops=dict(arrowstyle="<->", color=CGRAY, lw=0.7)); axl.text(P.T_min + reserve / 2 + 2, 37.2, f"{reserve:.0f} N cancel the payload's\nspecific force", fontsize=6.3, ha="center", va="bottom", color=CGRAY)
    axl.annotate("", (P.T_min + reserve, 34.8), (s_hi, 34.8), arrowprops=dict(arrowstyle="<->", color=CH, lw=0.7)); axl.text(s_hi + 0.5, 34.8, f"T̄ = {P.T_bar[0]:.1f} N for braking", fontsize=6.3, va="center", color=CH)
    axl.text(fm * 0.93, fm * 0.44, "‖u_i‖ = f_max", fontsize=7, rotation=-64, ha="center", va="center")
    axl.text(P.T_min + 0.5, P.rho[0] - 0.5, f"ρ = {P.rho[0]:.0f} N (assumed reserve)", fontsize=6.3, va="top", color=CH)
    axl.text(s_hi + 0.5, need_perp, " needed by the\n maneuver (ν)", fontsize=6.3, va="center", color=CH)
    axl.set_xlim(0, fm * 1.02); axl.set_ylim(0, fm * 1.02); axl.set_aspect("equal"); axl.set_xlabel("parallel component s_i [N]"); axl.set_ylabel("perpendicular ‖u_i^⊥‖ [N]")
    d = json.load(open(RESULTS / "optional" / "e7_stress" / "results.json"))["variants"]
    fs = sorted(int(k.split("_")[1]) for k in d if k.startswith("fmax_"))
    inf = [np.mean([r["infeasible_ticks"] for r in d[f"fmax_{f}"]["per_trial"]]) for f in fs]; rate = [np.mean([r["contact"] for r in d[f"fmax_{f}"]["per_trial"]]) for f in fs]
    axr.bar(fs, inf, width=1.4, color=CGRAY, alpha=0.6, lw=0); axr.set_ylabel("infeasible ticks per trial (bars)"); axr.set_xlabel("f_max [N]")
    ax2 = axr.twinx(); ax2.plot(fs, rate, "o-", color=CRED, ms=3); ax2.set_ylim(0, 1.0); ax2.set_ylabel("wall-contact rate (line)", color=CRED); ax2.spines["top"].set_visible(False)
    axr.axvline(42, color=CINK, lw=0.7, ls="--"); axr.text(41.5, max(inf) * 0.95, "Assumption on f_max\nholds from here ", fontsize=6.5, va="top", ha="right")
    files = [save(fl, out, "paper_fig6a_thrust_budget", tight=True), save(fr, out, "paper_fig6b_fmax_sweep", tight=True)]
    group("fig6_thrust", files, f"Where the thrust goes. (a) the admissible thrust of one quadrotor in the (parallel, perpendicular) half-plane — the ball ‖u_i‖ ≤ f_max = {fm:.0f} N, the rectangle [T_min, T̄ + m a_max + m l ω̄²] × [0, ρ] the assumptions reserve (solid), the perpendicular thrust the braking maneuver actually needs (dashed), and the thrust vectors realized in one adversarial trial (dots): {reserve:.0f} of the {fm:.0f} N are reserved to cancel the payload's specific force and only T̄ = {P.T_bar[0]:.1f} N is left for braking. (b) the f_max sweep of E7 (10 states per value, adversarial nominal): infeasible ticks per trial (bars) and wall-contact rate (line) as f_max is reduced from 44 to 20 N; the assumption on f_max holds down to 42 N and the guarantee degrades gracefully below it (every contact is a sampled-data case).")

# ------------------------------------------------------------------ 7. D(nu)/h_c against nu
def fig7_nu(out: Path):
    from authority_barriers.experiments.e3_collocation import configuration
    p3 = load_set("A_N3"); z, w = configuration("A", p3); zd = np.zeros(p3.N)
    j = json.load(open(RESULTS / "core" / "e3_collocation_jobs" / "A_v3_k41_s6_dt0.01-0.1_reg.json")); hc, v0 = float(j["h_c"]), float(j["v"])
    nus = np.linspace(0.1, 2.5, 120)
    ratio = [SD.D_of(v0, z, zd, BarrierData(np.asarray(p3.T_bar, float), p3.T_min, nu, p3.z_bar, p3.m_L, 0.0, "nu")) / hc for nu in nus]
    nu_max = np.cos(p3.theta_q) ** 2 * p3.omega_bar ** 2 / (4.0 * (p3.z_bar + p3.w_bar))          # swing-rate consistency with nu_w = nu
    fig, ax = one(COL, 2.2)
    ax.axvspan(0, nu_max, color=CGRAY, alpha=0.12, lw=0); ax.text(nu_max * 0.55, 1.12, f"admissible\nν ≤ {nu_max:.2f} s⁻²", fontsize=6.5, va="bottom", ha="center", color=CGRAY)
    ax.plot(nus, ratio, color=CINK); ax.axhline(1.0, color=CGRAY, lw=0.5, ls=":"); ax.axvline(p3.nu, color=CH, lw=0.7, ls="--"); ax.text(p3.nu, max(ratio) * 0.95, " ν = 0.5 s⁻²", color=CH, fontsize=6.5)
    ax.set_xlabel("swing acceleration bound ν [s$^{-2}$]"); ax.set_ylabel("D(ν) / h_c"); ax.set_xlim(0, nus[-1])
    files = [save(fig, out, "paper_fig7_price_of_nu", tight=True)]
    group("fig7_price_of_nu", files, f"The looseness of the inner bound is the price of the swing-acceleration assumption: for configuration A at v₀ = {v0:.0f} m/s, the stopping distance D computed with a hypothetical swing acceleration bound ν, relative to the collocation's h_c = {hc:.2f} m; the values of ν admissible under the swing-rate consistency assumption (ν ≤ {nu_max:.2f} s⁻² with ω̄ = 1 s⁻¹) are shaded, and set A's ν = {p3.nu} s⁻² gives the factor {ratio[np.argmin(np.abs(nus - p3.nu))]:.1f}.")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(RESULTS / "core" / "figures" / "paper")); ap.add_argument("--no-render", action="store_true"); ap.add_argument("--only", default=None)
    args = ap.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    jobs = {"1": lambda: fig1_anatomy(out), "2": lambda: fig2_hocbf_vs_barrier(out), "3": lambda: fig3_storyboard(out, render=not args.no_render), "4": lambda: fig4_sandwich(out), "5": lambda: fig5_margin_map(out), "6": lambda: fig6_thrust(out), "7": lambda: fig7_nu(out)}
    for k, fn in jobs.items():
        if args.only and k not in args.only.split(","): continue
        plt.rcParams.update(LINES_TRIAL if k in "2567" else LINES)               # the same line widths whichever groups are built in one run
        try: fn()
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"figure {k} FAILED: {e!r}", flush=True)
    old = {}
    if (out / "captions.md").exists():                                          # keep the captions of the groups not rebuilt this run
        for line in (out / "captions.md").read_text().splitlines():
            if line.startswith("**fig"):
                k, _, rest = line[2:].partition("** (files: "); fl, _, c = rest.partition(") — "); old[k] = (fl.split(", "), c)
    old.update(CAPTIONS)
    (out / "captions.md").write_text("# Captions (claim first); one plot or frame per file, a figure is a group of files\n\n" + "\n\n".join(f"**{k}** (files: {', '.join(old[k][0])}) — {old[k][1]}" for k in sorted(old)) + "\n")
    print("done:", out)

if __name__ == "__main__":
    main()
