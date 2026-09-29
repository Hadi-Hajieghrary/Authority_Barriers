"""Trial-level figures, one plot per file: Meshcat renders of the simulation (payload, cables, quadrotors with
their attitudes, commanded thrust vectors, payload velocity, wall) at the key instants of a trial in a side view
and a rear three-quarter view (authority_barriers.simulator.viz), one time series per signal (distances and barrier,
speeds, accelerations, cable tensions, thrust magnitudes and tilts, swing states and rates, filter status,
solve time, attitude tracking error, tick-to-tick command change, altitude) and the swing phase plane
against the admissible set V. A direct-collocation trajectory (E3) is rendered the same way.

  python -m authority_barriers.experiments.trial_figures                       # the headline trials of the core results
  python -m authority_barriers.experiments.trial_figures --trial <stem>        # one trial log (path without .npz/.json)
  python -m authority_barriers.experiments.trial_figures --collocation <job json>
Output: <results>/core/figures/trials/<name>/ with an index.md of captions and a sidecar.json.
"""
from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from authority_barriers.theory import profile as PF
from authority_barriers.theory import stopping as SD
from authority_barriers.theory.authority import BarrierData
from authority_barriers.theory.params import E3, Params, load_set
from authority_barriers.theory.state import State
from authority_barriers.theory.taut_model import rhs_from_thrust, unpack
from authority_barriers.simulator.recorder import TrialLog
from authority_barriers.experiments.make_figures import C, COL_W, RES, constants, git_sha
from authority_barriers.viability.taut_system import thrust_from_input

QC = [C["blue"], C["orange"], C["aqua"], C["yellow"], C["gray"], "#8e44ad", "#c0392b", "#16a085"]   # one color per quadrotor


@dataclass
class Rec:
    """What the renderers need, independent of the source (Drake trial log or collocation trajectory)."""
    name: str
    p: Params
    t: np.ndarray                 # (M,)
    X: np.ndarray                 # (6 + 6N, M) packed taut states
    U: np.ndarray                 # (M, N, 3) commanded thrust vectors
    h: np.ndarray
    v: np.ndarray
    H: np.ndarray
    D: np.ndarray
    extras: dict = field(default_factory=dict)   # optional per-tick signals (same length M)
    meta: dict = field(default_factory=dict)
    source: str = "trial"

    def state(self, k: int) -> State:
        return unpack(self.X[:, k], self.p.N)


# ------------------------------------------------------------------ sources
def rec_from_log(stem: Path) -> Rec:
    L = TrialLog.load(stem)
    cfg = L.meta["config"]
    p = load_set(cfg["params"])
    walls = None
    if cfg.get("corridor_half_width") is not None:                     # corridor: distances and leans refer to the first wall; both slabs drawn
        from authority_barriers.simulator.harness import corridor_params
        walls = corridor_params(p, cfg["corridor_half_width"])
        p = walls[0]
    N = p.N
    U = L.cmd.T.reshape(-1, N, 3)
    ex = {"feasible": L.d("feasible"), "relaxed": L.d("relaxed"), "solve_time": L.d("solve_time"), "b_cmd": L.d("b"),
          "t_plant": L.t_plant, "x_plant": L.x_plant,
          "D_rel": L.d("D_rel"), "D_rob": L.d("D_rob"), "H_rob": L.d("H_rob"), "T_phys": L.cable[:N], "a_meas": L.cable[2 * N:2 * N + 3],
          "att_err": L.att_err if np.isfinite(L.att_err).any() and np.nanmax(L.att_err) > 0 else None, "walls": walls}
    h, v, H, D = L.d("h"), L.d("v"), L.d("H"), L.d("D")
    if walls is not None:                                                # corridor: the logged h, v refer to the first wall; use the critical wall per tick
        from authority_barriers.theory.authority import BarrierData
        datas = [BarrierData.nominal(pw) for pw in walls]
        hs, vs, Ds = [], [], []
        for k in range(L.t_diag.size):
            st = unpack(L.taut[:, k], N)
            hs.append([st.h(pw) for pw in walls]); vs.append([st.v(pw) for pw in walls])
            Ds.append([SD.D_of(st.v(pw), st.z(pw), st.zd(pw), dw) for pw, dw in zip(walls, datas)])
        hs, vs, Ds = np.array(hs), np.array(vs), np.array(Ds)
        crit = np.argmin(hs - Ds, axis=1)
        idx = np.arange(L.t_diag.size)
        h, v, D = hs[idx, crit], vs[idx, crit], Ds[idx, crit]; H = h - D
        ex["h_walls"], ex["critical_wall"] = hs.T, crit                  # time along the last axis, like the other extras
    return Rec(Path(stem).name, p, L.t_diag, L.taut, U, h, v, H, D, ex,
               {"config": cfg, "termination": L.termination, "x0": L.meta["x0"], "git_sha": L.meta.get("git_sha")}, "trial")


def rec_from_collocation(job: Path, which: str = "best") -> Rec:
    d = json.loads(Path(job).read_text())
    tr = d["best_trajectory"]
    p = load_set("A_N3") if d.get("params", {}).get("N", 3) == 3 else load_set("A")
    data = BarrierData.nominal(p)
    t = np.asarray(tr["times"], float); X = np.asarray(tr["x"], float).T; Uin = np.asarray(tr["u"], float)
    N = p.N
    U = np.array([thrust_from_input(unpack(X[:, k], N), p, Uin[k]) for k in range(t.size)])
    sts = [unpack(X[:, k], N) for k in range(t.size)]
    h = np.array([s.h(p) for s in sts]); v = np.array([s.v(p) for s in sts])
    D = np.array([SD.D_of(s.v(p), s.z(p), s.zd(p), data) for s in sts])
    Tm = np.array([rhs_from_thrust(s, U[k], p)["T"] for k, s in enumerate(sts)]).T
    name = f"collocation_{d['config']}_v{d['v']:g}"
    return Rec(name, p, t, X, U, h, v, h - D, D, {"T_model_given": Tm, "D_rel": np.full(t.size, d["D_rel"])},
               {"config": d["config"], "v": d["v"], "h0": d["h_c"], "D": d["D"], "D_rel": d["D_rel"], "knots": len(t)}, "collocation")


# ------------------------------------------------------------------ helpers
def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _line_plot(out: Path, name: str, series, ylabel: str, refs=(), xlabel: str = "t [s]", ylim=None, step=False):
    """One axes: series = [(label, x, y, style dict)], refs = [(y, label)] horizontal reference lines."""
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    for label, x, y, st in series:
        (ax.step if step else ax.plot)(x, y, label=label, **st)
    for y, label in refs:
        ax.axhline(y, color=C["gray"], lw=0.6, ls="--")
        ax.text(ax.get_xlim()[1] if len(series) == 0 else float(np.max(series[0][1])), y, f" {label}", fontsize=6, color=C["gray"], va="bottom", ha="right")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if len(series) > 1 or (series and series[0][0]):
        ax.legend(frameon=False, ncol=min(3, len(series)), fontsize=6, handlelength=1.4, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout()
    _save(fig, out, name)


def key_moments(rec: Rec):
    """(index, label) of the instants worth a snapshot, in time order, at least 0.15 s apart."""
    t, H, h, v, p = rec.t, rec.H, rec.h, rec.v, rec.p
    zmin = np.array([np.min(rec.state(k).z(p)) for k in range(t.size)])
    swing = np.array([np.max(np.abs(rec.state(k).zd(p))) for k in range(t.size)])
    stop = np.where(v <= 0.0)[0]
    k_stop = int(stop[0]) if stop.size and stop[0] > 0 else None
    k_end_man = k_stop if k_stop is not None else t.size - 1          # the maneuver ends when the payload stops
    cand = [(0, "start")]
    feas = rec.extras.get("feasible")
    if feas is not None and np.any(feas < 0.5):
        cand.append((int(np.argmax(feas < 0.5)), "first infeasible tick"))
    later = t > 0.1
    if later.any():
        kk = np.where(later)[0]
        cand.append((int(kk[np.argmax(swing[kk])]), "fastest swing"))
    past = np.where(zmin > 0.0)[0]
    if past.size and past[0] > 0:
        cand.append((int(past[0]), "all cables behind the payload"))
    if k_end_man > 2:
        dec = -np.gradient(v[:k_end_man + 1], t[:k_end_man + 1])
        cand.append((int(np.argmax(dec)), "peak braking"))
    cand.append((int(np.nanargmin(H)), "min H"))
    if k_stop is not None:
        cand.append((k_stop, "stop (v = 0)"))
    cand.append((int(np.argmin(h)), "closest approach (min h)"))
    cand.append((t.size - 1, "end"))
    cand.sort()
    priority = ("first infeasible tick", "stop (v = 0)", "end")
    out = []
    for k, lab in cand:
        if out and t[k] - t[out[-1][0]] < 0.15 and lab not in priority and out[-1][1] not in priority:
            continue
        if out and k == out[-1][0]:
            continue
        out.append((k, lab))
    return out


VIEWS = {"side": "medium side view (wall on the right)", "rear": "medium three-quarter view from behind and above",
         "close_side": "close-up from the side at the height of the vehicles", "close_front": "close-up from the wall side looking back at the team",
         "close_low": "close-up from below and beside, looking up at the rotors and cables", "close_top": "close-up from above: the formation",
         "arena_side": "arena: the whole approach and the wall from the side, with the payload's path", "arena_high": "arena: the whole approach from behind and high"}


def render_snapshots(rec: Rec, moments, out: Path, views=tuple(VIEWS)) -> list[tuple[str, str, dict]]:
    """Meshcat renders of the real scene at the key instants: the logged plant state for attitude-loop trials (real
    attitudes), the taut state with the attitudes implied by the commanded thrust otherwise (Sec. II model and the
    collocation trajectory). Returns (file name, caption, camera) per image; empty if the renderer is unavailable."""
    try:
        from authority_barriers.simulator.viz import SceneRenderer, stamp
    except Exception as exc:                                    # pydrake Meshcat / Playwright missing: no snapshots
        print(f"snapshots skipped: {exc!r}")
        return []
    cfg = rec.meta.get("config") if isinstance(rec.meta.get("config"), dict) else {}
    use_plant = cfg.get("actuator") == "attitude" and rec.extras.get("x_plant") is not None
    R = SceneRenderer(rec.p, walls=rec.extras.get("walls"))
    path_all = rec.X[0:3, :]
    stop = np.where(rec.v <= 0.0)[0]                          # the arena is the maneuver: up to one second after the payload stops
    k_win = int(min(rec.t.size - 1, np.searchsorted(rec.t, rec.t[stop[0]] + 1.0))) if stop.size and stop[0] > 0 else rec.t.size - 1
    R.set_arena(path_all[:, :k_win + 1])
    done = []
    try:
        for j, (k, label) in enumerate(moments):
            if use_plant:
                kp = int(np.clip(np.searchsorted(rec.extras["t_plant"], rec.t[k]), 0, rec.extras["t_plant"].size - 1))
                focus = R.show(x_plant=rec.extras["x_plant"][:, kp], u=rec.U[k], path=path_all[:, :k + 1])
            else:
                st = rec.state(k)
                focus = R.show(st=st, u=rec.U[k], v_L=st.v_L, path=path_all[:, :k + 1])
            xL = focus["xL"]
            info = f"{label}:  t = {rec.t[k]:.2f} s   h = {rec.h[k]:.3f} m   v = {rec.v[k]:.2f} m/s   H = {rec.H[k]:.2f} m   altitude {xL[2]:.1f} m"
            for view in views:
                cam = R.camera(view, focus)
                nm = f"snapshot_{j}_{view}_{label.split(' ')[0].replace('(', '')}"
                R.screenshot(out / f"{nm}.png", settle=1.2 if j == 0 else 0.6)
                stamp(out / f"{nm}.png", info)
                done.append((nm, f"{VIEWS.get(view, view)}; Meshcat render of the {'logged plant state' if use_plant else 'taut-cable state with thrust-aligned attitudes'} (quadrotors colored per vehicle, cables gray, colored lines: commanded thrust scaled to 1 m at f_max, black line: payload velocity, gray line: payload path, slab: the wall). {info}", cam))
    finally:
        R.close()
    return done


def render(rec: Rec, out: Path, title: str = "") -> list[str]:
    p, N, t = rec.p, rec.p.N, rec.t
    out.mkdir(parents=True, exist_ok=True)
    for old in list(out.glob("*.png")) + list(out.glob("*.pdf")):      # no stale files from an earlier rendering
        old.unlink()
    lines = [f"# {rec.name}", "", title, "", f"Constants: set {p.name} (N = {N}); commit {git_sha()}.", "", "## Snapshots (Meshcat renders of the simulation)", ""]
    moments = key_moments(rec)
    cams = []
    for nm, cap, cam in render_snapshots(rec, moments, out):
        lines.append(f"- `{nm}`: {cap}")
        cams.append({"file": nm, **cam})
    # ---- signals
    sts = [rec.state(k) for k in range(t.size)]
    z = np.array([s.z(p) for s in sts]).T; zd = np.array([s.zd(p) for s in sts]).T; w = np.array([s.w(p) for s in sts]).T
    om = np.sqrt(np.array([s.omega2() for s in sts]).T)
    vq = np.array([np.linalg.norm(s.v_L[None, :] + p.l_arr[:, None] * s.qd, axis=1) for s in sts]).T
    Tm = rec.extras.get("T_model_given")
    if Tm is None:
        Tm = np.array([rhs_from_thrust(s, rec.U[k], p)["T"] for k, s in enumerate(sts)]).T
    unorm = np.linalg.norm(rec.U, axis=2).T
    tilt = np.degrees(np.arccos(np.clip(rec.U[:, :, 2] / np.maximum(unorm.T, 1e-9), -1, 1))).T
    du = np.concatenate([[0.0], np.max(np.linalg.norm(np.diff(rec.U, axis=0), axis=2), axis=1)])
    alt = np.array([s.x_L[2] for s in sts])
    ex = rec.extras
    robust = isinstance(rec.meta.get("config"), dict) and bool(rec.meta["config"].get("robust"))
    qs = lambda i: {"color": QC[i % len(QC)], "lw": 1.0}
    # the maneuver window: up to one second after the payload first stops (the parked phase adds nothing but chatter)
    stop = np.where(rec.v <= 0.0)[0]
    t_win = float(min(t[-1], t[stop[0]] + 1.0)) if stop.size and stop[0] > 0 and not rec.extras.get("walls") else float(t[-1])   # corridor: whole trial
    m = t <= t_win + 1e-9
    W = lambda y: np.asarray(y)[..., m]
    tw = t[m]
    lines += ["", f"## Signals (one per file; maneuver window 0-{t_win:.2f} s, `_full` = whole trial)", ""]
    def distances(tt, sel, suffix):
        ser = [("h (distance to the wall)", tt, sel(rec.h), {"color": C["ink"]}), ("D (stopping distance)", tt, sel(rec.D), {"color": C["orange"], "ls": "--"})]
        if ex.get("D_rel") is not None and np.isfinite(ex["D_rel"]).any():
            ser.append(("D_rel (relaxed)", tt, sel(ex["D_rel"]), {"color": C["aqua"], "ls": ":"}))
        if robust and ex.get("D_rob") is not None:
            ser.append(("D_rob (robust)", tt, sel(ex["D_rob"]), {"color": C["yellow"], "ls": "-."}))
        _line_plot(out, "signal_distances" + suffix, ser, "[m]", refs=[(0.0, "wall")])
        ser = [("H = h - D", tt, sel(rec.H), {"color": C["blue"]})]
        if robust and ex.get("H_rob") is not None:
            ser.append(("H_rob = h - D_rob", tt, sel(ex["H_rob"]), {"color": C["yellow"], "ls": "-."}))
        _line_plot(out, "signal_barrier" + suffix, ser, "[m]", refs=[(0.0, "H = 0")])
    distances(tw, W, "")
    if t_win < t[-1] - 1e-9:
        distances(t, lambda y: np.asarray(y), "_full")
    lines.append("- `signal_distances`: distance to the wall h(t) with the stopping distances D and D_rel (and D_rob when the robust filter runs); the wall is h = 0.")
    lines.append("- `signal_barrier`: authority barrier H(t) (and H_rob for the robust filter); the filter keeps H >= 0 in continuous time, the sampled implementation may lose it between ticks.")
    t = tw; sts = [s_ for s_, keep in zip(sts, m) if keep]
    z, zd, w, om, vq, Tm, unorm, tilt, du, alt = (W(a) for a in (z, zd, w, om, vq, Tm, unorm, tilt, du, alt))
    ex = {k_: (W(v_) if isinstance(v_, np.ndarray) and k_ not in ("t_plant", "x_plant") else v_) for k_, v_ in ex.items()}
    vw = W(rec.v)
    _line_plot(out, "signal_speed_payload", [("v (toward the wall)", t, vw, {"color": C["ink"]}), ("|v_L|", t, np.array([np.linalg.norm(s.v_L) for s in sts]), {"color": C["gray"], "ls": ":"})], "[m/s]", refs=[(0.0, "")])
    lines.append("- `signal_speed_payload`: payload speed toward the wall v(t) and speed norm.")
    _line_plot(out, "signal_speed_quadrotors", [(f"quad {i + 1}", t, vq[i], qs(i)) for i in range(N)], "[m/s]"); lines.append("- `signal_speed_quadrotors`: quadrotor speed norms.")
    ser = []
    if ex.get("a_meas") is not None:
        ser.append(("measured braking acceleration y.a", t, -(ex["a_meas"].T @ p.n_vec), {"color": C["ink"]}))
    if ex.get("b_cmd") is not None:
        ser.append(("commanded b", t, ex["b_cmd"], {"color": C["orange"], "ls": "--"}))
    if not ser:
        acc = np.gradient(-vw, t); ser.append(("braking acceleration -dv/dt", t, acc, {"color": C["ink"]}))
    _line_plot(out, "signal_acceleration", ser, "[m/s$^2$]", refs=[(0.0, ""), (p.alpha_sat, "alpha_sat")]); lines.append("- `signal_acceleration`: braking acceleration of the payload along the wall normal (measured from the cable forces, and the value b commanded by the filter).")
    _line_plot(out, "signal_tensions_model", [(f"quad {i + 1}", t, Tm[i], qs(i)) for i in range(N)], "[N]", refs=[(p.T_min, "T_min"), (p.T_bar[0], "T_bar")])
    lines.append("- `signal_tensions_model`: cable tensions implied by the commanded thrust on the taut-cable model (the quantity the filter bounds), with the floor T_min and the cap T_bar of the braking maneuver.")
    if ex.get("T_phys") is not None:
        _line_plot(out, "signal_tensions_spring", [(f"quad {i + 1}", t, ex["T_phys"][i], qs(i)) for i in range(N)], "[N]", refs=[(p.T_min, "T_min"), (0.0, "slack")])
        lines.append("- `signal_tensions_spring`: tensions measured on the springs of the full-order plant (compliance transients after each command step; a cable is physically slack at zero).")
    _line_plot(out, "signal_thrust", [(f"quad {i + 1}", t, unorm[i], qs(i)) for i in range(N)], "[N]", refs=[(p.f_max[0], "f_max")]); lines.append("- `signal_thrust`: commanded thrust magnitudes |u_i| (throttle = |u_i| / f_max).")
    _line_plot(out, "signal_thrust_tilt", [(f"quad {i + 1}", t, tilt[i], qs(i)) for i in range(N)], "[deg]"); lines.append("- `signal_thrust_tilt`: tilt of the commanded thrust vector from the vertical.")
    _line_plot(out, "signal_command_change", [("max_i |u_i(k) - u_i(k-1)|", t, du, {"color": C["ink"]})], "[N]"); lines.append("- `signal_command_change`: tick-to-tick change of the commanded thrust (the jumps the inner loop must follow, C-10).")
    _line_plot(out, "signal_swing_z", [(f"quad {i + 1}", t, z[i], qs(i)) for i in range(N)], "z_i = q_i . y  [-]", refs=[(p.z_bar, "z_bar"), (-p.z_bar, "-z_bar"), (0.0, "")]); lines.append("- `signal_swing_z`: cable lean toward the wall z_i (negative = cable ahead of the payload, no braking authority).")
    _line_plot(out, "signal_swing_zd", [(f"quad {i + 1}", t, zd[i], qs(i)) for i in range(N)], "dz_i/dt  [1/s]", refs=[(0.0, "")]); lines.append("- `signal_swing_zd`: swing rates dz_i/dt.")
    _line_plot(out, "signal_swing_w", [(f"quad {i + 1}", t, w[i], qs(i)) for i in range(N)], "w_i = q_i . r  [-]", refs=[(p.w_bar, "w_bar"), (-p.w_bar, "-w_bar")]); lines.append("- `signal_swing_w`: lateral cable lean w_i.")
    _line_plot(out, "signal_swing_rate", [(f"quad {i + 1}", t, om[i], qs(i)) for i in range(N)], "|dq_i/dt|  [1/s]", refs=[(p.omega_bar, "omega_bar")]); lines.append("- `signal_swing_rate`: cable angular rates against the bound omega_bar of the operational set.")
    _line_plot(out, "signal_altitude", [("payload altitude", t, alt, {"color": C["ink"]})], "[m]"); lines.append("- `signal_altitude`: payload altitude (no floor is modeled; the payload may descend while braking).")
    if ex.get("feasible") is not None:
        _line_plot(out, "signal_filter_status", [("feasible", t, ex["feasible"], {"color": C["blue"]}), ("relaxed (slack on the barrier row)", t, ex["relaxed"], {"color": C["orange"], "ls": "--"})], "[0/1]", ylim=(-0.1, 1.3), step=True)
        lines.append("- `signal_filter_status`: feasibility of the filter's program at every tick and whether the relaxed program (D-15) was applied.")
        _line_plot(out, "signal_solve_time", [("Clarabel solve", t, 1e3 * ex["solve_time"], {"color": C["ink"]})], "[ms]", refs=[(5.0, "period")]); lines.append("- `signal_solve_time`: solve time of the second-order-cone program per tick against the 5 ms period.")
    if ex.get("att_err") is not None:
        _line_plot(out, "signal_attitude_error", [(f"quad {i + 1}", t, ex["att_err"][i], qs(i)) for i in range(N)], "|f R e3 - u|  [N]", refs=[(p.d_bar_i, "d_bar_i")]); lines.append("- `signal_attitude_error`: tracking error of the 1 kHz geometric attitude loop against the disturbance budget d_bar_i of Rem. 18.")
    # ---- swing phase plane
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    zmax = p.omega_bar * 1.05
    zz = np.linspace(-zmax, zmax, 400); lo, hi = PF.bounds_V(zz, p.nu, p.z_bar)
    ax.fill_betweenx(zz, lo, hi, color="#e8f1fb", lw=0, label="admissible set V(nu, z_bar)")
    ax.plot(lo, zz, color=C["gray"], lw=0.6); ax.plot(hi, zz, color=C["gray"], lw=0.6)
    for i in range(N):
        ax.plot(z[i], zd[i], color=QC[i % len(QC)], lw=0.9, label=f"quad {i + 1}")
        ax.plot(z[i][0], zd[i][0], marker="o", ms=3, color=QC[i % len(QC)]); ax.plot(z[i][-1], zd[i][-1], marker="s", ms=3, color=QC[i % len(QC)])
    ax.axvline(0.0, color=C["gray"], lw=0.5, ls=":")
    ax.set_xlabel("z_i = q_i . y  [-]"); ax.set_ylabel("dz_i/dt  [1/s]"); ax.legend(frameon=False, fontsize=6, ncol=2)
    fig.tight_layout(); _save(fig, out, "phase_swing")
    lines.append("- `phase_swing`: swing phase plane (z_i, dz_i/dt) of every cable (circle: start, square: end) inside the admissible set V of the braking maneuver.")
    (out / "index.md").write_text("\n".join(lines) + "\n")
    meta = {k_: v_ for k_, v_ in rec.meta.items()}
    side = {"caption": title, "moments": [{"index": int(k), "t": float(rec.t[k]), "label": lab} for k, lab in moments], "cameras": cams, "source": rec.source, "meta": meta, **constants(p)}
    (out / "sidecar.json").write_text(json.dumps(side, indent=1, default=str))
    return lines


# ------------------------------------------------------------------ headline trials of the core results
def _e1_pair():
    """An X_RF initial state of the (0.5, 10) pair that the HOCBF filter drives into the wall and the proposed
    filter keeps off it (perfect actuator)."""
    from authority_barriers.experiments.common import in_XRF
    p = load_set("A"); data = BarrierData.nominal(p)
    folder = RES / "core" / "e1_perfect"
    for f in sorted(glob.glob(str(folder / "e1p_k0p5_10_*.npz")), key=lambda s: int(s.split("_")[-1][:-4])):
        Lp = TrialLog.load(f[:-4]); x0 = Lp.meta["x0"]
        st0 = State(*(np.array(x0[k]) for k in ("x_L", "v_L", "q", "qd")))
        if in_XRF(st0, p, data)[0] and Lp.termination == "horizon" and Lp.d("h").min() >= 0:
            i = f.split("_")[-1][:-4]
            Lh = TrialLog.load(str(folder / f"e1_k0p5_10_{i}"))
            if Lh.termination == "wall_contact":
                return folder / f"e1_k0p5_10_{i}", folder / f"e1p_k0p5_10_{i}"
    return None, None


def headline_items():
    """(log stem, one-sentence description) of the headline trials of the campaign."""
    core = RES / "core"
    items = [
        (core / "e2_perfect_max_off0_layer1" / "e2_perfect_max_off0_layer1_75", "E2 variant C, paper's actuator model: the closest call among the trials that stay off the wall (H(x0) = 0.77 m, v0 = 3.4 m/s) — the full braking maneuver under the adversarial nominal (full thrust toward the wall): the cables swing from ahead of the payload to behind it, the filter rides H = 0 and parks the payload at h ~ D(v = 0)."),
        (core / "e2_perfect_max_off0" / "e2_perfect_max_off0_72", "E2 variant A, paper's actuator model: a wall contact by the sampled-data mechanism (largest accumulated deficit of the campaign, 1.0 m): a new maximum of the braking profile is born within one tick and H drops by ~1 m; the filter stays feasible."),
        (core / "e2_attitude_max_off0" / "e2_attitude_max_off0_69", "E2 variant A with the 1 kHz geometric attitude loop (Sec. VI setup): tracking-error peaks of 43 N at the filter's command jumps, the swing constraints and the feasibility of (17) are lost, wall contact (C-10)."),
        (core / "e5_scale1_layer1" / "e5_s1_28", "E5, robust filter under wind at d_bar, paper's actuator model, H_rob(x0) = 0.94 m: H_rob >= 0 is lost between ticks (sampled-data case) and the payload touches the wall."),
        (core / "e5_scale1_layer1" / "e5_s1_6", "E5, robust filter under wind at d_bar, paper's actuator model, H_rob(x0) = 0.94 m: stays off the wall."),
    ]
    hocbf, prop = _e1_pair()
    if hocbf is not None:
        items.insert(2, (hocbf, "E1, HOCBF filter with (k1, k2) = (0.5, 10) from an initial state that lies in X_RF: the program loses feasibility before tau while every cable still leans toward the wall, and the closed loop (relaxed when infeasible) hits the wall."))
        items.insert(3, (prop, "E1, the proposed filter from the same initial state: stays off the wall."))
    # optional track (P7-P8): the corridor trial with the closest approach and, from the Monte Carlo, the proposed filter's
    # closest call among the trials in which every hypothesis holds
    opt = RES / "optional"
    cj = opt / "e7_corridor" / "results.json"
    if cj.exists():
        per = json.loads(cj.read_text())["per_trial"]
        r = min(per, key=lambda r: min(r["min_hA"], r["min_hB"]))
        items.append((opt / "e7_corridor" / r["label"], f"E7 corridor (two walls at +-10 m), nominal transport along the corridor at {abs(r['v_along']):.1f} m/s with a lateral drift: the trial with the closest approach to a wall "
                      f"(min h = {min(r['min_hA'], r['min_hB']):.2f} m at wall {r['closest_wall']}); the nominal damps the drift within a metre and keeps the formation inside the operational set, so no row of the two-wall filter binds and the command equals the nominal (M8.2)."))
    mj = opt / "e7_montecarlo" / "results.json"
    if mj.exists():
        prop = json.loads(mj.read_text())["per_baseline"].get("proposed", [])
        ok = [r for r in prop if r["assumptions_hold"] and r["in_XRF"] and r["gust"] <= 1.0 and r["k_factor"] == 1.0 and r["mass_ok"] and not r["contact"]]
        if ok:
            r = min(ok, key=lambda r: r["min_h"])
            items.append((opt / "e7_montecarlo" / f"e7mc_proposed_{r['seed']}", f"E7 Monte Carlo, proposed filter, seed {r['seed']}: the closest call among the trials in which every hypothesis holds "
                          f"(f_max {r['f_max']:.1f} N, gust {r['gust']:.2f} d_bar, true m_L {r['m_L_true']:.2f} kg, swing-rate factor {r['qd_factor']:.2f}, H(x0) {r['H0']:.2f} m, v0 {r['v0']:.2f} m/s); min h = {r['min_h']:.3f} m."))
    return items


def headline():
    core = RES / "core"; items = headline_items()
    done = []
    for stem, title in items:
        if not Path(str(stem) + ".npz").exists():
            print(f"missing {stem}"); continue
        rec = rec_from_log(stem)
        out = RES / "core" / "figures" / "trials" / rec.name
        render(rec, out, title); done.append(rec.name); print(f"rendered {rec.name}")
    job = core / "e3_collocation_jobs" / "B_v4_k41_s6_dt0.01-0.1_reg.json"
    if job.exists():
        rec = rec_from_collocation(job)
        out = RES / "core" / "figures" / "trials" / rec.name
        render(rec, out, "E3 direct collocation, N = 3, configuration B (mixed cables) at v0 = 4 m/s: the trajectory found by the optimizer from the smallest feasible initial distance h_c = 0.56 D (41 knots), ending in X_RF; commanded thrust from the collocation inputs, tensions on the taut model.")
        done.append(rec.name); print(f"rendered {rec.name}")
    idx = ["# Trial figures (snapshots and per-signal records)", "", "One plot per file; each folder has an index.md with the captions and a sidecar.json with the constants.", ""]
    idx += [f"- [{n}]({n}/index.md)" for n in done]
    (RES / "core" / "figures" / "trials" / "index.md").write_text("\n".join(idx) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", default=None, help="trial log stem (path without .npz/.json)")
    ap.add_argument("--collocation", default=None, help="collocation job json")
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    if args.trial:
        rec = rec_from_log(Path(args.trial)); render(rec, RES / "core" / "figures" / "trials" / rec.name, args.title); print(f"rendered {rec.name}")
    elif args.collocation:
        rec = rec_from_collocation(Path(args.collocation)); render(rec, RES / "core" / "figures" / "trials" / rec.name, args.title); print(f"rendered {rec.name}")
    else:
        headline()


if __name__ == "__main__":
    main()
