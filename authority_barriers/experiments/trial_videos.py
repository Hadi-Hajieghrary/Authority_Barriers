"""Videos of the simulation, one per scenario: a recorded trial (or a direct-collocation trajectory) is replayed in slow
motion and shown in four synchronized panels.

  close-up   the team from beside the payload, turned slightly behind it and raised, in parallel projection, by a camera
             that moves with the payload. Vertical lines stay vertical in the picture and the wall normal lies along its
             horizontal axis, so the lean of every cable toward the wall or away from it is seen undistorted. Three thin
             lines from the payload mark the vertical and the two leans +-z_bar of the operational cone: a cable to the
             right of the vertical leans toward the wall and cannot brake, a cable to its left is behind the payload.
             The grid of 1 m behind the team and the panel of the wall show the motion of the payload.
  overview   the whole maneuver and the wall from the same direction with a fixed camera.
  plots      the distance h to the wall with the stopping distance D, and the leans z_i of the cables, with a cursor at
             the instant shown.
  readouts   t, h, v, D, H, the state of the filter, the number of cables behind the payload, the legend, and the cable
             directions seen from above: one dot per vehicle at (z_i, w_i), wall to the right, inside the box
             |z| <= z_bar, |w| <= w_bar of the operational set.

In the scene: quadrotors colored per vehicle with their commanded thrust as arrows of the same color, cables and payload
black, payload velocity as a black arrow, the distance h as a gray line to the wall, the stopping point x_L + D n of the
braking maneuver as a purple post (its gap to the wall is H), one dot per 0.1 s on the payload's path.

  python -m authority_barriers.experiments.trial_videos                      # every scenario
  python -m authority_barriers.experiments.trial_videos --only e1_hocbf,e7_corridor
  python -m authority_barriers.experiments.trial_videos --list
  python -m authority_barriers.experiments.trial_videos --gif                # <name>.gif from every video of the folder
Output: <results>/videos/<name>.mp4 (1920 x 1080, H.264, 30 frames per second), <name>.png (one frame), <name>.json (source,
window, cameras, constants, the numbers quoted in the description), README.md with one entry per video and, with --gif,
<name>.gif (960 x 540, 10 frames per second, 128 colors) for pages that show no video.
Requirements: the [viz] extra (Playwright with Chromium, Pillow) and ffmpeg.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from authority_barriers.theory import stopping as SD
from authority_barriers.theory.authority import BarrierData
from authority_barriers.theory.params import E3, Params
from authority_barriers.theory.taut_model import unpack
from authority_barriers.experiments.make_figures import C, RES, constants
from authority_barriers.experiments.sequence_figures import frame_indices
from authority_barriers.experiments.trial_figures import QC, Rec, _e1_pair, rec_from_collocation, rec_from_log

FPS = 30
FRAME = (1920, 1080)
BAR = 64                                  # height of the title bar
CLOSE, OVER = (1280, 720), (640, 360)     # the two rendered panels
PLOT_H, PLOT_Z = (640, 360), (640, 296)   # the two plots, below the overview
AZ, EL = 15.0, 18.0                       # direction of both cameras: beside the payload, turned behind it by AZ and raised by EL [deg]
CLOSE_WIDTH = 4.4                         # metres across the close-up
PAYLOAD_AT = (0.32, 0.30)                 # position of the payload in the close-up: fractions of the width (from the left) and height (from below)
FLOOR = 1.0                               # corridor: the floor grid lies this far below the lowest position of the payload [m]
MARGIN = 0.06                             # room kept between the team and the border of the close-up, as a fraction of its height
CRUMB = 0.1                               # one dot on the path per CRUMB seconds
SCALE = 0.75                              # drawn size of the quadrotors relative to the 0.3 m airframe
STYLE = {"cable_radius": 0.008, "thrust_length": 0.5, "thrust_radius": 0.012, "thrust_tip": 0.022, "velocity_scale": 0.25,
         "velocity_radius": 0.011, "h_radius": 0.005, "h_rgba": (0.72, 0.72, 0.70, 1.0), "D_radius": 0.022, "D_half_height": 0.35, "D_offset": -0.30, "arrow_heads": True}
INK, MUTED, RULE, PAPER = (20, 20, 20), (105, 105, 100), (200, 200, 196), (246, 246, 243)
GOOD, BAD, PURPLE = (20, 120, 70), (190, 40, 30), (122, 64, 191)


@dataclass
class Scenario:
    name: str
    source: Path                  # trial log (path without .npz/.json) or collocation job
    title: str
    setup: str                    # filter, actuation and team, under the title
    text: str                     # description for the README; {placeholders} are filled with numbers of the record
    kind: str = "trial"           # "trial" or "collocation"
    speed: float = 0.25           # seconds of the simulation per second of video
    tail: float = 0.8             # seconds of the simulation shown after the end of the maneuver
    az: float = AZ
    el: float = EL
    width: float = CLOSE_WIDTH
    extra: dict = field(default_factory=dict)


# ------------------------------------------------------------------ scenarios
def _share(n: int, k: int) -> str:
    """How many of the n recorded trials of a set end at the wall, as a clause."""
    return f"Of the {n} recorded trials, " + ("none ends" if k == 0 else "all end" if k == n else "1 ends" if k == 1 else f"{k} end") + " at the wall"


def _row(path: Path, *keys):
    d = json.loads(path.read_text())
    for k in keys:
        d = d[k]
    return d


def scenarios() -> list[Scenario]:
    """The scenarios, chosen from the recorded results by the rules stated with each of them."""
    core, opt = RES / "core", RES / "optional"
    perfect, S = "thrust vectors applied directly", []
    hocbf, prop = _e1_pair()                                     # the first initial state of E1 that lies in X_RF, HOCBF against the proposed filter
    if hocbf is not None:
        S.append(Scenario("e1_hocbf", hocbf, "HOCBF filter: every cable leans toward the wall", f"HOCBF on the wall distance, gains (0.5, 10) · {perfect} · N = 4",
                          "High-order control barrier function on the wall distance with the gains (k1, k2) = (0.5, 10). At the start every cable leans toward the wall, so no "
                          "tension can brake the payload. The program of the filter is infeasible from t = {t_inf_ms:.0f} ms on, and the payload reaches the wall at {v_end:.1f} m/s."))
        S.append(Scenario("e1_authority_barrier", prop, "Authority-barrier filter from the same initial state", f"authority-barrier filter · {perfect} · N = 4",
                          "The initial state of the video e1_hocbf with the authority-barrier filter: every cable swings behind the payload, the team brakes, and the payload "
                          "stops {min_h_window:.2f} m from the wall; in the {T_trial:.0f} s of the trial its closest approach is {min_h:.3f} m."))
    S.append(Scenario("e2_adversarial_command", core / "e2_perfect_max_off0_layer1" / "e2_perfect_max_off0_layer1_75", "Authority-barrier filter against full thrust toward the wall",
                      f"authority-barrier filter, adversarial nominal command · {perfect} · N = 4",
                      "The nominal command is full thrust toward the wall. From the margin H(x0) = {H0:.2f} m at {v0:.1f} m/s the filter swings the cables behind the payload and "
                      "brakes; the payload stops {min_h_window:.2f} m from the wall, and in the {T_trial:.0f} s of the trial its closest approach is {min_h:.4f} m. Among the trials of this "
                      "set that stay off the wall it is the one that comes closest."))
    S.append(Scenario("e2_sampled_contact", core / "e2_perfect_max_off0" / "e2_perfect_max_off0_72", "Sampled filter started at the boundary of the safe set",
                      f"authority-barrier filter at 200 Hz, adversarial nominal command · {perfect} · N = 4",
                      "Initial state at the boundary of the safe set, H(x0) = {H0:.2f} m. The barrier decreases between two samples of the 200 Hz filter, which the "
                      "continuous-time condition does not see, and the payload reaches the wall at {v_end:.2f} m/s; {feas}."))
    S.append(Scenario("e2_attitude_loop", core / "e2_attitude_max_off0" / "e2_attitude_max_off0_69", "Authority-barrier filter with the attitude loop",
                      "authority-barrier filter, adversarial nominal command · geometric attitude controller at 1 kHz · N = 4",
                      "A geometric attitude controller at 1 kHz tracks the commanded thrust vectors. Its tracking error reaches {att_err:.0f} N at the jumps of the command, "
                      "{feas}, and the payload reaches the wall at {v_end:.2f} m/s.", speed=0.2))
    job = core / "e3_collocation_jobs" / "A_v4_k41_s6_dt0.01-0.1_reg.json"
    S.append(Scenario("e3_collocation", job, "Optimized trajectory with three vehicles", "direct collocation, no filter · taut-cable model · N = 3",
                      "Direct collocation with three vehicles, every cable leaning toward the wall at {v0:.1f} m/s. The trajectory starts at h = {h0:.2f} m, {ratio:.2f} of the "
                      "stopping distance D = {D0:.2f} m of the swing-and-brake maneuver, and ends after {t_phase:.2f} s on the boundary of the safe set (H = {H_phase:.3f} m). From there the "
                      "video continues with the braking maneuver on the taut-cable model, which stops the payload at the wall, {min_h_mm:.1f} mm in front of it.", kind="collocation", tail=0.1))
    S.append(Scenario("e5_wind", core / "e5_scale1_layer1" / "e5_s1_6", "Robust filter under wind", f"authority-barrier filter with the disturbance budget, wind at the budget · {perfect} · N = 4",
                      "Wind bounded by the disturbance budget, barrier data reduced by that budget, adversarial nominal command. From {h0:.2f} m at {v0:.1f} m/s the payload "
                      "stops {min_h_window:.2f} m from the wall; in the {T_trial:.0f} s of the trial its closest approach is {min_h:.3f} m."))
    S.append(Scenario("e5_wind_contact", core / "e5_scale1_layer1" / "e5_s1_28", "Robust filter under wind: contact", f"authority-barrier filter with the disturbance budget, wind at the budget · {perfect} · N = 4",
                      "The setting of the video e5_wind from another initial state, {h0:.2f} m at {v0:.1f} m/s: the robust barrier is lost between two samples and the payload "
                      "reaches the wall at {v_end:.2f} m/s."))
    d6 = opt / "e6_distributed"
    if (d6 / "results.json").exists():                          # the trial in which the distributed filter reaches the wall first
        from authority_barriers.simulator.recorder import TrialLog
        ends = {k: TrialLog.load(d6 / f"e6d_distributed_{k}").t_diag[-1] for k in range(len(_row(d6 / "results.json", "per_trial"))) if (d6 / f"e6d_distributed_{k}.npz").exists()}
        k = min(ends, key=ends.get); n6 = sum(TrialLog.load(d6 / f"e6d_distributed_{j}").termination == "wall_contact" for j in ends)
        S.append(Scenario("e6_distributed", d6 / f"e6d_distributed_{k}", "Distributed filter with a delayed broadcast", f"distributed authority-barrier filter, adversarial nominal command · {perfect} · N = 4",
                          "Every vehicle solves its own program with the specific force of the payload broadcast one period late. The delayed value is not a commitment: "
                          "{feas} and the payload reaches the wall at {v_end:.2f} m/s. {share}; the video shows the one that reaches it first.",
                          extra={"share": _share(len(ends), int(n6))}))
    cj = opt / "e7_corridor" / "results.json"
    if cj.exists():                                             # the corridor trial with the closest approach to a wall
        r = min(_row(cj, "per_trial"), key=lambda r: min(r["min_hA"], r["min_hB"]))
        S.append(Scenario("e7_corridor", opt / "e7_corridor" / r["label"], "Transport through a corridor", f"two-wall authority-barrier filter, transport command · {perfect} · N = 4",
                          "Two walls 20 m apart. The team flies along the corridor at {v_along:.1f} m/s and drifts toward one wall at {v0:.1f} m/s; the nominal controller damps the "
                          "drift and the filter leaves its command unchanged. The closest approach in the {T_trial:.0f} s of the trial is {min_h:.2f} m. The camera looks along the corridor.", speed=0.5, tail=1.0, az=0.0, el=24.0, width=6.0,
                          extra={"v_along": abs(r["v_along"])}))
    mj = opt / "e7_montecarlo" / "results.json"
    if mj.exists():                                             # Monte Carlo: the closest call of the proposed filter among the trials in which every hypothesis holds
        rows = [r for r in _row(mj, "per_baseline", "proposed") if r["assumptions_hold"] and r["in_XRF"] and r["gust"] <= 1.0 and r["k_factor"] == 1.0 and r["mass_ok"] and not r["contact"]]
        if rows:
            r = min(rows, key=lambda r: r["min_h"]); ex = {"seed": r["seed"], "f_max": r["f_max"], "m_L": r["m_L_true"], "gust": r["gust"]}
            txt = "Monte Carlo trial {seed}: thrust limit {f_max:.1f} N, payload of {m_L:.2f} kg against 1 kg in the model, gusts at {gust:.2f} of the disturbance budget, transport toward the wall. "
            S.append(Scenario("e7_montecarlo", opt / "e7_montecarlo" / f"e7mc_proposed_{r['seed']}", "Transport toward the wall with modeling errors", f"authority-barrier filter, transport command · {perfect} · N = 4",
                              txt + "With the authority-barrier filter the payload stops {min_h_window:.2f} m from the wall; in the {T_trial:.0f} s of the trial its closest approach is {min_h:.3f} m.", extra=ex))
            S.append(Scenario("e7_unfiltered", opt / "e7_montecarlo" / f"e7mc_none_{r['seed']}", "The same transport without a filter", f"no filter, transport command · {perfect} · N = 4",
                              txt + "Without a filter the payload reaches the wall at {v_end:.2f} m/s.", extra=ex))
    sj = opt / "e7_stress" / "results.json"
    if sj.exists():
        V = _row(sj, "variants"); stress = opt / "e7_stress"; adv = f"authority-barrier filter, adversarial nominal command · {perfect} · N = 4"
        count = lambda key: {"share": _share(len(V[key]["per_trial"]), sum(bool(x["contact"]) for x in V[key]["per_trial"]))}
        r = max(V["fmax_20"]["per_trial"], key=lambda r: (r["contact"], r["infeasible_ticks"]))                     # the trial with a contact
        S.append(Scenario("e7_reduced_thrust", stress / r["label"], "Thrust limit reduced to 20 N", adv,
                          "With a thrust limit of 20 N in place of 44 N the thrust reserve that the barrier assumes does not exist: {feas} and the payload reaches the wall at "
                          "{v_end:.2f} m/s. {share}, which is the one shown.", extra=count("fmax_20")))
        r = V["mass_1.2_A"]["per_trial"][0]                                                                       # the first trial
        S.append(Scenario("e7_heavier_payload", stress / r["label"], "Payload 20 % heavier than the model", adv,
                          "The payload weighs 1.2 kg, the filter is designed for 1 kg. The payload reaches the wall at {v_end:.2f} m/s. {share}; the video shows the first of them.",
                          extra=count("mass_1.2_A")))
        r = max(V["soft"]["per_trial"], key=lambda r: r["slack_ticks"])                                          # the trial with the most samples with a slack cable
        S.append(Scenario("e7_soft_cables", stress / r["label"], "Cables ten times softer", adv,
                          "Cable stiffness divided by ten. A cable is slack at {slack} samples of the trial, which is outside the taut-cable model. The payload stops {min_h_window:.2f} m from the "
                          "wall; in the {T_trial:.0f} s of the trial its closest approach is {min_h:.3f} m. {share}; the video shows the one with the most samples with a slack cable.",
                          extra={"slack": r["slack_ticks"], **count("soft")}))
        r = max([x for x in V["omega"]["per_trial"] if x["contact"]] or V["omega"]["per_trial"], key=lambda r: r["t_end"])   # the contact that occurs last
        S.append(Scenario("e7_fast_swing", stress / r["label"], "Swing rates above the operational limit", adv,
                          "The cables start with swing rates of 1.5 times the limit of the operational set, where the stopping distance is not defined: {feas} and the payload "
                          "reaches the wall at {v_end:.2f} m/s. {share}; the video shows the one that reaches it last.",
                          extra=count("omega")))
    return [s for s in S if Path(str(s.source) + ("" if s.kind == "collocation" else ".npz")).exists()]


# ------------------------------------------------------------------ the record on the time axis of the video
def with_maneuver(rec: Rec, dt: float = 0.005, extra: float = 0.8) -> Rec:
    """The collocation trajectory, which ends in the safe set with the payload in motion, followed by the braking maneuver
    from its end state on the taut-cable model until the payload is at rest. The knots of the trajectory, 0.04 s apart,
    are resampled at `dt` by cubic interpolation of the positions with the velocities of the state."""
    from scipy.interpolate import CubicHermiteSpline
    from authority_barriers.theory import profile as PF
    from authority_barriers.theory.maneuver import plan
    from authority_barriers.theory.taut_model import integrate_capped
    p, N, X = rec.p, rec.p.N, rec.X
    data = BarrierData.nominal(p)
    ip, iv = np.r_[0:3, 6:6 + 3 * N], np.r_[3:6, 6 + 3 * N:6 + 6 * N]
    t1 = np.arange(rec.t[0], rec.t[-1] - 1e-9, dt)
    X1 = np.zeros((X.shape[0], t1.size))
    X1[ip] = CubicHermiteSpline(rec.t, X[ip].T, X[iv].T)(t1).T
    X1[iv] = np.array([np.interp(t1, rec.t, X[i]) for i in iv])
    U1 = np.array([[np.interp(t1, rec.t, rec.U[:, i, j]) for j in range(3)] for i in range(N)]).transpose(2, 0, 1)
    stT = rec.state(rec.t.size - 1)
    _, _, t2 = PF.switching(stT.z(p), stT.zd(p), data.nu, data.zeta_bar)
    horizon = float(np.max(t2)) + max(stT.v(p), 0.0) / data.alpha_sat + extra
    tr, timed_out, _ = integrate_capped(stT, lambda t, st: plan(st, p, data).u, horizon, p, 300.0, dt_out=dt, rtol=1e-9, atol=1e-11, terminate_on_slack=True)
    if timed_out or tr.terminated:
        raise RuntimeError("the braking maneuver from the end of the collocation trajectory could not be integrated")
    t = np.r_[t1, rec.t[-1] + tr.t]; X = np.hstack([X1, tr.X.T]); U = np.vstack([U1, tr.U])
    sts = [unpack(X[:, k], N) for k in range(t.size)]
    for s in sts:
        s.q /= np.linalg.norm(s.q, axis=1)[:, None]              # the views write the unit directions back into X
    h = np.array([s.h(p) for s in sts]); v = np.array([s.v(p) for s in sts])
    D = np.array([SD.D_of(s.v(p), s.z(p), s.zd(p), data) for s in sts])
    return Rec(rec.name, p, t, X, U, h, v, h - D, D, {"phase": float(rec.t[-1])}, {**rec.meta, "maneuver_horizon": horizon}, "collocation")


class Clip:
    """A record with its window, the wall that the cameras refer to, and the signals relative to that wall."""

    def __init__(self, sc: Scenario):
        self.sc = sc
        rec = self.rec = with_maneuver(rec_from_collocation(sc.source)) if sc.kind == "collocation" else rec_from_log(sc.source)
        cfg = rec.meta.get("config") if isinstance(rec.meta.get("config"), dict) else {}
        self.cfg = cfg
        k_end = frame_indices(rec, 4)[-1]
        self.k1 = k1 = int(min(rec.t.size - 1, np.searchsorted(rec.t, rec.t[k_end] + sc.tail)))
        if rec.meta.get("termination") == "wall_contact":        # a trial that ends at the wall is shown to its end
            self.k1 = k1 = rec.t.size - 1
        self.t = rec.t[:k1 + 1]
        self.walls = rec.extras.get("walls") or [rec.p]
        sts = [rec.state(k) for k in range(k1 + 1)]
        if len(self.walls) > 1:                                  # corridor: the wall of the closest approach
            j = int(np.argmin(rec.extras["h_walls"][:, :k1 + 1].min(axis=1))); pw = self.walls[j]; data = BarrierData.nominal(pw)
            self.h = np.array([s.h(pw) for s in sts]); self.v = np.array([s.v(pw) for s in sts])
            self.D = np.array([SD.D_of(s.v(pw), s.z(pw), s.zd(pw), data) for s in sts])
        else:
            pw = rec.p; self.h, self.v, self.D = rec.h[:k1 + 1], rec.v[:k1 + 1], rec.D[:k1 + 1]
        self.h_trial_min = float(min(s.h(pw) for s in (rec.state(k) for k in range(rec.t.size))))             # closest approach of the whole record
        self.pw = pw
        ov = cfg.get("param_overrides") or {}
        if ov:                                                   # the constants of the trial, for the scale of the thrust arrows
            from authority_barriers.simulator.harness import apply_overrides
            pw = apply_overrides(pw, SimpleNamespace(param_overrides=ov))
        self.p = pw
        self.H = self.h - self.D
        self.z = np.array([s.z(self.pw) for s in sts]).T                                     # (N, M)
        self.w = np.array([s.w(self.pw) for s in sts]).T
        feas = rec.extras.get("feasible")
        self.feasible = None if feas is None or cfg.get("filter") == "none" else np.asarray(feas[:k1 + 1]) > 0.5
        self.plant = cfg.get("actuator") == "attitude" and rec.extras.get("x_plant") is not None
        self.contact = rec.meta.get("termination") == "wall_contact" and k1 == rec.t.size - 1
        self.phase = rec.extras.get("phase")                     # collocation: the instant at which the braking maneuver takes over

    def facts(self) -> dict:
        """The numbers that the description quotes, measured on the window of the video."""
        rec, k1 = self.rec, self.k1
        f = {"h0": float(self.h[0]), "v0": float(self.v[0]), "H0": float(self.H[0]), "D0": float(self.D[0]), "min_h_window": float(max(self.h.min(), 0.0)), "min_h": float(max(self.h_trial_min, 0.0)),
             "min_h_mm": 1e3 * float(max(self.h_trial_min, 0.0)), "T_trial": float(rec.t[-1]), "t_end": float(self.t[-1]),
             "v_end": float(self.v[-1]), "contact": bool(self.contact), "n_ticks": int(self.t.size), "ratio": float(self.h[0] / self.D[0]) if self.D[0] > 0 else float("nan")}
        if self.phase is not None:
            f["t_phase"] = float(self.phase); f["H_phase"] = float(np.interp(self.phase, self.t, self.H))
        if self.feasible is not None:
            bad = np.where(~self.feasible)[0]
            f["n_inf"] = int(bad.size); f["t_inf"] = float(self.t[bad[0]]) if bad.size else float("nan")
            run = np.where(~self.feasible & np.r_[~self.feasible[1:], False])[0]              # infeasible at two consecutive samples
            if run.size: f["t_inf"] = float(self.t[run[0]])
            f["t_inf_ms"] = 1e3 * f["t_inf"]
            f["feas"] = ("the program of the filter is feasible at every sample" if bad.size == 0 else "the program of the filter is infeasible at every sample" if bad.size == self.t.size
                         else f"the program of the filter is infeasible at {bad.size} of {self.t.size} samples")
        if rec.extras.get("att_err") is not None:
            f["att_err"] = float(np.nanmax(rec.extras["att_err"][:, :k1 + 1]))
        f.update(self.sc.extra)
        return f

    def at(self, tau: float) -> SimpleNamespace:
        """State, command and signals at the time tau by interpolation between the samples of the record."""
        rec, t = self.rec, self.t; N = rec.p.N
        tau = float(np.clip(tau, t[0], t[-1]))
        k = int(np.clip(np.searchsorted(t, tau, side="right") - 1, 0, t.size - 2)); a = (tau - t[k]) / (t[k + 1] - t[k])
        mix = lambda y: (1 - a) * y[..., k] + a * y[..., k + 1]
        X = mix(rec.X).copy()
        st = unpack(X, N); st.q = st.q / np.linalg.norm(st.q, axis=1)[:, None]
        U = (1 - a) * rec.U[k] + a * rec.U[k + 1]
        U = U * np.minimum(1.0, self.p.f_max_arr / np.maximum(np.linalg.norm(U, axis=1), 1e-9))[:, None]      # the plant applies a command above the thrust limit at the limit
        f = SimpleNamespace(t=tau, st=st, U=U, h=float(mix(self.h)), v=float(mix(self.v)), D=float(mix(self.D)), H=float(mix(self.H)),
                            z=st.z(self.pw), w=st.w(self.pw), feasible=None if self.feasible is None else bool(self.feasible[k]), x_plant=None, k=k)
        if self.plant:
            tp, xp = rec.extras["t_plant"], rec.extras["x_plant"]
            j = int(np.clip(np.searchsorted(tp, tau, side="right") - 1, 0, tp.size - 2)); b = float(np.clip((tau - tp[j]) / (tp[j + 1] - tp[j]), 0.0, 1.0))
            f.x_plant = (1 - b) * xp[:, j] + b * xp[:, j + 1]
        return f


# ------------------------------------------------------------------ cameras and scenes
def axes(pw: Params, az: float, el: float):
    """Direction from the scene to the camera (beside the payload, turned behind it by az, raised by el) and the axes of
    the picture: to the right, which is toward the wall, and upward."""
    a, e = np.radians(az), np.radians(el); n, r = pw.n_vec, pw.r_vec
    d = np.cos(e) * (np.cos(a) * r - np.sin(a) * n) + np.sin(e) * E3
    right = np.cos(a) * n + np.sin(a) * r
    return d, right, np.cross(d, right)


class Stage:
    """A Meshcat scene with its camera in parallel projection: the close-up, whose camera moves with the payload, or the
    overview, whose camera is fixed on the whole maneuver."""

    def __init__(self, clip: Clip, size: tuple, follow: bool):
        from authority_barriers.simulator.viz import SceneRenderer
        self.clip, self.size, self.follow = clip, size, follow
        sc, pw, rec = clip.sc, clip.pw, clip.rec
        self.R = R = SceneRenderer(clip.p, width=size[0], height=size[1], walls=clip.walls, visual_scale=SCALE if follow else 1.5 * SCALE,
                                   style=STYLE if follow else {**STYLE, "cable_radius": 0.016, "thrust_radius": 0.024, "thrust_tip": 0.04, "velocity_radius": 0.028, "h_radius": 0.014, "D_radius": 0.05})
        self.d, self.right, self.up = axes(pw, sc.az, sc.el)
        P = rec.X[0:3, :clip.k1 + 1].T                           # the payload in the window
        lat, alt, n, r = P @ pw.r_vec, P[:, 2], pw.n_vec, pw.r_vec
        team = np.vstack([P + dz * E3 for dz in (-0.3, 1.0 + STYLE["thrust_length"])])
        if len(clip.walls) > 1:                                  # corridor: both walls along the flight, a floor grid below the team
            R.set_wall_panel((lat.min() - 8.0, lat.max() + 8.0), (alt.min() - 3.0, alt.max() + 3.0), thickness=0.15)
            across = sorted(float(n @ (w.d0 * w.n_vec)) for w in clip.walls)
            self.floor = float(alt.min()) - FLOOR
            R.set_floor((lat.min() - 8.0, lat.max() + 8.0), (across[0], across[1]), self.floor)
            pts = np.vstack([team] + [P + (w.d0 - P @ w.n_vec)[:, None] * w.n_vec for w in clip.walls])
        else:                                                    # a wall ahead: its panel from the payload's lateral position toward the camera, a grid behind the team
            R.set_wall_panel((lat[-1] - 0.1, lat[-1] + 14.0), (alt.min() - 3.0, alt.max() + 8.0))      # from the payload at the end of the window toward the camera
            R.set_backdrop(float(clip.h.max()) + 2.0, (alt.min() - 3.0, alt.max() + 3.5), float(lat.min()) - 1.2, wall=pw)
            beyond = np.clip(np.where(np.isfinite(clip.D), clip.D, 0.0) - clip.h, 0.0, 2.0)                # stopping points behind the wall, up to 2 m
            pts = np.vstack([team, P + (clip.h + 0.3 + beyond)[:, None] * n, P - 0.6 * n])
        if follow:                                               # the frame of the scenario, widened if the team of this record needs more room
            rel = []
            for k in range(0, clip.k1 + 1, 4):
                st = rec.state(k); Q = clip.p.l_arr[:, None] * st.q
                U = rec.U[k] * np.minimum(1.0, clip.p.f_max_arr / np.maximum(np.linalg.norm(rec.U[k], axis=1), 1e-9))[:, None]
                rel += [Q, Q + (STYLE["thrust_length"] + 6.0 * STYLE["thrust_tip"]) * U / clip.p.f_max_arr[:, None]]
            rel = np.vstack(rel); x, y = rel @ self.right, rel @ self.up; a = size[1] / size[0]
            left, right, down, up = max(-x.min(), 0.3), max(x.max(), 1.2), max(-y.min(), 0.45), max(y.max(), 0.5)
            self.width = W = max(sc.width, (left + right) / (1.0 - 2.0 * MARGIN * a), (down + up) / (a * (1.0 - 2.0 * MARGIN)))
            fx = float(np.clip(PAYLOAD_AT[0], left / W + MARGIN * a, 1.0 - right / W - MARGIN * a))
            fy = float(np.clip(PAYLOAD_AT[1], down / (a * W) + MARGIN, 1.0 - up / (a * W) - MARGIN))
            self.at = (fx, fy)
            self.offset = (0.5 - fx) * W * self.right + (0.5 - fy) * W * a * self.up
            self.scale = size[0] / W                             # pixels per metre
        else:                                                    # the smallest frame that holds the maneuver and the wall
            x, y = pts @ self.right, pts @ self.up
            self.center = 0.5 * (x.min() + x.max()) * self.right + 0.5 * (y.min() + y.max()) * self.up + (pts.mean(axis=0) @ self.d) * self.d
            self.width = 1.12 * max(np.ptp(x), np.ptp(y) * size[0] / size[1]) + 0.6
            self.scale = size[0] / self.width
        R.parallel(self.width)
        self.crumbs = 0
        self.target = None

    def pixel(self, P: np.ndarray) -> tuple:
        """Position of the point P in the picture (pixels from the upper left corner)."""
        rel = np.asarray(P, float) - self.target
        return self.size[0] / 2 + self.scale * float(rel @ self.right), self.size[1] / 2 - self.scale * float(rel @ self.up)

    def shot(self, f: SimpleNamespace, path: np.ndarray, crumbs: np.ndarray):
        R, clip = self.R, self.clip
        if f.x_plant is not None:
            focus = R.show(x_plant=f.x_plant, u=f.U, v_L=f.st.v_L, path=path, D=f.D, h=f.h)
        else:
            focus = R.show(st=f.st, u=f.U, v_L=f.st.v_L, path=path, D=f.D, h=f.h)
        self.focus = focus
        if self.follow:
            R.set_reference(focus["xL"], wall=clip.pw)
        if len(clip.walls) > 1:
            R.set_plumb(focus["xL"], self.floor)
        R.set_breadcrumbs(crumbs, radius=0.03 if self.follow else 0.07, start=self.crumbs); self.crumbs = len(crumbs)
        self.target = focus["xL"] + self.offset if self.follow else self.center
        R.meshcat.SetCameraPose(self.target + 40.0 * self.d, self.target)
        return R.capture()

    def margin(self, f: SimpleNamespace) -> float:
        """Smallest distance of the payload, the vehicles and the tips of the thrust arrows from the border of the picture,
        as a fraction of its height; negative if one of them lies outside."""
        p = self.clip.p; xL = self.focus["xL"]; Q = self.focus["quads"]
        pts = [xL] + list(Q) + [Q[i] + (STYLE["thrust_length"] + 6.0 * STYLE["thrust_tip"]) * f.U[i] / p.f_max_arr[i] for i in range(p.N)]
        m = min(min(x, self.size[0] - x, y, self.size[1] - y) for x, y in (self.pixel(P) for P in pts))
        return m / self.size[1]

    def close(self):
        self.R.close()


# ------------------------------------------------------------------ plots
class Plot:
    """A plot over the window, drawn once; every frame adds the cursor and the values at the instant shown."""

    def __init__(self, size: tuple, t: np.ndarray, series: list, ylabel: str, refs=(), shade=None, marks=(), top=None):
        """series: (label, values, color, line style); refs: (value, label) horizontal lines; shade: boolean mask of the
        samples to mark with a strip along the time axis (the program of the filter infeasible); marks: (time, text)
        vertical lines; top: largest value of the vertical axis."""
        fig = plt.figure(figsize=(size[0] / 100, size[1] / 100), dpi=100)
        ax = fig.add_axes([0.115, 50.0 / size[1], 0.865, 1.0 - 86.0 / size[1]])
        if shade is not None and np.any(shade):
            ax.fill_between(t, 0.0, 0.045, where=shade, step="post", transform=ax.get_xaxis_transform(), color="#c9302c", lw=0)
            ax.plot([], [], color="#c9302c", lw=5, label="program infeasible")
        for x, txt in marks:
            ax.axvline(x, color=C["gray"], lw=0.9, ls=":")
            ax.text(x, 0.97, " " + txt, fontsize=9, color=C["gray"], va="top", ha="left", transform=ax.get_xaxis_transform())
        for y, lab in refs:
            ax.axhline(y, color=C["gray"], lw=0.8, ls="--")
            if lab:
                ax.text(t[-1], y, lab + " ", fontsize=9, color=C["gray"], va="bottom", ha="right")
        for lab, y, col, ls in series:
            ax.plot(t, y, color=col, ls=ls, lw=1.8, label=lab)
        lo = min(min(np.nanmin(y) for _, y, _, _ in series), min([v for v, _ in refs] or [0.0])); hi = max(max(np.nanmax(y) for _, y, _, _ in series), max([v for v, _ in refs] or [0.0]))
        if top is not None:
            hi = min(hi, top)
        pad = 0.08 * max(hi - lo, 1e-3); ax.set_ylim(lo - 1.6 * pad, hi + 1.6 * pad); ax.set_xlim(t[0], t[-1])
        ax.set_xlabel("t [s]", fontsize=11, labelpad=1); ax.set_ylabel(ylabel, fontsize=11); ax.tick_params(labelsize=10)
        ax.legend(frameon=False, fontsize=9.5, ncol=6, loc="lower right", bbox_to_anchor=(1.0, 1.0), handlelength=1.6, columnspacing=1.0, borderaxespad=0.1)
        fig.canvas.draw()
        from PIL import Image
        self.base = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        (x0, y0), (x1, y1) = ax.transData.transform((t[0], lo - 1.6 * pad)), ax.transData.transform((t[-1], hi + 1.6 * pad))
        self.map = (t[0], t[-1], x0, x1, lo - 1.6 * pad, hi + 1.6 * pad, size[1] - y0, size[1] - y1)
        self.colors = [tuple(int(col.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)) for _, _, col, _ in series]
        plt.close(fig)

    def frame(self, tau: float, values):
        from PIL import ImageDraw
        t0, t1, x0, x1, v0, v1, y0, y1 = self.map
        im = self.base.copy(); d = ImageDraw.Draw(im)
        x = x0 + (tau - t0) / (t1 - t0) * (x1 - x0)
        d.line([(x, y1), (x, y0)], fill=(60, 60, 60), width=2)
        for v, col in zip(values, self.colors):
            if np.isfinite(v) and v0 <= v <= v1:
                y = y0 + (float(v) - v0) / (v1 - v0) * (y1 - y0)
                d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=col, outline=(255, 255, 255))
        return im


# ------------------------------------------------------------------ the frame
def fonts():
    from PIL import ImageFont
    f = lambda name, size: ImageFont.truetype(font_manager.findfont(name), size)
    return SimpleNamespace(title=f("DejaVu Sans:bold", 27), setup=f("DejaVu Sans", 17), num=f("DejaVu Sans Mono", 30), unit=f("DejaVu Sans", 19), text=f("DejaVu Sans", 19),
                           small=f("DejaVu Sans", 16), tiny=f("DejaVu Sans", 13), label=f("DejaVu Sans:bold", 18), banner=f("DejaVu Sans:bold", 26))


def legend(d, F, x: int, y: int, N: int):
    """The legend of the scene, in two columns."""
    rgb = lambda hexa: tuple(int(hexa.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    rows = [("quads", f"quadrotors 1 to {N}, commanded thrust"), ("cable", "cables and payload"), ("vel", "payload velocity (0.25 m per m/s)"),
            ("cone", "vertical and cone limits ±z̄"), ("h", "distance h to the wall"), ("D", "stopping point x_L + D n"), ("dots", "payload every 0.1 s"), ("grid", "grids of 1 m")]
    for j, (key, txt) in enumerate(rows):
        cx, cy = x + (j // 4) * 450, y + (j % 4) * 31
        if key == "quads":
            for i in range(N):
                d.rectangle([cx + 11 * i, cy + 4, cx + 11 * i + 8, cy + 18], fill=rgb(QC[i % len(QC)]))
        elif key == "cable":
            d.line([(cx, cy + 18), (cx + 30, cy + 4)], fill=INK, width=3); d.ellipse([cx + 32, cy + 5, cx + 44, cy + 17], fill=INK)
        elif key == "vel":
            d.line([(cx, cy + 11), (cx + 32, cy + 11)], fill=INK, width=4); d.polygon([(cx + 44, cy + 11), (cx + 30, cy + 4), (cx + 30, cy + 18)], fill=INK)
        elif key == "cone":
            d.line([(cx + 22, cy + 20), (cx + 22, cy + 2)], fill=(90, 90, 90), width=2); d.line([(cx + 22, cy + 20), (cx + 12, cy + 2)], fill=(160, 160, 155), width=2); d.line([(cx + 22, cy + 20), (cx + 32, cy + 2)], fill=(160, 160, 155), width=2)
        elif key == "h":
            d.line([(cx, cy + 11), (cx + 44, cy + 11)], fill=(115, 115, 115), width=2)
        elif key == "D":
            d.line([(cx + 22, cy + 1), (cx + 22, cy + 21)], fill=PURPLE, width=5)
        elif key == "dots":
            for i in range(4):
                d.ellipse([cx + 12 * i, cy + 8, cx + 12 * i + 6, cy + 14], fill=(77, 77, 77))
        else:
            for i in range(4):
                d.line([(cx + 14 * i, cy + 3), (cx + 14 * i, cy + 19)], fill=(168, 168, 163), width=1)
            d.line([(cx, cy + 6), (cx + 42, cy + 6)], fill=(168, 168, 163), width=1); d.line([(cx, cy + 16), (cx + 42, cy + 16)], fill=(168, 168, 163), width=1)
        d.text((cx + 56, cy), txt, fill=INK, font=F.small)


def formation(F, size: tuple, clip: Clip, f: SimpleNamespace, tail: float = 0.4):
    """The cable directions seen from above: one dot per vehicle at its leans (z_i, w_i), the wall to the right and the camera
    below as in the close-up, the box |z| <= z_bar, |w| <= w_bar of the operational set, and the line z = 0 that separates the
    cables that lean toward the wall from those behind the payload. A dot outside the picture is drawn at its border."""
    from PIL import Image, ImageDraw
    rgb = lambda hexa: tuple(int(hexa.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    im = Image.new("RGB", size, (255, 255, 255)); d = ImageDraw.Draw(im)
    x0, y0, x1, y1 = 0, 0, size[0], size[1]; zb, wb = clip.pw.z_bar, clip.pw.w_bar
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1) + 12
    s = min((0.5 * (x1 - x0) - 8) / (1.3 * zb), (0.5 * (y1 - y0) - 26) / (1.08 * wb))       # pixels per unit of lean
    at = lambda z, w: (float(np.clip(cx - s * z, 9, size[0] - 9)), float(np.clip(cy + s * w, 30, size[1] - 9)))
    d.text((x0, y0), "cable directions from above", fill=MUTED, font=F.small)
    d.rectangle([cx, cy - s * wb, cx + s * zb, cy + s * wb], fill=(251, 240, 238))           # leaning toward the wall: no braking
    d.rectangle([cx - s * zb, cy - s * wb, cx + s * zb, cy + s * wb], outline=(150, 150, 146), width=1)
    for yy in np.arange(cy - s * wb, cy + s * wb, 10):
        d.line([(cx, yy), (cx, min(yy + 5, cy + s * wb))], fill=(90, 90, 90), width=1)
    d.text((cx - s * zb + 6, cy - s * wb + 3), "behind", fill=MUTED, font=F.tiny); d.text((cx + s * zb - 6 - d.textlength("toward the wall", font=F.tiny), cy - s * wb + 3), "toward the wall", fill=MUTED, font=F.tiny)
    d.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=INK)
    k0 = max(0, f.k - int(round(tail / max(np.median(np.diff(clip.t)), 1e-6))))
    for i in range(clip.p.N):
        col = rgb(QC[i % len(QC)])
        pts = [at(z, w) for z, w in zip(clip.z[i, k0:f.k + 1], clip.w[i, k0:f.k + 1])] + [at(f.z[i], f.w[i])]
        if len(pts) > 1:
            d.line(pts, fill=tuple(int(0.5 * (c + 255)) for c in col), width=2)
    for i in range(clip.p.N):
        x, y = at(f.z[i], f.w[i])
        d.ellipse([x - 8, y - 8, x + 8, y + 8], fill=rgb(QC[i % len(QC)]), outline=(255, 255, 255))
    return im


def compose(clip: Clip, f: SimpleNamespace, close, over, plots, F, stage: Stage, banner: tuple | None = None):
    """The frame of the video from the two renders, the plots and the readouts."""
    from PIL import Image, ImageDraw
    sc, N = clip.sc, clip.p.N
    im = Image.new("RGB", FRAME, (255, 255, 255)); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, FRAME[0], BAR], fill=PAPER)
    d.text((24, 5), sc.title, fill=INK, font=F.title); d.text((24, 38), sc.setup, fill=MUTED, font=F.setup)
    note = f"slow motion, {sc.speed:g} × real time"
    d.text((FRAME[0] - 24 - d.textlength(note, font=F.text), 21), note, fill=MUTED, font=F.text)
    im.paste(close, (0, BAR)); im.paste(over, (CLOSE[0], BAR)); im.paste(plots[0].frame(f.t, [f.h, f.D]), (CLOSE[0], BAR + OVER[1])); im.paste(plots[1].frame(f.t, f.z), (CLOSE[0], BAR + CLOSE[1]))
    d.text((14, BAR + 10), "close-up: the camera moves with the payload", fill=MUTED, font=F.small)
    d.text((CLOSE[0] + 12, BAR + 8), "overview: fixed camera", fill=MUTED, font=F.small)
    x, _ = stage.pixel(f.st.x_L + f.h * clip.pw.n_vec)           # labels that follow the wall and the stopping point in the close-up
    if 10 < x + 12 < CLOSE[0] - 60:
        d.text((x + 12, BAR + 40), "wall", fill=MUTED, font=F.label)
    if np.isfinite(f.D) and f.D > 0:
        x, y = stage.pixel(f.st.x_L + f.D * clip.pw.n_vec + (STYLE["D_offset"] - 0.6 * STYLE["D_half_height"]) * E3)
        if 10 < x + 12 < CLOSE[0] - 160 and 30 < y < CLOSE[1] - 30:
            d.text((x + 12, BAR + y - 10), "stopping point", fill=PURPLE, font=F.label)
    for xy in ([(CLOSE[0], BAR), (CLOSE[0], FRAME[1])], [(0, BAR), (FRAME[0], BAR)], [(0, BAR + CLOSE[1]), (FRAME[0], BAR + CLOSE[1])], [(CLOSE[0], BAR + OVER[1]), (FRAME[0], BAR + OVER[1])]):
        d.line(xy, fill=RULE, width=1)
    x, y = 24, BAR + CLOSE[1] + 14                               # readouts
    for name, val, unit in (("t", f.t, "s"), ("h", f.h, "m"), ("v", f.v, "m/s"), ("D", f.D, "m"), ("H", f.H, "m")):
        d.text((x, y + 8), name + (" = h − D =" if name == "H" else " ="), fill=MUTED, font=F.unit); x += d.textlength(name + (" = h − D = " if name == "H" else " = "), font=F.unit)
        s = (f"{val:6.2f}" if name == "H" else f"{val:5.2f}") if np.isfinite(val) else "  –  "
        d.text((x, y), s, fill=INK, font=F.num); x += d.textlength(s, font=F.num) + 6
        d.text((x, y + 8), unit, fill=MUTED, font=F.unit); x += d.textlength(unit, font=F.unit) + 34
    y += 52; behind = int(np.sum(f.z > 0))
    if f.feasible is None:
        state, col = "no filter: " + (("optimized trajectory" if f.t <= clip.phase else "braking maneuver") if sc.kind == "collocation" else "nominal command"), MUTED
    elif clip.cfg.get("filter") == "none":
        state, col = "no filter: nominal command", MUTED
    else:
        state, col = ("program of the filter feasible", GOOD) if f.feasible else ("program of the filter infeasible", BAD)
    d.ellipse([24, y + 5, 40, y + 21], fill=col); d.text((50, y), state, fill=col, font=F.text)
    d.text((474, y), f"cables behind the payload (z > 0): {behind} of {N}", fill=GOOD if behind == N else INK, font=F.text)
    legend(d, F, 24, y + 44, N)
    im.paste(formation(F, (332, 232), clip, f), (940, BAR + CLOSE[1] + 58))
    if banner is not None:
        txt, col = banner; w = d.textlength(txt, font=F.banner)
        x0 = (CLOSE[0] - w) / 2; y0 = BAR + 44
        d.rectangle([x0 - 16, y0 - 8, x0 + w + 16, y0 + 40], fill=(255, 255, 255), outline=col, width=2); d.text((x0, y0), txt, fill=col, font=F.banner)
    return im


# ------------------------------------------------------------------ rendering
def banners(clip: Clip, facts: dict) -> tuple:
    """What the first and the last frame say."""
    z0 = clip.z[:, 0]; N = z0.size; k = int(np.sum(z0 < 0))
    first = "start: every cable leans toward the wall" if k == N else "start: every cable is behind the payload" if k == 0 else f"start: {k} of {N} cables lean toward the wall"
    if len(clip.walls) > 1:
        first = f"start: drift toward the wall on the right at {facts['v0']:.1f} m/s"
    if clip.contact:
        last = (f"the payload reaches the wall at {facts['v_end']:.2f} m/s", BAD)
    else:
        last = ("off the wall; closest approach of the trial: " + (f"{facts['min_h']:.2f} m" if facts["min_h"] >= 0.1 else f"{1e3 * facts['min_h']:.1f} mm" if facts["min_h"] < 0.01
                else f"{facts['min_h']:.3f} m"), GOOD)
    return (first, INK), last


def render(sc: Scenario, out: Path, fps: int = FPS, step: int = 1) -> dict:
    """One video; `step` > 1 renders every step-th frame only (a check of the pictures, not a video to keep)."""
    from PIL import Image
    out.mkdir(parents=True, exist_ok=True)
    clip = Clip(sc); facts = clip.facts(); F = fonts(); t = clip.t
    taus = np.arange(t[0], t[-1] + 1e-9, sc.speed / fps * step)
    if taus[-1] < t[-1] - 1e-6:
        taus = np.r_[taus, t[-1]]
    colors = QC[:clip.p.N]
    shade = None if clip.feasible is None else ~clip.feasible
    marks = [] if clip.phase is None else [(clip.phase, "braking maneuver")]
    D_fin = np.where(np.isfinite(clip.D), clip.D, np.nan)
    plots = (Plot(PLOT_H, t, [("h", clip.h, C["ink"], "-"), ("D", D_fin, C["orange"], "--")], "distance to the wall [m]", refs=[(0.0, "")], shade=shade, marks=marks,
                  top=1.6 * float(clip.h.max())),
             Plot(PLOT_Z, t, [(f"z{i + 1}", clip.z[i], colors[i], "-") for i in range(clip.p.N)], "cable lean z",
                  refs=[(clip.pw.z_bar, "z̄: behind the payload"), (-clip.pw.z_bar, "−z̄: toward the wall"), (0.0, "")], marks=marks))
    close, over = Stage(clip, CLOSE, follow=True), Stage(clip, OVER, follow=False)
    crumb_t = np.arange(t[0], t[-1] + 1e-9, CRUMB); crumb_x = np.array([clip.at(x).st.x_L for x in crumb_t])
    first, last = banners(clip, facts)
    video = out / f"{sc.name}.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{FRAME[0]}x{FRAME[1]}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium",
                           "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)], stdin=subprocess.PIPE)
    margins, n_written = [], 0
    k_poster = int(np.argmax(np.all(clip.z > 0, axis=0))) if np.any(np.all(clip.z > 0, axis=0)) else t.size // 2       # all cables behind the payload for the first time
    j_poster = int(np.argmin(np.abs(taus - t[k_poster])))
    try:
        for j, tau in enumerate(taus):
            f = clip.at(tau)
            path = clip.rec.X[0:3, :max(f.k + 1, 2)]
            crumbs = crumb_x[crumb_t <= tau + 1e-9]
            for _ in range(2 if j == 0 else 1):                  # the first picture is taken twice: the page has then loaded every object
                a, b = close.shot(f, path, crumbs), over.shot(f, path, crumbs)
            margins.append(close.margin(f))
            hold = j == 0 or j == len(taus) - 1
            im = compose(clip, f, a, b, plots, F, close, banner=(first if j == 0 else last) if hold else None)
            for _ in range(int((1.2 if j == 0 else 2.0) * fps / step) if hold else 1):
                ff.stdin.write(im.tobytes()); n_written += 1
            if j == j_poster:
                compose(clip, f, a, b, plots, F, close).save(out / f"{sc.name}.png")
    finally:
        close.close(); over.close()
        ff.stdin.close(); code = ff.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg ended with code {code} for {video}")
    if min(margins) < 0.0:
        raise RuntimeError(f"{sc.name}: the team leaves the close-up (margin {min(margins):.3f} of the height)")
    text = sc.text.format(**facts)
    side = {"video": video.name, "title": sc.title, "setup": sc.setup, "description": text, "source": str(Path(sc.source).relative_to(RES)) if Path(sc.source).is_relative_to(RES) else str(sc.source),
            "window_s": [float(t[0]), float(t[-1])], "speed": sc.speed, "frames_per_second": fps, "frames": n_written, "duration_s": n_written / fps, "size": list(FRAME),
            "cameras": {"projection": "parallel", "azimuth_deg": sc.az, "elevation_deg": sc.el, "close_up_width_m": float(close.width), "payload_at": list(close.at), "overview_width_m": float(over.width),
                        "reference_wall_normal": clip.pw.n_vec.tolist(), "smallest_margin_of_the_team_in_the_close_up": float(min(margins))},
            "facts": facts, **constants(clip.p)}
    (out / f"{sc.name}.json").write_text(json.dumps(side, indent=1, default=str))
    return side


def to_gif(video: Path, width: int = 960, fps: int = 10, colors: int = 128) -> Path:
    """The video as an animated GIF with one palette for the whole video; only the rectangle that changes is stored per frame."""
    gif = video.with_suffix(".gif")
    vf = (f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors={colors}:stats_mode=diff[p];"
          "[b][p]paletteuse=dither=none:diff_mode=rectangle")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vf", vf, str(gif)], check=True)
    return gif


def write_readme(out: Path):
    """README.md of the folder from the sidecars that it holds."""
    sides = [json.loads(f.read_text()) for f in sorted(out.glob("*.json"))]
    L = ["# Videos of the simulation", "",
         "One video per scenario, written by `python -m authority_barriers.experiments.trial_videos` from the trial logs and the collocation jobs of this results directory. "
         f"Each video replays the record in slow motion at {FPS} frames per second ({FRAME[0]} x {FRAME[1]}, H.264); the states between two samples of the record are interpolated.", "",
         "## How to read a video", "",
         "- **Close-up** (left): the team from beside the payload, turned slightly behind it and raised, in parallel projection; the camera moves with the payload. The wall is to the "
         "right. Vertical lines stay vertical and the wall normal lies along the horizontal axis of the picture, so the lean of a cable is seen undistorted. The three thin lines from "
         "the payload are the vertical and the leans ±z̄ of the operational cone: a cable to the right of the vertical leans toward the wall and cannot brake, a cable to its left is "
         "behind the payload and can. The grid of 1 m behind the team and the panel of the wall show the motion of the payload; the panel begins at the lateral position of the "
         "payload and extends toward the camera, so its edge in the picture is the point that the payload approaches. In the corridor the camera looks along the corridor, from "
         "behind the team and above, over a floor grid; the walls are seen edge-on.",
         "- **Overview** (upper right): the whole maneuver and the wall from the same direction, fixed camera; the quadrotors are drawn larger there.",
         "- **Plots** (right): distance h to the wall and stopping distance D; leans z_i of the cables (z_i > 0: behind the payload). The cursor is the instant shown; a red strip "
         "along the time axis marks the samples at which the program of the filter is infeasible.",
         "- **Readouts** (below): t, h, approach speed v, D, H = h − D, the state of the filter, the number of cables behind the payload.",
         "- **Cable directions from above** (lower middle): one dot per vehicle at its leans (z_i, w_i), with the wall to the right and the camera below; the box is the "
         "operational set |z| ≤ z̄, |w| ≤ w̄, the dashed line separates the cables that lean toward the wall from those behind the payload, the tails show the last 0.4 s.",
         "- **Scene**: quadrotors colored per vehicle, commanded thrust as arrows of the same color (0.5 m at the thrust limit), cables and payload black, payload velocity as a black "
         "arrow (0.25 m per m/s), the distance h as a gray line, the stopping point x_L + D n as a purple post, one dot per 0.1 s on the path of the payload. A command above the thrust "
         "limit is drawn at the limit, which is what the plant applies. The quadrotors are drawn at "
         f"{SCALE:g} of a 0.3 m airframe; the model treats them as points.", "",
         "## The videos", "", "| Video | Scenario | Record | Window [s] | Speed | Length [s] |", "|---|---|---|---|---|---|"]
    for s in sides:
        gif = Path(s["video"]).with_suffix(".gif").name
        also = f", [`{gif}`]({gif})" if (out / gif).exists() else ""
        L.append(f"| [`{s['video']}`]({s['video']}){also} | **{s['title']}.** {s['description']} | `{s['source']}` | {s['window_s'][0]:.2f} to {s['window_s'][1]:.2f} | {s['speed']:g} | {s['duration_s']:.1f} |")
    L += ["", "A video ends when the maneuver is over: at the wall, or shortly after the payload has stopped; the closest approach is stated for the whole trial. `<name>.gif` is the video at 960 x 540 and 10 frames per second, `<name>.png` is one frame of the video, `<name>.json` holds the source, the window, the cameras, the constants and the numbers quoted above.", ""]
    (out / "README.md").write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="names of the scenarios to render, separated by commas")
    ap.add_argument("--list", action="store_true", help="print the scenarios and stop")
    ap.add_argument("--out", default=None, help="output folder (default <results>/videos)")
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--step", type=int, default=1, help="render every step-th frame only (check of the pictures)")
    ap.add_argument("--readme", action="store_true", help="write README.md from the sidecars in the output folder and stop")
    ap.add_argument("--gif", action="store_true", help="convert the videos of the output folder to GIF, write README.md and stop")
    args = ap.parse_args()
    S = scenarios()
    if args.list:
        for s in S:
            print(f"{s.name:26s} {s.title}  [{Path(s.source).name}]")
        return
    if args.gif:
        out = Path(args.out) if args.out else RES / "videos"
        for v in sorted(out.glob("*.mp4")):
            g = to_gif(v); print(f"{g.name}: {g.stat().st_size / 1e6:.1f} MB", flush=True)
        write_readme(out); return
    if args.readme:
        write_readme(Path(args.out) if args.out else RES / "videos"); return
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed")
    if args.only:
        names = args.only.split(","); S = [s for s in S if s.name in names]
    out = Path(args.out) if args.out else RES / "videos"
    for s in S:
        side = render(s, out, fps=args.fps, step=args.step)
        print(f"{s.name}: {side['frames']} frames, {side['duration_s']:.1f} s, smallest margin of the team {side['cameras']['smallest_margin_of_the_team_in_the_close_up']:.2f}", flush=True)
    write_readme(out)


if __name__ == "__main__":
    main()
