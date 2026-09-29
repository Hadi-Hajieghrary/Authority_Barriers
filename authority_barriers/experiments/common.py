"""Shared helpers for the experiment drivers: result folders, per-trial metrics, membership tests,
attribution replay (M3.4), summaries."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from authority_barriers.theory import hocbf as HB
from authority_barriers.theory import profile as PF
from authority_barriers.theory import stopping as SD
from authority_barriers.theory.authority import BarrierData
from authority_barriers.theory.params import Params
from authority_barriers.theory.state import State
from authority_barriers.theory.taut_model import integrate, unpack
from authority_barriers.simulator.filter_system import DIAG_FIELDS
from authority_barriers.simulator.recorder import TrialLog
from authority_barriers.simulator.harness import _worker

RESULTS = Path(os.environ.get("SIM_RESULTS_DIR", Path(__file__).resolve().parents[2] / "results")).resolve()   # overridable for smoke runs


def out_dir(track: str, name: str) -> Path:
    d = RESULTS / track / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def in_XRF(st: State, p: Params, data: BarrierData) -> tuple[bool, float]:
    """Membership in X_RF (eq. Xrf): H >= 0 and swing states in their admissible sets."""
    z, zd, w, wd = st.z(p), st.zd(p), st.w(p), st.wd(p)
    inV = bool(np.all(PF.in_V(z, zd, data.nu, data.zeta_bar)) and np.all(PF.in_V(w, wd, p.nu_w, p.w_bar)))
    H = st.h(p) - SD.D_of(st.v(p), z, zd, data)
    return bool(inV and H >= 0.0), float(H)


def swing_states_in_V(log: TrialLog, p: Params, data: BarrierData, tol: float = 1e-7) -> np.ndarray:
    """Per tick: are all swing states in V(nu, z_bar) x V(nu_w, w_bar)?"""
    ok = np.ones(log.t_diag.size, bool)
    for k in range(log.t_diag.size):
        st = unpack(log.taut[:, k], p.N)
        ok[k] = bool(np.all(PF.in_V(st.z(p), st.zd(p), data.nu, data.zeta_bar, tol)) and
                     np.all(PF.in_V(st.w(p), st.wd(p), p.nu_w, p.w_bar, tol)))
    return ok


def replay_on_taut_ode(log: TrialLog, p: Params, t_end: float | None = None):
    """Attribution (M3.4): apply the logged zero-order-hold thrust commands to the exact taut-cable
    model from the same x0 and return the replayed min h and the trajectory."""
    x0 = log.meta["x0"]
    st0 = State(np.array(x0["x_L"]), np.array(x0["v_L"]), np.array(x0["q"]), np.array(x0["qd"]))
    t_cmd, U = log.t_diag, log.cmd
    t_end = float(t_cmd[-1]) if t_end is None else t_end

    def policy(t, st):
        k = min(int(np.searchsorted(t_cmd, t, side="right") - 1), U.shape[1] - 1)
        return U[:, max(k, 0)].reshape(p.N, 3)

    tr = integrate(st0, policy, t_end, p, dt_out=p.dt_filter, rtol=1e-8, atol=1e-10, terminate_on_slack=True)
    h = np.array([tr.state_at(k).h(p) for k in range(tr.t.size)])
    return float(np.min(h)), tr


def classify_violation(log: TrialLog, p: Params, data: BarrierData, H_margin: float) -> str:
    """model gap / sampled-data / theory (candidate), per the M3.4 procedure."""
    H0 = float(log.d("H")[0])
    if H0 < H_margin:
        return "sampled-data (H(x0) below the one-tick margin)"
    if np.any(log.d("clamped") > 0):
        return "sampled-data (swing state left V between ticks)"
    stretch = log.cable[p.N:2 * p.N]
    if np.any(stretch <= 0) or np.nanmax(log.att_err) > p.d_bar_i:
        return "model gap (slack cable or attitude error beyond budget)"
    h_replay, _ = replay_on_taut_ode(log, p)
    if h_replay < 0:
        return "theory-class candidate (taut replay also violates): ESCALATE"
    return "model gap (taut replay stays safe)"


def summary_table(rows: list[dict], cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def save_json(path: Path, obj) -> None:
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return str(o)
    Path(path).write_text(json.dumps(obj, indent=1, default=default))


def sampled_deficit(t: np.ndarray, H: np.ndarray, kappa: float):
    """Per-tick deficit of H relative to the continuous-time barrier condition Hdot >= -kappa H:
    d_k = max(0, H_k (1 - kappa dt_k) - H_{k+1}); the accumulated deficit m(t_k) = sum_{j<k} d_j is the
    sampled-data loss of Rem. 19 measured on the log (model-free). Returns (d, m, max rate d_k/dt_k)."""
    t = np.asarray(t, float); H = np.asarray(H, float)
    if t.size < 2:
        return np.zeros(0), np.zeros(t.size), 0.0
    dt = np.diff(t)
    d = np.maximum(0.0, H[:-1] * (1.0 - kappa * dt) - H[1:])
    m = np.concatenate([[0.0], np.cumsum(d)])
    return d, m, float(np.max(d / dt))


def sampled_allowance(log: TrialLog, p: Params):
    """D-20 (revised): (accumulated deficit m(t), max deficit rate, within) with within = every
    contact depth <= m(t) at that time, i.e. the wall would not have been reached without the
    intra-tick losses that the continuous-time condition does not see."""
    t, H, h = log.t_diag, log.d("H"), log.d("h")
    d, m, rate = sampled_deficit(t, H, p.kappa_H)
    depth = np.maximum(0.0, -h)
    within = bool(np.all(depth <= m + 1e-9))
    return m, rate, within


def calibrate_margin(p: Params, filt_factory, nominal, n_states: int = 5, t_f: float = 3.0, seed: int = 99, v_range=(1.0, 4.0)):
    """delta = max deficit rate d_k / dt over sampled closed loops of the raw filter on the exact taut
    model (Rem. 19's sampled-data margin, measured); also returns the per-state rates and the
    accumulated losses."""
    from authority_barriers.theory.closed_loop import run_sampled
    from authority_barriers.theory.sampling import sample_XRF_layer
    data = BarrierData.nominal(p)
    rng = np.random.default_rng(seed)
    rates, losses = [], []
    filt = filt_factory()
    for st0 in sample_XRF_layer(rng, p, n_states, data, v_range=v_range):
        cl = run_sampled(st0, filt, nominal, p, t_f)
        _, m, rate = sampled_deficit(cl["t"], cl["H"], p.kappa_H)
        rates.append(rate); losses.append(float(m[-1]))
    return float(np.max(rates)), rates, losses


def _worker_indexed(args):
    k, job = args
    return k, _worker(job)


def run_many_cached(cfgs, x0s, out: Path, n_workers: int = 8):
    """`harness.run_many` with a per-trial cache: a trial whose log (out/<label>.npz + .json) exists with
    the same configuration and initial state is loaded instead of re-run; every new log is saved the
    moment it completes, so an interrupted campaign resumes where it stopped. Returns (logs, n_cached)."""
    from multiprocessing import get_context
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    logs, todo = [None] * len(cfgs), []
    for k, (cfg, x0) in enumerate(zip(cfgs, x0s)):
        stem = out / cfg.label
        want = json.loads(json.dumps(cfg.to_dict()))
        if Path(str(stem) + ".npz").exists() and Path(str(stem) + ".json").exists():
            try:
                L = TrialLog.load(stem)
                old = L.meta.get("config", {})
                have = {**want, **old}                                 # fields added later take their defaults ...
                same_cfg = all(have.get(k) == want[k] for k in want if k not in ("wall_cap",))
                if old.get("nominal_version", 0) < want.get("nominal_version", 0) and old.get("nominal", want.get("nominal")) in ("velocity", "corridor", "hover"):
                    same_cfg = False                                   # ... except logs of an older transport/hover nominal (C-15)
                if L.termination == "wall_cap":                        # a truncated trial is re-run if the cap was raised
                    same_cfg = same_cfg and (want.get("wall_cap") is not None and L.meta["wall_time"] >= want["wall_cap"])
                same_x0 = all(np.allclose(np.asarray(L.meta["x0"][key], float), getattr(x0, key), rtol=0.0, atol=1e-12)
                              for key in ("x_L", "v_L", "q", "qd"))
                if same_cfg and same_x0:
                    logs[k] = L
                    continue
                print(f"cache: {stem.name} differs in {'config' if not same_cfg else 'x0'}; re-running", flush=True)
            except Exception as exc:
                print(f"cache: could not load {stem}: {exc!r}; re-running", flush=True)
        todo.append(k)
    n_cached = len(cfgs) - len(todo)
    print(f"cache: {n_cached}/{len(cfgs)} trials loaded from {out.name}; running {len(todo)}", flush=True)
    jobs = [(k, (cfgs[k].to_dict(), {"x_L": x0s[k].x_L.tolist(), "v_L": x0s[k].v_L.tolist(), "q": x0s[k].q.tolist(),
                                     "qd": x0s[k].qd.tolist()}, cfgs[k].params)) for k in todo]
    if n_workers <= 1 or len(jobs) <= 1:
        for k, L in map(_worker_indexed, jobs):
            L.save(out / cfgs[k].label)
            logs[k] = L
    elif jobs:
        with get_context("spawn").Pool(min(n_workers, len(jobs))) as pool:
            for k, L in pool.imap_unordered(_worker_indexed, jobs):
                L.save(out / cfgs[k].label)
                logs[k] = L
    return logs, n_cached
