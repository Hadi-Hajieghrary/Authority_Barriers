"""E8: the filter with altitude barriers on the exact taut-cable model (Thm. 12(i) in its own setting).

The theorem is stated for the taut-cable model with inputs that change continuously in time. The rows of this
experiment keep the model and vary the sampling period of the zero-order hold and the rows of the filter that
bound the swing accelerations (|zddot_i| <= 1.25 nu, |wddot_i| <= 1.25 nu_w):

  R1   period 1 ms, with the acceleration rows   (the setting of the theorem up to the hold)
  R2   period 5 ms, with the acceleration rows   (the period of the filter in the full-order simulation)
  R1n  period 1 ms, without them
  R2n  period 5 ms, without them

Without the acceleration rows the filter follows the nominal input with swing accelerations of about 20 1/s^2, forty
times nu; the stopping distance then changes within one period by more than its gradient predicts.

Initial states: the swing states, transverse states and approach speeds of E2 (sample_XRF_layer, seed 11), placed in
the boundary layer of the safe set of the filter, H(x0) in [0, 0.05 D(x0)], at an altitude of 15 m and at rest
vertically. Nominal input: the adversarial one of E2 (full thrust toward the wall, tilted 40 degrees upward).

Criterion for R1, fixed before the run: no wall contact, min H >= -2 mm, the altitude inside its band in every trial.
A contact in R1 would contradict the theorem; R2 measures what the hold of 5 ms costs on the model of the theorem.

Comparison (option --compare): one state of the safe set of the filter (v = 3 m/s, every cable at -0.8 z_bar, h = 1.02 D),
the adversarial nominal, period 1 ms; the HOCBF filter of Thm. 5 with the three gain pairs of E1 against the filter with the
altitude barriers. Four closed loops, about one minute.

  python -m authority_barriers.experiments.e8_exact_model [--n 100] [--t-final 8] [--workers 8] [--smoke]
  python -m authority_barriers.experiments.e8_exact_model --compare
Output: <results>/core/e8_exact_model/ with summary.md, results.json, one cached record per trial, and the figures
fig_e8_distances, fig_e8_altitude of one trial of R1; with --compare the records e8_compare_*.npz, compare.json and the
figure fig_e8_comparison.
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from authority_barriers.theory.authority import BarrierData
from authority_barriers.theory.params import load_set
from authority_barriers.theory.sampling import sample_XRF_layer
from authority_barriers.theory.state import State
from authority_barriers.experiments.common import out_dir

ROWS = {"R0": (5e-4, 1.25), "R1": (1e-3, 1.25), "R2": (5e-3, 1.25), "R1n": (1e-3, None), "R2n": (5e-3, None)}     # period [s], factor of the acceleration rows
SUBSTEPS = {}                 # states evaluated inside each hold per row (none: the minima between updates are measured by --refine)
SET = "A_alt"
TOL = {"integrator": (1e-9, 1e-11), "integrator_tight": (1e-12, 1e-14), "solver_tight": 1e-10}
REFINE = [("hold_1ms", 1e-3, "integrator", None), ("hold_0.5ms", 5e-4, "integrator", None), ("hold_0.25ms", 2.5e-4, "integrator", None),
          ("hold_1ms_integrator_tight", 1e-3, "integrator_tight", None), ("hold_1ms_solver_tight", 1e-3, "integrator", "solver_tight")]


def _trial(job):
    """One closed loop; returns the record as a dict of arrays (run in a worker process)."""
    row, k, dt, accel, t_final, x0, substeps, rtol, atol, solver_tol = (job + (0, *TOL["integrator"], None))[:10]
    from authority_barriers.simulator.nominal import AdversarialNominal
    from authority_barriers.theory.closed_loop import run_sampled
    from authority_barriers.theory.filters import ProposedFilter
    p = load_set(SET)
    st = State(*(np.asarray(x0[key], float) for key in ("x_L", "v_L", "q", "qd")))
    t0 = time.perf_counter()
    cl = run_sampled(st, ProposedFilter(p, altitude=True, dt=dt, accel_bound=accel, solver_tol=solver_tol), AdversarialNominal(p), p, t_final,
                     dt=dt, rtol=rtol, atol=atol, substeps=substeps)
    cl["wall_time"] = time.perf_counter() - t0
    return row, k, cl


def _x0(x: np.ndarray, N: int) -> dict:
    return {"x_L": x[0:3].tolist(), "v_L": x[3:6].tolist(), "q": x[6:6 + 3 * N].reshape(N, 3).tolist(), "qd": x[6 + 3 * N:6 + 6 * N].reshape(N, 3).tolist()}


def _between(cl: dict) -> dict:
    """Minima between updates (NaN when the record has none)."""
    return {key: (float(np.nanmin(cl[key])) if key in cl and np.isfinite(cl[key]).any() else float("nan"))
            for key in ("h_between", "H_between", "margin_z_between", "margin_w_between", "H_up_between", "H_down_between")} | \
        {key: (float(np.max(cl[key])) if key in cl else float("nan")) for key in ("T_viol", "f_viol")}


def refine(out: Path, p, t_final: float, workers: int):
    """The recorded contact trials under finer holds and tighter tolerances: hold 1, 0.5, 0.25 ms (integrator rtol 1e-9,
    atol 1e-11, Clarabel defaults), and at 1 ms the integrator tightened to 1e-12 / 1e-14 and the solver to 1e-10, each
    alone; ten states evaluated inside every hold. Writes refine.md, refine.json and fig_e8_refine."""
    trials = [(row, k) for row in ("R1", "R2") for k in range(len(list(out.glob(f"e8_{row}_*.npz"))))
              if (out / f"e8_{row}_{k}.npz").exists() and str(np.load(out / f"e8_{row}_{k}.npz", allow_pickle=False)["termination"]) == "wall_contact"]
    jobs, recs = [], {}
    for row, k in trials:
        x0 = _x0(np.load(out / f"e8_{row}_{k}.npz", allow_pickle=False)["x"][0], p.N)
        for name, dt, integ, solv in REFINE:
            f = out / f"e8_refine_{row}_{k}_{name}.npz"
            if f.exists():
                z = np.load(f, allow_pickle=False); recs[(row, k, name)] = {key: (str(z[key]) if key == "termination" else z[key]) for key in z.files}; continue
            jobs.append(((row, k, name), None, dt, ROWS["R1"][1], t_final, x0, 10, *TOL[integ], None if solv is None else TOL[solv]))
    print(f"refine: {len(trials)} contact trials, {len(recs)} records cached, running {len(jobs)}", flush=True)

    def keep(key, cl):
        np.savez_compressed(out / f"e8_refine_{key[0]}_{key[1]}_{key[2]}.npz", t_final=t_final, **{k2: np.asarray(v) for k2, v in cl.items()}); recs[key] = cl

    if workers <= 1 or len(jobs) <= 1:
        for j in jobs:
            key, _, cl = _trial(j); keep(key, cl)
    else:
        with get_context("spawn").Pool(min(workers, len(jobs))) as pool:
            for key, _, cl in pool.imap_unordered(_trial, jobs):
                keep(key, cl); print(f"  {key}: {cl['termination']} min h {np.min(cl['h']):.3e}", flush=True)
    table = {}
    for (row, k, name), cl in recs.items():
        m = metrics(cl, p) | _between(cl)
        table.setdefault(f"e8_{row}_{k}", {})[name] = m
    lines = [f"# E8 refinement of the recorded contact trials (set {p.name})", "",
             "Each contact trial of R1 and R2 rerun from the same state with the same nominal input: hold 1, 0.5 and 0.25 ms with the integrator at "
             "rtol 1e-9 / atol 1e-11 and Clarabel at its defaults; at 1 ms also the integrator at 1e-12 / 1e-14 and the solver at 1e-10, each alone. "
             "Minima between updates are taken over ten states inside every hold.", "",
             "| trial | run | termination | min h [mm] | min h between [mm] | min H [mm] | min H between [mm] | min swing margin z, w | min H_up, H_down between [m] | "
             "max tension / thrust violation [N] | infeasible |", "|" + "---|" * 11]
    for label, runs in table.items():
        for name, _, _, _ in REFINE:
            if name not in runs: continue
            m = runs[name]
            lines.append(f"| {label} | {name} | {m['termination']} | {1e3 * m['min_h']:.4f} | {1e3 * m['h_between']:.4f} | {1e3 * m['min_H']:.3f} | {1e3 * m['H_between']:.3f} | "
                         f"{m['margin_z_between']:.4f}, {m['margin_w_between']:.4f} | {m['H_up_between']:.3f}, {m['H_down_between']:.3f} | {m['T_viol']:.1e} / {m['f_viol']:.1e} | {m['infeasible']} |")
    (out / "refine.md").write_text("\n".join(lines) + "\n")
    (out / "refine.json").write_text(json.dumps({"runs": [list(r) for r in REFINE], "tolerances": TOL, "trials": table}, indent=1))
    # figure: smallest wall distance against the hold, one line per state
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator
    from authority_barriers.experiments.make_figures import C, COL_W, constants, save
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    holds = [2.5e-4, 5e-4, 1e-3]; names = {2.5e-4: "hold_0.25ms", 5e-4: "hold_0.5ms", 1e-3: "hold_1ms"}
    seen = set()
    for label, runs in table.items():
        state = label.split("_")[-1]
        if state in seen: continue                      # the same state recorded under two rows
        seen.add(state)
        hmin = [min(runs[names[h]]["min_h"], runs[names[h]]["h_between"]) if names[h] in runs else np.nan for h in holds]
        ax.plot([1e3 * h for h in holds], [1e3 * v for v in hmin], marker="o", ms=3, label=f"state {state}")
    ax.axhline(0.0, color=C["gray"], lw=0.6)
    ax.set_xscale("log"); ax.set_xlabel("hold [ms]"); ax.set_ylabel("min wall distance [mm]")
    ax.set_xticks([0.25, 0.5, 1.0]); ax.set_xticklabels(["0.25", "0.5", "1"]); ax.xaxis.set_minor_locator(NullLocator())
    ax.legend(frameon=False, ncol=2, fontsize=6); fig.tight_layout()
    save(fig, out / "figures", "fig_e8_refine", "E8 refinement: smallest wall distance (negative: penetration) of each recorded contact state against the hold of "
         "the filter, same states and nominal input, minima at the updates and between them. Expected if the theorem holds and the contacts are a "
         "sampled-data effect: the distance grows as the hold shrinks. Refuting: a penetration that does not shrink with the hold.",
         constants(p, {"runs": [list(r) for r in REFINE], "tolerances": TOL}))
    print("\n".join(lines))
    return table


def bench(out: Path, p, data, n: int = 200, seed: int = 11):
    """Single-process timing on n states of the E8 sample: the stopping distance with its gradients, the filter with the wall
    barrier alone (nominal data, no altitude rows), and the complete filter of E8 (altitude and swing rows, period 1 ms);
    median, 95th percentile and maximum in ms. Writes bench.json."""
    import platform
    from authority_barriers.simulator.nominal import AdversarialNominal
    from authority_barriers.theory import stopping as SD
    from authority_barriers.theory.filters import ProposedFilter
    rng = np.random.default_rng(seed)
    x0s = sample_XRF_layer(rng, p, n, data, v_range=(1.0, 4.0))
    nom = AdversarialNominal(p)
    filters = {"wall_only": ProposedFilter(p, altitude=False, dt=1e-3), "complete": ProposedFilter(p, altitude=True, dt=1e-3, accel_bound=ROWS["R1"][1])}
    times = {"D_with_gradients": [], "wall_only": [], "complete": []}
    for st in x0s:
        t0 = time.perf_counter(); SD.evaluate(st.v(p), st.z(p), st.zd(p), data); times["D_with_gradients"].append(time.perf_counter() - t0)
        for name, F in filters.items():
            r = F.solve(st, nom(0.0, st)); times[name].append(r.solve_time)
    stats = {name: {"median_ms": 1e3 * float(np.median(t)), "p95_ms": 1e3 * float(np.percentile(t, 95)), "max_ms": 1e3 * float(np.max(t)), "n": len(t)} for name, t in times.items()}
    cpu = platform.processor()
    try:
        cpu = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
    except (OSError, IndexError):
        pass
    env = {"cpu": cpu, "python": platform.python_version(),
           "drake": __import__("importlib.metadata").metadata.version("drake"), "solver": "Clarabel (bundled with Drake), default tolerances", "workers": 1}
    (out / "bench.json").write_text(json.dumps({"set": p.name, "states": n, "seed": seed, "period": 1e-3, "timing_ms": stats, "environment": env}, indent=1))
    for name, s in stats.items():
        print(f"{name:18s} median {s['median_ms']:.2f} ms  p95 {s['p95_ms']:.2f} ms  max {s['max_ms']:.2f} ms  (n = {s['n']})")
    print(env)
    return stats


def metrics(cl: dict, p) -> dict:
    alt = cl["x"][:, 2]
    return {"termination": str(cl["termination"]), "contact": str(cl["termination"]) == "wall_contact", "slack": str(cl["termination"]) == "slack",
            "t_end": float(cl["t"][-1]), "H0": float(cl["H"][0]), "v0": float(cl["v"][0]), "min_h": float(np.min(cl["h"])), "min_H": float(np.nanmin(cl["H"])),
            "min_H_up": float(np.nanmin(cl["H_up"])), "min_H_down": float(np.nanmin(cl["H_down"])), "climb": float(np.max(alt) - alt[0]),
            "drop": float(alt[0] - np.min(alt)), "in_band": bool(np.min(alt) >= p.alt_min and np.max(alt) <= p.alt_max),
            "infeasible": int(np.sum(~cl["feasible"].astype(bool))), "samples": int(cl["t"].size),
            "solve_ms_median": float(1e3 * np.nanmedian(cl["solve_time"])), "wall_time": float(cl["wall_time"])}


def figures(out: Path, cl: dict, p, label: str):
    """Distances and altitude of one trial, one plot per file."""
    import matplotlib.pyplot as plt
    from authority_barriers.experiments.make_figures import C, COL_W, constants, save
    t = cl["t"]
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    ax.plot(t, cl["h"], color=C["ink"], label="h"); ax.plot(t, cl["D"], color=C["orange"], ls="--", label="D")
    ax.axhline(0.0, color=C["gray"], lw=0.6); ax.set_xlabel("t [s]"); ax.set_ylabel("[m]"); ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    side = constants(p, {"trial": label, "sampling_period": float(t[1] - t[0]), "T_h": p.T_h, "nu_dec": p.nu_dec})
    save(fig, out, "fig_e8_distances", f"E8, trial {label} on the exact taut-cable model: distance h to the wall and stopping distance D of the maneuver "
         "with the hover floor. Expected if Thm. 12(i) holds: h >= D. Refuting: h < D by more than the effect of the hold, or a contact.", side)
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    ax.plot(t, cl["x"][:, 2], color=C["ink"], label="altitude")
    ax.plot(t, p.alt_max - cl["H_up"], color=C["orange"], ls="--", label="altitude + bound on the climb")
    ax.axhline(p.alt_max, color=C["gray"], lw=0.6, ls="--"); ax.set_xlabel("t [s]"); ax.set_ylabel("[m]"); ax.legend(frameon=False, ncol=1)
    fig.tight_layout()
    save(fig, out, "fig_e8_altitude", f"E8, trial {label}: altitude of the payload and the altitude it can reach under the maneuver (altitude plus the bound "
         f"on the climb), against the upper edge {p.alt_max:g} m of the band. Expected: the dashed curve stays below the edge. Refuting: the altitude leaves the band.", side)


GAINS = ((0.5, 10.0), (1.0, 20.0), (2.0, 40.0))     # gain pairs (k1, k2) of E1 [1/s]


def compare_state(p, data):
    """The state of the comparison: v = 3 m/s, every cable at -0.8 z_bar, transverse leans spread, h = 1.02 D."""
    from authority_barriers.theory import stopping as SD
    z = -0.8 * p.z_bar * np.ones(p.N); w = np.linspace(-0.6, 0.6, p.N) * p.w_bar
    D0 = SD.D_of(3.0, z, np.zeros(p.N), data)
    return State.from_swing(p, 1.02 * D0, 3.0, z, w, np.zeros(p.N), np.zeros(p.N))


def compare(out: Path, p, data, t_final: float = 3.0, dt: float = 1e-3):
    """HOCBF filter (three gain pairs) and the filter with the altitude barriers from one state; one figure."""
    import matplotlib.pyplot as plt
    from authority_barriers.experiments.make_figures import C, COL_W, constants, save
    from authority_barriers.simulator.nominal import AdversarialNominal
    from authority_barriers.theory import hocbf as HB
    from authority_barriers.theory.closed_loop import run_sampled
    from authority_barriers.theory.filters import HOCBFFilter, ProposedFilter
    st = compare_state(p, data)
    x0 = np.concatenate([st.x_L, st.v_L, st.q.ravel(), st.qd.ravel()])
    runs = [(f"hocbf_{k1:g}_{k2:g}".replace(".", "p"), lambda k1=k1, k2=k2: HOCBFFilter(p, HB.LinearClassK(k1), HB.LinearClassK(k2))) for k1, k2 in GAINS]
    runs.append(("proposed", lambda: ProposedFilter(p, altitude=True, dt=dt, accel_bound=ROWS["R1"][1])))
    recs = {}
    for name, make in runs:
        f = out / f"e8_compare_{name}.npz"
        if f.exists():
            z = np.load(f, allow_pickle=False)
            if np.allclose(z["x"][0], x0, atol=1e-12) and abs(float(z["t_final"]) - t_final) < 1e-12:
                recs[name] = {key: (str(z[key]) if key == "termination" else z[key]) for key in z.files}; continue
        cl = run_sampled(st, make(), AdversarialNominal(p), p, t_final, dt=dt)
        np.savez_compressed(f, t_final=t_final, **{key: np.asarray(v) for key, v in cl.items()})
        recs[name] = cl
    rows = {}
    for name, cl in recs.items():
        ok = np.asarray(cl["feasible"]).astype(bool); bad = np.where(~ok)[0]
        zs = np.asarray(cl["x"])[:, 6:6 + 3 * p.N].reshape(-1, p.N, 3) @ p.y_vec
        cross = np.where(zs.min(axis=1) > 0.0)[0]
        rows[name] = {"termination": str(cl["termination"]), "t_end": float(cl["t"][-1]), "min_h": float(np.min(cl["h"])), "v_end": float(cl["v"][-1]),
                      "v_max": float(np.max(cl["v"])), "feasible_at_0": bool(ok[0]), "t_infeasible": float(cl["t"][bad[0]]) if bad.size else None,
                      "z_min_at_infeasible": float(zs[bad[0]].min()) if bad.size else None, "z_max_at_infeasible": float(zs[bad[0]].max()) if bad.size else None,
                      "t_all_cables_braking": float(cl["t"][cross[0]]) if cross.size else None,
                      "min_H": float(np.min(cl["H"])) if name == "proposed" else None,
                      "climb": float(np.max(np.asarray(cl["x"])[:, 2]) - x0[2])}
    (out / "compare.json").write_text(json.dumps({"set": p.name, "period": dt, "t_final": t_final, "h0": st.h(p), "v0": st.v(p), "D0": float(recs["proposed"]["D"][0]),
                                                  "z0": st.z(p).tolist(), "w0": st.w(p).tolist(), "runs": rows}, indent=1))
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    for (k1, k2), (name, _), col in zip(GAINS, runs, (C["blue"], C["aqua"], C["yellow"])):
        cl, r = recs[name], rows[name]
        ax.plot(cl["h"], cl["v"], color=col, label=f"HOCBF $({k1:g},{k2:g})$")
        if r["t_infeasible"] is not None:
            k = int(np.argmin(np.asarray(cl["feasible"]).astype(bool)))
            ax.plot(cl["h"][k], cl["v"][k], marker="x", ms=5, mew=1.3, color=col, ls="none")
        if r["termination"] == "wall_contact":
            ax.plot(0.0, cl["v"][-1], marker="<", ms=4, color=col, ls="none", clip_on=False, zorder=5)
    cl, r = recs["proposed"], rows["proposed"]
    back = np.where(np.asarray(cl["v"]) < 0.0)[0]
    k = int(back[0]) if back.size else len(cl["t"])          # the approach, up to the stop
    ax.plot(cl["h"][:k], cl["v"][:k], color=C["ink"], label="authority filter")
    ax.plot(cl["D"][:k], cl["v"][:k], color=C["orange"], ls="--", label=r"$D_{\mathrm{res}}$ of this run")
    if r["t_all_cables_braking"] is not None:
        kc = int(np.searchsorted(cl["t"], r["t_all_cables_braking"]))
        ax.plot(cl["h"][kc], cl["v"][kc], marker="o", ms=4, mfc="white", color=C["ink"], ls="none", zorder=5)
    if back.size:
        ax.plot(cl["h"][k - 1], 0.0, marker="s", ms=3.5, color=C["ink"], ls="none", clip_on=False, zorder=6)
    ax.axvline(0.0, color=C["gray"], lw=2.0)
    ax.set_xlim(1.03 * st.h(p), -0.15); ax.set_ylim(0.0, 1.4 * max(r["v_max"] for r in rows.values()))     # room for the legend
    ax.set_xlabel("distance to the wall $h$ [m]"); ax.set_ylabel("speed toward the wall $v$ [m/s]")
    ax.legend(frameon=False, ncol=1, loc="upper left", handlelength=1.8, borderaxespad=0.2)
    fig.tight_layout()
    save(fig, out / "figures", "fig_e8_comparison",
         "E8, comparison on the exact taut-cable model from one state of the safe set (v = 3 m/s, every cable at -0.8 z_bar, h = 1.02 D), adversarial nominal, "
         "period 1 ms, in the plane of wall distance and approach speed (the wall is the right edge): the HOCBF filter with the three gain pairs of E1 "
         "(cross: first infeasible program, after which the relaxed program is applied; triangle: wall contact) and the filter with the altitude barriers "
         "(circle: every cable in the braking half-space; square: stop), with the stopping distance of its states drawn at the same speed. Expected if Thm. 5 and "
         "Thm. 12(i) hold: each HOCBF program is feasible at the start and infeasible before a cable enters the braking half-space; h >= D and no contact under the "
         "filter. Refuting: a HOCBF run that stays feasible, or a contact under the filter.",
         constants(p, {"sampling_period": dt, "T_h": p.T_h, "nu_dec": p.nu_dec, "gains": [list(g) for g in GAINS], "runs": rows}))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--t-final", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="4 states, 1 s")
    ap.add_argument("--compare", action="store_true", help="the comparison with the HOCBF filter from one state (four closed loops) and nothing else")
    ap.add_argument("--refine", action="store_true", help="the recorded contact trials under finer holds and tighter tolerances, and nothing else")
    ap.add_argument("--bench", action="store_true", help="single-process timing of D and of the filters on 200 states, and nothing else")
    args = ap.parse_args()
    if args.smoke:
        args.n, args.t_final = 4, 1.0
    p = load_set(SET); data = BarrierData.certified(p)
    out = out_dir("core", "e8_exact_model")
    if args.compare:
        for name, r in compare(out, p, data).items():
            print(name, json.dumps(r))
        return
    if args.refine:
        refine(out, p, args.t_final, args.workers); return
    if args.bench:
        bench(out, p, data); return
    rng = np.random.default_rng(args.seed)
    x0s = sample_XRF_layer(rng, p, args.n, data, v_range=(1.0, 4.0))
    t0 = time.time()
    jobs, recs = [], {}
    for row, (dt, accel) in ROWS.items():
        for k, st in enumerate(x0s):
            f = out / f"e8_{row}_{k}.npz"
            if f.exists():
                z = np.load(f, allow_pickle=False)
                if np.allclose(z["x"][0], np.concatenate([st.x_L, st.v_L, st.q.ravel(), st.qd.ravel()]), atol=1e-12) and abs(float(z["t_final"]) - args.t_final) < 1e-12:
                    recs[(row, k)] = {key: z[key] for key in z.files}; continue
            jobs.append((row, k, dt, accel, args.t_final, {"x_L": st.x_L.tolist(), "v_L": st.v_L.tolist(), "q": st.q.tolist(), "qd": st.qd.tolist()}, SUBSTEPS.get(row, 0)))
    print(f"cache: {len(recs)} of {len(ROWS) * len(x0s)} records loaded; running {len(jobs)}", flush=True)

    def keep(row, k, cl):
        np.savez_compressed(out / f"e8_{row}_{k}.npz", t_final=args.t_final, **{key: np.asarray(v) for key, v in cl.items()})
        recs[(row, k)] = cl

    if args.workers <= 1 or len(jobs) <= 1:
        for j in jobs:
            keep(*_trial(j))
    else:
        with get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
            for row, k, cl in pool.imap_unordered(_trial, jobs):
                keep(row, k, cl); print(f"  {row} {k}: {cl['termination']}", flush=True)
    per = {row: [metrics(recs[(row, k)], p) | _between(recs[(row, k)]) | {"label": f"e8_{row}_{k}"} for k in range(len(x0s))] for row in ROWS}
    lines = [f"# E8: the filter with altitude barriers on the exact taut-cable model (Thm. 12(i)) — set {p.name}, N = {p.N}", "",
             f"{len(x0s)} initial states (swing states and speeds of E2, seed {args.seed}) in the boundary layer H(x0) in [0, 0.05 D], adversarial nominal, "
             f"horizon {args.t_final:g} s; hover floor T_h = {p.T_h:g} N, swing deceleration {p.nu_dec:g} 1/s^2, altitude band [{p.alt_min:g}, {p.alt_max:g}] m; "
             f"wall {time.time() - t0:.0f} s with {args.workers} workers.", "",
             "| row | period [ms] | acceleration rows | trials | contacts | stopped by a slack cable | min h [m] | min H [m] | min H_up [m] | min H_down [m] | largest climb [m] | largest drop [m] | "
             "altitude in the band | infeasible samples | trials with an infeasible sample | median solve [ms] | min h between updates [m] | min H between updates [m] |", "|" + "---|" * 18]
    summary = {}
    for row, (dt, accel) in ROWS.items():
        M = per[row]
        s = {"period_ms": 1e3 * dt, "accel_factor": accel, "trials": len(M), "contacts": sum(m["contact"] for m in M), "slack_stops": sum(m["slack"] for m in M),
             "min_h": min(m["min_h"] for m in M), "min_H": min(m["min_H"] for m in M), "min_H_up": min(m["min_H_up"] for m in M),
             "min_H_down": min(m["min_H_down"] for m in M), "climb": max(m["climb"] for m in M), "drop": max(m["drop"] for m in M),
             "in_band": sum(m["in_band"] for m in M), "infeasible": sum(m["infeasible"] for m in M), "samples": sum(m["samples"] for m in M),
             "trials_infeasible": sum(m["infeasible"] > 0 for m in M), "solve_ms": float(np.median([m["solve_ms_median"] for m in M])),
             "min_h_between": float(np.min([m["h_between"] for m in M])) if np.isfinite([m["h_between"] for m in M]).all() else None,
             "min_H_between": float(np.min([m["H_between"] for m in M])) if np.isfinite([m["H_between"] for m in M]).all() else None}
        summary[row] = s
        hb = "—" if s["min_h_between"] is None else f"{s['min_h_between']:.5f}"
        Hb = "—" if s["min_H_between"] is None else f"{s['min_H_between']:.5f}"
        lines.append(f"| {row} | {s['period_ms']:g} | {'yes' if accel else 'no'} | {s['trials']} | {s['contacts']} | {s['slack_stops']} | {s['min_h']:.4f} | {s['min_H']:.4f} | {s['min_H_up']:.3f} | "
                     f"{s['min_H_down']:.3f} | {s['climb']:.2f} | {s['drop']:.3f} | {s['in_band']} of {s['trials']} | {s['infeasible']} of {s['samples']} | "
                     f"{s['trials_infeasible']} | {s['solve_ms']:.2f} | {hb} | {Hb} |")
    r1 = summary["R1"]
    ok = r1["contacts"] == 0 and r1["min_H"] >= -2e-3 and r1["in_band"] == r1["trials"]
    lines[0] += f" — R1 {'PASS' if ok else 'FAIL'}"
    lines += ["", f"Criterion for R1 (fixed before the run): no contact, min H >= -2 mm, altitude in the band in every trial: {'met' if ok else 'NOT met'}.", "",
              "Trials that end at the wall or with a slack cable:", ""]
    for row in ROWS:
        for m in per[row]:
            if m["contact"] or m["slack"]:
                lines.append(f"- {m['label']}: {m['termination']} at t = {m['t_end']:.3f} s; H(x0) = {m['H0']:.4f} m, v0 = {m['v0']:.2f} m/s, min H = {m['min_H']:.4f} m, "
                             f"infeasible samples {m['infeasible']}")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    (out / "results.json").write_text(json.dumps({"rows": summary, "per_trial": per, "criterion_R1_met": bool(ok), "set": p.name, "seed": args.seed,
                                                  "t_final": args.t_final}, indent=1))
    k_fig = int(np.argmax([m["v0"] if not (m["contact"] or m["slack"]) else -1.0 for m in per["R1"]]))     # the fastest approach that ends regularly
    figures(out / "figures", recs[("R1", k_fig)], p, f"e8_R1_{k_fig}")
    print("\n".join(lines[:8 + len(ROWS)]))


if __name__ == "__main__":
    main()
