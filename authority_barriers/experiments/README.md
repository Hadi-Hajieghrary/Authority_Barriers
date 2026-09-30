# `authority_barriers/experiments/` — the experiments of Sec. VI, their figures and videos

One module per experiment. Each driver samples the initial states its question needs, runs the trials
through `authority_barriers.simulator.harness` (in parallel, with a per-trial cache so that an interrupted run resumes),
computes a verdict per trial from the log *after* the simulation, attributes every violation, and writes
two files into `results/<track>/<experiment>/`: `summary.md` (the human-readable tables with the
verdict in the title line) and `results.json` (the same numbers per trial, machine-readable). Every
summary states the observation that was expected if the claim holds and the one that would refute it,
as fixed before the runs. `results/` is the default results directory (`SIM_RESULTS_DIR` overrides it); it is
written by the drivers and is not part of the repository.

Statement labels (Thm. 5, Thm. 12, Prop. 17, Rem. 18, Sec. VI, …) are those of the full-length manuscript;
`authority_barriers/README.md`, Sec. 0, gives their numbers in the paper.

## Map of the experiments

| Driver | Paper item | Question and setup | Per-trial verdict | Outputs (under `results/`) |
|---|---|---|---|---|
| `e0_worked_example.py` | Sec. I / III example | Reproduce the paper's numbers (ψ₁ = 0.15 m/s, β = −2.15 m/s², α^ex = −1 m/s², μ = 1.15 m/s², t_ψ = 0.18 s, τ = 0.25 s, x0 ∈ D) | exact match | printed (M0.1) |
| `e1_thm5.py` | Thm. 5, Table I row 1, Q1 | From 100 states in D (all cables leaning toward the wall, 2–4 m/s), the HOCBF filter with three gain pairs (1, 20), (0.5, 10), (2, 40): does it lose feasibility before τ while h > 0? Plus the proposed filter from the same states. | PASS iff the program becomes infeasible (μ < 0, two consecutive ticks) at t ≤ τ − dt with h > 0, cables still in Σ_y and ‖q̇_i‖ ≤ ω̄ up to then; also the proof bound μ ≤ α₂(ψ₁) on [0, τ]. A trial feasible through τ under the hypothesis **refutes** Thm. 5. For the proposed filter: min h ≥ 0 whenever x0 ∈ X_RF. | `core/e1_<actuator>[_A_N3]/` |
| `e2_thm12i.py` | Thm. 12(i), Table I row 2, Rem. 19 | From the boundary layer of X_RF (H(x0) ∈ [H_min, H_min + 0.05 D], or H(x0) uniform in [0, 1] m with `--layer-abs 1`), the proposed filter under the *adversarial* nominal (full thrust toward the wall): invariance and feasibility. Variants: A (raw filter), C (absolute layer, measures the contact rate against H(x0)), B (rows at all local maxima, D-21). | PASS iff min h ≥ 0, feasible at every tick, model tension ≥ T_min − 0.05 N, swing states in V. A non-pass is attributed: sampled-data case (contact depth within the accumulated inter-tick deficit of H), infeasible ticks, swing state left V, or the M3.4 replay. | `core/e2_<actuator>_<rows>_off<H_min>[_layer1]/` with `<rows>` = `max` or `lmax` |
| `e3_kernel.py` | Thm. 12(ii), Table I row 3, Q2 | 4-D viability kernel (set A1) on a ladder of grids; boundary h* at 20 fixed cable states × 10 speeds against h = D and h = D_rel. | M4.2: change of h* between grids ≤ max(2 % h*, 1 cell); M4.3: D_rel − tol ≤ h* ≤ D + tol at every probe. A probe with h* < D_rel **refutes** Thm. 12(ii). | `core/e3_kernel/`, `core/e3_kernel_summary.md` |
| `e3_collocation.py` | Thm. 12(ii) "in three dimensions" | N = 3 (set A_N3), three fixed cable configurations (A: all leaning toward the wall; B: mixed; C: all at z̄) × 8 speeds: smallest feasible initial distance h_c by bisection of direct-collocation searches into X_RF; witness from h0 = D; independent re-simulation. | M4.4 PASS iff D_rel − 1 cm ≤ h_c ≤ D; a verified trajectory from h0 < D_rel − 1 cm **refutes** Thm. 12(ii); solver failures are never evidence. | `core/e3_collocation_summary.md`, `core/e3_collocation_jobs/` |
| `e4_prop17.py` | Prop. 17, Table I row 4 | Runtime of D + gradients and of the filter for N = 2..8 (M5.1); the distributed decomposition Ḣ = const + Σ local_i against the coupled row, algebraically at 300 random states and along 20 E2 logs with the logged commands (M5.2); the mismatch when â is the *measured* specific force instead of the commanded one. | slope of runtime vs N ≤ 1.3; identity to 1e-10 / 1e-8 | `core/e4/` |
| `e5_robust.py` | Rem. 18, Table I row 5 | The robust filter (D_rob) under wind clipped at d̄ (`--scale 1`) from X_RF^rob: min_t H_rob ≥ 0. Two layers (boundary layer, and H_rob(x0) uniform in [0, 1] m); both actuator modes. `--scale 2` (E7 gusts at 2 d̄) deliberately violates the remark. | PASS iff min_t H_rob ≥ 0; losses attributed (sampled-data case vs inner-loop model gap) | `core/e5_scale1[_attitude][_layer1]/`, `optional/e5_scale2_layer0.5_off0.8/` |
| `e6_backup_compare.py` | Sec. VI Q3 (optional) | The backup-CBF filter that integrates π♯ numerically vs the proposed closed form on 60 X_RF states: \|D_num − D\|/D, the command difference against 1 % f_max, the wall time per evaluation, and the ratio of integrator calls to breakpoints (the paper's speed prediction). | agreement within 1e-3 and 1 % f_max; speed ratio reported | `optional/e6_backup/` |
| `e6_distributed.py` | Prop. 17(b) in closed loop (optional, D-27; not in the paper) | The distributed filter with a one-step-lagged broadcast of â against the centralized one from the same 10 states on the full plant, adversarial nominal: \|ΔH\|, the â mismatch (charged to the d̄ budget), contacts and infeasible ticks. | \|ΔH\| within the budget charged to the mismatch; mismatch ≤ 0.05 N force-equivalent | `optional/e6_distributed/` |
| `e3b_kernel6d.py` | Q2, two cables (optional) | Coarse 6-D kernel (set A2) against D and D_rel at 20 sampled swing states × 6 speeds; the resolution is the result. | both bounds within 2 cells | `optional/e3b_kernel6d/` |
| `e7_stress.py` | "limits of the guarantee" (optional) | `fmax`: f_max from 44 to 20 N in 2 N steps with Assumption 7 checked per value (10 states each); `mass`: true m_L 0.8 / 1.2 kg with the nominal filter and with the filter designed for the heaviest payload (set A_mL12, D-26; 20 states each); `soft`: cable stiffness /10; `omega`: initial swing rates 1.5 ω̄ (outside X_op). Adversarial nominal, H(x0) ∈ [0.8, 1.3] m (above the measured sampled-data margin), 10 s. | contacts per variant, each attributed (sampled-data case or not); M8.3 PASS iff no contact is attributable to the reduced thrust; M8.4 PASS iff 0 contacts with the mass rule | `optional/e7_stress/` |
| `e7_corridor.py` | "50 trials of nominal transport in a 20 m corridor" (optional, D-7) | Two walls at ±10 m (`MultiWallFilter`), transport along the corridor at 2–4 m/s with a lateral drift U[−1, 1] m/s and a formation-hold nominal, 50 random states in X_RF of both walls, 15 s: contacts and ‖u − u_nom‖/‖u_nom‖ per tick against the paper's "at most 5 %"; the barrier rows alone are re-solved every 25th tick to decompose the change. `--gentle`: a sensitivity subset with a gentler nominal (k_swing 1.0, k_q 0.5). | 0 contacts; change ≤ 5 % (else the sentence is amended, D-7) | `optional/e7_corridor/` (`summary[_gentle].md`, `results[_gentle].json`) |
| `e7_montecarlo.py` | "a Monte Carlo campaign of 500 trials" (optional, D-24, D-28) | 500 seeds; per seed the stress factors are drawn (gust scale U[0, 2] d̄, f_max U[20, 44] N, true m_L ±20 %, soft cables with probability 0.3, swing rates × U[0.5, 1.5], H(x0) U[0.8, 1.5] m, v0 U[1, 4] m/s), transport toward the wall for 10 s, four baselines in closed loop (unfiltered, HOCBF (1, 20), HOCBF + per-quadrotor cable barriers, proposed); the backup filter is represented by E6's equivalence. | per baseline: contact rate, contacts within every hypothesis (Assumption 7, X_op, gusts ≤ d̄, stiff cables), trials with infeasible ticks, min h, min H, solve time; the proposed filter's contacts broken down by factor | `optional/e7_montecarlo/` |
| `e8_exact_model.py` | Thm. 12(i) in its own setting | The filter with altitude barriers (set `A_alt`: hover floor, swing deceleration, altitude band) on the exact taut-cable model through `closed_loop.run_sampled`: the 100 swing states and speeds of E2 (seed 11) in the boundary layer of the safe set, adversarial nominal, 6 s. Rows: sampling period 1 ms (R1) and 5 ms (R2) with the rows that bound the swing accelerations, and the same two periods without them (R1n, R2n). Option `--compare` runs four closed loops from one state of the safe set (v = 3 m/s, every cable at −0.8 z̄, h = 1.02 D) at 1 ms: the HOCBF filter of E1 with its three gain pairs and the filter with altitude barriers, and writes `compare.json` and the figure `fig_e8_comparison` (wall distance against approach speed). | R1: no contact, min H ≥ −2 mm, altitude in its band in every trial (fixed before the run); a contact in R1 would contradict the theorem up to the hold of 1 ms | `core/e8_exact_model/` (`summary.md`, `results.json`, one record per trial, `figures/`) |
| `bench_library.py` | M1.7 | runtime of the closed-form barrier and the filter | — | printed |

Actuator modes: `--actuator perfect` (the paper's Sec. II model, the mode that can refute a theorem) and
`--actuator attitude` (the Sec. VI setup with the 1 kHz geometric loop, which measures the inner-loop gap,
C-10); every closed-loop experiment of the necessary track was run in both (D-19). `--smoke` reduces trial
counts and horizons for a quick end-to-end check (3–4 trials, 3 s); run through
`authority_barriers.reproduce --smoke` it writes to the results directory `results_smoke`.

## Shared helpers — `common.py`
- `run_many_cached(cfgs, x0s, out, n_workers)`: runs only the trials whose log (`<label>.npz` + `.json`)
  is missing or was produced with a different configuration or initial state (the wall cap is ignored in
  the comparison; a log truncated by a cap is re-run if the cap was raised). This is what makes every
  driver resumable.
- `in_XRF`, `swing_states_in_V`: membership tests from a state / along a log.
- `sampled_deficit`, `sampled_allowance` (D-20 revised): the per-tick deficit d_k = max(0, H_k(1 − κ dt) − H_{k+1})
  of the continuous-time barrier condition, its accumulation m(t) (Rem. 19's sampled-data loss measured on
  the log, model-free), and the test "every contact depth ≤ m(t)" that defines a *sampled-data case*.
- `replay_on_taut_ode`, `classify_violation` (the M3.4 attribution procedure): H(x0) below the one-tick
  margin or a swing state clamped into V → sampled-data; a slack cable or an attitude error beyond the
  budget → model gap; otherwise the logged zero-order-hold commands are replayed on the exact taut-cable
  model from the same x0: a replay that also violates is a *theory-class candidate* to escalate, one that
  stays safe is a model gap.
- `calibrate_margin`: the sampled-data margin δ measured on the model alone (`closed_loop.run_sampled`).
- `out_dir`, `summary_table`, `save_json`: result folders (`SIM_RESULTS_DIR` overrides `results`),
  markdown tables, JSON with numpy types.

## Figures
- `make_figures.py --core`: the Table I figures and table sidecars from the core results — E1 μ(t)
  vs the proof bound and the onset distribution per gain pair and actuator, E2 contact rate and contact
  speed against H(x0) per actuator, the kernel boundary against the two bounds per probed cable state, the
  collocation map per configuration, the E5 violation rate against H_rob(x0); `table_e1b_thm12i`,
  `table_e4_prop17`, `table_e5_rem18` as JSON sidecars; `results/core/figures/index.md`.
- `make_figures_optional.py`: E6 command difference and cost ratio, E6 distributed â mismatch and min h,
  E3b boundary against the bounds at 6 states, E7 f_max sweep with the Assumption-7 threshold, corridor
  thrust-change distribution, Monte Carlo contact rates and contact rate against f_max, gusts at 2 d̄;
  `results/optional/figures/index.md`.
- `sequence_figures.py`: consecutive snapshots of the simulation from one fixed camera per scenario, one frame per
  file — four frames evenly spaced over the maneuver, framed so that the whole maneuver and the wall are in view and
  cropped to the content (wall panel with a 1 m grid, earlier payload positions as translucent spheres, the distance h
  and the stopping point x_L + D n drawn on the scene, t, h, v, D, H in a bar above the frame). Output
  `sequence_<j>.png` in each headline trial's folder under `results/core/figures/trials/`, plus a "Sequence" section
  in the trial's `index.md`.
- `paper_figures.py`: seven groups of figures that carry the arguments of the paper, one plot or one rendered frame
  per file; a figure is a group of files that the paper composes (anatomy of D: swings, authority, braking profile; HOCBF vs
  authority-barrier filter: side-view sketches and three traces; storyboard of one maneuver: six Meshcat frames and
  the h/D, z_i, tension plots; the viability sandwich per configuration; the sampled-data margin map and its
  mechanism; the thrust budget and the f_max sweep; D(ν)/h_c against ν). Vector PDF at the printed width, 8 pt fonts,
  direct labels, one palette. Output `results/core/figures/paper/` with `captions.md` (one claim-first caption per
  group). The paper includes files of the groups 1, 4, 5, 6 and 7; `IEEE_ACC2027/collect_figures.py` takes them into
  `IEEE_ACC2027/figures/`.
- `trial_figures.py`: trial-level figures for the headline trials (`headline()`): Meshcat renders of the
  real simulation at the key instants (start, first infeasible tick, fastest swing, cables past vertical,
  peak braking, minimum H, stop, closest approach, end) from eight camera views, one time series per
  signal (distances and barrier, speeds, accelerations, model and spring tensions, thrust magnitude and
  tilt, tick-to-tick command change, swing states and rates, filter status, solve time, attitude error,
  altitude) and the swing phase plane against V; also the B 4 m/s collocation trajectory rendered the same
  way. Output `results/core/figures/trials/<trial>/` with an `index.md` and a `sidecar.json` each.

Rules for every figure: one plot per file; a JSON sidecar with the constants the paper
requires with every figure (integrator, step, solver, γ = κ_H H, ρ_i, T̄_i, ν, the set, N, f_max, …) and a
caption that states the expected and the refuting observation.

## Videos
`trial_videos.py` replays sixteen recorded scenarios in slow motion, one video each (1920 x 1080, H.264, 30 frames per
second): the HOCBF filter and the authority-barrier filter from the same initial state (E1); the adversarial nominal
command with the closest call, a contact by the sampled-data mechanism and a contact with the attitude loop (E2); a
direct-collocation trajectory for N = 3, continued by the braking maneuver from its end state (E3); the robust filter
under wind without and with a contact (E5); the distributed filter with a delayed broadcast (E6); the corridor, one
Monte Carlo trial with and without a filter, and four stress cases: thrust limit 20 N, payload of 1.2 kg, soft cables,
swing rates above the limit (E7). `scenarios()` states the rule by which each record is chosen.

A frame has four parts.
- *Close-up*: the team from beside the payload, turned 15° behind it and raised by 18°, in parallel projection, by a
  camera that moves with the payload. Vertical lines stay vertical and the wall normal lies along the horizontal axis
  of the picture, so the lean of a cable is seen undistorted. Thin lines from the payload mark the vertical and the
  leans ±z̄ of the operational cone: a cable to the right of the vertical leans toward the wall and cannot brake, a
  cable to its left is behind the payload. A grid of 1 m behind the team and the panel of the wall show the motion of
  the payload; the panel begins at the lateral position of the payload and extends toward the camera, so its edge in
  the picture is the point that the payload approaches. For the corridor the camera looks along the corridor, from
  behind the team and above, over a floor grid.
- *Overview*: the whole maneuver and the wall from the same direction, fixed camera.
- *Plots*: h with D, and the leans z_i, with a cursor at the instant shown and a strip where the program of the filter
  is infeasible.
- *Readouts*: t, h, v, D, H, the state of the filter, the number of cables behind the payload, the legend, and the
  cable directions seen from above (one dot per vehicle at (z_i, w_i) inside the box of the operational set).

The frame of the close-up is widened when the team of a record needs more room, and the rendering stops with an error
if the payload, a vehicle or the tip of a thrust arrow leaves the picture. States between two samples of a record are
interpolated (linearly at the 5 ms of the trial logs, cubically in the positions between the knots of the collocation
trajectory). A command above the thrust limit is drawn at the limit, which is what the plant applies. The quadrotors
are drawn at 0.75 of a 0.3 m airframe; the model treats them as points.

`--gif` converts the videos of the folder to `<name>.gif` (960 x 540, 10 frames per second, 128 colors), which the main
README shows. The folder `results/videos/` is part of the repository; the rest of `results/` is not.

Output `results/videos/<name>.mp4`, one frame as `<name>.png`, the sidecar `<name>.json` (record, window, cameras,
constants, the numbers quoted in the description) and `README.md` (how to read a video, one entry per video). Needs
the `[viz]` extra and `ffmpeg`.

## Running
```
python -m authority_barriers.experiments.e1_thm5 --actuator perfect --workers 8
python -m authority_barriers.experiments.e2_thm12i --actuator perfect --rows maximizers --h-offset 0 --layer-abs 1.0
python -m authority_barriers.experiments.e3_kernel --grid 121 --zn 41 --h-max 10 --threads 4 ; python -m authority_barriers.experiments.e3_kernel --compare
python -m authority_barriers.experiments.e3_collocation --configs A,B,C --workers 6
python -m authority_barriers.experiments.e4_prop17 --logs "results/core/e2_perfect_max_off0/*.npz"
python -m authority_barriers.experiments.e5_robust --scale 1 --actuator perfect [--layer-abs 1.0]
python -m authority_barriers.experiments.e6_backup_compare --n 30 --workers 3 ; python -m authority_barriers.experiments.e6_distributed --n 10
python -m authority_barriers.experiments.e3b_kernel6d --shape 25 25 13 13 13 13 --threads 8
python -m authority_barriers.experiments.e7_stress --variant all ; python -m authority_barriers.experiments.e7_corridor --n 50 ; python -m authority_barriers.experiments.e7_montecarlo --n 500 --workers 6
python -m authority_barriers.experiments.e8_exact_model --t-final 6 --workers 8                     # about 30 min
python -m authority_barriers.experiments.e8_exact_model --compare                                   # about 1 min
python -m authority_barriers.experiments.trial_videos [--only e1_hocbf,e7_corridor] [--list]      # after the experiments
```
or all of it in order through `python -m authority_barriers.reproduce`. The sequences in which the experiments were
launched for the paper are in `scripts/run_*.sh` (see `scripts/README.md`).
