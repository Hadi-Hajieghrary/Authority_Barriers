# `authority_barriers/simulator/` — the full-order simulator and trial harness

This package builds the closed-loop simulation the paper's Sec. VI describes: rigid quadrotors and a
rigid payload in a Drake `MultibodyPlant`, cables as unilateral spring–damper elements that can go slack,
thrust and torque applied at the vehicles, a 1 kHz geometric attitude controller, bounded wind, and the
200 Hz safety filter of `authority_barriers/theory` closing the loop on the *projected* taut-cable state. One call
(`harness.run_trial`) turns a `TrialConfig` and an initial `State` into a `TrialLog`; the experiment
drivers in `authority_barriers/experiments` do nothing else with Drake.

Statement labels (Thm. 5, Thm. 12, Prop. 17, Rem. 18, Sec. VI, …) are those of the full-length manuscript;
`authority_barriers/README.md`, Sec. 0, gives their numbers in the paper.

## What one trial looks like

```
 x0 (taut-cable State) ─set_initial_state─▶ MultibodyPlant (N free quadrotor bodies + payload; continuous, time_step = 0)
                                                 ▲  applied spatial forces        │ plant state (1 kHz log)
                    ┌────────────────────────────┼─────────────────────────────┐   │
                    │ CableForces (unilateral k, c)                              │   ▼
                    │ WindForces (optional, filtered Gaussian, clipped at d̄)     │ StateProjector ─▶ packed taut-cable state
                    │ ThrustTorqueApplier ◀─ GeometricAttitudeController (1 kHz) │        │
                    │            or DirectThrustApplier ("perfect" actuator)     │        ▼
                    └────────────────────────────────────────────────────────────┘ FilterSystem (200 Hz): u_nom = nominal(t, x),
                                            ▲ thrust commands (zero-order hold)         u = filter.solve(x, u_nom); diagnostics
                                            └───────────────────────────────────────────┘
 Monitor: wall contact (h < 0), wall-clock cap; loggers at 1 kHz (plant state) and 200 Hz (diag, commands, cable data, taut state, attitude error)
```

Integration: implicit radau3 at accuracy 1e-6 with a 1 ms maximum step (decision D-17); the choice was
cross-checked against error-controlled RK3 (payload position within 1e-6 m over a trial, M2.5).

## Modules

| Module | Role |
|---|---|
| `plant.py` | `build_plant(p, builder, visualize, ...)`: N free rigid quadrotor bodies (mass m_i, inertia J) and a point-like payload in a continuous `MultibodyPlant`; `PlantInfo` (bodies, state index maps); `set_initial_state` places quadrotor i at x_L + (l_i + e_i) q_i with the cable pre-stretch e_i that matches the taut-model tension (so that t = 0 is consistent with the theory's state) and the attitudes implied by the first command; `read_state` recovers the taut-cable `State` from the plant state. With `visualize=True` the bodies get colored geometry (`QUAD_COLORS`) for Meshcat. |
| `cables.py` | `CableForces`: T_i = max(0, k e_i + c ė_i) for stretched cables, zero when slack (e_i = ‖p_i − x_L‖ − l_i); force −T_i q_i on the quadrotor, +T_i q_i on the payload. `cable_constants(p)` picks k so that the stretch under f_max stays below 0.1 % of the cable length (k = 5e4 N/m for set A) and c for a damping ratio 0.4 against the reduced mass (c = 139 N s/m). Its `cable_data` output port carries the tensions, the stretches and the measured payload specific force a_meas at every filter tick. |
| `actuation.py` | `ThrustTorqueApplier`: per-quadrotor commands [f_i, τ_i] (thrust magnitude along the body z-axis, clipped to [0, f_max,i], and a body torque) → `ExternallyAppliedSpatialForce`s at the centres of mass; quaternion/state-indexing helpers shared with the controller. |
| `attitude.py` | `GeometricAttitudeController`: the discrete 1 kHz SO(3) tracking law of Lee, Leok and McClamroch (Ω_d = 0) that turns each commanded thrust *vector* into [f_i, τ_i]; natural frequency `omega_n` (300 rad/s in the campaign, the practical limit of a 1 kHz loop, decision D-8 revised); publishes the tracking error d_att,i = ‖f_i R_i e3 − u_i‖ that Rem. 18 charges to the disturbance budget d̄_i (0.15 N of it). Non-direct-feedthrough by construction (no algebraic loop). |
| `direct_thrust.py` | `DirectThrustApplier`: the "perfect" actuator of the paper's Sec. II model — the commanded thrust vectors are applied directly as world-frame forces at the quadrotor CoMs (norm clipped at f_max). |
| `wind.py` | `WindForces` (decision D-16): first-order filtered Gaussian gusts (τ_w = 0.5 s, sampled at 100 Hz and held), clipped in norm at `scale`·d̄_L on the payload and `scale`·d̄_i on each quadrotor; `scale=1` is Rem. 18's budget, `scale=2` the deliberate violation of E7. |
| `projector.py` | `StateProjector`: plant state vector → packed taut-cable state (q_i from the vehicle–payload offsets, q̇_i tangent), so that the filter and the recorder see exactly the quantities the theory is written in. |
| `filter_system.py` | `FilterSystem`: the 200 Hz filter as a discrete `LeafSystem`. At each tick it evaluates the nominal controller and the filter on the projected state, stores the thrust commands (zero-order hold until the next tick) and a diagnostics vector (`DIAG_FIELDS`, below) in discrete state, and exposes both from state (no direct feedthrough). |
| `nominal.py` | The unfiltered controllers evaluated inside the tick. `VelocityTransportNominal` is a geometric transport controller for the taut-cable team (the transport controller of Sec. VI, used by E1, the corridor and the Monte Carlo): the desired payload specific force a_d (velocity error, gravity, altitude PD) with its horizontal part limited to the lean the operational set allows (half the swing box: 0.5 z̄ toward a wall, 0.5 w̄ along a corridor); a formation reference around a_d's direction with the cables placed on an ellipse of half the swing box (radii 0.5 z̄ along y, 0.5 w̄ along r; the vehicles surround the payload and stay inside the box with a margin); bounded least-squares tensions for m_L a_d on the current cable directions (vertical component weighted 3×); and perpendicular thrusts u_i^⊥ = m_i P_i a + m_i l_i (k_q (q_ref,i − q_i)_⊥ − k_swing q̇_i) — the feed-forward that eq. cable requires plus a PD on the cable direction. Without the feed-forward m_i P_i a a tilted cable accelerates away from vertical and the team collapses on its own (compromise C-15, found 2026-09-26 through the corridor snapshots; the affected experiments were re-run). `AdversarialNominal` (full thrust toward the wall, tilted 40° up so the team also pulls the payload — the worst case for E2/E5/E6/E7 stress); `HoverNominal` (the transport controller with zero velocity command). |
| `harness.py` | `TrialConfig` (all knobs of a trial, below), `corridor_params` (two walls with normals ±r at ±half width), `apply_overrides` (per-trial constants, e.g. a different f_max for the plant and the filter), `make_filter` / `make_nominal` (string → object), `build_diagram`, `run_trial` (with the termination monitor) and `run_many` (parallel processes; Meshcat never starts in workers). |
| `recorder.py` | `TrialLog`: the arrays of a trial plus a JSON sidecar with every constant a figure must report; `save`/`load` (`<label>.npz` + `<label>.json`). |
| `viz.py` | `SceneRenderer`: replays a logged plant state (or a taut-cable state with the attitudes implied by the commanded thrust) in a visualization diagram with Meshcat, draws the cables, the commanded thrust vectors, the payload velocity and the wall(s), places the camera relative to the payload (eight named views: `side`, `rear`, `close_side`, `close_front`, `close_low`, `close_top`, `arena_side`, `arena_high`) or fixes it once for a whole sequence (`frame`: three-quarter view fitted to a set of points, kept in front of every wall), and screenshots the page through headless Chromium (Playwright). Cables, thrust vectors and markers are thin cylinders (WebGL lines are 1 px), `set_wall_grid` shrinks the wall to a gridded panel (1 m cells) around a point, and `show` can draw earlier payload positions as translucent spheres and the distance h with the stopping point x_L + D n. `stamp` writes a caption bar into the PNG. For sequences of many frames (`experiments/trial_videos.py`): `capture` returns the picture after the page has reported a serial number that is sent after the scene, so the picture shows every change made before it; `parallel` sets a parallel projection; `set_wall_panel`, `set_backdrop`, `set_floor`, `set_reference`, `set_plumb` and `set_breadcrumbs` draw the references against which a camera that moves with the payload shows the motion; `style` replaces the sizes and colors of what `show` draws and `visual_scale` the drawn size of the quadrotors. The renderers of a process share one browser, one page each. Nothing here touches the simulation. |

## `TrialConfig` — the knobs of a trial

| Field | Meaning |
|---|---|
| `params` | parameter set name (`A` for every closed-loop trial; `A_N3` for the N = 3 replication; `A_mL12` for the mass rule) |
| `filter` | `proposed` (filter (17)), `hocbf` (Sec. III baseline with `gains` = (k1, k2)), `none` (unfiltered), `distributed` (D-27), `cable_cbf` (E7 baseline), `backup` (E6) |
| `robust` | use the robust barrier data D_rob of Rem. 18 |
| `swing_mode` | `sampled` (Remark 19's one-tick rows, the campaign default) or `tangency` (exact (17d)) |
| `hdot_margin` | additional slack δ on the barrier rows (0 = the raw filter (17)) |
| `rows` | `maximizers` (the paper's (17c)) or `local_maxima` (D-21 comparison, reverted after E2) |
| `nominal`, `v_cmd` | `velocity` (transport at v_cmd toward the wall), `adversarial`, `hover`, `corridor` (transport along the corridor with formation hold) |
| `t_final`, `seed` | horizon [s]; seed of the wind and of any per-trial randomness |
| `wind_scale` | 0 = no wind, 1 = clipped at d̄, 2 = at 2 d̄ |
| `integrator`, `accuracy`, `max_step` | radau3, 1e-6, 1e-3 s (D-17) |
| `m_L_true` | the plant's payload mass if it differs from the filter's nominal (E7 mass factor) |
| `cable_k_factor` | multiplies the cable stiffness (0.1 = "softened cables") |
| `omega_n_att` | natural frequency of the attitude loop (300 rad/s) |
| `actuator` | `attitude` (Sec. VI setup) or `perfect` (Sec. II model) |
| `log_mu` | also log the HOCBF feasibility margin μ (E1) |
| `terminate_on_wall` | stop at the first tick with h < 0 |
| `wall_cap` | wall-clock seconds per trial; the monitor stops the trial (`termination = "wall_cap"`), reported as truncated (C-12) |
| `param_overrides` | constants overriding the named set for the plant and the filter (E7: f_max sweep, mass rule) |
| `corridor_half_width` | two walls at ±half width (E7 corridor, `MultiWallFilter`) |
| `qd_scale` | scales the sampled initial swing rates (1.5 = outside X_op) |
| `nominal_k_swing`, `nominal_k_q` | gains of the transport nominal (the corridor's gentler-nominal subset uses 1.0 / 0.5) |
| `label` | file stem of the log |

`run_trial` ends with `termination` ∈ {`horizon`, `wall_contact`, `wall_cap`} in the log's metadata.

## What a trial log contains (`recorder.TrialLog`)

| Array | Shape | Content |
|---|---|---|
| `t_plant`, `x_plant` | (K,), (nx, K) | the plant state at 1 kHz (quaternion, position, angular and linear velocity per free body) |
| `t_diag`, `diag` | (M,), (32, M) | the 200 Hz diagnostics, one row per `DIAG_FIELDS` entry: `t, h, v, H, D, D_rel, D_rob, feasible, relaxed, slack, solve_time, b` (payload braking acceleration along y implied by the tensions), `du_norm` (‖u − u_nom‖), `utilization` (max_i ‖u_i‖/f_max,i), `minT_cable` (smallest physical tension), `max_omega` (max_i ‖q̇_i‖), `n_rows` (barrier rows), `in_V`, `clamped`, `t_star`, `beta, mu, psi1` (HOCBF quantities when logged), `in_Sigma_y`, `min_z`, `a_meas_norm`, `alpha_y` (authority at the current lean), `H_rob`, `hdot0` (first barrier-row value), `unom_norm`, `lambda`, `a_hat_err` (distributed variant) |
| `cable` | (2N + 3, M) | physical tensions T_i, stretches e_i and the measured payload specific force a_meas at the ticks |
| `cmd` | (3N, M) | the zero-order-held thrust commands |
| `att_err` | (N, M) | attitude tracking error at the ticks (NaN with the perfect actuator) |
| `taut` | (6 + 6N, M) | the projected taut-cable state the command was computed for |
| `meta` (JSON) | — | the `TrialConfig`, the parameter set and derived constants, cable k and c, integrator settings, filter period, solver, κ_H, gains, wall time, x0, `termination`, the diagnostics field list and the git commit |

`log.d("H")` returns one diagnostic by name. Everything an experiment reports (min h, contacts,
feasibility, tensions, swing-set membership, the sampled-data deficit, the attribution replay) is computed
from these arrays after the trial; nothing is decided inside the simulation.

## Actuator modes and what they measure
- `perfect` is the paper's Sec. II model: the thrust vector is applied as commanded. The theorems are
  stated for it, so it is the mode that can refute them; its only unmodeled effects are the cable
  compliance and the sampled filter.
- `attitude` is the paper's Sec. VI setup. The geometric loop cannot follow the sampled filter's
  tick-to-tick command jumps within the 0.15 N budget of Rem. 18: peaks of 17–43 N were measured
  (compromise C-10), which is why every closed-loop experiment was run in both modes (decision D-19) and
  why attitude-mode violations are attributed to the inner loop rather than to the theory.

## Fidelity and budgets (gate G2)
The following properties were checked numerically in the simulation study; the checks are not part of the
repository. The plant with stiff unilateral cables reproduces the taut-cable ODE under identical open-loop thrust
commands (M2.1); the attitude loop tracks a 30° step and a rotating command within the budget in steady
state (M2.2); the logging channels, determinism (identical logs from identical seeds) and the parallel
runner work (M2.4); slack cables and bounded wind behave as specified.

## Practical notes
- Wall time: perfect-actuator trials cost seconds to minutes per simulated second; attitude-mode trials
  with an active filter cost 30–860 s per simulated second because radau3 collapses its step through the
  inner loop's transients (C-12). Long trials therefore run under `wall_cap` and are reported as truncated
  when the cap fires; a truncated trial is never counted as a pass or a violation.
- `run_many` spawns worker processes; set `OMP_NUM_THREADS=1` in the environment to keep Clarabel and
  numpy from oversubscribing the cores.
- `viz.py` needs the `[viz]` extra (`playwright`, `pillow`) and `playwright install chromium`; the Meshcat
  page's widgets are hidden through CSS before the screenshot.
