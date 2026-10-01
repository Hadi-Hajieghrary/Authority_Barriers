# `authority_barriers/theory/` — the theory as a numerical library

Everything the paper defines in closed form lives here: the taut-cable model, the time-optimal swing,
the authority and the stopping distance with its exact gradients, the safety filters as second-order
cone programs, the samplers of the sets the theorems talk about, and the constants with their
assumption checks. The package depends on numpy, scipy and pydrake's solver bindings (Clarabel) only;
it never builds a Drake plant. Every experiment, the Drake filter system and the kernel computations
call into it.

Notation follows the paper (`IEEE_ACC2027/`): payload mass m_L, quadrotor masses m_i, cable
lengths l_i, unit cable directions q_i (payload → quadrotor), tensions T_i, thrusts u_i, wall normal n,
braking direction y = −n, transverse direction r = e3 × y, distance h, approach speed v, swing
coordinates z_i = q_iᵀy, w_i = q_iᵀr.

Statement labels (Thm. 5, Thm. 12, Prop. 17, Rem. 18, Sec. VI, …) are those of the full-length manuscript;
`authority_barriers/README.md`, Sec. 0, gives their numbers in the paper.

## Minimal example

```python
import numpy as np
from authority_barriers.theory.params import load_set
from authority_barriers.theory.authority import BarrierData
from authority_barriers.theory.state import State
from authority_barriers.theory import stopping
from authority_barriers.theory.filters import ProposedFilter

p = load_set("A")                                   # N = 4, the constants of the simulations
data = BarrierData.nominal(p)                       # (T_bar, T_min, nu, z_bar, m_L)
st = State.from_swing(p, h=6.0, v=3.0, z=np.full(p.N, -0.15), w=np.zeros(p.N),
                      zd=np.zeros(p.N), wd=np.zeros(p.N))   # 6 m from the wall at 3 m/s, cables toward the wall
res = stopping.evaluate(st.v(p), st.z(p), st.zd(p), data)   # D, maximizing times, gradients of E
H = st.h(p) - res.D                                 # authority barrier: D = 5.295 m, H = 0.705 m
u_nom = np.tile([20.0, 0.0, 12.0], (p.N, 1))        # nominal thrust vectors [N]: toward the wall and up
out = ProposedFilter(p).solve(st, u_nom)            # filter (17): out.u, out.T, out.feasible, out.H, out.D
```

## Modules

### `params.py` — constants and assumptions
- `Params` (frozen dataclass): the full constant set (N, m_L, m, l, f_max, T_min, T̄, ρ, ρ_fb, ν, ν_w,
  a_max, θ_q, z̄, w̄, ω̄, θ_max, g, κ_H, dt_filter, dt_attitude, d̄_L, d̄_i, wall normal n and offset d0)
  with derived properties (per-cable arrays, y/r vectors, `T_bar_rel`, `nu_rel`, `alpha_sat`,
  `alpha_sat_rel`, `T_max`, thrust-to-weight `twr`).
- `derive_set(cfg)` builds a consistent set from the *free* constants of a YAML file: T̄ from hovering at
  the maximum cable tilt with margin, then a_max (so that all cables at T̄ in parallel remain admissible,
  decision D-4), then ω̄ from Assumption 14, ρ from Assumption 13 and f_max from Assumption 7, each with
  a 5 % margin. `load_set("A")` reads `authority_barriers/configs/params_A.yaml`.
- `check_assumptions(p)` lists every inequality the theory needs as lhs ≤ rhs with its relative margin;
  `all_ok` is the gate for the stress experiments (E7 reports at which f_max Assumption 7 fails).
- `paper_example()` is the Sec. I worked example (E0 only; not an assumption-consistent set).

### `state.py` — the taut-cable state
`State(x_L, v_L, q, qd)` with the wall-coordinate accessors `h(p)`, `v(p)`, `z(p)`, `w(p)`, `zd(p)`,
`wd(p)`, `omega2()` (= ‖q̇_i‖²) and `State.from_swing(p, h, v, z, w, zd, wd)`, which builds a state from
the coordinates the theorems are written in (q_i = z_i y + w_i r + ϱ_i e3 with ϱ_i = √(1 − z_i² − w_i²)).

### `taut_model.py` — the taut-cable model (Sec. II, Lemma 1)
Right-hand sides of the point-mass model in the structured form of Lemma 1: `rhs_from_thrust(st, u, p)`
solves the closure equation for the payload's specific force a := ẍ_L + g e3 from the thrusts and returns
a, the tensions T_i and the cable accelerations; `rhs_from_tensions` does the same from (T, u^⊥). Both
are dtype-agnostic (float or pydrake `AutoDiffXd`), which is what lets the collocation differentiate
through the model. `integrate(st0, policy, t_f, p, ...)` runs scipy's DOP853/RK45 on the ambient ODE with
slack detection (a tension crossing zero ends the integration when `terminate_on_slack`), keeps the
sphere invariants ‖q_i‖ = 1, q_i·q̇_i = 0 (measured by `drift`), and returns a `Trajectory`;
`integrate_capped` adds a wall-clock cap (the braking maneuver's switching layer can make error-controlled
integrators crawl, compromise C-14).

### `profile.py` — swing of one coordinate (Sec. IV-C, App. A)
For one swing coordinate ζ with rate ω and the acceleration bound ν: the switching times
(`switching` → r, t1, t2), the piecewise-quadratic profile ζ̃(t) with first and second derivatives
(`zeta_tilde`), its partial derivatives with respect to the initial data (`d_zeta_tilde`, App. A(ii),
used by the barrier gradients), the admissible set V(ν, z̄) (`bounds_V`, `in_V`, `clamp_to_V`), the
initial acceleration `accel0` and the zero crossings of ζ̃ (where φ_i changes slope). With `nu_dec` < ν the swing
accelerates at ν and decelerates at `nu_dec` (`pieces` → r, t1, t2, the acceleration of the first piece and the
halves of its derivatives); it then arrives through the interior of V(ν, z̄), and from a state on or above the
curve ω² = 2 ν_dec (z̄ − ζ) it decelerates at the constant rate ω²/(2(z̄ − ζ)). Without `nu_dec` every function
returns the time-optimal swing.

### `authority.py` — directional authority and barrier data (Def. 8, Lemma 9)
`phi(z)` = T̄ z₊ − T_min z₋ per cable, `BarrierData` = the tuple (T̄, T_min, ν, ζ̄, m_L, offset) with which
α_y and D are evaluated, and the three instances used in the campaign: `BarrierData.nominal(p)` (the
paper's data), `.relaxed(p)` (T̄_rel = f_max-based caps and ν_rel: the relaxed authority of Thm. 12(ii)'s
outer bound D_rel) and `.robust(p)` (the substitutions of Rem. 18: ν_rob = ν − d̄_i/(m l) − d̄_L/(m_L l),
offset −d̄_L/m_L). `support_function_bruteforce` is the test oracle for Lemma 9. `BarrierData.certified(p)` is the
data of the maneuver with altitude barriers (paper, Sec. IV-D): the floor of φ is the hover floor T_h of the set,
with N T_h cos θ_q ≥ m_L g, and the swing decelerates at `nu_dec`.

### `maneuver.py` — the braking maneuver π♯ (Prop. 15)
`plan(st, p, data)` returns a `Plan` with the tensions (T̄_i if z_i > 0 else T_min), the initial swing
accelerations of the time-optimal profile, the transverse law `w_law` (saturated PD that keeps (w, ẇ)
in V(ν_w, w̄)), the perpendicular thrusts that realize both through the Gram matrix of eq. "gram"
(`swing_thrust`), the resulting thrust vectors and the braking acceleration b = α_y(z); `admissible`
checks the plan against U(x) (thrust norms, tension floor, D-4 cone). With the certified data the floor of the
tensions is T_h and the maneuver has a second phase (`holding`, `hold_tension`): once the stopping time of
`altitude.stop_time` is zero, one uniform tension holds the altitude, with a linear layer around zero vertical speed.

### `altitude.py` — altitude barriers of the maneuver (paper, Sec. IV-D)
`stop_time(mn)`: the end t_s of the braking phase as the largest of N + 1 candidate times, each smooth in the state
and decreasing at unit rate along the maneuver; `climb`, `drop`: the bounds Δ↑, Δ↓ on the climb and the drop of the
payload with their partial derivatives; `evaluate(st, p, mn)`: H↑ = (alt_max − χ) − Δ↑ and H↓ = (χ − alt_min) − Δ↓
with one barrier and gradient per candidate time; `rows`: the affine rows Ḣ ≥ −κ_alt H of the filter in the
variables of the cone program.

### `stopping.py` — stopping distance and its gradients (eq. D, Prop. 17(a))
`build(v, zeta, omega, data)` assembles the piecewise-polynomial `Maneuver` (α̃ quadratic, ṽ cubic, E
quartic between the at most 4N merged breakpoints: switch, arrival, zero crossings); `evaluate` returns a
`StoppingResult` with D = max_t E, *all* maximizing times t* (ties within 1e-9; also the local maxima),
the gradients ∂E/∂v, ∂E/∂ζ_i, ∂E/∂ω_i at each maximizer, the number of breakpoints, the `Maneuver`
itself, and flags (`in_V`, `clamped` when a swing state had to be projected into V). `D_of` and `H_of` are the scalar
shortcuts. `stopping_oracle.py` recomputes D on a dense time grid (test only, M1.2).

### `socp.py` — the shared cone program
`ConeProgram(p, st)`: decision variables x = (T_1..T_N, c_1..c_N) with c_i ∈ R² the tangent-plane thrust
coordinates, so that u_i = s_i(T) q_i + B_i c_i is affine in x; rows ‖u_i‖ ≤ f_max,i (Lorentz cones),
T_i ≥ T_min, ‖Σ T_j q_j‖ ≤ m_L a_max (D-4) and, if θ_max < π/2, the tilt cone. `tangent_basis(q)` is the
deterministic orthonormal basis of the plane ⊥ q.

### `filters.py` — the filters of Sec. V and the baselines
All return a `FilterResult` (thrusts u, tensions T, tangent coordinates c, `feasible`, `relaxed`, slack,
solver status, solve time, the payload's braking-acceleration row value b, H, D, t*, the barrier-row
values Ḣ_j, and a `rows` dict of extras).
- `ProposedFilter` — filter (17): objective Σ‖u_i − u_i^nom‖², the cone scaffold, one barrier row
  Ḣ_j ≥ −κ_H H − margin per maximizing time (`rows="maximizers"`, the paper's (17c); `"local_maxima"` is
  the D-21 comparison), and the swing constraints in mode `"sampled"` (Remark 19: each swing state must
  still be in V after one tick, a convex rotated-cone condition) or `"tangency"` (the exact (17d) rows,
  active only on ∂V). If the program is infeasible it re-solves with a slack on the barrier rows
  (`relax_on_infeasible`, decision D-15) and reports `relaxed=True`; if that program fails as well it
  returns the plan's own thrust (status `fallback_plan`). A swing state outside V is projected into V before
  D is evaluated (`clamped=True`). Options: `altitude=True` adds the rows of the altitude barriers and uses the
  certified barrier data; `solver_tol=t` sets Clarabel's feasibility and gap tolerances to t (its defaults, 1e-8,
  otherwise); `accel_bound=c` adds |z̈_i| ≤ c ν and |ẅ_i| ≤ c ν_w, which bound the change of the swing
  rates within one sampling period.
- `HOCBFFilter` — the high-order CBF baseline of Sec. III: row b(T) ≥ β(x) with β from the class-K pair
  (α₁, α₂) and, optionally, a class-K swing-rate row so that Thm. 5's hypothesis ‖q̇_i‖ ≤ ω̄ holds by
  construction.
- `decomposition_terms` — Prop. 17(b): with a common commitment â, Ḣ_j = const + Σ_i local_i(T_i, c_i).
- `BackupIntegratedFilter` (optional track, E6) — same program as (17) but the barrier data come from
  integrating π♯ numerically (RK45 at rtol 1e-8, central differences for the gradients: 2(1 + 2N) + 1
  integrations per evaluation); the "backup-CBF" baseline of Sec. VI Q3.
- `DistributedProposedFilter` (E6, decision D-27) — every vehicle solves its own small program with the
  one-step-lagged broadcast â; a scalar λ allocates the required Ḣ among the vehicles' spare capacities;
  the true coupled row is evaluated afterwards for the log (`a_hat_err`).
- `CableCBFBaseline` (E7) — the HOCBF filter plus second-order class-K rows on z̄ ∓ z_i and w̄ ∓ w_i
  ("independent per-quadrotor cable barriers").
- `MultiWallFilter` (E7 corridor) — the rows of one `ProposedFilter` per wall in one program; the result
  carries `H_walls`, `h_walls` and the critical wall.

### `hocbf.py` — the quantities of Sec. III and Thm. 5
Class-K functions (`LinearClassK`), ψ₁ = ḣ + α₁(h), β(x), the exact directional authority α^ex_y(x)
(a small SOCP), the feasibility margin μ = α^ex_y − β (K_HO nonempty iff μ ≥ 0, Prop. 3), the times
t_ψ and τ (minimum time for a cable to leave Σ_y), and the membership tests `in_Sigma_y`, `in_Xop`,
`in_Cho`, `in_D`.

### `sampling.py` — samplers of the sets in the theorems
`sample_V` (uniform in the admissible swing set), `sample_Xop`, `sample_XRF_layer` (states in the boundary
layer of X_RF: swing states in V, h = D + h_offset + δ with δ ~ U[0, 0.05 D] or an absolute layer) and
`sample_D` (constructive sampler of Thm. 5's open set D: cables well inside Σ_y at speed).

### `planar.py` — planar reductions for the kernel solvers
The one-cable (h, v, z, ż) and two-cable (h, v, z1, ż1, z2, ż2) systems with inputs (T_i, η_i),
η_i := y·u_i^⊥; the admissible input set as an ellipse (thrust limit) cut by the tension strip; its
support points, feasibility tests and per-node speed bounds (for the CFL condition).

### `closed_loop.py` — the sampled loop on the exact model
`run_sampled(filter, nominal, x0, ...)`: the zero-order-hold filter on the taut ODE itself (no plant); each hold
is integrated with DOP853 (`rtol`, `atol`, default 1e-9 and 1e-11). The record holds, per tick, the largest violation
of the tension floor and of the thrust limit by the applied command (`T_viol`, `f_viol`) and, with `substeps=k > 0`,
the minima over k states inside the hold of h, H, the swing margins of z and w, and the altitude barriers
(`h_between`, `H_between`, `margin_z_between`, `margin_w_between`, `H_up_between`, `H_down_between`; NaN otherwise),
computed by `between_samples`. Used by the attribution procedure, by `common.calibrate_margin` to measure Rem. 19's
sampled-data deficit on the model alone, and by the experiment E8 (`experiments/e8_exact_model.py`).

## Data flow of one filter tick

```
State st ──▶ stopping.evaluate(v, z, ż, data) ──▶ D, t*_j, ∂E/∂ζ, ∂E/∂ω
                                                        │
socp.ConeProgram(p, st) + filters.swing_affine(...)      ▼
   rows: thrust balls, T ≥ T_min, D-4 cone, Ḣ_j(x) = A_j x + b_j ≥ −κ_H H, one-tick swing cones
   objective: Σ‖u_i − u_i^nom‖²
                     │ Clarabel
                     ▼
FilterResult(u, T, c, feasible, H, D, Ḣ_j, ...)
```

## Invariants of the library (gate G1)
The following properties were checked numerically in the simulation study; the checks are not part of the
repository. Lemma 9's support function; D against the dense-grid oracle (M1.2); the analytic gradients of E against
finite differences (M1.1, relative 1e-5); the plan realizes b = α_y and the closed-form swings on the exact
model with H nondecreasing (M1.4); the plan satisfies the filter rows and Clarabel finds it feasible
(M1.5); Prop. 17(b)'s identity holds to 1e-13 with â = a(T) and fails with a wrong â (M1.6); sign(μ)
predicts HOCBF feasibility (M1.9); the sphere invariants survive 10 s of integration without projection
(M1.11); the planar reductions equal the ambient model.
