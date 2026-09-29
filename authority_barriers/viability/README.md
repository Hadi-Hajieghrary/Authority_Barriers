# `authority_barriers/viability/` — viability-kernel computations (Thm. 12(ii), Sec. VI Q2)

Theorem 12(ii) says that the *viability kernel* of the constraint set — the set of states from which
some admissible input keeps the payload off the wall and the cables inside the operational set forever —
lies between two closed-form bounds: it contains {h ≥ D(x)} (the braking maneuver works from there,
Thm. 12(i)) and is contained in {h ≥ D_rel(x)} (no admissible input can beat the relaxed authority).
Sampling trajectories cannot decide this (failing to find a safe trajectory proves nothing), so this
package computes the kernel by two independent methods:

1. **Level-set (Hamilton–Jacobi) solvers** on the reduced planar models, where the state is small enough
   for a grid: the one-cable system (h, v, z, ż) in 4-D (`hj4d.py`, the core experiment E3) and the
   two-cable system (h, v, z1, ż1, z2, ż2) in 6-D (`hj6d.py`, the optional experiment E3b).
2. **Direct collocation** on the full three-dimensional N = 3 model (`collocation3d.py`, `taut_system.py`,
   experiment E3 "in three dimensions"): from a fixed initial state, search for an input trajectory that
   respects every constraint and ends in X_RF; bisect the initial distance to find the smallest feasible
   one, h_c.

Plus the validation of the level-set solver on problems with closed-form kernels (`validate.py`) and a
cross-check with an independent solver (`crosscheck_hj.py`, compromise C-7).

Statement labels (Thm. 5, Thm. 12, Prop. 17, Rem. 18, Sec. VI, …) are those of the full-length manuscript;
`authority_barriers/README.md`, Sec. 0, gives their numbers in the paper.

## Reading the results

For a fixed swing state and approach speed, the computed kernel boundary is the smallest distance h*
that is still viable. The claim holds when D_rel ≤ h* ≤ D at every probe (up to the grid cell). A probe
with h* < D_rel (a viable state that the outer bound says is not viable) would refute Thm. 12(ii); a
probe with h* > D would refute Thm. 12(i) (a state the maneuver is supposed to save that is not viable).
The collocation reports h_c, which by construction can only *over*-estimate the true boundary (the
optimizer under-approximates the kernel); a verified trajectory from h0 < D_rel − 1 cm would be a
refutation, while a solver failure is never evidence.

Outcome of the runs made for the paper: both bounds hold at every probe on every grid (4-D: 200 probes on
four grids and in the Richardson-extrapolated limit; 6-D: 120 probes on the coarse grid) and in all 24
collocation pairs; the boundary sits far from D_rel and close to D only where the authority is already
saturated (all cables at z̄: h_c/D = 0.98), i.e. the inner bound is tight when no swing is needed and the
sandwich is loose (factor 3–4 in h) when the cables must first swing. The kernel boundary converges only
at first order in the h-cell (a kink of the value function at the bang-bang boundary), which is why the
Sec. VI sentence "refined until the boundary changes by less than a stated tolerance" is amended to state
the achieved tolerance (C-7).

**Reservation.** At speeds above about 2.4 m/s the kernel boundaries of these runs lie below the elementary bound
v²/(2 a_max z̄) on the stopping distance, by up to 1.6 m (on the finest 4-D grid at 144 of 200 probes, on the 6-D
grid at 64 of 120), so they are not boundaries of the viability kernel there. In the region that the solver marks
as viable the value equals the margin of the swing rate at the edge of its axis, which points to trajectories
that leave the grid there. The statements above describe the numbers as computed; the collocation results respect
the bound, and the paper uses only those.

## Modules

### `hj4d.py` — level-set solver on the 4-D grid
- Dynamics (from `authority_barriers.theory.planar`): ḣ = −v, v̇ = −T z/m_L, ż = ż, z̈ = η/(m l) − ż² z/(1 − z²) with
  inputs (T, η) in the state-dependent convex set U(z, ż) = tension strip [T_min, m_L a_max] ∩ thrust
  ellipse. Constraint set C = {h ≥ 0, |z| ≤ z̄, ż² ≤ ω̄²(1 − z²)}.
- Method: the "freeze" form of the level-set method (Mitchell et al. 2005; ToolboxLS `termRestrictUpdate`):
  V(x, 0) = clipped signed distance to C, V_τ = min(0, max_u ∇V·f); V is nonincreasing in pseudo-time,
  {V ≥ 0} shrinks to the viability kernel. The Hamiltonian's maximum over U has a closed form through the
  support function of the ellipse-strip set (`EllipseStripInputs`, exact) or the K-point inner
  approximation (`PolygonInputs`, cross-checks).
- Numerics: WENO3 (or ENO2) one-sided derivatives, two ghost cells by linear extrapolation, local
  Lax–Friedrichs dissipation with per-node speed bounds, TVD-RK2 in pseudo-time under a CFL condition,
  convergence when max|V(τ) − V(τ − 0.5 s)| < tol; torch float64 on the CPU, optionally fused by
  `torch.compile`; checkpoint/resume every N minutes (`evolve(..., checkpoint=...)`).
- Grid: `Grid` (axes, spacing), with the h axis extended *below* the wall by a pad so that the value of a
  lost state (the penetration depth) is representable — without it the kernel comes out far too large.
- Bias: dissipation, an inner input approximation and the finite horizon all *shrink* the computed
  kernel (raise h*), so a computed h* below D_rel cannot be their artifact, while h* slightly above D is
  expected on coarse grids.
- Outputs: `KernelResult` (V on the grid, iterations, convergence flag, wall time, pseudo-time, history)
  saved as `.npz` + `.json`; `boundary_h(V, grid, v, z, ż)` reads h* at a probe by interpolation.
- `kernel_2d_double_integrator` runs the same numerics on ḣ = −v, v̇ = −b for validation.

### `hj6d.py` — the 6-D two-cable kernel (E3b, decisions D-25, C-3)
Same numerics on (h, v, z1, ż1, z2, ż2) with set A2. The two cables couple through the payload's
specific force; the Hamiltonian is linear in (T1, T2, η1, η2) with state-dependent coefficients, and its
maximum separates per cable once the input set is *inner*-approximated per cable (`TwoCableInputs`: an
ellipse-strip per cable whose centre is shifted by the other cable's tension over its range, and the D-4
cone replaced by per-cable tension caps). The kernel is therefore under-approximated (h* biased upward),
which is stated with the result. The run made for the paper used the grid 30 × 25 × 13⁴ (21.4 M nodes, 261 min on
8 threads, 5.2 GB).

### `validate.py` — closed-form checks of the solver (M4.1)
(a) the double integrator ḣ = −v, v̇ = −b, |b| ≤ a: kernel {h ≥ v₊²/(2a)}; (b) the one-cable system with
frozen cables (η = 0, ż = 0 slices): for z > 0 the kernel in the slice is {h ≥ v₊²/(2 a_max z)}, for
z < 0 it is empty at v ≥ 0. Boundary errors of 0.1–0.9 cells were measured.

### `crosscheck_hj.py` — independent solver (C-7's pre-registered mitigation)
The same viability problem (same nodes, initial level set, exact support function and dissipation bounds)
solved by the `hj_reachability` toolbox (JAX; WENO5, TVD-RK3, Lax–Friedrichs) with its
`backwards_reachable_tube` postprocessor, which is the same freeze form. Compared on the 61-grid: h*
differs by 0.012 m in median and 0.143 m (0.43 cells) at most; the sign of V differs at 0.46 % of the
nodes; both bounds hold for the toolbox's kernel too. `--validate` runs the 2-D double integrator against
the closed form first. Needs the `[crosscheck]` extra.

### `taut_system.py` — the taut-cable model as a Drake system
`TautCableSystem` (scalar-convertible `LeafSystem`, continuous state in the `taut_model` layout, one input
port [T_1..T_N, c_11, c_12, ..., c_N1, c_N2] with the Gram parametrization u_i^⊥ = c_i1 P_i y + c_i2 P_i r
of eq. "gram"), so that `pydrake.planning.DirectCollocation` can differentiate the dynamics with
`AutoDiffXd`. Conversions between (T, u^⊥), thrust vectors, plans and the input vector round-trip exactly.

### `collocation3d.py` — direct-collocation viability search (M4.4)
`find_trajectory(x0, p, ...)`: DirectCollocation with K knots (41 in the campaign) and equal time steps in
[dt_min, dt_max]; at every knot the constraints of C (h ≥ 0, cables inside the cone, swing rates ≤ ω̄) and
of U(x) (thrust norms, tension floor, D-4 cone) as normalized nonlinear rows, plus the terminal set
H(x_T) ≥ 0 and swing states in V — a subset of X_RF from which π♯ is safe forever, so a finite-horizon
trajectory is conclusive. The terminal constraint's derivative is the analytic gradient of D
(`stopping.evaluate`) at the smallest maximizing time (a valid Clarke-gradient element; ties are counted).
Initial guesses: the replay of π♯ on the taut ODE, a straight line to a saturated-rest state, or the
previous solution shifted along the wall normal. Solver SNOPT (Ipopt as fallback) with an iteration limit;
the bundled SNOPT ignores its wall-time option, so the per-solve bound is the iteration limit (C-14).
`verify_trajectory` re-simulates the taut ODE with the solution's inputs (DOP853, rtol 1e-9), measures
the path violations, checks the terminal membership and replays the braking maneuver from the endpoint
(capped at 600 s of wall time; a timed-out replay is reported as inconclusive, C-14). Per (configuration,
speed) pair the driver `experiments/e3_collocation.py` first tries a *witness* from h0 = D (the maneuver
itself must be found feasible there) and then bisects h0 in [0, D] to the smallest feasible distance h_c;
every NLP solution is cached under `results/core/e3_collocation_jobs/<config>_v<v>_k<knots>_...json`.

## Outputs at a glance

| File | Written by | Content |
|---|---|---|
| `results/core/e3_kernel/e3_kernel_<grid>.npz/.json` | `experiments/e3_kernel.py` | V on the grid; the probe table (z, ż, v, h*, D, D_rel, normalized gaps), convergence data, verdicts (M4.2 grid change, M4.3 bounds) |
| `results/core/e3_kernel/crosscheck_hj_61.json` | `crosscheck_hj.py` | probe-wise comparison of the two solvers |
| `results/core/e3_kernel_summary.md` | `e3_kernel.py --compare` | one line per grid, the convergence steps between grids, the cross-check line, the probe table |
| `results/core/e3_collocation_summary.md`, `e3_collocation_jobs/` | `e3_collocation.py` | per pair: D_rel, h_c, D, h_c/D, witness, M4.4 verdict, solves/fails, verification |
| `results/optional/e3b_kernel6d/e3b_kernel6d_<shape>.*` | `e3b_kernel6d.py` | the 6-D kernel and its probe table |

## Runtime
4-D grids: 7 min (61, 61, 21, 21), 50 min (81, 81, 31, 31), 124 min (121, 121, 31, 31; h ≤ 10 m), 268 min (121, 121, 41, 41)
on 4 threads under load; 6-D: 261 min on 8 threads; the full collocation map (24 pairs, 266 NLPs) about
ten hours on 6 workers. Kernel runs checkpoint themselves and resume from `<file>.ckpt`.
