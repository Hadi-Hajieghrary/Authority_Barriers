# `authority_barriers/` — the Python package

The package holds the theory of the paper as a numerical library, a full-order Drake simulator of the team
with its cable-suspended payload, the solvers for the viability kernel, and the experiments with their figure
generators. The experiments were designed to *falsify* the theorems, not to illustrate them: every experiment
states in advance what observation would refute the claim it tests, and every violation that occurred was
attributed to a cause before it was reported.

Each subfolder has its own `README.md` with the details of that layer; this file gives the picture that
ties them together.

## 0. Labels used in the code

Comments, docstrings and the result files that the experiments write cite the statements by the numbers of the
full-length manuscript on which the simulation study was planned. The paper in `IEEE_ACC2027/` numbers them as
follows:

| In the code | In the paper | | In the code | In the paper |
|---|---|---|---|---|
| Lemma 1 | Lemma 1 | | Thm. 12 | Theorem 9 |
| Prop. 3 | Eq. (7) | | Assumption 13 | Assumption 10 |
| Def. 4 | Definition 2 | | Assumption 14 | Assumption 11 |
| Thm. 5 | Theorem 3 | | Prop. 15 | Proposition 12 |
| Assumption 7 | Assumption 4 | | Rem. 16 | Sec. IV-C, after Theorem 9 |
| Def. 8 | Definition 5 | | Prop. 17 | Proposition 13 |
| Lemma 9 | Lemma 6 | | Rem. 18 | not in the paper |
| Def. 10 | Definition 7 | | Rem. 19 | Remark 14, a different text |
| Lemma 11 | Lemma 8 | | filter (17) | Eq. (17) |

"Sec. VI" and "Table I" in the code are the simulation protocol and the theorem-to-experiment matrix of the
full-length manuscript.

## 1. The problem in one page

**System.** N quadrotors (mass m_i) carry one point-like payload (mass m_L) through cables of length l_i.
The state used by the theory is x = (x_L, ẋ_L, q_1..q_N, q̇_1..q_N): payload position and velocity plus
the unit cable directions q_i (from the payload toward quadrotor i) and their rates. Quadrotor i applies a
thrust vector u_i with ‖u_i‖ ≤ f_max,i. While the cables are taut the payload's acceleration is set by the
cable tensions T_i (eq. "load" of the paper: m_L ẍ_L = Σ T_i q_i − m_L g e3), and the tensions plus the
part of each thrust perpendicular to its cable determine how the cables swing (Lemma 1 of the paper,
`theory/taut_model.py`).

**Wall coordinates.** A vertical wall with horizontal normal n stands at distance
h = d0 − nᵀx_L from the payload; v = nᵀẋ_L is the approach speed (positive toward the wall);
y = −n is the braking direction and r = e3 × y the transverse horizontal direction. Each cable has the
swing coordinates z_i = q_iᵀy (how far the cable leans toward/away from the wall; z_i > 0 means the
quadrotor is *behind* the payload and can brake it) and w_i = q_iᵀr, with rates ż_i, ẇ_i.

**Operational set X_op.** |z_i| ≤ z̄, |w_i| ≤ w̄ (cables inside a cone of half-angle θ_q) and swing rates
‖q̇_i‖ ≤ ω̄.

**Authority.** The best braking acceleration the team can give the payload with the cables at their
current lean is α_y(z) = (1/m_L) Σ_i φ_i(z_i), φ_i(z) = T̄_i z₊ − T_min z₋ (Def. 8 / Lemma 9): a cable
leaning backward (z_i > 0) brakes with the tension cap T̄_i, a cable leaning forward can do no better
than the tension floor T_min. Building authority takes time because the cables must first be swung to
z_i = z̄, which is limited by the swing acceleration bound ν (Assumption 13).

**Braking maneuver π♯ and stopping distance D.** From any state, swing every cable time-optimally to z̄
(bang-bang with acceleration ±ν, `theory/profile.py`), pull with T̄_i as soon as z_i > 0, hold the
transverse swing inside its admissible set, and keep the whole thing inside the thrust limits
(Prop. 15, `theory/maneuver.py`). Integrating the resulting braking acceleration gives the predicted
speed ṽ(t) and the distance E(t; x) = ∫₀ᵗ ṽ; the **stopping distance** is D(x) = max_t E(t; x)
(`theory/stopping.py`), and the **authority barrier** is H(x) = h − D(x).

**Sets and claims tested.**
- X_RF = {H ≥ 0, swing states inside V(ν, z̄) × V(ν_w, w̄)} is the set from which π♯ reaches a stop
  before the wall. **Thm. 12(i)**: the filter (17) keeps X_RF invariant and stays feasible on it.
- V(ν, z̄) is the set of swing states (ζ, ζ̇) that can be brought to rest inside [−z̄, z̄] with
  acceleration ≤ ν (a parabola-bounded region, eq. "box").
- **Thm. 5**: a high-order CBF (HOCBF) filter on h alone, started from the set D of states whose cables
  all lean toward the wall (Σ_y) at speed, *loses feasibility* before the time τ at which a cable could
  leave Σ_y — braking authority cannot be created fast enough.
- **Thm. 12(ii)**: the true viability kernel of the constraint set lies between two closed-form bounds,
  h = D_rel(x) (relaxed authority: tension caps from the full thrust, the largest admissible swing
  acceleration ν_rel) and h = D(x).
- **Prop. 17**: Ḣ splits into per-vehicle terms once a common commitment â to the payload's specific
  force is broadcast (distributed implementation), and the barrier data costs O(N log N).
- **Rem. 18**: with the barrier data shrunk by the disturbance budget (d̄_L, d̄_i), H_rob stays ≥ 0
  under wind bounded by d̄. **Rem. 19**: the sampled (200 Hz) implementation and the inner attitude loop
  cost a margin.

**The filter (17).** At every tick (5 ms): minimize Σ‖u_i − u_i^nom‖² over the admissible inputs U(x)
(thrust balls, tension floor, and the payload specific-force bound a_max, decision D-4) subject to
Ḣ_j ≥ −κ_H H for every maximizing time t_j of E (one linear row each, Prop. 17) and to the swing-set
rows (Remark 19's one-tick membership in V, written as rotated second-order cones). It is a small SOCP
solved by Clarabel in 2–5 ms (`theory/filters.py`, `theory/socp.py`).

## 2. Layout

| Folder | Contents | README |
|---|---|---|
| `theory/` | The theory as code: taut-cable model, swing profile, stopping distance and its gradients, the SOCP filters (proposed, HOCBF, backup, distributed, cable-CBF, two-wall), samplers, parameter sets and the assumption checker. Pure numpy/scipy + Clarabel; no Drake plant. | [theory/README.md](theory/README.md) |
| `simulator/` | The full-order simulator: Drake `MultibodyPlant` with rigid quadrotors and payload, unilateral spring-damper cables, thrust/torque actuation, 1 kHz geometric attitude loop, bounded wind, the 200 Hz filter as a discrete system, loggers, the trial harness and the Meshcat renderer. | [simulator/README.md](simulator/README.md) |
| `viability/` | Viability-kernel computations for Thm. 12(ii): the level-set (Hamilton–Jacobi) solver on the 4-D planar one-cable system and the 6-D two-cable system, an independent cross-check, closed-form validation cases, and the 3-D direct-collocation search. | [viability/README.md](viability/README.md) |
| `experiments/` | One driver per experiment (E0–E7), the shared helpers (trial cache, attribution procedure, sampled-data accounting) and the figure generators (core, optional, trial-level). | [experiments/README.md](experiments/README.md) |
| `configs/` | Parameter sets (YAML) and their derived constants (JSON). | [configs/README.md](configs/README.md) |
| `reproduce.py` | Every experiment in one command (`--core`, `--all`, `--smoke`, `--results`, `--overwrite`). | — |
| `gate.py` | Lists the result files that a gate of the simulation study needs, prints their summaries and runs the scope audit. | — |

Outside the package:

| Folder | Contents | README |
|---|---|---|
| `scripts/` | Build script of the paper and the sequences in which the experiments were launched. | [../scripts/README.md](../scripts/README.md) |
| `IEEE_ACC2027/` | The manuscript and its figure files. | [../IEEE_ACC2027/README.md](../IEEE_ACC2027/README.md) |

## 3. How the pieces fit

```
configs/params_*.yaml ──derive_set──▶ Params (constants + Assumptions 7/13/14 checked)
        │
        ▼
theory/     State ─▶ stopping.evaluate ─▶ D, T*, ∇E ─▶ filters.ProposedFilter.solve ─▶ thrust commands u
        │                                                     ▲
        │                                                     │ u_nom from simulator/nominal.py
        ▼                                                     │
simulator/harness.run_trial: plant + cables + (attitude loop) + wind + FilterSystem(200 Hz) ─▶ TrialLog (.npz + .json)
        │
        ▼
experiments/e*.py: sample initial states, run trials (cached, parallel), compute per-trial verdicts,
                   attribute violations, write results/<track>/<experiment>/{summary.md, results.json}
        │
        ▼
experiments/make_figures*.py, trial_figures.py, sequence_figures.py, paper_figures.py
        │                          ─▶ results/*/figures/ (one plot or frame per file, JSON sidecars)
        ▼
IEEE_ACC2027/collect_figures.py ─▶ IEEE_ACC2027/figures/
```
`results/` is the default results directory; it is written by the experiments and is not part of the repository.
The gates G0 to G9, the decisions D-x and the compromises C-x that the comments cite belong to the plan of the
simulation study; Sec. 4 explains the terms.

The kernel computations (`viability/`) run beside this pipeline on the reduced planar models and feed
`experiments/e3_kernel.py`, `e3b_kernel6d.py` and `e3_collocation.py`.

## 4. Vocabulary used everywhere

- **Necessary track (P0–P6) / optional track (P7–P9)**: the plan of the simulation study splits
  the work into phases with a **gate** (G0–G9) each; a gate lists metrics **M#.#** with targets, the
  measured values and a verdict (PASS / CONDITIONAL / FAIL / reported).
- **D-#**: a design decision recorded in the plan (e.g. D-4: a_max is an enforced input constraint; D-5:
  ball thrust set; D-19: every closed-loop experiment in both actuator modes; D-24: the backup filter is
  not run in closed loop).
- **C-#**: a compromise in the compromise ledger (e.g. C-4: sampled-data margin; C-7: first-order
  convergence of the kernel boundary; C-10: the 1 kHz attitude loop exceeds the disturbance budget; C-12:
  wall-time caps on stalled trials; C-14: SNOPT ignores its time limit).
- **Actuator modes**: `perfect` = the paper's Sec. II model (thrust *vectors* applied directly, no
  attitude dynamics); `attitude` = the paper's Sec. VI setup (a 1 kHz geometric attitude controller turns
  the commanded vector into thrust magnitude and body torque). The theorems are stated for the first;
  the second measures the inner-loop gap.
- **Attribution (M3.4)**: a trial that touches the wall or loses a barrier condition is classified as a
  *sampled-data case* (the loss happened between two 5 ms ticks with the filter feasible; the contact depth
  is within the accumulated deficit that the continuous-time condition does not see), a *model gap* (slack
  cable or attitude-tracking error beyond the budget, or the exact taut-cable replay of the logged commands
  stays safe), or a *theory-class candidate* (the replay also violates — to be escalated). No theory-class
  candidate occurred in the campaign.
- **Truncated**: a trial stopped by a wall-clock cap (C-12) is reported and counted neither as a pass nor
  as a violation.
- **Expected / refuting observation**: every summary and figure caption states both, as fixed before the runs.

## 5. Commands

```
pip install -e .                             # in the Drake dev container (pydrake 1.51.1)
python -m authority_barriers.reproduce --core --results results_reproduced   # necessary track from fixed seeds
python -m authority_barriers.reproduce --all  --results results_reproduced   # plus the optional track
python -m authority_barriers.reproduce --all --smoke        # reduced version for an end-to-end check (about 15 min; writes results_smoke)
python -m authority_barriers.gate G8                        # print the evidence of a gate; scope audit
python -m authority_barriers.experiments.make_figures --core
python -m authority_barriers.experiments.make_figures_optional
python -m authority_barriers.experiments.trial_figures      # Meshcat snapshots and one plot per signal (pip install -e .[viz]; playwright install chromium)
python -m authority_barriers.experiments.sequence_figures   # fixed-camera sequences; after trial_figures
python -m authority_barriers.experiments.paper_figures      # the figures of the paper
```

Every driver is resumable: finished trials, collocation jobs and kernel integration states are cached, so
re-running a command after an interruption continues where it stopped. `reproduce` stops instead of writing into
a results directory that already holds summaries, unless `--results` or `--overwrite` is given.

## 6. Findings of the runs made for the paper

The paper reports E1, E2, the collocation part of E3 and the run times of E4 in its Sec. VI. Every experiment
writes a `summary.md` with its verdict and its tables.

- In short: the loss of feasibility (Thm. 5) held in every one of 690 trials; the positive results
  (Thm. 12(i)/(ii), Prop. 17, Rem. 18) were not contradicted in continuous time; every wall contact of
  the proposed filter was attributed to the sampled implementation (the largest initial margin with a contact
  is 0.78 m at a period of 5 ms, and larger under wind), to the 1 kHz inner loop (tracking errors two orders
  above the budget at the filter's command jumps), or to a hypothesis violated by a stress factor. A broadcast
  of the specific force that lags by one period is not a commitment: the distributed filter reached the wall
  in 10 of 10 trials. In the corridor the filter left the nominal thrust unchanged (the transport nominal is
  the one of compromise C-15). In the Monte Carlo study the HOCBF filters were infeasible in every trial and
  reached the wall in at least 498 of 500; the proposed filter reached it in 152 of 500, each time in a trial
  that violates a hypothesis; with independently drawn stress factors the study ranks the filters and
  certifies little on its own.
