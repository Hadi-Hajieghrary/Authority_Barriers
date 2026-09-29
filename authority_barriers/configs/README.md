# `authority_barriers/configs/` — parameter sets

The constants of the campaign. Each `params_<name>.yaml` lists only the *free* constants; everything
the assumptions of the paper constrain is derived from them by `authority_barriers.theory.params.derive_set` with a
5 % margin, and the result is written to `derived/params_<name>.json` by

```
python -m authority_barriers.theory.params A A1 A2 A_N3 A_mL12     # regenerates derived/*.json and prints the assumption margins
```

`authority_barriers.theory.params.load_set("<name>")` derives the set on the fly (the JSON files are the human-readable
record that the figures' sidecars cite).

Statement labels (Thm. 5, Thm. 12, Prop. 17, Rem. 18, Sec. VI, …) are those of the full-length manuscript;
`authority_barriers/README.md`, Sec. 0, gives their numbers in the paper.

## Free constants (YAML)
N (team size), m_L (payload mass), m (quadrotor mass), l (cable length), T_min (tension floor), θ_q
(cable cone half-angle), z̄ and w̄ as fractions of sin θ_q, ν and ν_w (swing acceleration bounds),
a floor for a_max, ρ_fb (feedback reserve inside the swing reserve ρ), the margins, κ_H (γ(H) = κ_H H),
and the disturbance budget d̄_L, d̄_i (decision D-16: the attitude error ≤ 0.15 N, wind ≤ 0.1 N, the
â lag ≤ 0.05 N must fit in d̄_i = 0.3 N, and ν_rob = ν − d̄_i/(m l) − d̄_L/(m_L l) must stay positive).

## Derived constants (in order)
1. **T̄** (tension cap, Assumption 7): the tension needed to hover the payload at the maximum cable tilt
   with a 20 % hover margin, at least T_min + 1 N, rounded up to 0.1 N.
2. **a_max** (bound on the payload specific force, decision D-4): the configured floor, but at least
   N T̄/m_L (so that all cables at T̄ in parallel remain admissible) and 1.15 g; enforced in every filter
   and input set as the second-order cone ‖Σ T_j q_j‖ ≤ m_L a_max.
3. **ω̄** (swing-rate bound) from Assumption 14 with margin.
4. **ρ** (swing reserve) from Assumption 13 with margin: the perpendicular thrust needed for the swing
   accelerations ν, ν_w at the maximum specific force and swing rate, plus ρ_fb.
5. **f_max** from Assumption 7 (magnitude): √((T̄ + m a_max + m l ω̄²)² + ρ²) with margin.

The derived JSON also lists T̄_rel (the relaxed caps of the outer bound D_rel), ν_rel, the saturated
authorities α_sat and α_sat,rel, the lowest braking authority a_low and the thrust-to-weight ratio.
`check_assumptions(p)` prints every inequality with its margin; `all_ok(p)` is what the stress
experiment uses to say at which f_max Assumption 7 fails.

## The sets

| Set | Use | Key values |
|---|---|---|
| `A` | every closed-loop Drake trial (E1, E2, E4, E5, E6, E7) | N = 4, m_L = 1 kg, m_i = 1.5 kg, l_i = 1 m, T_min = 1 N, θ_q = 20°, z̄ = 0.239, w̄ = 0.171, ν = ν_w = 0.5 s⁻², derived T̄ = 3.2 N, ρ = 34 N, f_max = 44 N, a_max = 13 m/s², ω̄ = 1 s⁻¹; κ_H = 1, filter period 5 ms, attitude period 1 ms; ball thrust set (θ_max = π, decision D-5) |
| `A1` | the planar one-cable kernel (E3) | one cable, same per-cable constants |
| `A2` | the planar two-cable kernel (E3b) | two cables |
| `A_N3` | the N = 3 replication of E1 (D-9) and the 3-D collocation | m_L scaled to 0.75 kg so that the per-cable constants are unchanged |
| `A_mL12` | the mass rule of E7 (D-26): the filter designed for the heaviest payload of the ±20 % interval | m_L = 1.2 kg; f_max kept at 44 N by an override (the derived 44.5 N would break the thrust-to-weight bound), Assumption 7 still holds with a 4.4 % margin |

Why the ball thrust set: with hover-consistent constants the tilt sufficient condition of the paper's
Sec. IV-A cannot hold for any θ_max ≤ 90° (the reserve ρ ≈ 34 N dwarfs the parallel component floor),
so the simulations use θ_max = π.
