"""Constants of the campaign and the assumption checker (Assumptions 7, 13, 14 of the paper).

Sets: A (N = 4 team, every Drake trial), A1 / A2 (planar kernel sets, N = 1 / 2), and the
paper's Sec. I example (E0 only). Ball thrust set theta_max = pi (D-5); a_max is an enforced
input constraint (D-4). Notation follows the paper.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

E3 = np.array([0.0, 0.0, 1.0])
CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@dataclass(frozen=True)
class Params:
    name: str
    N: int
    m_L: float            # payload mass [kg]
    m: tuple              # quadrotor masses [kg]
    l: tuple              # cable lengths [m]
    f_max: tuple          # thrust magnitude limits [N]
    T_min: float          # tension floor [N]
    T_bar: tuple          # tension caps [N] (Assumption 7)
    rho: tuple            # swing reserve [N] (Assumption 7)
    rho_fb: tuple         # feedback reserve inside rho [N] (Assumption 13, Rem. 18)
    nu: float             # swing acceleration bound for z [1/s^2] (Assumption 13)
    nu_w: float           # same for w
    a_max: float          # bound on the payload specific force [m/s^2] (D-4)
    theta_q: float        # cable cone half-angle [rad] (eq. Xop)
    z_bar: float
    w_bar: float
    omega_bar: float      # swing-rate bound [rad/s]
    theta_max: float = math.pi   # tilt limit; pi means the ball (D-5)
    g: float = 9.81
    kappa_H: float = 1.0         # gamma(H) = kappa_H * H (D-11)
    dt_filter: float = 5e-3
    dt_attitude: float = 1e-3
    d_bar_L: float = 0.1         # D-16 (revised 2026-09-24): nu - d_bar_i/(m l) - d_bar_L/(m_L l) must stay > 0
    d_bar_i: float = 0.3
    n: tuple = (1.0, 0.0, 0.0)   # wall normal, horizontal unit vector (eq. h)
    d0: float = 0.0              # wall plane n^T x = d0
    # ---- altitude barriers and swing deceleration (sets with a hover floor; None = not used)
    T_h: float | None = None     # hover floor of the tensions of the maneuver [N]: N T_h cos(theta_q) >= m_L g
    nu_dec: float | None = None  # deceleration of the swing [1/s^2], < nu
    alt_min: float | None = None # altitude band of the payload [m]
    alt_max: float | None = None
    c_dn: float | None = None    # vertical deceleration of the altitude hold while the payload climbs [m/s^2]
    c_up: float | None = None    # vertical acceleration of the altitude hold while the payload descends [m/s^2]
    kappa_alt: float = 1.0       # gamma(H) = kappa_alt * H for the altitude barriers
    hold_lead: float = 0.1       # the altitude hold begins this long before the last swing arrives [s]

    # ---- wall geometry
    @property
    def n_vec(self) -> np.ndarray:
        return np.asarray(self.n, float)

    @property
    def y_vec(self) -> np.ndarray:          # braking direction
        return -self.n_vec

    @property
    def r_vec(self) -> np.ndarray:          # transverse horizontal direction r = e3 x y
        return np.cross(E3, self.y_vec)

    # ---- per-cable arrays
    @property
    def m_arr(self): return np.asarray(self.m, float)
    @property
    def l_arr(self): return np.asarray(self.l, float)
    @property
    def f_max_arr(self): return np.asarray(self.f_max, float)
    @property
    def T_bar_arr(self): return np.asarray(self.T_bar, float)
    @property
    def rho_arr(self): return np.asarray(self.rho, float)
    @property
    def rho_fb_arr(self): return np.asarray(self.rho_fb, float)

    # ---- derived data of Prop. 15
    @property
    def T_bar_rel(self) -> np.ndarray:
        return self.f_max_arr + self.m_arr * self.a_max + self.m_arr * self.l_arr * self.omega_bar ** 2

    @property
    def nu_rel(self) -> float:
        return float(np.max(self.f_max_arr / (self.m_arr * self.l_arr) + self.a_max / self.l_arr + self.omega_bar ** 2))

    @property
    def alpha_sat(self) -> float:           # authority with every cable at (z_bar, 0)
        return float(np.sum(self.T_bar_arr) * self.z_bar / self.m_L)

    @property
    def alpha_sat_rel(self) -> float:
        return float(np.sum(self.T_bar_rel) * self.z_bar / self.m_L)

    # ---- vertical acceleration of the payload during the braking phase of the maneuver (tensions in [T_h, T_bar])
    @property
    def c_low(self) -> float:               # smallest: every cable at the floor and at the edge of the cone
        return float(self.N * self.T_h * math.cos(self.theta_q) / self.m_L - self.g)

    @property
    def c_high(self) -> float:              # largest: every cable at the cap and vertical
        return float(np.sum(self.T_bar_arr) / self.m_L - self.g)

    @property
    def a_low(self) -> float:               # lower bound on q_i^T a used by the tilt condition
        return self.N * self.T_min * math.cos(2 * self.theta_q) / self.m_L

    @property
    def twr(self) -> np.ndarray:            # thrust-to-weight of each quadrotor
        return self.f_max_arr / (self.m_arr * self.g)

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}

    def to_json(self, path: Path) -> None:
        d = self.to_dict()
        d["derived"] = {
            "T_bar_rel": self.T_bar_rel.tolist(), "nu_rel": self.nu_rel,
            "alpha_sat": self.alpha_sat, "alpha_sat_rel": self.alpha_sat_rel,
            "a_low": self.a_low, "twr": self.twr.tolist(),
        }
        Path(path).write_text(json.dumps(d, indent=2))

    @staticmethod
    def from_dict(d: dict) -> "Params":
        d = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k != "derived"}
        return Params(**d)


def _percable(x, N):
    return tuple(float(v) for v in (np.full(N, float(x)) if np.isscalar(x) else np.asarray(x, float)))


def derive_set(cfg: dict) -> Params:
    """Fix T_bar, a_max, omega_bar, rho, f_max from the free constants so that Assumptions 7
    (magnitude), 13, 14, the D-4 consistency of the braking maneuver, and hover consistency
    hold with relative margin cfg['margin'] (measured as (rhs - lhs)/rhs). Order: T_bar -> a_max -> omega_bar -> rho -> f_max."""
    N = int(cfg["N"]); m_L = float(cfg["m_L"]); g = float(cfg.get("g", 9.81))
    m = np.asarray(_percable(cfg["m"], N)); l = np.asarray(_percable(cfg["l"], N))
    T_min = float(cfg["T_min"]); thq = math.radians(float(cfg["theta_q_deg"]))
    z_bar = float(cfg["z_bar_frac"]) * math.sin(thq); w_bar = float(cfg["w_bar_frac"]) * math.sin(thq)
    nu = float(cfg["nu"]); nu_w = float(cfg.get("nu_w", nu)); rho_fb = float(cfg.get("rho_fb", 1.0))
    d = float(cfg.get("margin", 0.05)); hov = float(cfg.get("hover_margin", 1.2))
    # tension cap: lift the payload at the maximum cable tilt with margin; at least T_min + 1 N
    T_bar = max(T_min + 1.0, hov * m_L * g / (N * math.cos(thq)))
    T_bar = math.ceil(T_bar * 10) / 10
    # a_max: configured floor, D-4 consistency (all cables at T_bar and parallel stay admissible), hover
    a_max = max(float(cfg.get("a_max", 0.0)), N * T_bar / m_L, 1.15 * g)
    a_max = math.ceil(a_max * 2) / 2
    # Assumption 14 with margin
    om2 = (4 * nu * z_bar + 4 * nu_w * w_bar) / math.cos(thq) ** 2 / (1 - d)
    omega_bar = math.ceil(math.sqrt(om2) * 20) / 20
    # Assumption 13 with margin
    eta_z = m * (l * nu + a_max + l * omega_bar ** 2 * z_bar)
    eta_w = m * (l * nu_w + a_max + l * omega_bar ** 2 * w_bar)
    rho = (np.hypot(eta_z, eta_w) / math.cos(thq) + rho_fb) / (1 - d)
    rho = np.ceil(rho * 2) / 2
    # Assumption 7 (magnitude) with margin
    f_max = np.hypot(T_bar + m * a_max + m * l * omega_bar ** 2, rho) / (1 - d)
    f_max = np.ceil(f_max * 2) / 2
    return Params(
        name=str(cfg.get("name", "set")), N=N, m_L=m_L, m=tuple(m.tolist()), l=tuple(l.tolist()),
        f_max=tuple(f_max.tolist()), T_min=T_min, T_bar=tuple([T_bar] * N), rho=tuple(rho.tolist()),
        rho_fb=tuple([rho_fb] * N), nu=nu, nu_w=nu_w, a_max=a_max, theta_q=thq, z_bar=z_bar, w_bar=w_bar,
        omega_bar=omega_bar, theta_max=math.radians(float(cfg["theta_max_deg"])) if "theta_max_deg" in cfg else math.pi,
        g=g, kappa_H=float(cfg.get("kappa_H", 1.0)), d_bar_L=float(cfg.get("d_bar_L", 0.1)),
        d_bar_i=float(cfg.get("d_bar_i", 0.3)),
        **_altitude(cfg, N, m_L, g, thq, nu, T_min, T_bar),
    )


def _altitude(cfg: dict, N: int, m_L: float, g: float, thq: float, nu: float, T_min: float, T_bar: float) -> dict:
    """Constants of the altitude barriers, present when the configuration has an altitude band: the hover floor
    (the hover share at the edge of the cone, rounded up to 0.1 N), the swing deceleration, and the vertical
    accelerations of the altitude hold, which the uniform tension m_L (g + c) / sum_i varrho_i realizes inside
    [T_min, T_bar] for every cable configuration of the cone."""
    if "alt_band" not in cfg:
        return {}
    T_h = math.ceil(float(cfg.get("hover_floor", 1.0)) * m_L * g / (N * math.cos(thq)) * 10) / 10
    c_dn = float(cfg.get("c_dn", math.floor((g - N * T_min / m_L) * 2) / 2 - 0.5))
    c_up = float(cfg.get("c_up", math.floor((N * T_bar * math.cos(thq) / m_L - g) * 2) / 2))
    lo, hi = cfg["alt_band"]
    return {"T_h": T_h, "nu_dec": float(cfg.get("nu_dec_frac", 0.8)) * nu, "alt_min": float(lo), "alt_max": float(hi),
            "c_dn": c_dn, "c_up": c_up, "kappa_alt": float(cfg.get("kappa_alt", 1.0))}


def load_set(name: str) -> Params:
    cfg = yaml.safe_load((CONFIG_DIR / f"params_{name}.yaml").read_text())
    return derive_set(cfg)


def paper_example() -> Params:
    """Constants of the Sec. I example (E0). Not an assumption-consistent set: T_bar, rho, nu,
    a_max are placeholders that do not enter the example's quantities (a_max is non-binding)."""
    return Params(
        name="paper_example", N=2, m_L=1.0, m=(1.0, 1.0), l=(1.0, 1.0), f_max=(20.0, 20.0), T_min=1.0,
        T_bar=(5.0, 5.0), rho=(5.0, 5.0), rho_fb=(0.0, 0.0), nu=1.0, nu_w=1.0, a_max=50.0,
        theta_q=math.radians(45.0), z_bar=0.85 * math.sin(math.radians(45.0)),
        w_bar=0.5 * math.sin(math.radians(45.0)), omega_bar=2.0,
    )


# ---------------------------------------------------------------- assumption checker
@dataclass
class Margin:
    name: str
    lhs: float
    rhs: float
    ok: bool
    margin: float   # relative slack (rhs - lhs) / scale


def check_assumptions(p: Params, min_margin: float = 0.05) -> dict:
    """Every inequality the theory needs, as lhs <= rhs, with its relative margin.
    Assumption 7's tilt condition is reported only when theta_max < pi (D-5 makes it vacuous)."""
    out: dict[str, Margin] = {}

    def add(name, lhs, rhs, scale=None):
        scale = abs(rhs) if scale is None else scale
        out[name] = Margin(name, float(lhs), float(rhs), bool(lhs <= rhs),
                           float((rhs - lhs) / scale) if scale > 0 else float("nan"))

    sec = 1.0 / math.cos(p.theta_q)
    add("Xop: z_bar^2 + w_bar^2 <= sin^2(theta_q)", p.z_bar ** 2 + p.w_bar ** 2, math.sin(p.theta_q) ** 2)
    add("A14: sec^2(theta_q)(4 nu z_bar + 4 nu_w w_bar) <= omega_bar^2",
        sec ** 2 * (4 * p.nu * p.z_bar + 4 * p.nu_w * p.w_bar), p.omega_bar ** 2)
    for i in range(p.N):
        m, l, fm, Tb, rho, rfb = p.m_arr[i], p.l_arr[i], p.f_max_arr[i], p.T_bar_arr[i], p.rho_arr[i], p.rho_fb_arr[i]
        eta_z = m * (l * p.nu + p.a_max + l * p.omega_bar ** 2 * p.z_bar)
        eta_w = m * (l * p.nu_w + p.a_max + l * p.omega_bar ** 2 * p.w_bar)
        add(f"A13[{i}]: sec(theta_q) sqrt(eta_z^2 + eta_w^2) + rho_fb <= rho", sec * math.hypot(eta_z, eta_w) + rfb, rho)
        add(f"A7mag[{i}]: sqrt((T_bar + m a_max + m l omega_bar^2)^2 + rho^2) <= f_max",
            math.hypot(Tb + m * p.a_max + m * l * p.omega_bar ** 2, rho), fm)
        if p.theta_max < math.pi - 1e-9:
            s_low = p.T_min + m * (p.a_low - l * p.omega_bar ** 2)
            if s_low <= 0:
                add(f"A7tilt[{i}]: s_low > 0 (violated: s_low = {s_low:.3f} N)", 1.0, 0.0, scale=1.0)
            else:
                add(f"A7tilt[{i}]: theta_q + atan(rho/s_low) <= theta_max", p.theta_q + math.atan(rho / s_low), p.theta_max)
        add(f"caps[{i}]: T_min <= T_bar", p.T_min, Tb)
    add("D-4 consistency: sum_i T_bar_i <= m_L a_max (maneuver tensions admissible)", float(np.sum(p.T_bar_arr)), p.m_L * p.a_max)
    add("hover: T_min <= m_L g / N", p.T_min, p.m_L * p.g / p.N)
    add("hover: 1.2 m_L g <= N min_i(T_bar_i) cos(theta_q)", 1.2 * p.m_L * p.g, p.N * float(np.min(p.T_bar_arr)) * math.cos(p.theta_q))
    add("hover: 1.15 g <= a_max", 1.15 * p.g, p.a_max)
    add("alpha_sat > 0", 0.0, p.alpha_sat)
    if p.T_h is not None:
        add("floor: m_L g <= N T_h cos(theta_q) (no descent while braking)", p.m_L * p.g, p.N * p.T_h * math.cos(p.theta_q))
        add("floor: T_min <= T_h", p.T_min, p.T_h)
        add("floor: T_h <= min_i T_bar_i", p.T_h, float(np.min(p.T_bar_arr)))
        add("swing: nu_dec < nu", p.nu_dec, p.nu)
        add("hold: T_min <= m_L (g - c_dn) / N", p.T_min, p.m_L * (p.g - p.c_dn) / p.N)
        add("hold: m_L (g + c_up) / (N cos(theta_q)) <= min_i T_bar_i", p.m_L * (p.g + p.c_up) / (p.N * math.cos(p.theta_q)), float(np.min(p.T_bar_arr)))
        add("band: alt_min < alt_max", p.alt_min, p.alt_max)
    add("thrust-to-weight <= 3", float(np.max(p.twr)), 3.0)
    return out


def all_ok(margins: dict, min_margin: float = 0.05,
           ignore_prefixes=("thrust-to-weight", "alpha_sat", "caps", "Xop", "hover", "D-4", "floor", "swing", "hold", "band")) -> bool:
    """True when every inequality holds and every assumption row (A7/A13/A14) has margin >= min_margin."""
    for k, mg in margins.items():
        if not mg.ok:
            return False
        if not k.startswith(ignore_prefixes) and mg.margin < min_margin:
            return False
    return True


def format_margins(margins: dict) -> str:
    w = max(len(k) for k in margins)
    lines = [f"{'condition':{w}s} {'lhs':>10s} {'rhs':>10s} {'margin':>8s}  ok"]
    for k, mg in margins.items():
        lines.append(f"{k:{w}s} {mg.lhs:10.4f} {mg.rhs:10.4f} {mg.margin:8.3f}  {'PASS' if mg.ok else 'FAIL'}")
    return "\n".join(lines)


def summary(p: Params) -> str:
    return (f"set {p.name}: N={p.N} m_L={p.m_L} m={p.m[0]} l={p.l[0]} T_min={p.T_min} T_bar={p.T_bar[0]} "
            f"a_max={p.a_max} theta_q={math.degrees(p.theta_q):.1f}deg z_bar={p.z_bar:.4f} w_bar={p.w_bar:.4f} "
            f"nu={p.nu} nu_w={p.nu_w} omega_bar={p.omega_bar} rho={p.rho[0]} rho_fb={p.rho_fb[0]} f_max={p.f_max[0]} "
            f"(twr={p.twr[0]:.2f}) alpha_sat={p.alpha_sat:.3f} alpha_sat_rel={p.alpha_sat_rel:.1f} "
            f"T_bar_rel={p.T_bar_rel[0]:.1f} nu_rel={p.nu_rel:.1f}")


if __name__ == "__main__":
    import sys
    names = sys.argv[1:] or ["A", "A1", "A2"]
    ok_all = True
    for nm in names:
        p = load_set(nm)
        mg = check_assumptions(p)
        ok = all_ok(mg)
        ok_all &= ok
        print(summary(p)); print(format_margins(mg)); print(f"==> {nm}: {'ALL HOLD (margins >= 5 %)' if ok else 'FAIL'}\n")
        (CONFIG_DIR / "derived").mkdir(exist_ok=True)
        p.to_json(CONFIG_DIR / "derived" / f"params_{nm}.json")
    sys.exit(0 if ok_all else 1)
