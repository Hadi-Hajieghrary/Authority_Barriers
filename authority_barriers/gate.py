"""Gate report: lists the evidence artifacts each gate needs, prints their summaries, and runs the scope
audit of the phase-gate procedure (no optional-track module may exist before G6 PASSED, D-18).
Usage: python -m authority_barriers.gate G3          (after the experiments have been run into results/)"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
GATES = {
    "G1": ["core/G1_pytest.log"],
    "G2": ["core/G2_harness.log"],
    "G3": ["core/e1_perfect/summary.md", "core/e2_perfect_max_off0/summary.md", "core/e2_perfect_max_off0_layer1/summary.md",
           "core/e2_perfect_lmax_off0/summary.md", "core/e1_attitude/summary.md", "core/e2_attitude_max_off0/summary.md",
           "core/e2_attitude_max_off0_layer1/summary.md", "core/e1_perfect_A_N3/summary.md"],
    "G4": ["core/e3_kernel_summary.md", "core/e3_collocation_summary.md", "core/e3_kernel/crosscheck_hj_61.json"],
    "G5": ["core/e4/summary.md", "core/e5_scale1/summary.md", "core/e5_scale1_layer1/summary.md", "core/e5_scale1_attitude/summary.md",
           "core/e5_scale1_attitude_layer1/summary.md"],
    "G6": ["core/figures/index.md", "logs/reproduce_smoke_clean.log"],
    # optional track (§13): P7 practicality and the coarse 6-D kernel, P8 limits, P9 integration
    "G7": ["optional/e6_backup/summary.md", "optional/e6_distributed/summary.md", "optional/e3b_kernel6d/e3b_kernel6d_25x25x13x13x13x13_summary.md"],
    "G8": ["optional/e5_scale2_layer0.5_off0.8/summary.md", "optional/e7_stress/summary.md", "optional/e7_corridor/summary.md",
           "optional/e7_corridor/summary_gentle.md", "optional/e7_montecarlo/summary.md"],
    "G9": ["optional/figures/index.md", "core/figures/trials/index.md", "logs/reproduce_smoke_all_clean.log"],
}


OPTIONAL_TRACK = ["authority_barriers/viability/hj6d.py", "authority_barriers/experiments/e6_backup_compare.py", "authority_barriers/experiments/e7_stress.py",
                  "authority_barriers/experiments/e7_corridor.py", "authority_barriers/experiments/e7_montecarlo.py"]
OPTIONAL_SYMBOLS = ["BackupIntegratedFilter", "CableCBFBaseline"]


def scope_audit() -> int:
    """Optional-track modules must not exist before G6 PASSED (D-18); once the ledger records that verdict their presence is expected."""
    present = [f for f in OPTIONAL_TRACK if (ROOT / f).exists()]
    symbols = [s for s in OPTIONAL_SYMBOLS if s in (ROOT / "authority_barriers" / "theory" / "filters.py").read_text()]
    ledger = ROOT / "docs" / "simulation_plan.md"
    g6_passed = ledger.exists() and any(l.startswith("### Gate G6") and "PASSED" in l for l in ledger.read_text().splitlines())
    print(f"\nscope audit (D-18): optional-track modules present: {present or 'none'}; optional filter classes: {symbols or 'none'}"
          f"{'; G6 PASSED is recorded, so the optional track may exist' if g6_passed else ''}")
    return 0 if g6_passed else len(present) + len(symbols)


def main(gate: str) -> int:
    missing = 0
    for rel in GATES[gate]:
        f = RES / rel
        print(f"\n=== {rel} ===")
        if not f.exists():
            print("MISSING"); missing += 1; continue
        txt = f.read_text()
        print(txt if len(txt) < 6000 else txt[-6000:])
    extra = scope_audit() if gate == "G6" else 0
    print(f"\n{gate}: {'all evidence present' if not missing else f'{missing} artifact(s) missing'}"
          f"{'; scope audit clean' if gate == 'G6' and not extra else ('; SCOPE VIOLATION' if extra else '')}")
    return 1 if (missing or extra) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "G1"))
