"""Collect the figure files of the paper from the results of the experiments.

Every file under ``IEEE_ACC2027/figures/`` is produced from one file under ``results/`` and is named after the
figure it belongs to (``\\label{fig:<label>}`` in the LaTeX source). ``FIGURES`` is the
complete list: destination file, figure label, source file, the module that generates the source, and the
operation that turns the source into the destination (``copy``: byte-identical; ``crop``: the columns and
rows of the source listed in ``CROPS``, that is, the plot without its white margins; ``trim``: the rendered
frame with its outer columns and a band of rows cut away).

    python IEEE_ACC2027/collect_figures.py            # write IEEE_ACC2027/figures/ from results/ (files that match are left alone)
    python IEEE_ACC2027/collect_figures.py --check    # verify IEEE_ACC2027/figures/ against results/ (exit status 1 on a mismatch)

The results root is ``results/`` next to this folder, or ``$SIM_RESULTS_DIR`` if that variable is set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
REPO = PAPER.parent
FIG = PAPER / "figures"

PAPER_FIGS = "core/figures/paper"
TRIAL = "core/figures/trials/e2_perfect_max_off0_layer1_75"
COLLOCATION = "core/figures/trials/collocation_B_v4"

# Rendered frames of the collocation sequence: keep columns [100, 1188) and remove rows [100, 250), a band
# that shows nothing but the wall panel above the team.
SEQUENCE_TRIM = {"columns": [100, 1188], "drop_rows": [100, 250]}

# Plots shown without their white margins: the pixels of the source that are kept, as (first column, first row,
# column after the last, row after the last).
CROPS = {
    "profile_a_swings.png": (0, 2, 680, 341),
    "profile_b_authority.png": (14, 2, 697, 319),
    "profile_c_excursion.png": (25, 0, 744, 341),
    "trial_a_distances.png": (3, 16, 712, 331),
    "sandwich_b.png": (20, 20, 440, 570),
    "sandwich_c.png": (20, 20, 440, 570),
    "nu_sensitivity.png": (24, 2, 755, 456),
}

FIGURES = [
    # destination, figure label, source (relative to the results root), generator, operation
    ("profile_a_swings.png", "profile", f"{PAPER_FIGS}/paper_fig1a_swings.png", "experiments.paper_figures", "crop"),
    ("profile_b_authority.png", "profile", f"{PAPER_FIGS}/paper_fig1b_authority.png", "experiments.paper_figures", "crop"),
    ("profile_c_excursion.png", "profile", f"{PAPER_FIGS}/paper_fig1c_profile.png", "experiments.paper_figures", "crop"),
    ("e1_margin.png", "e1", "core/figures/fig_e1_thm5_perfect_k0p5_10_mu.png", "experiments.make_figures", "copy"),
    ("sampled_margin_map.png", "sampled", f"{PAPER_FIGS}/paper_fig5a_margin_map.png", "experiments.paper_figures", "copy"),
    ("trial_a_distances.png", "trial-distances", f"{TRIAL}/signal_distances.png", "experiments.trial_figures", "crop"),
    ("trial_b_swing.png", "trial-swing", f"{TRIAL}/signal_swing_z.png", "experiments.trial_figures", "copy"),
    ("trial_c_phase.png", "trial-phase", f"{TRIAL}/phase_swing.png", "experiments.trial_figures", "copy"),
    ("sandwich_a.png", "sandwich-a", f"{PAPER_FIGS}/paper_fig4a_sandwich_A.png", "experiments.paper_figures", "copy"),
    ("sandwich_b.png", "sandwich-bc", f"{PAPER_FIGS}/paper_fig4b_sandwich_B.png", "experiments.paper_figures", "crop"),
    ("sandwich_c.png", "sandwich-bc", f"{PAPER_FIGS}/paper_fig4c_sandwich_C.png", "experiments.paper_figures", "crop"),
    ("sequence_0.png", "sequence", f"{COLLOCATION}/sequence_0.png", "experiments.sequence_figures", "trim"),
    ("sequence_1.png", "sequence", f"{COLLOCATION}/sequence_1.png", "experiments.sequence_figures", "trim"),
    ("sequence_2.png", "sequence", f"{COLLOCATION}/sequence_2.png", "experiments.sequence_figures", "trim"),
    ("sequence_3.png", "sequence", f"{COLLOCATION}/sequence_3.png", "experiments.sequence_figures", "trim"),
    ("thrust_budget.png", "thrust", f"{PAPER_FIGS}/paper_fig6a_thrust_budget.png", "experiments.paper_figures", "copy"),
    ("nu_sensitivity.png", "nu", f"{PAPER_FIGS}/paper_fig7_price_of_nu.png", "experiments.paper_figures", "crop"),
]


def results_root() -> Path:
    return Path(os.environ.get("SIM_RESULTS_DIR", REPO / "results")).resolve()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trimmed(source: Path):
    """The frame of the paper as an RGB array: outer columns and the wall-only band of rows removed."""
    import numpy as np
    from PIL import Image

    frame = np.asarray(Image.open(source).convert("RGB"))
    c0, c1 = SEQUENCE_TRIM["columns"]
    r0, r1 = SEQUENCE_TRIM["drop_rows"]
    return np.vstack([frame[:r0, c0:c1], frame[r1:, c0:c1]])


def cropped(source: Path, name: str):
    """The part of the plot that the paper shows, as an image with the mode and the resolution of the source."""
    from PIL import Image

    image = Image.open(source)
    return image.crop(CROPS[name]), image.info.get("dpi")


def write(dest: Path, source: Path, operation: str) -> None:
    if operation == "copy":
        dest.write_bytes(source.read_bytes())
    elif operation == "crop":
        image, dpi = cropped(source, dest.name)
        image.save(dest, optimize=True, **({"dpi": dpi} if dpi else {}))
    else:
        from PIL import Image

        Image.fromarray(trimmed(source)).save(dest, optimize=True)


def matches(dest: Path, source: Path, operation: str) -> bool:
    if not dest.is_file():
        return False
    if operation == "copy":
        return sha256(dest) == sha256(source)
    import numpy as np
    from PIL import Image

    if operation == "crop":
        return np.array_equal(np.asarray(Image.open(dest).convert("RGBA")),
                              np.asarray(cropped(source, dest.name)[0].convert("RGBA")))
    return np.array_equal(np.asarray(Image.open(dest).convert("RGB")), trimmed(source))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="verify the figure files instead of writing them")
    args = parser.parse_args()
    root = results_root()
    FIG.mkdir(exist_ok=True)
    manifest, failures = [], []
    for name, label, rel, generator, operation in FIGURES:
        source, dest = root / rel, FIG / name
        if not source.is_file():
            failures.append(f"missing source: {source}")
            continue
        if args.check:
            if not matches(dest, source, operation):
                failures.append(f"{dest.relative_to(REPO)} differs from {rel} ({operation})")
        elif not matches(dest, source, operation):
            write(dest, source, operation)
        entry = {"file": name, "figure": f"fig:{label}", "source": f"results/{rel}",
                 "generator": f"authority_barriers.{generator}", "operation": operation,
                 "source_sha256": sha256(source)}
        if operation == "crop":
            entry["kept_columns_and_rows"] = list(CROPS[name])
        manifest.append(entry)
    if not args.check and not failures:
        record = {"figures": manifest, "trim": SEQUENCE_TRIM}
        (FIG / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    expected = {name for name, *_ in FIGURES} | {"manifest.json"}
    for extra in sorted(p.name for p in FIG.iterdir() if p.name not in expected):
        failures.append(f"IEEE_ACC2027/figures/{extra} is not in the list of figures")
    for line in failures:
        print("MISMATCH", line)
    print(f"{len(FIGURES) - len([f for f in failures if 'differs' in f or 'missing' in f])}/{len(FIGURES)} figure files "
          f"{'verified' if args.check else 'written'} (results root: {root})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
