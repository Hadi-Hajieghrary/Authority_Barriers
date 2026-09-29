# `IEEE_ACC2027/` — the manuscript

*Explicit Two-Sided Viability Bounds for Safety Filters with Configuration-Dependent Braking Authority.*
IEEE conference format, twelve figures, no supplementary material.

## Files

| File | Content |
|---|---|
| `main.tex` | document class, packages, float parameters, title, author block, abstract, keywords, the `\input` of every section, and the bibliography commands |
| `macros.tex` | theorem-like environments (one shared counter) and the notation macros |
| `sections/01_introduction.tex` … `07_conclusion.tex` | one file per section |
| `References.bib` | the bibliography as BibTeX entries; typeset with the style `IEEEtran`, numbered in the order of first citation |
| `figures/` | the figure files and `manifest.json`, which records the result file that each of them was taken from |
| `collect_figures.py` | takes every figure file from a results directory, or verifies the figure files against it |
| `latexmkrc` | `latexmk` settings: pdflatex, all generated files under `build/` |
| `IEEEtran.cls`, `IEEEtran.bst` | the IEEEtran class, version 1.8b, and its BibTeX style, version 1.14 (both 2015/08/26), distributed under the LaTeX Project Public License as stated in their headers |
| `build/` | generated; not tracked |

## Building

```
cd IEEE_ACC2027 && latexmk                 # pdflatex and bibtex as often as needed; writes build/main.pdf
../scripts/build_paper.sh           # builds; where results/ exists, the figure files are verified against it first
latexmk -C                          # removes the generated files
```

An editor that calls `latexmk` with its own output directory writes the generated files next to `main.tex`;
`.gitignore` covers both places.

## Figures

Every file in `figures/` comes from one file that the experiments write under the results directory (`results/`,
which is not part of the repository), and is named after the label of its figure.
`collect_figures.py` holds the list; `figures/manifest.json` records, for each file, the figure label, the source,
the module that generates the source, the operation (`copy`, `crop` or `trim`) and the SHA-256 checksum of the source.

| Figure | Files | Source under `results/core/figures/` |
|---|---|---|
| 1 `fig:scene` | TikZ drawing in `sections/01_introduction.tex` | — |
| 2 `fig:profile` | `profile_a_swings.png`, `profile_b_authority.png`, `profile_c_excursion.png` | `paper/paper_fig1a_swings.png`, `paper_fig1b_authority.png`, `paper_fig1c_profile.png`, cropped |
| 3 `fig:e1` | `e1_margin.png` | `fig_e1_thm5_perfect_k0p5_10_mu.png` |
| 4 `fig:sampled` | `sampled_margin_map.png` | `paper/paper_fig5a_margin_map.png` |
| 5 `fig:trial-distances` | `trial_a_distances.png` | `trials/e2_perfect_max_off0_layer1_75/signal_distances.png`, cropped |
| 6 `fig:trial-swing` | `trial_b_swing.png` | `trials/e2_perfect_max_off0_layer1_75/signal_swing_z.png`, trimmed in LaTeX |
| 7 `fig:trial-phase` | `trial_c_phase.png` | `trials/e2_perfect_max_off0_layer1_75/phase_swing.png` |
| 8 `fig:sandwich-a` | `sandwich_a.png` | `paper/paper_fig4a_sandwich_A.png` |
| 9 `fig:sandwich-bc` | `sandwich_b.png` (a), `sandwich_c.png` (b) | `paper/paper_fig4b_sandwich_B.png`, `paper/paper_fig4c_sandwich_C.png`, cropped |
| 10 `fig:sequence` | `sequence_0.png` … `sequence_3.png` | `trials/collocation_B_v4/sequence_0.png` … `sequence_3.png`, trimmed |
| 11 `fig:thrust` | `thrust_budget.png` | `paper/paper_fig6a_thrust_budget.png` |
| 12 `fig:nu` | `nu_sensitivity.png` | `paper/paper_fig7_price_of_nu.png`, cropped |

```
python IEEE_ACC2027/collect_figures.py            # write figures/ from results/ (or from $SIM_RESULTS_DIR); files that match stay
python IEEE_ACC2027/collect_figures.py --check    # verify figures/ against results/
```

Both commands need the results of the experiments E1, E2 and E3 (collocation) and the figures made from them
(`python -m authority_barriers.reproduce --core --results results`, with the extra `viz` installed for the rendered
frames).

The ten figures of Sec. VI are declared at the top of `sections/06_numerical_study.tex`, ahead of the text and in
the order of their numbers. Where they land depends on two rules of LaTeX. Figures appear in the order of their
numbers, one-column and two-column figures alike, so a figure that finds no room holds back every later one. A
two-column figure is placed at the top of a page, at the earliest the page after the one on which it is declared.

`main.tex` sets the space between two figures to 6 pt and the space between a figure and the text to 8 pt; the
class sets 0.85 and 1.55 baselines (10 and 19 pt). Where a column is not filled, because the next heading or
figure does not fit, LaTeX stretches these spaces to fill it.

Fig. 9 has two subfigures, set with `\subfloat` of the package `subfig`, which `main.tex` loads with the option
`caption=false` so that the captions keep the format of the class, and with no space above the subfigures.
Fig. 6 is included with `trim` and `clip`, which cut the legend below the plot. In its source the legend overlaps
the label of the time axis, so the cut also removes the lowest part of that label.

A copied file is identical to its source byte for byte. A cropped file is the plot of the source without its
white margins: the panels of Fig. 2, Fig. 5, the two plots of Fig. 9 and Fig. 12; `CROPS` in
`collect_figures.py` and `figures/manifest.json` give the columns and rows that are kept. A trimmed frame is the
rendered frame with columns 100 to 1187 kept and rows 100 to 249 removed; the removed band shows only the wall
panel above the team.

## Numbering

The paper numbers its definitions, lemmas, theorems, propositions, assumptions and remarks with one counter. The
code and the result files that it writes use the numbers of the full-length manuscript.
`authority_barriers/README.md`, Sec. 0, relates the two.

## Open items

- `main.tex`: the author block gives e-mail addresses and no affiliations.
- Length: on 2026-09-29 the paper builds to 9 pages; page 9 holds the references. The page limit of the venue was
  not checked.
- `IEEEtran.cls` is the generic class; a venue may require its own author kit.
