# `IEEE_ACC2027/` — the manuscript

*Explicit Two-Sided Viability Bounds for Safety Filters with Configuration-Dependent Braking Authority.*
IEEE conference format, five figures and one table, no supplementary material.

## Files

| File | Content |
|---|---|
| `main.tex` | document class, packages, float parameters, title, author block, abstract, keywords, the `\input` of every section, and the bibliography commands |
| `macros.tex` | theorem-like environments (one shared counter) and the notation macros |
| `sections/01_introduction.tex` … `07_conclusion.tex` | one file per section |
| `References.bib` | the bibliography as BibTeX entries; typeset with the style `IEEEtran`, numbered in the order of first citation |
| `figures/` | the figure files |
| `latexmkrc` | `latexmk` settings: pdflatex, all generated files under `build/` |
| `IEEEtran.cls`, `IEEEtran.bst` | the IEEEtran class, version 1.8b, and its BibTeX style, version 1.14 (both 2015/08/26), distributed under the LaTeX Project Public License as stated in their headers |
| `build/` | generated; not tracked |

## Building

```
cd IEEE_ACC2027 && latexmk                 # pdflatex and bibtex as often as needed; writes build/main.pdf
../scripts/build_paper.sh           # builds IEEE_ACC2027/build/main.pdf
latexmk -C                          # removes the generated files
```

An editor that calls `latexmk` with its own output directory writes the generated files next to `main.tex`;
`.gitignore` covers both places.

## Figures

Every file in `figures/` comes from one file that the experiments write under the results directory (`results/`,
which is not part of the repository), and is named after the label of its figure.
The table gives the source of every file.

| Figure | Files | Source under `results/core/figures/` |
|---|---|---|
| 1 `fig:scene` | TikZ drawing in `sections/01_introduction.tex` | — |
| 2 `fig:profile` | `profile_a_swings.png`, `profile_b_authority.png`, `profile_c_excursion.png` | `paper/paper_fig1a_swings.png`, `paper_fig1b_authority.png`, `paper_fig1c_profile.png`, cropped |
| 3 `fig:sequence` | `sequence_0.png` … `sequence_3.png` | `trials/collocation_B_v4/sequence_*.png` |
| 4 `fig:exact` | `exact_comparison.png` | `../e8_exact_model/figures/fig_e8_comparison.png` |
| 5 `fig:sandwich-a` | `sandwich_a.png` | `paper/paper_fig4a_sandwich_A.png` (with the certified stopping distance, `paper_figures --only 4`) |
| not included | `sampled_margin_map.png`, `e1_margin.png`, `trial_a_distances.png`, `trial_b_swing.png`, `trial_c_phase.png`, `sandwich_b.png`, `sandwich_c.png`, `thrust_budget.png`, `nu_sensitivity.png` | `paper/paper_fig5a_margin_map.png`, `fig_e1_thm5_perfect_k0p5_10_mu.png`, `trials/e2_perfect_max_off0_layer1_75/signal_distances.png` (cropped), `signal_swing_z.png`, `phase_swing.png`, `paper/paper_fig4b_sandwich_B.png`, `paper_fig4c_sandwich_C.png`, `paper_fig6a_thrust_budget.png`, `paper_fig7_price_of_nu.png` |

The sources are written by the experiments E1, E2 and E3 (collocation) and the figure generators
(`python -m authority_barriers.reproduce --core --results results`, with the extra `viz` installed for the rendered
frames). Fig. 4 is written by `python -m authority_barriers.experiments.e8_exact_model --compare`: one state of the
safe set on the exact taut-cable model, the HOCBF filter with three gain pairs against the filter of the paper, in
the plane of wall distance and approach speed.

The three figures of Sec. VI are declared at the top of `sections/06_numerical_study.tex`, ahead of the text and in
the order of their numbers. Where they land depends on two rules of LaTeX. Figures appear in the order of their
numbers, one-column and two-column figures alike, so a figure that finds no room holds back every later one. A
two-column figure is placed at the top of a page, at the earliest the page after the one on which it is declared.

`main.tex` sets the space between two figures to 6 pt and the space between a figure and the text to 8 pt; the
class sets 0.85 and 1.55 baselines (10 and 19 pt). Where a column is not filled, because the next heading or
figure does not fit, LaTeX stretches these spaces to fill it.

Fig. 3 spans both columns and is declared first, so it takes the top of the page after the one on which Sec. VI
begins, and the two one-column figures follow it.

A cropped file is the plot of the source without its white margins: the panels of Fig. 2.

## Numbering

The paper numbers its definitions, lemmas, theorems, propositions, assumptions and remarks with one counter. The
code and the result files that it writes use the numbers of the full-length manuscript.
`authority_barriers/README.md`, Sec. 0, relates the two.

## Open items

- `main.tex`: the author block gives e-mail addresses and no affiliations.
- Length: the paper builds to 9 pages, of which the last holds about half a column (the end of the references).
  ACC 2027 accepts 8 pages (6 without page charges).
- `IEEEtran.cls` is the generic class; a venue may require its own author kit.
