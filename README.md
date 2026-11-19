# Reproduction package: the paper's numerical study

This folder reproduces every figure and table of the paper
"Operator Calculus for Population-Based Optimization: Modular Convergence and
Finite-Population Guarantees" (arXiv:2606.14289): the numerical study, the three verified
algorithms, the practitioner's guide, and the explanatory figures.  The runners of the studies
are frozen: they verify the SHA-256 hashes of their own sources, of `objectives.py` and of the
protocols, so these files must not be edited.  For applying the results to new
problems and algorithms, use the EvoScope toolbox (https://github.com/pma-papers/evoscope);
its tests check that it reproduces the numbers produced here.

## Figures and tables of the paper

The numbering is that of the current version of the paper: an article with Sections I-X and
Figs. 1-8, and supplementary material with Sections S1-S11.  The earlier two-part version
(arXiv v3: article and Online Appendix, Figures 1-3 and S1-S8) corresponds to the tag
`arxiv-v3` of this repository; its numbering is given in the last column.

| Study (runner) | Article | Supplement | arXiv v3 |
|---|---|---|---|
| One-step checks of the eight variants (`one_step_study.py`) | Section VII, Fig. 6(a)-(b) | S1.1-S1.4: Tables S1-S4, Figures S1-S2 | Section 7, Figure 2(a)-(b); S1.1-S1.4 |
| Exploration diagnostics (`exploration_study.py`) | Section VII | S1.5: Figure S3 | S1.5: Figure S3 |
| Finite-population trajectories (`trajectory_study.py`, `discovery_inference.py`) | Section VII, Fig. 6(c)-(d), measured hitting-time bound (9) | S1.6-S1.7: Figures S4-S6, Tables S5-S6; S2.2-S2.3: Tables S9-S10 | Section 7, Figure 2(c)-(d), equation (12); S1.6-S1.7, S2.2-S2.3 |
| Verified instances (`verified_instances.py`) | Section VIII, Fig. 8 | S4.8 | Section 8, Figure 3; S4.8 |
| Practitioner's guide (`guide_figures.py`, `discovery_bound_design.py`) | Fig. 7 | S2.5-S2.6: Figure S7, Table S11 | S2.5-S2.6: Figures S7-S8, Table S11 |
| Explanatory figures, drawn with EvoScope (`explanatory_figures.py`) | Figs. 1-5 | | |

| Figure or table | Script | File in `figures/` or `results/` |
|---|---|---|
| Fig. 1 (operators and their effects) | `explanatory_figures.py` | `explanatory_operators.pdf` |
| Fig. 2 (the four conditions) | `explanatory_figures.py` | `explanatory_conditions.pdf` |
| Fig. 3 (distance between populations) | `explanatory_figures.py` | `explanatory_population_distance.pdf` |
| Fig. 4 (evolution equation and drift inequality) | `explanatory_figures.py` | `explanatory_evolution_equation.pdf` |
| Fig. 5 (convergence targets) | `explanatory_figures.py` | `explanatory_convergence_targets.pdf` |
| Fig. 6 (numerical study) | `article_figures.py` | `article_main_results.pdf` |
| Fig. 7 (measured hitting-time bound) | `guide_figures.py` | `guide_discovery_bound.pdf` |
| Fig. 8 (three verified algorithms) | `article_figures.py` | `article_verified_instances.pdf` |
| Figure S1 (composition-theorem checks) | `consistency_row_figure.py` | `canonical_consistency.pdf` |
| Figure S2 (estimated drift coefficients) | `dissipation_estimates.py` | `dissipation_estimates.pdf` |
| Figure S3 (exploration diagnostics) | `supplement_figures.py` (drawn by `exploration_figures.py`) | `supplement_exploration.pdf` |
| Figure S4 (archive failure against function evaluations) | `supplement_figures.py` (drawn by `budget_curves.py`) | `supplement_budget_curves.pdf` |
| Figure S5 (offspring failure by generation 200) | `finite_population_failures_figure.py` | `canonical_finite_population_failures.pdf` |
| Figure S6 (dynamic contrasts, unequal wells) | `supplement_figures.py` (drawn by `relabel_dynamic_contrasts.py`) | `supplement_dynamic_contrasts_wells.pdf` |
| Figure S7 (drift-coefficient estimation) | `guide_figures.py` | `guide_lambda_estimate.pdf` |
| Figure S8 (landscape conditions and operators) | `article_figures.py` | `article_landscape_operators.pdf` |
| Table S1 (test problems) | `objectives.py` | |
| Tables S3-S4 (one-step design, constants, validation) | `one_step_figures.py` | `results/one-step/*.csv` |
| Tables S5-S6 (finite-population results, audit batch sizes) | `trajectory_figures.py`; Table S5 also `reproduce_paper_figures.py` | `results/trajectories/finite_population_audit.json` |
| Table S10 (worked finite-budget criterion) | `criterion_worked_example.py` | `results/analysis/criterion_worked_example.json` |
| Table S9 (reference-run estimates) | `reference_run_modulus.py` | `results/analysis/reference_run_modulus.json` |
| Table S11 (design of the measured bound) | `discovery_bound_design.py` | `results/analysis/discovery_bound_design.json` |
| `supplement_figures.py` | `figures/supplement_exploration.pdf`, `supplement_budget_curves.pdf`, `supplement_dynamic_contrasts_wells.pdf` (Figures S3, S4, S6: the figures of `exploration_figures.py`, `budget_curves.py` and `relabel_dynamic_contrasts.py` in the current wording); `results/analysis/supplement_figures_audit.json` | one-step, exploration, trajectories |
| `explanatory_figures.py` | `figures/explanatory_*.pdf` (Figs. 1-5); `results/analysis/explanatory_figures_audit.json` | EvoScope; no study data |

The figures of the current version use the terms of that version (for example "time step" for
"macro-step", "failure frequency" for "non-discovery", "function evaluations" for "objective calls",
"full variant" for "full assembly", and expected values in place of the functionals V, E and W).
`code/wording.py` holds these replacements.  They change text only: `article_figures.py` and
`guide_figures.py` apply them when a figure is saved, and `supplement_figures.py` redraws the three
supplement figures concerned with their original scripts in a temporary folder.  Every one of these
scripts checks the plotted numbers against the audits of the study.  The study scripts themselves
keep the wording of the study and write the figures under their original names.

`code/article_figures.py` draws its figures at a width of 6 in with lettering of at least 9 pt,
with the plotting functions of the studies.  `code/guide_figures.py` draws the two figures of
the practitioner's guide from the one-step arrays and the trajectory arrays of unequal wells
and asserts that the recomputed bounds equal the stored analysis, and
`code/discovery_bound_design.py` computes Table S11 with the frozen inference module.
`code/explanatory_figures.py` draws Figs. 1-5 with EvoScope: they are illustrations with fixed
seeds, not results of the study, and their numbers are recorded in
`results/analysis/explanatory_figures_audit.json`.  The 256 runs behind Fig. 4 take about 13 minutes; their
arrays are kept in `results/explanatory-cache/` (not shipped) so that a second run takes seconds.
`paper_guide_figures_evoscope.py` (in this folder; it needs EvoScope on the Python path)
redraws the practitioner's-guide figures with EvoScope and checks the toolbox against the
study's analysis.  The text below refers to the study and its outputs; the section numbers in
it are those of this README.

The code is released under the MIT License (see `LICENSE`).


This package contains the complete code and the pre-registered protocols of every
numerical experiment reported in the paper, together with compact summaries of
the results (JSON summaries and audits, CSV tables, two small analysis arrays).
All objectives are synthetic and defined in the code; no external data,
pretrained models or accelerators are used.  The large array files are not
included: every one of them is regenerated by the runners from the fixed seeds
(Section 2), and the shipped summaries record their SHA-256 hashes.

## Requirements

Python 3.10 or newer with NumPy, SciPy and Matplotlib.  The reported results were
produced with Python 3.13.3, NumPy 2.4.2, SciPy 1.17.1 and Matplotlib 3.10.8:

```sh
python -m pip install -r requirements.txt
```

The explanatory figures (`explanatory_figures.py`) also need EvoScope, installed
(`python -m pip install git+https://github.com/pma-papers/evoscope`) or given by its `src/` folder
(`--evoscope PATH`).  They were drawn with EvoScope 0.2.0.dev0.

Run every command from the package root (the directory containing this file).
If Matplotlib cannot write its cache, set `MPLCONFIGDIR` to a temporary directory.

## 1. Quick start (about 30 seconds without the explanatory figures)

```sh
python code/regenerate_arrays.py            # one-step calibration and validation arrays, verified-instances arrays
python code/one_step_figures.py             # one-step tables and figures
python code/consistency_row_figure.py       # Figure S1: composition-theorem checks
python code/dissipation_estimates.py        # Figure S2: drift coefficients
python code/finite_population_failures_figure.py   # Figure S5: finite-population failures (from the shipped analysis arrays)
python code/verified_instances_figures.py   # study figure of the verified instances
python code/explanatory_figures.py --evoscope PATH   # Figs. 1-5 (illustrations, no study data; about 15 minutes, 13 of them for Fig. 4)
```

`regenerate_arrays.py` runs `one_step_study.py --calibrate` (2 s) and `--validate`
(7 s) and `verified_instances.py --run` (2 s), compares each regenerated array file
with the SHA-256 recorded in the shipped audits and reports `MATCH`; the shipped
summaries stay in place (the one-step runner never writes into a folder that
already holds a summary, so the helper sets the shipped summaries aside while it
runs and restores them).  It does nothing when the files are already present.

The remaining figures of the paper need the exploration run (80 s: Figure S3) or the
trajectory run (13 minutes: Figs. 6-8 and Figures S4, S6-S8), see Sections 2 and 3; then

```sh
python code/article_figures.py              # Figs. 6 and 8, Figure S8
python code/guide_figures.py                # Fig. 7, Figure S7
python code/supplement_figures.py           # Figures S3, S4, S6 in the current wording
```

## 2. The experiments: runners, run times and outputs

Every runner has `--check` (read-only implementation checks, seconds),
`--smoke` (a reduced run into a separate directory, seconds) and the full mode.
All seeds are fixed in the sources; the saved summaries record the seed
construction and the SHA-256 of sources, protocols and outputs, and the runners
verify the hashes of their own source, of `objectives.py`, of their protocol and
of their inputs.  Output directories are created exclusively: no saved result is
ever overwritten.

| Study | Runner and full command | Time | Output (not shipped) |
|---|---|---|---|
| One-step assembly checks | `one_step_study.py --calibrate`, then `--validate` | 2 s + 7 s | `results/one-step/canonical-{calibration,validation}/canonical_arrays.npz` (2.8 MB + 11.3 MB) next to the shipped summaries; use `regenerate_arrays.py`, see Section 1 |
| Exploration diagnostics on the saved validation populations | `exploration_study.py --run` | 80 s | `results/exploration/exploration-full/` (20 `.npz` files and a summary, about 90 MB); needs the validation arrays |
| Finite-population trajectories, discovery bounds, dynamic contrasts | `trajectory_study.py --run --workers 4` | 13 min (four processes) | `results/trajectories/finite-population-full/*.npz` (80 files, 1.1 GB) next to the shipped per-case summaries; move the shipped folder aside first (`mv results/trajectories/finite-population-full results/trajectories/finite-population-shipped`) and compare the two sets of `.json` files afterwards |
| Verified instances (CBO, CMA-ES-type flow, recombinative ES) | `verified_instances.py --run --output DIR` | 2 s | `DIR/arrays.npz` (14.7 MB) and `DIR/summary.json`; `regenerate_arrays.py` moves `arrays.npz` next to the shipped summary |

With the versions listed above, the calibration, validation and verified-instances
runs reproduced every saved array bitwise (this is what `regenerate_arrays.py`
checks), the exploration run reproduced every table in `results/one-step/` and
`results/exploration/` byte for byte, and the trajectory run reproduced all 80 raw
trajectory files bitwise, as checked against the SHA-256 hashes in the shipped
summaries.  Bitwise equality across platforms or library versions is not
guaranteed; if `regenerate_arrays.py` reports `DIFFERS`, the plotting scripts
still run from the regenerated arrays, but the checks that are pinned to the
recorded hashes (`exploration_study.py`, the relabel scripts) will fail.

After the trajectory run, `extract_dynamic_traces.py` copies the N=32,
order-S-then-R traces of the unequal-wells and quartic instances (all 128 runs,
eight assemblies, 201 generations, four metrics) from four hash-checked raw files
into `results/trajectories/dynamic_traces_N32.npz` (16 MB), the input of the
dynamic-contrast part of `reproduce_paper_figures.py`.

## 3. Figures, tables and audits

The plotting scripts write their figures to `figures/` and overwrite the derived
tables and audits in `results/`, so a rerun can be compared file by file with the
shipped copies (for example with `git diff --no-index` or `cmp`).  The table
says which regenerated arrays each script needs: "one-step" and "verified" are
produced by `regenerate_arrays.py`, "exploration" and "trajectories" by the runs
of Section 2, "traces" by `extract_dynamic_traces.py`.

| Script | Output | Needs |
|---|---|---|
| `one_step_figures.py` | `figures/one_step_consistency.pdf`, `one_step_components.pdf`; the three CSV tables and `one_step_figure_audit.json` in `results/one-step/` | one-step |
| `exploration_figures.py` | `figures/exploration_noise.pdf`, `exploration_components.pdf`; the five CSV tables and `exploration_analysis.json` in `results/exploration/` | one-step, exploration |
| `trajectory_figures.py` (with `discovery_inference.py`) | `figures/trajectory_failures.pdf`, `trajectory_discovery_bounds.pdf`; `results/trajectories/finite_population_analysis.npz` and `finite_population_audit.json` | trajectories |
| `dynamic_contrasts.py` | `figures/trajectory_contrasts_{wells,quartic}.pdf`; `results/trajectories/dynamic_contrast_audit.json` | trajectories (N=32 files) |
| `relabel_trajectory_figures.py` | `figures/*_relabeled.pdf` (population wording); `results/figure-labels/population_figure_labels.json` | the figures of `one_step_figures.py` and `exploration_figures.py` |
| `relabel_dynamic_contrasts.py` | `figures/canonical_dynamic_contrasts_{wells,quartic}.pdf` (study wording); `results/figure-labels/dynamic_contrast_labels.json` | trajectories (N=32 files) and the figures of `dynamic_contrasts.py` |
| `relabel_exploration_figures.py` | `figures/canonical_exploration.pdf` (study wording), `canonical_exploration_components.pdf`, `canonical_discovery_bounds.pdf`; `results/analysis/notation_labels.json` | the figures of `exploration_figures.py` (it calls the same plotting function, which hashes them) |
| `consistency_row_figure.py` | `figures/canonical_consistency.pdf` (Figure S1); `results/analysis/consistency_row_audit.json` | one-step |
| `dissipation_estimates.py` | `figures/dissipation_estimates.pdf` (Figure S2); `results/analysis/dissipation_estimates_audit.json` | one-step |
| `finite_population_failures_figure.py` | `figures/canonical_finite_population_failures.pdf` (Figure S5); audit | nothing beyond the shipped files |
| `verified_instances_figures.py` | `figures/verified_instances.pdf` (study figure; Fig. 8 is drawn by `article_figures.py`); `results/verified-instances/verified_instances_figure_audit.json` | verified |
| `budget_curves.py` | `figures/canonical_budget_curves.pdf` (study wording); `results/analysis/budget_curves.npz` and audit | trajectories |
| `reference_run_modulus.py` | `results/analysis/reference_run_modulus.json` (Table S9) | trajectories |
| `criterion_worked_example.py` | `results/analysis/criterion_worked_example.json` (Table S10) | trajectories |
| `article_figures.py` | `figures/article_landscape_operators.pdf`, `article_main_results.pdf`, `article_verified_instances.pdf` (Figure S8, Figs. 6 and 8); `results/analysis/article_figures_audit.json` | one-step, verified, trajectories (N=32 files) |
| `guide_figures.py` | `figures/guide_discovery_bound.pdf`, `guide_lambda_estimate.pdf` (Fig. 7, Figure S7); `results/analysis/guide_figures_audit.json` | one-step, trajectories (unequal wells, broad starts; or `--trajectories DIR`) |
| `discovery_bound_design.py` | `results/analysis/discovery_bound_design.json` (Table S11) | nothing beyond `discovery_inference.py` |
| `main_text_figures.py` | `figures/main_results.pdf`, `certified_rates.pdf` (earlier versions); `results/analysis/main_results_figure_audit.json` | one-step, verified, trajectories (N=32 files), `budget_curves.npz` (shipped) |
| `schematic_figures.py` | `figures/landscape.pdf`, `operators.pdf`, `landscape_operators.pdf` (earlier versions) | nothing (illustrations) |
| `reproduce_paper_figures.py` | the eight data figures of the one-step, exploration and trajectory studies in `reproduced-figures/`, in the manuscript's wording, and the finite-population table | one-step, traces |

The relabelling scripts apply exactly the wording changes of the manuscript to
the original study figures and verify that no plotted value changes; with
`--check` they write nothing and compare against the shipped audits.

`reproduce_paper_figures.py` is the digest-checked entry point: it calls the
original plotting functions on the regenerated one-step arrays, the shipped
tables and audits and the extracted traces, compares each figure's plotted data
(axes, limits, and every line, image, patch and collection) with a digest recorded
when the manuscript figure was produced, recomputes the dynamic-contrast bootstrap
summaries and the finite-population table, and reports `MATCH` for each item.
Nothing in `results/` is written.

## Contents

- `code/`: all sources.  `objectives.py` defines the ten test problems (instances
  and objective values) and nothing else.  Runners: `one_step_study.py`,
  `exploration_study.py`, `trajectory_study.py` (with `discovery_inference.py`),
  `verified_instances.py`.  Helpers: `regenerate_arrays.py`,
  `extract_dynamic_traces.py`.  Plotting, analysis and relabelling scripts, including those
  of the article's figures (`article_figures.py`) and of the practitioner's guide
  (`guide_figures.py`, `discovery_bound_design.py`): Section 3.
- `protocols/`: the pre-registered experiment protocols (hash-checked by the runners).
- `results/one-step/`: the calibration and validation summaries of the one-step
  study (every generator action and finite-step statistic per instance, family and
  assembly), the derived CSV tables and the figure audit.
- `results/exploration/`: the complete per-population exploration diagnostics as CSV
  tables and the analysis summary.
- `results/trajectories/`: the per-case summaries of the trajectory run
  (`finite-population-full/*.json`, with the SHA-256 of every raw file), the
  configuration-level analysis arrays `finite_population_analysis.npz` (0.5 MB), and
  the finite-population and dynamic-contrast audits.
- `results/figure-labels/`: audits of the presentation-only label changes, including
  the plotted-data digests used by `reproduce_paper_figures.py`.
- `results/verified-instances/`: the summary and figure audit of the three verified instances.
- `results/analysis/`: outputs of the derived analyses (budget curves, reference-run
  modulus, worked criterion example, practitioner's-guide figures and design table, figure
  audits, notation labels).
- `figures/`: default output directory of the plotting scripts.

## Provenance of the names and hash records

The development copies of these files carried protocol-version tags (a trailing
version number) in file, module, folder and figure names and in a few JSON keys.
Those tags were removed when this package was assembled.  No numerical value, array, table row or
seed was changed: the `.csv` and `.npz` files are byte-identical to the
development copies, and the JSON files differ only in renamed keys and path
strings, in two reworded label-audit sentences that named versions, and in the
hash records described next.

The runners and audits verify SHA-256 hashes of their own source files, of the
objective module, of the protocol files and of their input files.  Because
renaming changes the bytes of the scripts and protocols (and the renamed JSON
files), every such hash record was recomputed from the renamed files ("re-sealed")
so that the unchanged self-checks pass; the self-check logic itself is untouched.
The recorded hashes of the array files that are not shipped are those of the
development copies, which the runners reproduce bitwise.  Hashes of figure files
are recorded for files written with reproducible PDF metadata and were refreshed
from the regenerated figures; figures saved with a creation date
(`exploration_figures.py`, `trajectory_figures.py` and the relabelled copies) keep
the values recorded during development and are not expected to be reproduced byte
for byte.

## Scope

The experiments are finite-sample checks of the paper's generator identities and
hitting-time recursion on synthetic test problems.  They are not benchmarks of
optimizer performance and not proofs of uniform convergence assumptions.
