# Imported Baseline Results

This directory preserves reusable baseline outputs from the user's related
repository:

`git@github.com:Al-Toretto/Interpretable-Lazy-Classification-for-Numerical-Data-using-Interval-Pattern-Structures.git`

Source commit:

`6cf605a52d7effd75e9dd83dea02b0700843dcc4`

The copied source tree was temporarily placed under `old_baselines/` and then
removed after extracting these small result artifacts.

## Contents

- `repeated_macro_f1/raw_repeat_results.csv`: per-dataset, per-seed test
  macro-F1 results, selected hyperparameters, runtime, and status.
- `repeated_macro_f1/summary.csv`: mean macro-F1 summaries over repeats.
- `repeated_macro_f1/latex_table_body.tex`: LaTeX table body from the source
  experiment.
- `repeated_macro_f1/summary.txt`: short source summary.
- `repeated_sizes/raw_repeat_sizes.csv`: per-repeat compactness/size results.
- `repeated_sizes/summary.csv`: size and compactness summaries over repeats.
- `repeated_sizes/compactness.csv`: dataset-level compactness summary.
- `repeated_sizes/latex_table_body.tex`: LaTeX table body from the source
  size experiment.

## Reuse Notes

These files include the expensive numerical-data baselines:

- `fcalc`: deterministic FCALC/LazyFCA baseline;
- `fcalc_rand`: randomized FCALC/LazyFCA baseline;
- `ips_knn`: compact interval pattern-structure KNN baseline.

They also include classical ML baselines from the related paper. Results use
10 stratified 80/20 splits with split seeds `1998` through `2007`.

The files store macro-F1, selected parameters, elapsed time, and compactness
statistics. They do not store predictions or confusion matrices, so binary
positive-class F1 cannot be recovered from them. For comparability, the ranking
experiments in this repository should use macro-F1 as the primary paper metric
when compared against these imported baselines.

To convert these preserved files into a result folder after a ranking run:

```bash
python3 experiments/import_baseline_results.py --run-name ranking_macro_f1_10splits
```

The importer writes:

- `imported_baseline_results.csv`;
- `imported_baseline_summary.csv`;
- `paper_comparison_macro_f1.csv`.

Dataset naming differences:

- source `spam` corresponds to this repository's `spambase`;
- source results include `wine`, which is not part of the current ranking
  experiment config;
- source results do not include `churn`, because the imported FCALC/IPSKNN
  baselines are numerical-data methods.
