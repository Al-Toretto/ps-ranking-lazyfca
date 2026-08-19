# Paper Result Artifacts

This directory contains compact artifacts for checking the reported top-k
ranking experiment.

The artifacts are generated from the completed run
`experiments/results/ranking_macro_f1_10splits/` with:

```bash
python3 experiments/export_paper_artifacts.py
python3 experiments/generate_paper_figures.py
```

The reported dataset suite contains eleven numerical datasets:

`breast_cancer`, `ionosphere`, `parkinsons`, `rice`, `sonar`, `spambase`,
`waveform`, `vehicle`, `page_blocks`, `glass`, and `image_segmentation`.

All predictive summaries use macro-F1 over seeds `1998..2007`.

## Main Files

- `summary_by_dataset_metric.csv`: mean/std/95% CI for every ranked metric and
  retained-candidate budget.
- `topk_plot_data.csv`: compact plot-ready version of the summary.
- `dataset_diagnostics.csv`: split sizes, class counts, feature counts, and
  candidate-pool diagnostics.
- `imported_baseline_summary.csv`: imported FCALC, FCALC(rand.), IPS-KNN, and
  classical ML baseline summaries.
- `paper_comparison_macro_f1.csv`: best compact-budget ranked rows plus
  imported baselines.

## Table-Ready Files

- `table_qwp_fixed_k.csv`: fixed `k = 1, 3, 5, 10` query-weighted precision
  results plus the post-hoc compact-budget upper bound.
- `table_compact_budget_comparison.csv`: query-weighted precision top-k,
  random top-k, FCALC, FCALC(rand.), and IPS-KNN.
- `table_classical_context.csv`: query-weighted precision top-k and classical
  ML baselines.
- `table_qwp_compactness.csv`: compactness summary for query-weighted
  precision.

The two `diagnostic_metric_screening_*.csv` files are mechanical screenings
from the enabled metrics. They are useful for auditing but are not intended to
be a byte-for-byte recreation of the manuscript's manually grouped
representative metric table.

## Retained-Candidate Example

- `retained_candidates_rice_seed1998_query122_top5.csv`
- `retained_candidate_rice_seed1998_query122_rank5_interval.csv`
- `retained_candidate_example_metadata.json`

These files reproduce the Rice retained-candidate example used to illustrate
the interval evidence.
