# Top-k Ranking of Lazy Interval-Pattern Classifier Candidates

This repository contains the code, configuration, and compact result artifacts
for experiments on ranking and pruning local interval-pattern classifier
candidates.

The method keeps the standard source-query candidate pool used in lazy
interval-pattern classification, ranks candidates from all classes in one
global pool, retains only the top `k`, and predicts by majority vote among the
retained candidates with deterministic tie-breaks.

## Repository Layout

- `lazyfca/`: LazyFCA-style source-query candidate generation, metric
  computation, and explanations.
- `experiments/run_experiments.py`: incremental experiment runner for ranked
  top-k and random top-k.
- `experiments/config.yaml`: paper-level configuration: eleven numerical
  datasets, seeds `1998..2007`, macro-F1, and the retained-candidate grid.
- `experiments/import_baseline_results.py`: imports preserved FCALC,
  FCALC(rand.), IPS-KNN, and classical ML baseline summaries.
- `experiments/export_paper_artifacts.py`: exports compact public summaries and
  retained-candidate examples.
- `experiments/generate_paper_figures.py`: regenerates the two plotted
  macro-F1 figures from `results/paper/topk_plot_data.csv`.
- `experiments/imported_baselines/`: preserved baseline result summaries from
  the related interval-pattern benchmark.
- `baselines/interval_lazy_methods/`: cleaned FCALC, FCALC(rand.), and IPS-KNN
  implementation code kept for provenance and optional reruns.
- `datasets/`: local dataset location. Data files are not tracked by default.
- `results/paper/`: compact public result artifacts used to check the reported
  tables, examples, and figures.

The manuscript source files and private writing notes are intentionally not
tracked in this public experiment repository.

## Installation

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Data

The paper-level run uses eleven public numerical datasets:

`breast_cancer`, `ionosphere`, `parkinsons`, `rice`, `sonar`, `spambase`,
`waveform`, `vehicle`, `page_blocks`, `glass`, and `image_segmentation`.

Place the dataset files under `datasets/` using the filenames listed in
`datasets/README.md` and `experiments/config.yaml`. The data files are ignored
by git so their redistribution terms can be handled separately.

The optional `churn` entry remains in the config for local experiments, but it
is disabled by default and is not part of the reported numerical benchmark.

## Existing Result Artifacts

The compact public artifacts are in:

```text
results/paper/
```

Useful files:

- `summary_by_dataset_metric.csv`: macro-F1 mean/std/95% CI for all ranked
  metrics and all `k` values.
- `topk_plot_data.csv`: plot-ready subset of the summary.
- `table_qwp_fixed_k.csv`: query-weighted precision at `k = 1, 3, 5, 10`.
- `table_compact_budget_comparison.csv`: query-weighted precision top-k,
  random top-k, FCALC, FCALC(rand.), and IPS-KNN.
- `table_classical_context.csv`: compact ranked method and classical ML
  baselines.
- `table_qwp_compactness.csv`: compactness summary for query-weighted
  precision.
- `retained_candidates_rice_seed1998_query122_top5.csv`: retained-candidate
  example.
- `figures/`: regenerated plotting outputs.

## Quick Smoke Test

After placing the datasets, run a small end-to-end check:

```bash
python3 experiments/run_experiments.py \
  --smoke \
  --run-name smoke_reviewer_check \
  --datasets rice \
  --seeds 1998 \
  --metrics query_weighted_precision \
  --methods global_topk random_topk
```

The runner is incremental. Existing chunks are skipped unless `--force` is
used.

## Full Reproduction

The full ranked experiment can be computationally expensive. It streams one
query at a time to avoid storing full explanations in memory.

```bash
python3 experiments/run_experiments.py \
  --run-name ranking_macro_f1_10splits
```

Then import preserved baseline summaries and build the combined comparison:

```bash
python3 experiments/import_baseline_results.py \
  --run-name ranking_macro_f1_10splits
```

Export compact public artifacts:

```bash
python3 experiments/export_paper_artifacts.py
```

Regenerate figures:

```bash
python3 experiments/generate_paper_figures.py
```

## Reported Protocol

- Ten stratified 80/20 train-test splits.
- Split seeds: `1998, ..., 2007`.
- Primary predictive metric: macro-F1 for binary and multi-class datasets.
- Retained-candidate grid: `k = 1, ..., 10` plus 24 geometrically spaced
  values from 12 to the full training-set size.
- Random top-k uses five random repeats inside each split; repeats are averaged
  before split-level uncertainty is computed.
- Result tables report mean and 95% confidence-interval half-width over split
  seeds.

## Baselines

The imported baseline summaries include:

- deterministic FCALC/LazyFCA with CV-selected aggregation;
- randomized FCALC/LazyFCA with CV-selected aggregation and sampling
  parameters;
- IPS-KNN;
- kNN, SVM, Random Forest, XGBoost, and other classical baselines from the
  same repeated benchmark.

The preserved imported results use the same split seeds and macro-F1 reporting
protocol. The raw baseline prediction vectors were not preserved, so the common
comparison metric is macro-F1.
