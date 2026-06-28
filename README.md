# Ranked LazyFCA Pattern Hypotheses

This repository contains the cleaned working code and paper draft for a PhD
paper on ranking and pruning LazyFCA pattern-structure hypotheses for compact
interpretable classification.

Working title:

`Compact Local Classification with Ranked Pattern-Structure Hypotheses`

## What This Project Studies

LazyFCA avoids constructing a full concept or pattern-concept lattice by
generating local hypotheses for each query object. In interval pattern
structures, these hypotheses are interpretable interval descriptions over
features. The problem is that one query can generate many hypotheses, and
counting all of them can be noisy, harmful, and difficult to interpret.

The main method in this repository is global pooled top-k ranking:

1. Generate LazyFCA hypotheses for every class.
2. Put all class-specific hypotheses into one common pool.
3. Rank the pool by an importance metric.
4. Keep only the top `k` hypotheses.
5. Predict by retained class counts, with score/prior/label tie-breaks.

The paper asks whether a small ranked subset can preserve or improve macro-F1
while making the local explanation much smaller.

## Repository Map

- `lazyfca/`
  Core LazyFCA implementation, explanations, classifiers, metrics, and metric
  calculators.

- `experiments/`
  Config-driven incremental experiment runner.

- `experiments/config.yaml`
  Main editable experiment configuration. It controls datasets, seeds, metrics,
  methods, output paths, and k-grid options.

- `paper/`
  LaTeX draft, bibliography, and AI-assistant context for literature review and
  writing.

- `docs/project_context.md`
  Full technical/research handoff. Read this first in new chats.

- `docs/README.md`
  Short index explaining which context files to send to different AI chats.

- `docs/experiment_plan.md`
  Full experimental plan, including methods, outputs, validation checks, and
  baseline-extension strategy.

- `docs/paper_plan.md`
  Full paper-writing plan, including positioning, section-by-section content,
  claims to make, claims to avoid, and how to discuss baselines.

## Data

Dataset files are expected locally under `datasets/`, using the paths in
`experiments/config.yaml`. They are ignored by git by default because
their redistribution status should be checked separately.

Current configured datasets:

- `breast_cancer`
- `churn`
- `page_blocks`
- `parkinsons`
- `rice`
- `sonar`
- `spambase`
- `waveform`
- `ionosphere`
- `image_segmentation`
- `vehicle`
- `glass`

## Running Experiments

Smoke run:

```bash
python3 experiments/run_experiments.py --smoke --run-name smoke
```

Single dataset:

```bash
python3 experiments/run_experiments.py --run-name trial_rice --datasets rice --seeds 1998
```

Full configured run:

```bash
python3 experiments/run_experiments.py --run-name ranking_macro_f1_10splits
```

Import preserved FCALC/randomized FCALC/IPS-KNN baselines and build the paper
comparison table after the ranking run:

```bash
python3 experiments/import_baseline_results.py --run-name ranking_macro_f1_10splits
```

The runner is incremental. Completed chunks are skipped unless `--force` is
used.

Results are written under:

```text
experiments/results/<run_name>/
```

This directory is ignored by git.

Important output files after both commands:

- `topk_results.csv`: per-split ranking results;
- `summary_by_dataset_metric.csv`: split-level mean/std/95% CI by dataset,
  method, metric, and `k`;
- `topk_plot_data.csv`: compact plot-ready version of the summary;
- `imported_baseline_results.csv`: preserved FCALC/randomized FCALC/IPSKNN
  rows mapped to this repository's dataset names;
- `imported_baseline_summary.csv`: mean/std/95% CI for imported baselines;
- `paper_comparison_macro_f1.csv`: ranking best-per-metric rows plus imported
  baseline summaries.

## Important Research Notes

The old raw-count all-hypotheses implementation is kept as
`current_vanilla_lazyfca`, but it is disabled by default because it is often too
weak to serve as the main paper baseline.

`random_topk` is a sanity baseline for arbitrary pruning.

The main LazyFCA-family baselines are now imported from the user's related
interval-pattern lazy-classification code:

- `fcalc_deterministic`: deterministic FCALC/LazyFCA with aggregation selected
  by cross-validation on the training split.
- `fcalc_randomized`: randomized FCALC/LazyFCA with aggregation and sampling
  parameters selected on the training split.
- `ips_knn`: optional compact external interval-pattern kNN baseline, disabled
  by default.

IPS-KNN may outperform the ranked LazyFCA method. That does not invalidate the
paper, because IPS-KNN changes the classifier family and uses a single compact
reason. This paper studies how to improve hypothesis selection inside LazyFCA
aggregation. IPS-KNN can be used as related work, a diagnostic comparison, or
future integration depending on results.

The imported baselines are numeric-only. The runner skips `churn` for those
methods because that dataset has raw categorical predictors.

Reusable historical baseline outputs are preserved under
`experiments/imported_baselines/`. They use split seeds `1998..2007` and store
macro-F1, selected parameters, runtime, and compactness summaries. Because
those files do not include predictions or confusion matrices, the paper-level
comparison metric is macro-F1 for both binary and multi-class datasets.

The related paper by the same author that implements deterministic FCALC,
randomized FCALC, and IPS-KNN has been accepted and can be cited. Local
PDF/LaTeX copies of that manuscript and the earlier IPS-KNN position paper are
under `paper/my_other_papers/`; use them for accurate citation and rephrase any
technical descriptions.

## Code Provenance

This is a cleaned research repository derived from earlier LazyFCA work. If
additional code is copied from another paper or repository, keep it separated
under a clearly named folder, preserve its license/citation information, and
document any modifications.
