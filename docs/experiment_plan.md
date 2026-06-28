# Experiment Plan

This file describes the experimental plan for the paper. It should be updated
whenever a dataset, method, metric, or baseline decision changes.

## Objective

Evaluate whether ranking and pruning LazyFCA local pattern-structure hypotheses
produces compact explanations while preserving or improving classification
performance.

The central object of study is not a new black-box classifier. It is the
aggregation stage of LazyFCA-style local classification.

## Hypotheses

H1. All-hypotheses LazyFCA aggregation is noisy and can be harmful.

H2. A global ranking over hypotheses from all classes can remove many harmful
or redundant hypotheses.

H3. Small top-k subsets can often match or exceed deterministic/randomized
FCALC LazyFCA-family baselines on at least some datasets.

H4. Query-locality-aware metrics should be especially useful for compact local
explanations.

H5. Singleton hypotheses with `tp=1` can contain local prototype-like evidence,
but high singleton rates require careful interpretation.

## Experiment Runner

Main runner:

```bash
python3 experiments/run_experiments.py
```

Config:

```text
experiments/config.yaml
```

Default output root:

```text
experiments/results/<run_name>/
```

Important CLI options:

```bash
--run-name NAME
--datasets rice sonar
--metrics query_weighted_log_odds_ratio precision
--methods global_topk random_topk
--seeds 1998 1999 2000
--force
--smoke
```

The runner is incremental. Existing chunks are skipped unless `--force` is
used.

Default seeds are `1998..2007`, matching the imported baseline paper splits.
This allows the ranked LazyFCA experiment to be compared against the preserved
FCALC/randomized FCALC/IPSKNN macro-F1 outputs without rerunning those expensive
baselines.

Primary full run:

```bash
python3 experiments/run_experiments.py --run-name ranking_macro_f1_10splits
```

After the ranking run, import the preserved baselines and build comparison
tables:

```bash
python3 experiments/import_baseline_results.py --run-name ranking_macro_f1_10splits
```

## Execution Model

Do not cache full explanations. Full LazyFCA explanations can exceed memory
limits on larger datasets.

The runner streams one query at a time:

1. fit LazyFCA on one train split;
2. explain one test query;
3. compute rankings and predictions for all missing chunks;
4. update compact prediction stores;
5. update diagnostics;
6. discard the explanation;
7. continue to the next query.

Only compact result CSVs are persisted.

## Current Methods

### Vanilla LazyFCA

Counts all hypotheses. Included as a reference. It is often weak, which is part
of the motivation.

### Global Top-k Ranked LazyFCA

For each query:

1. flatten all class-specific hypotheses into one pool;
2. rank by a metric;
3. retain top `k`;
4. class score is retained hypothesis count;
5. ties are broken by summed metric score, class prior, lower label.

### Random Top-k

Randomly orders hypotheses and keeps top `k`. This tests whether ranking is
better than arbitrary pruning.

When random top-k has multiple random repeats inside the same train/test split,
summaries first average those repeats within each seed. Mean, standard
deviation, and confidence interval are then computed over split seeds, so random
repeats are not treated as independent dataset splits.

## Imported Baseline Methods

### Deterministic FCALC/LazyFCA

The main vanilla LazyFCA-family baseline is deterministic FCALC/LazyFCA from
the user's related repository. Aggregation is selected by cross-validation on
the training split, so the paper should describe it as "deterministic FCALC
with CV-selected aggregation," not as the weak raw-count implementation.

Historical results for this baseline have been imported under
`experiments/imported_baselines/`, so it is disabled in the default config to
avoid rerunning the expensive grid unless explicitly requested.

### Randomized FCALC/LazyFCA

The randomized FCALC/LazyFCA baseline intersects the query with random
subsamples/batches instead of only individual training objects. Aggregation,
number of iterations, and subsample size are selected on the training split.

This is the strongest LazyFCA-family baseline in the current runner.

Historical results for this baseline have also been imported. They use split
seeds `1998..2007` and report macro-F1.

### IPS-KNN

IPS-KNN is implemented as an optional numeric-only baseline. It is disabled by
default because it changes the classifier family and may be better framed as
related work or diagnostic comparison.

All imported baselines are numeric-only; `churn` is absent/skipped for them.

## Datasets

Current default datasets:

| Name | Path | Target | Drop | Type |
| --- | --- | --- | --- | --- |
| breast_cancer | `datasets/breast_cancer.csv` | `diagnosis` | `id` | binary |
| churn | `datasets/churn_pr.csv` | `Class` | none | binary |
| page_blocks | `datasets/page_blocks.csv` | `class` | none | multi-class |
| parkinsons | `datasets/parkinsons.csv` | `class` | none | binary |
| rice | `datasets/rice_pr.csv` | `Class` | none | binary |
| sonar | `datasets/sonar.csv` | `class` | none | binary |
| spambase | `datasets/spambase.csv` | `class` | none | binary |
| waveform | `datasets/waveform.csv` | `class` | none | binary in local file |
| ionosphere | `datasets/ionosphere.data` | `class` | none | binary |
| image_segmentation | `datasets/image_segmentation.csv` | `class` | none | multi-class |
| vehicle | `datasets/vehicle.csv` | `class` | none | multi-class |
| glass | `datasets/glass.data` | `type` | `id` | multi-class |

## Preprocessing

For every dataset and seed:

- stratified 80/20 train/test split;
- target label-encoded as `0..C-1`;
- numeric columns passed directly;
- categorical columns one-hot encoded as boolean;
- no numeric scaling;
- same k-grid and metric set across datasets unless config changes.

## K Grid

Plot and summarize `k` increasing from 1 upward.

Config currently uses:

- low values: `1..10`;
- geometric values from `12` to full training size;
- full training size included.

Vanilla LazyFCA is a horizontal reference, not the start of the top-k curve.

## Metrics

Primary paper metrics should probably include:

- `precision`;
- `log_odds_ratio`;
- `query_similarity`;
- `query_weighted_precision`;
- `query_weighted_log_odds_ratio`;
- `wracc`;
- possibly `interval_tightness` and `delta_stability` if they remain strong.

The full config may compute all metrics for discovery, but the paper should
not discuss every metric in equal detail. Group them into families and report
the strongest or most interpretable representatives.

## Evaluation Metrics

Primary classification score:

- macro-F1 for both binary and multi-class datasets.

This changed from the earlier positive-class-F1 convention for binary datasets
because the imported FCALC/randomized FCALC/IPS-KNN baseline outputs store
macro-F1 but do not store predictions or confusion matrices. Future paper
comparisons should therefore use macro-F1 as the common metric.

For each dataset/method/metric/`k`, the runner reports:

- mean over split seeds;
- sample standard deviation over split seeds;
- 95% confidence-interval half-width using the Student-t critical value;
- number of split seeds in `runs`;
- number of raw result rows in `repeat_rows`.

Also store:

- accuracy;
- precision;
- recall;
- macro-F1;
- weighted-F1;
- AUC-ROC when available;
- confusion matrix;
- retained hypothesis count;
- compression ratio.

Compactness summary:

- best F1;
- best `k`;
- smallest `k` within 1% of best F1;
- smallest `k` within 3% of best F1;
- smallest `k` within 5% of best F1;
- compression ratio at best `k`;
- retained mean.

Diagnostics:

- class counts;
- train/test sizes;
- numeric/categorical/encoded feature counts;
- mean/min/max generated classifiers per query;
- singleton `tp=1` rate;
- `fp=0` rate.

## Output Files

Under `experiments/results/<run_name>/`:

- `chunks/`
- `diagnostics/`
- `manifest.jsonl`
- `topk_results.csv`
- `vanilla_lazyfca.csv`
- `summary_by_dataset_metric.csv`
- `compactness_summary.csv`
- `dataset_diagnostics.csv`
- `topk_plot_data.csv`
- `imported_baseline_results.csv`
- `imported_baseline_summary.csv`
- `paper_comparison_macro_f1.csv`
- `plots/`

Do not commit result folders to GitHub by default.

The imported baseline files are created by
`experiments/import_baseline_results.py`, not by the main ranking runner.

## Validation Checks

After a full run, verify:

```text
expected default chunks = datasets * seeds * (metric_count global + 1 random)
```

With 12 datasets, 10 seeds, 36 metrics, and default methods
`global_topk + random_topk`:

```text
12 * 10 * (36 + 1) = 4440 chunks
```

If `current_vanilla_lazyfca` is explicitly enabled, add `12 * 10 = 120`
chunks/rows.

Expected top-k rows:

For each dataset/seed:

- global top-k: `36 metrics * 34 k = 1224`;
- random top-k: `5 repeats * 34 k = 170`;
- default total: `1394` rows.

For 12 datasets and 10 seeds:

```text
12 * 10 * 1394 = 167280 rows
```

If `current_vanilla_lazyfca` is enabled, add:

```text
12 * 10 = 120
```

Expected compactness rows:

```text
12 datasets * (36 global metrics + 1 random metric) = 444
```

Expected diagnostics rows:

```text
12 datasets * 10 seeds = 120
```

Expected diagnostics rows:

```text
12 datasets * 10 seeds = 120
```

## Known Findings From Earlier Full Run

The 8-dataset full run was internally consistent:

- 55,800 `topk_results.csv` rows;
- 10,064 summary rows;
- 296 compactness rows;
- 40 vanilla rows;
- no missing primary F1 or accuracy.

Vanilla LazyFCA often had very poor binary performance under the older
positive-class-F1 diagnostic because it collapsed to one class under
all-hypotheses counting. This supports the motivation but should be explained
carefully; the paper comparison should use macro-F1.

Some observed best metrics in the earlier 8-dataset run included:

- `delta_stability`;
- `interval_tightness`;
- `query_weighted_precision`;
- `lift`.

Do not overinterpret these before the 12-dataset final analysis.

## Analysis Plan

1. Check run completeness and diagnostics.
2. Compare best metric per dataset.
3. Compare primary selected metrics across datasets.
4. Measure compactness at smallest `k` within 1%, 3%, and 5% of best F1.
5. Compare global top-k against random top-k.
6. Compare global top-k against vanilla LazyFCA.
7. Inspect singleton rates and whether high singleton datasets favor
   locality/tightness metrics.
8. Identify 3-5 metrics worth discussing in the paper.
9. Produce compact tables for the paper.
10. Generate plots:
    - one top-k curve per dataset for selected metrics;
    - one compactness summary plot;
    - one vanilla-vs-best ranked comparison;
    - optional singleton-rate diagnostic plot.

## Baseline Extension Plan

If copying the newly published code:

1. Put it in a separate folder, not directly into the current implementation.
2. Add a README noting source, citation, license/permission, and modifications.
3. Create wrapper functions with a common interface:
   - `fit(X_train, y_train)`;
   - `predict(X_test)`;
   - optional `explain(X_test)`;
   - optional compactness outputs.
4. First run on one or two small datasets.
5. Compare runtime, F1, and explanation size.
6. Decide whether to include in paper.

Recommended baseline priority:

1. imported deterministic FCALC/LazyFCA;
2. imported randomized FCALC/LazyFCA;
3. imported IPS-KNN as diagnostic or related-work comparison;
4. ranked randomized LazyFCA only as future work or an extra experiment.

## Stop Conditions

The paper can be considered experimentally sufficient if:

- 12 datasets complete over seeds `1998..2007`;
- random top-k and global top-k are complete;
- imported FCALC/randomized FCALC/IPSKNN comparison files are generated;
- compactness summaries are stable;
- at least two or three ranking metrics consistently beat or match random top-k
  and are competitive with imported LazyFCA-family baselines at much smaller
  explanation budgets;
- limitations around IPS-KNN/randomized LazyFCA are honestly stated.
