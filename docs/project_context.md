# Full Project Context

This file is the main handoff document for future chats working on this
repository. Read it before changing experiments, paper text, baselines, or
analysis code.

## Project Identity

The project is a PhD research paper in machine learning, artificial
intelligence, Formal Concept Analysis (FCA), and pattern structures.

Working title:

`Compact Local Classification with Ranked Pattern-Structure Hypotheses`

Research topic:

`Pattern Structures for Interpretable Machine Learning`

The paper studies whether LazyFCA-style local classification can be made more
accurate and more interpretable by ranking and pruning the many local
pattern-structure hypotheses generated for each query object.

## Theoretical Background

Formal Concept Analysis starts from a formal context `(G, M, I)` where `G` is a
set of objects, `M` is a set of attributes, and `I` tells which object has which
attribute. FCA derives formal concepts. A formal concept is a maximal set of
objects sharing a maximal set of attributes. Concepts are ordered by
generality/specificity and form a concept lattice.

Pattern structures generalize FCA from binary attributes to complex
descriptions. A pattern structure is usually written as `(G, (D, sqcap),
delta)`, where:

- `G` is the set of objects;
- `D` is a description space;
- `sqcap` is a meet or similarity operation on descriptions;
- `delta` maps each object to its description.

This makes FCA tools applicable to numerical intervals, sets, sequences, graphs,
relational descriptions, and other complex data types.

In interval pattern structures, numerical objects are described by intervals.
For two scalar values `x` and `s`, their common interval description is:

`[min(x, s), max(x, s)]`

For multiple numeric features, this becomes a hyper-rectangle. This is
interpretable because each feature receives an explicit value interval.

## LazyFCA Classification

Full concept or pattern-concept lattice construction can be expensive. LazyFCA
avoids constructing the whole lattice and instead classifies each query object
locally.

For a query object `q` and a class `c`:

1. Take each training object `s` of class `c`.
2. Intersect or meet `q` and `s`.
3. The result is a local pattern hypothesis for class `c`.
4. Evaluate the hypothesis against supporters from class `c` and opposers from
   all other classes.
5. Repeat for all classes.
6. Aggregate the generated hypotheses to predict the query label.

Each hypothesis has:

- a predicted class label;
- an interpretable pattern description;
- coverage values such as `tp`, `fp`, `tn`, `fn`;
- optional geometric and query-local metrics.

Important caveat:

Every generated hypothesis is created from a query and a same-class source
training object. Therefore `tp >= 1` by construction. A hypothesis with `tp=1`
is not necessarily invalid, but it should be interpreted as a singleton
source-query hypothesis, not as a broadly generalized multi-object rule.

## Problem Motivation

LazyFCA avoids full lattice construction, but it can still generate many local
hypotheses per query. Even if every individual hypothesis is interpretable, the
final explanation can become too large when hundreds or thousands of hypotheses
vote.

Naive all-hypotheses LazyFCA aggregation can also be harmful. Many generated
hypotheses may be:

- too specific;
- singleton-only;
- redundant;
- broad but misleading;
- biased toward the majority class;
- pure only because the interval is narrow;
- harmful when counted equally with better hypotheses.

This is why vanilla LazyFCA is included as a reference, but it should not be
treated as a gold-standard baseline. In the current experiments, vanilla
LazyFCA often collapses toward one class under all-hypotheses counting. Earlier
binary diagnostics showed this as `0` positive-class F1 on some datasets; the
paper-level comparison now uses macro-F1 for all datasets.

## Core Paper Idea

For each query:

1. Generate all class-specific LazyFCA hypotheses.
2. Put hypotheses from all classes into one common pool.
3. Rank the whole pool by a hypothesis-importance metric.
4. Keep only the top `k` hypotheses.
5. Predict by counting retained hypotheses per class.
6. If class counts tie, break ties by summed ranking score, then training class
   prior, then lower encoded class label.

This is called global pooled top-k ranking.

The global pool is important. It lets hypotheses from all classes compete. This
is different from keeping the top `k` hypotheses per class. A per-class quota
can force harmful hypotheses from a class to remain. A common pool lets the
ranking function delete harmful hypotheses regardless of class.

Main research question:

Can a small top-k subset of ranked local pattern-structure hypotheses preserve
or improve predictive quality while making the local explanation much more
compact?

## Intended Contribution

The contribution is not state-of-the-art black-box accuracy.

The realistic contribution is:

- identify the noisy aggregation problem in LazyFCA;
- propose global pooled ranking and compact top-k selection;
- compare many hypothesis-importance metrics systematically;
- measure both predictive performance and compactness;
- analyze singleton source-query hypotheses;
- discuss whether locality-aware metrics improve compact local explanations.

The paper should be positioned as an interpretable ML / pattern-structure /
FCA improvement paper, suitable for a modest journal target.

## Metrics

Metrics are implemented in:

- `lazyfca/metrics.py`
- `lazyfca/calculators.py`

The experiment config can enable or disable metrics in:

- `experiments/config.yaml`

Important metric families:

- contingency counts: `tp`, `fp`, `tn`, `fn`;
- purity: `precision`, `error_rate`;
- coverage/support: `support`, `supporter_opposer_ratio`;
- combined purity-support: `precision_log_tp`, `precision_sqrt_tp`;
- rule-learning/statistical: `lift`, `wracc`, `chi_squared`, `g_test`,
  `information_gain`, `gini_gain`, `matthews_correlation`, `youdens_j`;
- interval geometry: `interval_tightness`, `description_volume`,
  `simplicity_prior`;
- query locality: `query_binary_similarity`, `query_numeric_similarity`,
  `query_similarity`;
- hybrid locality-purity: `query_weighted_precision`,
  `query_weighted_log_odds_ratio`, `query_weighted_wracc`, and variants;
- FCA-inspired robustness: `stability`, `robustness`, `delta_stability`.

Earlier exploratory work suggested that `log_odds_ratio`,
`query_similarity`, `query_weighted_precision`, and
`query_weighted_log_odds_ratio` may be promising, but systematic experiments
are needed before making claims.

Note on naming:

The implemented `log_odds_ratio` behaves like a smoothed support-sensitive
purity score. In the draft, describe the exact formula used instead of relying
on the conventional statistical meaning of "log odds ratio" unless the
implementation is revised.

## Experiment Framework

Main files:

- `experiments/config.yaml`
- `experiments/run_experiments.py`

The runner is config-driven and incremental. It writes result chunks per
dataset, seed, method, and metric. If a chunk exists, it is skipped unless
`--force` is used.

The runner streams one query at a time. It does not cache full LazyFCA
explanations because explanation caching caused extreme memory use in earlier
runs, exceeding 100 GB on a larger dataset. Streaming computes the needed
metrics for one query, updates compact prediction stores, discards the
explanation, and moves to the next query.

CLI examples:

```bash
python3 experiments/run_experiments.py --smoke --run-name smoke
python3 experiments/run_experiments.py --run-name trial_rice --datasets rice --seeds 1998
python3 experiments/run_experiments.py --run-name ranking_macro_f1_10splits
python3 experiments/import_baseline_results.py --run-name ranking_macro_f1_10splits
python3 experiments/run_experiments.py --run-name ranking_macro_f1_10splits --datasets ionosphere vehicle
python3 experiments/run_experiments.py --run-name ranking_macro_f1_10splits --metrics query_weighted_log_odds_ratio
```

Important behavior:

- adding a new dataset runs only missing chunks for that dataset;
- adding a metric runs only missing metric chunks;
- rerunning an interrupted experiment skips completed chunks;
- chunk files are considered complete only after they are written;
- diagnostics are stored per dataset/seed and combined into
  `dataset_diagnostics.csv`.

## Default Datasets

Original systematic set:

- `breast_cancer`: `datasets/breast_cancer.csv`, target `diagnosis`, drop `id`;
- `churn`: `datasets/churn_pr.csv`, target `Class`;
- `page_blocks`: `datasets/page_blocks.csv`, target `class`;
- `parkinsons`: `datasets/parkinsons.csv`, target `class`;
- `rice`: `datasets/rice_pr.csv`, target `Class`;
- `sonar`: `datasets/sonar.csv`, target `class`;
- `spambase`: `datasets/spambase.csv`, target `class`;
- `waveform`: `datasets/waveform.csv`, target `class`.

Added later:

- `ionosphere`: `datasets/ionosphere.data`, target `class`;
- `image_segmentation`: `datasets/image_segmentation.csv`, target `class`;
- `vehicle`: `datasets/vehicle.csv`, target `class`;
- `glass`: `datasets/glass.data`, target `type`, drop `id`.

Preprocessing:

- target labels are label-encoded as `0..C-1`;
- train/test split is stratified;
- default split is 80/20;
- default seeds are `[1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007]`
  so new ranking runs align with imported baseline results;
- numeric columns are passed directly;
- categorical columns are one-hot encoded as boolean columns;
- numeric scaling is not used.

## Default Methods

Currently implemented in the experiment runner:

1. `current_vanilla_lazyfca`
   - all generated hypotheses are counted by the current implementation;
   - disabled by default;
   - used only as a diagnostic because it often gives very poor F1.

2. `global_topk`
   - all class hypotheses are pooled;
   - ranking metric chooses top `k`;
   - class score is retained hypothesis count;
   - ties use summed ranking score, class prior, lower label.

3. `random_topk`
   - random order top-k;
   - sanity baseline for whether ranking is better than arbitrary pruning.

4. `fcalc_deterministic`
   - deterministic FCALC/LazyFCA imported from the user's related paper;
   - aggregation is selected by cross-validation on the training split;
   - this is the main vanilla LazyFCA-family baseline;
   - disabled by default because historical macro-F1 results are imported.

5. `fcalc_randomized`
   - randomized FCALC/LazyFCA imported from the same related code;
   - aggregation, number of iterations, and subsample size are selected by
     cross-validation on the training split;
   - this is the main randomized LazyFCA-family baseline;
   - disabled by default because historical macro-F1 results are imported.

6. `ips_knn`
   - interval pattern structure KNN;
   - uses a compact interval-pattern reason for classification;
   - optional and disabled by default because it changes the classifier family.

The imported baselines are numeric-only. The runner skips `churn` for these
methods because it contains raw categorical predictors.

The paper's proposed method should remain the simple ranked LazyFCA variant:
standard object-query hypothesis generation, global pooled ranking, top-k
retention, and majority vote over retained hypotheses with score/prior/label
tie-breaks. Do not fold FCALC's aggregation hyperparameter search into the
proposed method for this paper; use FCALC and randomized FCALC as baselines.

## IPS-KNN and Randomized LazyFCA Question

The user has a newly published paper/codebase that implements:

- vanilla LazyFCA;
- randomized LazyFCA;
- IPS-KNN.

The useful code has been cleaned into:

```text
baselines/interval_lazy_methods/
```

The original nested repository, copied datasets, and bulky old outputs were
removed. Wrappers expose `fit`, `predict`, `get_params_used`, and
`get_compactness`.

Historical baseline outputs were preserved under:

```text
experiments/imported_baselines/
```

These imported files contain macro-F1, selected parameters, runtimes, and
compactness/size summaries for deterministic FCALC, randomized FCALC, IPS-KNN,
and classical ML baselines over split seeds `1998..2007`. They do not contain
test predictions or confusion matrices, so binary positive-class F1 cannot be
recovered from them. For paper-level comparison with these baselines, use
macro-F1 as the primary metric.

Important citation/source note:

The user's related paper that implements deterministic FCALC/LazyFCA,
randomized FCALC/LazyFCA, and IPS-KNN has been accepted and is expected to be
published soon. It is the paper associated with the repository from which the
baseline code and historical results were taken. The earlier position paper
that introduced IPS-KNN is also locally available.

Local files:

```text
paper/my_other_papers/manuscript.pdf
paper/my_other_papers/manuscript.tex
paper/my_other_papers/position_paper.pdf
paper/my_other_papers/position_paper.tex
```

Use these files as author-owned source material for understanding and citation,
but do not copy text verbatim into the new paper. Rephrase carefully, cite the
accepted paper for FCALC/randomized FCALC/IPSKNN implementation and experimental
baseline details, and cite the position paper for the initial IPS-KNN idea when
historically appropriate. Before submission, make sure `paper/mybib.bib`
contains the final publication metadata for the accepted paper.

Use this command after a ranking run to map the imported baseline rows and
build comparison CSVs:

```bash
python3 experiments/import_baseline_results.py --run-name ranking_macro_f1_10splits
```

Experimental stance:

- First inspect/import IPS-KNN macro-F1 results to understand its performance.
- Do not immediately commit to using IPS-KNN as a main baseline.
- If IPS-KNN beats the ranking method, the paper can still be justified as a
  study of compacting LazyFCA hypothesis aggregation, not as a claim that top-k
  ranking beats all compact local classifiers.
- IPS-KNN can be discussed as related work and as evidence that compact local
  interval-pattern explanations are valuable.
- A limited comparison may be included if it is framed as family-level context:
  "IPS-KNN changes the classifier family, whereas our method improves
  hypothesis selection inside LazyFCA."
- If IPS-KNN is too dominant, it may be safer to mention it in related work and
  future work rather than make it a central baseline.

Randomized LazyFCA is more directly relevant:

- It is still LazyFCA-family.
- It addresses hypothesis specificity at generation time.
- Our ranking addresses hypothesis selection after generation.
- These approaches are complementary.
- A strong experiment would test ranking on randomized LazyFCA hypotheses:
  random generation plus top-k ranking.

Possible method matrix:

| Generation | Selection/Aggregation |
| --- | --- |
| object-query LazyFCA | all hypotheses |
| object-query LazyFCA | global top-k ranked |
| object-query LazyFCA | random top-k |
| randomized/batch LazyFCA | all generated hypotheses |
| randomized/batch LazyFCA | global top-k ranked |
| IPS-KNN | single compact reason |

This matrix may be too much for the first paper if time is limited. Prioritize:

1. global top-k results rerun with seeds `1998..2007` and macro-F1 primary;
2. imported deterministic FCALC/LazyFCA baseline;
3. imported randomized FCALC/LazyFCA baseline;
4. optional imported IPS-KNN diagnostic comparison.

## Paper Draft

Main file:

- `paper/main.tex`

Bibliography:

- `paper/mybib.bib`

The related work section already covers:

- FCA foundations;
- pattern structures;
- LazyFCA and interval pattern structures;
- the credit-scoring randomized LazyFCA paper;
- IPS-KNN;
- associative classification;
- rule ordering and pruning;
- WRAcc and subgroup discovery;
- compact rule sets/rule lists;
- LIME, local explanations, prototypes.

Do not upload downloaded PDFs to a public repository unless licensing permits.
The paper PDFs are useful locally but should usually be ignored by git.

## Current Paper Claim

Safe claim:

Ranked global top-k selection can make LazyFCA local explanations dramatically
more compact and can often preserve or improve macro-F1 relative to naive
all-hypotheses aggregation.

Unsafe claim:

This is the best interpretable classifier or beats all baselines.

Likely final story:

LazyFCA generates interpretable local hypotheses but naive aggregation is noisy.
Ranking makes the explanation budget explicit. The method is useful when the
goal is to keep the LazyFCA/pattern-structure hypothesis machinery while
reducing harmful hypothesis accumulation.

## Important Result Interpretation Notes

Vanilla LazyFCA often collapses to one class under all-hypotheses counting. In
earlier diagnostics this produced poor binary positive-class F1; under the
current paper convention it should be discussed through macro-F1 and class-level
diagnostics. This is not necessarily a bug; it is part of the motivation.

Plots should show `k` increasing from 1 upward. Vanilla LazyFCA should be a
horizontal reference line, not the starting point of the curve. Starting from
"all hypotheses" and decreasing to `k=1` visually exaggerates the weird jump
from bad vanilla to better compact subsets.

Compactness should be reported with:

- best F1;
- best `k`;
- smallest `k` within 1%, 3%, and 5% of best F1;
- compression ratio;
- singleton `tp=1` rate;
- total classifier count.

## Repository Hygiene

This repository is intended to be a clean personal research repo. Avoid
committing:

- `paper/refs/` PDFs;
- `paper/refs_text/` extracted text;
- `paper/main.pdf`;
- `experiments/results/`;
- `__pycache__/`;
- `.nbc`, `.nbi`, `.pyc`;
- old exploratory notebooks unless deliberately selected;
- large generated plots;
- temporary logs.

If a future chat sees these files tracked, clean them with `.gitignore` and
`git rm --cached` rather than deleting local working copies blindly.

## Next Work Items

1. Finish or verify the full 12-dataset experiment.
2. Analyze which metrics are consistently strong.
3. Produce paper tables and plots from `compactness_summary.csv` and
   `summary_by_dataset_metric.csv`.
4. Run/import the preserved FCALC, randomized FCALC, and IPS-KNN baselines with
   `experiments/import_baseline_results.py`.
5. Decide how prominently IPS-KNN should appear based on imported results.
6. Treat ranked randomized LazyFCA as future work unless there is extra time.
7. Update `paper/main.tex` results and discussion.
8. Clean bibliography metadata.
9. Prepare reproducibility instructions.
