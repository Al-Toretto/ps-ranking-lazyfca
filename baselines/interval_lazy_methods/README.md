# Interval Lazy Method Baselines

This package provides pattern-structure baselines for the ranked LazyFCA
experiments:

- `FCalcBaseline(randomized=False)`: deterministic FCALC/LazyFCA with
  aggregation selected by cross-validation on the training split.
- `FCalcBaseline(randomized=True)`: randomized FCALC/LazyFCA with aggregation,
  number of iterations, and subsample size selected on the training split.
- `IPSKNNBaseline`: compact interval-pattern kNN baseline.

These baselines are numeric-only. The experiment runner skips datasets with raw
categorical predictors for these methods.

