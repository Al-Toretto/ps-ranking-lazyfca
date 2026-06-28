# Baseline Implementations

Use this folder for external or separately developed methods that are compared
against the ranked LazyFCA experiments.

Current local package:

```text
baselines/
  interval_lazy_methods/
```

It contains cleaned code from the user's related interval-pattern
lazy-classification repository:

- deterministic FCALC/LazyFCA;
- randomized FCALC/LazyFCA;
- IPS-KNN.

The wrappers expose:

```python
fit(X_train, y_train)
predict(X_test)
get_params_used()
get_compactness()
```

The nested source repository, copied datasets, old outputs, and old experiment
scripts were removed. See `interval_lazy_methods/SOURCE.md` for provenance.
