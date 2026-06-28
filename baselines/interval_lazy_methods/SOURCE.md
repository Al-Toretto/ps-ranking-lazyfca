# Source Notice

This package contains selected code adapted from the user's related repository:

`git@github.com:Al-Toretto/Interpretable-Lazy-Classification-for-Numerical-Data-using-Interval-Pattern-Structures.git`

Copied from commit:

`6cf605a full output example`

The source repository belongs to the same author/user and accompanies a
separate paper on interpretable lazy classification with interval pattern
structures.

Local changes in this repository:

- removed the nested git repository, datasets, outputs, and old experiment
  scripts;
- kept only the algorithm code needed for FCALC/LazyFCA, randomized FCALC, and
  IPS-KNN baselines;
- wrapped the algorithms behind a shared `fit` / `predict` /
  `get_params_used` / `get_compactness` interface;
- adapted imports so the code works as a local package under `baselines/`;
- made experiment splitting, result chunks, and summaries controlled by this
  repository's `experiments/run_experiments.py`.

