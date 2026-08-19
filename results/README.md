# Result Artifacts

This directory is for compact, public result artifacts used to check the
reported experiments. Large raw experiment runs remain under
`experiments/results/` and are ignored by git.

The paper artifact bundle is generated with:

```bash
python3 experiments/export_paper_artifacts.py
```

It writes filtered result summaries, table-ready CSV files, retained-candidate
examples, and plot-ready data under `results/paper/`.
