# Documentation Index

Use this folder as the project memory for parallel AI chats.

## Start Here

- `project_context.md`
  Full project context: FCA/pattern-structure background, LazyFCA mechanics,
  motivation, ranking method, metrics, datasets, runtime constraints, results
  interpretation, and baseline strategy.

- `experiment_plan.md`
  Full experiment plan: runner design, config options, implemented methods,
  candidate future methods, k-grid, outputs, validation checks, analysis plan,
  and stop conditions.

- `paper_plan.md`
  Full paper-writing plan: positioning, contribution claims, section-by-section
  content, related-work needs, result narrative, threats to validity, and how
  to discuss IPS-KNN/randomized LazyFCA.

## Which Files To Send To Another Chat

For experiment-code work:

- `docs/project_context.md`
- `docs/experiment_plan.md`
- `experiments/config.yaml`
- `experiments/run_experiments.py`
- `experiments/import_baseline_results.py`

For paper writing:

- `docs/project_context.md`
- `docs/paper_plan.md`
- `paper/main.tex`
- `paper/mybib.bib`

For literature review or reference search:

- `paper/context_for_ai_assistants.txt`
- `docs/paper_plan.md`
- `paper/gpt_deep_research_results.md`
- `paper/my_other_papers/`

For randomized LazyFCA or IPS-KNN integration:

- `docs/project_context.md`
- `docs/experiment_plan.md`
- `NOTICE.md`

## Important Positioning Reminder

The paper is not trying to prove that ranked LazyFCA beats every compact
pattern-structure classifier. The central claim is narrower and safer:
LazyFCA generates many local interpretable hypotheses, naive aggregation can be
noisy, and global pooled ranking can select compact subsets that often improve
or preserve predictive performance.

The expensive deterministic FCALC, randomized FCALC, and IPS-KNN results are
imported from the related repository and compared through macro-F1 over the
same seeds `1998..2007`. Ranking on randomized hypotheses is a natural future
extension, but it is not part of the main proposed method unless explicitly
added later.

The related FCALC/randomized FCALC/IPSKNN paper by the same author has been
accepted and can be cited. Its PDF/LaTeX files, plus the earlier IPS-KNN
position paper, are stored locally under `paper/my_other_papers/`. Rephrase
carefully and cite them; do not copy manuscript text verbatim.
