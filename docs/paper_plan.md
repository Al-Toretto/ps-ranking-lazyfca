# Paper Plan

This file is the writing and publication plan for the paper. It complements:

- `docs/project_context.md`
- `docs/experiment_plan.md`
- `paper/context_for_ai_assistants.txt`

## Working Title

`Compact Local Classification with Ranked Pattern-Structure Hypotheses`

Alternative titles:

- `Ranking Local Pattern-Structure Hypotheses for Compact LazyFCA Classification`
- `Compact LazyFCA Classification by Global Top-k Hypothesis Ranking`
- `From Noisy Hypothesis Aggregation to Compact Pattern-Structure Explanations`

## Target Positioning

This is not a tier-1 state-of-the-art classifier paper. It is a focused
interpretable machine learning paper about pattern structures, LazyFCA, and
compact local explanations.

Likely venue level: modest journal, possibly Q3.

The paper should be honest and useful:

- LazyFCA is mathematically interpretable.
- But naive all-hypotheses aggregation is noisy.
- Ranking hypotheses makes the explanation budget explicit.
- Compact top-k subsets can improve or preserve F1.
- The method stays inside the LazyFCA / pattern-structure family.

## Core Problem Statement

LazyFCA avoids constructing full pattern concept lattices by classifying each
query locally. However, it may generate a large number of hypotheses per query.
Using all hypotheses equally can harm performance and makes the explanation too
large for practical interpretation.

The paper asks:

Can we rank local pattern-structure hypotheses so that a small top-k subset
retains most of the predictive quality and yields a compact interpretable
explanation?

## Main Contribution Claims

Safe contributions:

1. A global pooled ranking framework for LazyFCA hypotheses where hypotheses
   from all classes compete in one common pool.
2. A systematic comparison of hypothesis-importance metrics for compact
   LazyFCA classification.
3. An empirical accuracy-compactness analysis over multiple tabular datasets.
4. A diagnostic analysis of singleton source-query hypotheses (`tp=1`).
5. Evidence that naive all-hypotheses LazyFCA aggregation can be harmful.

Possible contribution if randomized LazyFCA is added:

6. A comparison between generation-time broadening by randomized LazyFCA and
   post-generation pruning by ranking.
7. A test of whether ranking can further compact randomized LazyFCA.

Do not claim:

- state-of-the-art predictive accuracy;
- superiority to all interpretable classifiers;
- superiority to IPS-KNN unless results genuinely show it;
- that every top-k hypothesis is a generalized rule.

## Paper Structure

### Abstract

Should state:

- LazyFCA avoids full lattice construction but generates many local hypotheses.
- Many hypotheses reduce interpretability and can hurt voting.
- We rank all class hypotheses in a common pool and retain top-k.
- We evaluate multiple ranking metrics across datasets.
- We measure F1 and compactness.
- Results show whether compact ranking improves/preserves performance.

### 1. Introduction

Goals:

- introduce FCA and pattern structures as interpretable mathematical tools;
- explain why full lattices are expensive;
- explain LazyFCA;
- identify the second bottleneck: too many hypotheses and noisy aggregation;
- motivate global pooled ranking;
- present contributions.

Important phrasing:

Each hypothesis may be interpretable, but a large unranked set of hypotheses is
not necessarily an interpretable explanation.

### 2. Background

Subsections:

1. Formal Concept Analysis
2. Pattern Structures
3. Interval Pattern Structures
4. LazyFCA Classification

Need define:

- formal context `(G, M, I)`;
- derivation operators;
- formal concept `(A, B)`;
- concept lattice;
- pattern structure `(G, (D, sqcap), delta)`;
- pattern concept;
- interval meet;
- query-source hypothesis;
- supporters/opposers;
- `tp`, `fp`, `tn`, `fn`.

### 3. Related Work

Already drafted in `paper/main.tex`.

Must cover:

- FCA foundations: Wille; Ganter and Wille.
- Pattern structures: Ganter and Kuznetsov; Kuznetsov; Kaytoue/Buzmakov/Napoli.
- LazyFCA / interval pattern structures:
  - credit scoring paper;
  - randomized/subsample LazyFCA;
  - IPS-KNN.
- Associative classification:
  - CBA;
  - CMAR;
  - CPAR;
  - lazy associative classification;
  - rule ranking and pruning.
- Rule quality:
  - support/confidence;
  - WRAcc;
  - subgroup discovery;
  - concept stability.
- Local explanations:
  - LIME;
  - prototype selection;
  - ProtoPNet or case-based explanations.
- Compact rule models:
  - Bayesian Rule Lists;
  - Bayesian Rule Sets;
  - CORELS;
  - decision sets.

Gap:

Existing work studies local rules, FCA concepts, associative classification,
compact rule sets, and local explanations, but the specific problem of ranking
query-dependent pattern-structure hypotheses from all classes in one pool is
underexplored.

### 4. Method

Subsections:

1. LazyFCA Hypothesis Generation
2. Global Pooled Ranking
3. Prediction from Top-k Hypotheses
4. Ranking Metrics
5. Singleton Hypotheses

Need formalize:

Let `H(q)` be all hypotheses generated for query `q`.

Each hypothesis `h` has:

- class `y(h)`;
- pattern description `d(h)`;
- metric vector `m(h)`;
- ranking score `r(h)`.

Top-k:

`H_k(q) = top_k(H(q), r)`

Prediction:

`ŷ(q) = argmax_c |{h in H_k(q): y(h)=c}|`

Tie-breaking:

1. count;
2. sum of ranking scores;
3. training prior;
4. lower encoded label.

Metrics should be grouped, not listed as 36 unrelated formulas.

Primary metrics to explain in detail:

- precision;
- support;
- WRAcc;
- implemented log-odds-like score;
- query similarity;
- query-weighted precision;
- query-weighted log-odds-like score;
- interval tightness if results justify it;
- delta stability if results justify it.

### 5. Experimental Design

Subsections:

1. Datasets
2. Preprocessing
3. Methods Compared
4. Evaluation Metrics
5. Implementation and Incremental Reproducibility

Datasets:

Use the 12 configured datasets if the final run is complete.

Primary score:

- macro-F1 for both binary and multi-class datasets.

Reason: the reusable deterministic FCALC, randomized FCALC, and IPS-KNN
baseline outputs from the related repository contain macro-F1 but do not
contain predictions/confusion matrices, so binary positive-class F1 cannot be
recovered without rerunning the expensive baselines.

Baselines:

Minimum:

- imported deterministic FCALC/LazyFCA with CV-selected aggregation;
- imported randomized FCALC/LazyFCA with CV-selected aggregation and sampling
  parameters;
- random top-k;
- global top-k by ranking metrics.

Possible extension:

- ranked randomized LazyFCA, only if there is extra time;
- IPS-KNN diagnostic comparison.

The proposed method should not absorb FCALC's aggregation hyperparameter
framework in this paper. The clean contribution is ranking/pruning over the
simple LazyFCA hypothesis pool, where prediction is the majority class among
retained hypotheses with deterministic tie-breaks. FCALC and randomized FCALC
are stronger LazyFCA-family baselines, not variants of the proposed method.

### 6. Results

Main result tables:

- ranked LazyFCA macro-F1 mean +/- 95% CI over seeds `1998..2007`;
- imported FCALC/randomized FCALC/IPSKNN macro-F1 mean +/- 95% CI;
- compactness table with best `k`, smallest `k` within 1/3/5% of best macro-F1,
  retained hypotheses, and compression ratio.

1. Vanilla LazyFCA vs best ranked top-k per dataset.
2. Best or selected metrics per dataset.
3. Compactness summary: smallest k within 1%, 3%, 5% of best.
4. Diagnostics: generated hypotheses and singleton rates.

Main figures:

1. Top-k F1 curves for selected metrics per dataset.
2. Compression ratio or retained hypotheses at near-best F1.
3. Optional metric win-count or rank plot.
4. Optional singleton-rate plot.

Avoid overwhelming the paper with all 36 metrics. Put full results in
supplementary material or repository CSVs.

### 7. Discussion

Points to discuss:

- why vanilla LazyFCA is often bad;
- why global top-k can outperform all-hypotheses aggregation;
- which metric families work best;
- whether locality-aware metrics are robust;
- why interval tightness or delta stability may perform well if confirmed;
- how singleton-heavy datasets behave;
- compactness vs accuracy;
- randomized LazyFCA and IPS-KNN as related but different directions.

Important nuance:

If IPS-KNN beats the method, the discussion should say that IPS-KNN changes the
classification mechanism, whereas this paper studies compacting hypothesis
aggregation inside LazyFCA. The paper can then motivate ranking as useful for
FCA-style hypothesis sets, not as the universal best lazy classifier.

### 8. Threats to Validity

Include:

- limited dataset suite;
- local CSV preprocessing choices;
- class imbalance;
- macro-F1 hides per-class failures unless class-level diagnostics are also
  inspected;
- no full lattice comparison;
- runtime constraints;
- generated hypotheses are query-dependent;
- `tp=1` by construction;
- not all possible baselines included;
- possible sensitivity to metric implementation details.

### 9. Conclusion

Should conclude only what experiments support.

Likely final message:

Ranking and pruning local pattern-structure hypotheses can substantially reduce
LazyFCA explanation size and often improve or preserve F1 relative to naive
all-hypotheses aggregation. Global pooled ranking is a practical post-generation
tool for making LazyFCA classification more compact and less noisy.

## Baseline Strategy

### Vanilla LazyFCA

Include it. It is the natural reference. Do not apologize for poor performance;
explain that this is the aggregation problem the paper addresses.

### Random Top-k

Include it. It shows that improvements are due to ranking and not merely due to
using fewer hypotheses.

### Randomized LazyFCA

Strong candidate to add.

Reasons:

- same family as LazyFCA;
- directly from the credit-scoring line of work;
- addresses the specificity problem differently;
- reviewer may expect it.

Potential experiments:

- randomized LazyFCA all generated hypotheses;
- randomized LazyFCA top-k ranked;
- compare against object-query global top-k.

### IPS-KNN

Handle carefully.

IPS-KNN uses a compact interval-pattern reason and may outperform the ranking
method. It is a related method, but not the same research question.

Options:

1. Mention in related work only.
2. Run privately and include only if results are not damaging or can be framed
   clearly.
3. Include as an "external compact lazy pattern-structure classifier" but state
   that it changes the model family.

If IPS-KNN dominates:

- do not hide that it exists;
- cite it;
- say this paper focuses on improving LazyFCA hypothesis aggregation;
- present IPS-KNN as future integration or as motivation for compact local
  reasons.

Possible bridge:

IPS-KNN supports the idea that compact local interval-pattern classifiers can be
meaningful. Our work asks whether a similarly compact subset can be selected
from the many hypotheses generated by LazyFCA.

## Reference and Bibliography Work

Need clean `paper/mybib.bib`.

Known issues:

- some entries may be arXiv versions but PDFs are published versions;
- some entries have missing journal/venue fields;
- `bien2011prototype` and `letham2015interpretable` had empty journal warnings
  during compilation;
- downloaded PDFs should not be committed publicly.

Important references already used:

- Wille, FCA origin.
- Ganter and Wille, FCA monograph.
- Ganter and Kuznetsov, pattern structures and projections.
- Kuznetsov, pattern structures for complex data.
- Masyutin, Kashnitsky, Kuznetsov, lazy classification with interval pattern
  structures for credit scoring.
- Tomat, IPS-KNN.
- Tomat accepted paper on deterministic FCALC/LazyFCA, randomized
  FCALC/LazyFCA, and IPS-KNN. This is the paper associated with the baseline
  code/results imported into this repository. Local author-owned PDF/LaTeX
  sources are in `paper/my_other_papers/`.
- Liu et al., CBA.
- Li et al., CMAR.
- Yin and Han, CPAR.
- Baralis et al., lazy associative classification.
- Thabtah, associative classification review.
- Todorovski, Flach, Lavrac, WRAcc.
- Atzmueller et al., subgroup discovery.
- Kuznetsov and Ignatov, concept stability.
- Ribeiro et al., LIME.
- Bien and Tibshirani, prototype selection.
- Letham/Rudin, Bayesian rule lists.
- Angelino/Rudin, CORELS.

Local author-owned papers:

- `paper/my_other_papers/manuscript.pdf`
- `paper/my_other_papers/manuscript.tex`
- `paper/my_other_papers/position_paper.pdf`
- `paper/my_other_papers/position_paper.tex`

Use the accepted manuscript to cite FCALC/randomized FCALC/IPSKNN baselines and
their implementation/protocol. Use the position paper when discussing the first
IPS-KNN idea. Rephrase all reused technical descriptions; do not copy text
verbatim from these manuscripts into the new paper.

## Immediate Writing Tasks

1. Update title and abstract after final results.
2. Replace TODOs in Background.
3. Tighten Related Work after bibliography cleanup.
4. Write Method section formally.
5. Generate result tables from final outputs.
6. Write Results and Discussion.
7. Add limitations and baseline decisions.
8. Remove or move implementation notes that do not belong in final paper.

## External AI Instructions

If another AI chat is asked to help:

- give it this file and `docs/project_context.md`;
- tell it not to invent citations;
- ask it to write in LaTeX using keys from `paper/mybib.bib`;
- ask it to preserve the modest claim;
- ask it to distinguish LazyFCA ranking from IPS-KNN and randomized LazyFCA;
- ask it to avoid claiming state-of-the-art accuracy.
