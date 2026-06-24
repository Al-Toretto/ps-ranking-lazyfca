# Literature Review Plan and Source Map for Ranking and Pruning LazyFCA Pattern-Structure Hypotheses

## Project framing

Your attached brief is already unusually sharp about the paper’s real contribution: the target is **not** generic FCA background and **not** a claim of state-of-the-art black-box accuracy, but a study of **global pooled top-k ranking over class-specific LazyFCA hypotheses** in order to reduce noisy aggregation, preserve or improve predictive quality, and make local explanations compact and interpretable. The brief also makes two points that should strongly shape the review: first, your ranking is over a **single shared pool across classes**, not separate per-class quotas; second, many generated hypotheses can be **singleton source-query rules** with `tp = 1`, so the review must not treat every high-purity hypothesis as evidence of broad generalization. fileciteturn0file0

That framing suggests a literature review with four linked aims. The first is to establish the mathematical lineage from FCA to pattern structures. The second is to position LazyFCA-like local hypothesis generation against adjacent lazy, associative, and case-based classifiers. The third is to review how interpretable rule systems rank, order, prune, and compact rule sets. The fourth is to justify the specific families of importance scores you are experimenting with—support, confidence, precision, WRAcc, lift, odds-style scores, locality-weighted variants, and FCA-specific stability notions—without pretending that all of these arise from one unified literature. fileciteturn0file0turn20search2turn13academia1turn23view0

## Literature review plan

The cleanest chapter architecture is to make the review **problem-driven rather than field-driven**. Start with FCA and pattern structures only to the extent needed to formalize local hypotheses, descriptions, extents/intents, and complex descriptions. Then move immediately to the classification problem: why full lattice construction is often impractical, why local query-time hypothesis generation is attractive, and why using all generated hypotheses can be noisy. That transition is important because it connects the classical FCA literature to your actual ranking and pruning problem instead of leaving the review as disconnected preliminaries. fileciteturn0file0turn20search2turn13academia2

After that, the review should split the prior work into three neighboring strands. One strand is **lazy or local classification**: papers that classify per query, use neighbors or instance-local evidence, or select local rules at prediction time. The second strand is **compact interpretable rule learning**: ordered rule lists, decision sets, associative classifiers, and pruning frameworks that trade accuracy against model size. The third strand is **hypothesis importance metrics**: rule-quality measures from association rule mining and subgroup discovery, FCA-specific concept importance measures such as stability, and locality-aware scoring ideas from interpretable ML. This lets you state a precise gap: existing literatures study local rules, FCA concepts, associative classification, compact rule sets, and local explanations, but the specific combination of **pattern-structure local hypotheses + global pooled ranking across classes + compact top-k explanation subsets** appears underexplored in the sources I could verify confidently. fileciteturn0file0turn25search0turn29view0turn36academia2turn44academia2

The review should end with a short methodological bridge into your own paper. That bridge should argue that your ranking functions are not merely “features” of rules; they operationalize competing notions of importance: purity, prevalence, exceptionality, statistical contrast, locality, and conceptual robustness. That is the right place to explain why different datasets may favor different scores, especially when many candidate hypotheses are pure singletons or when broad rules dominate by raw support. Those caveats are already in your brief and should be cited again in the paper’s discussion section. fileciteturn0file0

## Source map for foundations, pattern structures, and the LazyFCA neighborhood

*Group: FCA foundations*

**Rudolf Wille. _Restructuring Lattice Theory: An Approach Based on Hierarchies of Concepts_. In _Ordered Sets_, 445–470, 1982. Stable source: Reidel/Springer chapter record.**  
**Problem solved:** establishes FCA as a mathematically grounded theory of concepts and concept hierarchies rather than just a lattice-theoretic construction.  
**Relation to your paper:** essential for defining formal contexts, concepts, and lattice-ordered descriptions before you generalize to pattern structures and local hypotheses.  
**Best placement:** Background.  
**Summary:** This is the conceptual origin point for FCA in the sense most later work uses it. Cite it when you introduce extent, intent, Galois connection, and the idea that concept order expresses generality/specificity.  
**Relevance confidence:** Very high (0.99). citeturn20search4turn20search2

**Bernhard Ganter and Rudolf Wille. _Formal Concept Analysis: Mathematical Foundations_. Springer, 1999. ISBN 978-3-540-62771-5.**  
**Problem solved:** provides the canonical systematic treatment of FCA definitions, derivation operators, concept lattices, implications, and algorithms.  
**Relation to your paper:** this is the standard citation for all formal definitions you will use before introducing pattern structures and lazy local hypothesis generation.  
**Best placement:** Background and Method.  
**Summary:** If you need exactly one foundational monograph for FCA notation, this is it. It anchors the formalism of contexts, intents/extents, and closure operators that your paper inherits.  
**Relevance confidence:** Very high (0.99). citeturn20search2turn19search4

**Claudio Carpineto and Giovanni Romano. _Concept Data Analysis: Theory and Applications_. Wiley, 2004. Stable identifier: Wiley book record.**  
**Problem solved:** connects FCA and concept lattices to broader data-analytic practice, especially concept-based analysis and applications.  
**Relation to your paper:** useful when you want to motivate FCA as a practical data-analysis framework rather than a purely algebraic object.  
**Best placement:** Background or Discussion.  
**Summary:** This book is less foundational than Ganter–Wille, but more application-facing. It helps justify why FCA remains relevant for interpretable machine learning and knowledge discovery.  
**Relevance confidence:** High (0.90). citeturn20search2

*Group: Pattern structures and complex data descriptions*

**Mehdi Kaytoue, Sergei O. Kuznetsov, and Amedeo Napoli. _Revisiting Numerical Pattern Mining with Formal Concept Analysis_. arXiv:1111.5689, 2011.**  
**Problem solved:** addresses how FCA can work directly with numerical data rather than only via conceptual scaling into binary contexts.  
**Relation to your paper:** highly relevant because your setting includes numerical features and interval-like local descriptions; it supports the move from binary FCA to complex descriptions.  
**Best placement:** Background and Method.  
**Summary:** The paper argues that direct work with numerical descriptions can avoid information loss and inefficiency caused by binarization. For your paper, it is one of the strongest bridges from classical FCA to pattern-structure-like reasoning over numeric attributes.  
**Relevance confidence:** Very high (0.96). citeturn13academia1turn32academia1

**Aleksey Buzmakov, Sergei O. Kuznetsov, and Amedeo Napoli. _Revisiting Pattern Structure Projections_. arXiv:1506.05018, 2015.**  
**Problem solved:** studies projections for reducing the computational burden of pattern structures while preserving relevant structure.  
**Relation to your paper:** important for discussion because your ranking/pruning currently shrinks the **retained explanation set after generation**, whereas projection work attacks complexity earlier in the pipeline.  
**Best placement:** Related Work and Discussion.  
**Summary:** The paper generalizes pattern-structure projections and shows they form a semilattice. In your paper, it is the right citation when distinguishing *post-generation pruning* from *pre-generation structural reduction*.  
**Relevance confidence:** Very high (0.95). citeturn13academia2turn7academia5

**Aleksey Buzmakov, Elias Egho, Nicolas Jay, Sergei O. Kuznetsov, Amedeo Napoli, and Chedy Raïssi. _On Mining Complex Sequential Data by Means of FCA and Pattern Structures_. arXiv:1504.02255, 2015.**  
**Problem solved:** shows how pattern structures can mine complex sequential data with meaningful projections and condensed pattern representations.  
**Relation to your paper:** broadens the argument that pattern structures are not just interval tricks for tabular data, but a general framework for structured hypotheses.  
**Best placement:** Background or Related Work.  
**Summary:** This paper is especially useful if you want a sentence showing that pattern structures extend naturally beyond simple numeric tables. That helps position your work as one instance of a larger concept-based framework for interpretable structured descriptions.  
**Relevance confidence:** High (0.90). citeturn13academia0turn9academia0

**Aimene Belfodil, Sergei Kuznetsov, and Mehdi Kaytoue. _On Pattern Setups and Pattern Multistructures_. arXiv:1906.02963, 2019.**  
**Problem solved:** revisits the algebraic assumptions behind pattern structures and studies broader settings where descriptions form only partial orders or multilattices.  
**Relation to your paper:** useful if you want to signal awareness of the limits of classical meet-semilattice assumptions when descriptions become richer.  
**Best placement:** Background or Discussion.  
**Summary:** This is not necessary for a minimal paper, but it strengthens the theoretical perimeter around “pattern-structure hypotheses.” It is especially helpful if reviewers push on how general your description language really is.  
**Relevance confidence:** Medium-high (0.82). citeturn14academia2turn7academia2

*Group: LazyFCA and nearby lazy/local FCA-style classification*

**E. Baralis, S. Chiusano, and P. Garza. _A Lazy Approach to Associative Classification_. _IEEE Transactions on Knowledge and Data Engineering_, 20(2):156–171, 2008. doi:10.1109/TKDE.2007.190677.**  
**Problem solved:** classifies instances by selecting and exploiting relevant rules lazily at prediction time rather than building one fixed global classifier.  
**Relation to your paper:** this is one of the strongest neighboring citations for the idea that local, query-time rule selection can improve classification and interpretability. It is not FCA, but it is a very strong analogue to your local-hypothesis setting.  
**Best placement:** Related Work.  
**Summary:** The paper is a close methodological neighbor because it couples lazy inference with rule-based classification and implicitly with compact local evidence. It is particularly valuable when positioning your contribution against the broader lazy rule-classification literature.  
**Relevance confidence:** Very high (0.95). citeturn23view0turn25search0

**Yury Kashnitsky and Dmitry I. Ignatov. _Can FCA-Based Recommender System Suggest a Proper Classifier?_ arXiv:1504.05473, 2015.**  
**Problem solved:** recommends a classifier for a given instance based on neighborhood behavior, using FCA to organize which classifiers perform well on similar objects.  
**Relation to your paper:** not LazyFCA itself, but highly relevant for the combination of FCA, local object neighborhoods, and instance-specific model choice.  
**Best placement:** Related Work or Discussion.  
**Summary:** This paper matters because it shows FCA being used at query level rather than only as a global concept miner. It helps justify your claim that object-local FCA reasoning is a legitimate design pattern in interpretable classification.  
**Relevance confidence:** High (0.88). citeturn44academia3turn45academia0

**Marcel Boersma, Krishna Manoorkar, Alessandra Palmigiano, Mattia Panettiere, Apostolos Tzimoulis, and Nachoem Wijnberg. _Flexible Categorization Using Formal Concept Analysis and Dempster-Shafer Theory_. arXiv:2408.15012, 2024.**  
**Problem solved:** develops an FCA-grounded framework for explainable categorization and discusses a meta-algorithm for outlier detection and classification with local and global explanations.  
**Relation to your paper:** useful as modern evidence that FCA is actively being connected to explainable classification rather than only unsupervised concept analysis.  
**Best placement:** Related Work or Discussion.  
**Summary:** This source is newer and more XAI-facing than classical FCA papers. It does not solve your top-k hypothesis pruning problem, but it helps establish a contemporary conversation around FCA-based local and global explanations.  
**Relevance confidence:** Medium-high (0.80). citeturn45academia1

**Dmitry I. Ignatov. _Introduction to Formal Concept Analysis and Its Applications in Information Retrieval and Related Fields_. arXiv:1703.02819, 2017.**  
**Problem solved:** tutorial overview of FCA applications, including machine learning and data mining.  
**Relation to your paper:** a practical bridge citation when you want one source that explicitly connects FCA to machine learning audiences.  
**Best placement:** Background.  
**Summary:** This is not a core method citation, but it helps orient readers outside the FCA community. It is especially useful if your target journal is broader than FCA-specific venues.  
**Relevance confidence:** Medium (0.76). citeturn44academia0

## Source map for ranking, pruning, local explanations, metrics, and similarity-aware selection

*Group: Pattern or rule ranking in interpretable classification*

**Benjamin Letham, Cynthia Rudin, Tyler H. McCormick, and David Madigan. _Interpretable Classifiers Using Rules and Bayesian Analysis: Building a Better Stroke Prediction Model_. arXiv:1511.01644, 2015.**  
**Problem solved:** learns sparse decision lists whose rules are ordered and selected under a probabilistic framework.  
**Relation to your paper:** one of the cleanest citations for the idea that interpretable classification depends heavily on **ordering and compactness**, not just the existence of human-readable rules.  
**Best placement:** Related Work.  
**Summary:** Bayesian Rule Lists show that small ordered rule sets can remain accurate and clinically usable. For your paper, this work supports the claim that shrinking and ordering a rule collection is itself a substantive modeling problem.  
**Relevance confidence:** Very high (0.94). citeturn39academia2turn38academia2

**Fulton Wang and Cynthia Rudin. _Falling Rule Lists_. arXiv:1411.5899, 2014.**  
**Problem solved:** builds ordered decision lists in which estimated risk monotonically decreases down the list.  
**Relation to your paper:** relevant because it treats rule ordering as semantically meaningful and explicitly tied to importance or risk level.  
**Best placement:** Related Work or Discussion.  
**Summary:** Falling Rule Lists are useful when arguing that rankings over interpretable rules are not merely heuristic conveniences; they can encode meaningful monotone structure. That can inspire discussion of whether your top-k pool should also preserve a calibrated semantic order.  
**Relevance confidence:** High (0.86). citeturn39academia1

**Elaine Angelino, Nicholas Larus-Stone, Daniel Alabi, Margo Seltzer, and Cynthia Rudin. _Learning Certifiably Optimal Rule Lists for Categorical Data_. arXiv:1704.01701, 2017.**  
**Problem solved:** learns sparse rule lists with certificates of optimality under regularized empirical risk.  
**Relation to your paper:** this is an important contrast case: instead of pruning local hypotheses after generation, CORELS-style work searches globally for an optimal compact rule list.  
**Best placement:** Related Work and Discussion.  
**Summary:** The paper is highly relevant for discussing compactness versus accuracy and the difference between global interpretable model induction and local explanation selection. It helps reviewers see that your contribution is about **ranking generated local hypotheses**, not solving optimal global rule-list search.  
**Relevance confidence:** Very high (0.93). citeturn41academia0

**Himabindu Lakkaraju, Ece Kamar, Rich Caruana, and Jure Leskovec. _Interpretable & Explorable Approximations of Black Box Models_. arXiv:1707.01154, 2017.**  
**Problem solved:** learns small decision sets to approximate black-box models while trading off fidelity, interpretability, and non-redundancy.  
**Relation to your paper:** relevant as a compactness-aware decision-set baseline from interpretable ML, especially for discussion of ambiguity and redundancy.  
**Best placement:** Related Work.  
**Summary:** BETA is valuable because it explicitly optimizes a small collection of human-readable rules for coverage and fidelity. Your paper tackles a different regime—ranking hypotheses already generated from LazyFCA—but the overlap on compact explanations is real.  
**Relevance confidence:** High (0.84). citeturn42academia2

*Group: Rule pruning, associative classification, subgroup discovery, and compact rule sets*

**Bing Liu, Wynne Hsu, and Yiming Ma. _Integrating Classification and Association Rule Mining_, 1998. Stable source: CiteSeerX 10.1.1.48.8380.**  
**Problem solved:** introduces classification based on association rules, the CBA family, where rules predict the class label.  
**Relation to your paper:** this is the classical entry point for rule ranking and pruning in classification; it provides one of the strongest non-FCA baselines for your discussion.  
**Best placement:** Related Work.  
**Summary:** Even though the retrieved source did not expose the venue cleanly, this is the standard seminal CBA paper and should be in your review. It frames classification as selecting and ordering high-quality class association rules, which is conceptually close to selecting high-quality LazyFCA hypotheses.  
**Relevance confidence:** High (0.89). citeturn23view0turn25search0

**Wenmin Li, Jiawei Han, and Jian Pei. _CMAR: Accurate and Efficient Classification Based on Multiple Class-Association Rules_. In _Proceedings 2001 IEEE International Conference on Data Mining_, 369–376, 2001. doi:10.1109/ICDM.2001.989541.**  
**Problem solved:** improves associative classification by leveraging multiple class association rules rather than one first-match rule.  
**Relation to your paper:** relevant because your method also retains multiple hypotheses and aggregates them, rather than relying on a single selected rule.  
**Best placement:** Related Work or Discussion.  
**Summary:** CMAR is useful when contrasting single-rule, ordered-rule, and multi-rule aggregation strategies. Your top-k pooled LazyFCA can be framed as an interpretable local analogue of multi-rule evidence aggregation.  
**Relevance confidence:** Very high (0.92). citeturn23view0

**Xiaoxin Yin and Jiawei Han. _CPAR: Classification Based on Predictive Association Rules_. In _Proceedings of the 2003 SIAM International Conference on Data Mining_, 331–335, 2003. doi:10.1137/1.9781611972733.40.**  
**Problem solved:** blends associative classification with predictive-rule ideas to improve both accuracy and compactness.  
**Relation to your paper:** relevant for discussions of ranking, compactness, heuristic search, and predictive rather than merely frequent rules.  
**Best placement:** Related Work.  
**Summary:** CPAR is a strong citation when you discuss rules that are chosen for predictive usefulness instead of raw frequency or support alone. That distinction maps directly onto your motivation for rejecting harmful or overly broad LazyFCA hypotheses.  
**Relevance confidence:** Very high (0.91). citeturn23view0

**Tomas Kliegr and Ebroul Izquierdo. _QCBA: Improving Rule Classifiers Learned from Quantitative Data by Recovering Information Lost by Discretisation_. arXiv:1711.10166, 2017.**  
**Problem solved:** post-processes rule classifiers to recover numeric information lost by discretization and introduces pruning/tuning steps to make models smaller and better.  
**Relation to your paper:** especially relevant because your hypotheses over numerical descriptions face the same question of how to preserve meaningful intervals while keeping explanations compact.  
**Best placement:** Related Work or Discussion.  
**Summary:** QCBA is one of the best adjacent citations for post-hoc improvement of rule sets learned on quantitative data. It supports the idea that compactness-oriented post-processing can improve both interpretability and accuracy.  
**Relevance confidence:** Very high (0.93). citeturn22academia0

**Martin Atzmueller, Frank Puppe, and Hans-Peter Buscher. _Exploiting Background Knowledge for Knowledge-Intensive Subgroup Discovery_. In _Proceedings of IJCAI_, 647–652, 2005. Stable PDF source: IJCAI proceedings.**  
**Problem solved:** studies subgroup discovery as a search over interpretable descriptions guided by quality functions, constraints, and background knowledge.  
**Relation to your paper:** extremely useful for your metric discussion because subgroup discovery explicitly balances subgroup size with target enrichment and uses search-time quality measures to rank descriptions.  
**Best placement:** Related Work, Method, and Discussion.  
**Summary:** This is not a classifier-ranking paper in the LazyFCA sense, but it provides a nearby interpretability tradition where descriptive rules are ranked by exceptional quality relative to a population. That is conceptually very close to arguing for WRAcc-like or enrichment-style scoring of local hypotheses.  
**Relevance confidence:** Very high (0.94). citeturn29view0

*Group: Local interpretable classifiers and instance-based explanation methods*

**Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. _“Why Should I Trust You?”: Explaining the Predictions of Any Classifier_. arXiv:1602.04938, 2016.**  
**Problem solved:** explains individual predictions by fitting a sparse interpretable model locally around the query using locality weights.  
**Relation to your paper:** perhaps the strongest non-FCA citation for the idea that explanations should be **local** and that locality itself can be part of the weighting function.  
**Best placement:** Related Work and Method.  
**Summary:** LIME lets you connect your query-weighted metrics to a broader XAI logic: local evidence near the test object should matter more than globally strong but locally irrelevant patterns. This is especially helpful when motivating query-similarity-weighted precision or odds scores.  
**Relevance confidence:** Very high (0.97). citeturn36academia2

**Jacob Bien and Robert Tibshirani. _Prototype Selection for Interpretable Classification_. arXiv:1202.5933, 2012.**  
**Problem solved:** selects a small subset of representative training instances as prototypes for interpretation and classification.  
**Relation to your paper:** highly relevant to your `tp = 1` discussion, because singleton source-query hypotheses often behave like prototype- or nearest-neighbor-style local evidence.  
**Best placement:** Related Work and Discussion.  
**Summary:** This paper helps you frame singleton local hypotheses as potentially legitimate instance-based explanations rather than mere degenerate rules. It gives you a principled bridge between rule-style explanations and case/prototype-style explanations.  
**Relevance confidence:** High (0.90). citeturn37academia2

**Chaofan Chen, Oscar Li, Chaofan Tao, Alina Jade Barnett, Jonathan Su, and Cynthia Rudin. _This Looks Like That: Deep Learning for Interpretable Image Recognition_. arXiv:1806.10574, 2018.**  
**Problem solved:** builds interpretable classification through learned prototypes that support case-based “this looks like that” reasoning.  
**Relation to your paper:** useful as a modern demonstration that prototype-style local evidence can be a first-class interpretability mechanism in classification.  
**Best placement:** Discussion.  
**Summary:** ProtoPNet is from a different modality, but conceptually it is very helpful for defending local-support explanations grounded in comparisons to representative cases. It strengthens the argument that not every good explanation must be a broad global rule.  
**Relevance confidence:** Medium-high (0.81). citeturn36academia1

**Hilde J. P. Weerts, Werner van Ipenburg, and Mykola Pechenizkiy. _Case-Based Reasoning for Assisting Domain Experts in Processing Fraud Alerts of Black-Box Machine Learning Models_. arXiv:1907.03334, 2019.**  
**Problem solved:** uses similar previous instances and local explanation similarity to support trust in individual predictions.  
**Relation to your paper:** important for connecting object similarity, explanation similarity, and local evidence reuse.  
**Best placement:** Discussion.  
**Summary:** This source is a good modern case-based XAI citation when you discuss why nearest-neighbor-like explanations can still be useful, especially in high-cost domains. It complements prototype selection by emphasizing trust calibration via similar prior instances.  
**Relevance confidence:** Medium-high (0.79). citeturn37academia0

*Group: Metrics relevant to hypothesis importance*

**Fadi Thabtah. _A Review of Associative Classification Mining_. _The Knowledge Engineering Review_, 22(1):37–65, 2007. doi:10.1017/S0269888907001026.**  
**Problem solved:** surveys associative classification algorithms and the role of metrics such as support and confidence in rule ordering and filtering.  
**Relation to your paper:** this is one of the best compact review citations for classic rule-quality metrics in predictive rule systems.  
**Best placement:** Background or Method.  
**Summary:** Use this paper to anchor support, confidence, and the general rule-ranking vocabulary from associative classification. It is especially useful when justifying why rule quality should be treated as a design axis rather than a postscript.  
**Relevance confidence:** Very high (0.93). citeturn23view0turn26search1

**Martin Atzmueller, Frank Puppe, and Hans-Peter Buscher. _Exploiting Background Knowledge for Knowledge-Intensive Subgroup Discovery_. 2005.**  
**Problem solved:** formalizes subgroup discovery quality functions that compare subgroup target frequency against the reference population and subgroup size.  
**Relation to your paper:** relevant for WRAcc-style and enrichment-style thinking, where interestingness is not just purity but also exceptional coverage.  
**Best placement:** Method.  
**Summary:** The paper explicitly discusses the ingredients of subgroup quality functions: target share difference, subgroup size, and search-time ranking. That makes it a valuable bridge for WRAcc, coverage-aware ranking, and “exceptionality versus support” arguments in your metric section.  
**Relevance confidence:** Very high (0.92). citeturn29view0

**Sergei O. Kuznetsov and Dmitry I. Ignatov. _Concept Stability for Constructing Taxonomies of Web-Site Users_. arXiv:0905.1424, 2009.**  
**Problem solved:** uses the stability index to reduce the huge number of concepts and retain structurally meaningful ones.  
**Relation to your paper:** this is the most direct FCA-specific importance measure in the verified source set and is therefore essential if you discuss stability-like ranking of hypotheses or concepts.  
**Best placement:** Background, Method, and Discussion.  
**Summary:** Stability is especially important because it shifts “importance” away from mere support or purity toward robustness of the description under object removal. For your paper, that gives a concept-theoretic counterweight to purely predictive contingency metrics.  
**Relevance confidence:** Very high (0.94). citeturn32academia0turn33academia1

**Ayao Bobi, Rokia Missaoui, and Mohamed Hamza Ibrahim. _Enhancing Actionable Formal Concept Identification with Base-Equivalent Conceptual-Relevance_. arXiv:2312.14421, 2023.**  
**Problem solved:** proposes a concept-relevance measure and explicitly compares it to the well-known stability index for identifying actionable formal concepts.  
**Relation to your paper:** useful as a modern citation showing that FCA concept importance is still an active topic and that stability is not the only game in town.  
**Best placement:** Discussion.  
**Summary:** This is a newer, more application-facing relevance measure paper. It helps you argue that FCA-based importance ranking remains open, which supports the legitimacy of testing multiple hypothesis-ranking metrics in LazyFCA.  
**Relevance confidence:** Medium-high (0.80). citeturn33academia0

*Group: Similarity or locality combined with rule weighting or selection*

**E. Baralis, S. Chiusano, and P. Garza. _A Lazy Approach to Associative Classification_. 2008.**  
**Problem solved:** performs lazy, query-dependent selection of rules rather than using one fixed global rule ordering.  
**Relation to your paper:** this is the closest high-confidence analogue to “generate many rules, then retain only locally relevant ones.”  
**Best placement:** Related Work and Method.  
**Summary:** If you need one non-FCA citation for local rule selection at prediction time, use this one. It makes your global pooled top-k selection look like a pattern-structure-specific descendant of the broader lazy rule-classification idea.  
**Relevance confidence:** Very high (0.95). citeturn23view0turn25search0

**Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. _“Why Should I Trust You?”: Explaining the Predictions of Any Classifier_. 2016.**  
**Problem solved:** uses locality-weighted perturbation samples to select a sparse explanation around a specific query.  
**Relation to your paper:** directly relevant to your query-similarity weighted metrics because it operationalizes the idea that local proximity should change explanation weighting.  
**Best placement:** Method and Discussion.  
**Summary:** LIME belongs here as much as in the local-explanation section because it explicitly combines local weighting with sparse explanation selection. That is conceptually close to ranking LazyFCA hypotheses by a hybrid of predictive quality and query closeness.  
**Relevance confidence:** Very high (0.96). citeturn36academia2

**Léonard Kwuida and Dmitry I. Ignatov. _On Interpretability and Similarity in Concept-Based Machine Learning_. arXiv:2102.12723, 2021.**  
**Problem solved:** discusses similarity and attribute-importance ideas in concept-based machine learning and how they support classification and explanation.  
**Relation to your paper:** this is one of the most directly relevant conceptual citations for mixing concept-based explanations with similarity-aware importance.  
**Best placement:** Related Work, Method, and Discussion.  
**Summary:** The paper is especially useful because it speaks your language: interpretability, similarity, and concept-based ML in one place. It can support your argument that similarity-weighted metrics are not ad hoc engineering, but part of a broader interpretability logic.  
**Relevance confidence:** Very high (0.92). citeturn44academia2

**Yury Kashnitsky and Dmitry I. Ignatov. _Can FCA-Based Recommender System Suggest a Proper Classifier?_ 2015.**  
**Problem solved:** uses neighborhood correctness to recommend a classifier for an instance.  
**Relation to your paper:** relevant because it combines local similarity with prediction-time selection among competing predictive mechanisms.  
**Best placement:** Discussion.  
**Summary:** This is a good supporting citation when you want to emphasize that instance-level locality can guide not only explanations but even the choice of predictive evidence. It helps contextualize your preference for locality-aware ranking functions.  
**Relevance confidence:** High (0.87). citeturn44academia3turn45academia0

## Recommended citation architecture for the paper

In **Background**, cite Wille, Ganter–Wille, and one practical FCA bridge such as Carpineto–Romano or Ignatov’s tutorial. Then move quickly to Kaytoue and Buzmakov to justify pattern structures for numerical and structured descriptions. This section should be mathematically compact; the point is to give readers enough formal footing to understand local hypotheses, not to reproduce an FCA textbook. citeturn20search4turn20search2turn13academia1turn13academia2turn44academia0

In **Related Work**, organize the literature around the paper’s real competitors: lazy/local rule classification, associative classification, compact rule lists, decision sets, and local explanation methods. Baralis–Chiusano–Garza, CBA/CMAR/CPAR, Bayesian Rule Lists, CORELS, LIME, and prototype-based explanation papers together give you a very credible map of adjacent work. That map also lets you say something substantive: your paper is not competing with one monolithic “interpretable classifier” literature, but sits at the intersection of **local explanation**, **rule compaction**, and **FCA/pattern-structure hypothesis generation**. citeturn23view0turn25search0turn39academia2turn41academia0turn36academia2turn37academia2

In **Method**, reserve citations for the metrics themselves. Use Thabtah for support/confidence-oriented rule ranking, Atzmueller for subgroup-quality and exceptionality logic, Kuznetsov–Ignatov for stability, and LIME/Kwuida–Ignatov for locality-weighted scoring. That combination is powerful because it lets you say each score instantiates a distinct philosophy of importance: frequency, purity, contrast, robustness, or locality. The attached brief’s own observations about singleton `tp = 1` hypotheses then give you the paper-specific reason these philosophies matter empirically. citeturn26search1turn29view0turn32academia0turn36academia2turn44academia2fileciteturn0file0

In **Discussion**, explicitly separate two benefits: **explanation compaction** and **generation-time efficiency**. Your current method reduces the retained explanation set after generation, but it does not yet avoid generating all local hypotheses—a distinction the attached brief makes very clearly. That is where Buzmakov’s projection work becomes useful, because it suggests future paths for earlier-stage structural reduction. Discussion is also where you should interpret singleton rules as potentially prototype-like or case-based explanations rather than either dismissing them or overclaiming them as generalized rules. fileciteturn0file0turn13academia2turn37academia2turn37academia0

## Open questions and limitations

The strongest limitation in the current verified source map is that I could not confidently verify, from retrievable primary pages, a directly indexed paper explicitly using the name **“LazyFCA”** or the original **Ganter–Kuznetsov pattern-structures chapter** without risking bibliographic invention. I therefore excluded any such citation from the high-confidence map rather than guessing. Relatedly, some classic association-classification references were recoverable through reliable index pages and review pages, but not always with perfectly exposed venue metadata, so I kept the citation text conservative when necessary. citeturn23view0turn25search0turn14academia2

The second limitation is metric coverage. I found strong anchors for support/confidence-style measures, subgroup-discovery quality functions, and FCA stability, but not a clean single primary-source chain for every metric in your implementation list, especially the odds-ratio/log-likelihood family and delta-stability/robustness variants as they are specifically instantiated in your codebase. For the paper, that is manageable: you can cite the review and subgroup-discovery sources for the general metric families, cite concept-stability work for FCA-specific robustness, and then define your exact operational formulas in the Method section from your implementation. The attached brief is especially important here because it already records the exact metric families, hybrid locality-weighted variants, and the `tp = 1` caveat that should shape how you discuss them. fileciteturn0file0turn26search1turn29view0turn32academia0