Current capabilities
====================

BuildML 2.6.3 centers on :class:`buildml.Session`. Classical classification and regression
are available in the core installation. Other domains use the same Session
and operation history, with additional dependencies where required.

This page summarizes supported domains, optional dependencies, principal
methods, and limitations. Tutorials live in :doc:`guide-index`.

Data and workflow
-----------------

* Ingest Pandas DataFrames and CSV, Parquet, Arrow, and Excel sources.
* Record source detection, scale estimates, mode and engine choices, and
  loading warnings in an ingest report.
* Assign feature, target, identifier, group, time, weight, and ignored
  roles.
* Create random, stratified, group-aware, or chronological partitions, or
  inject externally designed row memberships.
* Save and validate checkpoints containing data, roles, partitions,
  history, optional preprocess plans, and an integrity manifest.
* Switch or materialize through Pandas, Polars, or DuckDB where
  installed. DuckDB connections close via ``close_native`` or a context
  manager.

Preparation
-----------

* Drop columns and extract date parts.
* Fit imputation, categorical encoding, scaling, outlier handling,
  binning, feature selection, text features, and dimensionality
  reduction on training rows.
* Resample only the training partition when ``buildml[imbalanced]`` is
  installed.
* Apply registered custom transforms (Session-global; not fold-local in
  CV).
* Use ``PreprocessRecipe`` for fold-local preparation inside
  ``cv_score``, ``grid_search``, ``optuna_search``,
  ``evolutionary_search``, and ``nested_cv_score``.

Classical models and diagnostics
--------------------------------

* Fit sklearn-compatible classifiers and regressors on the training
  partition.
* Compare named estimators under one partition and ranking metric.
  Override the default partition (test) during iterative selection.
* Evaluate classification or regression metrics with task baselines.
* Run cross-validation, grid search, randomized search, Optuna search,
  and nested CV without scoring Session test rows in inner loops.
* Inspect calibration, threshold tradeoffs, learning curves, permutation
  importance, error slices, and task-adaptive plot boards.

Domain catalog
--------------

Optional backend packages are loaded by the operations that need them.
Domains expose capability matrices to report installed backend availability.

Unsupervised
   Cluster on train (``session.unsupervised.fit``), assign holdout labels
   without refit, evaluate geometric validity. Optional PCA components from
   ``reduce_dimensions``. Bundle: ``buildml.unsupervised_bundle.v1``. Cluster
   labels describe the fitted grouping; they are not verified class labels.
   Anomaly scoring has a separate API. Guides: :doc:`quickstart-unsupervised`,
   :doc:`unsupervised-deep`.

Ensembles
   ``session.ensemble.fit_voting`` / ``fit_stacking`` / ``fit_blending``.
   Stacking CV and blend holdouts stay inside train. Bundle:
   ``buildml.ensemble_bundle.v1``. Distinct from passing one RandomForest to
   ``fit``. Guides: :doc:`quickstart-ensemble`, :doc:`ensemble-deep`.

AutoML
   ``session.automl.run``: joint model-family and fold-local preprocess search
   under a trial budget (``cv`` / ``nested`` / ``validation``). Same
   Session-global preprocess refusal as classical CV. Extra:
   ``buildml[optuna]`` for the Optuna method; industry families via
   ``automl-industry``. Neural architecture search and causal discovery are
   outside this API. Guides: :doc:`quickstart-automl`, :doc:`automl-deep`.

Forecasting
   ``session.forecast.fit`` / ``generate`` / ``evaluate``. Needs a ``time``
   role and ``time_split``. Univariate by default; optional numeric exogenous
   columns with disclosed future-exog requirements. Supported forecasting
   methods and backend requirements are listed in the forecasting guides.
   Guides: :doc:`quickstart-forecasting`, :doc:`forecasting-deep`.

Time-series analysis
   ``session.timeseries.analyze`` / ``decompose`` / ``diagnostics``. Analysis
   only: no forecast model is fitted here. Distinct from
   ``session.forecast.fit``. Additional analysis backends use
   ``buildml[timeseries]``. Guides:
   :doc:`quickstart-timeseries-analysis`, :doc:`timeseries-analysis-deep`.

Anomaly / fraud
   ``session.anomaly.fit`` / ``score`` / ``evaluate`` / ``tune_threshold``.
   IsolationForest, LOF (novelty), One-Class SVM, or supervised HGB when a
   binary target exists. Distinct from EDA IsolationForest screens and
   ``handle_outliers``. Scores identify unusual observations or classify a
   supplied target. They do not establish whether fraud occurred or explain
   its cause. Guides: :doc:`quickstart-anomaly`, :doc:`anomaly-deep`.

Semi-supervised
   ``session.semisupervised.fit`` on scarce labeled plus unlabeled train rows
   (target NaNs mark unlabeled). Holdout metrics are labeled-only. Distinct
   from anomaly novelty and from self-supervised pretext. Guides:
   :doc:`quickstart-semisupervised`, :doc:`semisupervised-deep`.

Self-supervised
   ``session.ssl.fit_pretext`` (masked tabular autoencoder lite),
   ``transform``, ``finetune_head``, ``evaluate``. This API does not pretrain
   language models. Vision / audio freeze-finetune remains
   ``session.dl.load_backbone``. Guides: :doc:`quickstart-selfsupervised`,
   :doc:`selfsupervised-deep`.

Active learning
   Train-pool query (``suggest_query``) then human labels (``label_rows``).
   Pool is train target NaNs. Labels must come from the user or an external
   labeling process. Budget caps are enforced. Guides:
   :doc:`quickstart-active-learning`, :doc:`active-learning-deep`.

Online / continual
   sklearn ``partial_fit`` on train chunks. Validation and test are never used
   for updates. Updates require an incremental estimator. Distributed stream
   processing must be managed outside BuildML. Guides:
   :doc:`quickstart-online-learning`, :doc:`online-learning-deep`.

Multi-task
   sklearn ``MultiOutput*`` / chains support two or more same-type targets.
   The optional Torch ``shared_trunk_multihead`` model also supports mixed
   classification and regression targets. Guides:
   :doc:`quickstart-multi-task`, :doc:`multi-task-deep`.

Meta-learning
   Episodic few-shot (``prototypical`` / ``warm_start``). Holdout evaluation
   prefers novel task ids and discloses overlaps. The included examples target
   small episodic learning tasks. Guides: :doc:`quickstart-meta-learning`,
   :doc:`meta-learning-deep`.

Federated
   Local FedAvg / FedProx simulation on a client or group column. The
   simulation does not provide network transport or cryptographic secure
   aggregation. Guides: :doc:`quickstart-federated`, :doc:`federated-deep`.

Probabilistic
   BayesianRidge / GaussianProcess / Naive Bayes, plus train-only split
   conformal intervals. Evaluate NLL / coverage. These adapters do not expose
   general-purpose probabilistic programming or Markov chain Monte Carlo
   inference. Guides: :doc:`quickstart-probabilistic`,
   :doc:`probabilistic-deep`.

Causal
   ``session.causal.declare_assumptions`` then fit / estimate / evaluate /
   refute. Backdoor ATE via T-learner / IPW / AIPW. Refuses incomplete
   assumptions. EDA associations alone are insufficient input for causal
   conclusions. Guides: :doc:`quickstart-causal`, :doc:`causal-deep`.

Graph ML
   Node classify: NetworkX (``buildml[graph]``), pure-Torch GCN
   (``buildml[torch]``), optional PyG (``buildml[graph-pyg]``).
   Knowledge-graph embeddings have a separate API (``session.kg``). Graph
   database hosting is outside BuildML. Guides: :doc:`quickstart-graph`,
   :doc:`graph-deep`.

Symbolic
   Declared or induced if-then rules with traces; sklearn hybrid overlays. The
   supported rule formats and inference methods are listed in the symbolic
   learning guides. Guides: :doc:`quickstart-symbolic`, :doc:`symbolic-deep`.

Case-based reasoning
   Train-only case memory. Distinct from RAG. Guides: :doc:`quickstart-cbr`,
   :doc:`cbr-deep`.

Imitation + RL
   Behavioral cloning and contextual bandits in core. Gymnasium tabular TD and
   REINFORCE-lite via ``buildml[rl]``; SB3 via ``buildml[rl-industry]``.
   Robotics integration and multi-agent simulation environments are external
   to BuildML. Guides: :doc:`quickstart-imitation-rl`,
   :doc:`imitation-rl-deep`.

TDA
   Local Vietoris–Rips plus vectorization, then a sklearn head
   (``buildml[tda]``). Guides: :doc:`quickstart-tda`, :doc:`tda-deep`.

Recommenders
   User-based, item-based, and content-based recommendations, evaluated with
   ranking metrics. Retrieval for text generation and learning-to-rank models
   have separate APIs. Guides: :doc:`quickstart-recommenders`,
   :doc:`recommenders-deep`.

Learning to rank
   Query–item feature rows. sklearn fallback plus industry GBDT rankers.
   Distinct from RAG and from recommenders. Guides: :doc:`quickstart-ranking`,
   :doc:`ranking-deep`.

Knowledge graphs
   Embed head-relation-tail triples with TransE or DistMult, or use PyKEEN
   RotatE and ComplEx via ``buildml[kg-industry]``. Symbolic queries are also
   supported. Database hosting has to be provided separately. Guides:
   :doc:`quickstart-kg`, :doc:`kg-deep`.

Decisions
   Thresholds, cost matrices, top-K / knapsack / LP. Supported solvers and
   problem forms are documented in the optimization guides. Guides:
   :doc:`quickstart-optimize`, :doc:`optimize-deep`.

Fairness
   Compare demographic parity, disparate impact, and equal opportunity on a
   holdout, with optional stability estimates and intersectional groups.
   Suggestion methods return thresholds or weights for review; they do not
   apply mitigation automatically or assess legal compliance. Optional SHAP
   via ``explain_shap`` (``buildml[shap]``). Guides:
   :doc:`quickstart-fairness`, :doc:`fairness-deep`.

Synthetic data
   Generate data with bootstrap, copula, SMOTE, or optional SDV methods.
   Evaluate statistical similarity and train-on-synthetic/test-on-real
   performance. These methods do not provide differential privacy. Guides:
   :doc:`quickstart-synthetic`, :doc:`synthetic-deep`.

NLP
   A text column on the Session dataset: corpus profile, document classify,
   token attribution, topics, keyphrases, summaries, entities, sentiment,
   language. Bag-of-n-grams by default; encoders via ``buildml[nlp]``. This
   namespace focuses on document analysis and classification. Transformer
   fine-tuning uses the Torch text API; retrieval for generation uses RAG.
   Translation and text generation are not NLP namespace operations. Guides:
   :doc:`quickstart-nlp`, :doc:`nlp-deep`.

Teaching and reports
--------------------

* Explain any public operation before or after execution from a
  versioned catalog.
* Resolve workflow operations as done, available, blocked, or skipped.
* Preview with ``dry_run``; summarize history and heuristic risks.
* Export EDA, evaluation, diagnostic, and walkthrough reports as local
  HTML. ``html_format="research"`` writes the static EDA review
  report. ``html_format="studio"`` writes an offline app snapshot.
* Launch the local Industry EDA App when ``buildml[dashboard]`` is
  installed (report panels, review checklists, and concept explanations). Gate marks are
  browser-tab UI state and are never persisted.

Optional stacks (same Session)
------------------------------

**Torch** (``buildml[torch]``)
   Tabular, text, image, and audio loaders; built-in MLP / text / fusion
   modules; fold-local CV / search / nested; AMP; single-node and torchrun
   multi-node DDP; TorchScript / ONNX export.

**Speech** (``buildml[speech]``)
   ASR transcription (stub or transformers), WER / CER, speech classify
   finetune-lite. Foundation-model pretraining is outside this API.

**Pretrained backbones** (``buildml[vision]`` / ``[speech]`` / ``[pretrained]``)
   Curated ResNet / ViT / audio / speech hooks with
   ``weights=none|mock|pretrained``. Only the documented architectures and
   weight sources are supported.

**Serve** (``buildml[serve]``)
   Local FastAPI for classical pipeline bundles and TorchScript (``/health``,
   ``/metadata``, ``/predict``, ``/predict/batch``). The server binds to
   localhost by default. Managed cloud identity and access control require
   deployment configuration outside BuildML.

**RAG** (core; ``buildml[rag]`` for semantic embeddings and reranking)
   Corpus ingest, retrieve, grounded generate with citations, evaluate,
   bundle. The default ``embedder="auto"`` selects semantic embeddings when
   the optional backend is available and hashing otherwise. Set
   ``embedder="hashing"`` explicitly for the local hashing workflow.
   Hosted vector database management is outside this API.

**AI operator** (``buildml[ai]``)
   Advisor, multi-step plan, confirmed execute (default), and explicit
   ``run_autonomous`` under hard caps. Execution follows configured tool
   permissions and operation limits.

Boundaries
----------

BuildML does not infer valid grouped or temporal evaluation boundaries.
It does not make causal claims from associations, EDA, or feature
importance. Causal effect estimation is a separate path that refuses to
run without an explicit estimand. There is no out-of-core sklearn
training mode. Checkpoints do not contain fitted models, and model
bundles do not contain the Session dataset or split history.

Each domain lists its supported operations and limitations. Managed cloud
identity, foundation-model pretraining, and hosted vector databases require
external systems.

Proof suite
-----------

Runnable scripts are available in ``examples/``. End-to-end evidence lives in
``proofs/``. Re-run the harness from a source checkout::

   python examples/classical_loan_loop.py
   python -m proofs._lib.run_all --tier all

Domain mappings are in :doc:`guide-index` and ``proofs/README.md``.
``buildml[production]`` remains best-effort on Python 3.13.

Where to read more
------------------

* Index / learning path: :doc:`guide-index` · :doc:`guides`
* Classical: :doc:`classical-end-to-end`, :doc:`leakage-cv-recipes`,
  :doc:`preprocess-depth`, :doc:`classical-diagnostics-search`
* Engines / EDA / artifacts: :doc:`engines-polars-duckdb`,
  :doc:`eda-teaching-studio`, :doc:`artifacts-checkpoints-bundles`
* Torch / speech / serve: :doc:`torch-deep`, :doc:`speech-asr-finetune`,
  :doc:`pretrained-backbones`, :doc:`serve-deploy`
* RAG / AI: :doc:`rag-deep`, :doc:`ai-operator-safety`,
  :doc:`ai-tools-operator-patterns`
