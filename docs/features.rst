Current capabilities
====================

BuildML 2.5.0 centers on :class:`buildml.Session`. Classical tabular work
is the core path. Optional domains attach to the same Session, the same
history, and the same explain catalog.

This page is a map of what is shipped. Each domain has a one-line job,
the extra (if any), the main calls, and a hard line around what it is
not. Tutorials live in :doc:`guide-index`.

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

Core ``import buildml`` stays light. Domain methods attach to the same
Session. Each refined domain exposes a capability matrix for installed
backends.

Unsupervised
   Cluster on train (``session.unsupervised.fit``), assign holdout labels
   without refit, evaluate geometric validity. Optional PCA components
   from ``reduce_dimensions``. Bundle: ``buildml.unsupervised_bundle.v1``.
   Not a ground-truth taxonomy. Anomaly scoring is a separate path.
   Guides: :doc:`quickstart-unsupervised`, :doc:`unsupervised-deep`.

Ensembles
   ``session.ensemble.fit_voting`` / ``fit_stacking`` / ``fit_blending``.
   Stacking CV and blend holdouts stay inside train.
   Bundle: ``buildml.ensemble_bundle.v1``. Distinct from passing one
   RandomForest to ``fit``.
   Guides: :doc:`quickstart-ensemble`, :doc:`ensemble-deep`.

AutoML
   ``session.automl.run``: joint model-family and fold-local preprocess
   search under a trial budget (``cv`` / ``nested`` / ``validation``).
   Same Session-global preprocess refusal as classical CV. Extra:
   ``buildml[optuna]`` for the Optuna method; industry families via
   ``automl-industry``. Not NAS, not causal discovery.
   Guides: :doc:`quickstart-automl`, :doc:`automl-deep`.

Forecasting
   ``session.forecast.fit`` / ``generate`` / ``evaluate``. Needs a
   ``time`` role and ``time_split``. Univariate by default; optional
   numeric exogenous columns with disclosed future-exog requirements.
   Not a full econometrics suite, not ARIMA productization, not a Torch
   sequence forecaster.
   Guides: :doc:`quickstart-forecasting`, :doc:`forecasting-deep`.

Time-series analysis
   ``session.timeseries.analyze`` / ``decompose`` / ``diagnostics``.
   Analysis only: no forecast model is fitted here. Distinct from
   ``session.forecast.fit``. Depth behind ``buildml[timeseries]`` /
   ``timeseries-prophet`` / ``timeseries-ml``.
   Guides: :doc:`quickstart-timeseries-analysis`,
   :doc:`timeseries-analysis-deep`.

Anomaly / fraud
   ``session.anomaly.fit`` / ``score`` / ``evaluate`` / ``tune_threshold``.
   IsolationForest, LOF (novelty), One-Class SVM, or supervised HGB when
   a binary target exists. Distinct from EDA IsolationForest screens and
   ``handle_outliers``. Not a graph-fraud or streaming platform; no
   causal fraud claims.
   Guides: :doc:`quickstart-anomaly`, :doc:`anomaly-deep`.

Semi-supervised
   ``session.semisupervised.fit`` on scarce labeled plus unlabeled train
   rows (target NaNs mark unlabeled). Holdout metrics are labeled-only.
   Distinct from anomaly novelty and from self-supervised pretext.
   Guides: :doc:`quickstart-semisupervised`, :doc:`semisupervised-deep`.

Self-supervised
   ``session.ssl.fit_pretext`` (masked tabular autoencoder lite),
   ``transform``, ``finetune_head``, ``evaluate``. Not BERT-from-scratch.
   Vision / audio freeze-finetune remains ``session.dl.load_backbone``.
   Guides: :doc:`quickstart-selfsupervised`, :doc:`selfsupervised-deep`.

Active learning
   Train-pool query (``suggest_query``) then human labels
   (``label_rows``). Pool is train target NaNs. The core never invents
   an oracle. Budget caps are enforced.
   Guides: :doc:`quickstart-active-learning`, :doc:`active-learning-deep`.

Online / continual
   sklearn ``partial_fit`` on train chunks. Validation and test are never
   used for updates. Silent full refits are refused. Not a distributed
   streaming platform.
   Guides: :doc:`quickstart-online-learning`, :doc:`online-learning-deep`.

Multi-task
   sklearn ``MultiOutput*`` / chains. Requires two or more same-type
   targets. Mixed classification plus regression is refused. Not a deep
   multi-head research platform.
   Guides: :doc:`quickstart-multi-task`, :doc:`multi-task-deep`.

Meta-learning
   Episodic few-shot (``prototypical`` / ``warm_start``). Holdout
   evaluation prefers novel task ids and discloses overlaps. Not
   foundation-model meta-learning or MAML-at-scale.
   Guides: :doc:`quickstart-meta-learning`, :doc:`meta-learning-deep`.

Federated
   Local FedAvg / FedProx simulation on a client or group column. Not a
   Flower / OpenFL network stack; not cryptographic secure aggregation.
   Guides: :doc:`quickstart-federated`, :doc:`federated-deep`.

Probabilistic
   BayesianRidge / GaussianProcess / Naive Bayes, plus train-only split
   conformal intervals. Evaluate NLL / coverage. Not PyMC / Stan /
   NumPyro MCMC.
   Guides: :doc:`quickstart-probabilistic`, :doc:`probabilistic-deep`.

Causal
   ``session.causal.declare_assumptions`` then fit / estimate / evaluate
   / refute. Backdoor ATE via T-learner / IPW / AIPW. Refuses incomplete
   assumptions. Never invents causality from EDA.
   Guides: :doc:`quickstart-causal`, :doc:`causal-deep`.

Graph ML
   Node classify: NetworkX (``buildml[graph]``), pure-Torch GCN
   (``buildml[torch]``), optional PyG (``buildml[graph-pyg]``). Not Neo4j
   and not knowledge-graph embeddings (use ``session.kg``).
   Guides: :doc:`quickstart-graph`, :doc:`graph-deep`.

Symbolic
   Declared or induced if-then rules with traces; sklearn hybrid
   overlays. Not Prolog / Z3 / fuzzy / expert-system products.
   Guides: :doc:`quickstart-symbolic`, :doc:`symbolic-deep`.

Case-based reasoning
   Train-only case memory. Distinct from RAG.
   Guides: :doc:`quickstart-cbr`, :doc:`cbr-deep`.

Imitation + RL
   Behavioral cloning and contextual bandits in core. Gymnasium tabular
   TD and REINFORCE-lite via ``buildml[rl]``; SB3 via
   ``buildml[rl-industry]``. Not a robotics or multi-agent world-sim
   product.
   Guides: :doc:`quickstart-imitation-rl`, :doc:`imitation-rl-deep`.

TDA
   Local Vietoris–Rips plus vectorization, then a sklearn head
   (``buildml[tda]``).
   Guides: :doc:`quickstart-tda`, :doc:`tda-deep`.

Recommenders
   User / item CF plus content, with ranking metrics. Distinct from RAG
   and from LTR.
   Guides: :doc:`quickstart-recommenders`, :doc:`recommenders-deep`.

Learning to rank
   Query–item feature rows. sklearn fallback plus industry GBDT rankers.
   Distinct from RAG and from recommenders.
   Guides: :doc:`quickstart-ranking`, :doc:`ranking-deep`.

Knowledge graphs
   ``(h,r,t)`` TransE / DistMult, or PyKEEN RotatE / ComplEx via
   ``buildml[kg-industry]``, plus symbolic query. Not Neo4j, Graph ML, or
   RAG.
   Guides: :doc:`quickstart-kg`, :doc:`kg-deep`.

Decisions
   Thresholds, cost matrices, top-K / knapsack / LP. Not a general OR
   platform.
   Guides: :doc:`quickstart-optimize`, :doc:`optimize-deep`.

Fairness
   Observational DP / DI / EO on a holdout, optional stability bands,
   intersectional groups. Opt-in suggest helpers return thresholds or
   weights only. Not automatic mitigation and not legal certification.
   Optional SHAP via ``explain_shap`` (``buildml[shap]``).
   Guides: :doc:`quickstart-fairness`, :doc:`fairness-deep`.

Synthetic data
   Bootstrap / copula / SMOTE; optional SDV backends. Fidelity and TSTR.
   Not differential-privacy synthesis.
   Guides: :doc:`quickstart-synthetic`, :doc:`synthetic-deep`.

NLP
   A text column on the Session dataset: corpus profile, document
   classify, token attribution, topics, keyphrases, summaries, entities,
   sentiment, language. Bag-of-n-grams by default; encoders via
   ``buildml[nlp]``. Not multi-label, not span labelling, not generation,
   not translation, not transformer fine-tuning (Torch text path), and
   not retrieval for generation (RAG).
   Guides: :doc:`quickstart-nlp`, :doc:`nlp-deep`.

Teaching and reports
--------------------

* Explain any public operation before or after execution from a
  versioned catalog.
* Resolve workflow operations as done, available, blocked, or skipped.
* Preview with ``dry_run``; summarize history and heuristic risks.
* Export EDA, evaluation, diagnostic, and walkthrough reports as local
  HTML. ``html_format="research"`` writes the static industry readiness
  sheet. ``html_format="studio"`` writes an offline app snapshot.
* Launch the local Industry EDA App when ``buildml[dashboard]`` is
  installed (cockpit, readiness gates, concept academy). Gate marks are
  browser-tab UI state and are never persisted.

Optional stacks (same Session)
------------------------------

**Torch** (``buildml[torch]``)
   Tabular, text, image, and audio loaders; built-in MLP / text / fusion
   modules; fold-local CV / search / nested; AMP; single-node and
   torchrun multi-node DDP; TorchScript / ONNX export.

**Speech** (``buildml[speech]``)
   ASR transcription (stub or transformers), WER / CER, speech classify
   finetune-lite. Not training a Whisper-scale foundation model from
   scratch.

**Pretrained backbones** (``buildml[vision]`` / ``[speech]`` / ``[pretrained]``)
   Curated ResNet / ViT / audio / speech hooks with
   ``weights=none|mock|pretrained``. Not a full Hugging Face / TorchVision
   zoo product.

**Serve** (``buildml[serve]``)
   Local FastAPI for classical pipeline bundles and TorchScript
   (``/health``, ``/metadata``, ``/predict``, ``/predict/batch``).
   Localhost bind by default. Not managed cloud IAM.

**RAG** (``buildml[rag]``)
   Corpus ingest, retrieve, grounded generate with citations, evaluate,
   bundle. Hashing embeddings are the default; semantic embedders are
   optional. Not a hosted vector-DB product.

**AI operator** (``buildml[ai]``)
   Advisor, multi-step plan, confirmed execute (default), and explicit
   ``run_autonomous`` under hard caps. Not unconstrained agency.

Boundaries
----------

BuildML does not infer valid grouped or temporal evaluation boundaries.
It does not make causal claims from associations, EDA, or feature
importance. Causal effect estimation is a separate path that refuses to
run without an explicit estimand. There is no out-of-core sklearn
training mode. Checkpoints do not contain fitted models, and model
bundles do not contain the Session dataset or split history.

The honesty lines next to each domain ("not a full zoo", "not managed
cloud IAM", "not FM-from-scratch") are product-scope boundaries around
shipped paths. They are not stubs for missing APIs.

Proof suite
-----------

End-to-end evidence lives in the repository ``proofs/`` directory. It is
not a smoke folder. Re-run from a source checkout::

   python -m proofs._lib.run_all --tier all

Domain-to-proof mappings are in :doc:`guide-index` and
``proofs/README.md``. ``buildml[production]`` remains best-effort on
Python 3.13.

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
