Current capabilities
====================

BuildML 2.5.0 centers on :class:`buildml.Session` for tabular workflows.
Optional Torch, RAG, and AI operator paths reuse the same Session state, history,
and explain catalog.

Data and workflow state
-----------------------

* Ingest Pandas DataFrames and CSV, Parquet, Arrow, and Excel sources.
* Record source detection, scale estimates, mode and engine choices, and
  loading warnings in an ingest report.
* Assign feature, target, identifier, group, time, weight, and ignored roles.
* Create random, stratified, group-aware, or chronological partitions, or
  inject externally designed row memberships.
* Save and validate checkpoints containing data, roles, partitions, history,
  optional preprocess plans, and an integrity manifest.
* Switch or materialize through Pandas, Polars, or DuckDB engines where
  installed; DuckDB connections close via ``close_native`` or context managers.

Preparation
-----------

* Drop columns and extract date parts.
* Fit imputation, categorical encoding, scaling, outlier handling, binning,
  feature selection, text features, and dimensionality reduction on training
  rows.
* Resample only the training partition when the ``imbalanced`` extra is
  installed.
* Apply registered custom transforms (Session-global; not fold-local in CV).
* Use ``PreprocessRecipe`` for fold-local preparation inside ``cv_score``,
  ``grid_search``, ``optuna_search``, ``evolutionary_search``, and ``nested_cv_score``.

Models and diagnostics
----------------------

* Fit sklearn-compatible classifiers and regressors on the training partition.
* Compare named estimators under one partition and ranking metric.
* Evaluate classification or regression metrics with task baselines.
* Run cross-validation, grid search, randomized search, Optuna search, and
  nested CV without scoring Session test rows in inner loops.
* Inspect calibration, threshold tradeoffs, learning curves, permutation
  importance, error slices, and task-adaptive plot boards.
* Fit unsupervised clusterers on train (``session.unsupervised.fit``), assign holdout labels
  without refit, and evaluate geometric validity (optional external ARI/NMI).
  Optionally cluster train-fitted PCA components from ``reduce_dimensions``.
  Persist via ``buildml.unsupervised_bundle.v1``. Not a claim of ground-truth
  taxonomy. Dedicated anomaly/fraud scoring is ``session.anomaly.fit`` (separate path).
* Fit native ensembles on train (``session.ensemble.fit_voting``, ``session.ensemble.fit_stacking``,
  ``session.ensemble.fit_blending``) with leakage-safe meta-learner fitting (stacking CV and
  blend holdouts stay inside train). Evaluate with classical supervised metrics
  (``session.ensemble.evaluate``). Persist via ``buildml.ensemble_bundle.v1``; classical
  ``save_pipeline`` also works because ``fit_result`` is set. Distinct from
  passing a single RandomForest to ``fit``.
* Run AutoML beyond single-estimator HPO (``session.automl.run``): joint model-family
  and fold-local preprocess-strategy search under a trial budget, with
  ``cv`` / ``nested`` / ``validation`` selection (Session test held out).
  Same Session-global preprocess refusal as classical CV/search. Optional
  voting of diverse top families. Evaluate via ``session.automl.evaluate``; persist
  via ``buildml.automl_bundle.v1``. Not NAS, not causal discovery, not a fully
  automated AI scientist. Randomized/grid need no extra; ``method='optuna'``
  reuses ``buildml[optuna]``.
* Fit classical forecasters on train (``session.forecast.fit``) with ``time`` role +
  ``time_split`` (random/stratified/group splits refused). Generate horizons
  (``session.forecast.generate``) and evaluate MAE/RMSE/MAPE
  (``session.forecast.evaluate``; rolling one-step or origin). Univariate by default;
  optional numeric exogenous columns with disclosed future-exog requirements.
  Persist via ``buildml.forecast_bundle.v1``. Not a full econometrics suite,
  not ARIMA productization, not a digital twin, and not a Torch sequence
  forecaster.
* Analyse ordered series on train (``session.timeseries.analyze`` / ``session.timeseries.decompose`` /
  ``session.timeseries.diagnostics``) with a ``time`` role: stationarity, seasonality hints,
  change points, and decomposition reports via ``session.timeseries.capability_matrix``.
  Analysis-only floor (no forecast model is fitted here). Distinct from
  ``session.forecast.fit``. Optional depth behind ``buildml[timeseries]`` /
  ``timeseries-prophet`` / ``timeseries-ml``.
* Fit anomaly / fraud detectors on train (``session.anomaly.fit``): IsolationForest,
  LOF (novelty), One-Class SVM, or supervised HGB when a binary target exists.
  Score/flag holdout rows (``session.anomaly.score``) and evaluate with disclosed
  thresholds / alert rates plus optional PR-AUC and precision/recall@k
  (``session.anomaly.evaluate``). Novelty mode fits a normal-only train subset.
  Persist via ``buildml.anomaly_bundle.v1``. Distinct from EDA IsolationForest
  screens and ``handle_outliers`` fences. Not a graph-fraud or streaming
  platform; no causal fraud claims. ``ClusterPlan`` labels remain a separate
  structure API.
* Fit semi-supervised classifiers on scarce labeled + unlabeled train rows
  (``session.semisupervised.fit``): label propagation, label spreading, or self-training.
  Target NaNs mark unlabeled (sklearn ``-1`` internally). Predict/evaluate holdout
  with labeled-only metrics (``session.semisupervised.predict`` / ``session.semisupervised.evaluate``).
  Persist via ``buildml.semisupervised_bundle.v1``. Distinct from anomaly novelty
  and from self-supervised pretext. Validation/test never invent selection labels.
* Fit self-supervised tabular pretext on train (``session.ssl.fit_pretext``): masked
  tabular autoencoder lite, representation export (``session.ssl.transform``), supervised
  head attach (``session.ssl.finetune_head``), and labeled holdout eval (``session.ssl.evaluate``).
  Persist via ``buildml.selfsupervised_bundle.v1``. Not BERT-from-scratch.
  Vision/audio/speech freeze/finetune remains ``session.dl.load_backbone`` /
  ``session.dl.attach_head`` under optional Torch/speech extras.
* Active learning human-in-the-loop loop on the train pool
  (``session.active_learning.fit`` → ``session.active_learning.suggest_query`` → ``session.active_learning.label_rows`` →
  ``session.active_learning.evaluate``). Uncertainty strategies (least confidence /
  margin / entropy), committee (bagged vote entropy), and
  ``expected_model_change_lite``. Pool = train target NaNs (never validation/test).
  Labels come from the user: core never invents an oracle. Budget caps enforced.
  Persist via ``buildml.activelearning_bundle.v1``. Distinct from semi-supervised
  propagation and self-supervised pretext.
* Online / continual learning via sklearn ``partial_fit``
  (``session.online.fit`` → ``session.online.partial_fit`` → ``session.online.evaluate`` /
  ``session.online.predict``). Warm-start on a train chunk; update on subsequent train
  chunks or role-aligned frames. Classifiers require ``classes=`` on first fit
  (explicit or discovered from the full train target vocabulary: labels only).
  Validation/test never used for updates. Silent full refits are refused
  (optional ``allow_refit_fallback`` is always disclosed). Optional lite
  chunk-vs-init mean-shift disclosure. Persist via ``buildml.online_bundle.v1``.
  Honesty: batch/stream-chunk Session updates: not a distributed streaming
  platform.
* Multi-task / multi-output learning via sklearn ``MultiOutput*`` /
  ``ClassifierChain`` / ``RegressorChain`` (``session.multitask.fit`` →
  ``session.multitask.predict`` / ``session.multitask.evaluate``). Requires ``>= 2`` same-type
  targets (multiple ``role='target'`` columns or explicit ``targets=``).
  Train-only fit; holdout evaluation reports per-task and unweighted aggregate
  metrics. Classical ``Session.fit`` remains single-target. Persist via
  ``buildml.multitask_bundle.v1``. Honesty: shared-feature multi-output: not a
  deep multi-head MTL research platform; mixed classification+regression is
  refused.
* Meta-learning via episodic few-shot protocols on Session task/group columns
  (``session.metalearning.fit`` → ``session.metalearning.adapt`` / ``session.metalearning.evaluate``).
  Algorithms: ``prototypical`` (tabular nearest-centroid) and ``warm_start``
  (pooled sklearn init + support adapt). Meta-train uses train only; holdout
  evaluation prefers novel task ids and discloses overlaps. Persist via
  ``buildml.metalearning_bundle.v1``. Honesty: practical tabular few-shot /
  episodic Session protocol: not foundation-model meta-learning or
  MAML-at-scale.
* Federated learning via local FedAvg / FedProx simulation on a client/group
  column (``session.federated.fit`` → ``session.federated.evaluate`` / ``session.federated.predict``).
  Aggregates sklearn linear/SGD ``coef_`` / ``intercept_`` with sample-size
  weights. Local updates use train only; holdout evaluation never trains.
  Persist via ``buildml.federated_bundle.v1``. Honesty: local FL simulation for
  research/teaching/workflows: not a Flower/OpenFL network stack; not
  cryptographic secure aggregation.
* Fit probabilistic models on train (``session.probabilistic.fit``): BayesianRidge,
  GaussianProcess, or Naive Bayes, plus train-only split conformal intervals
  (``session.probabilistic.predict_interval``). Evaluate NLL / coverage (``session.probabilistic.evaluate``).
  Persist via ``buildml.probabilistic_bundle.v1``. Not PyMC/Stan/NumPyro MCMC.
* Fit assumption-declared causal estimators (``session.causal.declare_assumptions`` →
  ``session.causal.fit`` / ``session.causal.estimate`` / ``session.causal.evaluate`` / ``session.causal.refute``).
  Backdoor ATE via T-learner / IPW / AIPW; refuses incomplete assumptions and
  never invents causality from EDA. Persist via ``buildml.causal_bundle.v1``.
* Fit Graph ML node classifiers (``session.graph.set_spec`` → ``session.graph.fit`` /
  ``session.graph.predict`` / ``session.graph.evaluate``): NetworkX classical node features
  (``buildml[graph]``), pure-Torch GCN (``buildml[torch]``), or optional PyG
  GCN/SAGE/GAT via ``buildml[graph-pyg]``. Persist via
  ``buildml.graph_bundle.v1``. Not a Neo4j product and not a knowledge-graph
  embedding path (use ``session.kg.fit``).
* Fit symbolic / neuro-symbolic rules (``session.symbolic.fit`` /
  ``session.symbolic.fit_neuro``): declared or induced if-then rules with traces;
  sklearn hybrid overlays. Persist via ``buildml.symbolic_bundle.v1``. Not
  Prolog/Z3 / fuzzy / expert-system products.
* Fit case-based reasoners (``session.cbr.fit`` → ``session.cbr.retrieve`` / ``session.cbr.predict`` /
  ``session.cbr.retain``). Train-only case memory; distinct from RAG. Persist via
  ``buildml.cbr_bundle.v1``.
* Fit imitation / RL policies (``session.rl.fit_imitation`` / ``session.rl.fit``): behavioral
  cloning and contextual bandits in core; optional Gymnasium tabular TD control
  (Q-learning / SARSA / Expected SARSA / Double Q-learning) and REINFORCE-lite
  via ``buildml[rl]``; SB3 PPO/DQN/A2C via ``buildml[rl-industry]``. Persist via
  imitation/RL bundles. Not a robotics / multi-agent world-sim product.
* Fit TDA pipelines (``session.tda.fit`` → ``session.tda.transform`` / ``session.tda.predict``): local
  Vietoris–Rips + vectorization → sklearn head (``buildml[tda]``). Persist via
  ``buildml.tda_bundle.v1``.
* Fit recommenders (``session.recommender.fit`` → ``session.recommender.recommend`` /
  ``session.recommender.evaluate``): user/item CF + content with ranking metrics.
  Persist via ``buildml.recommender_bundle.v1``. Distinct from RAG / LTR.
* Fit learning-to-rank models (``session.ranking.fit`` → ``session.ranking.rank`` / ``session.ranking.evaluate``)
  on query–item feature rows. Persist via ``buildml.ranker_bundle.v1``.
* Fit knowledge-graph embeddings (``session.kg.fit`` → ``session.kg.score_triples`` /
  ``session.kg.predict_links`` / ``session.kg.query``): native TransE/DistMult or PyKEEN
  RotatE/ComplEx (``buildml[kg-industry]``) + symbolic query.
  Persist via ``buildml.kg_bundle.v1``. Not Neo4j / Graph ML / RAG.
* Fit decision policies (``session.decision.fit`` → ``session.decision.apply`` /
  ``session.decision.evaluate``): thresholds, cost matrices, top-K / knapsack / LP.
  Persist via ``buildml.decision_bundle.v1``. Not a general OR platform.
* Fit synthetic-data generators (``session.synthetic.fit`` → ``session.synthetic.sample`` /
  ``session.synthetic.evaluate``): bootstrap / Gaussian copula / SMOTE with fidelity +
  TSTR. Persist via ``buildml.synthetic_bundle.v1``. Not DP synthesis.
* Report observational group disparity on a holdout
  (``session.fairness.evaluate`` / ``session.fairness.attach_to_last_eval``):
  demographic parity difference, disparate impact ratio, equalized odds
  TPR/FPR gaps, per-group classical metrics, optional bootstrap /
  stratified-subsample stability bands, and intersectional sensitive
  columns (composite group keys). Opt-in post-hoc helpers
  (``session.fairness.suggest_thresholds``,
  ``session.fairness.suggest_reweighing``) return thresholds or sample
  weights only — not automatic mitigation and not legal certification.
  Requires a fitted classifier and caller-declared sensitive column(s).
  Optional SHAP attribution via ``explain_shap`` (``buildml[shap]``).
* Model and analyse a text column on the Session dataset
  (``session.nlp.profile_corpus`` → ``session.nlp.fit_classifier`` → ``session.nlp.predict`` /
  ``session.nlp.evaluate`` / ``session.nlp.interpret``), plus
  unsupervised description on the same split (``session.nlp.fit_topics`` /
  ``session.nlp.assign_topics``, ``session.nlp.extract_keyphrases``, ``session.nlp.analyze_sentiment``,
  ``session.nlp.extract_entities``, ``session.nlp.summarize``, ``session.nlp.detect_language``). Bag-of-n-grams
  is the always-available default; frozen sentence-transformer and transformer
  encoders come with ``buildml[nlp]``, spaCy NER with
  ``buildml[nlp-industry]``. Normalization is deterministic; vocabulary,
  document frequencies, topic components, and heads are train-only. Persist via
  ``buildml.nlp_bundle.v1``. Honesty: single-label document classification and
  analysis: not multi-label, not span/sequence labelling, not generation or
  abstractive summarization, not translation, not transformer fine-tuning
  (Torch text path), and not document retrieval for generation
  (``buildml.rag``).

Explanation, audit, and reports
--------------------------------

* Explain any public operation before or after execution from a versioned
  operation catalog with concept links.
* Resolve workflow operations as done, available, blocked, or skipped.
* Preview operations with ``dry_run``; summarize history and heuristic risks.
* Export EDA, evaluation, diagnostic, and workflow walkthrough reports as
  local, self-contained HTML. ``html_format="research"`` writes BUILDML STATIC
  EDA (Industry readiness sheet) with Offline HTML as the sole header export
  (no CSV or PDF briefing buttons on Static);
  ``html_format="studio"`` writes an offline Industry App snapshot.
* Launch an optional local Industry EDA App (Cockpit spine 01-08 / Readiness
  Gates / Concept Academy covering ~204 BuildML concept notes with
  dataset-adaptive Session examples, plus domain boards) when
  ``buildml[dashboard]`` is installed. Offline HTML is the primary app-header
  export; CSV/PDF remain on App API routes for automation. Gate session marks
  are UI-only and are never persisted.

Optional extras (same Session)
------------------------------

* **Torch** (``buildml[torch]``): tabular loaders, text/sequence loaders
  (``session.dl.make_text_loaders``), multimodal fusion for tabular/text/image/audio
  mixes (``session.dl.make_multimodal_loaders`` /
  ``session.dl.make_image_loaders`` /
  ``session.dl.make_audio_loaders``; path or array/waveform cells;
  train-only normalize stats; audio multimodal uses a small 1D-CNN branch),
  built-in MLP / text / fusion modules when ``session.dl.fit`` omits a module,
  fold-local ``session.dl.cross_validate``, nested ``session.dl.nested_cv`` /
  ``session.dl.search``, optional CUDA AMP, single-node and torchrun multi-node
  ``session.dl.fit_ddp`` (``multi_node=True``), TorchScript/ONNX ``session.dl.export``,
  evaluation, and trainer bundles (optional ``multimodal_preprocess`` meta;
  load does not rebuild loaders). Optional ``buildml[onnx]`` adds the
  ``onnx`` checker package. ``soundfile`` is included in ``buildml[torch]``
  (also via ``buildml[audio]``) for audio path cells.
* **Speech** (``buildml[speech]``): ASR transcription
  (``session.dl.transcribe``, stub CI-safe or transformers Whisper-class),
  ``session.dl.evaluate_asr`` WER/CER scoring, and speech classify finetune-lite
  (``session.dl.make_speech_loaders`` / ``session.dl.fit_speech`` /
  ``session.dl.domain_adapt_speech``). Integration path: not training a
  Whisper-scale foundation model from scratch
  (``session.dl.refuse_speech_pretrain`` states that explicitly).
* **Pretrained backbones** (``buildml[vision]`` / ``buildml[speech]`` /
  ``buildml[pretrained]``): expanded curated ResNet/ViT/audio/speech encoder
  hooks via ``session.dl.load_backbone`` / ``list_pretrained_backbones`` with
  ``weights=none|mock|pretrained`` (mock is CI-safe), plus
  ``session.dl.attach_head`` for classify/probe heads. Not a full HF/TorchVision
  zoo product.
* **Serve** (``buildml[serve]``): managed local FastAPI serving
  (``buildml-serve`` / ``python -m buildml.serving`` /
  ``session.dl.serve``) for classical pipeline bundles and TorchScript
  artifacts, with ``/health``, ``/metadata``, ``/predict``, ``/predict/batch``,
  and OpenAPI docs. Localhost bind by default; optional API-key/Bearer
  middleware and optional local SSL cert/key pair: still not a managed cloud
  IAM / cert product. Prefer TLS at a reverse proxy for non-local exposure.
  TorchServe directory packaging (``session.dl.pack_torchserve``) and TensorRT
  ``trtexec`` plans (``session.dl.prepare_tensorrt``) are recipe helpers only.
* **K8s templates** (``session.dl.emit_k8s_ddp`` / ``session.dl.emit_k8s_serve`` /
  ``deploy/k8s``): Indexed Job + Service (+ optional ConfigMap) torchrun DDP
  emitters and serve Deployment templates: not live multi-cluster
  orchestration or a Helm/control-plane product.
* **RAG** (``buildml[rag]``): corpus ingest, chunk, embed, retrieve, grounded
  ``session.rag.generate`` with citations and cheap faithfulness hooks, evaluate,
  upsert/delete, and bundle save/load. Hashing embeddings are the CI-safe
  default; semantic embedders are optional behind the same API. Not a hosted
  vector-DB product.
* **AI operator** (``buildml[ai]``): advisor, multi-step plan, confirmed execute
  (default), and explicit ``session.ai.run_autonomous`` operator automation under hard
  caps (allowlist, max steps, blocked sample egress, transcript audit). Tool
  allowlist spans classical, RAG, and Torch (including nested search,
  multimodal loaders, speech, zoo heads, ASR eval, and K8s emitters).

Boundaries
----------

BuildML does not infer valid grouped or temporal evaluation boundaries. It
does not make causal claims from associations, EDA, or feature importance.
Causal effect estimation is a **separate** Session path
(``session.causal.declare_assumptions`` / ``session.causal.fit``) that refuses to run without
an explicit estimand and identification acknowledgements. There is no
``OUT_OF_CORE`` sklearn training mode; engine choice does not make every
sklearn-facing operation out-of-core. Checkpoints do not contain fitted models,
and model bundles do not contain the Session dataset or split history. The AI
operator defaults to propose→confirm→execute; autonomy is opt-in automation
inside an allowlist, not unconstrained agency, and does not replace domain
review of roles, splits, or metrics.

**Shipped vs scope:** image/audio multimodal fusion (including preprocess
restore), speech ASR/finetune-lite + WER/CER, curated pretrained backbone hooks
with attachable heads, torchrun multi-node DDP, K8s Job/Deployment templates,
local managed serving (API keys, metadata/batch/OpenAPI, optional local SSL),
TorchServe/TRT recipes, and local RAG generate/faithfulness are real library
paths. The matching honesty lines (“not a full zoo”, “not managed cloud IAM”,
“not live multi-cluster”, “not FM-from-scratch”, “not a hosted vector DB”) are
product-scope boundaries around those shipped paths: not stubs for missing
APIs.

Where to read more
------------------

Capability narratives with runnable examples live in the guide set (canonical
Markdown under ``guides/``, also rendered here):

* Index / learning path: :doc:`guide-index` · :doc:`guides`
* Classical: :doc:`classical-end-to-end`, :doc:`leakage-cv-recipes`,
  :doc:`preprocess-depth`, :doc:`classical-diagnostics-search`
* Unsupervised: :doc:`quickstart-unsupervised`, :doc:`unsupervised-deep`
* Ensembles: :doc:`quickstart-ensemble`, :doc:`ensemble-deep`
* AutoML: :doc:`quickstart-automl`, :doc:`automl-deep`
* Forecasting: :doc:`quickstart-forecasting`, :doc:`forecasting-deep`
* Time-series analysis: :doc:`quickstart-timeseries-analysis`,
  :doc:`timeseries-analysis-deep`
* Anomaly / fraud: :doc:`quickstart-anomaly`, :doc:`anomaly-deep`
* Semi-supervised: :doc:`quickstart-semisupervised`, :doc:`semisupervised-deep`
* Self-supervised: :doc:`quickstart-selfsupervised`, :doc:`selfsupervised-deep`
* Active learning: :doc:`quickstart-active-learning`, :doc:`active-learning-deep`
* Online / continual: :doc:`quickstart-online-learning`, :doc:`online-learning-deep`
* Multi-task / multi-output: :doc:`quickstart-multi-task`, :doc:`multi-task-deep`
* Meta-learning: :doc:`quickstart-meta-learning`, :doc:`meta-learning-deep`
* Federated learning: :doc:`quickstart-federated`, :doc:`federated-deep`
* Bayesian / probabilistic: :doc:`quickstart-probabilistic`, :doc:`probabilistic-deep`
* Causal ML (assumption-declared): :doc:`quickstart-causal`, :doc:`causal-deep`
* Graph ML: :doc:`quickstart-graph`, :doc:`graph-deep`
* Symbolic / CBR / IL+RL / TDA: :doc:`quickstart-symbolic`, :doc:`quickstart-cbr`,
  :doc:`quickstart-imitation-rl`, :doc:`quickstart-tda`
* Recommenders / LTR / KG / decisions / synthetic: :doc:`quickstart-recommenders`,
  :doc:`quickstart-ranking`, :doc:`quickstart-kg`, :doc:`quickstart-optimize`,
  :doc:`quickstart-synthetic`
* NLP (text column on the Session dataset): :doc:`quickstart-nlp`, :doc:`nlp-deep`
* Engines / EDA / artifacts: :doc:`engines-polars-duckdb`,
  :doc:`eda-teaching-studio`, :doc:`artifacts-checkpoints-bundles`
* Torch / speech / serve: :doc:`torch-deep`, :doc:`speech-asr-finetune`,
  :doc:`pretrained-backbones`, :doc:`serve-deploy`
* RAG / AI: :doc:`rag-deep`, :doc:`ai-operator-safety`,
  :doc:`ai-tools-operator-patterns`

Proof suite (Tier A/B/C)
------------------------

End-to-end evidence that Session domains work with honest splits and holdout
metrics lives in the repository ``proofs/`` directory (not thin smoke):

* **Tier A: 63/63:** one deep project per major domain (classical through NLP,
  plus ensembles, Torch tabular/text, Industry EDA adaptability, and a
  **REAL_PUBLIC_DATASET** cohort: breast_cancer / diabetes / wine / Adult fairness)
* **Tier B: 36/36:** baseline Aegis/Harbor/Atlas/Pulse/Ledger/Nexus plus 30
  expansion products (Meridian, Helix, Citadel, Nova, Zenith, …)
* **Tier C: 58/62:** same-split industry twins writing ``comparison.json``
  (qualitative competitive bar 5-B: workflow parity over tiny metric gaps;
  real-public cohort may be Tier A evidence-first)

Re-run from a source checkout::

   python -m proofs._lib.run_all --tier all

Domain → proof mappings are in ``guides/README.md`` (rendered here as
:doc:`guide-index`) and ``proofs/README.md``. TDA prefers an editable
``pip install -e ".[tda]"``. ``buildml[production]`` remains best-effort on
Python 3.13 (environment markers skip broken upstream wheels).

Install with ``pip install buildml`` for Session 2.5.x. Legacy 1.x remains
available only under an explicit pin (``buildml==1.0.9``).
