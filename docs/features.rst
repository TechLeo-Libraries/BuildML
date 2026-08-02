Current capabilities
====================

BuildML 2.4.0a1 centers on :class:`buildml.Session` for tabular workflows.
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
* Fit unsupervised clusterers on train (``fit_clusters``), assign holdout labels
  without refit, and evaluate geometric validity (optional external ARI/NMI).
  Optionally cluster train-fitted PCA components from ``reduce_dimensions``.
  Persist via ``buildml.unsupervised_bundle.v1``. Not a claim of ground-truth
  taxonomy. Dedicated anomaly/fraud scoring is ``fit_anomaly`` (separate path).
* Fit native ensembles on train (``fit_voting``, ``fit_stacking``,
  ``fit_blending``) with leakage-safe meta-learner fitting (stacking CV and
  blend holdouts stay inside train). Evaluate with classical supervised metrics
  (``evaluate_ensemble``). Persist via ``buildml.ensemble_bundle.v1``; classical
  ``save_pipeline`` also works because ``fit_result`` is set. Distinct from
  passing a single RandomForest to ``fit``.
* Run AutoML beyond single-estimator HPO (``run_automl``): joint model-family
  and fold-local preprocess-strategy search under a trial budget, with
  ``cv`` / ``nested`` / ``validation`` selection (Session test held out).
  Same Session-global preprocess refusal as classical CV/search. Optional
  voting of diverse top families. Evaluate via ``evaluate_automl``; persist
  via ``buildml.automl_bundle.v1``. Not NAS, not causal discovery, not a fully
  automated AI scientist. Randomized/grid need no extra; ``method='optuna'``
  reuses ``buildml[optuna]``.
* Fit classical forecasters on train (``fit_forecast``) with ``time`` role +
  ``time_split`` (random/stratified/group splits refused). Generate horizons
  (``generate_forecast``) and evaluate MAE/RMSE/MAPE
  (``evaluate_forecast``; rolling one-step or origin). Univariate by default;
  optional numeric exogenous columns with disclosed future-exog requirements.
  Persist via ``buildml.forecast_bundle.v1``. Not a full econometrics suite,
  not ARIMA productization, not a digital twin, and not a Torch sequence
  forecaster.
* Fit anomaly / fraud detectors on train (``fit_anomaly``): IsolationForest,
  LOF (novelty), One-Class SVM, or supervised HGB when a binary target exists.
  Score/flag holdout rows (``score_anomalies``) and evaluate with disclosed
  thresholds / alert rates plus optional PR-AUC and precision/recall@k
  (``evaluate_anomaly``). Novelty mode fits a normal-only train subset.
  Persist via ``buildml.anomaly_bundle.v1``. Distinct from EDA IsolationForest
  screens and ``handle_outliers`` fences. Not a graph-fraud or streaming
  platform; no causal fraud claims. ``ClusterPlan`` labels remain a separate
  structure API.
* Fit semi-supervised classifiers on scarce labeled + unlabeled train rows
  (``fit_semisupervised``): label propagation, label spreading, or self-training.
  Target NaNs mark unlabeled (sklearn ``-1`` internally). Predict/evaluate holdout
  with labeled-only metrics (``predict_semisupervised`` / ``evaluate_semisupervised``).
  Persist via ``buildml.semisupervised_bundle.v1``. Distinct from anomaly novelty
  and from self-supervised pretext. Validation/test never invent selection labels.
* Fit self-supervised tabular pretext on train (``fit_ssl_pretext``): masked
  tabular autoencoder lite, representation export (``transform_ssl``), supervised
  head attach (``finetune_ssl_head``), and labeled holdout eval (``evaluate_ssl``).
  Persist via ``buildml.selfsupervised_bundle.v1``. Not BERT-from-scratch.
  Vision/audio/speech freeze/finetune remains ``load_pretrained_backbone`` /
  ``attach_backbone_head`` under optional Torch/speech extras.
* Active learning human-in-the-loop loop on the train pool
  (``fit_active_learner`` → ``suggest_query`` → ``label_rows`` →
  ``evaluate_active_learning``). Uncertainty strategies (least confidence /
  margin / entropy), committee (bagged vote entropy), and
  ``expected_model_change_lite``. Pool = train target NaNs (never validation/test).
  Labels come from the user — core never invents an oracle. Budget caps enforced.
  Persist via ``buildml.activelearning_bundle.v1``. Distinct from semi-supervised
  propagation and self-supervised pretext.
* Online / continual learning via sklearn ``partial_fit``
  (``fit_online`` → ``partial_fit_online`` → ``evaluate_online`` /
  ``predict_online``). Warm-start on a train chunk; update on subsequent train
  chunks or role-aligned frames. Classifiers require ``classes=`` on first fit
  (explicit or discovered from the full train target vocabulary — labels only).
  Validation/test never used for updates. Silent full refits are refused
  (optional ``allow_refit_fallback`` is always disclosed). Optional lite
  chunk-vs-init mean-shift disclosure. Persist via ``buildml.online_bundle.v1``.
  Honesty: batch/stream-chunk Session updates — not a distributed streaming
  platform.
* Multi-task / multi-output learning via sklearn ``MultiOutput*`` /
  ``ClassifierChain`` / ``RegressorChain`` (``fit_multitask`` →
  ``predict_multitask`` / ``evaluate_multitask``). Requires ``>= 2`` same-type
  targets (multiple ``role='target'`` columns or explicit ``targets=``).
  Train-only fit; holdout evaluation reports per-task and unweighted aggregate
  metrics. Classical ``Session.fit`` remains single-target. Persist via
  ``buildml.multitask_bundle.v1``. Honesty: shared-feature multi-output — not a
  deep multi-head MTL research platform; mixed classification+regression is
  refused.
* Meta-learning via episodic few-shot protocols on Session task/group columns
  (``fit_metalearning`` → ``adapt_to_task`` / ``evaluate_metalearning``).
  Algorithms: ``prototypical`` (tabular nearest-centroid) and ``warm_start``
  (pooled sklearn init + support adapt). Meta-train uses train only; holdout
  evaluation prefers novel task ids and discloses overlaps. Persist via
  ``buildml.metalearning_bundle.v1``. Honesty: practical tabular few-shot /
  episodic Session protocol — not foundation-model meta-learning or
  MAML-at-scale.
* Federated learning via local FedAvg / FedProx simulation on a client/group
  column (``fit_federated`` → ``evaluate_federated`` / ``predict_federated``).
  Aggregates sklearn linear/SGD ``coef_`` / ``intercept_`` with sample-size
  weights. Local updates use train only; holdout evaluation never trains.
  Persist via ``buildml.federated_bundle.v1``. Honesty: local FL simulation for
  research/teaching/workflows — not a Flower/OpenFL network stack; not
  cryptographic secure aggregation.
* Fit probabilistic models on train (``fit_probabilistic``): BayesianRidge,
  GaussianProcess, or Naive Bayes, plus train-only split conformal intervals
  (``predict_interval``). Evaluate NLL / coverage (``evaluate_probabilistic``).
  Persist via ``buildml.probabilistic_bundle.v1``. Not PyMC/Stan/NumPyro MCMC.
* Fit assumption-declared causal estimators (``declare_causal_assumptions`` →
  ``fit_causal`` / ``estimate_causal`` / ``evaluate_causal`` / ``refute_causal``).
  Backdoor ATE via T-learner / IPW / AIPW; refuses incomplete assumptions and
  never invents causality from EDA. Persist via ``buildml.causal_bundle.v1``.
* Fit Graph ML node classifiers (``set_graph`` → ``fit_graph`` /
  ``predict_graph`` / ``evaluate_graph``): NetworkX classical node features
  (``buildml[graph]``) or pure-Torch GCN (``buildml[torch]``). Persist via
  ``buildml.graph_bundle.v1``. Not Neo4j, not PyG, not a KG product.
* Fit symbolic / neuro-symbolic rules (``fit_symbolic`` /
  ``fit_neuro_symbolic``): declared or induced if-then rules with traces;
  sklearn hybrid overlays. Persist via ``buildml.symbolic_bundle.v1``. Not
  Prolog/Z3 / fuzzy / expert-system products.
* Fit case-based reasoners (``fit_cbr`` → ``retrieve_cases`` / ``predict_cbr`` /
  ``retain_cbr``). Train-only case memory; distinct from RAG. Persist via
  ``buildml.cbr_bundle.v1``.
* Fit imitation / RL policies (``fit_imitation`` / ``fit_rl``): behavioral
  cloning and contextual bandits in core; optional Gymnasium REINFORCE-lite
  via ``buildml[rl]``. Persist via imitation/RL bundles. Not a robotics /
  multi-agent world-sim product.
* Fit TDA pipelines (``fit_tda`` → ``transform_tda`` / ``predict_tda``): local
  Vietoris–Rips + vectorization → sklearn head (``buildml[tda]``). Persist via
  ``buildml.tda_bundle.v1``.
* Fit recommenders (``fit_recommender`` → ``recommend`` /
  ``evaluate_recommender``): user/item CF + content with ranking metrics.
  Persist via ``buildml.recommender_bundle.v1``. Distinct from RAG / LTR.
* Fit learning-to-rank models (``fit_ranker`` → ``rank`` / ``evaluate_ranker``)
  on query–item feature rows. Persist via ``buildml.ranker_bundle.v1``.
* Fit knowledge-graph embeddings (``fit_kg`` → ``score_triples`` /
  ``predict_links`` / ``query_kg``): native TransE/DistMult or PyKEEN
  RotatE/ComplEx (``buildml[kg-industry]``) + symbolic query.
  Persist via ``buildml.kg_bundle.v1``. Not Neo4j / Graph ML / RAG.
* Fit decision policies (``fit_decision_policy`` → ``apply_decisions`` /
  ``evaluate_decisions``): thresholds, cost matrices, top-K / knapsack / LP.
  Persist via ``buildml.decision_bundle.v1``. Not a general OR platform.
* Fit synthetic-data generators (``fit_synthesizer`` → ``sample_synthetic`` /
  ``evaluate_synthetic``): bootstrap / Gaussian copula / SMOTE with fidelity +
  TSTR. Persist via ``buildml.synthetic_bundle.v1``. Not DP synthesis.

Explanation, audit, and reports
--------------------------------

* Explain any public operation before or after execution from a versioned
  operation catalog with concept links.
* Resolve workflow operations as done, available, blocked, or skipped.
* Preview operations with ``dry_run``; summarize history and heuristic risks.
* Export EDA, evaluation, diagnostic, and workflow walkthrough reports as
  local, self-contained HTML.
* Launch an optional local EDA Teaching Studio dashboard when
  ``buildml[dashboard]`` is installed.

Optional extras (same Session)
------------------------------

* **Torch** (``buildml[torch]``): tabular loaders, text/sequence loaders
  (``make_text_torch_loaders``), multimodal fusion for tabular/text/image/audio
  mixes (``make_multimodal_torch_loaders`` /
  ``make_image_multimodal_torch_loaders`` /
  ``make_audio_multimodal_torch_loaders``; path or array/waveform cells;
  train-only normalize stats; audio multimodal uses a small 1D-CNN branch),
  built-in MLP / text / fusion modules when ``fit_torch`` omits a module,
  fold-local ``cross_validate_torch``, nested ``nested_cv_torch`` /
  ``search_torch``, optional CUDA AMP, single-node and torchrun multi-node
  ``fit_torch_ddp`` (``multi_node=True``), TorchScript/ONNX ``export_torch``,
  evaluation, and trainer bundles (optional ``multimodal_preprocess`` meta;
  load does not rebuild loaders). Optional ``buildml[onnx]`` adds the
  ``onnx`` checker package. ``soundfile`` is included in ``buildml[torch]``
  (also via ``buildml[audio]``) for audio path cells.
* **Speech** (``buildml[speech]``): ASR transcription
  (``transcribe_speech``, stub CI-safe or transformers Whisper-class),
  ``evaluate_asr`` WER/CER scoring, and speech classify finetune-lite
  (``make_speech_torch_loaders`` / ``fit_speech_torch`` /
  ``domain_adapt_speech_torch``). Integration path — not training a
  Whisper-scale foundation model from scratch
  (``refuse_speech_foundation_pretrain`` states that explicitly).
* **Pretrained backbones** (``buildml[vision]`` / ``buildml[speech]`` /
  ``buildml[pretrained]``): expanded curated ResNet/ViT/audio/speech encoder
  hooks via ``load_pretrained_backbone`` / ``list_pretrained_backbones`` with
  ``weights=none|mock|pretrained`` (mock is CI-safe), plus
  ``attach_backbone_head`` for classify/probe heads. Not a full HF/TorchVision
  zoo product.
* **Serve** (``buildml[serve]``): managed local FastAPI serving
  (``buildml-serve`` / ``python -m buildml.serving`` /
  ``Session.serve_bundle``) for classical pipeline bundles and TorchScript
  artifacts, with ``/health``, ``/metadata``, ``/predict``, ``/predict/batch``,
  and OpenAPI docs. Localhost bind by default; optional API-key/Bearer
  middleware and optional local SSL cert/key pair — still not a managed cloud
  IAM / cert product. Prefer TLS at a reverse proxy for non-local exposure.
  TorchServe directory packaging (``pack_torchserve``) and TensorRT
  ``trtexec`` plans (``prepare_tensorrt_export``) are recipe helpers only.
* **K8s templates** (``emit_k8s_ddp_job`` / ``emit_k8s_serve_deployment`` /
  ``deploy/k8s``): Indexed Job + Service (+ optional ConfigMap) torchrun DDP
  emitters and serve Deployment templates — not live multi-cluster
  orchestration or a Helm/control-plane product.
* **RAG** (``buildml[rag]``): corpus ingest, chunk, embed, retrieve, grounded
  ``rag_generate`` with citations and cheap faithfulness hooks, evaluate,
  upsert/delete, and bundle save/load. Hashing embeddings are the CI-safe
  default; semantic embedders are optional behind the same API. Not a hosted
  vector-DB product.
* **AI operator** (``buildml[ai]``): advisor, multi-step plan, confirmed execute
  (default), and explicit ``ai_run_autonomous`` operator automation under hard
  caps (allowlist, max steps, blocked sample egress, transcript audit). Tool
  allowlist spans classical, RAG, and Torch (including nested search,
  multimodal loaders, speech, zoo heads, ASR eval, and K8s emitters).

Boundaries
----------

BuildML does not infer valid grouped or temporal evaluation boundaries. It
does not make causal claims from associations, EDA, or feature importance.
Causal effect estimation is a **separate** Session path
(``declare_causal_assumptions`` / ``fit_causal``) that refuses to run without
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
product-scope boundaries around those shipped paths — not stubs for missing
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

* **Tier A — 25/25:** one deep project per major domain (classical through IL/RL)
* **Tier B — 6/6:** Aegis, Harbor, Atlas, Pulse, Ledger, Nexus cross-domain products
* **Tier C — 25/25:** same-split industry twins writing ``comparison.json``
  (qualitative competitive bar 5-B — workflow parity over tiny metric gaps)

Re-run from a source checkout::

   python -m proofs._lib.run_all --tier all

Domain → proof mappings are in ``guides/README.md`` (rendered here as
:doc:`guide-index`) and ``proofs/README.md``. TDA prefers an editable
``pip install -e ".[tda]"``. ``buildml[production]`` remains best-effort on
Python 3.13 (environment markers skip broken upstream wheels).

Install honesty stays unchanged: PyPI ``buildml`` is legacy 1.x until a 2.x
wheel ships — use a GitHub or editable install for Session APIs above.
