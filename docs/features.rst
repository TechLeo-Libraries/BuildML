Current capabilities
====================

BuildML 2.3.0a1 centers on :class:`buildml.Session` for tabular workflows.
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
  ``grid_search``, ``optuna_search``, and ``nested_cv_score``.

Models and diagnostics
----------------------

* Fit sklearn-compatible classifiers and regressors on the training partition.
* Compare named estimators under one partition and ranking metric.
* Evaluate classification or regression metrics with task baselines.
* Run cross-validation, grid search, randomized search, Optuna search, and
  nested CV without scoring Session test rows in inner loops.
* Inspect calibration, threshold tradeoffs, learning curves, permutation
  importance, error slices, and task-adaptive plot boards.

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
  (``transcribe_speech``, stub CI-safe or transformers Whisper-class) and
  speech classify finetune-lite (``make_speech_torch_loaders`` /
  ``fit_speech_torch`` / ``domain_adapt_speech_torch``). Integration path —
  not training a Whisper-scale foundation model from scratch
  (``refuse_speech_foundation_pretrain`` states that explicitly).
* **Pretrained backbones** (``buildml[vision]`` / ``buildml[speech]`` /
  ``buildml[pretrained]``): curated ResNet/ViT, Wav2Vec2, and Whisper-encoder
  hooks via ``load_pretrained_backbone`` with ``weights=none|mock|pretrained``
  (mock is CI-safe). Not a full HF/TorchVision zoo product.
* **Serve** (``buildml[serve]``): managed local FastAPI serving
  (``buildml-serve`` / ``python -m buildml.serving`` /
  ``Session.serve_bundle``) for classical pipeline bundles and TorchScript
  artifacts. Localhost bind by default; optional API-key/Bearer middleware —
  still not a managed cloud IAM product. Prefer TLS at a reverse proxy for
  non-local exposure. TorchServe directory packaging
  (``pack_torchserve``) and TensorRT ``trtexec`` plans
  (``prepare_tensorrt_export``) are recipe helpers only.
* **K8s DDP templates** (``emit_k8s_ddp_job`` / ``deploy/k8s``): example
  Indexed Job + torchrun YAML emitters — not live multi-cluster orchestration.
* **RAG** (``buildml[rag]``): corpus ingest, chunk, embed, retrieve, grounded
  ``rag_generate`` with citations, evaluate, upsert/delete, and bundle
  save/load. Hashing embeddings are the CI-safe default; semantic embedders are
  optional behind the same API.
* **AI operator** (``buildml[ai]``): advisor, multi-step plan, confirmed execute
  (default), and explicit ``ai_run_autonomous`` operator automation under hard
  caps (allowlist, max steps, blocked sample egress, transcript audit). Tool
  allowlist spans classical, RAG, and Torch (including nested search,
  multimodal loaders, and speech).

Boundaries
----------

BuildML does not infer valid grouped or temporal evaluation boundaries. It
does not make causal claims from associations or feature importance. There is
no ``OUT_OF_CORE`` sklearn training mode; engine choice does not make every
sklearn-facing operation out-of-core. Checkpoints do not contain fitted models,
and model bundles do not contain the Session dataset or split history. The AI
operator defaults to propose→confirm→execute; autonomy is opt-in automation
inside an allowlist, not unconstrained agency, and does not replace domain
review of roles, splits, or metrics. Image/audio multimodal fusion and a
separate speech ASR/finetune-lite path are shipped; torchrun multi-node DDP,
K8s Job templates, local managed serving (optional API keys), TorchServe
packaging, and TensorRT plans are alpha library helpers (not multi-cluster
orchestration, not a managed cloud, not Whisper-scale FM training from scratch).
