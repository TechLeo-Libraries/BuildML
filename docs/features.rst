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
does not make causal claims from associations or feature importance. There is
no ``OUT_OF_CORE`` sklearn training mode; engine choice does not make every
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
* Engines / EDA / artifacts: :doc:`engines-polars-duckdb`,
  :doc:`eda-teaching-studio`, :doc:`artifacts-checkpoints-bundles`
* Torch / speech / serve: :doc:`torch-deep`, :doc:`speech-asr-finetune`,
  :doc:`pretrained-backbones`, :doc:`serve-deploy`
* RAG / AI: :doc:`rag-deep`, :doc:`ai-operator-safety`,
  :doc:`ai-tools-operator-patterns`

Install honesty stays unchanged: PyPI ``buildml`` is legacy 1.x until a 2.x
wheel ships — use a GitHub or editable install for Session APIs above.
