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
  (``make_text_torch_loaders``), built-in MLP / text classifier when
  ``fit_torch`` omits a module, fold-local ``cross_validate_torch``, evaluation,
  and trainer bundles. Torch CV is not nested hyperparameter search.
* **RAG** (``buildml[rag]``): corpus ingest, chunk, embed, retrieve, grounded
  ``rag_generate`` with citations, evaluate, upsert/delete, and bundle
  save/load. Hashing embeddings are the CI-safe default; semantic embedders are
  optional behind the same API.
* **AI operator** (``buildml[ai]``): advisor, multi-step plan, confirmed execute
  with egress controls and a typed tool allowlist spanning classical, RAG
  (including generate), and Torch (including text loaders and fold-local CV).

Boundaries
----------

BuildML does not infer valid grouped or temporal evaluation boundaries. It
does not make causal claims from associations or feature importance. There is
no ``OUT_OF_CORE`` sklearn training mode; engine choice does not make every
sklearn-facing operation out-of-core. Checkpoints do not contain fitted models,
and model bundles do not contain the Session dataset or split history. The AI
operator guides workflows behind confirmations; it is not an autonomous agent
and does not replace domain review of roles, splits, or metrics. Multimodal
fusion and nested Torch search remain out of scope for this alpha.
