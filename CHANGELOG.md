# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
with pre-release tags for alpha (`aN`) builds.

## [Unreleased]

### Added

- Cost-sensitive `tune_threshold(fp_cost=..., fn_cost=...)` with recommended
  threshold, expected cost, and structured operating points.
- Stronger `error_slices`: multi-column segments, richer metrics, small-n
  handling, optional HTML export.
- Richer `dry_run` / `summarize_history` audit UX (ranked risks, prerequisite
  graph summary, suggested next ops) surfaced on walkthrough HTML.

## [2.2.0a1] — RAG alpha — 2026-08-01

First retrieval (RAG) alpha on the BuildML 2.x `Session` API. Exit criteria and
known limits are defined in [`docs/rag-alpha-gate.md`](docs/rag-alpha-gate.md).
Classical alpha remains at `2.0.0a1`; DL alpha remains at `2.1.0a1`. This line
adds optional retrieve / evaluate / bundle — **not** generate or an LLM operator.

### Added

- Optional `buildml.rag` domain: corpus ingest → chunk → embed/index → retrieve →
  evaluate → upsert/delete → `buildml.rag_bundle.v1` save/load.
- Session delegates: `rag_ingest_corpus`, `rag_chunk`, `rag_embed_and_index`,
  `rag_retrieve`, `rag_evaluate`, `rag_upsert`, `rag_delete`, `save_rag_bundle`,
  `load_rag_bundle`; result slots `rag_index_result` / `rag_retrieve_result` /
  `rag_eval_result`.
- Default embedder `buildml.hashing_embed.v1` (CPU hashing) and NumPy cosine
  store; optional sentence-transformers / cross-encoder behind `buildml[rag]`.
- Hybrid retrieve (dense + BM25, RRF or weighted), metadata filters, eval depth
  (recall@k, MRR, nDCG@k, hit-rate@k, document|chunk relevance,
  `compare_retrieval_configs`), walkthrough `rag_status`.
- Explain catalog/concept coverage; RAG quickstart, glossary terms, alpha gate,
  and release checklist.
- CI `rag` job (Python 3.11–3.12) with unit, integration, and RAG alpha smoke.

### Known limits (RAG alpha)

- Hashing default is lexical/hashed, not semantic retrieval quality.
- Local-first NumPy store; no hosted vector-DB product path.
- **No generate (`rag_generate`) and no LLM operator / agent product
  (`buildml.ai`).**
- No Teaching Studio RAG cockpit redesign; structured results + `rag_status` only.
- CPU merge gate for RAG CI; GPU embed/rerank optional when available.
- Session checkpoints never embed the vector index.
- Public RAG APIs and bundle formats may change before a stable RAG release.

### Verification

- Gate checklist: `docs/rag-alpha-gate.md` sign-off section.
- Tag only after remote CI is green (see `docs/release-checklist-rag-a1.md`).

## [2.1.0a1] — DL alpha — 2026-08-01

First deep-learning alpha on the BuildML 2.x `Session` API. Exit criteria and
known limits are defined in [`docs/dl-alpha-gate.md`](docs/dl-alpha-gate.md).
Classical alpha remains documented at `2.0.0a1`; this line adds optional Torch.

### Added

- Optional `buildml.dl` domain behind `buildml[torch]` (alias `buildml[dl]`):
  tabular partition → DataLoaders → train loop → evaluate → trainer bundle.
- Session delegates: `make_torch_loaders`, `fit_torch`, `evaluate_torch`,
  `torch_training_curve`, `save_torch_bundle`, `load_torch_bundle`; result slot
  `dl_train_result`.
- Trainer bundle schema `buildml.torch_bundle.v1` (distinct from Session
  checkpoints and classical pipeline bundles).
- M2 depth: early stopping, grad clip, LR schedulers, group/time split honor,
  resume training, structured training-curve report, walkthrough Torch status.
- Explain catalog/concept coverage for Torch ops; DL quickstart, glossary terms,
  alpha gate, and release checklist.
- CI `torch` job (Python 3.11–3.12) with unit, integration, and DL alpha smoke.

### Known limits (DL alpha)

- CPU merge gate; no GPU CI on every PR. Tabular numeric features first.
- No built-in model zoo; caller supplies `nn.Module`.
- Materialized Pandas/NumPy tensors; no Polars/DuckDB zero-copy into loaders.
- Classical preprocess is not auto-applied before loaders.
- No fold-local Torch CV, DDP, mixed precision, or ONNX/TorchScript product path.
- RAG / LLM operator remain out of scope.
- Public Torch APIs and bundle formats may change before a stable DL release.

### Verification

- Gate checklist: `docs/dl-alpha-gate.md` sign-off section.
- Tag only after remote CI is green (see `docs/release-checklist-dl-a1.md`).

## [2.0.0a1] — classical alpha — 2026-08-01

First classical-ML alpha of the BuildML 2.x `Session` API. Exit criteria and
known limits are defined in [`docs/classical-alpha-gate.md`](docs/classical-alpha-gate.md).

### Added

- Stateful `Session` workflow: ingest → roles → EDA → split → train-fitted
  preprocess → CV/search → fit → evaluate → checkpoint / pipeline persist.
- Fold-local `PreprocessRecipe` for CV/search (dates, text, outliers, impute,
  encode, binning, scale, PCA reduce, select) with Session-global-only custom
  transforms and resample documented and disclosed.
- Optional Polars/DuckDB engines for project/filter/sample/aggregate before
  Pandas materialization (sklearn still needs an in-memory design matrix).
- `Dataset.aggregate` supports `median` and integer percentiles `q0`..`q100`
  (continuous/linear interpolation; DuckDB `quantile_cont`).
- Teaching Studio / walkthrough disclosures for engine/lazy-native status,
  nested-CV `warm_start_studies`, and fold-local vs Session-global preprocess
  scope (text/PCA, custom, resample).
- Offline HTML reports, explain catalog/concepts, classical alpha-gate smoke
  test, and CI matrix (core, engines, Optuna, extras).

### Known limits (alpha)

- Custom transforms and resample stay Session-global (not in `PreprocessRecipe`).
- Native engines do not enable out-of-core sklearn fitting.
- Hashing text features are not invertible; PCA explained variance is unsupervised.
- Deep learning / RAG / LLM operator, fairness, and SHAP-style explainability
  are out of classical alpha scope.
- Public APIs and checkpoint/pipeline formats may change before stable 2.0.

### Verification

- Local gate checklist: see `docs/classical-alpha-gate.md` sign-off section.
- Tag only after remote CI is green on the push that includes this release
  candidate (see `docs/release-checklist-a1.md`).

## [1.x]

Archival release notes for the retired 1.x line live in
[`docs/history.rst`](docs/history.rst). The 1.x `SupervisedLearning` facade is
not part of the 2.x public API.
