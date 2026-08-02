# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
with pre-release tags for alpha (`aN`) builds.

## [Unreleased]

### Added

- **Pass L audio multimodal:** extend multimodal fusion to audio path/waveform
  columns fused with tabular and/or text and/or image. Train-only audio
  amplitude mean/std, built-in small 1D-CNN fusion branch (honest alpha — not a
  speech foundation model), Session facades
  (`make_multimodal_torch_loaders(..., audio_column=)` /
  `make_audio_multimodal_torch_loaders`), `fit_torch` / `export_torch` refuse
  silent tabular rebuilds, AI tools/executor/autonomy allowlist/planner wiring,
  and teaching-surface sync. `soundfile` is included in `buildml[torch]` (also
  via `buildml[audio]`) for path cells; waveform arrays work with Torch alone.
  CI torch job runs Pass L tests with `.[torch,onnx]`.

### Fixed

- **Pass N leftovers after Pass M:** torch trainer bundles persist
  `multimodal_preprocess` (audio/image stats, source SR, layout) with load-path
  honesty (meta restored; DataLoaders not auto-rebuilt); ambiguous 2D waveform
  arrays raise instead of silent flatten; repeat-pad kept as alpha pooling
  choice (documented + short-clip pool-signal test) rather than widening the
  forward/export contract with length-masked pooling.
- **Pass M adversarial re-audit after Pass L:** short audio clips are
  repeat-padded (not zero-filled) so default `audio_max_samples` does not wipe
  the 1D-CNN pool; train-only amp stats use pre-pad lengths; media path/array
  columns are refused as inferred text without `audio_column=`/`image_column=`;
  `fit_torch` / `evaluate_torch` / `fit_torch_ddp` refuse silent tabular loader
  rebuild after multimodal/text fit (not only `export_torch`); ONNX export uses
  multimodal `input_layout` names; AI tool schemas/executor forward
  `audio_sample_rate` / `audio_max_samples` / `audio_source_sample_rate`;
  `docs/features.rst` no longer falsely claims audio multimodal is deferred.
- **Pass K adversarial re-audit after Pass J:** ONNX export broke on Torch ≥2.9
  (`dynamo=True` default requires `onnxscript`). Export now uses
  `dynamo=False` and requires `buildml[onnx]` up front. AI registry/executor
  coverage for `make_image_multimodal_torch_loaders` is regression-tested;
  image multimodal ONNX smoke added. CI torch job installs `.[torch,onnx]`.
  Maintainer architecture note no longer claims image multimodal is deferred.

### Added

- **Pass J image multimodal:** extend multimodal fusion beyond tabular⊕text to
  include image path/array columns fused with tabular and/or text. Train-only
  image channel mean/std, built-in CNN fusion branch, Session facades
  (`make_multimodal_torch_loaders(..., image_column=)` /
  `make_image_multimodal_torch_loaders`), `fit_torch` / `export_torch` wiring
  that refuses silent tabular rebuilds, AI tools/executor/autonomy allowlist,
  and teaching-surface sync. Pillow is included in `buildml[torch]` for path
  cells.

### Fixed

- **Pass H adversarial re-audit after Pass G:** AI registry listed
  `make_multimodal_torch_loaders` / `search_torch` / `nested_cv_torch` /
  `export_torch` but executor had no dispatch handlers (dead wires). Tool schemas
  now accept `param_grid` / `param_distributions`. Multimodal fusion `forward`
  supports both tuple and dual-arg calling so TorchScript/ONNX export works.
  Randomized Torch search seeds scipy-like `rvs` with an int (not
  ``numpy.Generator``). CI torch/ai jobs run Pass G tests. README Torch extra
  row matches shipped depth. Export refuses silent tabular loader rebuild after
  text/multimodal fit.

### Added

- **Pass G deferred depth:** nested Torch HPO (`Session.nested_cv_torch` /
  `search_torch` with fold-local normalize), tabular+text multimodal fusion
  (`make_multimodal_torch_loaders` + built-in fusion module), explicit
  `ai_run_autonomous` operator automation (confirm opt-in, allowlist, max steps,
  blocked sample egress, transcript audit), CUDA AMP via
  `TrainConfig.mixed_precision` / `fit_torch(..., mixed_precision=True)`,
  single-node `fit_torch_ddp`, and TorchScript/ONNX `export_torch`. Optional
  `buildml[onnx]` extra for ONNX checker smoke tests.

### Fixed

- **Pass F adversarial re-audit:** closed soft-leakage docstring/concept regressions
  that Pass E’s line-local lint missed (`nested_cv_score` param docs still claimed
  refuse only when no fold-local recipe is provided; Concept Academy still taught
  “without a fold recipe → limited honesty”). Restored UTF-8 in
  `buildml/explain/concepts/*` after cp1252 mojibake (`Â±`, `â†'`, `â€"`, Greek).
  Copy lint now checks adjacent-line windows and rejects mojibake markers.
  Stale DL gate/checklist copy that denied built-in MLP, text loaders, and
  fold-local Torch CV on current HEAD was corrected.
- **Pass E re-audit:** corrected soft-leakage teaching regressions that still claimed
  a fold-local `PreprocessRecipe` alone bypasses Session-global CV refuse (README,
  guides, workflow guide, overlays). Added copy-lint rule
  `soft-leakage-false-claim`. Overlay tuple bugs (missing trailing commas) fixed.
- Stale maintainer honesty after Phase C: fold-local Torch CV and `rag_generate`
  are documented as shipped; architecture review / alpha-gate / phase-plan copy
  no longer contradict HEAD.

### Added

- Catalog parameter auto-fill from `operation_index.json` plus stricter
  signature↔catalog param parity; dashboard Teaching Studio concept keys are
  gated against `CONCEPT_NOTES`.
- Scoped mypy and Phase C domain tests (`test_dl_phase_c`, `test_rag_generate`,
  `test_ai_phase_c`) in CI.
- Concepts package split (`buildml/explain/concepts/{classical,dl,rag,ai}.py`)
  replacing the monolithic hand blob.
- **Phase D teaching sync:** generated Session operation index
  (`buildml/explain/generated/operation_index.json`), domain catalog overlays
  under `buildml/explain/overlays/`, and CI gate
  `python scripts/sync_teaching_surface.py --check` so Session ↔ catalog ↔ AI
  tools cannot drift silently. `make_text_torch_loaders` added to the default
  AI tool allowlist.
- **Phase C domain depth (RAG / DL / AI):** product names now match shipped
  capability rather than thin retrieve-only / tabular-only / advisor-only slices.
- **RAG generate:** `Session.rag_generate` grounded generation with citations,
  pluggable chat providers (Session AI provider, `MockProvider`, or
  `EchoGroundedProvider` for offline CI), empty-retrieval / missing-index hard
  failures, and `embedder="auto"` (semantic when `buildml[rag]` importable).
- **DL depth:** built-in tabular MLP + text embedding classifier; optional
  `fit_torch()` without a hand-rolled module; classical plan disclosure /
  `apply_plans=` bridge on `make_torch_loaders`; fold-local
  `cross_validate_torch`; `make_text_torch_loaders` sequence/text modality.
- **AI operator depth:** default tool registry covers classical + RAG retrieve /
  generate + Torch train/eval/CV; plan steps carry `parameters`; multi-step
  `ai_run_plan` can orchestrate grounded RAG and Torch tools; MockProvider
  supports queued multi-turn tool calls.
- Cost-sensitive `tune_threshold(fp_cost=..., fn_cost=...)` with recommended
  threshold, expected cost, and structured operating points.
- Stronger `error_slices`: multi-column segments, richer metrics, small-n
  handling, optional HTML export.
- Richer `dry_run` / `summarize_history` audit UX (ranked risks, prerequisite
  graph summary, suggested next ops) surfaced on walkthrough HTML.

### Changed

- Session facade typing for DL/RAG/AI public results and key method signatures
  (still a thin delegate; logic remains in domain/ops packages).
- RAG / DL / AI maintainer locks and quickstarts updated so “no generate” is no
  longer the product ceiling; hashing remains the CI-safe default embedder with
  semantic path first-class behind `buildml[rag]`.

## [2.3.0a1] — AI operator alpha — 2026-08-02

First AI operator alpha on the BuildML 2.x `Session` API. Exit criteria and
known limits are defined in [`maintainers/ai-alpha-gate.md`](maintainers/ai-alpha-gate.md).
Classical alpha remains at `2.0.0a1`; DL alpha at `2.1.0a1`; RAG alpha at
`2.2.0a1`. This line adds optional LLM-assisted workflow guidance — **not**
autonomous agents or auto-execution.

### Added

- Optional `buildml.ai` domain behind `buildml[ai]` (alias `buildml[llm]`):
  LLM-assisted workflow guidance via typed tool registry, privacy-aware egress,
  and propose-confirm-execute patterns.
- Session delegates: `ai_configure`, `ai_egress_preview`, `ai_dry_run`,
  `ai_advisor`, `ai_plan`, `ai_execute`, `ai_status`, `save_ai_transcript`,
  `load_ai_transcript`; result slots `ai_result` / `ai_transcript`.
- Provider protocol (OpenAI-compatible) with BYO API key; `MockProvider` for
  CI and offline testing.
- Tool registry with typed allowlist: read-only tools (describe, explain,
  workflow_status), write tools (set_roles, impute, split, fit), destructive
  tools (drop_columns) with confirmation policies.
- Egress privacy controls: `STATS_ONLY` default, column allow/deny lists,
  `ai_egress_preview` manifest, `ai_dry_run` payload inspection.
- Multi-step planner with batch approve and budget tracking (token/cost limits).
- Transcript schema `buildml.ai_transcript.v1` (distinct from checkpoint/bundle);
  API keys and raw data never persisted by default.
- Injection hardening: detection of malicious prompts, column names, and RAG
  chunks; tool registry is the trust boundary.
- Explain catalog/concept coverage for AI ops; AI quickstart, alpha gate, and
  release checklist.
- CI `ai` job (Python 3.11–3.12) with unit tests using `MockProvider` only.

### Known limits (AI alpha)

- **Bring-your-own API key.** BuildML never ships, proxies, or embeds keys.
- **Default egress is STATS_ONLY.** Raw rows require explicit opt-in and
  confirmation. Provider sees whatever egress payload the user approved.
- **Propose → confirm → execute.** No autonomous agent or auto-execution.
- **Tool registry is the trust boundary.** Cannot execute arbitrary code or
  tools not in the allowlist.
- **Transcript ≠ checkpoint ≠ bundle.** Three distinct artifacts.
- **Not a replacement for Teaching Studio.** Supplements, not replaces.
- **Not fine-tuning LLMs.** Use DL domain or external tools.
- **Advice must be verified.** Evidence-bound recommendations are not infallible.
- **No local-only provider path.** OpenAI-compatible protocol only; local LLM
  support is later.
- **CI runs with MockProvider only.** No real API keys in tests.
- **Public AI APIs and transcript formats may change before a stable release.**

### Verification

- Gate checklist: `maintainers/ai-alpha-gate.md` sign-off section.
- Tag only after remote CI is green (see `maintainers/release-checklist-ai-a1.md`).

## [2.2.0a1] — RAG alpha — 2026-08-01

First retrieval (RAG) alpha on the BuildML 2.x `Session` API. Exit criteria and
known limits are defined in [`maintainers/rag-alpha-gate.md`](maintainers/rag-alpha-gate.md).
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

- Gate checklist: `maintainers/rag-alpha-gate.md` sign-off section.
- Tag only after remote CI is green (see `maintainers/release-checklist-rag-a1.md`).

## [2.1.0a1] — DL alpha — 2026-08-01

First deep-learning alpha on the BuildML 2.x `Session` API. Exit criteria and
known limits are defined in [`maintainers/dl-alpha-gate.md`](maintainers/dl-alpha-gate.md).
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

- Gate checklist: `maintainers/dl-alpha-gate.md` sign-off section.
- Tag only after remote CI is green (see `maintainers/release-checklist-dl-a1.md`).

## [2.0.0a1] — classical alpha — 2026-08-01

First classical-ML alpha of the BuildML 2.x `Session` API. Exit criteria and
known limits are defined in [`maintainers/classical-alpha-gate.md`](maintainers/classical-alpha-gate.md).

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

- Local gate checklist: see `maintainers/classical-alpha-gate.md` sign-off section.
- Tag only after remote CI is green on the push that includes this release
  candidate (see `maintainers/release-checklist-a1.md`).

## [1.x]

Archival release notes for the retired 1.x line live in
[`docs/history.rst`](docs/history.rst). The 1.x `SupervisedLearning` facade is
not part of the 2.x public API.
