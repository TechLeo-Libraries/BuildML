# BuildML guides

User-facing tutorials for BuildML 2.x. Markdown under `guides/` is the
**canonical** source; Sphinx renders the same files on Read the Docs when the
hosted build is current
([buildml.readthedocs.io](https://buildml.readthedocs.io/)).

**Install honesty:** PyPI `buildml` is still legacy **1.x**. Session 2.x
requires a GitHub or editable install (see
[installation](../docs/installation.rst)):

```bash
pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"
# then extras as needed, e.g. pip install "buildml[torch,rag,ai,serve]"
```

Apache-2.0 · [TechLeo-Libraries/BuildML](https://github.com/TechLeo-Libraries/BuildML)

---

## Suggested learning path

| Stage | Read | Outcome |
| --- | --- | --- |
| 0 | [Installation](../docs/installation.rst), [concepts](../docs/concepts.rst), [workflow guide](../docs/workflow-guide.rst) | Vocabulary, install honesty, stage decisions |
| 1 | [Classical quickstart](quickstart-classical.md) → [Classical end-to-end](classical-end-to-end.md) | Dirty data → roles → split → prep → fit → evaluate |
| 2 | [Leakage, recipes, weights, hard-refuse CV](leakage-cv-recipes.md) | Why BuildML refuses poisoned CV; good vs bad patterns |
| 3 | [Engines](engines-polars-duckdb.md), [EDA / Teaching Studio](eda-teaching-studio.md) | Prep at scale; explore before mutating |
| 4 | [Diagnostics & model search](classical-diagnostics-search.md), [Artifacts](artifacts-checkpoints-bundles.md) | Calibration, CV/HPO, checkpoint vs pipeline |
| 5 | Optional: [Torch](quickstart-torch.md) → [Torch deep](torch-deep.md), [Speech](speech-asr-finetune.md), [Pretrained](pretrained-backbones.md) | DL on the same Session |
| 6 | Optional: [RAG](quickstart-rag.md) → [RAG deep](rag-deep.md) | Retrieve, grounded generate, eval, bundle |
| 7 | Optional: [AI](quickstart-ai.md) → [AI safety](ai-operator-safety.md) → [AI tools](ai-tools-operator-patterns.md) | Advisor → confirm → execute; autonomy caps |
| 8 | Optional: [Serve & deploy recipes](serve-deploy.md) | Local FastAPI, TorchServe/TRT/K8s templates |
| ∞ | [Glossary](glossary.md), [features](../docs/features.rst) | Terms and capability boundaries |

---

## Session domain → guide map

Every major `Session` surface maps to at least one deep guide. Quickstarts stay
short on-ramps; deep guides carry use cases, many examples, failure modes, and
cross-links.

| Session domain | Primary APIs | Guide(s) |
| --- | --- | --- |
| Ingest / roles / splits | `ingest`, `set_roles`, `split`, `group_split`, `time_split`, `inject_split` | [Classical E2E](classical-end-to-end.md), [Leakage](leakage-cv-recipes.md) |
| Preprocess (Session-global) | `impute`, `encode`, `scale`, `handle_outliers`, `bin`, `select_features`, `text_features`, `reduce_dimensions`, `extract_dates`, `resample`, custom transforms | [Classical E2E](classical-end-to-end.md), [Preprocess depth](preprocess-depth.md) |
| Classical fit / eval | `fit`, `predict`, `evaluate`, `compare_models` | [Classical E2E](classical-end-to-end.md), [Diagnostics & search](classical-diagnostics-search.md) |
| CV / search / nested | `cv_score`, `grid_search`, `randomized_search`, `optuna_search`, `nested_cv_score`, `PreprocessRecipe` | [Leakage](leakage-cv-recipes.md), [Diagnostics & search](classical-diagnostics-search.md) |
| Diagnostics | `calibration`, `tune_threshold`, `feature_importance`, `error_slices`, `learning_curve`, `eval_plots` | [Diagnostics & search](classical-diagnostics-search.md) |
| Engines | `with_engine`, `to_engine`, `dataset.filter_expr` / `project` / `aggregate`, DuckDB lifecycle | [Engines](engines-polars-duckdb.md) |
| EDA / teaching | `eda`, `eda_app`, `explain`, `workflow`, `walkthrough`, `dry_run` | [EDA / Teaching Studio](eda-teaching-studio.md) |
| Artifacts | `checkpoint_*`, `save_model`, `save_pipeline`, `predict_from_pipeline`, torch/rag/ai artifacts | [Artifacts](artifacts-checkpoints-bundles.md) |
| Torch tabular / text | `make_torch_loaders`, `make_text_torch_loaders`, `fit_torch`, `evaluate_torch` | [Torch quickstart](quickstart-torch.md), [Torch deep](torch-deep.md) |
| Torch multimodal | `make_multimodal_*`, image/audio loaders, concat/gated fusion, frozen `multimodal_preprocess` restore | [Torch deep](torch-deep.md) |
| Torch CV / HPO / AMP / DDP / export | `cross_validate_torch`, `search_torch`, `nested_cv_torch`, `fit_torch_ddp`, `export_torch` | [Torch deep](torch-deep.md) |
| Speech | `transcribe_speech`, `evaluate_asr` (WER/CER), `SpeechContract`, `make_speech_torch_loaders`, `fit_speech_torch`, `domain_adapt_speech_torch`, refuse FM pretrain | [Speech](speech-asr-finetune.md) |
| Pretrained backbones | `list_pretrained_backbones`, `load_pretrained_backbone`, `attach_backbone_head` | [Pretrained](pretrained-backbones.md) |
| RAG | `rag_ingest_corpus` … `rag_generate` (+ faithfulness), `rag_evaluate`, bundle | [RAG quickstart](quickstart-rag.md), [RAG deep](rag-deep.md) |
| AI operator | `ai_configure` … `ai_run_autonomous`, transcripts | [AI quickstart](quickstart-ai.md), [AI safety](ai-operator-safety.md), [AI tools](ai-tools-operator-patterns.md) |
| Serve / packs | `serve_bundle` (`/metadata`, `/predict/batch`, optional local HTTPS), `pack_torchserve`, `prepare_tensorrt_export`, `emit_k8s_ddp_job`, `emit_k8s_serve_deployment` | [Serve & deploy](serve-deploy.md) |

---

## Quickstarts (on-ramps)

| Guide | Extra | Summary |
| --- | --- | --- |
| [Classical](quickstart-classical.md) | core | Split, preprocess, fit, evaluate, CV, teaching, bundles |
| [Torch](quickstart-torch.md) | `buildml[torch]` | Tabular + text + multimodal + speech pointers |
| [RAG](quickstart-rag.md) | `buildml[rag]` | Ingest → retrieve → generate → evaluate → bundle |
| [AI operator](quickstart-ai.md) | `buildml[ai]` | Advisor, plan, confirmed execute, autonomy caps |

---

## Deep guides (encyclopedic)

| Guide | Focus |
| --- | --- |
| [Classical end-to-end](classical-end-to-end.md) | Dirty data → pipeline bundle with many use cases |
| [Leakage, recipes, weights, hard-refuse CV](leakage-cv-recipes.md) | Good/bad examples; fold-local honesty; weight role |
| [Preprocess depth](preprocess-depth.md) | Encode variants, dates, text features, custom transforms, resample |
| [Engines (Polars / DuckDB)](engines-polars-duckdb.md) | Prep then sklearn; lifecycle; honesty on out-of-core |
| [EDA / Teaching Studio](eda-teaching-studio.md) | Findings, HTML, live dashboard, explain/workflow |
| [Diagnostics & model search](classical-diagnostics-search.md) | Calibration, thresholds, compare_models, grid/random/Optuna/nested |
| [Artifacts: checkpoints vs bundles](artifacts-checkpoints-bundles.md) | What each artifact contains and does not |
| [Torch deep](torch-deep.md) | Tabular, text, multimodal (gated fusion + preprocess restore), CV/search/nested, AMP/DDP, export |
| [Speech ASR + classify](speech-asr-finetune.md) | Stub/transformers ASR, WER/CER, SpeechContract, finetune-lite, FM refuse |
| [Pretrained backbones](pretrained-backbones.md) | Expanded catalog, `attach_backbone_head`, mock vs pretrained |
| [RAG deep](rag-deep.md) | Hybrid retrieve, grounded generate + faithfulness, eval_only hygiene, upsert |
| [AI operator safety](ai-operator-safety.md) | Egress, confirm gates, autonomy residual risk |
| [AI tools & operator patterns](ai-tools-operator-patterns.md) | Allowlist, plan execution, classical/RAG/Torch tool chains |
| [Serve & deploy recipes](serve-deploy.md) | FastAPI metadata/batch/HTTPS, TorchServe compose, K8s Job + serve Deploy |

---

## Reference

- [Glossary](glossary.md)
- [Features / boundaries](../docs/features.rst)
- [Sphinx package API](../docs/package.rst)
- Runnable mirrors (optional): [`examples/`](../examples/)

## Intentional gaps

Guides cover **public Session surfaces** and common operator patterns. They do
**not** claim:

- Causal inference, fairness certification, or SHAP-first explainability
- Full Hugging Face / TorchVision zoo productization
- Managed cloud IAM, multi-cluster orchestration, or Whisper-scale FM training
- Exhaustive parameter tables for every knob (use `session.explain(...)` and
  the generated operation catalog kept in sync by CI)

When an API is alpha, guides say so and show the honest limit next to the example.
