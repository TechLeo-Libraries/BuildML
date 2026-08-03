# BuildML guides

Tutorials for BuildML 2.x. These Markdown pages are the source of truth; the
same content appears on Read the Docs when the hosted build is current
([buildml.readthedocs.io](https://buildml.readthedocs.io/)).

**Install honesty:** PyPI `buildml` is still legacy **1.x**. Session 2.x
requires a GitHub or editable install (see
[installation](../docs/installation.rst)):

```bash
pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"
# then extras as needed, e.g. pip install "buildml[torch,rag,ai,serve]"
```

Apache-2.0 · [TechLeo-Libraries/BuildML](https://github.com/TechLeo-Libraries/BuildML)

**Deep proofs (not smoke):** see the [proof suite](../proofs/README.md) for
industry-standard Tier A/B/C projects with holdout metrics, leakage controls,
and Tier C `comparison.json` twins. Mapping table at the bottom of this page.

---

## Suggested learning path

| Stage | Read | Outcome |
| --- | --- | --- |
| 0 | [Installation](../docs/installation.rst), [Safe install & runtime](safe-install-and-runtime.md), [concepts](../docs/concepts.rst), [workflow guide](../docs/workflow-guide.rst) | Vocabulary, staged venv install, and runtime probe before optional extras |
| 0b | New to machine learning? `session.learn()` then `session.explain("<step>")`: see [EDA / Teaching Studio](eda-teaching-studio.md#teaching-surfaces-explain-learn-workflow-walkthrough) | Plain-language concepts and operations, in reading order |
| 1 | [Classical quickstart](quickstart-classical.md) → [Classical end-to-end](classical-end-to-end.md) | Dirty data → roles → split → prep → fit → evaluate |
| 2 | [Leakage, recipes, weights, hard-refuse CV](leakage-cv-recipes.md) | Why BuildML refuses poisoned CV; good vs bad patterns |
| 3 | [Engines](engines-polars-duckdb.md), [EDA / Teaching Studio](eda-teaching-studio.md) | Prep at scale; explore before mutating |
| 4 | [Diagnostics & model search](classical-diagnostics-search.md), [Artifacts](artifacts-checkpoints-bundles.md) | Calibration, CV/HPO, checkpoint vs pipeline |
| 5 | Optional: [Torch](quickstart-torch.md) → [Torch deep](torch-deep.md), [Speech](speech-asr-finetune.md), [Pretrained](pretrained-backbones.md) | DL on the same Session |
| 6 | Optional: [Unsupervised](quickstart-unsupervised.md) → [Unsupervised deep](unsupervised-deep.md) | Clustering, PCA integration, eval, bundle |
| 7 | Optional: [Ensembles](quickstart-ensemble.md) → [Ensemble deep](ensemble-deep.md) | Voting, stacking, holdout blending, bundle |
| 8 | Optional: [AutoML](quickstart-automl.md) → [AutoML deep](automl-deep.md) | Family + recipe search beyond HPO, nested/validation, bundle |
| 9 | Optional: [Forecasting](quickstart-forecasting.md) → [Forecasting deep](forecasting-deep.md) | time_split lag/baseline forecasts, eval, bundle |
| 9b | Optional: [TS analysis](quickstart-timeseries-analysis.md) → [TS analysis deep](timeseries-analysis-deep.md) | Stationarity / seasonality / change points / decompose (analysis-only; not `fit_forecast`) |
| 10 | Optional: [Anomaly](quickstart-anomaly.md) → [Anomaly deep](anomaly-deep.md) | IsolationForest/LOF/OCSVM + supervised fraud path, bundle |
| 11 | Optional: [Semi-supervised](quickstart-semisupervised.md) → [Semi-supervised deep](semisupervised-deep.md) | Scarce labels + unlabeled train; propagation / self-training |
| 12 | Optional: [Self-supervised](quickstart-selfsupervised.md) → [Self-supervised deep](selfsupervised-deep.md) | Masked tabular pretext → embeddings → head |
| 13 | Optional: [Active learning](quickstart-active-learning.md) → [Active learning deep](active-learning-deep.md) | Train-pool query → human labels → refit → eval |
| 14 | Optional: [Online / continual](quickstart-online-learning.md) → [Online deep](online-learning-deep.md) | Train-chunk `partial_fit` → eval → bundle |
| 15 | Optional: [Multi-task](quickstart-multi-task.md) → [Multi-task deep](multi-task-deep.md) | sklearn / industry GBDT / torch multi-head → per-task eval → bundle |
| 16 | Optional: [Meta-learning](quickstart-meta-learning.md) → [Meta-learning deep](meta-learning-deep.md) | Episodic few-shot → adapt → eval → bundle |
| 17 | Optional: [Federated](quickstart-federated.md) → [Federated deep](federated-deep.md) | Local FedAvg/FedProx → eval → bundle |
| 18 | Optional: [NLP](quickstart-nlp.md) → [NLP deep](nlp-deep.md) | Profile a corpus → classify documents → attribute tokens → topics/keyphrases/summaries/entities/sentiment → bundle |
| 19 | Optional: [RAG](quickstart-rag.md) → [RAG deep](rag-deep.md) | Retrieve, grounded generate, eval, bundle |
| 20 | Optional: [AI](quickstart-ai.md) → [AI safety](ai-operator-safety.md) → [AI tools](ai-tools-operator-patterns.md) | Advisor → confirm → execute; autonomy caps |
| 21 | Optional: [Serve & deploy recipes](serve-deploy.md) | Local FastAPI, TorchServe/TRT/K8s templates |
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
| CV / search / nested | `cv_score`, `grid_search`, `randomized_search`, `optuna_search`, `evolutionary_search`, `nested_cv_score`, `PreprocessRecipe` | [Leakage](leakage-cv-recipes.md), [Diagnostics & search](classical-diagnostics-search.md) |
| Diagnostics | `calibration`, `tune_threshold`, `feature_importance`, `error_slices`, `learning_curve`, `eval_plots` | [Diagnostics & search](classical-diagnostics-search.md) |
| Engines | `with_engine`, `to_engine`, `dataset.filter_expr` / `project` / `aggregate`, DuckDB lifecycle | [Engines](engines-polars-duckdb.md) |
| EDA / teaching | `eda`, `eda_app`, `explain`, `learn`, `workflow`, `walkthrough`, `dry_run` | [EDA / Teaching Studio](eda-teaching-studio.md) |
| Artifacts | `checkpoint_*`, `save_model`, `save_pipeline`, `predict_from_pipeline`, torch/rag/unsupervised/ensemble/automl/forecast/anomaly/semisupervised/ssl/activelearning/online/multitask/metalearning/federated/symbolic/cbr/imitation/rl/ai artifacts | [Artifacts](artifacts-checkpoints-bundles.md) |
| Unsupervised | `fit_clusters`, `assign_clusters`, `evaluate_clusters`, unsupervised bundle (+ `reduce_dimensions` for PCA) | [Unsupervised quickstart](quickstart-unsupervised.md), [Unsupervised deep](unsupervised-deep.md) |
| Ensembles | `fit_voting`, `fit_stacking`, `fit_blending`, `evaluate_ensemble`, ensemble bundle | [Ensemble quickstart](quickstart-ensemble.md), [Ensemble deep](ensemble-deep.md) |
| AutoML | `run_automl`, `evaluate_automl`, automl bundle | [AutoML quickstart](quickstart-automl.md), [AutoML deep](automl-deep.md) |
| Forecasting | `fit_forecast`, `generate_forecast`, `evaluate_forecast`, forecast bundle | [Forecasting quickstart](quickstart-forecasting.md), [Forecasting deep](forecasting-deep.md) |
| Time-series analysis | `analyze_timeseries`, `ts_decompose`, `ts_diagnostics` | [TS analysis quickstart](quickstart-timeseries-analysis.md), [TS analysis deep](timeseries-analysis-deep.md) |
| Anomaly / fraud | `fit_anomaly`, `score_anomalies`, `evaluate_anomaly`, `tune_anomaly_threshold`, anomaly bundle; backends sklearn / PyOD / torch | [Anomaly quickstart](quickstart-anomaly.md), [Anomaly deep](anomaly-deep.md) |
| Semi-supervised | `fit_semisupervised`, `predict_semisupervised`, `evaluate_semisupervised`, semisupervised bundle | [Semi-supervised quickstart](quickstart-semisupervised.md), [Semi-supervised deep](semisupervised-deep.md) |
| Self-supervised | `fit_ssl_pretext`, `transform_ssl`, `finetune_ssl_head`, `evaluate_ssl`, ssl bundle | [Self-supervised quickstart](quickstart-selfsupervised.md), [Self-supervised deep](selfsupervised-deep.md) |
| Active learning | `fit_active_learner`, `suggest_query`, `label_rows`, `evaluate_active_learning`, AL bundle | [Active learning quickstart](quickstart-active-learning.md), [Active learning deep](active-learning-deep.md) |
| Online / continual | `fit_online`, `partial_fit_online`, `evaluate_online`, `predict_online`, online bundle | [Online quickstart](quickstart-online-learning.md), [Online deep](online-learning-deep.md) |
| Multi-task / multi-output | `fit_multitask`, `predict_multitask`, `evaluate_multitask`, multitask bundle | [Multi-task quickstart](quickstart-multi-task.md), [Multi-task deep](multi-task-deep.md) |
| Meta-learning | `fit_metalearning`, `adapt_to_task`, `evaluate_metalearning`, metalearning bundle | [Meta-learning quickstart](quickstart-meta-learning.md), [Meta-learning deep](meta-learning-deep.md) |
| Federated learning | `fit_federated`, `evaluate_federated`, `predict_federated`, federated bundle | [Federated quickstart](quickstart-federated.md), [Federated deep](federated-deep.md) |
| Bayesian / probabilistic | `fit_probabilistic`, `predict_probabilistic`, `predict_interval`, `evaluate_probabilistic`, probabilistic bundle | [Probabilistic quickstart](quickstart-probabilistic.md), [Probabilistic deep](probabilistic-deep.md) |
| Causal ML | `declare_causal_assumptions`, `fit_causal`, `estimate_causal`, `evaluate_causal`, `refute_causal`, causal bundle | [Causal quickstart](quickstart-causal.md), [Causal deep](causal-deep.md) |
| Graph ML | `set_graph`, `fit_graph`, `predict_graph`, `evaluate_graph`, graph bundle | [Graph quickstart](quickstart-graph.md), [Graph deep](graph-deep.md) |
| Symbolic / neuro-symbolic | `fit_symbolic`, `predict_symbolic`, `evaluate_symbolic`, `fit_neuro_symbolic`, `predict_neuro_symbolic`, `evaluate_neuro_symbolic`, symbolic bundle | [Symbolic quickstart](quickstart-symbolic.md), [Symbolic deep](symbolic-deep.md) |
| Case-based reasoning | `fit_cbr`, `retrieve_cases`, `predict_cbr`, `evaluate_cbr`, `retain_cbr`, CBR bundle | [CBR quickstart](quickstart-cbr.md), [CBR deep](cbr-deep.md) |
| Imitation + RL | `fit_imitation`, `predict_imitation_action`, `evaluate_imitation`, `fit_rl`, `act_rl`, `evaluate_rl`, IL/RL bundles | [IL+RL quickstart](quickstart-imitation-rl.md), [IL+RL deep](imitation-rl-deep.md) |
| TDA | `fit_tda`, `transform_tda`, `predict_tda`, `evaluate_tda`, TDA bundle | [TDA quickstart](quickstart-tda.md), [TDA deep](tda-deep.md) |
| Recommenders | `fit_recommender`, `recommend`, `evaluate_recommender`, recommender bundle | [Recommenders quickstart](quickstart-recommenders.md), [Recommenders deep](recommenders-deep.md) |
| Search / LTR | `fit_ranker`, `rank`, `evaluate_ranker`, ranker bundle | [LTR quickstart](quickstart-ranking.md), [LTR deep](ranking-deep.md) |
| Knowledge graphs | `fit_kg`, `score_triples`, `predict_links`, `query_kg`, `evaluate_kg`, KG bundle | [KG quickstart](quickstart-kg.md), [KG deep](kg-deep.md) |
| Optimisation / decisions | `fit_decision_policy`, `apply_decisions`, `evaluate_decisions`, decision bundle | [Decisions quickstart](quickstart-optimize.md), [Decisions deep](optimize-deep.md) |
| Fairness / SHAP | `evaluate_fairness`, `fairness_capability_matrix`; `explain_shap` (`buildml[shap]`) | [features](../docs/features.rst), [stability](../docs/stability.md) |
| Synthetic data | `fit_synthesizer`, `sample_synthetic`, `evaluate_synthetic`, `synthetic_capability_matrix`, synthetic bundle | [Synthetic quickstart](quickstart-synthetic.md), [Synthetic deep](synthetic-deep.md) |
| NLP (text column) | `nlp_capability_matrix`, `profile_text_corpus`, `detect_language`, `fit_text_classifier`, `predict_text`, `evaluate_text_classifier`, `interpret_text_prediction`, `fit_topics`, `assign_topics`, `extract_keyphrases`, `analyze_sentiment`, `extract_entities`, `summarize_text`, NLP bundle | [NLP quickstart](quickstart-nlp.md), [NLP deep](nlp-deep.md) |
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
| [Unsupervised](quickstart-unsupervised.md) | core | Clustering, PCA integration, eval, unsupervised bundle |
| [Ensembles](quickstart-ensemble.md) | core | Voting, stacking, holdout blending, ensemble bundle |
| [AutoML](quickstart-automl.md) | core (`buildml[optuna]` for Optuna method) | Family + recipe search beyond HPO, automl bundle |
| [Forecasting](quickstart-forecasting.md) | core | time_split lag/baseline forecasts, eval, forecast bundle |
| [Time-series analysis](quickstart-timeseries-analysis.md) | core; depth via `timeseries` / `timeseries-prophet` / `timeseries-ml` | `analyze_timeseries` / decompose / diagnostics (no forecast fit) |
| [Anomaly / fraud](quickstart-anomaly.md) | core + `anomaly-industry` + `torch` | sklearn/PyOD/torch AE + supervised HGB/XGB/LGBM; validation threshold tuning |
| [Semi-supervised](quickstart-semisupervised.md) | core | Label propagation / spreading / self-training; scarce labels |
| [Self-supervised](quickstart-selfsupervised.md) | core (torch optional for zoo transfer) | Masked tabular pretext → head; zoo freeze/finetune separate |
| [Active learning](quickstart-active-learning.md) | core | Train-pool uncertainty query → human labels → refit / bundle |
| [Online / continual](quickstart-online-learning.md) | core | Train-chunk `partial_fit` → holdout eval → online bundle |
| [Multi-task / multi-output](quickstart-multi-task.md) | core + `multitask-industry` / `torch` | sklearn / GBDT / torch shared-trunk → per-task + aggregate eval → multitask bundle |
| [Meta-learning](quickstart-meta-learning.md) | core | Episodic few-shot (prototypical / warm_start) → adapt → eval → bundle |
| [Federated learning](quickstart-federated.md) | core | Local FedAvg / FedProx → holdout eval → federated bundle |
| [Bayesian / probabilistic](quickstart-probabilistic.md) | core | BayesianRidge / GP / NB + train-only conformal → NLL/coverage → bundle |
| [Causal ML](quickstart-causal.md) | core | Declared CausalAssumptions → T-learner / IPW / AIPW ATE → bundle |
| [Graph ML](quickstart-graph.md) | `buildml[graph]` (+ `torch` for GCN; `graph-pyg` for PyG) | Node classify: NetworkX classical + pure-Torch GCN + PyG GCN/SAGE/GAT; ≠ KG / Neo4j |
| [Symbolic / neuro-symbolic](quickstart-symbolic.md) | core | Declared/tree/list rules → traces; sklearn hybrid → symbolic bundle |
| [Case-based reasoning](quickstart-cbr.md) | core + `cbr-industry` | Train case memory → kNN retrieve/reuse (ANN when installed) → traces → CBR bundle (≠ RAG) |
| [Imitation + RL](quickstart-imitation-rl.md) | core (+ `rl`, `rl-industry`) | BC; contextual bandit; tabular Q-learning/SARSA; REINFORCE-lite; SB3 PPO/DQN/A2C + imitation BC/GAIL |
| [TDA](quickstart-tda.md) | `buildml[tda]` | Local VR persistence → images/landscapes/silhouettes → sklearn head |
| [Recommenders](quickstart-recommenders.md) | core | User/item CF (kNN, SVD/NMF) + content; ranking metrics; ≠ RAG / EDA Findings |
| [Search / LTR](quickstart-ranking.md) | core + `ranking-industry` | Query–item feature rows + relevance; sklearn fallback + GBDT rankers; ≠ RAG / recommenders |
| [Knowledge graphs](quickstart-kg.md) | core | (h,r,t) TransE/DistMult + symbolic query; ≠ Graph ML / Neo4j / RAG |
| [Optimisation / decisions](quickstart-optimize.md) | core | Thresholds / cost matrices / top-K / knapsack / LP; ≠ general OR |
| [Synthetic data](quickstart-synthetic.md) | core native; `smote` → `imbalanced`; SDV → `synthetic-industry` | Bootstrap / copula / SMOTE + optional CTGAN/TVAE/CopulaGAN; fidelity/TSTR/SDMetrics; ≠ DP / resample |
| [NLP](quickstart-nlp.md) | core; `nlp` (embeddings, langdetect, NLTK); `nlp-industry` (spaCy NER) | Corpus profile → document classify → exact token attribution → topics/keyphrases/summaries/entities/sentiment/language → NLP bundle; ≠ RAG / generation / fine-tuning |
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
| [Unsupervised deep](unsupervised-deep.md) | Clustering methods, PCA integration, validity honesty, unsupervised bundles |
| [Ensemble deep](ensemble-deep.md) | Voting / stacking / blending, train-only meta fit, ensemble bundles |
| [AutoML deep](automl-deep.md) | Family + recipe strategy search, nested/validation selection, automl bundles |
| [Forecasting deep](forecasting-deep.md) | time_split lag/baselines, generate vs eval protocols, exog honesty, forecast bundles |
| [Time-series analysis deep](timeseries-analysis-deep.md) | Analysis-only floor: stationarity, seasonality, change points, decompose; distinct from forecasting |
| [Anomaly deep](anomaly-deep.md) | unsupervised/novelty/supervised modes, thresholds/alert rates, imbalance metrics, anomaly bundles |
| [Semi-supervised deep](semisupervised-deep.md) | Scarce labels, propagation / self-training, labeled-only eval, semisupervised bundles |
| [Self-supervised deep](selfsupervised-deep.md) | Masked tabular pretext, embeddings, head finetune, ssl bundles |
| [Active learning deep](active-learning-deep.md) | Train-pool query strategies, human labels, budget caps, AL bundles |
| [Online / continual deep](online-learning-deep.md) | sklearn `partial_fit` family, class discovery, disclosed refit fallback, online bundles |
| [Multi-task deep](multi-task-deep.md) | Backend routing, capability matrix, industry GBDT, torch mixed heads, benchmarks |
| [Meta-learning deep](meta-learning-deep.md) | Episodic few-shot, prototypical / warm_start, novel-task eval, metalearning bundles |
| [Federated deep](federated-deep.md) | Local FedAvg / FedProx, client/group partitioning, privacy limits, federated bundles |
| [Probabilistic deep](probabilistic-deep.md) | BayesianRidge / GP / NB, train-only split conformal, NLL/coverage, probabilistic bundles |
| [Causal deep](causal-deep.md) | Assumption-declared backdoor ATE, T-learner / IPW / AIPW, placebo disclose, causal bundles |
| [Graph deep](graph-deep.md) | Node classify: NetworkX classical + pure-Torch GCN + PyG GCN/SAGE/GAT, inductive/transductive, graph bundles |
| [Symbolic deep](symbolic-deep.md) | Declared/tree/list rules, traces, neuro-symbolic overlay/features/repair, symbolic bundles |
| [CBR deep](cbr-deep.md) | Train-only case memory, metrics/reuse/retain, CBR≠RAG, CBR bundles |
| [Imitation + RL deep](imitation-rl-deep.md) | BC, bandits, offline DM/IPS, tabular TD control, REINFORCE-lite, SB3 industry, capability matrix |
| [TDA deep](tda-deep.md) | Local VR (ripser), images/landscapes/silhouettes, train-only head, TDA bundles |
| [Recommenders deep](recommenders-deep.md) | Train-only CF/content, known-item protocol, Precision@K/Recall@K/nDCG@K/MAP@K, bundles |
| [LTR deep](ranking-deep.md) | Train-only tabular LTR, group_split queries, nDCG@K/MAP@K/MRR@K, sklearn/industry/torch backends, bundles |
| [KG deep](kg-deep.md) | Train-only triples, TransE/DistMult, filtered MRR/Hits@K, symbolic query, KG bundles |
| [Decisions deep](optimize-deep.md) | Cost-sensitive thresholds, cost matrices, top-K/knapsack/LP, decision bundles; ≠ OR platform |
| [Synthetic deep](synthetic-deep.md) | Train-only native + SDV backends, capability matrix, fidelity/TSTR/SDMetrics, validate_synthetic, merge provenance, privacy limits |
| [NLP deep](nlp-deep.md) | Deterministic normalization vs train-only vocabulary, contamination screening, exact token attribution and when it is refused, NPMI topics, unsupervised description limits, NLP≠RAG, NLP bundles |
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

- Legal fairness certification / automatic bias mitigation (observational
  `evaluate_fairness` and optional `explain_shap` via `buildml[shap]` are shipped)
- Causal claims from EDA / associations / feature importance (use the separate
  assumption-declared causal path: never from EDA alone)
- PyMC / Stan / NumPyro MCMC or Bayesian deep nets (sklearn BayesianRidge / GP /
  NB + train-only split conformal is the shipped probabilistic surface)
- Full Hugging Face / TorchVision zoo productization
- Managed cloud IAM, multi-cluster orchestration, or Whisper-scale FM training
- Graph fraud, online streaming fraud platforms, or causal fraud attribution
  (batch Session anomaly path with disclosed thresholds is the shipped surface)
- AGI symbolic reasoners, Prolog/Z3 engines, fuzzy-logic products, or full
  expert-system suites (tabular if-then rules + sklearn hybrid is the shipped
  symbolic / neuro-symbolic surface)
- Full econometrics / ARIMA / Torch-sequence forecasting productization
  (classical lag/baseline Session path is the shipped forecast surface)
- Neuromorphic/SNN, swarm zoo, digital twins, AV/robotics stacks, TTS, full
  COCO detection/segmentation suite
- Exhaustive parameter tables for every knob (use `session.explain(...)`,
  `session.learn(...)`, and the generated operation catalog kept in sync by CI)

When an API is alpha, guides say so and show the honest limit next to the example.

---

## Proof suite deep-links

Industry-standard Tier A/B/C projects live under
[`proofs/`](../proofs/README.md) (**57/57** Tier A, **36/36** Tier B, **57/57**
Tier C: not smoke). Re-run: `python -m proofs._lib.run_all --tier all`.

| Domain | Proof project |
| --- | --- |
| Classical | [loan-approval-classical](../proofs/loan-approval-classical/), [mortgage-default-classical](../proofs/mortgage-default-classical/), [claim-severity-regression](../proofs/claim-severity-regression/) · Tier B [ledger-underwriting-studio](../proofs/ledger-underwriting-studio/), [cornerstone-mortgage-suite](../proofs/cornerstone-mortgage-suite/) |
| Ensembles | [voting-ensemble-attrition](../proofs/voting-ensemble-attrition/), [stacking-credit-risk](../proofs/stacking-credit-risk/), [blending-payment-risk](../proofs/blending-payment-risk/) · Tier B [citadel-ensemble-desk](../proofs/citadel-ensemble-desk/), [keystone-underwrite-ml](../proofs/keystone-underwrite-ml/) |
| Torch / DL | [torch-tabular-underwrite](../proofs/torch-tabular-underwrite/), [torch-text-intent](../proofs/torch-text-intent/) · Tier B [nova-torch-bench](../proofs/nova-torch-bench/) |
| AutoML | [churn-automl-search](../proofs/churn-automl-search/) · Tier B [ledger-underwriting-studio](../proofs/ledger-underwriting-studio/), [orbit-multitask-hub](../proofs/orbit-multitask-hub/), [keystone-underwrite-ml](../proofs/keystone-underwrite-ml/) |
| Anomaly | [network-intrusion-anomaly](../proofs/network-intrusion-anomaly/), [payment-rail-anomaly](../proofs/payment-rail-anomaly/), [iot-sensor-anomaly](../proofs/iot-sensor-anomaly/) · Tier B [aegis-fraud-platform](../proofs/aegis-fraud-platform/), [sentinel-iot-watch](../proofs/sentinel-iot-watch/), [rivulet-stream-risk](../proofs/rivulet-stream-risk/), [volt-sensor-fusion](../proofs/volt-sensor-fusion/) |
| Forecast / TS | [store-sales-forecast](../proofs/store-sales-forecast/), [energy-load-forecast](../proofs/energy-load-forecast/) · Tier B [harbor-demand-desk](../proofs/harbor-demand-desk/), [ballast-energy-desk](../proofs/ballast-energy-desk/), [terrace-retail-mesh](../proofs/terrace-retail-mesh/) |
| RAG | [support-kb-rag](../proofs/support-kb-rag/), [policy-handbook-rag](../proofs/policy-handbook-rag/) · Tier B [pulse-support-copilot](../proofs/pulse-support-copilot/), [parchment-policy-copilot](../proofs/parchment-policy-copilot/), [helix-knowledge-mesh](../proofs/helix-knowledge-mesh/), [zenith-support-os](../proofs/zenith-support-os/) |
| Clustering | [cluster-customer-segments](../proofs/cluster-customer-segments/), [sku-embedding-clusters](../proofs/sku-embedding-clusters/) · Tier B [canyon-segment-studio](../proofs/canyon-segment-studio/), [forge-synth-lab](../proofs/forge-synth-lab/) |
| Recommenders | [movie-recs-collaborative](../proofs/movie-recs-collaborative/), [catalog-recs-implicit](../proofs/catalog-recs-implicit/) · Tier B [meridian-recs-commerce](../proofs/meridian-recs-commerce/), [compass-catalog-recs](../proofs/compass-catalog-recs/) |
| LTR | [search-relevance-ltr](../proofs/search-relevance-ltr/), [sponsored-ad-ltr](../proofs/sponsored-ad-ltr/) · Tier B [pulse-support-copilot](../proofs/pulse-support-copilot/), [aurora-ad-ranker](../proofs/aurora-ad-ranker/), [meridian-recs-commerce](../proofs/meridian-recs-commerce/) |
| Knowledge graphs | [kg-biomed-linkpred](../proofs/kg-biomed-linkpred/), [logistics-kg-linkpred](../proofs/logistics-kg-linkpred/) · Tier B [helix-knowledge-mesh](../proofs/helix-knowledge-mesh/), [lattice-supply-graph](../proofs/lattice-supply-graph/) |
| TDA | [credit-tda-shape](../proofs/credit-tda-shape/), [process-tda-shape](../proofs/process-tda-shape/) · Tier B [prism-shape-monitor](../proofs/prism-shape-monitor/), [kiln-process-tda](../proofs/kiln-process-tda/), [volt-sensor-fusion](../proofs/volt-sensor-fusion/) |
| Semi / AL / SSL | [semi-label-efficiency](../proofs/semi-label-efficiency/), [radiology-semi-labels](../proofs/radiology-semi-labels/), [active-labeling-budget](../proofs/active-labeling-budget/), [defect-active-budget](../proofs/defect-active-budget/), [ssl-representation-probe](../proofs/ssl-representation-probe/), [tabular-ssl-probe](../proofs/tabular-ssl-probe/) · Tier B [atlas-label-studio](../proofs/atlas-label-studio/), [beacon-label-factory](../proofs/beacon-label-factory/), [zenith-support-os](../proofs/zenith-support-os/) |
| Online / Graph / Decisions | [stream-fraud-online](../proofs/stream-fraud-online/), [clickstream-online](../proofs/clickstream-online/), [graph-fraud-rings](../proofs/graph-fraud-rings/), [peer-lending-graph](../proofs/peer-lending-graph/), [cost-sensitive-collections](../proofs/cost-sensitive-collections/), [campaign-budget-optimize](../proofs/campaign-budget-optimize/) · Tier B [aegis-fraud-platform](../proofs/aegis-fraud-platform/), [rivulet-stream-risk](../proofs/rivulet-stream-risk/), [lattice-supply-graph](../proofs/lattice-supply-graph/) |
| Probabilistic | [prob-interval-risk](../proofs/prob-interval-risk/), [weather-prob-intervals](../proofs/weather-prob-intervals/) · Tier B [harbor-demand-desk](../proofs/harbor-demand-desk/), [ballast-energy-desk](../proofs/ballast-energy-desk/), [nexus-federated-clinical](../proofs/nexus-federated-clinical/), [relay-edge-federated](../proofs/relay-edge-federated/), [nova-torch-bench](../proofs/nova-torch-bench/) |
| Causal | [causal-treatment-effect](../proofs/causal-treatment-effect/), [uplift-marketing-causal](../proofs/uplift-marketing-causal/) · Tier B [ledger-underwriting-studio](../proofs/ledger-underwriting-studio/), [apex-uplift-studio](../proofs/apex-uplift-studio/), [cornerstone-mortgage-suite](../proofs/cornerstone-mortgage-suite/), [keystone-underwrite-ml](../proofs/keystone-underwrite-ml/) |
| Multi-task | [multi-target-underwriting](../proofs/multi-target-underwriting/), [sku-multitask-retail](../proofs/sku-multitask-retail/) · Tier B [orbit-multitask-hub](../proofs/orbit-multitask-hub/), [terrace-retail-mesh](../proofs/terrace-retail-mesh/) |
| Meta-learning | [few-shot-domain-adapt](../proofs/few-shot-domain-adapt/), [coldstart-meta-adapt](../proofs/coldstart-meta-adapt/) · Tier B [quasar-meta-adapt](../proofs/quasar-meta-adapt/), [dynamo-click-lab](../proofs/dynamo-click-lab/) |
| Federated | [federated-hospital-sim](../proofs/federated-hospital-sim/), [edge-fleet-federated](../proofs/edge-fleet-federated/) · Tier B [nexus-federated-clinical](../proofs/nexus-federated-clinical/), [relay-edge-federated](../proofs/relay-edge-federated/) |
| Symbolic / CBR | [policy-rules-neuro-symbolic](../proofs/policy-rules-neuro-symbolic/), [compliance-neuro-symbolic](../proofs/compliance-neuro-symbolic/), [case-memory-claims](../proofs/case-memory-claims/), [warranty-cbr-memory](../proofs/warranty-cbr-memory/) · Tier B [pulse-support-copilot](../proofs/pulse-support-copilot/), [mosaic-warranty-desk](../proofs/mosaic-warranty-desk/), [scaffold-compliance-ai](../proofs/scaffold-compliance-ai/) |
| Synthetic | [synthetic-privacy-utility](../proofs/synthetic-privacy-utility/), [tabular-synth-utility](../proofs/tabular-synth-utility/) · Tier B [forge-synth-lab](../proofs/forge-synth-lab/) |
| Imitation + RL | [imitation-cartpole-control](../proofs/imitation-cartpole-control/), [tabular-q-frozenlake](../proofs/tabular-q-frozenlake/) · Tier B [vector-control-deck](../proofs/vector-control-deck/) |
| NLP (text) | [ticket-routing-nlp](../proofs/ticket-routing-nlp/), [torch-text-intent](../proofs/torch-text-intent/) · Tier B [folio-claims-nlp](../proofs/folio-claims-nlp/), [zenith-support-os](../proofs/zenith-support-os/) |

Full inventory and Tier C comparison status: [proofs/README.md](../proofs/README.md).
