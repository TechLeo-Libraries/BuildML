# BuildML guides

These are the tutorials. Markdown here is the source. Read the Docs
renders the same files at
[buildml.readthedocs.io](https://buildml.readthedocs.io/).

```bash
pip install buildml
# then extras as needed, e.g. pip install "buildml[torch,rag,ai,serve]"
```

Apache-2.0 · [TechLeo-Libraries/BuildML](https://github.com/TechLeo-Libraries/BuildML)

## Start here

Do this much before you pick a domain.

| | Read | What you should be able to do |
| --- | --- | --- |
| 1 | [Installation](../docs/installation.rst), then [a first Session](../docs/usage.rst) | Install, run ingest → roles → split → prepare → fit → evaluate |
| 2 | [Concepts](../docs/concepts.rst) and the [workflow guide](../docs/workflow-guide.rst) | Know why the order exists, and when random split is the wrong split |
| 3 | [Classical quickstart](quickstart-classical.md), then [leakage and recipes](leakage-cv-recipes.md) | Repeat the loop on messier data; understand why CV refuses poisoned prep |

If machine-learning vocabulary is new, start a Session and run
`session.learn()`, then `session.explain("split")`. The
[EDA / Teaching Studio](eda-teaching-studio.md#teaching-surfaces-explain-learn-workflow-walkthrough)
page is the long form of that. Domain work lives on `session.<domain>.*`.

After that, pick a domain from the map below. Quickstarts are short
on-ramps. Deep guides carry use cases, failure modes, and cross-links.
Paste the loop from [`examples/`](../examples/). Then run one proof from
the [evidence index](../proofs/README.md).

---

## Session domain → guide map

Pick a domain from the quickstart table below. Each row is a job, not an
API dump. The deep page is there when you need failure modes and
refuses. Method lists live on `session.explain("<name>")` and the
[package reference](../docs/package.rst).

---

## Quickstarts (on-ramps)

| Guide | Extra | Summary |
| --- | --- | --- |
| [Classical](quickstart-classical.md) | core | Split, preprocess, fit, evaluate, CV, teaching, bundles |
| [Unsupervised](quickstart-unsupervised.md) | core | Clustering, PCA integration, eval, unsupervised bundle |
| [Ensembles](quickstart-ensemble.md) | core | Voting, stacking, holdout blending, ensemble bundle |
| [AutoML](quickstart-automl.md) | core (`buildml[optuna]` for Optuna method) | Family + recipe search beyond HPO, automl bundle |
| [Forecasting](quickstart-forecasting.md) | core | time_split lag/baseline forecasts, eval, forecast bundle |
| [Time-series analysis](quickstart-timeseries-analysis.md) | core; depth via `timeseries` / `timeseries-prophet` / `timeseries-ml` | `session.timeseries.analyze` / decompose / diagnostics (no forecast fit) |
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
| [Case-based reasoning](quickstart-cbr.md) | core + `cbr-industry` (or `cbr-faiss`) | Train case memory → kNN retrieve/reuse (ANN when installed) → traces → CBR bundle (≠ RAG) |
| [Imitation + RL](quickstart-imitation-rl.md) | core (+ `rl`, `rl-industry`) | BC; contextual bandit; tabular Q-learning/SARSA; REINFORCE-lite; SB3 PPO/DQN/A2C + imitation BC/GAIL |
| [TDA](quickstart-tda.md) | `buildml[tda]` | Local VR persistence → images/landscapes/silhouettes → sklearn head |
| [Recommenders](quickstart-recommenders.md) | core | User/item CF (kNN, SVD/NMF) + content; ranking metrics; ≠ RAG / EDA Findings |
| [Search / LTR](quickstart-ranking.md) | core + `ranking-industry` | Query–item feature rows + relevance; sklearn fallback + GBDT rankers; ≠ RAG / recommenders |
| [Knowledge graphs](quickstart-kg.md) | core | (h,r,t) TransE/DistMult + symbolic query; ≠ Graph ML / Neo4j / RAG |
| [Optimisation / decisions](quickstart-optimize.md) | core | Thresholds / cost matrices / top-K / knapsack / LP; ≠ general OR |
| [Fairness](quickstart-fairness.md) | core | Observational DP/DI/EO + intersectional groups + stability bands; opt-in suggest helpers; ≠ legal certification |
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
| [EDA / Teaching Studio](eda-teaching-studio.md) | Findings, Static Offline HTML, Industry App (Cockpit/Gates/Academy), explain/workflow |
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
| [Fairness deep](fairness-deep.md) | Observational DP/DI/EO, intersectional groups, stability bands, classical bridge, opt-in threshold/reweighing suggestions; ≠ legal certification |
| [Synthetic deep](synthetic-deep.md) | Train-only native + SDV backends, capability matrix, fidelity/TSTR/SDMetrics, validate_synthetic, merge provenance, privacy limits |
| [NLP deep](nlp-deep.md) | Deterministic normalization vs train-only vocabulary, contamination screening, exact token attribution and when it is refused, NPMI topics, unsupervised description limits, NLP≠RAG, NLP bundles |
| [Torch deep](torch-deep.md) | Tabular, text, multimodal (gated fusion + preprocess restore), CV/search/nested, AMP/DDP, export |
| [Speech ASR + classify](speech-asr-finetune.md) | Stub/transformers ASR, WER/CER, SpeechContract, finetune-lite, FM refuse |
| [Pretrained backbones](pretrained-backbones.md) | Expanded catalog, `session.dl.attach_head`, mock vs pretrained |
| [RAG deep](rag-deep.md) | Hybrid retrieve, grounded generate + faithfulness, eval_only hygiene, upsert |
| [AI operator safety](ai-operator-safety.md) | Egress, confirm gates, autonomy residual risk |
| [AI tools & operator patterns](ai-tools-operator-patterns.md) | Allowlist, plan execution, classical/RAG/Torch tool chains |
| [Serve & deploy recipes](serve-deploy.md) | FastAPI metadata/batch/HTTPS, TorchServe compose, K8s Job + serve Deploy |

---

## Reference

- [Glossary](glossary.md)
- [Features / boundaries](../docs/features.rst)
- [Sphinx package API](../docs/package.rst)
- Paste scripts (the guide contract): [`examples/`](../examples/)

## What these pages do not cover

If a surface is not in the map above, it is not a Session product. That
includes legal fairness certification, causality from EDA, PyMC/Stan,
a full Hugging Face zoo, managed cloud IAM, and Whisper-scale pretrain.
Each domain guide states its own refuse next to the example.
`session.explain` and `session.learn` cover knob-level detail that would
drown a tutorial.

When an API is alpha, the page says so.

---

## Paste, then evidence

[`examples/`](../examples/) is the paste contract. [`proofs/`](../proofs/README.md)
is one end-to-end run per job. Composition slugs in the harness are not
extra products.

| Domain | Paste | Evidence |
| --- | --- | --- |
| Classical | [classical_loan_loop.py](../examples/classical_loan_loop.py) | [loan-approval-classical](../proofs/loan-approval-classical/), [breast-cancer-classical](../proofs/breast-cancer-classical/) |
| Leakage / CV | [leakage_cv_recipe.py](../examples/leakage_cv_recipe.py) | [loan-approval-classical](../proofs/loan-approval-classical/) |
| Unsupervised | [unsupervised_cluster_loop.py](../examples/unsupervised_cluster_loop.py) | [cluster-customer-segments](../proofs/cluster-customer-segments/), [wine-cluster-segments](../proofs/wine-cluster-segments/) |
| Ensembles | [ensemble_vote_stack_loop.py](../examples/ensemble_vote_stack_loop.py) | [voting-ensemble-attrition](../proofs/voting-ensemble-attrition/) |
| AutoML | [automl_search_loop.py](../examples/automl_search_loop.py) | [churn-automl-search](../proofs/churn-automl-search/) |
| Anomaly | [anomaly_iforest_loop.py](../examples/anomaly_iforest_loop.py) | [network-intrusion-anomaly](../proofs/network-intrusion-anomaly/) |
| Forecast | [forecast_lag_loop.py](../examples/forecast_lag_loop.py) | [store-sales-forecast](../proofs/store-sales-forecast/) |
| Time-series analysis | [timeseries_analyze_loop.py](../examples/timeseries_analyze_loop.py) | [store-sales-forecast](../proofs/store-sales-forecast/) |
| RAG | [rag_hashing_loop.py](../examples/rag_hashing_loop.py) | [support-kb-rag](../proofs/support-kb-rag/) |
| Recommenders | [recommender_item_knn_loop.py](../examples/recommender_item_knn_loop.py) | [movie-recs-collaborative](../proofs/movie-recs-collaborative/) |
| LTR | [ranking_pointwise_loop.py](../examples/ranking_pointwise_loop.py) | [search-relevance-ltr](../proofs/search-relevance-ltr/) |
| Knowledge graphs | [kg_transe_loop.py](../examples/kg_transe_loop.py) | [kg-biomed-linkpred](../proofs/kg-biomed-linkpred/) |
| TDA | [tda_loop.py](../examples/tda_loop.py) | [credit-tda-shape](../proofs/credit-tda-shape/) |
| Semi-supervised | [semisupervised_label_propagation_loop.py](../examples/semisupervised_label_propagation_loop.py) | [semi-label-efficiency](../proofs/semi-label-efficiency/) |
| Active learning | [activelearning_margin_loop.py](../examples/activelearning_margin_loop.py) | [active-labeling-budget](../proofs/active-labeling-budget/) |
| Self-supervised | [selfsupervised_masked_tabular_loop.py](../examples/selfsupervised_masked_tabular_loop.py) | [ssl-representation-probe](../proofs/ssl-representation-probe/) |
| Online | [online_partial_fit_loop.py](../examples/online_partial_fit_loop.py) | [stream-fraud-online](../proofs/stream-fraud-online/) |
| Multi-task | [multitask_multioutput_loop.py](../examples/multitask_multioutput_loop.py) | [multi-target-underwriting](../proofs/multi-target-underwriting/) |
| Meta-learning | [metalearning_prototypical_loop.py](../examples/metalearning_prototypical_loop.py) | [few-shot-domain-adapt](../proofs/few-shot-domain-adapt/) |
| Federated | [federated_fedavg_loop.py](../examples/federated_fedavg_loop.py) | [federated-hospital-sim](../proofs/federated-hospital-sim/) |
| Probabilistic | [probabilistic_bayesian_ridge.py](../examples/probabilistic_bayesian_ridge.py) | [prob-interval-risk](../proofs/prob-interval-risk/) |
| Causal | [causal_aipw_ate.py](../examples/causal_aipw_ate.py) | [causal-treatment-effect](../proofs/causal-treatment-effect/) |
| Graph | [graph_node_classification.py](../examples/graph_node_classification.py) | [graph-fraud-rings](../proofs/graph-fraud-rings/) |
| Symbolic | [symbolic_rules_loop.py](../examples/symbolic_rules_loop.py) | [policy-rules-neuro-symbolic](../proofs/policy-rules-neuro-symbolic/) |
| CBR | [cbr_knn_loop.py](../examples/cbr_knn_loop.py) | [case-memory-claims](../proofs/case-memory-claims/) |
| Decisions | [decision_threshold_loop.py](../examples/decision_threshold_loop.py) | [cost-sensitive-collections](../proofs/cost-sensitive-collections/) |
| Synthetic | [synthetic_copula_loop.py](../examples/synthetic_copula_loop.py) | [synthetic-privacy-utility](../proofs/synthetic-privacy-utility/) |
| NLP | [nlp_text_classifier_loop.py](../examples/nlp_text_classifier_loop.py) | [ticket-routing-nlp](../proofs/ticket-routing-nlp/) |
| Imitation + RL | [imitation_rl_loop.py](../examples/imitation_rl_loop.py) | [imitation-cartpole-control](../proofs/imitation-cartpole-control/) |
| Fairness | [fairness_observational_loop.py](../examples/fairness_observational_loop.py) | [loan-fairness-observational](../proofs/loan-fairness-observational/), [adult-fairness-observational](../proofs/adult-fairness-observational/) |
| Torch | [torch_tabular_mlp_loop.py](../examples/torch_tabular_mlp_loop.py) | [torch-tabular-underwrite](../proofs/torch-tabular-underwrite/) |
| Industry EDA | | [eda-industry-adaptability](../proofs/eda-industry-adaptability/) |

Harness lists and composition scripts: [proofs/README.md](../proofs/README.md).
