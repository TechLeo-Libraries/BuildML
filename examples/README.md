# BuildML examples

Small scripts that mirror guide snippets for copy-paste outside Markdown.
These are **not** a CI gate; prefer `tests/` for behavioral guarantees and
`proofs/` for end-to-end product evidence.

**Install (GitHub 2.x):**

```bash
pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"
# or from a checkout: pip install -e ".[dev]"
```

Bundle reload calls pass `trusted=True` because they load artifacts the same
script just wrote. Public loaders default to `trusted=False`.

| Script | Guide |
| --- | --- |
| `classical_loan_loop.py` | [classical-end-to-end](../guides/classical-end-to-end.md) |
| `leakage_cv_recipe.py` | [leakage-cv-recipes](../guides/leakage-cv-recipes.md) |
| `evolutionary_search_loop.py` | [classical-diagnostics-search](../guides/classical-diagnostics-search.md) |
| `unsupervised_cluster_loop.py` | [quickstart-unsupervised](../guides/quickstart-unsupervised.md) |
| `ensemble_vote_stack_loop.py` | [quickstart-ensemble](../guides/quickstart-ensemble.md) |
| `automl_search_loop.py` | [quickstart-automl](../guides/quickstart-automl.md) |
| `forecast_lag_loop.py` | [quickstart-forecasting](../guides/quickstart-forecasting.md) |
| `anomaly_iforest_loop.py` | [quickstart-anomaly](../guides/quickstart-anomaly.md) |
| `semisupervised_label_propagation_loop.py` | [quickstart-semisupervised](../guides/quickstart-semisupervised.md) |
| `selfsupervised_masked_tabular_loop.py` | [quickstart-selfsupervised](../guides/quickstart-selfsupervised.md) |
| `activelearning_margin_loop.py` | [quickstart-active-learning](../guides/quickstart-active-learning.md) |
| `online_partial_fit_loop.py` | [quickstart-online-learning](../guides/quickstart-online-learning.md) |
| `multitask_multioutput_loop.py` | [quickstart-multi-task](../guides/quickstart-multi-task.md) |
| `metalearning_prototypical_loop.py` | [quickstart-meta-learning](../guides/quickstart-meta-learning.md) |
| `federated_fedavg_loop.py` | [quickstart-federated](../guides/quickstart-federated.md) |
| `probabilistic_bayesian_ridge.py` | [quickstart-probabilistic](../guides/quickstart-probabilistic.md) |
| `causal_aipw_ate.py` | [quickstart-causal](../guides/quickstart-causal.md) |
| `graph_node_classification.py` | [quickstart-graph](../guides/quickstart-graph.md) |
| `symbolic_rules_loop.py` | [quickstart-symbolic](../guides/quickstart-symbolic.md) |
| `cbr_knn_loop.py` | [quickstart-cbr](../guides/quickstart-cbr.md) |
| `imitation_rl_loop.py` | [quickstart-imitation-rl](../guides/quickstart-imitation-rl.md) |
| `tda_loop.py` | [quickstart-tda](../guides/quickstart-tda.md) (`buildml[tda]`) |
| `recommender_item_knn_loop.py` | [quickstart-recommenders](../guides/quickstart-recommenders.md) |
| `ranking_pointwise_loop.py` | [quickstart-ranking](../guides/quickstart-ranking.md) |
| `kg_transe_loop.py` | [quickstart-kg](../guides/quickstart-kg.md) |
| `decision_threshold_loop.py` | [quickstart-optimize](../guides/quickstart-optimize.md) |
| `synthetic_copula_loop.py` | [quickstart-synthetic](../guides/quickstart-synthetic.md) |
| `nlp_text_classifier_loop.py` | [quickstart-nlp](../guides/quickstart-nlp.md) |
| `rag_hashing_loop.py` | [rag-deep](../guides/rag-deep.md) |

Run from the repo root after install:

```bash
python examples/classical_loan_loop.py
python examples/forecast_lag_loop.py
python examples/nlp_text_classifier_loop.py
# …or any script listed above
```
