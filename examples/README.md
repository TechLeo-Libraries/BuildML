# BuildML examples

These scripts are the paste contract. They match the guide snippets so you
can run a loop without copying out of Markdown. They are not proofs: no
JSON harness, no industry twin. Behavioral guarantees live in `tests/`.
End-to-end evidence lives in [`proofs/`](../proofs/README.md).

Start here for the classical loop. The guide snippet is a 12-row table
you can read; the file draws 120 rows so the printed metrics are not
three-row noise. Use
[loan-approval-classical](../proofs/loan-approval-classical/) when you
want the same spine on a fuller synthetic credit table, and
[breast-cancer-classical](../proofs/breast-cancer-classical/) when you
want a public dataset.

**Install:**

```bash
pip install buildml
# from a checkout: pip install -e ".[dev]"
# extras only when the matching guide says so, e.g. buildml[torch], buildml[tda]
```

```bash
python examples/classical_loan_loop.py
python examples/leakage_cv_recipe.py
python examples/forecast_lag_loop.py
```

Bundles write under `examples/.artifacts/` (next to the script), not
your current working directory. Reload calls pass `trusted=True` because
they load a file the same script just wrote. Public loaders default to
`trusted=False`.

There is no paste script for the AI operator: that extra needs a provider
and confirmation gates. Use [quickstart-ai](../guides/quickstart-ai.md).

| Script | Guide | Evidence |
| --- | --- | --- |
| `classical_loan_loop.py` | [classical-end-to-end](../guides/classical-end-to-end.md), [classical quickstart](../guides/quickstart-classical.md) | [loan-approval-classical](../proofs/loan-approval-classical/) |
| `leakage_cv_recipe.py` | [leakage-cv-recipes](../guides/leakage-cv-recipes.md) | [loan-approval-classical](../proofs/loan-approval-classical/) |
| `evolutionary_search_loop.py` | [classical-diagnostics-search](../guides/classical-diagnostics-search.md) | [loan-approval-classical](../proofs/loan-approval-classical/) |
| `unsupervised_cluster_loop.py` | [quickstart-unsupervised](../guides/quickstart-unsupervised.md) | [cluster-customer-segments](../proofs/cluster-customer-segments/) |
| `ensemble_vote_stack_loop.py` | [quickstart-ensemble](../guides/quickstart-ensemble.md) | [voting-ensemble-attrition](../proofs/voting-ensemble-attrition/) |
| `automl_search_loop.py` | [quickstart-automl](../guides/quickstart-automl.md) | [churn-automl-search](../proofs/churn-automl-search/) |
| `forecast_lag_loop.py` | [quickstart-forecasting](../guides/quickstart-forecasting.md) | [store-sales-forecast](../proofs/store-sales-forecast/) |
| `timeseries_analyze_loop.py` | [quickstart-timeseries-analysis](../guides/quickstart-timeseries-analysis.md) | [store-sales-forecast](../proofs/store-sales-forecast/) |
| `anomaly_iforest_loop.py` | [quickstart-anomaly](../guides/quickstart-anomaly.md) | [network-intrusion-anomaly](../proofs/network-intrusion-anomaly/) |
| `semisupervised_label_propagation_loop.py` | [quickstart-semisupervised](../guides/quickstart-semisupervised.md) | [semi-label-efficiency](../proofs/semi-label-efficiency/) |
| `selfsupervised_masked_tabular_loop.py` | [quickstart-selfsupervised](../guides/quickstart-selfsupervised.md) | [ssl-representation-probe](../proofs/ssl-representation-probe/) |
| `activelearning_margin_loop.py` | [quickstart-active-learning](../guides/quickstart-active-learning.md) | [active-labeling-budget](../proofs/active-labeling-budget/) |
| `online_partial_fit_loop.py` | [quickstart-online-learning](../guides/quickstart-online-learning.md) | [stream-fraud-online](../proofs/stream-fraud-online/) |
| `multitask_multioutput_loop.py` | [quickstart-multi-task](../guides/quickstart-multi-task.md) | [multi-target-underwriting](../proofs/multi-target-underwriting/) |
| `metalearning_prototypical_loop.py` | [quickstart-meta-learning](../guides/quickstart-meta-learning.md) | [few-shot-domain-adapt](../proofs/few-shot-domain-adapt/) |
| `federated_fedavg_loop.py` | [quickstart-federated](../guides/quickstart-federated.md) | [federated-hospital-sim](../proofs/federated-hospital-sim/) |
| `probabilistic_bayesian_ridge.py` | [quickstart-probabilistic](../guides/quickstart-probabilistic.md) | [prob-interval-risk](../proofs/prob-interval-risk/) |
| `causal_aipw_ate.py` | [quickstart-causal](../guides/quickstart-causal.md) | [causal-treatment-effect](../proofs/causal-treatment-effect/) |
| `graph_node_classification.py` | [quickstart-graph](../guides/quickstart-graph.md) | [graph-fraud-rings](../proofs/graph-fraud-rings/) |
| `symbolic_rules_loop.py` | [quickstart-symbolic](../guides/quickstart-symbolic.md) | [policy-rules-neuro-symbolic](../proofs/policy-rules-neuro-symbolic/) |
| `cbr_knn_loop.py` | [quickstart-cbr](../guides/quickstart-cbr.md) | [case-memory-claims](../proofs/case-memory-claims/) |
| `imitation_rl_loop.py` | [quickstart-imitation-rl](../guides/quickstart-imitation-rl.md) | [imitation-cartpole-control](../proofs/imitation-cartpole-control/) |
| `tda_loop.py` | [quickstart-tda](../guides/quickstart-tda.md) (`buildml[tda]`) | [credit-tda-shape](../proofs/credit-tda-shape/) |
| `recommender_item_knn_loop.py` | [quickstart-recommenders](../guides/quickstart-recommenders.md) | [movie-recs-collaborative](../proofs/movie-recs-collaborative/) |
| `ranking_pointwise_loop.py` | [quickstart-ranking](../guides/quickstart-ranking.md) | [search-relevance-ltr](../proofs/search-relevance-ltr/) |
| `kg_transe_loop.py` | [quickstart-kg](../guides/quickstart-kg.md) | [kg-biomed-linkpred](../proofs/kg-biomed-linkpred/) |
| `decision_threshold_loop.py` | [quickstart-optimize](../guides/quickstart-optimize.md) | [cost-sensitive-collections](../proofs/cost-sensitive-collections/) |
| `synthetic_copula_loop.py` | [quickstart-synthetic](../guides/quickstart-synthetic.md) | [synthetic-privacy-utility](../proofs/synthetic-privacy-utility/) |
| `nlp_text_classifier_loop.py` | [quickstart-nlp](../guides/quickstart-nlp.md) | [ticket-routing-nlp](../proofs/ticket-routing-nlp/) |
| `rag_hashing_loop.py` | [quickstart-rag](../guides/quickstart-rag.md) | [support-kb-rag](../proofs/support-kb-rag/) |
| `fairness_observational_loop.py` | [quickstart-fairness](../guides/quickstart-fairness.md) | [loan-fairness-observational](../proofs/loan-fairness-observational/) |
| `torch_tabular_mlp_loop.py` | [quickstart-torch](../guides/quickstart-torch.md) (`buildml[torch]`) | [torch-tabular-underwrite](../proofs/torch-tabular-underwrite/) |
