# Proofs

These projects run a Session end to end: a split, train-only fit, a holdout
number, and JSON under `results/` (gitignored). They are evidence for the
guides, not a second documentation set, and not a catalog of shipped
products.

Paste the loop first. That is [`examples/`](../examples/). Then run one
proof for the job you actually have.

They are not smoke tests. They are not a claim that every industry wheel
installs on your machine.

## How to run

From the repo root, after `pip install buildml` (or `pip install -e ".[dev]"`
on a checkout):

```bash
python proofs/loan-approval-classical/script.py
python proofs/loan-approval-classical/baseline_industry.py
python examples/classical_loan_loop.py
```

The harness re-runs many proofs at once. Use it when you are checking the
suite, not when you are learning a domain:

```bash
python -m proofs._lib.run_all --smoke
python -m proofs._lib.run_all --tier all
```

`--smoke` re-runs a core CI subset and treats `skipped_missing_extra` /
`partial` as failure unless you also pass `--allow-skip`. Process exit 0
alone is not enough under `--smoke`. Read `proofs/<slug>/results/*.json`
and look for `"status": "completed"` on a run you actually did. The
README tables here are an index of scripts, not a scoreboard of your
machine.

Shared helpers live in [`_lib/`](_lib/) (seed, results writer, leakage
asserts, synthetic loaders, extra probes, Tier C `write_comparison`,
`run_all`).

Optional extras skip with JSON when the import probe fails. Prefer an
editable extra so `pyproject.toml` markers resolve:

```bash
pip install -e ".[tda,rl,rag,torch,dashboard]"
```

`find_spec` alone is not enough. Runtime import probes and
`scripts/probe_industry_extras.py` are the honesty path.

## Start with these

One paste script and one proof per job. Expansion clones and composition
scripts sit in the harness; they are not extra products.

| Job | Paste | Evidence |
| --- | --- | --- |
| A table, a target, a trusted holdout | [`examples/classical_loan_loop.py`](../examples/classical_loan_loop.py) | [loan-approval-classical](loan-approval-classical/) |
| Same spine on a public table | | [breast-cancer-classical](breast-cancer-classical/) (sklearn Wisconsin breast cancer) |
| Fold-local CV that refuses poisoned prep | [`examples/leakage_cv_recipe.py`](../examples/leakage_cv_recipe.py) | [loan-approval-classical](loan-approval-classical/) (`cv_score` + recipe) |
| Clusters | [`examples/unsupervised_cluster_loop.py`](../examples/unsupervised_cluster_loop.py) | [cluster-customer-segments](cluster-customer-segments/) |
| Public clusters + external labels | | [wine-cluster-segments](wine-cluster-segments/) |
| Voting / stacking / blending | [`examples/ensemble_vote_stack_loop.py`](../examples/ensemble_vote_stack_loop.py) | [voting-ensemble-attrition](voting-ensemble-attrition/) |
| Family + recipe search | [`examples/automl_search_loop.py`](../examples/automl_search_loop.py) | [churn-automl-search](churn-automl-search/) |
| Isolation Forest / anomaly | [`examples/anomaly_iforest_loop.py`](../examples/anomaly_iforest_loop.py) | [network-intrusion-anomaly](network-intrusion-anomaly/) |
| Chronological forecast | [`examples/forecast_lag_loop.py`](../examples/forecast_lag_loop.py) | [store-sales-forecast](store-sales-forecast/) |
| Time-series analysis (no forecast fit) | [`examples/timeseries_analyze_loop.py`](../examples/timeseries_analyze_loop.py) | [store-sales-forecast](store-sales-forecast/) (analysis then forecast) |
| RAG retrieve / generate | [`examples/rag_hashing_loop.py`](../examples/rag_hashing_loop.py) | [support-kb-rag](support-kb-rag/) |
| Recommenders | [`examples/recommender_item_knn_loop.py`](../examples/recommender_item_knn_loop.py) | [movie-recs-collaborative](movie-recs-collaborative/) |
| Learning to rank | [`examples/ranking_pointwise_loop.py`](../examples/ranking_pointwise_loop.py) | [search-relevance-ltr](search-relevance-ltr/) |
| Knowledge-graph triples | [`examples/kg_transe_loop.py`](../examples/kg_transe_loop.py) | [kg-biomed-linkpred](kg-biomed-linkpred/) |
| Persistence image + head | [`examples/tda_loop.py`](../examples/tda_loop.py) | [credit-tda-shape](credit-tda-shape/) |
| Scarce labels | [`examples/semisupervised_label_propagation_loop.py`](../examples/semisupervised_label_propagation_loop.py) | [semi-label-efficiency](semi-label-efficiency/) |
| Query a train pool | [`examples/activelearning_margin_loop.py`](../examples/activelearning_margin_loop.py) | [active-labeling-budget](active-labeling-budget/) |
| Masked tabular pretext | [`examples/selfsupervised_masked_tabular_loop.py`](../examples/selfsupervised_masked_tabular_loop.py) | [ssl-representation-probe](ssl-representation-probe/) |
| Stream `partial_fit` | [`examples/online_partial_fit_loop.py`](../examples/online_partial_fit_loop.py) | [stream-fraud-online](stream-fraud-online/) |
| Several targets | [`examples/multitask_multioutput_loop.py`](../examples/multitask_multioutput_loop.py) | [multi-target-underwriting](multi-target-underwriting/) |
| Few-shot adapt | [`examples/metalearning_prototypical_loop.py`](../examples/metalearning_prototypical_loop.py) | [few-shot-domain-adapt](few-shot-domain-adapt/) |
| Local FedAvg | [`examples/federated_fedavg_loop.py`](../examples/federated_fedavg_loop.py) | [federated-hospital-sim](federated-hospital-sim/) |
| Intervals | [`examples/probabilistic_bayesian_ridge.py`](../examples/probabilistic_bayesian_ridge.py) | [prob-interval-risk](prob-interval-risk/) |
| Declared ATE | [`examples/causal_aipw_ate.py`](../examples/causal_aipw_ate.py) | [causal-treatment-effect](causal-treatment-effect/) |
| Graph node classify | [`examples/graph_node_classification.py`](../examples/graph_node_classification.py) | [graph-fraud-rings](graph-fraud-rings/) |
| Rules + traces | [`examples/symbolic_rules_loop.py`](../examples/symbolic_rules_loop.py) | [policy-rules-neuro-symbolic](policy-rules-neuro-symbolic/) |
| Case memory | [`examples/cbr_knn_loop.py`](../examples/cbr_knn_loop.py) | [case-memory-claims](case-memory-claims/) |
| Cost-sensitive threshold | [`examples/decision_threshold_loop.py`](../examples/decision_threshold_loop.py) | [cost-sensitive-collections](cost-sensitive-collections/) |
| Synthetic rows | [`examples/synthetic_copula_loop.py`](../examples/synthetic_copula_loop.py) | [synthetic-privacy-utility](synthetic-privacy-utility/) |
| Text column classify | [`examples/nlp_text_classifier_loop.py`](../examples/nlp_text_classifier_loop.py) | [ticket-routing-nlp](ticket-routing-nlp/) |
| Imitation / bandit / tabular Q | [`examples/imitation_rl_loop.py`](../examples/imitation_rl_loop.py) | [imitation-cartpole-control](imitation-cartpole-control/) |
| Observational group rates | [`examples/fairness_observational_loop.py`](../examples/fairness_observational_loop.py) | [loan-fairness-observational](loan-fairness-observational/) |
| Adult census (public) | | [adult-fairness-observational](adult-fairness-observational/) |
| Torch tabular MLP | [`examples/torch_tabular_mlp_loop.py`](../examples/torch_tabular_mlp_loop.py) | [torch-tabular-underwrite](torch-tabular-underwrite/) |
| Diabetes progression (public) | | [diabetes-progression-regression](diabetes-progression-regression/) |
| Industry EDA surfaces | | [eda-industry-adaptability](eda-industry-adaptability/) (`buildml[dashboard]`) |

Guides for each row live in [`guides/README.md`](../guides/README.md).

Public-dataset proofs write provenance under `results.json` → `data`
(`name`, `source`, `license` / `provenance`, `n_rows`, `n_features`,
`task`, `evidence_tier=REAL_PUBLIC_DATASET`). Several scripts refuse a
perfect holdout score (`>= 1.0`) on noisy synthetics and on those public
tables.

## Same-split industry twins

When a proof ships `baseline_industry.py` (or an embedded twin), it
writes `results/comparison.json` on the **same split** as the Session
path. Deltas are descriptive on one draw. Read workflow parity and
leakage discipline, not tiny metric gaps. BuildML must fit and select on
train / validation and evaluate on held-out test; the twin must use the
same indices.

`"status": "filled"` means the twin ran and wrote the file. It does not
mean production certification.

Fairness observational reporting and Industry EDA have no sklearn metric
twin: [loan-fairness-observational](loan-fairness-observational/) and
[eda-industry-adaptability](eda-industry-adaptability/).

Re-run the twin after the matching `script.py`:

```bash
python proofs/breast-cancer-classical/script.py
python proofs/breast-cancer-classical/baseline_industry.py
python -m proofs._lib.run_all --tier C
```

## Composition scripts

Scripts under slugs such as `aegis-fraud-platform` or `harbor-demand-desk`
compose several Session surfaces on one synthetic table. They are
harness coverage. They are not fraud platforms, demand desks, or other
products BuildML ships.

Run one with `python proofs/<slug>/script.py`. The harness list is
`TIER_B` in [`_lib/run_all.py`](_lib/run_all.py).

## More domain proofs in the harness

A second synthetic table for the same Session surface (mortgage vs
consumer loan, payment-rail vs network intrusion) lives next to the
canonical proof above. CI smoke runs a subset of those plus the public
tables. Full slug lists: `TIER_A` in [`_lib/run_all.py`](_lib/run_all.py).

Time-series analysis has no fitted bundle. The floor note is
[timeseries-analysis-floor](timeseries-analysis-floor/). The user path is
the [TS analysis quickstart](../guides/quickstart-timeseries-analysis.md).

## Generated files

`proofs/**/results/` and `proofs/**/artifacts/` are gitignored. Runners
(`script.py`, `baseline_industry.py`), READMEs, and `_lib/` stay tracked.
