# BuildML Proof Suite

Deep, industry-standard proofs that BuildML works end-to-end — not thin smoke.
Each project uses honest splits, train-only fitting, holdout evaluation, persisted
artifacts where applicable, and JSON metrics under `results/` (gitignored).

| Tier | Count | What it proves |
| --- | ---: | --- |
| **A** | **26/26** | One named product scenario per major Session domain |
| **B** | **6/6** | Cross-domain products composing multiple Session surfaces |
| **C** | **26/26** | Same-split industry twin + `comparison.json` per Tier A |

**Status:** proof program **complete**. Optional deepeners (not failures):
`gymnasium` via `buildml[rl]` enables the CartPole REINFORCE path (BC core
always runs); TDA prefers editable `pip install -e ".[tda]"`.

**Richer backends exercised in this env when installed:**

| Package | Proof path deepened |
| --- | --- |
| `implicit` | `movie-recs-collaborative` → ALS (`backend=implicit`) |
| `sentence-transformers` | `support-kb-rag` → dense / auto embedder |
| `flaml` (+ GBDT extras) | `churn-automl-search`, Tier B `ledger-underwriting-studio` |
| AutoGluon | preferred when FLAML absent and `autogluon.tabular` imports |

Library note: default `Session.scale` / `encode` / `impute` (and related
numeric/categorical preprocess) skip `ignore` / `id` / `target` / `group` /
`time` / `weight` roles — knapsack cost columns stay usable without
feature-scoped workarounds (Tier B `aegis` / `ledger`).

---

## How to prove BuildML works

1. Install Session 2.x from this checkout (PyPI `buildml` is still legacy 1.x):
   ```bash
   pip install -e ".[dev]"
   # domain extras as needed, e.g.:
   pip install -e ".[tda,rl,rag]"
   ```
2. Run one project or the full harness from the repo root:
   ```bash
   .\.venv\Scripts\python.exe proofs\loan-approval-classical\script.py
   .\.venv\Scripts\python.exe -m proofs._lib.run_all --tier all
   ```
3. Read `proofs/<slug>/results/*.json` — look for `"status": "completed"` (or an
   honest `skipped_missing_extra` with a documented reason). Tier C writes
   `results/comparison.json` on the **same split** as the BuildML path.

Shared helpers live in [`_lib/`](_lib/) (seed, results writer, leakage asserts,
synthetic loaders, extra probes, Tier C `write_comparison`, `run_all`).

### Optional extras (TDA and others)

Prefer an **editable** install from the repo so `pyproject.toml` extras resolve:

```bash
pip install -e ".[tda]"
# or a fuller set, e.g.
pip install -e ".[tda,automl,rag,rl]"
```

`pip install "buildml[tda]"` against a **non-editable / older wheel** can fail
because that installed package metadata may not declare the `tda` extra even
when the current `pyproject.toml` does. If you already installed ripser/persim
directly, Tier A `credit-tda-shape` can still complete; the editable extra is
the supported path for fresh envs.

**Torch:** proof JSON may include a `torch` block (`skip_torch_paths=false`
when a working CPU/CUDA wheel imports). Windows / Python 3.13 often use a
CPU wheel from the PyTorch index.

---

## Interpreting Tier C comparisons (qualitative bar 5-B)

Tier C is a **same-split industry twin**, not a bake-off for bragging rights.

- Deltas are descriptive on **one synthetic (or public-demo) draw**.
- Prefer reading **workflow parity** and **leakage discipline** over tiny metric
  gaps: BuildML must fit/select on train/validation only and evaluate on held-out
  test; the twin must use the same indices.
- Competitive bar **5-B**: BuildML should be in the same qualitative band as a
  competent sklearn / classical baseline — not required to dominate every metric.
- `"status": "filled"` means the twin ran and wrote `comparison.json`; it does
  **not** mean production certification.

---

## Docs inventory (sync to R1–R6 reality)

| Area | Canonical guide(s) | Proof deep-link |
| --- | --- | --- |
| Root overview | [`README.md`](../README.md) | → `proofs/` |
| Guide index | [`guides/README.md`](../guides/README.md) | Mapping table + per-domain links |
| Classical | `quickstart-classical.md`, `classical-end-to-end.md` | → `loan-approval-classical` |
| AutoML | `quickstart-automl.md`, `automl-deep.md` | → `churn-automl-search` · Tier B `ledger-underwriting-studio` |
| Anomaly | `quickstart-anomaly.md` | → `network-intrusion-anomaly` · Tier B `aegis-fraud-platform` |
| Forecasting / TS | `quickstart-forecasting.md`, `quickstart-timeseries-analysis.md` | → `store-sales-forecast` · Tier B `harbor-demand-desk` |
| RAG | `quickstart-rag.md` | → `support-kb-rag` · Tier B `pulse-support-copilot` |
| Unsupervised | `quickstart-unsupervised.md` | → `cluster-customer-segments` |
| Recommenders / LTR | `quickstart-recommenders.md`, `quickstart-ranking.md` | → `movie-recs-collaborative`, `search-relevance-ltr` |
| KG / TDA / Causal / Graph / Prob | matching quickstarts | → matching Tier A slugs |
| Semi / AL / SSL | matching quickstarts | → Tier A + Tier B `atlas-label-studio` |
| Online / Decisions | matching quickstarts | → Tier A + Tier B `aegis` / `harbor` / `ledger` |
| Federated / Multi-task / Meta / Symbolic / CBR / Synthetic / IL+RL | matching quickstarts | → matching Tier A (+ Tier B where listed) |
| NLP (text) | `quickstart-nlp.md`, `nlp-deep.md` | → `ticket-routing-nlp` |
| Sphinx | `docs/index.rst`, `docs/features.rst`, `docs/guide-index.rst` | Mirror Markdown proof pointers |

---

## Tier A — Single-domain deep projects

| # | Project | Domain | Status | Notes |
| ---: | --- | --- | --- | --- |
| 1 | [loan-approval-classical](loan-approval-classical/) | Classical supervised | **completed** | Tier C `comparison.json` filled |
| 2 | [churn-automl-search](churn-automl-search/) | AutoML | **completed** | Tier C RandomizedSearchCV twin |
| 3 | [network-intrusion-anomaly](network-intrusion-anomaly/) | Anomaly | **completed** | Tier C IsolationForest twin |
| 4 | [store-sales-forecast](store-sales-forecast/) | Forecast + TS analysis | **completed** | Tier C SARIMAX / seasonal_naive |
| 5 | [support-kb-rag](support-kb-rag/) | RAG | **completed** | Tier C TF-IDF cosine twin |
| 6 | [movie-recs-collaborative](movie-recs-collaborative/) | Recommenders | **completed** | Tier C item-cosine twin |
| 7 | [search-relevance-ltr](search-relevance-ltr/) | Learning-to-rank | **completed** | Tier C Ridge pointwise twin |
| 8 | [kg-biomed-linkpred](kg-biomed-linkpred/) | Knowledge graphs | **completed** | Tier C co-occurrence PMI twin |
| 9 | [credit-tda-shape](credit-tda-shape/) | TDA | **completed** | ripser/persim; Tier C logistic twin |
| 10 | [semi-label-efficiency](semi-label-efficiency/) | Semi-supervised | **completed** | Tier C LabelPropagation twin |
| 11 | [active-labeling-budget](active-labeling-budget/) | Active learning | **completed** | Tier C margin-sampling twin |
| 12 | [stream-fraud-online](stream-fraud-online/) | Online / continual | **completed** | Tier C SGD partial_fit twin |
| 13 | [multi-target-underwriting](multi-target-underwriting/) | Multi-task | **completed** | Tier C MultiOutputClassifier twin |
| 14 | [few-shot-domain-adapt](few-shot-domain-adapt/) | Meta-learning | **completed** | Tier C NearestCentroid k-shot twin |
| 15 | [policy-rules-neuro-symbolic](policy-rules-neuro-symbolic/) | Symbolic | **completed** | Tier C DecisionTree twin |
| 16 | [case-memory-claims](case-memory-claims/) | CBR | **completed** | Tier C KNeighbors twin |
| 17 | [cost-sensitive-collections](cost-sensitive-collections/) | Optimize / decisions | **completed** | Tier C val cost-threshold twin |
| 18 | [synthetic-privacy-utility](synthetic-privacy-utility/) | Synthetic | **completed** | Tier C column-bootstrap twin |
| 19 | [cluster-customer-segments](cluster-customer-segments/) | Unsupervised | **completed** | Tier C KMeans+PCA twin |
| 20 | [ssl-representation-probe](ssl-representation-probe/) | Self-supervised | **completed** | Tier C PCA+probe twin |
| 21 | [causal-treatment-effect](causal-treatment-effect/) | Causal | **completed** | Tier C sklearn AIPW twin |
| 22 | [federated-hospital-sim](federated-hospital-sim/) | Federated | **completed** | Tier C pooled SGD twin |
| 23 | [graph-fraud-rings](graph-fraud-rings/) | Graph | **completed** | Tier C networkx+LR twin |
| 24 | [prob-interval-risk](prob-interval-risk/) | Probabilistic | **completed** | Tier C BayesianRidge+quantile |
| 25 | [imitation-cartpole-control](imitation-cartpole-control/) | IL + RL | **completed** | Tier C sklearn BC twin; gym REINFORCE when `gymnasium` installed |
| 26 | [ticket-routing-nlp](ticket-routing-nlp/) | NLP (text) | **completed** | Tier C `TfidfVectorizer`+`LogisticRegression` Pipeline twin |

Each Tier A README includes: business purpose, data source, leakage controls,
BuildML API steps, metrics, limitations, and an **Industry comparison** section.

---

## Tier B — Named cross-domain products

| Product | Domains combined | Status |
| --- | --- | --- |
| [Aegis Fraud Platform](aegis-fraud-platform/) | graph + anomaly + supervised + online + decision thresholds + optional rules | **completed** |
| [Harbor Demand Desk](harbor-demand-desk/) | TS analysis + forecast + optimize allocation + probabilistic intervals | **completed** |
| [Atlas Label Studio](atlas-label-studio/) | SSL + semi-supervised + active learning budget loop | **completed** |
| [Pulse Support Copilot](pulse-support-copilot/) | RAG + ranking + CBR case memory + symbolic guardrails | **completed** |
| [Ledger Underwriting Studio](ledger-underwriting-studio/) | classical + AutoML + causal assumptions + cost-sensitive decisions + calibration | **completed** |
| [Nexus Federated Clinical](nexus-federated-clinical/) | federated sim + probabilistic uncertainty + evaluation disclosures | **completed** |

---

## Tier C — Industry baseline twins

For each completed Tier A: `baseline_industry.py` **or** a comparison section
in `script.py` on the **same split**, writing `results/comparison.json`.

| Project | Tier C status |
| --- | --- |
| loan-approval-classical | **filled** (sklearn Pipeline twin) |
| network-intrusion-anomaly | **filled** (IsolationForest + val F1 threshold) |
| store-sales-forecast | **filled** (SARIMAX / seasonal_naive) |
| churn-automl-search | **filled** (RandomizedSearchCV) |
| cluster-customer-segments | **filled** (KMeans+PCA) |
| support-kb-rag | **filled** (TF-IDF cosine) |
| movie-recs-collaborative | **filled** (item-cosine) |
| search-relevance-ltr | **filled** (Ridge pointwise) |
| causal-treatment-effect | **filled** (sklearn AIPW) |
| semi-label-efficiency | **filled** (LabelPropagation) |
| active-labeling-budget | **filled** (margin sampling) |
| stream-fraud-online | **filled** (SGD partial_fit) |
| prob-interval-risk | **filled** (BayesianRidge + val quantile) |
| graph-fraud-rings | **filled** (networkx features + LR) |
| ssl-representation-probe | **filled** (PCA + logistic probe) |
| credit-tda-shape | **filled** (logistic on raw features) |
| kg-biomed-linkpred | **filled** (train co-occurrence PMI / filtered ranking) |
| multi-target-underwriting | **filled** (MultiOutputClassifier) |
| few-shot-domain-adapt | **filled** (NearestCentroid k-shot) |
| policy-rules-neuro-symbolic | **filled** (DecisionTreeClassifier) |
| case-memory-claims | **filled** (KNeighborsClassifier) |
| cost-sensitive-collections | **filled** (val cost-threshold sweep) |
| synthetic-privacy-utility | **filled** (independent column bootstrap) |
| federated-hospital-sim | **filled** (pooled centralized SGD) |
| imitation-cartpole-control | **filled** (sklearn BC; gym disclosed when present) |
| ticket-routing-nlp | **filled** (`Pipeline(TfidfVectorizer + LogisticRegression)`) |

---

## Cleanup / gitignore policy

- Generated `proofs/**/results/` and `proofs/**/artifacts/` are gitignored.
- Runners (`script.py`, `baseline_industry.py`), READMEs, and `_lib/` stay tracked.
- `benchmarks/**/results/` gitignored; benchmark runners kept.

---

## Optional backlog (not blockers)

1. AutoGluon on Py3.13 / Windows when upstream wheels resolve (FLAML covers
   industry AutoML today).
2. LightFM / learn2learn / giotto-tda / neuralforecast where markers still skip.
3. Release cut (PyPI 2.x) — only when explicitly requested.
