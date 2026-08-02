# BuildML

BuildML is a Python library for machine-learning workflows built around a
stateful `Session`. The Session holds the dataset, column roles,
train/validation/test membership, fitted preprocessing plans, an optional
estimator, and a record of every operation you run. Preprocessing learns from
the training partition only; validation and test rows receive frozen
transformations. That train-only boundary is enforced in the API.

**BuildML 2.4 alpha** (`2.4.0a2`) is pre-release software. Public methods, report
schemas, and serialized bundle formats may change before a stable 2.x release.
The public 2.x entry point is `buildml.Session`.

| Path | What it is |
| --- | --- |
| Classical tabular | Main path — ingest → roles → split → preprocess → fit → evaluate |
| Torch DL | Optional multimodal / speech / vision extras on the same Session |
| RAG | Optional retrieve → generate → evaluate |
| AI operator | Optional LLM-assisted plan/execute with allowlists |
| R1–R6 industry depth | Optional backends + capability matrices per domain |

---

## Install

**Python 3.10–3.13.**

> **Install honesty:** PyPI `buildml` is still the legacy **1.x** line
> (`1.0.9`, MIT). It does **not** install Session 2.x. Until a 2.x wheel is
> published, install from GitHub or an editable checkout:

```bash
# GitHub (Session 2.x)
pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"

# Editable source checkout (recommended for development / proofs)
pip install -e ".[dev]"
```

### Optional extras (scannable)

| Extra | Install | Adds |
| --- | --- | --- |
| Viz / EDA | `buildml[viz]`, `[reports]`, `[eda]`, `[dashboard]` | matplotlib/seaborn; Sweetviz/profiling; local EDA app |
| Engines | `buildml[engines]` | Polars + DuckDB adapters |
| Search / AutoML | `buildml[optuna]`, `[automl]`, `[automl-industry]` | Optuna; native AutoML; FLAML / AutoGluon / GBDT families |
| Imbalance | `buildml[imbalanced]` | imbalanced-learn resample |
| Torch / DL | `buildml[torch]` / `[dl]` / `[audio]` | Tabular + multimodal Torch path |
| Speech / Vision | `buildml[speech]`, `[vision]`, `[pretrained]` | ASR + finetune-lite; torchvision backbones |
| Serve / ONNX | `buildml[serve]`, `[onnx]` | Local FastAPI serve; ONNX checker |
| RAG | `buildml[rag]`, `[rag-advanced]` | Dense/rerank backends; LangChain hooks |
| Graph / RL / TDA | `buildml[graph]`, `[graph-pyg]`, `[rl]`, `[rl-industry]`, `[tda]` | NetworkX / PyG; Gymnasium / SB3; ripser/persim |
| AI | `buildml[ai]` / `[llm]` | LLM operator (BYO API key) |
| Classical bundle | `buildml[all-classical]` | engines + imbalanced + eda + excel + dashboard + optuna + automl |
| Industry meta | `buildml[production]` | R1–R6 industry extras — **best-effort** (see below) |

```bash
pip install "buildml[production]"   # after GitHub / editable install above
```

### `buildml[production]` honesty

R1–R6 industry refinement is **complete** (capability matrices, backend routing,
benchmark smokes, guides). `buildml[production]` is a **best-effort** meta-extra:
it pulls domain depth plus `*-industry` adapters — **not** a guarantee that every
nested industry wheel installs on every platform.

On **Python 3.13** (especially Windows) some nested pins are skipped via
environment markers when upstream wheels are missing or broken (LightFM,
learn2learn/qpth, giotto-tda, neuralforecast, skope-rules, …). Core sklearn paths
and markers that resolve still install. Check each domain’s capability matrix
(e.g. `Session.automl_capability_matrix()`) and the [proof suite](proofs/README.md)
for what actually runs in your environment.

It does **not** include dashboard, serve, or AI operator extras.

---

## A classical workflow

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression

from buildml import Session

frame = pd.DataFrame(
    {
        "age": [21, None, 35, 40, 29, 33, 52, 47],
        "income": [40, 55, 60, 80, 50, 70, 90, 65],
        "approved": [0, 1, 0, 1, 0, 1, 1, 0],
    }
)

session = Session.ingest(frame)
session.set_roles(
    {"age": "feature", "income": "feature", "approved": "target"}
)
session.split(test_size=0.25, stratify=True, random_state=42)

# Fit on train; apply frozen transforms everywhere else.
# Default impute/encode/scale touch feature-role columns only —
# ignore / id / target / group / time / weight stay unmutated
# (pass columns=[...] to force-include).
session.impute(strategy="median")
session.handle_outliers(method="iqr", action="cap")
session.scale(method="standard")
session.fit(LogisticRegression(max_iter=500), task="classification")

result = session.evaluate(partition="test")
print(result.metrics)
```

When rows are not exchangeable, use `group_split`, `time_split`, or
`inject_split` with memberships you designed outside BuildML.

Cross-validation and hyperparameter search draw folds from the **training**
partition only. The Session test holdout is not used for ranking.

```python
from sklearn.tree import DecisionTreeClassifier
from buildml.preprocess import PreprocessRecipe

cv = session.cv_score(
    LogisticRegression(max_iter=500),
    cv=5,
    preprocess=PreprocessRecipe(impute="median", scale="standard"),
)
print(cv.mean_metrics, cv.std_metrics)

search = session.grid_search(
    DecisionTreeClassifier(random_state=0),
    param_grid={"max_depth": [2, 4, 6], "min_samples_leaf": [1, 5]},
    cv=5,
)
print(search.best_params, search.best_score)
```

Pass a `PreprocessRecipe` when encoding, binning, feature selection, or outlier
fences should be refit inside each fold — on **unpoisoned** data (no prior
Session-global impute/encode/scale). Opt in only with
`allow_session_global_preprocess=True` when you intentionally accept leakage-biased
scores, or re-ingest / checkpoint-load unpoisoned data first.

---

## Artifacts and inspection

```python
session.checkpoint_save("artifacts/checkpoint")
restored = Session.checkpoint_load("artifacts/checkpoint")

session.save_pipeline("artifacts/pipeline", evaluate_partition="test")
loaded = Session.ingest(frame).load_pipeline("artifacts/pipeline")
loaded.apply_preprocess_plans()

before = session.explain("scale", moment="before")
steps = session.workflow()
walkthrough = session.walkthrough(export_html="artifacts/workflow.html")
```

Checkpoints store data workflow state. Model and pipeline bundles store fitted
estimators and preprocess plans — they do not embed each other.
`workflow()` marks operations as done / available / blocked / skipped from API
prerequisites; it does not judge domain fit.

---

## EDA and reports

```python
session.eda(export_html="artifacts/eda.html")

# pip install "buildml[dashboard]"
handle = session.eda_app(port=8765)
# handle.url -> http://127.0.0.1:8765/

session.evaluate(
    partition="test",
    include_plots=True,
    export_html="artifacts/evaluation.html",
)
```

Reports surface screening evidence. They do not establish causality, fairness,
or deployment readiness on their own. Deeper classical topics (target encoding,
PCA, calibration, learning curves, permutation importance) live in the
[classical quickstart](guides/quickstart-classical.md) and
[workflow guide](docs/workflow-guide.rst).

---

## Domains at a glance

Core `import buildml` stays light (numpy, pandas, scikit-learn). Domain methods
attach to the same Session; classical `fit` / `evaluate` stay unchanged. Each
refined domain exposes an honest **capability matrix** reporting which backends
are installed.

| Area | Guide | Notes |
| --- | --- | --- |
| Classical | [quickstart-classical](guides/quickstart-classical.md) | Roles, splits, preprocess, fit, CV/search |
| Unsupervised / ensembles | [unsupervised](guides/quickstart-unsupervised.md), [ensemble](guides/quickstart-ensemble.md) | Core clustering + voting/stacking |
| AutoML | [automl](guides/quickstart-automl.md) | Native + Optuna; FLAML/AutoGluon via industry |
| Forecast / TS | [forecasting](guides/quickstart-forecasting.md) | `time_split` lags/baselines |
| Anomaly | [anomaly](guides/quickstart-anomaly.md) | IsolationForest / LOF / OCSVM + supervised |
| Semi / SSL / AL / Online | matching quickstarts | sklearn floor; industry/torch deepen |
| Multi-task / Meta / Federated | matching quickstarts | MultiOutput / few-shot / FedAvg sim |
| Probabilistic / Causal | matching quickstarts | Conformal; assumption-declared ATE |
| Graph / Symbolic / CBR | matching quickstarts | NetworkX/Torch/PyG; rules; case memory |
| Recommenders / LTR / KG | matching quickstarts | CF + content; GBDT rankers; TransE-style |
| Optimize / Synthetic / IL+RL | matching quickstarts | Thresholds/knapsack; SDV optional; BC + bandits |
| TDA | [tda](guides/quickstart-tda.md) | ripser/persim (`buildml[tda]`) |
| Torch | [torch](guides/quickstart-torch.md) | Tabular / text / image / audio fusion |
| RAG | [rag](guides/quickstart-rag.md) | Hashing default; sentence-transformers optional |
| AI operator | [ai](guides/quickstart-ai.md) | Propose→confirm→execute; allowlisted autonomy |

Torch covers tabular MLP, text/sequence, and multimodal fusion; speech
(`buildml[speech]`) is ASR + finetune-lite — not Whisper-scale FM training from
scratch. RAG defaults to lexical hashing; semantic embeddings and grounded
`rag_generate` are first-class when extras resolve. The AI operator defaults to
propose→confirm→execute — not unconstrained agency.

Full guide index: [`guides/README.md`](guides/README.md).

---

## Proof suite

End-to-end evidence that Session domains work with honest splits and holdout
metrics lives under [`proofs/`](proofs/README.md) — **not** smoke tests.

| Tier | Status | Meaning |
| --- | --- | --- |
| A | **25/25** | One deep project per major domain |
| B | **6/6** | Named products composing multiple Session surfaces |
| C | **25/25** | Same-split industry twin + `comparison.json` (qualitative bar 5-B) |

```bash
# Full harness from repo root
python -m proofs._lib.run_all --tier all

# Single project
python proofs/loan-approval-classical/script.py
```

Install domain extras as needed before running (editable install preferred):

```bash
pip install -e ".[tda,rl,rag,recommenders-industry,automl-industry]"
# richer backends when wheels resolve:
#   implicit → movie-recs ALS
#   sentence-transformers → support-kb-rag dense embeddings
#   flaml / autogluon.tabular → churn-automl / ledger industry AutoML
```

TDA prefers `pip install -e ".[tda]"`. Gymnasium / SB3 deepen the IL+RL path via
`buildml[rl]` / `buildml[rl-industry]` when wheels resolve. See
[`proofs/README.md`](proofs/README.md) for the inventory, Tier C interpretation,
and re-run instructions.

---

## Alpha status

This is pre-release software. Bundle schema version strings, report layouts,
and method signatures may change. There is no out-of-core sklearn training,
first-class SHAP or fairness reporting, or unconstrained LLM agency.
See [CHANGELOG.md](CHANGELOG.md) for release notes and
[guides/glossary.md](guides/glossary.md) for terminology.

---

## Documentation

- [Proof suite](proofs/README.md) — Tier A/B/C inventory, harness, Tier C interpretation
- [Guides](guides/README.md) — quickstarts (each major domain links its proof) and glossary
- [Concepts](docs/concepts.rst) — roles, partitions, train-fitted plans
- [Workflow guide](docs/workflow-guide.rst) — ordering, leakage, diagnostics
- [Sphinx docs](docs/index.rst) — installation, features, API reference, legacy boundary
- [Changelog](CHANGELOG.md) — release notes

---

## BuildML 1.x legacy boundary

BuildML 1.x (`SupervisedLearning` and the old module layout) lives under
`buildml/_legacy/` for reference only. It is not imported from the 2.x package
root. There is no compatibility shim that re-exports 1.x APIs from
`import buildml`.

---

## Author and license

**Leonard Onyiriuba** — [LinkedIn](https://www.linkedin.com/in/chukwubuikem-leonard-onyiriuba/) · leonard.c.onyiriuba@gmail.com

Issues: [GitHub](https://github.com/TechLeo-Libraries/BuildML/issues)

Apache License 2.0.
