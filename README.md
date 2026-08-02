# BuildML

BuildML is a Python library for tabular machine-learning workflows built around
a stateful `Session`. The Session holds the dataset, column roles,
train/validation/test membership, fitted preprocessing plans, an optional
estimator, and a record of every operation you run. Preprocessing learns from
the training partition only; validation and test rows receive frozen
transformations. That train-only boundary is enforced in the API, not buried in
docstrings you might miss.

Version **2.3.0a1** is alpha software. Public methods, report schemas, and
serialized bundle formats may change before a stable 2.x release. Classical
tabular ML is the main path; deep learning, retrieval, and LLM-assisted
operations are optional extras on the same Session.

## Install

Python 3.10–3.13.

```bash
pip install buildml
```

Common optional groups:

```bash
pip install "buildml[viz]"        # matplotlib, seaborn
pip install "buildml[reports]"    # Sweetviz, ydata-profiling
pip install "buildml[eda]"        # viz + reports
pip install "buildml[dashboard]"  # local interactive EDA dashboard
pip install "buildml[engines]"    # Polars and DuckDB adapters
pip install "buildml[optuna]"     # Optuna hyperparameter search
pip install "buildml[torch]"      # tabular Torch path (alias: buildml[dl])
pip install "buildml[rag]"        # optional dense/rerank backends
pip install "buildml[ai]"         # LLM operator (alias: buildml[llm])
pip install "buildml[all-classical]"
```

From a source checkout:

```bash
pip install -e ".[dev]"
```

## A classical workflow

Suppose you are modeling loan approval from applicant age, income, and a binary
outcome. Missing ages and differently scaled numeric columns are common here;
you also want a held-out test set that never influenced imputation or scaling.

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

# Each step below fits on train and applies frozen values everywhere else.
session.impute(strategy="median")
session.handle_outliers(method="iqr", action="cap")
session.scale(method="standard")
session.fit(LogisticRegression(max_iter=500), task="classification")

result = session.evaluate(partition="test")
print(result.metrics)
```

Random splitting assumes rows are exchangeable. When they are not—patients
nested under clinics, or sales rows ordered in time—use `group_split`,
`time_split`, or `inject_split` with memberships you designed outside BuildML.

Cross-validation and hyperparameter search draw folds from the training
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
session.evaluate(partition="test")  # confirm once after selection
```

Pass a `PreprocessRecipe` when encoding, binning, feature selection, or
outlier fences should be refit inside each fold. Custom transforms and
resampling stay Session-global; if you already ran full-train preprocess on the
Session, CV results record that limitation.

## Artifacts and inspection

Checkpoints store data workflow state (dataset, roles, partitions, history).
Model and pipeline bundles store fitted estimators and preprocess plans—they do
not embed each other.

```python
session.checkpoint_save("artifacts/checkpoint")
restored = Session.checkpoint_load("artifacts/checkpoint")

session.save_pipeline("artifacts/pipeline", evaluate_partition="test")
loaded = Session.ingest(frame).load_pipeline("artifacts/pipeline")
loaded.apply_preprocess_plans()
```

Before changing state, you can read the operation catalog against live Session
state:

```python
before = session.explain("scale", moment="before")
steps = session.workflow()
walkthrough = session.walkthrough(export_html="artifacts/workflow.html")
```

`workflow()` marks operations as done, available, blocked, or skipped from API
prerequisites. It does not judge whether your split or model fits the domain.

## EDA and reports

Exploratory analysis can export offline HTML or open a local dashboard when the
`dashboard` extra is installed:

```python
session.eda(export_html="artifacts/eda.html")

# pip install "buildml[dashboard]"
handle = session.eda_app(port=8765)
# handle.url -> http://127.0.0.1:8765/
# handle.stop() when finished

session.evaluate(
    partition="test",
    include_plots=True,
    export_html="artifacts/evaluation.html",
)
```

Reports surface screening evidence. They do not establish causality, fairness,
or deployment readiness on their own.

Target encoding, PCA, feature selection, calibration, threshold tuning,
learning curves, and permutation importance are covered in the
[classical quickstart](docs/quickstart-alpha.md) and
[workflow guide](docs/workflow-guide.rst).

## Optional extras

Core `import buildml` does not require Torch, RAG backends, or an LLM provider.
Each extra adds methods on the same Session; classical `fit` / `evaluate` stay
unchanged.

| Extra | Install | What it adds |
|-------|---------|--------------|
| Torch | `buildml[torch]` | `make_torch_loaders`, `fit_torch`, `evaluate_torch`, bundle save/load |
| RAG | `buildml[rag]` | Corpus ingest, chunk, embed, retrieve, evaluate; default embedder is lexical hashing |
| AI | `buildml[ai]` | Advisor, plan, and confirmed execute against real Session methods; BYO API key |
| Dashboard | `buildml[dashboard]` | Interactive local EDA via `eda_app()` |

Runnable quickstarts:

- [Classical](docs/quickstart-alpha.md)
- [Torch](docs/quickstart-dl-alpha.md)
- [RAG](docs/quickstart-rag-alpha.md)
- [AI operator](docs/quickstart-ai-alpha.md)

Torch training in this alpha is a tabular numeric slice without fold-local CV.
RAG defaults to hashing embeddings, not downloaded sentence models, and has no
generation path. The AI operator proposes and executes only registered tools
after explicit confirmation.

## Alpha status

This is pre-release software. Bundle schema version strings, report layouts,
and method signatures may change. There is no out-of-core sklearn training,
first-class SHAP or fairness reporting, autonomous LLM agents, or RAG answer
generation yet. See [CHANGELOG.md](CHANGELOG.md) for release notes and
[docs/glossary.md](docs/glossary.md) for terminology.

## Documentation

- [Concepts](docs/concepts.rst) — roles, partitions, train-fitted plans
- [Workflow guide](docs/workflow-guide.rst) — ordering, leakage, diagnostics
- [Sphinx docs](docs/index.rst) — installation, API reference, legacy boundary

BuildML 1.x (`SupervisedLearning` and the old module layout) lives under
`buildml/_legacy/` for reference only. It is not imported from the 2.x package
root.

## Author and license

**Leonard Onyiriuba** — [LinkedIn](https://www.linkedin.com/in/chukwubuikem-leonard-onyiriuba/) · leonard.c.onyiriuba@gmail.com

Issues: [GitHub](https://github.com/TechLeo-Dev/BuildML/issues)

MIT License.
