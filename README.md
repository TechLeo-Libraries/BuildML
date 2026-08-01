# BuildML 2.0 alpha

BuildML is a Python library for stateful classical machine-learning workflows. A
`Session` owns the dataset, semantic column roles, partition membership, fitted
preprocessing plans, the active estimator, and operation history. The 2.0 API
also explains available operations and exports local HTML reports that do not
depend on a network connection.

BuildML 2.0 is an alpha release. APIs and checkpoint formats may change before
the stable 2.0 release.

## Install

BuildML supports Python 3.10 through 3.13.

```bash
pip install buildml
```

Optional dependencies are grouped by use:

```bash
pip install "buildml[viz]"          # matplotlib and seaborn
pip install "buildml[reports]"      # Sweetviz and ydata-profiling
pip install "buildml[eda]"          # viz + reports
pip install "buildml[dashboard]"    # local EDA Teaching Studio (FastAPI + Plotly)
pip install "buildml[engines]"      # Polars and DuckDB adapters
pip install "buildml[optuna]"       # Optuna hyperparameter search
pip install "buildml[torch]"        # tabular Torch thin slice (alias: buildml[dl])
pip install "buildml[imbalanced]"   # imbalanced-learn samplers
pip install "buildml[excel]"        # Excel input support
pip install "buildml[all-classical]"
```

For a source checkout:

```bash
pip install -e ".[dev]"
```

## A leakage-aware first workflow

Split before any operation that learns values. BuildML fits its imputer,
encoder, scaler, outlier fences, bin edges, feature selectors, sampler, and
estimator on the training partition. Validation and test rows receive frozen
transformations.

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
session.impute(strategy="median")
session.handle_outliers(method="iqr", action="cap")
session.scale(method="standard")
session.fit(LogisticRegression(max_iter=500), task="classification")

result = session.evaluate(partition="test")
print(result.metrics)
```

Deeper train-fitted preparation options:

```python
# Detect / cap / drop using train-only fences
session.handle_outliers(columns=["income"], method="iqr", action="detect")

# Quantile or uniform bins with frozen edges
session.bin(columns=["age"], strategy="quantile", n_bins=4)

# Encoding beyond one-hot / ordinal
session.encode(columns=["city"], method="infrequent", min_frequency=0.05)
# Target encoding uses out-of-fold values on train; prefer fold-local
# recipes inside cv_score when the encoding itself is tuned.
session.encode(columns=["city"], method="target", n_folds=5)

# Variance / univariate / model-based selection (numeric features)
session.select_features(strategy="univariate", k=20)
print(session.last_preprocess.findings)  # evidence-linked result

# Text → numeric features (train-fitted count / TF-IDF / hashing)
session.text_features(columns=["review"], method="tfidf", max_features=128)

# PCA with explained-variance reporting (scale first when magnitudes differ)
session.reduce_dimensions(method="pca", n_components=0.95)

# Registered custom transform (fit on train only; serialize when picklable)
Session.register_transform(
    "clip_deciles",
    fit=lambda train, params: {
        "lo": float(train.iloc[:, 0].quantile(0.1)),
        "hi": float(train.iloc[:, 0].quantile(0.9)),
    },
    transform=lambda frame, art: frame.clip(art["lo"], art["hi"]),
    description="Clip the first column to train deciles",
)
session.apply_custom_transform("clip_deciles", columns=["income"])

# Preview and audit without mutating state
preview = session.dry_run(["impute", "scale", "fit"])
preview.show()  # ranked risks, prerequisite gaps, suggested next ops
summary = session.summarize_history()
summary.show()
# walkthrough.audit_summary repeats the same ranked audit cues for offline HTML
```

Random splitting assumes rows are exchangeable. Prefer first-class helpers when
that assumption fails:

```python
session.group_split(test_size=0.2, validation_size=0.2)  # requires role "group"
session.time_split(test_size=0.2, validation_size=0.2)   # requires role "time"
```

For externally governed memberships, use `session.inject_split(...)`. BuildML
checks that indices are disjoint and in bounds; it cannot infer whether the
domain boundary itself is valid or detect semantic target proxies.

## Cross-validation and search

`cv_score`, `grid_search`, and `randomized_search` score folds drawn only from
the train partition. The Session test holdout is not used for fold membership
or ranking. Pass a `PreprocessRecipe` to refit dates/text/impute/scale/
encode(infrequent|target)/binning/reduce(PCA)/select/outliers inside each fold;
custom transforms and resample stay Session-global. If Session preprocess
already ran on full train, the result records that limitation.

```python
from sklearn.tree import DecisionTreeClassifier

from buildml.preprocess import PreprocessRecipe

cv = session.cv_score(
    LogisticRegression(max_iter=500),
    cv=5,
    preprocess=PreprocessRecipe(
        impute="median",
        scale="standard",
        encode="infrequent",
        select="model",  # or variance / univariate; fold-train only
        outliers="iqr",  # optional fold-local fences (cap/detect)
    ),
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

## Pipeline bundles and model cards

`save_model` stores the estimator and feature contract. `save_pipeline` stores
fitted preprocess plans (impute, encode, scale, dates, outliers, binning,
feature selection, and resample lineage), the estimator, and a model card
(schema, metrics, history summary, lineage). A pipeline bundle is not a
checkpoint; keep data resume and model recipes as separate artifacts.
Checkpoints may also restore optional plan objects for mid-loop resume, but
never embed the estimator. Bundle metadata uses `buildml.pipeline_bundle.v2`
and `plans.joblib` uses `buildml.plans.v2`; older flat plan dicts still load.

`apply_preprocess_plans` / `Session.apply_preprocess_plans` replay fitted plans
in score-time order (dates → impute → outliers → encode → binning → scale →
feature_select). Resample plans are lineage-only and are not reapplied at score
time. Soft materialization gates warn near ~250 MiB; set
`BUILDML_MATERIALIZATION_HARD_LIMIT_BYTES` or `hard_limit_bytes` to refuse
oversized copies.

```python
session.save_pipeline("artifacts/pipeline", evaluate_partition="test")
restored = Session.ingest(frame).load_pipeline("artifacts/pipeline")
print(restored.model_card.title if restored.model_card else None)
print(restored.model_card.lineage.get("plans_present"))
restored.apply_preprocess_plans()  # rebuild feature contract from restored plans
```

## Explanations and workflow state

The operation catalog documents prerequisites, alternatives, ordering,
assumptions, leakage risks, state changes, and how to read each result.
Explanations are read-only.

```python
before = session.explain("feature_importance", moment="before")
workflow = session.workflow()
walkthrough = session.walkthrough(export_html="artifacts/workflow.html")
```

`workflow()` identifies operations that are done, available, blocked, or
skipped for the current state. `walkthrough()` joins those statuses to recorded
history and writes a self-contained HTML file when `export_html` is supplied.
Availability means API prerequisites are satisfied; it does not establish that
an operation is appropriate for the data-generating process.

## EDA and model reports

```python
# Offline Teaching Studio snapshot (same SPA surface as eda_app; needs dashboard)
eda = session.eda(
    export_html="artifacts/eda_studio.html",
    html_format="studio",  # default
)
# Layered research HTML shell (matplotlib embeds; needs viz)
research = session.eda(
    include_plots=True,
    export_html="artifacts/eda_research.html",
    html_format="research",
    export_figures="artifacts/eda-figures",
)

# Live product UI: domain boards, Teaching Studio, Concept Academy, PDF/CSV/offline HTML
# Requires: pip install "buildml[dashboard]"  (includes plotly, reportlab, kaleido)
handle = session.eda_app(port=8765)   # or session.open_eda_dashboard()
# handle.url -> http://127.0.0.1:8765/
# If the port is busy: session.eda_app(port=8766)
# handle.stop() when finished

evaluation = session.evaluate(
    partition="test",
    include_plots=True,
    export_html="artifacts/evaluation.html",
)
```

`eda_app()` starts a local FastAPI Teaching Studio with interactive Plotly
boards, long-form per-domain Teaching Studio panels (thresholds, pitfalls,
dataset-specific worked examples, practice checklists), a searchable Concept
Academy with structured notes (definition through anti-patterns), CSV evidence
exports (including Pearson/Spearman pairs and Cramér's V when present), an
**offline HTML** snapshot of the same Studio SPA, and a structured **PDF
briefing** with cover/meta, contents, findings, domain evidence, captioned
Plotly PNG stills (via kaleido), Teaching Studio excerpts, and
methods/limitations. Interactive hover/zoom stays in the live app or offline
Studio HTML. Plotly figures follow the SPA light/dark theme (ink, grid,
annotations, heatmaps, gauges, series). Missing dashboard dependencies raise
an install hint for `pip install 'buildml[dashboard]'`; an occupied port raises
a clear alternate-port message.

`session.eda(export_html=...)` defaults to that same Teaching Studio offline
snapshot (`html_format="studio"`). Use `html_format="research"` for the layered
research HTML shell (restyled typography, sticky nav, severity-aware cards, and
responsive tables; matplotlib embeds when `include_plots=True`). EDA findings
are screening evidence, not causal conclusions. Small samples, repeated review
of test results, related rows across partitions, and collection changes can make
apparently strong patterns misleading.

Focused diagnostics include `calibration`, `tune_threshold`, `error_slices`,
`learning_curve`, `feature_importance`, and `eval_plots`. Choose thresholds on
validation data (`fp_cost` / `fn_cost` for expected-cost minimization) and
assess the fixed choice once on test data. `error_slices` supports multi-column
segments and keeps small-n rows out of the primary ranking. Permutation
importance describes fitted-model reliance for a chosen score and partition;
it is not causal importance.

## Checkpoints and model artifacts

```python
session.checkpoint_save("artifacts/checkpoint")
# Native sidecar knobs (optional; defaults keep prior behavior):
# session.checkpoint_save(
#     "artifacts/checkpoint",
#     sidecar_layout="auto",          # or "single" / "partitioned"
#     sidecar_partition_rows=25_000,
#     sidecar_compression="zstd",
# )
restored = Session.checkpoint_load("artifacts/checkpoint")

session.save_model("artifacts/model")
restored.load_model("artifacts/model")
```

A checkpoint contains canonical data, roles, partition membership, operation
history, and an integrity manifest. Optional Polars/DuckDB native sidecars use
zstd and auto layout by default (partition at ≥50k rows). It does not contain
the fitted estimator. A model bundle contains the fitted estimator and feature
contract. Only load model bundles from trusted sources because estimator
serialization is pickle-compatible.

For DuckDB, prefer `with Session.ingest(..., engine="duckdb") as session:` (or
`with session.dataset:`) so owned connections close on exit. Simple portable
filters: `from buildml.data import portable_filter_expr`.

## Alpha status (2.0.0a1)

Version `2.0.0a1` is the classical-ML alpha. APIs and checkpoint/pipeline
formats may change before stable 2.0. Release readiness is defined by the
[classical alpha gate](docs/classical-alpha-gate.md); changelog notes are in
[CHANGELOG.md](CHANGELOG.md).

### What is gated

- Leakage-aware Session path: train-fitted preprocess, fold-local
  `PreprocessRecipe` inside CV/search, held-out Session test.
- End-to-end smoke: ingest → roles → EDA → split → prep → CV/fit → evaluate →
  checkpoint + pipeline roundtrip → `predict_from_pipeline`.
- Docs/catalog coverage for the learner path and editorial copy lint.
- CI on Python 3.10–3.13 (core), plus optional engines / Optuna / extras jobs.

### Known limits (do not claim as done)

- Custom transforms and resample stay Session-global (not fold-local).
- Polars/DuckDB help project/filter/sample/aggregate; sklearn still needs an
  in-memory design matrix (not out-of-core fitting).
- Hashing text features are not invertible; PCA explained variance is
  unsupervised.
- RAG / LLM operator, fairness, and SHAP-style explainability remain out of
  classical alpha scope. A tabular Torch thin slice is available behind
  `buildml[torch]` (see below); it is separate from classical `Session.fit`.

### Local smoke

```bash
pip install -e ".[dev]"
ruff check buildml tests scripts docs/conf.py
python scripts/lint_user_copy.py
pytest tests/integration/test_classical_alpha_smoke.py -q
# Full suite (what CI `test` runs):
pytest --cov=buildml --cov-report=term-missing
```

Optional engines / Optuna / Torch:

```bash
pip install -e ".[dev,engines,optuna]"
pytest tests/unit/test_engine_aggregate.py tests/unit/test_nested_cv_optuna.py -q

pip install -e ".[torch]"
pip install pytest
pytest tests/unit/test_dl_torch_slice.py tests/integration/test_dl_torch_smoke.py -q
```

Tabular Torch slice (optional): after split, `make_torch_loaders` → `fit_torch` →
`evaluate_torch` → `save_torch_bundle`. Core `import buildml` never requires Torch.
Trainer bundles (`buildml.torch_bundle.v1`) are not Session checkpoints. Design lock:
[docs/dl-m0-lock.md](docs/dl-m0-lock.md).

Tag only after remote CI is green on the release candidate push. See
[docs/release-checklist-a1.md](docs/release-checklist-a1.md).

## Current scope and limitations

- The public 2.x entry point is `buildml.Session`.
- Pandas is the canonical sklearn materialization path. Selecting Polars or
  DuckDB supports conversion and engine-aware paths; it does not make every
  Session operation lazy or out-of-core.
- Classification and regression accept sklearn-compatible estimators.
- Random, stratified, grouped, and temporal splits are built in; use
  `inject_split` for externally governed memberships.
- Learned preprocessing requires a split and is fitted on training rows.
  Outlier, binning, encoding, and feature-selection plans expose structured
  evidence through `session.last_preprocess`.
- Checkpoints validate bundle integrity and structural compatibility, not
  semantic equivalence or complete source-data provenance.
- Reports explain computed evidence and known limits. They do not validate
  business costs, causal claims, fairness, deployment constraints, or row
  independence.
- The API, report schema, and serialized formats remain subject to alpha
  changes.

See the [alpha quickstart](docs/quickstart-alpha.md),
[classical alpha gate](docs/classical-alpha-gate.md),
[workflow guide](docs/workflow-guide.rst), [concept guide](docs/concepts.rst),
and [glossary](docs/glossary.md).

## BuildML 1.x legacy boundary

The 1.x `SupervisedLearning` facade and its old module layout are not part of
the 2.x public API. Reference source is isolated under `buildml/_legacy/` and is
excluded from the package build. There is no compatibility shim. Historical
release notes and architecture records remain in the repository as archival
material and must not be used as 2.x usage guidance.

## Author

**Leonard Onyiriuba**

- **Email:** leonard.c.onyiriuba@gmail.com
- **LinkedIn:** [Leonard Onyiriuba](https://www.linkedin.com/in/chukwubuikem-leonard-onyiriuba/)

## Contact

- **Author:** Leonard Onyiriuba
- **Email:** leonard.c.onyiriuba@gmail.com
- **GitHub Issues:** [BuildML Issues](https://github.com/TechLeo-Dev/BuildML/issues)

## Contributors

- [Onyiriuba Leonard](https://www.linkedin.com/in/chukwubuikem-leonard-onyiriuba/)
  — project lead and maintainer; workwithtechleo@gmail.com
- [The TechLeo Community](https://www.linkedin.com/company/techleo/)
  — testers; techleo.ng@gmail.com

## License

BuildML is distributed under the MIT License.
