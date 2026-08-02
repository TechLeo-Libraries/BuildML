# BuildML

BuildML is a Python library for tabular machine-learning workflows built around
a stateful `Session`. The Session holds the dataset, column roles,
train/validation/test membership, fitted preprocessing plans, an optional
estimator, and a record of every operation you run. Preprocessing learns from
the training partition only; validation and test rows receive frozen
transformations. That train-only boundary is enforced in the API, not buried in
docstrings you might miss.

BuildML 2.4 alpha (`2.4.0a1`) is pre-release software. Public methods, report
schemas, and serialized bundle formats may change before a stable 2.x release.
The public 2.x entry point is `buildml.Session`. Classical tabular ML is the
main path; deep learning, retrieval, and LLM-assisted operations are optional
extras on the same Session.

## Install

Python 3.10–3.13.

> **Install honesty:** PyPI `buildml` is still the legacy **1.x** line
> (`1.0.9`, MIT). It does **not** install Session 2.x. Until a 2.x wheel is
> published, install from GitHub:

```bash
pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"
```

Common optional groups (after the GitHub / editable install above):

```bash
pip install "buildml[viz]"        # matplotlib, seaborn
pip install "buildml[reports]"    # Sweetviz, ydata-profiling
pip install "buildml[eda]"        # viz + reports
pip install "buildml[dashboard]"  # local interactive EDA dashboard
pip install "buildml[engines]"    # Polars and DuckDB adapters
pip install "buildml[optuna]"     # Optuna hyperparameter search
pip install "buildml[torch]"      # Torch DL path (alias: buildml[dl])
pip install "buildml[speech]"     # ASR + speech finetune-lite (adds transformers)
pip install "buildml[vision]"     # torchvision pretrained vision hooks
pip install "buildml[pretrained]" # vision + speech pretrained extras
pip install "buildml[serve]"      # managed local FastAPI model serving
pip install "buildml[onnx]"       # optional ONNX checker for export_torch
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

# In-tree NumPy GA HPO (population/selection/crossover/mutation/elitism) —
# not neuroevolution / NAS / swarm zoo. Same train-only CV contract.
evo = session.evolutionary_search(
    DecisionTreeClassifier(random_state=0),
    param_space={"max_depth": {"type": "int", "low": 2, "high": 8}},
    population_size=8,
    n_generations=4,
    cv=3,
    random_state=0,
)
print(evo.best_params, evo.best_score)
session.evaluate(partition="test")  # confirm once after selection
```

Pass a `PreprocessRecipe` when encoding, binning, feature selection, or
outlier fences should be refit inside each fold — on **unpoisoned** data
(no prior Session-global impute/encode/scale/…). Custom transforms and
resampling stay Session-global. If Session-global preprocess already ran,
CV/search **refuse even when a fold-local recipe is passed** (recipes do not
rebuild from raw rows). Opt in only with
`allow_session_global_preprocess=True` (scores remain leakage-biased), or
re-ingest / checkpoint-load unpoisoned data first.

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
[classical quickstart](guides/quickstart-classical.md) and
[workflow guide](docs/workflow-guide.rst).

## Optional extras

Core `import buildml` does not require Torch, RAG backends, or an LLM provider.
Each extra adds methods on the same Session; classical `fit` / `evaluate` stay
unchanged.

| Extra | Install | What it adds |
|-------|---------|--------------|
| Torch | `buildml[torch]` | Tabular + text + image + audio multimodal fusion, nested HPO, AMP, single-/multi-node DDP, TorchScript/ONNX export, fold-local CV, bundles (`buildml[audio]` aliases the same extra) |
| Speech | `buildml[speech]` | ASR transcription (`transcribe_speech`) + speech classify finetune-lite (`make_speech_torch_loaders` / `fit_speech_torch`); transformers Whisper-class optional |
| Vision | `buildml[vision]` | torchvision pretrained vision backbone hooks (`load_pretrained_backbone`) |
| Pretrained | `buildml[pretrained]` | Combines `vision` + `speech` for curated backbone hooks |
| Serve | `buildml[serve]` | Managed local FastAPI serving (`buildml-serve` / `Session.serve_bundle`) for pipeline + TorchScript |
| RAG | `buildml[rag]` | Ingest → chunk → embed → retrieve → **generate** → evaluate; hashing default, semantic optional |
| Graph | `buildml[graph]` | NetworkX classical node features for Graph ML (`fit_graph(method='classical')`); GCN uses `buildml[torch]` (no PyG) |
| RL | `buildml[rl]` | Optional Gymnasium REINFORCE-lite for `fit_rl(mode='gym_reinforce')`; BC + contextual bandits stay core |
| TDA | `buildml[tda]` | Persistent homology (ripser) + persistence images (persim); landscapes/silhouettes in-tree |
| AI | `buildml[ai]` | Advisor, multi-step plan/execute, optional allowlisted autonomy; classical + RAG + Torch tools; BYO API key |
| Dashboard | `buildml[dashboard]` | Interactive local EDA via `eda_app()` |

Runnable quickstarts:

- [Classical](guides/quickstart-classical.md)
- [Unsupervised](guides/quickstart-unsupervised.md) (core clustering + PCA integration; no extra)
- [Ensembles](guides/quickstart-ensemble.md) (voting / stacking / blending; no extra)
- [AutoML](guides/quickstart-automl.md) (family + recipe search beyond HPO; Optuna optional)
- [Forecasting](guides/quickstart-forecasting.md) (time_split lag/baseline forecasts; no extra)
- [Anomaly / fraud](guides/quickstart-anomaly.md) (IsolationForest/LOF/OCSVM + supervised; no extra)
- [Semi-supervised](guides/quickstart-semisupervised.md) (label propagation / spreading / self-training; no extra)
- [Self-supervised](guides/quickstart-selfsupervised.md) (masked tabular pretext → embeddings → head; no extra)
- [Active learning](guides/quickstart-active-learning.md) (train-pool query → human labels → refit; no extra)
- [Online / continual](guides/quickstart-online-learning.md) (train-chunk `partial_fit` → eval; no extra)
- [Multi-task](guides/quickstart-multi-task.md) (MultiOutput / Chain → per-task eval; no extra)
- [Meta-learning](guides/quickstart-meta-learning.md) (episodic few-shot prototypical / warm_start; no extra)
- [Federated](guides/quickstart-federated.md) (local FedAvg / FedProx simulation; no extra)
- [Bayesian / probabilistic](guides/quickstart-probabilistic.md) (BayesianRidge / GP / NB + train-only conformal; no extra)
- [Causal ML](guides/quickstart-causal.md) (assumption-declared backdoor ATE; no extra)
- [Graph ML](guides/quickstart-graph.md) (node classify: NetworkX classical + pure-Torch GCN; `buildml[graph]` / `buildml[torch]`)
- [Symbolic / neuro-symbolic](guides/quickstart-symbolic.md) (declared/tree rules + sklearn hybrid; no extra)
- [Case-based reasoning](guides/quickstart-cbr.md) (train case memory → retrieve/reuse; ≠ RAG; no extra)
- [Imitation + RL](guides/quickstart-imitation-rl.md) (BC + contextual bandit core; optional Gymnasium via `buildml[rl]`)
- [TDA](guides/quickstart-tda.md) (local Vietoris–Rips + vectorization → sklearn; `buildml[tda]`)
- [Recommenders](guides/quickstart-recommenders.md) (user/item CF + content; ranking metrics; core)
- [Search / LTR](guides/quickstart-ranking.md) (query–item feature rows + relevance; pointwise / RankSVM-lite; core)
- [Knowledge graphs](guides/quickstart-kg.md) (triples → TransE/DistMult + symbolic query; ≠ Graph ML / Neo4j / RAG; core)
- [Optimisation / decisions](guides/quickstart-optimize.md) (thresholds / cost matrices / top-K / knapsack / LP; ≠ general OR; core)
- [Synthetic data](guides/quickstart-synthetic.md) (bootstrap / Gaussian copula / SMOTE; fidelity + TSTR; ≠ DP / `resample`; core)
- [Torch](guides/quickstart-torch.md)
- [RAG](guides/quickstart-rag.md)
- [AI operator](guides/quickstart-ai.md)

Torch covers tabular MLP, text/sequence, and multimodal fusion across tabular /
text / image / audio (path or array/waveform columns; train-only normalize
stats). Audio multimodal fusion remains a small 1D-CNN branch; a separate
speech path (`buildml[speech]`) offers ASR transcription + finetune-lite
classification (integration — not training Whisper-scale FMs from scratch).
Also: fold-local CV, nested Torch HPO, optional CUDA AMP, single-node and
torchrun multi-node DDP, TorchScript/ONNX export, and local managed serving
via `buildml[serve]`. RAG defaults to lexical hashing embeddings; semantic
sentence-transformers and grounded `rag_generate` are first-class. The AI
operator defaults to propose→confirm→execute; optional `ai_run_autonomous` is
allowlisted operator automation with hard caps — not unconstrained agency.

## Alpha status

This is pre-release software. Bundle schema version strings, report layouts,
and method signatures may change. There is no out-of-core sklearn training,
first-class SHAP or fairness reporting, or unconstrained LLM agency.
See [CHANGELOG.md](CHANGELOG.md) for release notes and
[guides/glossary.md](guides/glossary.md) for terminology.

## Documentation

- [Guides](guides/README.md) — quickstarts and glossary (Markdown)
- [Concepts](docs/concepts.rst) — roles, partitions, train-fitted plans
- [Workflow guide](docs/workflow-guide.rst) — ordering, leakage, diagnostics
- [Sphinx docs](docs/index.rst) — installation, API reference, legacy boundary

## BuildML 1.x legacy boundary

BuildML 1.x (`SupervisedLearning` and the old module layout) lives under
`buildml/_legacy/` for reference only. It is not imported from the 2.x package
root. There is no compatibility shim that re-exports 1.x APIs from
`import buildml`.

## Author and license

**Leonard Onyiriuba** — [LinkedIn](https://www.linkedin.com/in/chukwubuikem-leonard-onyiriuba/) · leonard.c.onyiriuba@gmail.com

Issues: [GitHub](https://github.com/TechLeo-Libraries/BuildML/issues)

Apache License 2.0.
