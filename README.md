# BuildML

BuildML is a Python library for machine-learning workflows built around a
stateful `Session`. The Session holds the dataset, column roles,
train/validation/test membership, fitted preprocessing plans, an optional
estimator, and a record of every operation you run. Preprocessing learns from
the training partition only; validation and test rows receive frozen
transformations. That train-only boundary is enforced in the API.

**BuildML 2.5** (`2.5.0`) is the current stable Session 2.x line (GitHub Release
[`v2.5.0`](https://github.com/TechLeo-Libraries/BuildML/releases/tag/v2.5.0)).
PyPI may still resolve `2.4.0` until Trusted Publishing / twine upload for
`2.5.0` completes — see [`docs/pypi-2x-publish.md`](docs/pypi-2x-publish.md).
The public entry point is `buildml.Session`. For domains, **namespaced facades**
(`session.<domain>.*`) are the supported public API; flat domain aliases are
supported-but-deprecated until BuildML 3.0. See
[`docs/stability.md`](docs/stability.md).

| Path | What it is |
| --- | --- |
| Classical tabular | Main path: ingest → roles → split → preprocess → fit → evaluate |
| Torch DL | Optional multimodal / speech / vision extras on the same Session |
| RAG | Optional retrieve → generate → evaluate |
| AI operator | Optional LLM-assisted plan/execute with allowlists |
| Industry backends | Optional extras + capability matrices per domain |

---

## Install

**Python 3.10–3.13.**

```bash
# Session 2.x from PyPI (default)
pip install buildml

# Editable source checkout (recommended for development / proofs)
pip install -e ".[dev]"

# GitHub tip of main
pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"
```

Legacy **1.x** (`1.0.9`, MIT) remains on PyPI for pin-only installs:
`pip install "buildml==1.0.9"`.

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
| Industry meta | `buildml[production]` | R1–R6 industry extras: **best-effort** (see below) |

```bash
pip install "buildml[production]"
```

### `buildml[production]` honesty

Industry domains ship with capability matrices, backend routing, benchmark
smokes, and guides. `buildml[production]` is a **best-effort** meta-extra: it
pulls domain depth plus `*-industry` adapters. It is **not** a guarantee that
every nested industry wheel installs on every platform.

On **Python 3.13** (especially Windows) some nested pins are skipped via
environment markers when upstream wheels are missing or broken (LightFM,
learn2learn/qpth, giotto-tda, neuralforecast, skope-rules, …). Core sklearn paths
and markers that resolve still install. Check each domain’s capability matrix
(e.g. `session.automl.capability_matrix()`) and the [proof suite](proofs/README.md)
for what actually runs in your environment. For a machine-local inventory of
importable industry modules (never a hard fail), run:

```bash
python scripts/probe_industry_extras.py
```

It does **not** include dashboard, serve, or AI operator extras.

### Security notes (bundles + AI)

- **Pickle / joblib / torch bundles (opt-in).** Checkpoint `plans.joblib`,
  pipeline bundles, domain `*_plan.joblib`, and Torch trainer / TorchScript
  payloads can execute code on load. Public loaders default to `trusted=False`
  and raise `ValidationError` until you pass `trusted=True` for artifacts you
  created or fully trust: for example `Session.checkpoint_load(path, trusted=True)`,
  `session.anomaly.load_bundle(path, trusted=True)`,
  `Session().predict_from_pipeline(path, frame, trusted=True)`, or
  `buildml-serve --bundle … --trusted`. Prefer JSON sidecars / parquet /
  `Session.checkpoint_load(..., data_only=True)` (skips plans without needing
  `trusted`) or re-fitting when provenance is unclear. Optional `sha256`
  integrity in manifests detects *tampering after save*; it does **not** make a
  malicious author safe. **Residual risk:** `trusted=True` on an
  attacker-controlled artifact still executes code: untrusted pickle cannot be
  made safe.
- **AI operator.** Prompt-injection heuristics in `buildml.ai.security` are a
  best-effort layer (NFKC + zero-width / bidi strip, Latin-homoglyph fold,
  multi-line / base64-ish smuggle patterns, structured `InjectionFinding`
  reason codes). Primary controls remain the closed tool registry (runtime
  `register` refused), confirm-on-write for mutating tools, and egress levels in
  `buildml.ai.privacy`. **Residual risk:** paraphrase / novel attacks may
  bypass heuristics: do not treat pattern matching as injection-proof.

---

## A classical workflow

**Facades are the supported public API for domains** in 2.5.x
(`session.fairness.*`, `session.anomaly.*`, …). Flat domain aliases still work
but emit `DeprecationWarning` until BuildML 3.0. Classical core is dual on
purpose: flat `session.fit` / `session.evaluate` / … stay first-class without
warnings, and `session.classical.*` / `session.data.*` / `session.preprocess.*`
are equivalent paths — not a secondary API. See
[`docs/session-facade-migration.md`](docs/session-facade-migration.md).

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
# Default impute/encode/scale touch feature-role columns only :
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
fences should be refit inside each fold: on **unpoisoned** data (no prior
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
estimators and preprocess plans: they do not embed each other.
`workflow()` marks operations as done / available / blocked / skipped from API
prerequisites; it does not judge domain fit.

---

## Learning while you work

Every operation and concept is written for three reading levels. `beginner` is
the default and assumes **no** prior machine-learning vocabulary: plain-language
summary, an analogy, the steps in order, what each parameter means in practice,
the pitfalls, and a glossary of the terms the answer itself used.

```python
before = session.explain("split")          # level="beginner" by default
print(before.beginner.plain_summary)
print(before.beginner.analogy)
for step in before.beginner.steps:
    print("-", step)
for knob in before.beginner.key_parameters:
    print(knob.name, "→", knob.plain_meaning, "|", knob.typical_choice)

session.explain("split", moment="after")   # what it did, in this session
session.explain("split", level="advanced") # same facts, no hand-holding
```

`explain` is about **this session right now**: what is missing, what changed,
how to read the result. When the question is conceptual instead, use `learn`,
which takes a concept key, an operation name, or whatever word tripped you up:

```python
session.learn()                    # where to start, in reading order
session.learn("leakage")           # a term → the concept that teaches it
session.learn("stratified")        # spelling/punctuation is forgiving
brief = session.learn("data-splitting")
[note.key for note in brief.read_first]   # read these before this one
[note.key for note in brief.read_next]    # read these once it lands
```

| Level | Shows |
| --- | --- |
| `beginner` | Analogy, plain steps, in-line glossary, worked example |
| `intermediate` | Same facts, less scaffolding, more parameters |
| `advanced` | Full assumptions / leakage / failure lists, no glossary |

The same material backs `Session.explain`, `Session.learn`, `workflow()`,
`walkthrough()`, and the AI operator's `explain_operation` / `learn_concept`
tools, so no surface can drift from another. Teaching content explains ideas and
BuildML's contract; it does not inspect your data or certify that a choice is
appropriate for it.

---

## EDA and reports

```python
# Offline Industry App snapshot (default html_format="studio")
session.eda(export_html="artifacts/eda_studio.html")

# BUILDML STATIC EDA: Industry readiness sheet (Offline HTML primary in header)
session.eda(
    include_plots=True,
    export_html="artifacts/eda_research.html",
    html_format="research",
)

# pip install "buildml[dashboard]"
handle = session.eda_app(port=8765)
# handle.url -> http://127.0.0.1:8765/
# Cockpit spine 01-08, Readiness Gates, Concept Academy (~204 catalog lessons)
# Primary export in the app header: Offline HTML

session.evaluate(
    partition="test",
    include_plots=True,
    export_html="artifacts/evaluation.html",
)
```

Reports surface screening evidence. They do not establish causality, fairness,
or deployment readiness on their own. Local launch helpers and a multi-dataset
adaptability check live under `scripts/` (`launch_synthetic_eda_studio.py`,
`generate_static_eda_preview.py`, `eda_adaptability_gauntlet.py`). Tier A proof:
[`proofs/eda-industry-adaptability/`](proofs/eda-industry-adaptability/)
(12 datasets; regenerate via that script or the gauntlet). See
[EDA / Teaching Studio](guides/eda-teaching-studio.md), the
[classical quickstart](guides/quickstart-classical.md), and the
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
| Forecast | [forecasting](guides/quickstart-forecasting.md) | `time_split` lags/baselines |
| Time-series analysis | [timeseries-analysis](guides/quickstart-timeseries-analysis.md) | `session.timeseries.analyze` / decompose / diagnostics (no forecast fit) |
| Anomaly | [anomaly](guides/quickstart-anomaly.md) | IsolationForest / LOF / OCSVM + supervised |
| Semi / SSL / AL / Online | matching quickstarts | sklearn floor; industry/torch deepen |
| Multi-task / Meta / Federated | matching quickstarts | MultiOutput / few-shot / FedAvg sim |
| Probabilistic / Causal | matching quickstarts | Conformal; assumption-declared ATE |
| Graph / Symbolic / CBR | matching quickstarts | NetworkX/Torch/PyG; rules; case memory |
| Recommenders / LTR / KG | matching quickstarts | CF + content; GBDT rankers; TransE-style |
| Optimize / Synthetic / IL+RL | matching quickstarts | Thresholds/knapsack; SDV optional; BC + bandits + tabular Q-learning/SARSA |
| TDA | [tda](guides/quickstart-tda.md) | ripser/persim (`buildml[tda]`) |
| NLP | [nlp](guides/quickstart-nlp.md) | Document classify + token attribution, topics, keyphrases, summaries, entities, sentiment, language, corpus profile; `buildml[nlp]` adds encoders |
| Torch | [torch](guides/quickstart-torch.md) | Tabular / text / image / audio fusion |
| RAG | [rag](guides/quickstart-rag.md) | Hashing default; sentence-transformers optional |
| AI operator | [ai](guides/quickstart-ai.md) | Propose→confirm→execute; allowlisted autonomy |

Torch covers tabular MLP, text/sequence, and multimodal fusion; speech
(`buildml[speech]`) is ASR + finetune-lite: not Whisper-scale FM training from
scratch. RAG defaults to lexical hashing; semantic embeddings and grounded
`session.rag.generate` are first-class when extras resolve. NLP models a text column that
lives on the dataset: document classification and analysis, distinct from RAG
retrieval and from Torch fine-tuning. The AI operator defaults to
propose→confirm→execute: not unconstrained agency.

Full guide index: [`guides/README.md`](guides/README.md).

---

## Proof suite

End-to-end evidence that Session domains work with honest splits and holdout
metrics lives under [`proofs/`](proofs/README.md): **not** smoke tests.

| Tier | Status | Meaning |
| --- | --- | --- |
| A | **63/63** | One deep project per major domain (incl. ensembles + Torch + **REAL_PUBLIC_DATASET** cohort + Industry EDA) |
| B | **36/36** | Named products composing multiple Session surfaces |
| C | **58/62** | Same-split industry twin + `comparison.json` (qualitative bar 5-B; real-public cohort may be A-only) |

```bash
# Full harness from repo root
python -m proofs._lib.run_all --tier all

# Single project (synthetic or real public)
python proofs/loan-approval-classical/script.py
python proofs/breast-cancer-classical/script.py
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

## Stability and scope

BuildML **2.5.0** is the stable Session 2.x line. SemVer applies to the public
Session / facade surface; see [`docs/stability.md`](docs/stability.md).

Shipped: observational fairness disparity reports (`session.fairness.evaluate`) and
optional SHAP attribution (`explain_shap` via `buildml[shap]`). Still out of
scope: out-of-core sklearn training, legal fairness certification, and
unconstrained LLM agency. Speech ASR defaults to a CI-safe stub backend;
`backend="transformers"` is optional and not Whisper-scale FM training.

See [CHANGELOG.md](CHANGELOG.md) for release notes and
[guides/glossary.md](guides/glossary.md) for BuildML terminology:
`session.learn(term)` covers general machine-learning vocabulary.

---

## Documentation

- [Proof suite](https://github.com/TechLeo-Libraries/BuildML/blob/main/proofs/README.md) — Tier A/B/C inventory, harness, Tier C interpretation
- [Guides](https://github.com/TechLeo-Libraries/BuildML/blob/main/guides/README.md) — quickstarts (each major domain links its proof) and [glossary](https://github.com/TechLeo-Libraries/BuildML/blob/main/guides/glossary.md)
- [Concepts](https://buildml.readthedocs.io/en/latest/concepts.html) — roles, partitions, train-fitted plans
- [Workflow guide](https://buildml.readthedocs.io/en/latest/workflow-guide.html) — ordering, leakage, diagnostics
- [Sphinx docs](https://buildml.readthedocs.io/en/latest/) — installation, features, API reference, legacy boundary
- [Changelog](https://github.com/TechLeo-Libraries/BuildML/blob/main/CHANGELOG.md) — release notes

---

## BuildML 1.x legacy boundary

BuildML 1.x (`SupervisedLearning` and the old module layout) lives under
`buildml/_legacy/` for reference only. It is not imported from the 2.x package
root. There is no compatibility shim that re-exports 1.x APIs from
`import buildml`.

---

## Author and license

**Leonard Onyiriuba**: [LinkedIn](https://www.linkedin.com/in/chukwubuikem-leonard-onyiriuba/) · leonard.c.onyiriuba@gmail.com

Issues: [GitHub](https://github.com/TechLeo-Libraries/BuildML/issues)

Apache License 2.0.
