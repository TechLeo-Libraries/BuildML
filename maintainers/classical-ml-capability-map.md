# Classical ML capability map

Planning inventory of workflow turns BuildML v2 should account for.  
Goal: professionals are not limited; learners get a simple path through the same surface.

Status key:

| Tag | Meaning |
| --- | --- |
| **F** | Foundation / parity with corrected v1 intent |
| **D** | Depth for v2 professional completeness |
| **L** | Later (after classical depth or via another domain) |
| **X** | Explicit non-goal for classical domain (belongs elsewhere) |

---

## 1. Ingest and data understanding

| Capability | Tag | Notes |
| --- | --- | --- |
| Load CSV / Parquet / Arrow / DataFrame | F | Paths and in-memory objects |
| Schema inference + dtype report | F | Including nullability |
| Memory / row / partition estimates | F | Drive engine mode suggestions |
| Sample preview / head profiles | F | Required for large data |
| Data quality checks (constants, dupes, type conflicts) | D | Actionable issue list |
| Dataset catalog (multi-table later) | L | Single-table first |

## 2. Exploratory data analysis

| Capability | Tag | Notes |
| --- | --- | --- |
| Tabular summary (describe, dtypes, missingness) | F | |
| Univariate / bivariate plots | F | Optional plotting extras |
| Correlation / association views | F | Handle mixed types safely |
| Target-aware EDA | D | Class balance, target vs feature |
| Sample-aware EDA on large data | F | Never force full materialization for profile |
| HTML profiling reports | D | Extra; privacy warnings |
| Segment / group analysis | D | |

## 3. Cleaning and preparation

| Capability | Tag | Notes |
| --- | --- | --- |
| Drop / keep / rename columns | F | |
| Filter rows; deduplicate | F | |
| Missing-value strategies | F | Fit on train after split when used in modeling path |
| Outlier handling | F | Row-safe semantics; documented strategies |
| Replace / map values | F | |
| Type casting / downcast | F | Memory efficiency |
| Date/time parse + feature extraction | F | Correct `.dt` semantics |
| Binning / discretization | F | |
| Text normalization (basic) | D | Session + fold-local `PreprocessRecipe(text=...)` |
| Custom user transforms | D | Registered callables / sklearn-compatible |

## 4. Feature engineering and selection

| Capability | Tag | Notes |
| --- | --- | --- |
| Encoding (one-hot, ordinal, target-safe options) | F | Train-fit only |
| Scaling / normalization | F | Train-fit only |
| Polynomial / interaction features | F | Honest validation when searching degree |
| Feature selection utilities | F | No mutating shared X across search iters |
| Dimensionality reduction helpers | D | PCA via Session + fold-local `PreprocessRecipe(reduce='pca')` |
| Column roles (feature, target, group, time, id, weight) | F | Core to leakage control |
| Feature importance surfaces | D | Model-dependent; documented limits |

## 5. Splitting and resampling

| Capability | Tag | Notes |
| --- | --- | --- |
| Random train/test split | F | |
| Stratified split | F | |
| Train / validation / test | D | First-class three-way |
| Group-aware split | D | Prevent group leakage |
| Time-aware split | D | Critical professional path |
| External pre-split injection | F | Explicit APIs + validation |
| Imbalance resampling | F | Train-only; pipeline-aware |
| CV splitters catalog | D | KFold, Stratified, Group, TimeSeries |

## 6. Training and model selection

| Capability | Tag | Notes |
| --- | --- | --- |
| Fit caller-supplied sklearn-compatible estimator | F | |
| Multi-model compare | F | Structured results |
| Cross-validated scoring | F | Default honest path for selection |
| Nested CV (outer estimate after inner search) | D | `nested_cv_score` |
| Nested CV over fold-local recipe knobs | D | `recipe_grid` / `recipe_distributions` / Optuna `recipe_space` |
| Nested CV Optuna inner search | D | `inner_search='optuna'` + `buildml[optuna]` |
| Hyperparameter search wrappers | D | Grid / randomized / Optuna (`optuna_search`) |
| Pipeline fit (preprocess + model) | F | Single artifact |
| Probability estimation (`predict_proba`) | D | Where supported |

| Decision function / ranking scores | D | |
| Calibration | D | |
| Threshold tuning (binary/multiclass strategies) | D | |
| Class weight / sample weight | F | `ColumnRole.WEIGHT` → sklearn `sample_weight` on fit/evaluate/CV/search; unsupported estimators raise |
| Reproducible seeds | F | |

## 7. Evaluation and diagnostics

| Capability | Tag | Notes |
| --- | --- | --- |
| Classification metrics suite | F | Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, log-loss, … |
| Regression metrics suite | F | MAE, MSE, RMSE, R², MAPE (guarded), … |
| Confusion matrix / classification report | F | |
| ROC / PR curves | D | |
| Residual / prediction diagnostic plots | D | |
| Learning curves / validation curves | D | |
| Error analysis slices | D | By segment/column |
| Experiment comparison tables | D | |
| Fairness / subgroup metrics | L | Important; dedicated design later |
| Explainability (SHAP/permutation) | L | Extra; careful deps |

## 8. Inference, export, persistence

| Capability | Tag | Notes |
| --- | --- | --- |
| Predict on held-out / new data through fitted pipeline | F | `predict_from_pipeline` |
| Export metrics / predictions tables | F | |
| Persist fitted pipeline | D | joblib/skops-style + `schema_contract.json` |
| Score-time schema contract validation | D | missing / extra / wrong-type checks |
| Load fitted pipeline for inference-only session | D | |

| Checkpoint session mid-loop | F | See checkpoint design |
| Reattach externalized data | F | Validation matrix |
| Model cards / run summaries | D | Markdown/JSON |

## 9. Professional flexibility

| Capability | Tag | Notes |
| --- | --- | --- |
| Guided mode (BuildML owns split/fit scope) | F | Default |
| Professional mode (inject partitions, custom scorers) | D | Still leakage-safe |
| Escape to Pandas / Polars / Parquet anytime | F | |
| Persistent native Polars/DuckDB Dataset handles | D | project/filter/sample/aggregate before `to_pandas` |
| Polars LazyFrame native handle | D | collect-on-promote; not out-of-core sklearn |
| Polars `filter_expr` parity | D | SQL-style expr; keeps LazyFrames lazy when possible |
| Portable `filter_expr` helper | D | `portable_filter_expr` for simple Polars/DuckDB predicates |
| DuckDB Dataset connection lifecycle | D | owned/reused connection; `close_native`; `with dataset/session:` |
| Checkpoint native reattach | D | optional sidecar; public layout/compression/partition knobs |
| DuckDB Arrow/IPC attach | D | pyarrow Feather/IPC without Pandas bridge |
| DuckDB SQL filter/sample pushdown | D | mask/expr/sample before full Arrow collect |
| Nested CV Optuna study warm-start | D | `warm_start_studies=True` (default off; leakage-audited) |
| Lazy-native Teaching Studio disclosure | D | overview + walkthrough + cockpit |
| Warm-start Teaching Studio / walkthrough | D | factual policy/risks/shared state when enabled |
| Bring back with schema/role/split validation | F | |
| Dry-run plans (what would run / what would materialize) | D | Especially for large data |
| Operation history / audit trail | D | Supports teaching + debugging |

## 10. Explicitly outside classical v2 core

| Capability | Tag | Notes |
| --- | --- | --- |
| Deep learning trainers | X → `buildml.dl` | Same session patterns later |
| RAG / embeddings index / retrieval | X → `buildml.rag` | Later domain |
| LLM natural-language operator | X → `buildml.ai` | Later design |
| Full AutoML black box | L | Optional opinionated wrapper later; not the identity |
| Hosted server / multi-tenant SaaS | X | Library-first |

---

## Learner vs professional progressive disclosure

**Learner path (same objects, fewer knobs):**  
ingest → eda → clean → encode → split → fit → score → plot

**Professional path (same session, full knobs):**  
roles → quality checks → time/group splits → pipeline recipe → CV/search → probabilities/calibration/thresholds → diagnostics → persist → checkpoint/reattach

---

## v2 depth definition of done (classical)

Classical depth is not “feature count.” A turn is done when it has:

1. Session method(s) + underlying implementation (no duplication)
2. Leakage / partition rules documented and tested
3. Typed result object(s)
4. Extensive docstring + at least one executable example
5. Scale note (memory / lazy / materialization requirements)
6. CI coverage

---

## Suggested implementation waves

| Wave | Focus |
| --- | --- |
| W1 | Ingest, roles, split, clean/encode/scale, fit/predict/metrics, session basics |
| W2 | Pipelines, CV, multi-model compare, date features, imbalance, checkpoints |
| W3 | Probabilities, calibration, thresholds, rich plots/diagnostics |
| W4 | Time/group splits, search, persistence, experiment comparison |
| W5 | Quality checks, dry-run plans, audit trail, advanced extras |

This map should be revised as we discover practitioner gaps, but it is the completeness target for “not mediocre.”
