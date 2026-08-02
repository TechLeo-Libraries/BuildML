# BuildML architecture review

Read-only audit of repository HEAD `3d4ccae` · source release 1.0.9 · audited 1 Aug 2026.

> Canvas twin (open beside chat if available):  
> `C:/Users/leona/.cursor/projects/c-Users-leona-Desktop-Github-Projects-BuildML/canvases/buildml-architecture-review.canvas.tsx`  
>  
> Planning follow-ons: [reconstruction-roadmap.md](./reconstruction-roadmap.md) · [classical-ml-capability-map.md](./classical-ml-capability-map.md) · [ingest-engine-checkpoint-design.md](./ingest-engine-checkpoint-design.md)

## Snapshot

| Metric | Value |
| --- | --- |
| Public root class | 1 (`SupervisedLearning`) |
| Facade methods | 52 |
| Tracked files | 52 |
| Automated tests | 0 |

**Release blocker:** The source parses, but the package does not import in the audited environment because the root API eagerly imports optional visualization dependencies. Several deterministic code paths also fail independently of dependency installation.

**Product in one sentence:** BuildML is an educational, notebook-oriented Python toolkit that wraps a complete tabular supervised-learning workflow—exploration, dataframe mutation, feature engineering, splitting, estimator training, evaluation, tuning, visualization, and report export—behind one mutable object.

**Data flow:** Pandas DataFrame → `SupervisedLearning` mutable facade → Prepare (clean → encode → select → split) → Model (caller estimator → fit → predict) → Results (nested dicts, DataFrames, plots, HTML).

## Implementation status

- Broad feature surface and extensive docstrings exist; PyPI 1.0.9 was published on 27 Jan 2024.
- Best characterized as a **beta prototype**: no tests/CI, no tagged releases, stale packaging, duplicate APIs, and multiple release-blocking correctness defects.
- Latest commit: 4 Apr 2025.
- Runtime model: library only (no server, CLI, database, API route, auth, or container). Runs in the user’s Python process; state in memory; produces console output, Matplotlib/notebook plots, and local HTML/CSV/XLSX files.

## Architecture notes

The facade does **not** delegate to the functional modules. It reimplements them, so fixes can diverge and package users receive two subtly different APIs.

### API groups on `SupervisedLearning`

| Area | Important API | Responsibility |
| --- | --- | --- |
| State and workflow | `__init__`, `get_dataset`, `get_training_test_data`, `select_dependent_and_independent`, `split_data` | Holds original/current data, X/y, splits, models, scaler, polynomial data, and 17 workflow flags. |
| Cleaning and shape | `drop_columns`, `fix_missing_values`, `remove_outlier`, `replace_values`, `filter_data`, `remove_duplicates`, sort/index/rename helpers | Mostly mutates `__data`; filtering is a large branch matrix. |
| Types and features | `categorical_to_numerical`, datetime helpers, `extract_date_features`, `column_binning`, `select_features`, `polyreg_x` | One-hot encoding, casts, date parts, bins, sklearn selectors, polynomial expansion. |
| Explore and summarize | `eda`, `eda_visual`, `group_data`, category/unique helpers | Returns pandas summaries or renders plots. |
| Train and infer | `train_model_regressor/classifier`, predict/testing helpers | Fits caller-provided sklearn-compatible estimators. |
| Evaluate and compare | evaluation helpers; `build_*_from_features`; `build_multiple_*` | Nested dictionaries/DataFrames; optional CV. |
| Specialized modeling | KNN best-k, poly degree, classifier/linreg/poly graphs | Training-score K search, polynomial sweep, 2D visualizations. |
| I/O and profiling | large dataset load, memory downcast, Sweetviz / pandas profiling | Datatable ingestion and local reports. |

### Boundaries

- Public root → `automate` facade. Separately importable `build_model`, `preprocessing`, `date_features`, `eda`, and `output_dataset` expose functions but are not composed by the facade.
- State transitions are implicit boolean flags rather than a typed workflow or fitted pipeline.
- No runtime SaaS dependency. GitHub / PyPI / Read the Docs / Ko-fi are distribution and docs surfaces only.
- Profilers generate local reports that can contain full dataset values and distributions.

## Module inventory

| Path | Role | Contents |
| --- | --- | --- |
| `buildml/__init__.py` | Public package root | Eagerly exports only `SupervisedLearning`; stores 1.0.9 metadata. |
| `automate/_automate.py` | Primary user facade | Stateful ~6,604-line class with 52 workflow methods. |
| `build_model/_model.py` | Functional modeling API | 15 functions for split/selection/fit/metrics/sweeps/plots. |
| `preprocessing/_preprocessing.py` | Functional dataframe utilities | 21 cleaning/filter/type/scale/sample helpers. |
| `date_features/_date.py` | Date transforms | Two functions; extraction is currently broken. |
| `eda/_eda.py` | Exploration and reports | Summaries, Matplotlib/Seaborn, Sweetviz, ydata-profiling. |
| `output_dataset/_output_dataset.py` | File export | CSV/XLSX; not connected to root facade. |
| Six package `__init__.py` files | Namespace exports | Only `SupervisedLearning` is available from `buildml`. |
| `setup.py` + `requirements.txt` | Distribution | Setuptools 1.0.9; unpinned dependency list. |
| `docs/` | Sphinx site source (RTD) | Published HTML at buildml.readthedocs.io |
| `guides/` | User quickstarts and glossary (Markdown) | Included in Sphinx via MyST |
| `maintainers/` | Phase plans, gates, checklists | Git only |
| `.spyproject/` | Editor metadata | Tracked Spyder prefs; not runtime configuration. |

## Risk register

| Urgency | Finding | Impact / evidence | Location |
| --- | --- | --- | --- |
| P0 | Package cannot import in audited environment | Eager `seaborn` import via root facade | `buildml/__init__.py`; `automate/_automate.py` |
| P0 | Published dependency contract unsatisfiable/unbounded | `imbalanced_learn` + incorrect `imblearn`, legacy `scikit_learn`, no pins | `requirements.txt`; `setup.py` |
| P0 | Known runtime-breaking paths | Missing `.dt`; unreachable branch; undefined `items`; invalid `sort_index` args | `date_features/_date.py`; `preprocessing/_preprocessing.py` |
| P0 | Model-selection result keys disagree | Readers expect `Built Model`; producer returns `Model` → `KeyError` | `build_model/_model.py` |
| P1 | Preprocessing leakage / invalid inference scaling | Fit before split; prediction scalers never fit | `automate/_automate.py`; `build_model/_model.py` |
| P1 | Evaluation overstates generalization | KNN on train score; poly CV on unexpanded X; feature sweeps mutate X | automate + build_model |
| P1 | No tests or CI | No test tree, workflow, coverage, lint, type-check, build, or release automation | repository inventory |
| P1 | 6,500-line mutable facade duplicates lower layers | One class owns dataset state, flags, models, reports, plots, transforms | `automate/_automate.py` |
| P2 | Legacy packaging / false compatibility metadata | No `pyproject.toml`; docs recommend direct setup.py; `Python >=3.0` | `setup.py`; docs |
| P2 | Generated artifacts pollute history/worktree | Tracked `.pyc`; local wheels/sdists/egg-info/build/Sphinx output | git inventory |

## Recommended release gates

1. **Restore install/import correctness** — `pyproject.toml`, real Python support, canonical dependency names and bounded versions, optional extras, reproducible environment matrix.
2. **Lock behavior with tests before refactoring** — unit tests for public functions/state transitions; regressions for P0 defects; estimator/pandas/sklearn compatibility; executable docs examples.
3. **Separate workflow state from transformations** — sklearn-compatible transformers/pipelines; fitted preprocess artifacts with models; no leakage; structured results; compatibility facade only if required.
4. **Automate quality and release** — format/lint/type, coverage, wheel/sdist smoke, docs checks, trusted PyPI publish, changelog/version single-sourcing, artifact hygiene.

## Owner decisions needed

1. Preserve stateful `SupervisedLearning` API, or introduce composable sklearn-style transformers/pipelines?
2. Which Python versions and OSes are actually supported?
3. Should Sweetviz, ydata-profiling, Datatable, and Excel be optional extras so core imports without them?
4. Must 1.0.9 return-dictionary keys and in-place mutation remain backward compatible?
5. Educational notebook helper or production library?
6. Are externally supplied train/test datasets a supported workflow, or should BuildML own splitting/pipeline fitting?
7. Privacy policy for generated profiling reports?
8. Was 1.0.3 intentionally omitted from local `dist/`, and should historical artifacts stay outside source control?

## Inspection coverage

All 52 tracked paths were accounted for: 49 authored text/source/configuration files were read; three tracked bytecode binaries were inventoried. Generated/vendor build contents were inventoried without wasteful line-by-line review.

**Diagnostic outcomes:** All Python files parse. No Git tags. Origin points to `TechLeo-Libraries/BuildML`. Root import fails at missing `seaborn`. Generated `build/lib` samples match authored source hashes. Local env had NumPy 2.2.6 and lacked several declared dependencies; Sweetviz 2.3.1 reported incompatible with NumPy 2.x.
