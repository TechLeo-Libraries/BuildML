# Classical alpha gate

Concrete exit criteria for declaring BuildML 2.x classical ML **alpha-ready**.
This is a release checklist, not a capability wishlist. Items are either met by
CI/tests/docs today or listed as known limits.

Related docs: [quickstart-alpha.md](./quickstart-alpha.md) ·
[glossary.md](./glossary.md) · [workflow-guide.rst](./workflow-guide.rst) ·
[editorial-standards.md](../editorial-standards.md)

---

## Verdict rubric

| Status | Meaning |
| --- | --- |
| **Pass** | Every **must** criterion below is green in CI or explicitly verified |
| **Fail** | Any **must** criterion is red, missing, or contradicted by docs |
| **Conditional** | Musts pass, but a listed known limit blocks a claimed workflow |

Assess readiness after CI: **Pass** when all must IDs are green; otherwise
**Fail** or **Conditional** per the known-limits section.

---

## Must criteria

### Leakage and fit scope

| ID | Criterion | Evidence |
| --- | --- | --- |
| L1 | Train-fitted Session preprocess (impute/encode/scale/dates/bin/outliers/select/text/reduce) fits on train only | Unit + integration preprocess tests |
| L2 | `cv_score` / search / `nested_cv_score` never use Session test rows for fold membership or scoring | `test_selection_cv`, nested CV tests |
| L3 | Fold-local `PreprocessRecipe` refits dates, text, outliers, impute, encode, binning, scale, reduce (PCA), and select on fold-train only | `test_fold_recipe_extensions`, `test_fold_text_reduce`, selection CV tests |
| L4 | Session-global-only steps are documented: resample, `apply_custom_transform`, and any Session plan fitted before CV when not in the recipe | `SESSION_GLOBAL_ONLY_STEPS`, glossary, workflow guide |
| L5 | Target / infrequent encoding and supervised select inside CV do not peek at fold-eval labels | Selection + fold recipe leakage tests |

### End-to-end smoke

| ID | Criterion | Evidence |
| --- | --- | --- |
| S1 | Core path: ingest → roles → light EDA → split → prep → `cv_score` or search → fit → evaluate → checkpoint save/load → pipeline save/load → `predict_from_pipeline` | `tests/integration/test_classical_alpha_smoke.py` |
| S2 | Smoke runs on core install (no engines/optuna/dashboard extras required) | CI `test` job |
| S3 | Checkpoint and pipeline artifacts remain distinct (data vs fitted bundle) | Checkpoint/pipeline smoke + S1 |

### Docs and catalog

| ID | Criterion | Evidence |
| --- | --- | --- |
| D1 | Public Session methods used by the learner path have catalog entries | `buildml.explain.catalog` + explain tests |
| D2 | Fold-local vs Session-global preprocess limits are documented and disclosed | glossary, workflow guide, `PreprocessRecipe` / `SESSION_GLOBAL_ONLY_STEPS`, walkthrough + Teaching Studio `preprocess_scope_status` |
| D3 | Quickstart covers split → prep → fit → evaluate and CV with `PreprocessRecipe` | `docs/quickstart-alpha.md` |
| D4 | Editorial / user-copy lint clean | `scripts/lint_user_copy.py` in CI |

### CI and packaging

| ID | Criterion | Evidence |
| --- | --- | --- |
| C1 | `import buildml` on core install | CI import smoke |
| C2 | `ruff check` + pytest (with coverage) on Python 3.10–3.13 | `.github/workflows/ci.yml` |
| C3 | Wheel/sdist build succeeds | CI build step |
| C4 | Optional engines / Optuna / extras have dedicated jobs (skip-friendly) | CI `engines`, `optuna`, `extras` jobs |

---

## Should criteria (alpha-tolerant)

| ID | Criterion | Notes |
| --- | --- | --- |
| W1 | Native Polars/DuckDB `project` / `filter` / `aggregate` (incl. `median` / `qN`) before Pandas | Core Pandas path always; engines optional; continuous quantile interpolation |
| W2 | Nested CV + Optuna recipe knobs | Requires `buildml[optuna]` |
| W3 | Teaching Studio / rich plot boards; warm-start + preprocess-scope disclosures | Require dashboard/viz extras for live studio; offline walkthrough HTML is core |
| W4 | Schema contract coerce + `predict_from_pipeline` on new frames | Covered by pipeline tests; keep in smoke |

---

## Known limits (do not claim as done)

1. **Custom transforms stay Session-global.** Registered callables are not part of
   `PreprocessRecipe`; they are unfit for fold-local search honesty.
2. **Resample is Session-global.** Train-row rewrite is not applied inside CV folds.
3. **Sklearn still needs an in-memory design matrix.** Native Polars/DuckDB handles
   avoid full-width Pandas for project/filter/aggregate/sample; they do not enable
   out-of-core fitting.
4. **Hashing text features are not invertible.** Prefer TF-IDF when token names matter.
5. **PCA explained variance is unsupervised.** It is not a substitute for holdout metrics.
6. **Deep learning / RAG / LLM operator** are out of classical alpha scope.
7. **Fairness / SHAP-style explainability** remain later extras.

---

## Smoke path (canonical)

```text
Session.ingest → set_roles → eda(include_plots=False)
  → split → impute → scale
  → cv_score(..., preprocess=PreprocessRecipe(...))  # or grid_search
  → fit → evaluate(partition="test")
  → checkpoint_save / checkpoint_load
  → save_pipeline / load_pipeline
  → predict_from_pipeline(path, holdout_frame)
```

CI entry: `pytest tests/integration/test_classical_alpha_smoke.py -q`

---

## Sign-off checklist

Copy into a release note when cutting an alpha tag (see also
[release-checklist-a1.md](./release-checklist-a1.md)):

- [ ] L1–L5 green
- [ ] S1–S3 green on core CI
- [ ] D1–D4 green
- [ ] C1–C4 green
- [ ] Known limits reviewed; README/quickstart/`CHANGELOG.md` do not contradict them
- [ ] Version is `2.0.0a1` in `pyproject.toml` and `buildml/_version.py`
- [ ] Changelog / history notes name this gate document

Local verification for post-alpha polish (2026-08-01): classical smoke, fold
preprocess-scope Teaching Studio surfaces, and aggregate median/quantile tests
were exercised on the developer machine; confirm the full CI matrix (C1–C4
across Python versions) on the next push before tagging `v2.0.0a1`.
