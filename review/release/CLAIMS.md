# BuildML claim-to-evidence register

Release confidence is supported by reproducible checks of documented workflows and environments, with validation coverage and remaining limitations recorded explicitly.

Status: **acceptance pending**. This register maps 49 material claim groups to implementation, tests, assumptions and existing evidence. The user serves as the external human reviewer. No release or human approval is implied.

## Evidence interpretation

The baseline Git HEAD is `83d1b14571ad9ec44e30d986f256bb552488bda4`. The reviewed working tree also contains uncommitted repairs, so that HEAD alone does not identify the candidate. Bind acceptance to the source manifest created by `scripts/release_snapshot.py`, then attach the tested distribution hashes.

The prior full-suite XML records 1,263 passes, 151 skips and one teaching wording failure; a later 44-test scoped rerun verified its correction. Counts in the JSON refer to whole test files in that historical run, not claim-specific coverage or a fresh aggregate pass. Current conformal acceptance tests postdate that run. A later 12-test EDA log is separately recorded. Skips are unverified paths.

The README is the source for the next package long description. This register does not verify currently deployed PyPI or Read the Docs pages. Existing local 2.6.2 validation archives are not replacements for published distributions.

## Acceptance procedure

1. Identify the exact candidate source manifest and distribution hashes.
2. Have an independent reviewer challenge each material claim against the cited implementation, failure paths and tests. Test-file pass totals alone do not approve a claim.
3. Reproduce relevant tests against the candidate package. Record commands, environments, results and skips.
4. Record fixes and independent rechecks. Reviewer-authored changes need another reviewer.
5. Record the human decision for each material claim: accept, narrow, defer or reject. Open critical/high findings and unsupported material claims block release.

## Claim map

All entries await frozen-candidate acceptance. Sources and tests are review entry points, not assertions of exhaustive branch or statistical coverage.

### BML-CORE-001

Session coordinates data, roles, partitions, preprocessing, fitted models and operation history.

Public source: [README.md](../../README.md) — BuildML.

Implementation: [buildml/session/state.py](../../buildml/session/state.py), [buildml/session/data_ops.py](../../buildml/session/data_ops.py), [buildml/session/classical_ops.py](../../buildml/session/classical_ops.py).

Tests: [tests/integration/test_classical_alpha_smoke.py](../../tests/integration/test_classical_alpha_smoke.py), [tests/integration/test_classical_parity_flow.py](../../tests/integration/test_classical_parity_flow.py), [tests/unit/test_session_facades.py](../../tests/unit/test_session_facades.py).

Dependencies: core.

Review boundaries: Stateful orchestration does not infer the correct scientific study design.

### BML-DATA-001

Ingest supports DataFrames and documented file formats with detection and loading reports.

Public source: [docs/features.rst](../../docs/features.rst) — Data and workflow.

Implementation: [buildml/ingest/loaders.py](../../buildml/ingest/loaders.py), [buildml/ingest/pipeline.py](../../buildml/ingest/pipeline.py).

Tests: [tests/integration/test_import_and_ingest.py](../../tests/integration/test_import_and_ingest.py), [tests/unit/test_data_mode.py](../../tests/unit/test_data_mode.py).

Dependencies: core.

Review boundaries: Excel and optional engines require their dependencies. Individual formats and malformed inputs need separate acceptance; file-level tests are not exhaustive format validation.

### BML-DATA-002

Random, stratified, grouped, chronological and injected partitions are available.

Public source: [docs/features.rst](../../docs/features.rst) — Data and workflow.

Implementation: [buildml/session/data_ops.py](../../buildml/session/data_ops.py).

Tests: [tests/unit/test_group_time_splits.py](../../tests/unit/test_group_time_splits.py), [tests/unit/test_splits_and_leakage.py](../../tests/unit/test_splits_and_leakage.py).

Dependencies: core.

Review boundaries: Users must identify repeated entities and temporal ordering. Random partitions do not establish statistical independence.

### BML-PREP-001

Fit-capable core preprocessing learns on training rows and applies fitted plans to holdouts.

Public source: [docs/features.rst](../../docs/features.rst) — Preparation.

Implementation: [buildml/session/preprocess_ops.py](../../buildml/session/preprocess_ops.py).

Tests: [tests/unit/test_prep_depth_and_audit.py](../../tests/unit/test_prep_depth_and_audit.py), [tests/unit/test_preprocess.py](../../tests/unit/test_preprocess.py), [tests/unit/test_preprocess_depth.py](../../tests/unit/test_preprocess_depth.py), [tests/unit/test_preprocess_native_sync.py](../../tests/unit/test_preprocess_native_sync.py), [tests/unit/test_preprocess_role_skip.py](../../tests/unit/test_preprocess_role_skip.py), [tests/unit/test_preprocess_scope_teaching.py](../../tests/unit/test_preprocess_scope_teaching.py).

Dependencies: core.

Review boundaries: Custom transforms are Session-global, not automatically fold-local. External feature engineering can introduce leakage. Resampling requires the imbalanced extra.

### BML-CV-001

Classical CV/search use training-partition folds and support fold-local PreprocessRecipe.

Public source: [README.md](../../README.md) — What the Session protects.

Implementation: [buildml/model/selection.py](../../buildml/model/selection.py), [buildml/session/classical_ops.py](../../buildml/session/classical_ops.py).

Tests: [tests/unit/test_evolutionary_search.py](../../tests/unit/test_evolutionary_search.py), [tests/unit/test_nested_cv.py](../../tests/unit/test_nested_cv.py), [tests/unit/test_nested_cv_optuna.py](../../tests/unit/test_nested_cv_optuna.py), [tests/unit/test_nested_cv_warm_start.py](../../tests/unit/test_nested_cv_warm_start.py), [tests/unit/test_optuna_search.py](../../tests/unit/test_optuna_search.py), [tests/unit/test_selection_cv.py](../../tests/unit/test_selection_cv.py).

Dependencies: core.

Review boundaries: Start from unprocessed data; allow_session_global_preprocess explicitly accepts biased validation risk. Optuna is optional. Repeated consultation of test scores can still bias selection.

### BML-MODEL-001

Core classification/regression fit sklearn-compatible estimators and evaluate named partitions.

Public source: [docs/features.rst](../../docs/features.rst) — Classical models and diagnostics.

Implementation: [buildml/session/classical_ops.py](../../buildml/session/classical_ops.py).

Tests: [tests/integration/test_classical_alpha_smoke.py](../../tests/integration/test_classical_alpha_smoke.py), [tests/integration/test_classical_parity_flow.py](../../tests/integration/test_classical_parity_flow.py), [tests/unit/test_model_reporting_milestone4.py](../../tests/unit/test_model_reporting_milestone4.py).

Dependencies: core.

Review boundaries: Compatibility and metric validity depend on task/data. compare_models defaults to test; select validation during iterative model selection.

### BML-ARTIFACT-001

Checkpoints store workflow data/roles/splits/history; pipeline bundles store fitted plans and estimator.

Public source: [README.md](../../README.md) — Save, reload, and trust.

Implementation: [buildml/checkpoint/bundle.py](../../buildml/checkpoint/bundle.py), [buildml/checkpoint/validate.py](../../buildml/checkpoint/validate.py), [buildml/pipeline/persist.py](../../buildml/pipeline/persist.py), [buildml/pipeline/bundle.py](../../buildml/pipeline/bundle.py).

Tests: [tests/integration/test_checkpoint_pipeline_smoke.py](../../tests/integration/test_checkpoint_pipeline_smoke.py), [tests/integration/test_checkpoint_roundtrip.py](../../tests/integration/test_checkpoint_roundtrip.py), [tests/integration/test_pipeline_bundle_roundtrip.py](../../tests/integration/test_pipeline_bundle_roundtrip.py), [tests/integration/test_predict_from_pipeline.py](../../tests/integration/test_predict_from_pipeline.py), [tests/unit/test_checkpoint_native_reattach.py](../../tests/unit/test_checkpoint_native_reattach.py), [tests/unit/test_checkpoint_native_sidecar.py](../../tests/unit/test_checkpoint_native_sidecar.py), [tests/unit/test_checkpoint_sidecar_layout.py](../../tests/unit/test_checkpoint_sidecar_layout.py).

Dependencies: core.

Review boundaries: Trusted loading is required for pickle/joblib payloads; integrity checking does not establish trusted authorship. Checkpoints and pipeline bundles do not embed each other. Cross-version compatibility requires review.

### BML-ENGINES-001

Optional Polars and DuckDB materialization integrates with Session data.

Public source: [docs/features.rst](../../docs/features.rst) — Data and workflow.

Implementation: [buildml/session/data_ops.py](../../buildml/session/data_ops.py).

Tests: [tests/unit/test_dataset_native.py](../../tests/unit/test_dataset_native.py), [tests/unit/test_engine_aggregate.py](../../tests/unit/test_engine_aggregate.py), [tests/unit/test_engine_prep.py](../../tests/unit/test_engine_prep.py), [tests/unit/test_engines.py](../../tests/unit/test_engines.py).

Dependencies: polars / duckdb / engines.

Review boundaries: Availability depends on installed engines; no out-of-core sklearn training guarantee.

### BML-TEACH-001

Versioned explanations and workflow guidance describe public operations and expose state.

Public source: [docs/features.rst](../../docs/features.rst) — Teaching and reports.

Implementation: [buildml/session/workflow_ops.py](../../buildml/session/workflow_ops.py), [buildml/explain/generated/operation_index.json](../../buildml/explain/generated/operation_index.json).

Tests: [tests/unit/test_explain_beginner_layer.py](../../tests/unit/test_explain_beginner_layer.py), [tests/unit/test_explain_catalog.py](../../tests/unit/test_explain_catalog.py), [tests/unit/test_explain_runtime.py](../../tests/unit/test_explain_runtime.py), [tests/unit/test_session_discovery.py](../../tests/unit/test_session_discovery.py), [tests/unit/test_teaching_examples.py](../../tests/unit/test_teaching_examples.py), [tests/unit/test_teaching_surface_sync.py](../../tests/unit/test_teaching_surface_sync.py).

Dependencies: core.

Review boundaries: Teaching explains contracts, not scientific suitability. The historical full run includes one jargon failure repaired in a later scoped run; see teaching-closeout evidence.

### BML-EDA-001

EDA accepts explicit exploration partitions and discloses the separate train-versus-test drift scope.

Public source: [guides/eda-teaching-studio.md](../../guides/eda-teaching-studio.md) — Use case: findings before preparation.

Implementation: [buildml/session/eda_ops.py](../../buildml/session/eda_ops.py), [buildml/eda/profile.py](../../buildml/eda/profile.py), [buildml/eda/report.py](../../buildml/eda/report.py), [buildml/eda/analyzers/drift.py](../../buildml/eda/analyzers/drift.py).

Tests: [tests/unit/test_eda_audit_regressions.py](../../tests/unit/test_eda_audit_regressions.py).

Dependencies: core.

Review boundaries: Default exploration uses all rows. partition=train still permits the separately disclosed full train/test drift comparison. Using holdout findings for model/feature selection compromises an untouched test set. Drift results record train/test row counts and analyzed feature columns. Scores describe the supplied data representation; review prior preprocessing separately. Columns with too few usable observations and output limits constrain screening coverage.

### BML-EDA-002

Continuous declared feature columns remain eligible; unavailable drift differs from no flagged drift.

Public source: [docs/features.rst](../../docs/features.rst) — Teaching and reports.

Implementation: [buildml/eda/analyzers/quality.py](../../buildml/eda/analyzers/quality.py), [buildml/eda/analyzers/drift.py](../../buildml/eda/analyzers/drift.py), [buildml/eda/profile.py](../../buildml/eda/profile.py).

Tests: [tests/unit/test_eda_audit_regressions.py](../../tests/unit/test_eda_audit_regressions.py), [tests/unit/test_eda_depth.py](../../tests/unit/test_eda_depth.py).

Dependencies: core.

Review boundaries: Identifier screening and drift thresholds are heuristics; no warning does not establish absence of drift or deployment fitness.

### BML-EDA-003

EDA separates missing/nonfinite observations from usable data and discloses quality counts.

Public source: [docs/features.rst](../../docs/features.rst) — Teaching and reports.

Implementation: [buildml/eda/analyzers/target.py](../../buildml/eda/analyzers/target.py), [buildml/eda/profile.py](../../buildml/eda/profile.py), [buildml/eda/visualize.py](../../buildml/eda/visualize.py).

Tests: [tests/unit/test_eda_audit_regressions.py](../../tests/unit/test_eda_audit_regressions.py).

Dependencies: core.

Review boundaries: Task inference remains heuristic for integer-coded targets and can differ from intent. Exclusion changes denominators; review counts. EDA does not repair the original Session data.

### BML-EDA-004

Read-only EDA findings and recommendations are available through research HTML, offline studio and local dashboard.

Public source: [docs/features.rst](../../docs/features.rst) — Teaching and reports.

Implementation: [buildml/eda/html_report.py](../../buildml/eda/html_report.py), [buildml/dashboard/app.py](../../buildml/dashboard/app.py).

Tests: [tests/unit/test_eda_academy.py](../../tests/unit/test_eda_academy.py), [tests/unit/test_eda_adaptability.py](../../tests/unit/test_eda_adaptability.py), [tests/unit/test_eda_audit_regressions.py](../../tests/unit/test_eda_audit_regressions.py), [tests/unit/test_eda_cockpit_ux.py](../../tests/unit/test_eda_cockpit_ux.py), [tests/unit/test_eda_dashboard.py](../../tests/unit/test_eda_dashboard.py), [tests/unit/test_eda_depth.py](../../tests/unit/test_eda_depth.py), [tests/unit/test_eda_gates.py](../../tests/unit/test_eda_gates.py), [tests/unit/test_eda_sheet_coverage.py](../../tests/unit/test_eda_sheet_coverage.py), [tests/unit/test_reporting_shell.py](../../tests/unit/test_reporting_shell.py).

Dependencies: eda / viz / dashboard.

Review boundaries: Readiness gates are heuristic aids and tab-local UI marks, not persisted approvals. Route/schema tests do not replace browser/accessibility review.

### BML-DOMAIN-UNSUPERVISED

Train-fitted clustering with holdout assignment and geometric evaluation.

Public source: [docs/features.rst](../../docs/features.rst) — Unsupervised.

Implementation: [buildml/session/unsupervised_ops.py](../../buildml/session/unsupervised_ops.py).

Tests: [tests/integration/test_unsupervised_alpha_smoke.py](../../tests/integration/test_unsupervised_alpha_smoke.py), [tests/unit/test_unsupervised_m2_depth.py](../../tests/unit/test_unsupervised_m2_depth.py), [tests/unit/test_unsupervised_r2_depth.py](../../tests/unit/test_unsupervised_r2_depth.py), [tests/unit/test_unsupervised_slice.py](../../tests/unit/test_unsupervised_slice.py).

Dependencies: core; unsupervised extra.

Review boundaries: Clusters are not ground-truth classes. Capability availability and smoke tests are not benchmark-quality evidence.

### BML-DOMAIN-ENSEMBLE

Voting, stacking and blending keep internal folds/holdouts within train.

Public source: [docs/features.rst](../../docs/features.rst) — Ensembles.

Implementation: [buildml/session/ensemble_ops.py](../../buildml/session/ensemble_ops.py).

Tests: [tests/integration/test_ensemble_alpha_smoke.py](../../tests/integration/test_ensemble_alpha_smoke.py), [tests/unit/test_ensemble_m2_depth.py](../../tests/unit/test_ensemble_m2_depth.py), [tests/unit/test_ensemble_reporting_high.py](../../tests/unit/test_ensemble_reporting_high.py), [tests/unit/test_ensemble_slice.py](../../tests/unit/test_ensemble_slice.py).

Dependencies: core.

Review boundaries: Group/time dependencies still require appropriate fold design. Capability availability and smoke tests are not benchmark-quality evidence.

### BML-DOMAIN-AUTOML

Budgeted model-family and preprocessing search exposes backend capabilities.

Public source: [docs/features.rst](../../docs/features.rst) — AutoML.

Implementation: [buildml/session/automl_ops.py](../../buildml/session/automl_ops.py).

Tests: [tests/integration/test_automl_alpha_smoke.py](../../tests/integration/test_automl_alpha_smoke.py), [tests/unit/test_automl_m2_depth.py](../../tests/unit/test_automl_m2_depth.py), [tests/unit/test_automl_r5_industry.py](../../tests/unit/test_automl_r5_industry.py), [tests/unit/test_automl_reporting_high.py](../../tests/unit/test_automl_reporting_high.py), [tests/unit/test_automl_slice.py](../../tests/unit/test_automl_slice.py).

Dependencies: core; optuna / automl-industry.

Review boundaries: Installation and data constrain backend search; not NAS or causal discovery. Capability availability and smoke tests are not benchmark-quality evidence.

### BML-DOMAIN-FORECAST

Forecast fit/generate/evaluate follows time roles and time splits.

Public source: [docs/features.rst](../../docs/features.rst) — Forecasting.

Implementation: [buildml/session/forecast_ops.py](../../buildml/session/forecast_ops.py).

Tests: [tests/integration/test_forecasting_alpha_smoke.py](../../tests/integration/test_forecasting_alpha_smoke.py), [tests/unit/test_forecasting_m2_depth.py](../../tests/unit/test_forecasting_m2_depth.py), [tests/unit/test_forecasting_r3_depth.py](../../tests/unit/test_forecasting_r3_depth.py), [tests/unit/test_forecasting_slice.py](../../tests/unit/test_forecasting_slice.py).

Dependencies: core; see capability catalog.

Review boundaries: Required future exogenous values must be provided; no unrestricted econometrics/sequence-model platform. Capability availability and smoke tests are not benchmark-quality evidence.

### BML-DOMAIN-TIMESERIES

Time-series analysis/decomposition/diagnostics are distinct from forecast fitting.

Public source: [docs/features.rst](../../docs/features.rst) — Time-series analysis.

Implementation: [buildml/session/timeseries_ops.py](../../buildml/session/timeseries_ops.py).

Tests: [tests/unit/test_timeseries_r3_depth.py](../../tests/unit/test_timeseries_r3_depth.py), [tests/unit/test_timeseries_teaching_wiring.py](../../tests/unit/test_timeseries_teaching_wiring.py).

Dependencies: timeseries / timeseries-prophet / timeseries-ml.

Review boundaries: Optional algorithms need installed backends; analysis alone does not fit a forecast model. Capability availability and smoke tests are not benchmark-quality evidence.

### BML-DOMAIN-ANOMALY

Supported anomaly estimators score/evaluate and tune thresholds.

Public source: [docs/features.rst](../../docs/features.rst) — Anomaly / fraud.

Implementation: [buildml/session/anomaly_ops.py](../../buildml/session/anomaly_ops.py).

Tests: [tests/integration/test_anomaly_alpha_smoke.py](../../tests/integration/test_anomaly_alpha_smoke.py), [tests/unit/test_anomaly_industry_depth.py](../../tests/unit/test_anomaly_industry_depth.py), [tests/unit/test_anomaly_m2_depth.py](../../tests/unit/test_anomaly_m2_depth.py), [tests/unit/test_anomaly_slice.py](../../tests/unit/test_anomaly_slice.py).

Dependencies: core; anomaly-industry.

Review boundaries: Scores do not establish fraud or causality; threshold tuning must preserve evaluation independence. Capability availability and smoke tests are not benchmark-quality evidence.

### BML-DOMAIN-SEMISUPERVISED

Fitting uses labeled/unlabeled train rows; holdout metrics are labeled-only.

Public source: [docs/features.rst](../../docs/features.rst) — Semi-supervised.

Implementation: [buildml/session/semisupervised_ops.py](../../buildml/session/semisupervised_ops.py).

Tests: [tests/integration/test_semisupervised_alpha_smoke.py](../../tests/integration/test_semisupervised_alpha_smoke.py), [tests/unit/test_semisupervised_industry_depth.py](../../tests/unit/test_semisupervised_industry_depth.py), [tests/unit/test_semisupervised_m2_depth.py](../../tests/unit/test_semisupervised_m2_depth.py), [tests/unit/test_semisupervised_slice.py](../../tests/unit/test_semisupervised_slice.py).

Dependencies: core; semisupervised-industry.

Review boundaries: Missing targets indicate unlabeled rows; algorithm assumptions/class coverage matter. Capability availability and smoke tests are not benchmark-quality evidence.

### BML-DOMAIN-SSL

Documented pretext training, transforms, supervised heads and bundle roundtrips are supported.

Public source: [docs/features.rst](../../docs/features.rst) — Self-supervised.

Implementation: [buildml/session/selfsupervised_ops.py](../../buildml/session/selfsupervised_ops.py).

Tests: [tests/integration/test_selfsupervised_alpha_smoke.py](../../tests/integration/test_selfsupervised_alpha_smoke.py), [tests/unit/test_selfsupervised_m2_depth.py](../../tests/unit/test_selfsupervised_m2_depth.py), [tests/unit/test_selfsupervised_slice.py](../../tests/unit/test_selfsupervised_slice.py), [tests/unit/test_selfsupervised_torch.py](../../tests/unit/test_selfsupervised_torch.py).

Dependencies: core; torch / ssl.

Review boundaries: Five Torch tabular roundtrips have recorded execution; vision/HF downloads require separate acceptance. Capability availability and smoke tests are not benchmark-quality evidence.

### BML-DOMAIN-ACTIVE

Queries target the unlabeled train pool and respect budgets.

Public source: [docs/features.rst](../../docs/features.rst) — Active learning.

Implementation: [buildml/session/activelearning_ops.py](../../buildml/session/activelearning_ops.py).

Tests: [tests/integration/test_activelearning_alpha_smoke.py](../../tests/integration/test_activelearning_alpha_smoke.py), [tests/unit/test_activelearning_industry_depth.py](../../tests/unit/test_activelearning_industry_depth.py), [tests/unit/test_activelearning_m2_depth.py](../../tests/unit/test_activelearning_m2_depth.py), [tests/unit/test_activelearning_slice.py](../../tests/unit/test_activelearning_slice.py).

Dependencies: core; activelearning-industry.

Review boundaries: Human labels are required; the query strategy is not an oracle. Capability availability and smoke tests are not benchmark-quality evidence.

### BML-DOMAIN-ONLINE

Incremental updates use train chunks with supported estimators.

Public source: [docs/features.rst](../../docs/features.rst) — Online / continual.

Implementation: [buildml/session/online_ops.py](../../buildml/session/online_ops.py).

Tests: [tests/integration/test_online_alpha_smoke.py](../../tests/integration/test_online_alpha_smoke.py), [tests/unit/test_online_industry_depth.py](../../tests/unit/test_online_industry_depth.py), [tests/unit/test_online_m2_depth.py](../../tests/unit/test_online_m2_depth.py), [tests/unit/test_online_slice.py](../../tests/unit/test_online_slice.py).

Dependencies: core; online-industry.

Review boundaries: No distributed streaming guarantee; unsupported silent full refits are refused. Capability availability and smoke tests are not benchmark-quality evidence.

### BML-DOMAIN-MULTITASK

Multioutput/chained methods support multiple same-type targets.

Public source: [docs/features.rst](../../docs/features.rst) — Multi-task.

Implementation: [buildml/session/multitask_ops.py](../../buildml/session/multitask_ops.py).

Tests: [tests/integration/test_multitask_alpha_smoke.py](../../tests/integration/test_multitask_alpha_smoke.py), [tests/unit/test_multitask_industry_depth.py](../../tests/unit/test_multitask_industry_depth.py), [tests/unit/test_multitask_m2_depth.py](../../tests/unit/test_multitask_m2_depth.py), [tests/unit/test_multitask_slice.py](../../tests/unit/test_multitask_slice.py).

Dependencies: core; multitask-industry.

Review boundaries: Mixed classification/regression targets are unsupported in the documented path. Capability availability and smoke tests are not benchmark-quality evidence.

### BML-DOMAIN-META

Documented episodic methods evaluate held-out tasks and disclose overlap.

Public source: [docs/features.rst](../../docs/features.rst) — Meta-learning.

Implementation: [buildml/session/metalearning_ops.py](../../buildml/session/metalearning_ops.py).

Tests: [tests/integration/test_metalearning_alpha_smoke.py](../../tests/integration/test_metalearning_alpha_smoke.py), [tests/unit/test_metalearning_m2_depth.py](../../tests/unit/test_metalearning_m2_depth.py), [tests/unit/test_metalearning_r65_industry.py](../../tests/unit/test_metalearning_r65_industry.py), [tests/unit/test_metalearning_slice.py](../../tests/unit/test_metalearning_slice.py).

Dependencies: core; torch / metalearning-industry.

Review boundaries: Task identities/episode construction determine validity; no foundation-model meta-learning claim. Capability availability and smoke tests are not benchmark-quality evidence.

### BML-DOMAIN-FEDERATED

Local FedAvg/FedProx simulation uses declared clients/groups.

Public source: [docs/features.rst](../../docs/features.rst) — Federated.

Implementation: [buildml/session/federated_ops.py](../../buildml/session/federated_ops.py).

Tests: [tests/integration/test_federated_alpha_smoke.py](../../tests/integration/test_federated_alpha_smoke.py), [tests/unit/test_federated_industry_depth.py](../../tests/unit/test_federated_industry_depth.py), [tests/unit/test_federated_m2_depth.py](../../tests/unit/test_federated_m2_depth.py), [tests/unit/test_federated_slice.py](../../tests/unit/test_federated_slice.py).

Dependencies: core; federated-industry adapters.

Review boundaries: Local simulation is not network federation, secure aggregation or a privacy guarantee. Capability availability and smoke tests are not benchmark-quality evidence.

### BML-DOMAIN-PROBABILISTIC

Documented probabilistic models and train-contained conformal intervals expose coverage evaluation.

Public source: [docs/features.rst](../../docs/features.rst) — Probabilistic.

Implementation: [buildml/session/probabilistic_ops.py](../../buildml/session/probabilistic_ops.py).

Tests: [tests/integration/test_probabilistic_alpha_smoke.py](../../tests/integration/test_probabilistic_alpha_smoke.py), [tests/unit/test_probabilistic_industry.py](../../tests/unit/test_probabilistic_industry.py), [tests/unit/test_probabilistic_m2_depth.py](../../tests/unit/test_probabilistic_m2_depth.py), [tests/unit/test_probabilistic_slice.py](../../tests/unit/test_probabilistic_slice.py).

Dependencies: core; probabilistic-industry.

Review boundaries: Conformal guarantees require statistical assumptions; empirical coverage is not universal calibration. No general MCMC engine. Capability availability and smoke tests are not benchmark-quality evidence.

### BML-DOMAIN-CAUSAL

Explicit assumptions/estimand precede supported backdoor effect estimation and refutation.

Public source: [docs/features.rst](../../docs/features.rst) — Causal.

Implementation: [buildml/session/causal_ops.py](../../buildml/session/causal_ops.py).

Tests: [tests/integration/test_causal_alpha_smoke.py](../../tests/integration/test_causal_alpha_smoke.py), [tests/unit/test_causal_industry_depth.py](../../tests/unit/test_causal_industry_depth.py), [tests/unit/test_causal_m2_depth.py](../../tests/unit/test_causal_m2_depth.py).

Dependencies: core; causal-industry.

Review boundaries: Declarations do not prove assumptions; confounding, positivity and misspecification require expert review. Capability availability and smoke tests are not benchmark-quality evidence.

### BML-DOMAIN-GRAPH

Supported node-classification backends include NetworkX, Torch GCN and optional PyG.

Public source: [docs/features.rst](../../docs/features.rst) — Graph ML.

Implementation: [buildml/session/graph_ops.py](../../buildml/session/graph_ops.py).

Tests: [tests/integration/test_graph_alpha_smoke.py](../../tests/integration/test_graph_alpha_smoke.py), [tests/unit/test_graph_catalog.py](../../tests/unit/test_graph_catalog.py), [tests/unit/test_graph_m2_depth.py](../../tests/unit/test_graph_m2_depth.py), [tests/unit/test_graph_slice.py](../../tests/unit/test_graph_slice.py).

Dependencies: graph / torch / graph-pyg.

Review boundaries: Graph construction/transductive boundaries require review; not a graph database. Capability availability and smoke tests are not benchmark-quality evidence.

### BML-DOMAIN-SYMBOLIC

Declared/induced rules and supported hybrid methods return traces.

Public source: [docs/features.rst](../../docs/features.rst) — Symbolic.

Implementation: [buildml/session/symbolic_ops.py](../../buildml/session/symbolic_ops.py).

Tests: [tests/integration/test_symbolic_alpha_smoke.py](../../tests/integration/test_symbolic_alpha_smoke.py), [tests/unit/test_symbolic_industry.py](../../tests/unit/test_symbolic_industry.py), [tests/unit/test_symbolic_m2_depth.py](../../tests/unit/test_symbolic_m2_depth.py), [tests/unit/test_symbolic_slice.py](../../tests/unit/test_symbolic_slice.py).

Dependencies: core; symbolic-industry.

Review boundaries: Traceability does not establish rule validity or comprehensive logical inference. Capability availability and smoke tests are not benchmark-quality evidence.

### BML-DOMAIN-CBR

Case memory uses train rows with supported retrieval/prediction.

Public source: [docs/features.rst](../../docs/features.rst) — Case-based reasoning.

Implementation: [buildml/session/cbr_ops.py](../../buildml/session/cbr_ops.py).

Tests: [tests/integration/test_cbr_alpha_smoke.py](../../tests/integration/test_cbr_alpha_smoke.py), [tests/unit/test_cbr_industry_depth.py](../../tests/unit/test_cbr_industry_depth.py), [tests/unit/test_cbr_m2_depth.py](../../tests/unit/test_cbr_m2_depth.py), [tests/unit/test_cbr_slice.py](../../tests/unit/test_cbr_slice.py).

Dependencies: core; cbr-industry / cbr-faiss.

Review boundaries: Similarity choices determine validity; distinct from RAG. Capability availability and smoke tests are not benchmark-quality evidence.

### BML-DOMAIN-RL

Core behavioral cloning/contextual bandits and documented optional RL integrations are available.

Public source: [docs/features.rst](../../docs/features.rst) — Imitation + RL.

Implementation: [buildml/session/rl_ops.py](../../buildml/session/rl_ops.py).

Tests: [tests/integration/test_rl_alpha_smoke.py](../../tests/integration/test_rl_alpha_smoke.py), [tests/unit/test_rl_industry.py](../../tests/unit/test_rl_industry.py), [tests/unit/test_rl_m2_depth.py](../../tests/unit/test_rl_m2_depth.py), [tests/unit/test_rl_slice.py](../../tests/unit/test_rl_slice.py), [tests/unit/test_rl_tabular.py](../../tests/unit/test_rl_tabular.py).

Dependencies: core; rl / rl-industry.

Review boundaries: Offline/toy tests do not establish environment correctness, deployment safety or online exploration quality. Capability availability and smoke tests are not benchmark-quality evidence.

### BML-DOMAIN-TDA

Vietoris-Rips features and vectorization can feed a sklearn head.

Public source: [docs/features.rst](../../docs/features.rst) — TDA.

Implementation: [buildml/session/tda_ops.py](../../buildml/session/tda_ops.py).

Tests: [tests/integration/test_tda_alpha_smoke.py](../../tests/integration/test_tda_alpha_smoke.py), [tests/unit/test_tda_giotto_padding.py](../../tests/unit/test_tda_giotto_padding.py), [tests/unit/test_tda_m2_depth.py](../../tests/unit/test_tda_m2_depth.py), [tests/unit/test_tda_r5_industry.py](../../tests/unit/test_tda_r5_industry.py), [tests/unit/test_tda_slice.py](../../tests/unit/test_tda_slice.py).

Dependencies: tda / tda-industry.

Review boundaries: Absent optional dependencies were skipped; complexity and filtration assumptions constrain use. Capability availability and smoke tests are not benchmark-quality evidence.

### BML-DOMAIN-RECOMMENDER

Collaborative/content recommendation supports ranking evaluation.

Public source: [docs/features.rst](../../docs/features.rst) — Recommenders.

Implementation: [buildml/session/recommender_ops.py](../../buildml/session/recommender_ops.py).

Tests: [tests/integration/test_recommender_alpha_smoke.py](../../tests/integration/test_recommender_alpha_smoke.py), [tests/unit/test_recommender_industry.py](../../tests/unit/test_recommender_industry.py), [tests/unit/test_recommender_m2_depth.py](../../tests/unit/test_recommender_m2_depth.py), [tests/unit/test_recommender_slice.py](../../tests/unit/test_recommender_slice.py).

Dependencies: core; recommenders-industry / recommenders-lightfm.

Review boundaries: Cold-start, exposure bias and temporal evaluation require review. LightFM is a separate platform-limited extra. Capability availability and smoke tests are not benchmark-quality evidence.

### BML-DOMAIN-RANKING

Query-item ranking supports sklearn and installed Torch/industry alternatives.

Public source: [docs/features.rst](../../docs/features.rst) — Learning to rank.

Implementation: [buildml/session/ranking_ops.py](../../buildml/session/ranking_ops.py).

Tests: [tests/integration/test_ranking_alpha_smoke.py](../../tests/integration/test_ranking_alpha_smoke.py), [tests/unit/test_ranking_industry_depth.py](../../tests/unit/test_ranking_industry_depth.py), [tests/unit/test_ranking_m2_depth.py](../../tests/unit/test_ranking_m2_depth.py), [tests/unit/test_ranking_slice.py](../../tests/unit/test_ranking_slice.py).

Dependencies: core; torch / ranking-industry.

Review boundaries: Metrics require correct query groups; availability is not ranking-quality evidence. Capability availability and smoke tests are not benchmark-quality evidence.

### BML-DOMAIN-KG

Supported triple embeddings and symbolic queries operate on graph triples.

Public source: [docs/features.rst](../../docs/features.rst) — Knowledge graphs.

Implementation: [buildml/session/kg_ops.py](../../buildml/session/kg_ops.py).

Tests: [tests/integration/test_kg_alpha_smoke.py](../../tests/integration/test_kg_alpha_smoke.py), [tests/unit/test_kg_industry_depth.py](../../tests/unit/test_kg_industry_depth.py), [tests/unit/test_kg_m2_depth.py](../../tests/unit/test_kg_m2_depth.py), [tests/unit/test_kg_slice.py](../../tests/unit/test_kg_slice.py).

Dependencies: core; kg-industry.

Review boundaries: Negative-sampling/evaluation design matters; not a graph database or RAG engine. Capability availability and smoke tests are not benchmark-quality evidence.

### BML-DOMAIN-DECISION

Threshold/cost/top-K/knapsack and documented LP decisions are available.

Public source: [docs/features.rst](../../docs/features.rst) — Decisions.

Implementation: [buildml/session/decision_ops.py](../../buildml/session/decision_ops.py).

Tests: [tests/integration/test_optimize_alpha_smoke.py](../../tests/integration/test_optimize_alpha_smoke.py), [tests/unit/test_optimize_cvxpy_adapter.py](../../tests/unit/test_optimize_cvxpy_adapter.py), [tests/unit/test_optimize_industry_depth.py](../../tests/unit/test_optimize_industry_depth.py), [tests/unit/test_optimize_m2_depth.py](../../tests/unit/test_optimize_m2_depth.py), [tests/unit/test_optimize_slice.py](../../tests/unit/test_optimize_slice.py).

Dependencies: core; optional optimization solvers.

Review boundaries: Results depend on costs, constraints and solver availability; not a general optimization platform. Capability availability and smoke tests are not benchmark-quality evidence.

### BML-DOMAIN-FAIRNESS

Observational holdout fairness metrics support optional stability/intersectional analysis.

Public source: [docs/features.rst](../../docs/features.rst) — Fairness.

Implementation: [buildml/session/fairness_ops.py](../../buildml/session/fairness_ops.py).

Tests: [tests/unit/test_fairness_slice.py](../../tests/unit/test_fairness_slice.py).

Dependencies: core; shap optional.

Review boundaries: Groups, sample size and normative meaning require judgment; not automatic mitigation or legal certification. Capability availability and smoke tests are not benchmark-quality evidence.

### BML-DOMAIN-SYNTHETIC

Bootstrap/copula/SMOTE and optional SDV support fidelity/TSTR evaluation.

Public source: [docs/features.rst](../../docs/features.rst) — Synthetic data.

Implementation: [buildml/session/synthetic_ops.py](../../buildml/session/synthetic_ops.py).

Tests: [tests/integration/test_synthetic_alpha_smoke.py](../../tests/integration/test_synthetic_alpha_smoke.py), [tests/unit/test_synthetic_industry_depth.py](../../tests/unit/test_synthetic_industry_depth.py), [tests/unit/test_synthetic_m2_depth.py](../../tests/unit/test_synthetic_m2_depth.py), [tests/unit/test_synthetic_slice.py](../../tests/unit/test_synthetic_slice.py).

Dependencies: core; synthetic-industry / imbalanced as required.

Review boundaries: Fidelity does not establish privacy, representativeness or downstream benefit; no differential privacy guarantee. Capability availability and smoke tests are not benchmark-quality evidence.

### BML-DOMAIN-NLP

Core corpus/classification/topic/keyphrase/summary/entity/sentiment/language tools support text columns.

Public source: [docs/features.rst](../../docs/features.rst) — NLP.

Implementation: [buildml/session/nlp_ops.py](../../buildml/session/nlp_ops.py).

Tests: [tests/integration/test_nlp_alpha_smoke.py](../../tests/integration/test_nlp_alpha_smoke.py), [tests/unit/test_nlp_industry_depth.py](../../tests/unit/test_nlp_industry_depth.py), [tests/unit/test_nlp_m2_depth.py](../../tests/unit/test_nlp_m2_depth.py), [tests/unit/test_nlp_slice.py](../../tests/unit/test_nlp_slice.py).

Dependencies: core; nlp / nlp-industry.

Review boundaries: Extractive summaries can omit context or repeat source errors; optional pretrained models may download files. Capability availability and smoke tests are not benchmark-quality evidence.

### BML-DOMAIN-RAG

Corpus ingest, hashing/BM25/NumPy retrieval, cited generation and bundles are supported.

Public source: [docs/features.rst](../../docs/features.rst) — RAG.

Implementation: [buildml/session/rag_ops.py](../../buildml/session/rag_ops.py).

Tests: [tests/integration/test_rag_alpha_smoke.py](../../tests/integration/test_rag_alpha_smoke.py), [tests/integration/test_rag_smoke.py](../../tests/integration/test_rag_smoke.py), [tests/unit/test_rag_generate.py](../../tests/unit/test_rag_generate.py), [tests/unit/test_rag_m2_depth.py](../../tests/unit/test_rag_m2_depth.py), [tests/unit/test_rag_nlp_restore_high.py](../../tests/unit/test_rag_nlp_restore_high.py), [tests/unit/test_rag_r4_depth.py](../../tests/unit/test_rag_r4_depth.py), [tests/unit/test_rag_slice.py](../../tests/unit/test_rag_slice.py).

Dependencies: core; rag / rag-advanced.

Review boundaries: Retrieval/citations do not guarantee factual correctness; hosted providers need credentials and semantic models may download. Capability availability and smoke tests are not benchmark-quality evidence.

### BML-DOMAIN-AI

Advisor/planning, confirmed execution and explicitly autonomous execution apply documented caps.

Public source: [docs/features.rst](../../docs/features.rst) — AI operator.

Implementation: [buildml/session/ai_ops.py](../../buildml/session/ai_ops.py).

Tests: [tests/unit/test_ai_phase_c.py](../../tests/unit/test_ai_phase_c.py), [tests/unit/test_ai_slice.py](../../tests/unit/test_ai_slice.py).

Dependencies: ai.

Review boundaries: Mock tests do not establish hosted-provider reliability; credentials/network routes need separate validation. Capability availability and smoke tests are not benchmark-quality evidence.

### BML-DOMAIN-TORCH

Documented Torch loaders/modules/training, CV/search, distributed paths and exports are exposed.

Public source: [docs/features.rst](../../docs/features.rst) — Torch.

Implementation: [buildml/session/dl_ops.py](../../buildml/session/dl_ops.py).

Tests: [tests/integration/test_dl_alpha_smoke.py](../../tests/integration/test_dl_alpha_smoke.py), [tests/integration/test_dl_torch_smoke.py](../../tests/integration/test_dl_torch_smoke.py), [tests/unit/test_dl_m2_depth.py](../../tests/unit/test_dl_m2_depth.py), [tests/unit/test_dl_phase_c.py](../../tests/unit/test_dl_phase_c.py), [tests/unit/test_dl_torch_slice.py](../../tests/unit/test_dl_torch_slice.py).

Dependencies: torch / onnx.

Review boundaries: CPU-local tests do not establish GPU/AMP, multi-node DDP or every modality/export runtime. Capability availability and smoke tests are not benchmark-quality evidence.

### BML-SPEECH-001

Speech supports documented ASR stub/transformer modes and WER/CER evaluation.

Public source: [docs/features.rst](../../docs/features.rst) — Speech.

Implementation: [buildml/dl/speech.py](../../buildml/dl/speech.py).

Tests: [tests/unit/test_pass_o_speech_ddp_serve.py](../../tests/unit/test_pass_o_speech_ddp_serve.py), [tests/unit/test_speech_asr_default_high.py](../../tests/unit/test_speech_asr_default_high.py).

Dependencies: speech.

Review boundaries: Stub outputs are fixtures, not transcription-quality evidence; pretrained downloads/device behavior require separate validation.

### BML-PRETRAINED-001

Curated backbones support documented none/mock/pretrained weight sources.

Public source: [docs/features.rst](../../docs/features.rst) — Pretrained backbones.

Implementation: [buildml/dl/zoo.py](../../buildml/dl/zoo.py).

Tests: [tests/unit/test_pass_r_pretrained_serve_k8s.py](../../tests/unit/test_pass_r_pretrained_serve_k8s.py).

Dependencies: vision / speech / pretrained.

Review boundaries: Mock/random weights are not pretrained quality evidence. Downloads/licenses/architectures require per-source review.

### BML-SERVE-001

Local FastAPI exposes documented pipeline/TorchScript serving endpoints.

Public source: [docs/features.rst](../../docs/features.rst) — Serve.

Implementation: [buildml/serving/app.py](../../buildml/serving/app.py), [buildml/serving/launch.py](../../buildml/serving/launch.py).

Tests: [tests/unit/test_pass_o_speech_ddp_serve.py](../../tests/unit/test_pass_o_speech_ddp_serve.py), [tests/unit/test_pass_r_pretrained_serve_k8s.py](../../tests/unit/test_pass_r_pretrained_serve_k8s.py).

Dependencies: serve.

Review boundaries: Localhost defaults and route tests are not a security assessment or cloud IAM. Authentication/network boundaries/load need acceptance.

### BML-DIST-001

Metadata declares Python 3.10-3.13, four core dependencies and optional extras.

Public source: [README.md](../../README.md) — Optional extras.

Implementation: [pyproject.toml](../../pyproject.toml).

Tests: [tests/integration/test_import_and_ingest.py](../../tests/integration/test_import_and_ingest.py), [tests/unit/test_documentation_copy.py](../../tests/unit/test_documentation_copy.py).

Dependencies: core.

Review boundaries: Executed validation used existing Windows/Python 3.12 dependencies. Fresh online resolution and full OS/Python/extra matrix remain acceptance work. Production extras are best-effort.

### BML-DOCS-001

README supplies the package long description; sources and examples have local validation evidence.

Public source: [README.md](../../README.md) — BuildML.

Implementation: [pyproject.toml](../../pyproject.toml), [README.md](../../README.md), [docs/conf.py](../../docs/conf.py).

Tests: [tests/unit/test_copy_audit_coverage.py](../../tests/unit/test_copy_audit_coverage.py), [tests/unit/test_documentation_copy.py](../../tests/unit/test_documentation_copy.py), [tests/unit/test_teaching_examples.py](../../tests/unit/test_teaching_examples.py).

Dependencies: core.

Review boundaries: Local builds do not update deployed PyPI/Read the Docs. Current 2.6.2 validation archives must not replace existing published distributions.

### BML-CONFORMAL-001

Finite-sample conformal calibration rejects unattainable finite cutoffs and nonfinite calibration scores.

Public source: [docs/features.rst](../../docs/features.rst) — Probabilistic; [buildml/probabilistic/conformal.py](../../buildml/probabilistic/conformal.py) — conformal_quantile docstring.

Implementation: [buildml/probabilistic/conformal.py](../../buildml/probabilistic/conformal.py).

Tests: [tests/unit/test_conformal_acceptance.py](../../tests/unit/test_conformal_acceptance.py), [tests/unit/test_probabilistic_m2_depth.py](../../tests/unit/test_probabilistic_m2_depth.py).

Dependencies: core.

Review boundaries: With n calibration observations, the requested alpha must permit a finite order-statistic cutoff; otherwise the method rejects instead of silently returning a misleading bounded interval. Coverage assumptions still require exchangeability and an appropriate calibration design; time/group dependence is not resolved by the numeric cutoff check.

## Evidence retention

Historical logs are under gitignored `artifacts/`. Preserve them in the human-review package: a source checkout without those logs is not the complete evidence record. [claims.json](claims.json) records their paths and SHA-256 hashes at register creation, stable claim IDs, historical file-level counts, and unfilled independent/human acceptance fields.

The register groups material advertised capabilities; it does not state that every sentence in the repository has received individual acceptance. Further findings belong in the candidate review record.

## Latest local evidence

The current probabilistic/teaching rerun passed 34 tests with 3 optional skips. Six installed-wheel acceptance checks passed in a fresh Windows Python 3.12 environment with independently resolved core dependencies. See `artifacts/probabilistic-acceptance.log` and `artifacts/release-review/fresh-install.json`; the JSON register records their hashes. The 12-cell CI matrix and human decisions remain pending. The independent statistical report verifies the finite-sample cutoff repair; that is not human approval of all probabilistic claims.
