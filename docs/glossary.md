# BuildML glossary

Terms here describe the current BuildML 2.x API. They are not interchangeable with similarly named
objects in every machine-learning library.

**Action**  
A concrete, optionally executable response attached to a recommendation. An action names a Session
operation and parameters but does not run it from a report.

**Active fit**  
The `FitResult` currently attached to a Session. `fit`, `compare_models`, and `load_model` can replace
it. Checkpoint loading does not restore a model fit.

**Automatic choice**  
A deterministic choice made by BuildML from observed inputs, such as scale-aware ingest planning.
Reports label its decision origin `automatic` and should expose the selection rule.

**Canonical dataset**  
The Dataset-owned tabular state used by Session operations. In the current release, sklearn-facing
materialization is Pandas even when another engine is configured.

**Checkpoint**  
A resumable directory containing data, roles/metadata, split membership, operation history, and an
integrity manifest. It is not a fitted-model artifact.

**Concept note**  
A reusable technical explanation linked by operation catalog entries. Concept notes hold shared
material such as leakage boundaries; they do not replace operation-specific guidance.

**Data mode**  
The policy describing how BuildML intends to handle dataset scale, including memory-oriented modes.
Changing the mode after ingestion records policy metadata; it does not retroactively unload an
already materialized frame.

**Decision origin**  
One of `automatic`, `recommended`, or `explicit`, identifying whether BuildML selected a choice,
suggested it without mutation, or received it from the caller.

**Engine**  
The tabular execution/interchange implementation: currently Pandas, Polars, or DuckDB. Polars and
DuckDB support is optional and lazily imported. DuckDB connections are owned by root Dataset
handles (`close_native` or `with dataset:` / `with session:`), not by each `get_engine('duckdb')`
call.

**Filter expression**  
A SQL-style boolean predicate passed to `Dataset.filter_expr` for Polars/DuckDB native pushdown.
Simple comparisons that should work on both engines can be built with
`buildml.data.portable_filter_expr(column, op, value)`.

**Evidence**  
A traceable observation, metric, statistical test, artifact, or configuration value used to support
a finding. Evidence includes its source and limitations where relevant.

**Feature contract**  
The names, order, representation, and meaning of columns expected by a fitted estimator. Encoding,
date extraction, dropping columns, and external transformations can change this contract.

**Finding**  
An interpretation supported by evidence. A finding has severity and may identify affected columns;
it is distinct from the recommendation that may follow.

**Fit-capable operation**  
An operation that learns from values, including imputing, encoding, scaling, resampling, estimator
training, calibrator fitting, and threshold selection. Its learned part belongs on training or
validation data according to the decision being made.

**Holdout**  
Rows excluded from estimator fitting. In BuildML this usually means validation or test, but the
purpose of the partition must still be stated.

**Ingest report**  
Structured output describing source detection, estimated scale, selected mode/engine, and loading
warnings. A dry run can produce this report without a materialized Dataset.

**Injected split**  
Train, validation, and test membership supplied as positional indices by the caller. Use it for
grouped, temporal, regulated, or externally governed partitions.

**Leakage**  
Information reaching model development that would not be available at the prediction time being
simulated. BuildML's train-fit guards prevent some partition leakage but cannot detect semantic
target proxies or misuse in external code.

**Manifest**  
The checkpoint file that records bundle members and integrity information. Removing the manifest or
moving only part of a checkpoint makes reliable reattachment impossible.

**Materialization gate**  
A soft or hard check at Pandas/sklearn design-matrix boundaries. Soft gates warn near ~250 MiB;
hard gates refuse when `hard_limit_bytes` or `BUILDML_MATERIALIZATION_HARD_LIMIT_BYTES` is set.
`prepare_design_matrix` projects requested columns (and may sample on Polars/DuckDB) before those
gates; it does not provide out-of-core sklearn training.

**Native sidecar**  
Optional Parquet snapshot written beside checkpoint `frame.parquet` so Polars/DuckDB handles can
reattach without an eager rebuild from the Pandas export. Defaults: compression `zstd`, layout
`auto` (partition at ≥50k rows), 25k rows per partition. Override with `sidecar_compression`,
`sidecar_layout`, and `sidecar_partition_rows` on `checkpoint_save`. Older single-file sidecars
remain readable. Engine query plans are not serialized.

**Nested cross-validation**  
An outer loop that scores configurations chosen by an inner CV search on each outer-train subset.
`Session.nested_cv_score` keeps Session test/validation out of both loops. Inner means are selection
evidence; outer mean±std is the post-selection estimate. Optional `warm_start_studies=True` shares
Optuna trial history across outer folds only; Teaching Studio / walkthrough surfaces that policy when
present in history.

**Pipeline bundle format**  
Directory layout for fitted plans plus an estimator. Current labels are
`buildml.pipeline_bundle.v2` (meta) and `buildml.plans.v2` (`plans.joblib`), with a read path for
older flat plan dicts.

**Preprocess recipe**  
An unfitted fold-local preprocess specification (`PreprocessRecipe`) refit on each CV fold's
training rows. Supported fold-local steps include dates, text features, outliers, impute, encode,
binning, scale, PCA (`reduce`), and feature selection. Resample, registered custom transforms, and
any Session plan fitted on the full train partition before CV (when not expressed in the recipe)
remain Session-global concerns.

**Dataset project / aggregate**  
`Dataset.project` keeps a column subset; `Dataset.aggregate` computes grouped or global summaries
(`sum`, `mean`, `min`, `max`, `count`, `n_unique`, `std`, `median`, and integer percentiles
`q0`..`q100`). Both prefer attached Polars/DuckDB native ops before Pandas. Quantiles use
continuous/linear interpolation; pass `materialize=True` for Pandas-only semantics when
cross-engine tie behavior matters. They are tabular prep helpers, not fold-local modeling
transforms.

**Operation catalog**  
The Python registry of explanation specifications for every public callable Session operation.
Catalog entries document mechanics, ordering, risks, alternatives, state changes, and result use.

**Operation history**  
A list of Session calls and selected details. It supports audit and checkpoint resumption but is not
complete source-data provenance and does not prove that choices were valid.

**Model bundle**  
A separately persisted fitted estimator and its recorded feature contract. It does not contain the
Session dataset, partitions, or complete preprocessing workflow. Treat pickle-compatible model
bundles as trusted-input artifacts.

**Pipeline bundle**  
A directory that stores a fitted estimator together with Session preprocess plans (impute, encode,
scale, dates, outliers, binning, feature selection, and resample lineage) and a model card
(schema, metrics, history summary, lineage). It is not a checkpoint: it does not embed the dataset
or split membership. Safe to store beside a checkpoint; neither artifact contains the other.

**Model card**  
A JSON/Markdown summary written with a pipeline bundle. It records task, feature contract,
optional partition metrics, which preprocess plans are present, compact history, and lineage notes.
It is not a complete provenance proof.

**Partition**  
A named set of row positions: `train`, optional `validation`, or `test`. Session partition access
returns a copy.

**Recommendation**  
Advice supported by findings or evidence. It includes rationale, priority, and caveats and does not
change Session state.

**Reattach**  
Loading checkpoint state and validating that its data and metadata remain compatible. A `data_only`
reattach deliberately discards prior workflow semantics.

**Result reading**  
Catalog guidance on how to interpret an operation's output, including context and limitations. It is
not an automatic pass/fail verdict.

**Role**  
The semantic use assigned to a column, such as feature, target, identifier, or ignored. Role is not
inferred safely from dtype alone.

**Session**  
The thin public facade that owns workflow state and delegates computation to BuildML domain
packages. It should not duplicate analyzer, transform, estimator, or report-rendering logic.

**Split plan**  
The stored partition memberships and split metadata. Row-preserving transforms retain it; operations
that replace or resample rows must update it deliberately.

**Self-contained report**  
An HTML artifact whose required CSS, JavaScript, and small assets are embedded so it remains readable
without network access. The term describes packaging, not methodological completeness.

**Test partition**  
Rows reserved for estimating performance after feature, model, hyperparameter, and threshold
choices are fixed. Repeatedly consulting test results makes them selection data.

**Train-fitted**  
Learned exclusively from training rows and then applied with frozen parameters to other partitions.
Examples include imputation values, category vocabularies, scaling statistics, and model parameters.

**Validation partition**  
Rows used to compare models, tune thresholds, or make other iterative decisions without fitting the
estimator parameters on those rows.

**Workflow step status**  
One of `available`, `done`, `blocked`, or `skipped`. `available` means current API prerequisites
pass, not that the operation is recommended. A blocked or skipped step should include its reason.

