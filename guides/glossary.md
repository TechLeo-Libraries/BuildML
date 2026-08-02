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

**DataLoader (Torch)**  
A batched iterator over partition tensors built by `Session.make_torch_loaders`. Shuffle applies to
the train loader only. Validation and test loaders stay unshuffled for evaluation honesty.

**DeviceSpec**  
Resolved compute device for Torch training (`cpu`, `cuda`, or `mps`) plus any fallback warning when
the requested device was unavailable.

**Decision origin**  
One of `automatic`, `recommended`, or `explicit`, identifying whether BuildML selected a choice,
suggested it without mutation, or received it from the caller.

**dl_train_result**  
The Session slot holding the last Torch `TrainResult`. Distinct from classical `fit_result`. Cleared
only by a new Torch fit/load path; classical `fit` does not overwrite it.

**Early stopping (Torch)**  
Optional patience on a validation monitor (default `val_loss`) during `fit_torch`. Selecting the
epoch on the test partition turns test into selection data; use validation for stopping and test
once the recipe is fixed.

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
date extraction, dropping columns, and external transformations can change this contract. On the
Torch path, the contract also records task, optional class labels, and train-fit normalize
mean/std carried in the trainer bundle.

**Finding**  
An interpretation supported by evidence. A finding has severity and may identify affected columns;
it is distinct from the recommendation that may follow.

**Fit-capable operation**  
An operation that learns from values, including imputing, encoding, scaling, resampling, estimator
training, calibrator fitting, and threshold selection. Its learned part belongs on training or
validation data according to the decision being made.

**Hashing embedder**  
Default RAG embedder id `buildml.hashing_embed.v1`: sklearn `HashingVectorizer`
(`n_features=384`, L2-normalized). Deterministic and CPU-only; lexical/hashed,
not a semantic sentence model. Disclosures and catalog copy must say so.

**Hit-rate@k**  
Fraction of evaluation queries for which at least one relevant document (or
chunk, under chunk relevance mode) appears in the top-k retrieved results.

**Holdout**  
Rows excluded from estimator fitting. In BuildML this usually means validation or test, but the
purpose of the partition must still be stated.

**Ingest report**  
Structured output describing source detection, estimated scale, selected mode/engine, and loading
warnings. A dry run can produce this report without a materialized Dataset.

**Hybrid retrieve**  
Retrieval mode that blends dense vector ranking with lexical BM25 over chunk
text. Default fusion is reciprocal rank fusion (RRF, `rrf_k=60`); weighted
fusion is optional via retrieve config.

**IndexResult**  
Typed summary of a built RAG index: chunk/document counts, embedder id and
dimension, store backend, and disclosures. Stored on `session.rag_index_result`.

**Injected split**  
Train, validation, and test membership supplied as positional indices by the caller. Use it for
grouped, temporal, regulated, or externally governed partitions.

**Leakage**  
Information reaching model development that would not be available at the prediction time being
simulated. BuildML's train-fit guards prevent some partition leakage but cannot detect semantic
target proxies or misuse in external code.

**MRR (mean reciprocal rank)**  
Mean, over evaluation queries, of `1 / rank` of the first relevant hit (0 when
no relevant hit appears). Reported by `Session.rag_evaluate`.

**Manifest**  
The checkpoint file that records bundle members and integrity information. Removing the manifest or
moving only part of a checkpoint makes reliable reattachment impossible.

**Materialization gate**  
A soft or hard check at Pandas/sklearn design-matrix boundaries. Soft gates warn near ~250 MiB;
hard gates refuse when `hard_limit_bytes` or `BUILDML_MATERIALIZATION_HARD_LIMIT_BYTES` is set.
`prepare_design_matrix` projects requested columns (and may sample on Polars/DuckDB) before those
gates; it does not provide out-of-core sklearn training.

**nDCG@k**  
Normalized discounted cumulative gain at cutoff k for retrieval evaluation.
Uses graded or binary relevance from gold qrels; reported alongside recall@k
and MRR by `Session.rag_evaluate`.

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
Optuna trial history across outer folds only; the EDA dashboard and walkthrough
surfaces that policy when present in history.

**Qrels**  
Gold relevance judgments for retrieval evaluation: query → relevant `doc_id`
(document mode) or `chunk_id` (chunk mode) labels. Index corpus and qrel/query
sets must stay separate to avoid evaluation contamination.

**Pipeline bundle format**  
Directory layout for fitted plans plus an estimator. Current labels are
`buildml.pipeline_bundle.v2` (meta) and `buildml.plans.v2` (`plans.joblib`), with a read path for
older flat plan dicts.

**Preprocess recipe**  
An unfitted fold-local preprocess specification (`PreprocessRecipe`) refit on each CV fold's
training rows. Supported fold-local steps include dates, text features, outliers, impute, encode,
binning, scale, PCA (`reduce`), and feature selection. Resample and registered custom transforms
remain Session-global only. If Session fit-capable plans were already fitted on the full train
partition, CV/search refuse even when a fold-local recipe is passed — recipes run on the
already-transformed frame and do not rebuild from raw rows. Re-ingest unpoisoned data, or set
`allow_session_global_preprocess=True` as an explicit override (scores remain leakage-biased).

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

**Normalize (Torch loaders)**  
Optional train-fit feature mean/std computed in `make_torch_loaders` when `normalize=True`. Stats
are frozen on validation and test. This is not batch-norm inside the module and is not classical
`Session.scale`.

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

**RAG bundle**  
Directory schema `buildml.rag_bundle.v1` (`meta.json` + `chunks.jsonl` +
`embeddings.npy`) holding chunk config, embedder id/dim, embeddings, chunk
metadata, and optional eval snapshot. It is not a Session checkpoint and does
not embed dataset rows or Torch weights.

**rag_eval_result / rag_index_result / rag_retrieve_result**  
Session slots for the last RAG evaluate, index, and retrieve typed results.
Distinct from classical `fit_result` and Torch `dl_train_result`.

**RagEvalResult**  
Typed output of `rag_evaluate`: recall@k, MRR, nDCG@k, hit-rate@k, relevance
mode, retrieve mode, per-query rows, and disclosures/warnings.

**Recall@k**  
Fraction of relevant labels recovered in the top-k hits for a query (averaged
over the eval set). It is not classification accuracy.

**RetrieveResult**  
Typed ranked-hit list from `rag_retrieve`: mode (`dense` / `bm25` / `hybrid`),
fusion/rerank flags, filters, scores, and disclosures.

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

**Torch trainer bundle**  
Directory schema `buildml.torch_bundle.v1` (`meta.json` + `trainer.pt`) holding module weights,
optimizer (and optional scheduler) state, `TrainConfig`, epoch history, early-stop bookkeeping, and
the feature/label contract. It is not a Session checkpoint and does not embed dataset rows or split
indices.

**TrainConfig**  
Typed epoch-loop knobs for `fit_torch` (epochs, learning rate, device, grad clip, scheduler,
early-stopping patience/monitor). Defaults are documented on `buildml.dl.types.TrainConfig`.

**TrainingCurveReport**  
Structured per-epoch series plus interpretation, limitations, and disclosures (device, early-stop
partition, scheduler) from `Session.torch_training_curve`. It is teaching data, not a pass/fail
verdict.

**TrainResult**  
Typed output of `fit_torch`: module, config, device, history, optional early-stop record, and
feature contract. Stored on `session.dl_train_result`.

**Train-fitted**  
Learned exclusively from training rows and then applied with frozen parameters to other partitions.
Examples include imputation values, category vocabularies, scaling statistics, and model parameters.

**Validation partition**  
Rows used to compare models, tune thresholds, or make other iterative decisions without fitting the
estimator parameters on those rows.

**Workflow step status**  
One of `available`, `done`, `blocked`, or `skipped`. `available` means current API prerequisites
pass, not that the operation is recommended. A blocked or skipped step should include its reason.

