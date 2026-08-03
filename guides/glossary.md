# BuildML glossary

Terms here describe the current BuildML 2.x API. They are not interchangeable with similarly named
objects in every machine-learning library.

For general machine-learning vocabulary rather than BuildML's own objects — leakage, stratification,
calibration, ROC-AUC — call `session.learn("<term>")`, which returns a plain-language definition plus
the concept note that teaches it and what to read first.

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

**Case base (CBR)**  
Train-built tabular memory of cases (features + solution/label/outcome) used by
`fit_cbr` / `retrieve_cases` / `predict_cbr`. Validation and test rows never enter
the memory at fit time. Distinct from a RAG text corpus.

**CaseTrace**  
Per-query explanation from CBR retrieve/predict: neighbor case ids, distances,
weights, neighbor solutions, and the reused prediction.

**CBR bundle**  
Directory schema `buildml.cbr_bundle.v1` (`meta.json` + `cbr_plan.joblib`) holding
a `CbrPlan` (case memory + metric/reuse config). It is not a Session checkpoint
and is not interchangeable with `buildml.rag_bundle.v1`.

**Behavioral cloning (imitation)**  
Supervised state→action policy fitted by `fit_imitation` on Session train
demonstration rows only. Holdout `evaluate_imitation` compares predicted actions
to demonstration actions. Not inverse RL and not a robotics stack.

**Checkpoint**  
A resumable directory containing data, roles/metadata, split membership, operation history, and an
integrity manifest. It is not a fitted-model artifact.

**Concept note**  
A reusable explanation linked by operation catalog entries. Concept notes hold shared material such
as leakage boundaries; they do not replace operation-specific guidance. Each note is layered: a
plain-language summary, an analogy, beginner steps, misconceptions, a worked example, and the
technical sections, plus prerequisite and follow-on notes that give a reading order.

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

**evaluate_asr / WER / CER**  
`Session.evaluate_asr` (and `buildml.dl.speech.evaluate_asr`) scores ASR
hypotheses against references with word and character error rates via
Levenshtein edit distance. String metrics only — not a speech quality / MOS
product. Omitting `hypotheses=` reuses texts from the last `transcribe_speech`.

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

**Faithfulness (RAG)**  
Cheap grounding heuristic attached to `GenerateResult.faithfulness`
(`FaithfulnessReport`): citation-marker coverage plus answer↔context token
overlap. Not an NLI / LLM-as-judge product; high overlap does not prove
factual correctness. See `buildml.rag.generate.score_faithfulness`.

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

**Gated fusion (Torch multimodal)**  
Late-fusion mode (`fusion="gated"`) for `build_multimodal_fusion` that gates
modality branches before combining them. Default built-in `fit_torch` fusion
(when a module is omitted) remains concat; pass an explicit gated module to
opt in.

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

**multimodal_preprocess**  
Frozen multimodal fit meta (normalize stats, vocab, image/audio rates/layout)
optionally stored on Torch trainer bundles. `load_torch_bundle` restores it for
inspection; rebuild loaders with
`make_multimodal_torch_loaders(..., use_saved_preprocess=True)` or
`preprocess=`.

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

**Operation primer**  
The beginner-facing briefing attached to every operation explanation: plain summary, analogy, why it
exists, ordered steps, prerequisites in plain words, what each key parameter means in practice,
pitfalls, an in-line glossary, and a worked example. Derived from the catalog entry and its linked
concept notes, so it cannot drift from the expert sections it fronts. An operation may override any
section with hand-written prose.

**Learning level**  
`beginner` (default), `intermediate`, or `advanced`, accepted by `Session.explain` and
`Session.learn`. The level controls how much scaffolding is rendered — analogy, glossary, step
detail — never which facts are true. Assumptions, leakage risks, and failure modes are present at
every level.

**Learning brief**  
What `Session.learn` returns: the resolved subject (a concept note, an operation primer, or a
glossary term), plus `read_first` and `read_next` concept notes giving a reading order rather than an
index. Topic lookup accepts concept keys, operation names, and jargon, and tolerates spacing and
hyphenation differences.

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
change Session state. Distinct from **recommendation systems** (`fit_recommender` /
`recommend`), which rank catalog items from user–item interactions.

**Recommendation systems (Session)**  
`fit_recommender` learns from train user–item interactions (item/user kNN CF,
TruncatedSVD / NMF, or content profiles). `recommend` returns top-K train-catalog
items; `evaluate_recommender` scores Precision@K, Recall@K, nDCG@K, MAP@K under
a known-item protocol with cold-start disclosure. Not RAG; not EDA Recommendation
Findings; not a Netflix-scale platform.

**Recommender bundle**  
Directory schema `buildml.recommender_bundle.v1` (`meta.json` +
`recommender_plan.joblib`) holding a `RecommenderPlan`. Distinct from Session
checkpoints and from RAG / TDA bundles.

**Learning-to-rank / Search ranking (Session)**  
`fit_ranker` learns from train query–item (or query–document) feature rows
with relevance labels (pointwise Ridge/HGB or pairwise RankSVM-lite).
`rank` orders items per query; `evaluate_ranker` scores graded nDCG@K, MAP@K,
MRR@K. Prefer `group_split` on the query id. Not a search-engine product; not
RAG retrieve/generate; not recommender user–item CF.

**Ranker bundle**  
Directory schema `buildml.ranker_bundle.v1` (`meta.json` +
`ranker_plan.joblib`) holding a `RankerPlan`. Distinct from Session
checkpoints and from RAG / recommender bundles.

**Reinforcement learning (Session)**  
`fit_rl` covers contextual bandits on logged train tables (LinUCB / ε-greedy /
softmax) plus optional Gymnasium env loops behind `buildml[rl]`: tabular TD
control (`tabular_q` — Q-learning / SARSA / Expected SARSA / Double Q-learning)
and REINFORCE-lite (`gym_reinforce`); SB3 PPO/DQN/A2C behind
`buildml[rl-industry]`. Bandit holdout metrics are offline (DM/IPS); env-loop
metrics are online returns. Not a MuJoCo / robotics / multi-agent platform.

**Tabular TD control**  
Value-based RL that stores one action-value per (state, action) pair and
bootstraps: `Q(s,a) ← Q(s,a) + α[target − Q(s,a)]`. Q-learning uses the
off-policy target `r + γ max_a' Q(s',a')`; SARSA uses the on-policy
`r + γ Q(s',a')`. DQN is the same idea with a neural network replacing the
table. Continuous observations are discretized first; see
`RlPlan.config["discretizer"]`.

**RL bundle**  
Directory schema `buildml.rl_bundle.v1` (`meta.json` + `rl_plan.joblib`) holding
an `RlPlan`. Distinct from Session checkpoints and from imitation bundles.

**Imitation bundle**  
Directory schema `buildml.imitation_bundle.v1` (`meta.json` +
`imitation_plan.joblib`) holding an `ImitationPlan` (behavioral cloning policy).

**Topological Data Analysis (Session)**  
`fit_tda` builds local Vietoris–Rips persistence diagrams (ripser) on kNN train
neighborhoods, vectorizes them (persim images/landscapes or in-tree
silhouettes), and optionally fits a sklearn head — all on train only. Requires
`buildml[tda]`. Not a Mapper research suite.

**TDA bundle**  
Directory schema `buildml.tda_bundle.v1` (`meta.json` + `tda_plan.joblib`) holding
a `TdaPlan` (frozen PH vectorizer ± head). Distinct from Session checkpoints.

**Natural language processing (Session)**  
The `buildml.nlp` surface for one text column that lives on the Session dataset:
`profile_text_corpus`, `fit_text_classifier` → `predict_text` /
`evaluate_text_classifier` / `interpret_text_prediction`, plus `fit_topics` /
`assign_topics`, `extract_keyphrases`, `analyze_sentiment`, `extract_entities`,
`summarize_text`, and `detect_language`. Single-label document classification and
analysis — not multi-label, not span labelling, not generation, and not document
retrieval for generation (that is RAG).

**Text normalization plan**  
The deterministic, stateless part of a text pipeline — the normalization steps,
tokenizer settings, stopword list, and stemming or lemmatization choice — stored
on an `NlpTextPlan`. Because it learns nothing from the corpus it cannot leak, so
it replays freely on holdout rows. The vocabulary, document frequencies, and IDF
weights beside it are train-only.

**Token attribution (NLP)**  
`interpret_text_prediction` output: per token, the model's coefficient, the
token's value in this document, and their product. For a linear head on an
invertible vocabulary the products plus the intercept reconstruct the decision
score exactly, so this is an identity rather than an approximation. Refused for
hashing (no invertible vocabulary) and for dense backends (features are latent
dimensions).

**NPMI coherence (NLP topics)**  
Normalized pointwise mutual information over a topic's top terms, computed on the
train partition and bounded in [-1, 1]. The usual proxy for "are these topics
real" and the usual way to choose `n_topics`. Reconstruction error always falls as
topics are added, so it cannot serve the same purpose.

**Corpus contamination screen**  
The part of `profile_text_corpus` that counts holdout documents which are exact
duplicates of a train document, and those above a stated cosine similarity
threshold on character n-grams. It reports what it finds; it never silently drops
rows.

**NLP bundle**  
Directory schema `buildml.nlp_bundle.v1` (`meta.json` + `nlp_text_plan.joblib`
± `nlp_topic_plan.joblib`) holding the normalization plan, the train-fitted
representation, and the fitted head. Because the normalization plan travels with
the representation, a reload reproduces a holdout score exactly. Not a Session
checkpoint, not a `buildml.rag_bundle.v1`, and not a Torch trainer bundle.

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
optimizer (and optional scheduler) state, `TrainConfig`, epoch history, early-stop bookkeeping,
the feature/label contract, and optional `multimodal_preprocess` meta (frozen image/audio stats,
sample rates, layout). Load restores that meta for inspection but does not rebuild DataLoaders.
It is not a Session checkpoint and does not embed dataset rows or split indices.

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

