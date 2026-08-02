Concept guide
=============

The canonical short notes live in ``buildml.explain.CONCEPT_NOTES`` and are
returned by ``buildml.explain.get_concept(key)``. Session operation entries
link to those keys. This guide expands the choices that require project
judgment.

Roles and feature contracts
---------------------------

A role states how a column may be used. Dtype is not enough: an integer may be
a measurement, category, identifier, group key, or timestamp surrogate.
Review target, feature, identifier, group, time, weight, and ignored roles
before target-aware analysis.

Do not proceed when a feature would be unavailable at prediction time, was
created after the outcome, or is a direct proxy for the target. BuildML
validates role names and uses roles to select target/features. It cannot infer
the real-world availability or meaning of a field.

Canonical catalog keys: ``column-roles`` and ``feature-schema``.

Leakage and partitions
----------------------

Leakage occurs when development receives information unavailable at the
prediction point being simulated. BuildML's ``impute``, ``encode``, ``scale``,
``resample``, and ``fit`` require a split. Replacement statistics,
vocabularies, scale parameters, synthetic samples, and estimator parameters
are learned from training rows.

``split`` supports random and stratified membership. Random splitting assumes
independent, exchangeable rows. It is misleading for repeated customers,
households, devices, locations, matched records, or future prediction when
related rows or periods cross the boundary. In those cases, stop and design
the boundary outside BuildML, then call ``inject_split``. BuildML checks
indices for overlap, duplicates, range, and stored membership; it cannot prove
that groups or time windows were defined correctly.

Use validation for model, feature, hyperparameter, calibration, and threshold
choices. Use test once those choices are fixed. Repeatedly reading test results
turns test into selection data.

Canonical catalog keys: ``leakage-boundary``, ``data-splitting``, and
``evaluation-partitions``.

EDA interpretation
------------------

``Session.eda`` can report quality issues, distributions, associations,
outliers, target relationships, multivariate screens, and partition drift.
These are prompts for investigation:

* correlation and mutual information do not establish causation;
* a statistical flag can be negligible in effect size or unstable in a small
  sample;
* outliers may be valid rare cases rather than errors;
* full-data or test-aware exploration can leak choices into evaluation;
* sampled EDA can miss rare categories and tails;
* drift identifies changed distributions, not the resulting change in model
  quality.

Do not proceed to model claims when the observation unit, target timing,
duplicate policy, missingness mechanism, or partition design remains
unresolved. BuildML records findings, evidence, recommendations, and
limitations separately. Recommendations never mutate Session state.

Canonical catalog keys: ``diagnostic-uncertainty`` and ``dataset-drift``.

Preprocessing order
-------------------

Split first. A common order is impute, encode, then scale, but the estimator
and data semantics determine whether each stage belongs:

* skip imputation when the estimator handles missing values and that behavior
  is understood;
* use one-hot encoding for unordered low-cardinality categories; ordinal
  encoding invents numeric order unless the category is truly ordered;
* scaling matters for distance, margin, and regularized linear methods, but
  usually not for tree split ordering;
* resampling changes training prevalence and must not change validation or
  test rows;
* date parts can expose future or post-outcome information even though their
  calculation is deterministic.

Do not continue after a schema-changing operation until generated columns,
unknown-category behavior, null counts, and the estimator feature contract
have been checked. BuildML stores fitted plans and preserves frozen
train-derived parameters across partitions. It does not package all
preprocessing into the saved estimator bundle for deployment.

Canonical catalog keys: ``missing-data``, ``categorical-encoding``,
``feature-scaling``, ``class-imbalance``, and ``feature-schema``.

Baselines, model fit, and selection
-----------------------------------

A baseline anchors whether model complexity improves the chosen metric. For
classification, compare against prevalence or a simple policy. For regression,
compare against a train-derived central prediction. Evaluate candidates under
the same preparation, partitions, and metric.

``fit`` clones and trains one sklearn-compatible estimator on training rows.
``compare_models`` trains named candidates and ranks on one partition. Prefer
validation for ranking; its current default partition is test, so override
that default during iterative selection. The top-ranked candidate becomes the
active fit.

Do not proceed from a rounded score difference alone. Check variation,
failure slices, complexity, latency, calibration, and whether the gain exceeds
the baseline by enough to matter.

Canonical catalog keys: ``baselines``, ``model-selection``, and
``overfitting``.

Evaluation, calibration, thresholds, and importance
---------------------------------------------------

Always read a metric with its partition, sample count, positive class or
target unit, and baseline. Accuracy can hide minority-class failure. ROC and
precision-recall answer different questions under class imbalance. Regression
averages can hide asymmetric or subgroup errors.

Calibration asks whether predicted probabilities match observed frequencies;
it is separate from ranking quality. Fit calibrators on validation or
cross-validation data and assess them elsewhere. Threshold selection is a
decision-policy choice tied to false-positive and false-negative costs.
``tune_threshold`` reports a sweep (and optional expected-cost minimization
via ``fp_cost`` / ``fn_cost``) but does not change estimator prediction
behavior; choose on validation and confirm the fixed threshold on test.
``error_slices`` localizes holdout errors by one or more segment columns and
keeps small-n segments out of the primary ranking.

Permutation importance measures score change when a feature is shuffled. It
can split reliance among correlated features and is unstable on small
partitions. It does not measure causal effect or universal relevance.

Do not make a release claim when the relevant partition is too small, the
positive class has inadequate support, the deployment prevalence differs
materially, or the test partition influenced prior choices.

Canonical catalog keys: ``probability-calibration``, ``thresholds``,
``feature-importance``, and ``diagnostic-uncertainty``.

Drift
-----

Drift compares defined populations or periods. Interpret effect size, sample
support, missingness, and collection changes together. Train-test drift can
mean an invalid split, temporal change, or a different population. Feature
drift without labels does not measure model-quality drift.

Stop automated comparison when schemas, units, category meanings, or
observation definitions differ. BuildML's EDA drift analyzer compares stored
partitions; it cannot establish that they represent production windows.

Canonical catalog key: ``dataset-drift``.

Checkpoints and reproducibility
-------------------------------

A checkpoint stores canonical data, roles, split membership, operation
history, metadata, and ``MANIFEST.json`` hashes. ``checkpoint_load`` validates
the bundle. ``data_only=True`` deliberately discards prior workflow semantics.

A checkpoint is not a model artifact. ``save_model`` stores the active fitted
estimator and feature contract separately. Do not load an untrusted model
bundle because its serialization is pickle-compatible. Do not resume when
reattach validation reports incompatible or missing required state.

History records calls made through Session; it is not complete source-data
provenance and does not prove that methodological choices were valid.

Canonical catalog keys: ``checkpoint-integrity`` and ``reproducibility``.

Teaching surfaces: explain, workflow, walkthrough, dry_run
----------------------------------------------------------

BuildML maintains a versioned **operation catalog** for every public Session
callable. Each entry covers definition, purpose, pipeline role, mechanism,
parameters, prerequisites, usual ordering, alternatives, assumptions, failure
modes, leakage risks, state changes, and result reading. Shared background
lives in **concept notes** (``buildml.explain.CONCEPT_NOTES``), linked from
catalog entries by key.

``Session.explain(operation, moment="before"|"after")`` joins catalog text to
live Session state. A ``before`` explanation lists what must already be true
and what could go wrong. An ``after`` explanation adds the latest recorded
call, parameters, and state transition. Explanations report what BuildML
knows; they cannot prove that a partition matches deployment or that roles
exclude target proxies.

``Session.workflow()`` resolves every cataloged operation to one of:

* ``done`` — recorded in history or satisfied by current state;
* ``available`` — prerequisites pass (not a recommendation to run);
* ``blocked`` — prerequisites fail, with a reason;
* ``skipped`` — not applicable given current task or configuration.

``Session.walkthrough()`` combines workflow resolution, operation history,
unresolved catalog risks, and optional offline HTML export. It is the audit
view for handoff or self-review after a long session.

``Session.dry_run(...)`` previews one or more operations without mutating
state or appending history. ``Session.summarize_history()`` counts operations,
surfaces heuristic unresolved risks, and lists suggested next steps from the
prerequisite graph. Risks are review cues, not proof of invalid results.

``Session.eda()`` and ``session.eda_app()`` add findings (observations with
severity), evidence, and read-only recommendations. A **finding** states what
was observed, on which partition, with what measure, and with stated limits. A
**recommendation** proposes a response but does not mutate the Session.

Canonical catalog keys: ``operation-catalog``, ``workflow-resolution``, and
``diagnostic-uncertainty``.

Engines at a practical level
----------------------------

Three engines appear in current APIs: **Pandas** (default canonical frame),
**Polars**, and **DuckDB**. Path ingest with ``engine="polars"`` or
``engine="duckdb"`` loads natively when the extra is installed. Session
preprocess steps still materialize through Pandas for sklearn; native handles
are rebuilt after transforms so ``Dataset.project``, ``Dataset.aggregate``,
and ``prepare_design_matrix`` can prefer engine ops where implemented.

Practical guidance:

* Stay on Pandas for small and medium frames and the simplest mental model.
* Use Polars or DuckDB when filtering, projecting, or aggregating large files
  before sklearn materialization.
* Use ``portable_filter_expr`` for simple predicates shared across Polars and
  DuckDB; keep complex SQL engine-specific.
* Close DuckDB with ``with session:`` or ``session.close_native()`` — root
  datasets own the connection.
* Lazy Polars ``LazyFrame`` plans collect at sklearn boundaries; that is not
  out-of-core training.

Checkpoint sidecars optionally store Parquet snapshots so restore can reattach
a native handle without eager rebuild from the Pandas export. Sidecar layout,
compression, and row thresholds are configurable on ``checkpoint_save``.

Canonical catalog keys: ``data-engines`` and ``materialization-gates``.

Imbalance and resampling
------------------------

Class imbalance affects which metrics matter and whether resampling helps.
``Session.resample`` alters **training rows only** after a split. Validation
and test partitions stay untouched. Resampling changes training prevalence; compare
against a non-resampled baseline on the same partitions before claiming gain.

``resample_strategies()`` lists available samplers and when each is reasonable.
Resample plans are recorded for lineage and appear in pipeline bundles, but
they are not reapplied automatically at score time.

Canonical catalog key: ``class-imbalance``.
