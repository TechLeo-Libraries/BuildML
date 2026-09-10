Concept guide
=============

A few ideas decide whether a BuildML run is trustworthy. This page is
those ideas. The short in-library notes live in
``buildml.explain.CONCEPT_NOTES`` and come back through
``session.learn("leakage")`` (or whatever word tripped you up). Use that
when you are already in a Session. Use this page when you want the
judgment written out.

A Session is a unified, stateful ML lifecycle. Enforced leakage
safeguards, fold-local preprocessing, contextual teaching, workflow
guidance, checkpointing, and auditable export live on that same object.

Roles
-----

A role says how a column may be used. Dtype is not enough. An integer
can be a measurement, a category, an identifier, a group key, or a
stand-in for time. Columns you do not name default to ``feature``.
Only the names you pass to ``set_roles`` change.

Review target, feature, identifier, group, time, weight, and ignored
roles before you do anything target-aware. Supervised ``fit`` wants
exactly one ``target``.

Do not continue when a feature would be unavailable at prediction time,
was created after the outcome, or is a direct proxy for the target.
BuildML validates role names and uses them to pick target and features.
It cannot infer what a field means in the real world.

In a Session: ``session.learn("column-roles")``.

Leakage and partitions
----------------------

Leakage is when development sees information that would not exist at the
prediction point you are simulating. ``impute``, ``encode``, ``scale``,
``resample``, and ``fit`` require a split. Replacement statistics,
vocabularies, scale parameters, synthetic samples, and estimator
parameters are learned from training rows.

Random ``split`` assumes independent, exchangeable rows. That is the
wrong tool for repeated customers, households, devices, locations,
matched records, or predicting the future when related rows or periods
cross the boundary. In those cases, design the boundary yourself, then
call ``group_split``, ``time_split``, or ``inject_split``.
``group_split``'s sizes count groups, not rows.
``time_split`` holds out the most recent rows.
``inject_split`` takes positional indices (``0`` to ``n-1``), not
DataFrame labels, and refuses overlap. BuildML cannot prove that groups
or time windows were defined correctly.

Use validation for model, feature, hyperparameter, calibration, and
threshold choices. Use test once those choices are fixed. Reading test
results over and over turns test into selection data.

In a Session: ``session.learn("leakage")``.

EDA interpretation
------------------

``session.eda()`` can report quality issues, distributions, associations,
outliers, target relationships, multivariate screens, and partition
drift. Treat those as prompts, not conclusions:

* correlation and mutual information do not establish causation;
* a statistical flag can be tiny in effect or unstable in a small sample;
* outliers may be valid rare cases rather than errors;
* exploring the full table, or peeking at test, can leak choices into
  evaluation;
* sampled EDA can miss rare categories and tails;
* drift identifies changed distributions, not the resulting change in
  model quality.

Do not move to model claims while the observation unit, target timing,
duplicate policy, missingness mechanism, or partition design is still
unresolved. Findings, evidence, recommendations, and limitations are
recorded separately. Recommendations never mutate Session state.

In a Session: ``session.learn("diagnostic-uncertainty")``.

Preprocessing order
-------------------

Split first. A common order is impute, encode, then scale, but the
estimator and the data decide whether each stage belongs:

* skip imputation when the estimator handles missing values and you
  understand that behavior;
* use one-hot encoding for unordered low-cardinality categories;
  ordinal encoding invents numeric order unless the category is truly
  ordered;
* scaling matters for distance, margin, and regularized linear methods,
  and usually not for tree split ordering;
* resampling changes training prevalence and must not change validation
  or test rows;
* date parts can expose future or post-outcome information even though
  the calculation is deterministic.

After a schema-changing operation, check generated columns,
unknown-category behavior, null counts, and the estimator feature
contract before you continue. BuildML stores fitted plans and keeps
frozen train-derived parameters across partitions.

In a Session: ``session.learn("missing-data")``.

Baselines, fit, and selection
-----------------------------

A baseline tells you whether complexity improved the metric you chose.
For classification, compare against prevalence or a simple policy. For
regression, compare against a train-derived central prediction. Evaluate
candidates under the same preparation, partitions, and metric.

``fit`` clones and trains one sklearn-compatible estimator on training
rows. ``compare_models`` trains named candidates and ranks on one
partition. Prefer validation for ranking. The current default partition
for ``compare_models`` is test, so override that during iterative
selection. The top-ranked candidate becomes the active fit.

Do not proceed from a rounded score difference alone. Check variation,
failure slices, complexity, latency, calibration, and whether the gain
beats the baseline by enough to matter.

In a Session: ``session.learn("baselines")``.

Evaluation, calibration, thresholds, and importance
---------------------------------------------------

Read a metric with its partition, sample count, positive class or target
unit, and baseline. Accuracy can hide minority-class failure. ROC and
precision-recall answer different questions under imbalance. Regression
averages can hide asymmetric or subgroup errors.

Calibration asks whether predicted probabilities match observed
frequencies. That is separate from ranking quality. Fit calibrators on
validation or cross-validation data and assess them elsewhere.

Threshold selection is a decision-policy choice tied to false-positive
and false-negative costs. ``tune_threshold`` reports a sweep (and
optional expected-cost minimization via ``fp_cost`` / ``fn_cost``) but
does not change estimator prediction behavior. Choose on validation and
confirm the fixed threshold on test.

``error_slices`` localizes holdout errors by one or more segment
columns and keeps small-n segments out of the primary ranking.

Permutation importance measures score change when a feature is shuffled.
It can split reliance among correlated features and is unstable on small
partitions. It does not measure causal effect.

Do not make a release claim when the relevant partition is too small,
the positive class has inadequate support, deployment prevalence differs
materially, or the test partition already influenced prior choices.

In a Session: ``session.learn("probability-calibration")``.

Drift
-----

Drift compares defined populations or periods. Read effect size, sample
support, missingness, and collection changes together. Train-test drift
can mean an invalid split, temporal change, or a different population.
Feature drift without labels does not measure model-quality drift.

Stop automated comparison when schemas, units, category meanings, or
observation definitions differ. BuildML's EDA drift analyzer compares
stored partitions. It cannot establish that they represent production
windows.

In a Session: ``session.learn("dataset-drift")``.

Checkpoints and reproducibility
-------------------------------

A checkpoint stores canonical data, roles, split membership, operation
history, metadata, and ``MANIFEST.json`` hashes. ``checkpoint_load``
validates the bundle. ``data_only=True`` discards prior workflow
semantics on purpose.

A checkpoint is not a model artifact. ``save_model`` stores the active
fitted estimator and feature contract separately. Do not load an
untrusted model bundle; its serialization is pickle-compatible. Do not
resume when reattach validation reports incompatible or missing required
state.

History records calls made through Session. It is not complete
source-data provenance and does not prove that methodological choices
were valid.

In a Session: ``session.learn("checkpoint-integrity")``.

Teaching surfaces
-----------------

BuildML keeps a versioned operation catalog for every public Session
callable. Each entry covers definition, purpose, pipeline role,
mechanism, parameters, prerequisites, usual ordering, alternatives,
assumptions, failure modes, leakage risks, state changes, and how to
read the result.

``session.explain(operation, moment="before"|"after")`` joins that
catalog to live Session state. A ``before`` explanation lists what must
already be true and what could go wrong. An ``after`` explanation adds
the latest recorded call. Explanations report what BuildML knows. They
cannot prove that a partition matches deployment or that roles exclude
target proxies.

``explain`` and ``learn`` both accept ``level="beginner"`` (the
default), ``"intermediate"``, or ``"advanced"``. The level changes how
much scaffolding you see, never which facts are true. Assumptions,
leakage risks, and failure modes appear at every level.

``session.learn(topic)`` answers the question that comes before
``explain``: what is this, and what should I understand first. The topic
may be a concept key (``"leakage-boundary"``), an operation name
(``"split"``), or a piece of jargon (``"stratified"``). Spacing and
hyphenation are forgiven. Called with no topic it returns the foundation
concepts in reading order.

``session.workflow()`` resolves every cataloged operation to one of
``done``, ``available``, ``blocked``, or ``skipped``. Available means
prerequisites pass, not that you should run the step.

``session.walkthrough()`` joins workflow status, history, and unresolved
catalog risks, and can export offline HTML.

``session.dry_run(...)`` previews operations without mutating state.
``session.summarize_history()`` counts operations and surfaces heuristic
unresolved risks. Those risks are review cues, not proof of invalid
results.

Engines
-------

Three engines appear in current APIs: **Pandas** (the default canonical
frame), **Polars**, and **DuckDB**. Path ingest with
``engine="polars"`` or ``engine="duckdb"`` loads natively when the extra
is installed. Session preprocess still materializes through Pandas for
sklearn.

Practical guidance:

* Stay on Pandas for small and medium frames and the simplest mental
  model.
* Use Polars or DuckDB when filtering, projecting, or aggregating large
  files before sklearn materialization.
* Use ``portable_filter_expr`` for simple predicates shared across
  Polars and DuckDB. Keep complex SQL engine-specific.
* Close DuckDB with ``with session:`` or ``session.close_native()``.
* Lazy Polars ``LazyFrame`` plans collect at sklearn boundaries. That is
  not out-of-core training.

In a Session: ``session.learn("data-engines")``.

Imbalance and resampling
------------------------

Class imbalance affects which metrics matter and whether resampling
helps. ``session.resample`` alters **training rows only** after a split.
Validation and test stay untouched. Resampling changes training
prevalence. Compare against a non-resampled baseline on the same
partitions before you claim a gain.

``resample_strategies()`` lists available samplers and when each is
reasonable. Resample plans are recorded for lineage and appear in
pipeline bundles, but they are not reapplied automatically at score
time.

In a Session: ``session.learn("class-imbalance")``.
