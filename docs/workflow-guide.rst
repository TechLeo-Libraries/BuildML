Workflow guide
==============

This sequence is a decision framework, not a requirement to call every
method. At each boundary, stop when the assumptions do not match the project.

1. Ingest without forcing scale
-------------------------------

Use ``Session.ingest`` with a DataFrame or supported path. Read
``session.ingest_report`` for detected format, estimated size, recommended
mode and engine, and warnings. Use ``dry_run=True`` or ``read_nrows`` before a
large load.

Do not force memory mode merely to bypass a warning. In the current release,
changing mode after ingestion records policy metadata and does not move an
already materialized frame out of memory.

2. Assign and review roles
--------------------------

Call ``set_roles`` from semantic knowledge. Confirm one intended target and
exclude identifiers, post-outcome fields, and unavailable-at-scoring fields
from features.

.. code-block:: python

   session.set_roles(
       {
           "customer_id": "id",
           "event_time": "time",
           "amount": "feature",
           "outcome": "target",
       }
   )

Stop if the prediction time, observation unit, or target definition is
ambiguous.

3. Define partitions
--------------------

Use random or stratified ``split`` only for exchangeable rows. Add validation
when choices will be repeated. For groups, time, or externally governed
membership, call ``inject_split`` with positional indices.

Check counts, class support, chronology, group isolation, and duplicates.
BuildML enforces disjoint memberships and train-only fit scope; it cannot
detect related entities or future leakage from values alone.

4. Explore without mining test
------------------------------

Run EDA after roles are assigned. Prefer training or development evidence for
choices and reserve final test interpretation. A full-dataset report is
descriptive; it is not automatically valid model-selection evidence.

.. code-block:: python

   report = session.eda(
       include_plots=True,
       export_html="artifacts/eda.html",
   )

Read each finding's evidence, partition, denominator, severity, and limit.
Investigate collection errors before treating a statistical association as a
feature opportunity.

5. Prepare in train-fitted order
--------------------------------

Choose only operations required by the estimator and data:

.. code-block:: python

   session.impute(strategy="median")
   session.encode(method="onehot")
   session.scale(method="standard")

BuildML learns these plans on train and applies frozen parameters to other
partitions. Review generated columns and stored plans. If external preparation
uses ``to_pandas()``, manually select ``session.partition("train")`` for any
fit-capable work; the full exported frame contains every partition.

For selection-time honesty inside ``cv_score``, ``grid_search``,
``optuna_search``, ``evolutionary_search``, or ``nested_cv_score``, use a fold-local
``PreprocessRecipe`` (dates, text, impute, encode, binning, scale, reduce/PCA,
select, outliers) on **unpoisoned** data — do not fit Session-global plans on
the full train partition first. When Session-global fit-capable plans already
exist, CV/search **refuse** with ``LeakageError`` even if a fold-local recipe
is passed (recipes do not rebuild from raw/unpoisoned rows). Opt in only via
``allow_session_global_preprocess=True`` (explicit, loud override; scores
remain leakage-biased), or re-ingest / checkpoint-load unpoisoned data.
Resample and ``apply_custom_transform`` remain Session-global only.
``Session.text_features`` / ``Session.reduce_dimensions`` are also
Session-global unless the same steps are expressed in the recipe on
unpoisoned data.

When Polars or DuckDB is configured via ``with_engine`` or path ingest,
``Dataset`` attaches a native handle. Path ingest with those engines loads
natively (no Pandas-first pass). With Polars and ``mode='lazy'``,
``Dataset.native`` may be a ``LazyFrame``; collect happens on
``to_pandas()`` / sklearn materialization. That is not out-of-core sklearn
training. Session preprocess transforms rebuild ``Dataset.native`` after
Pandas work so later ``project`` / ``prepare_design_matrix`` calls keep using
engine ops. Checkpoint load rebuilds an eager native handle from the Parquet
payload when engine metadata allows (LazyFrame plans are not persisted).

When hyperparameters or fold-local recipe knobs such as ``select_k`` /
``n_bins`` are tuned, pass ``recipe_grid`` / ``recipe_distributions`` /
``recipe_space`` into ``nested_cv_score``, ``grid_search``,
``optuna_search``, or ``evolutionary_search`` with a ``PreprocessRecipe``.
For nested Optuna, use
``inner_search='optuna'`` with ``param_space`` / ``recipe_space``. Outer
folds record ``best_params`` / ``best_recipe_knobs`` without using Session
test rows. Optuna requires ``pip install 'buildml[optuna]'``.

``save_pipeline`` writes ``schema_contract.json``. ``predict_from_pipeline``
coerces compatible dtypes when safe, then validates role-aware required
columns and dtype families (missing/wrong-type columns raise clear errors).
Older bundles without a contract remain loadable.

6. Establish a baseline and fit
-------------------------------

Fit a simple candidate before adding complexity. ``compare_models`` can apply
one prepared representation to multiple estimators, but candidates that need
different preprocessing should be compared in separately controlled
workflows.

.. code-block:: python

   from sklearn.dummy import DummyClassifier
   from sklearn.linear_model import LogisticRegression

   comparison = session.compare_models(
       {
           "prevalence": DummyClassifier(strategy="prior"),
           "logistic": LogisticRegression(max_iter=500),
       },
       task="classification",
       partition="validation",
       ranking_metric="f1",
   )

The ranking metric must reflect error costs. Stop when the partition is too
small, a candidate has incompatible inputs, or the score difference is within
observed variation.

When hyperparameters or selection knobs are tuned with CV, use
``nested_cv_score`` for the post-selection estimate. Do not treat
``grid_search`` inner means as untouched generalization claims.
Optuna inner search may set ``warm_start_studies=True`` to share study
priors across outer folds; this remains opt-in and still never scores
Session test or validation rows.

7. Evaluate the fixed choice
----------------------------

Use validation during iteration and test after model and feature choices are
fixed.

.. code-block:: python

   result = session.evaluate(
       partition="test",
       include_plots=True,
       export_html="artifacts/evaluation.html",
   )

Inspect task baselines, confusion or residual structure, class prevalence,
sample count, and skipped diagnostics. A self-contained report preserves the
rendered evidence, not the correctness of the evaluation design.

8. Diagnose decisions
---------------------

For probability decisions, inspect calibration before selecting a threshold.
Select thresholds on validation and assess the fixed policy on test.
Use permutation importance as a model-reliance audit, not as a causal or
automatic feature-selection rule. Use learning curves only when their
cross-validation folds respect row dependencies.

9. Explain and hand off
-----------------------

.. code-block:: python

   status = session.workflow()
   before = session.explain("checkpoint_save", moment="before")
   preview = session.dry_run(["checkpoint_save"])
   summary = session.summarize_history()
   walkthrough = session.walkthrough(export_html="artifacts/workflow.html")

``workflow`` shows every cataloged operation as done, available, blocked, or
skipped. ``explain`` gives operation-level choices, prerequisites, and limits.
``dry_run`` previews calls without mutating state. ``summarize_history`` lists
operation counts and heuristic unresolved risks. ``walkthrough`` joins statuses
to history and unresolved catalog risks in offline HTML. Available operations
are possibilities, not recommendations.

10. Persist the right artifact
------------------------------

Use ``checkpoint_save`` for data, roles, partitions, Session history, and
optional preprocess plan objects. Use ``save_pipeline`` for fitted plans plus
the estimator and model card, or ``save_model`` for the estimator alone. Retain
evaluation context and dependency versions with the chosen artifacts.

Score new frames with ``predict_from_pipeline`` (or
``Session.predict_from_pipeline``) so preprocess plans and the estimator run
in one call. Resample plans remain lineage-only at score time.

After loading a checkpoint, inspect ``reattach_result`` before fitting. A
``data_only`` load is a fresh semantic start. Never deserialize an untrusted
model bundle.
