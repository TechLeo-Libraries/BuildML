Workflow guide
==============

This is the order I use when the job is a table and a holdout I need to
trust. It is a decision path, not a requirement to call every method. At
each boundary, stop when the assumptions do not match the project.

The order exists because each stage changes what the next stage is
allowed to see. Ingest does not learn. Roles decide what may be used.
The split locks the holdout. Preparation and fitting learn from train.
Evaluation reads a partition you name. If you invert that, BuildML
refuses. If you follow it on the wrong split (random rows when the unit
is a customer), the library cannot save you. That judgment is yours.

1. Ingest without forcing scale
-------------------------------

Use ``Session.ingest`` with a DataFrame or a supported path. Read
``session.ingest_report`` for detected format, estimated size,
recommended mode and engine, and warnings. Use ``dry_run=True`` or
``read_nrows`` before a large load.

Do not force memory mode merely to silence a warning. Changing mode
after ingestion records policy metadata. It does not move an already
materialized frame out of memory.

2. Assign and review roles
--------------------------

Call ``set_roles`` from what you know about the columns. Confirm one
intended target. Keep identifiers, post-outcome fields, and fields that
will not exist at scoring time out of features.

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
still ambiguous.

3. Define partitions
--------------------

Use random or stratified ``split`` only for exchangeable rows. Add
validation when choices will be repeated. For groups, time, or
membership decided outside BuildML, call ``group_split``,
``time_split``, or ``inject_split``.

Check counts, class support, chronology, group isolation, and
duplicates. BuildML enforces disjoint memberships and train-only fit
scope. It cannot detect related entities or future leakage from values
alone.

4. Explore without mining test
------------------------------

Run EDA after roles are assigned. Prefer training or development
evidence for choices. Reserve final test interpretation. A full-dataset
report is descriptive. It is not automatically valid model-selection
evidence.

.. code-block:: python

   report = session.eda(
       include_plots=True,
       export_html="artifacts/eda_research.html",
       html_format="research",
   )
   # Live app: session.eda_app(port=8765)  # needs buildml[dashboard]

Read each finding's evidence, partition, denominator, severity, and
limit. Investigate collection errors before you treat a statistical
association as a feature opportunity.

5. Prepare in train-fitted order
--------------------------------

Choose only the operations the estimator and the data need:

.. code-block:: python

   session.impute(strategy="median")
   session.encode(method="onehot")
   session.scale(method="standard")

BuildML learns these plans on train and applies frozen parameters to
the other partitions. Review generated columns and stored plans. If you
export with ``to_pandas()`` and prepare outside the Session, select
``session.partition("train")`` yourself; the full exported frame
contains every partition.

For ``cv_score``, ``grid_search``, ``optuna_search``,
``evolutionary_search``, or ``nested_cv_score``, use a fold-local
``PreprocessRecipe`` on data that has **not** already been prepared on
the whole training partition. If Session-global fit-capable plans
already exist, CV and search refuse with ``LeakageError`` even if you
pass a recipe. Recipes do not rebuild from raw rows. Opt in only with
``allow_session_global_preprocess=True``, or re-ingest / checkpoint-load
unpoisoned data. Resample and ``apply_custom_transform`` stay
Session-global only.

The full recipe, weight, and refuse patterns live in
:doc:`leakage-cv-recipes`. Engine ingest and materialization live in
:doc:`engines-polars-duckdb`.

When you tune hyperparameters or recipe knobs such as ``select_k`` /
``n_bins``, pass ``recipe_grid`` / ``recipe_distributions`` /
``recipe_space`` into search or ``nested_cv_score``. Outer folds record
``best_params`` / ``best_recipe_knobs`` without using Session test rows.
Optuna needs ``pip install 'buildml[optuna]'``.

6. Establish a baseline and fit
-------------------------------

Fit a simple candidate before adding complexity. ``compare_models`` can
apply one prepared representation to multiple estimators. Candidates
that need different preprocessing should be compared in separately
controlled workflows.

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

The ranking metric must reflect error costs. Stop when the partition is
too small, a candidate has incompatible inputs, or the score difference
is within observed variation.

When hyperparameters or selection knobs are tuned with CV, use
``nested_cv_score`` for the post-selection estimate. Do not treat
``grid_search`` inner means as untouched generalization claims.

7. Evaluate the fixed choice
----------------------------

Use validation during iteration and test after model and feature
choices are fixed.

.. code-block:: python

   result = session.evaluate(
       partition="test",
       include_plots=True,
       export_html="artifacts/evaluation.html",
   )

Inspect task baselines, confusion or residual structure, class
prevalence, sample count, and skipped diagnostics. A self-contained
report preserves the rendered evidence, not the correctness of the
evaluation design.

8. Diagnose decisions
---------------------

For probability decisions, inspect calibration before selecting a
threshold. Select thresholds on validation and assess the fixed policy
on test. Use permutation importance as a model-reliance audit, not as a
causal or automatic feature-selection rule. Use learning curves only
when their cross-validation folds respect row dependencies.

9. Explain and hand off
-----------------------

.. code-block:: python

   status = session.workflow()
   before = session.explain("checkpoint_save", moment="before")
   preview = session.dry_run(["checkpoint_save"])
   summary = session.summarize_history()
   walkthrough = session.walkthrough(export_html="artifacts/workflow.html")

``workflow`` shows every cataloged operation as done, available,
blocked, or skipped. ``explain`` is this Session, right now. ``learn``
is the idea behind a call, or any term you did not recognise.
``dry_run`` previews without mutating state. ``walkthrough`` joins
status, history, and unresolved catalog risks into offline HTML.
Available operations are possibilities, not recommendations.

10. Persist the right artifact
------------------------------

Use ``checkpoint_save`` for data, roles, partitions, Session history,
and optional preprocess plan objects. Use ``save_pipeline`` for fitted
plans plus the estimator and model card, or ``save_model`` for the
estimator alone. Keep evaluation context and dependency versions with
the artifacts you ship.

Score new frames with ``predict_from_pipeline`` so preprocess plans and
the estimator run in one call. Resample plans remain lineage-only at
score time. ``save_pipeline`` writes ``schema_contract.json``.
``predict_from_pipeline`` coerces compatible dtypes when safe, then
validates role-aware required columns. Older bundles without a contract
remain loadable.

After loading a checkpoint, inspect ``reattach_result`` before fitting.
A ``data_only`` load is a fresh semantic start. Never deserialize an
untrusted model bundle.
