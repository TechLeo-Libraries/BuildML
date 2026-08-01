Current capabilities
====================

Data and workflow state
-----------------------

* Ingest Pandas DataFrames and CSV, Parquet, Arrow, and Excel sources.
* Record source detection, scale estimates, mode and engine choices, and
  loading warnings in an ingest report.
* Assign feature, target, identifier, group, time, weight, and ignored roles.
* Create random or stratified partitions, or inject externally designed row
  memberships.
* Save and validate checkpoints containing data, roles, partitions, history,
  and an integrity manifest.

Preparation
-----------

* Drop columns and extract date parts.
* Fit imputation, categorical encoding, and scaling on training rows.
* Resample only the training partition when the ``imbalanced`` extra is
  installed.

Models and diagnostics
----------------------

* Fit sklearn-compatible classifiers and regressors.
* Compare named estimators under one partition and ranking metric.
* Evaluate classification or regression metrics and error diagnostics.
* Inspect calibration, threshold tradeoffs, learning curves, permutation
  importance, and task-adaptive plot boards.

Explanation and reports
-----------------------

* Explain an operation before or after execution from a versioned operation
  catalog.
* Resolve workflow operations as done, available, blocked, or skipped.
* Export EDA, evaluation, diagnostic, and workflow reports as local HTML.

Boundaries
----------

BuildML does not infer valid grouped or temporal evaluation boundaries. It
does not make causal claims from associations or feature importance. The
selected engine does not make every sklearn-facing operation out-of-core.
Checkpoints do not contain fitted models, and model bundles do not contain the
Session dataset or split history.
