"""Fit, predict, and evaluate a supervised model without leaking the holdout.

This is the classical machine-learning core: give it a dataset with a target, a
split, and any scikit-learn-compatible estimator, and it trains on the train
partition and scores on whichever partition you name.

The value it adds over calling scikit-learn directly is that the boundaries are
enforced rather than remembered. :func:`fit_estimator` refuses to run on
anything but the train partition, so the holdout cannot be trained on by
accident. Feature columns come from the dataset's roles, so an ID or a group key
cannot drift into the model as a feature — a leak that produces excellent scores
and a worthless model. A weight column that the estimator cannot accept raises
instead of being silently dropped.

Evaluation is deliberately broad. A single number hides the failure that matters:
accuracy looks fine on imbalanced data while the model predicts the majority
class throughout, and R² looks respectable while the residuals fan out at the
top of the range. :func:`evaluate_estimator` therefore returns a card — several
metrics, per-class or residual diagnostics, and recommendations pointing at what
the numbers suggest is wrong.

Nothing here selects a model for you. Fitting one estimator and reading its score
tells you how that estimator did, not whether it was a good choice; see
:mod:`buildml.model.compare` and :mod:`buildml.model.selection` for that.

See Also
--------
buildml.model.selection : Cross-validation and hyperparameter search.
buildml.model.compare : Scoring several estimators against each other.
buildml.model.diagnostics : Deeper analysis of a fitted model.
buildml.data.splits : Building the split this module fits within.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin, clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_recall_fscore_support,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.validation import has_fit_parameter

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset
from buildml.data.engines.prep import MaterializePrepResult, prepare_design_frame
from buildml.data.splits import SplitPlan, assert_fit_partition, frame_for_partition

TaskType = Literal["classification", "regression", "auto"]


@dataclass(slots=True)
class FitResult:
    """A trained model together with the contract it was trained under.

    The estimator alone is not enough to use safely later. Prediction needs the
    feature columns in the order the model saw them, and interpreting a score
    needs to know which task was inferred and how much data it was trained on.
    Keeping all of it together is what lets :func:`predict_estimator` detect a
    mismatched frame instead of producing quiet nonsense.

    Attributes
    ----------
    estimator:
        The fitted model. A clone of what you passed in, so your original is
        left untouched and can be reused.
    task:
        ``'classification'`` or ``'regression'``, as resolved at fit time. This
        determines which metrics evaluation computes.
    feature_columns:
        The columns used, in order. Prediction reindexes to exactly these, which
        is what catches a frame whose columns arrived in a different order.
    target_column:
        The column that was predicted.
    n_train_rows:
        How many rows the fit actually saw — after any sampling, so this is the
        real training size rather than the partition size.
    weight_column:
        The weight-role column, or ``None``. Kept so evaluation can weight its
        metrics the same way the fit weighted its loss.

    Notes
    -----
    **``n_train_rows`` is worth reading.** A model trained on a few hundred rows
    with many features will score erratically across resplits, and the row count
    is the quickest way to notice that before trusting a single holdout number.

    See Also
    --------
    EvaluateResult : What scoring this model produces.
    """

    estimator: Any
    task: Literal["classification", "regression"]
    feature_columns: tuple[str, ...]
    target_column: str
    n_train_rows: int
    weight_column: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Summarise the fit as plain data for history and logs.

        The estimator is reduced to its class name, since the fitted object
        itself is neither serialisable to JSON nor meaningful in a log.

        Returns
        -------
        dict
            The estimator's class name, task, feature columns, target, training
            row count, and weight column.

        Notes
        -----
        **This is a record, not a checkpoint.** Nothing here can rebuild the
        model; use the pipeline and checkpoint machinery to persist one.
        """
        return {
            "estimator": type(self.estimator).__name__,
            "task": self.task,
            "feature_columns": list(self.feature_columns),
            "target_column": self.target_column,
            "n_train_rows": self.n_train_rows,
            "weight_column": self.weight_column,
        }


@dataclass(slots=True)
class EvaluateResult:
    """Several metrics, supporting diagnostics, and what they seem to say.

    A card rather than a score, because one number is where model evaluation
    usually goes wrong. Accuracy of 0.97 on data that is 97% one class describes
    a model that has learned to say "no"; an R² of 0.85 says nothing about
    whether the errors are evenly spread or concentrated in the range you care
    about. Reading several metrics against each other is what surfaces those,
    and the diagnostics are there to confirm what the divergence implies.

    Attributes
    ----------
    partition:
        Which partition was scored. The single most important field here: a
        train score and a test score are different kinds of statement, and only
        one of them estimates future performance.
    task:
        ``'classification'`` or ``'regression'``, deciding which metrics appear.
    metrics:
        The headline numbers. For regression: MAE, MSE, RMSE, median absolute
        error, R², and MAPE where the target allows it. For classification:
        accuracy, balanced accuracy, weighted precision, recall and F1, macro
        F1, and — when the estimator gives probabilities — log loss, ROC AUC,
        and average precision.
    diagnostics:
        The detail behind the metrics: a confusion matrix, per-class scores and
        a full classification report, or a residual summary with quantiles.
    n_rows:
        How many rows were scored. Small partitions produce noisy metrics, and
        per-class figures on a rare class can rest on a handful of rows.
    recommendations:
        Plain-language observations drawn from the numbers — a negative R², an
        accuracy far above balanced accuracy, metrics that could not be
        computed. Prompts to investigate, not conclusions.

    Notes
    -----
    **Accuracy far above balanced accuracy means the model is riding class
    imbalance.** Balanced accuracy averages recall over classes, so it drops
    when a minority class is being missed while accuracy stays high.

    **A negative R² is not an error.** It means the predictions are worse than
    always guessing the training mean.

    **RMSE much larger than MAE means the errors are lopsided.** Squaring
    magnifies large misses, so the gap says a minority of predictions are badly
    wrong — worth finding individually rather than averaging away.

    See Also
    --------
    show : Printing this card.
    buildml.model.diagnostics : Going further than these summary numbers.
    """

    partition: str
    task: Literal["classification", "regression"]
    metrics: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    n_rows: int = 0
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert the card to plain data for history, logs, and reports.

        Everything is copied rather than referenced, so a caller mutating the
        returned dictionary cannot alter the recorded result.

        Returns
        -------
        dict
            ``partition``, ``task``, ``metrics``, ``diagnostics``, ``n_rows``,
            and ``recommendations``.

        Notes
        -----
        **``diagnostics`` can be large.** A classification report over many
        classes carries a nested entry per class, which is worth trimming before
        writing this into a compact log.
        """
        return {
            "partition": self.partition,
            "task": self.task,
            "metrics": dict(self.metrics),
            "diagnostics": self.diagnostics,
            "n_rows": self.n_rows,
            "recommendations": list(self.recommendations),
        }

    def show(self) -> None:
        """Print the metrics and the first ten recommendations.

        For reading a result at a prompt. Prints the task, partition, and row
        count on one line, then every metric, then the recommendations.

        Notes
        -----
        **The diagnostics are not printed**, since a confusion matrix and a
        per-class report do not fit a terminal digest. Read ``diagnostics``
        directly for those.

        **Recommendations are truncated at ten**, which in practice is all of
        them.
        """
        print(f"Evaluate · {self.task} · partition={self.partition} · n={self.n_rows}")
        for key, value in self.metrics.items():
            print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")
        for tip in self.recommendations[:10]:
            print(f"  - {tip}")


def weight_column(dataset: Dataset) -> str | None:
    """Find the column marked as sample weights, if there is one.

    Sample weights let some rows count for more than others during fitting —
    useful when rows represent different numbers of underlying events, when a
    rare class needs amplifying, or when recent observations should dominate
    older ones.

    Parameters
    ----------
    dataset:
        The dataset whose roles are inspected.

    Returns
    -------
    str or None
        The weight column's name, or ``None`` when no weight role is set.

    Raises
    ------
    ValidationError
        If more than one column carries the weight role. Two weight columns has
        no defined meaning, so this is a configuration error rather than
        something to resolve by picking one.

    See Also
    --------
    validate_sample_weights : Checking the values are usable.
    """
    cols = dataset.role_columns(ColumnRole.WEIGHT)
    if not cols:
        return None
    if len(cols) > 1:
        raise ValidationError(
            f"Expected at most one weight column, found {len(cols)}: {cols}"
        )
    return cols[0]


def resolve_feature_columns(dataset: Dataset) -> list[str]:
    """Decide which columns the model may see, excluding the ones that leak.

    Roles exist largely for this moment. When features are declared explicitly,
    those are used. Otherwise every column is a feature *except* the target and
    the roles that must never be modelled: IDs, group keys, time columns,
    weights, and anything explicitly ignored.

    Those exclusions are the point. A customer ID is often the strongest
    "predictor" in a dataset and predicts nothing at all — it memorises which
    row is which, scoring beautifully in training and collapsing on new
    customers. A group key leaks the same way. Time columns invite the model to
    learn "later rows are positive", which holds until the day you deploy.

    Parameters
    ----------
    dataset:
        The dataset whose roles decide the feature set.

    Returns
    -------
    list of str
        Feature column names, in dataset order.

    Raises
    ------
    ValidationError
        If no target is set, if no features remain after exclusions, or if the
        weight column is also declared a feature.

    Notes
    -----
    **A weight column cannot also be a feature.** Weights encode how much a row
    counts, and letting the model read that as a predictor lets it learn the
    weighting scheme rather than the target. This raises rather than resolving
    the ambiguity quietly.

    **Automatic selection is a fallback, not a recommendation.** It cannot know
    that a column recording the outcome under a different name is a leak. Set
    feature roles explicitly for anything you intend to deploy.

    See Also
    --------
    buildml.core.types.ColumnRole : The roles consulted here.
    """
    target = dataset.require_target()
    feature_cols = dataset.role_columns(ColumnRole.FEATURE)
    if not feature_cols:
        skip = {
            *dataset.role_columns(ColumnRole.TARGET),
            *dataset.role_columns(ColumnRole.ID),
            *dataset.role_columns(ColumnRole.IGNORE),
            *dataset.role_columns(ColumnRole.GROUP),
            *dataset.role_columns(ColumnRole.TIME),
            *dataset.role_columns(ColumnRole.WEIGHT),
        }
        feature_cols = [c for c in dataset.columns if c not in skip and c != target]
    weight = weight_column(dataset)
    if weight is not None and weight in feature_cols:
        raise ValidationError(
            f"Weight column '{weight}' cannot also be treated as a feature. "
            "Remove it from FEATURE roles or clear the WEIGHT role."
        )
    if not feature_cols:
        raise ValidationError("No feature columns available for modeling")
    return feature_cols


def validate_sample_weights(weights: pd.Series, *, column: str) -> pd.Series:
    """Check that a weight column is usable before it silently distorts the fit.

    Bad weights do not usually raise inside scikit-learn; they change the answer.
    A missing value coerced to zero drops a row from training. A negative weight
    asks the optimiser to make a row *more* wrong. All-zero weights make the
    fit degenerate. Each is caught here, where the message can name the column.

    Parameters
    ----------
    weights:
        The raw weight values, which may be any dtype.
    column:
        The column's name, used only so the error says which column is wrong.

    Returns
    -------
    pandas.Series
        The weights as floats, ready to pass to scikit-learn.

    Raises
    ------
    ValidationError
        If any value is missing or non-numeric, if any is negative, or if none
        is positive.

    Notes
    -----
    **Weights are not normalised.** Only their ratios matter to most estimators,
    so doubling every weight generally changes nothing — but weighted metrics
    and any absolute-scale interpretation do shift.

    **A zero weight excludes a row rather than erroring**, which is a legitimate
    way to mask rows. Only an entirely zero column is refused.

    See Also
    --------
    weight_column : Finding the column.
    fit_kwargs_for_sample_weight : Passing weights to an estimator.
    """
    numeric = pd.to_numeric(weights, errors="coerce")
    if numeric.isna().any():
        raise ValidationError(
            f"Weight column '{column}' contains non-numeric or missing values"
        )
    values = numeric.to_numpy(dtype=float)
    if np.any(values < 0):
        raise ValidationError(f"Weight column '{column}' must be non-negative")
    if not np.any(values > 0):
        raise ValidationError(f"Weight column '{column}' must contain at least one positive weight")
    return numeric.astype(float)


def fit_kwargs_for_sample_weight(estimator: Any, sample_weight: pd.Series | None) -> dict[str, Any]:
    """Pass weights to the estimator, or refuse if it cannot accept them.

    Not every scikit-learn estimator supports ``sample_weight``. The dangerous
    outcome is not an error but a success: weights configured, quietly ignored,
    and a model trained as though every row counted equally while the results
    are read as though they did not.

    Parameters
    ----------
    estimator:
        The estimator about to be fitted. Its ``fit`` signature is inspected.
    sample_weight:
        Validated weights, or ``None`` when no weight role is set.

    Returns
    -------
    dict
        ``{'sample_weight': array}``, or an empty dict when there are no
        weights, suitable for splatting into ``fit``.

    Raises
    ------
    ValidationError
        If weights were supplied but the estimator's ``fit`` does not accept
        ``sample_weight``. Choose an estimator that does, or clear the weight
        role — the error should not be worked around.

    Notes
    -----
    **Support is detected from the signature**, so an estimator that declares
    the parameter and ignores it cannot be caught here.

    **Pipelines need the step-prefixed spelling** that scikit-learn requires, so
    a bare pipeline is reported as unsupported rather than silently mis-routed.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> fit_kwargs_for_sample_weight(RandomForestClassifier(), None)
    {}
    >>> kwargs = fit_kwargs_for_sample_weight(
    ...     RandomForestClassifier(), pd.Series([1.0, 2.0, 0.5])
    ... )
    >>> kwargs["sample_weight"].tolist()
    [1.0, 2.0, 0.5]

    An estimator that cannot weight refuses rather than ignoring them:

    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from buildml.core.errors import ValidationError
    >>> try:
    ...     fit_kwargs_for_sample_weight(
    ...         KNeighborsClassifier(), pd.Series([1.0, 2.0])
    ...     )
    ... except ValidationError:
    ...     print("refused")
    refused

    See Also
    --------
    validate_sample_weights : Producing the weights this consumes.
    """
    if sample_weight is None:
        return {}
    if not has_fit_parameter(estimator, "sample_weight"):
        raise ValidationError(
            f"{type(estimator).__name__} does not accept sample_weight, but a "
            "ColumnRole.WEIGHT column is set. Choose an estimator that supports "
            "sample_weight, or clear the weight role."
        )
    return {"sample_weight": sample_weight.to_numpy(dtype=float)}


def _feature_target_frames(
    dataset: Dataset,
    split_plan: SplitPlan,
    partition: Literal["train", "validation", "test"],
    *,
    sample_rows: int | None = None,
    random_state: int | None = 0,
    materialize_prep: bool = True,
) -> tuple[pd.DataFrame, pd.Series, list[str], str, pd.Series | None]:
    """Build the X, y, and weights for one partition.

    The single place partition data becomes a design matrix, which is what keeps
    fit, predict, and evaluate consistent: all three resolve features from the
    same roles and validate weights the same way, so a column set that trains
    cannot fail to score.

    Parameters
    ----------
    dataset:
        The source data.
    split_plan:
        Partition membership.
    partition:
        Which partition to build.
    sample_rows:
        Cap on rows, applied by the engine before materialising. ``None`` takes
        everything.
    random_state:
        Seed for that sampling.
    materialize_prep:
        Whether to route through the engine's projection and sampling. ``False``
        slices the partition frame directly, skipping the engine path.

    Returns
    -------
    tuple
        ``(x, y, feature_columns, target_column, sample_weight)``.

    Raises
    ------
    ValidationError
        If no target is set, no features remain, or the weight column is absent
        or contains unusable values.

    Notes
    -----
    **Sampling changes what the numbers describe.** A sampled fit is trained on
    a subset, and a sampled evaluation scores one — neither is wrong, but both
    describe the sample rather than the partition.
    """
    target = dataset.require_target()
    feature_cols = resolve_feature_columns(dataset)
    weight = weight_column(dataset)

    frame = frame_for_partition(dataset, split_plan, partition)
    if not materialize_prep:
        sample_weight = None
        if weight is not None:
            if weight not in frame.columns:
                raise ValidationError(f"Weight column '{weight}' is missing from the dataset")
            sample_weight = validate_sample_weights(frame[weight], column=weight)
        return frame[feature_cols], frame[target], feature_cols, target, sample_weight

    # Engine-aware projection/sampling on the partition slice before sklearn.
    project_cols = [*feature_cols, target]
    if weight is not None:
        if weight not in frame.columns:
            raise ValidationError(f"Weight column '{weight}' is missing from the dataset")
        project_cols.append(weight)
    partition_ds = Dataset.from_pandas(
        frame,
        schema=dataset.schema,
        mode=dataset.mode,
        engine=dataset.engine,
        source=dataset.source,
        roles=dict(dataset.roles),
    )
    prep = prepare_design_frame(
        partition_ds,
        project_cols,
        sample_rows=sample_rows,
        random_state=random_state,
        context=f"estimator {partition} design matrix",
    )
    prepared = prep.frame
    sample_weight = None
    if weight is not None:
        sample_weight = validate_sample_weights(prepared[weight], column=weight)
    return (
        prepared[feature_cols],
        prepared[target],
        feature_cols,
        target,
        sample_weight,
    )


def materialize_partition_design(
    dataset: Dataset,
    split_plan: SplitPlan,
    partition: Literal["train", "validation", "test"] = "train",
    *,
    sample_rows: int | None = None,
    random_state: int | None = 0,
) -> MaterializePrepResult:
    """Narrow a partition to the modelling columns using the dataset's engine.

    scikit-learn needs a Pandas frame in memory, but the data may live in Polars
    or DuckDB. Doing the column projection and any row sampling in the engine
    first means only the needed columns cross that boundary, which on a wide
    table is the difference between materialising a few columns and all of them.

    Parameters
    ----------
    dataset:
        The source data.
    split_plan:
        Partition membership.
    partition:
        Which partition to materialise.
    sample_rows:
        Cap on rows. ``None`` takes the whole partition.
    random_state:
        Seed for the sampling, so a sampled run reproduces.

    Returns
    -------
    MaterializePrepResult
        The Pandas frame plus a record of what the engine did, including whether
        sampling occurred.

    Raises
    ------
    ValidationError
        If no target is set, no features remain, or the weight column is absent.

    Notes
    -----
    **This is not out-of-core training.** The projected frame still has to fit
    in memory; projection reduces its width, and sampling reduces its height.
    Neither lets you train on data larger than RAM.

    **Sampling is disclosed in the result** precisely because a model trained on
    a sample and reported without that context looks like a model trained on
    everything.

    See Also
    --------
    buildml.data.engines.prep.prepare_design_frame : The engine-side work.
    """
    target = dataset.require_target()
    feature_cols = resolve_feature_columns(dataset)
    weight = weight_column(dataset)
    project_cols = [*feature_cols, target]
    if weight is not None:
        project_cols.append(weight)
    frame = frame_for_partition(dataset, split_plan, partition)
    partition_ds = Dataset.from_pandas(
        frame,
        schema=dataset.schema,
        mode=dataset.mode,
        engine=dataset.engine,
        source=dataset.source,
        roles=dict(dataset.roles),
    )
    return prepare_design_frame(
        partition_ds,
        project_cols,
        sample_rows=sample_rows,
        random_state=random_state,
        context=f"estimator {partition} design matrix",
    )


def _infer_task(
    y: pd.Series,
    task: TaskType,
    estimator: Any,
) -> Literal["classification", "regression"]:
    """Work out whether this is a classification or a regression problem.

    Tried in order of reliability. An explicit ``task`` wins. Otherwise the
    estimator itself usually settles it, since scikit-learn classifiers and
    regressors are distinguishable by their base classes. Only when neither is
    available does the target's shape get a vote: numeric with many distinct
    values is treated as regression, and anything else as classification.

    Parameters
    ----------
    y:
        The training target.
    task:
        ``'classification'``, ``'regression'``, or ``'auto'`` to infer.
    estimator:
        The estimator, inspected for the scikit-learn mixins.

    Returns
    -------
    str
        ``'classification'`` or ``'regression'``.

    Notes
    -----
    **The value-count heuristic can be wrong, and quietly.** The threshold is
    more than ten distinct values or more than 20% of rows distinct. An integer
    target with many levels that is genuinely categorical — a product code, a
    postcode — reads as regression, and the metrics that follow are meaningless
    for it. Pass ``task`` explicitly when the target is not obviously one or the
    other.

    **The heuristic only runs for estimators that are neither mixin**, which in
    practice means custom or wrapped estimators.

    Examples
    --------
    The estimator settles it, whatever the target looks like:

    >>> import pandas as pd
    >>> from sklearn.linear_model import LogisticRegression
    >>> _infer_task(pd.Series([1.5, 2.5, 3.5]), "auto", LogisticRegression())
    'classification'

    With no mixin to consult, the target's shape decides — and here it decides
    wrongly, which is why ``task`` exists:

    >>> codes = pd.Series(range(50))
    >>> _infer_task(codes, "auto", object())
    'regression'
    >>> _infer_task(codes, "classification", object())
    'classification'
    """
    if task != "auto":
        return task
    if isinstance(estimator, ClassifierMixin):
        return "classification"
    if isinstance(estimator, RegressorMixin):
        return "regression"
    if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > max(10, int(0.2 * len(y))):
        return "regression"
    return "classification"


def fit_estimator(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    estimator: Any,
    *,
    task: TaskType = "auto",
    sample_rows: int | None = None,
    random_state: int | None = 0,
) -> FitResult:
    """Train an estimator on the train partition, and only the train partition.

    The central operation of classical supervised learning: show the model
    labelled examples so it can learn the mapping from features to target. What
    this adds over calling ``estimator.fit`` yourself is that the boundaries are
    enforced. The train partition is the only data the model can see, features
    come from roles rather than from whatever columns happen to be present, and
    a weight column that cannot be honoured stops the run.

    The estimator is cloned before fitting, so the object you pass in stays
    unfitted and can be reused across several runs.

    Parameters
    ----------
    dataset:
        The data, with a target role set.
    split_plan:
        The split. Required — fitting without one would mean training on
        everything, leaving nothing to score honestly against.
    estimator:
        Any scikit-learn-compatible estimator: anything with ``fit`` and
        ``predict`` and cloneable parameters. Pipelines, ensembles, and
        third-party estimators following the API all work.
    task:
        ``'classification'``, ``'regression'``, or ``'auto'`` to infer from the
        estimator and target.
    sample_rows:
        Train on at most this many rows. For iterating quickly on a large
        dataset; the resulting model is weaker than one trained on everything.
    random_state:
        Seed for that sampling.

    Returns
    -------
    FitResult
        The fitted estimator with the feature columns, task, target, training
        row count, and weight column it was trained under.

    Raises
    ------
    ValidationError
        If ``split_plan`` is ``None``, if no target or features are resolvable,
        if the weight column is unusable, or if weights are set and the
        estimator cannot accept them.

    Notes
    -----
    **Train is the only partition that can be fitted on, by construction.** This
    is the guard that makes a later holdout score mean something: a model that
    has seen the test rows scores well on them regardless of whether it learned
    anything general.

    **The estimator is cloned, so its learned state does not persist here.**
    Read the fitted model from the returned ``FitResult``, not from the object
    you passed in.

    **Preprocess before fitting, not after.** Scaling, encoding, and imputation
    must be fitted on train and applied to holdout; doing it across the whole
    dataset leaks the holdout's distribution into training. The preprocessing
    module handles this.

    **A fitted model is not a validated one.** This produces a model; whether it
    is any good is what evaluation and cross-validation are for.

    Examples
    --------
    Fit a gradient-boosted classifier::

        from sklearn.ensemble import HistGradientBoostingClassifier

        fit = fit_estimator(
            dataset, split_plan, HistGradientBoostingClassifier(random_state=0)
        )
        print(fit.task, fit.n_train_rows, fit.feature_columns)

    See Also
    --------
    evaluate_estimator : Scoring the result.
    predict_estimator : Generating predictions.
    buildml.model.selection.cross_validate_estimator : A more stable estimate.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None

    x_train, y_train, feature_cols, target, sample_weight = _feature_target_frames(
        dataset,
        split_plan,
        "train",
        sample_rows=sample_rows,
        random_state=random_state,
    )
    resolved_task = _infer_task(y_train, task, estimator)
    model = clone(estimator)
    model.fit(x_train, y_train, **fit_kwargs_for_sample_weight(model, sample_weight))
    return FitResult(
        estimator=model,
        task=resolved_task,
        feature_columns=tuple(feature_cols),
        target_column=target,
        n_train_rows=int(len(x_train)),
        weight_column=weight_column(dataset),
    )


def predict_estimator(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    fit_result: FitResult,
    *,
    partition: Literal["train", "validation", "test"] = "test",
    return_proba: bool = False,
) -> pd.Series | pd.DataFrame:
    """Generate predictions for a partition, as labels or as probabilities.

    Rebuilds the partition's design matrix, reindexes it to exactly the columns
    the model was trained on, and predicts. That reindexing matters: a frame
    with the right columns in the wrong order predicts confidently and wrongly
    if handed straight to scikit-learn, and here it cannot.

    Parameters
    ----------
    dataset:
        The data to predict on.
    split_plan:
        Partition membership.
    fit_result:
        The trained model and its column contract.
    partition:
        Which partition to predict for. Defaults to test.
    return_proba:
        Return class probabilities rather than labels. Ignored when the
        estimator has no ``predict_proba``, in which case labels come back.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Labels as a Series named ``prediction``, or one probability column per
        class named ``proba_<class>``. Either way indexed like the input rows,
        so predictions can be joined back to their source.

    Raises
    ------
    ValidationError
        If ``split_plan`` is ``None``, or if any training feature column is
        missing from the partition.

    Notes
    -----
    **``return_proba`` degrades silently.** An estimator without
    ``predict_proba`` returns labels instead of raising, so check the return
    type rather than assuming probabilities arrived.

    **Predicted probabilities are usually not calibrated.** A model's 0.8 rarely
    means "correct 80% of the time" — tree ensembles in particular tend to be
    over-confident. Calibrate before using probabilities as probabilities rather
    than as a ranking.

    **The default 0.5 threshold is a convention, not a decision.** On imbalanced
    data or when false positives and false negatives cost differently, choose a
    threshold from the probabilities instead of accepting the labels.

    See Also
    --------
    evaluate_estimator : Scoring predictions against the truth.
    """
    if split_plan is None:
        raise ValidationError("A split is required for partitioned prediction")
    x, _, _, _, _ = _feature_target_frames(dataset, split_plan, partition)
    missing = [c for c in fit_result.feature_columns if c not in x.columns]
    if missing:
        raise ValidationError(f"Missing feature columns for prediction: {missing}")
    x = x[list(fit_result.feature_columns)]
    preds = fit_result.estimator.predict(x)
    if return_proba and hasattr(fit_result.estimator, "predict_proba"):
        proba = fit_result.estimator.predict_proba(x)
        classes = getattr(fit_result.estimator, "classes_", range(proba.shape[1]))
        columns = [f"proba_{c}" for c in classes]
        return pd.DataFrame(proba, columns=columns, index=x.index)
    return pd.Series(preds, index=x.index, name="prediction")


def evaluate_estimator(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    fit_result: FitResult,
    *,
    partition: Literal["train", "validation", "test"] = "test",
) -> EvaluateResult:
    """Score a fitted model on a partition and explain what the numbers imply.

    Predicts, computes a spread of metrics appropriate to the task, gathers the
    diagnostics behind them, and adds plain-language observations about what the
    combination suggests.

    The spread is the point. One metric can only mislead: accuracy hides a
    majority-class predictor, R² hides heteroscedastic residuals, and precision
    hides a model that achieves it by almost never predicting positive. Metrics
    that disagree are the signal worth acting on, and the diagnostics are there
    to confirm what the disagreement means.

    Parameters
    ----------
    dataset:
        The data to score against.
    split_plan:
        Partition membership.
    fit_result:
        The trained model and its column contract.
    partition:
        Which partition to score. Defaults to test.

    Returns
    -------
    EvaluateResult
        Metrics, diagnostics, row count, and recommendations.

    Raises
    ------
    ValidationError
        If ``split_plan`` is ``None``, or if a training feature column is
        missing from the partition.

    Notes
    -----
    **Which partition you score decides what the number means.** Train tells you
    how well the model memorised. Validation is what you may tune against, and
    it stops being an unbiased estimate the moment you do. Test estimates future
    performance, and only while it stays untouched — score it once, at the end.

    **Metrics are weighted when a weight role is set.** A weighted metric answers
    a different question from an unweighted one, so weighted and unweighted runs
    are not comparable.

    **Some metrics are omitted rather than faked.** MAPE is skipped when the
    target contains zeros, and probability metrics when the estimator offers no
    probabilities; each absence appears in ``recommendations``.

    **Per-class figures on a rare class rest on very few rows.** Check
    ``support`` in the per-class diagnostics before reading a class F1 of 0.4 as
    a measurement rather than noise.

    Examples
    --------
    Score on validation while iterating, and keep test for the end::

        result = evaluate_estimator(
            dataset, split_plan, fit, partition="validation"
        )
        result.show()
        print(result.diagnostics["confusion_matrix"])

    See Also
    --------
    fit_estimator : Producing the model being scored.
    buildml.model.compare : Scoring several models on equal terms.
    buildml.model.diagnostics : Going beyond summary metrics.
    """
    if split_plan is None:
        raise ValidationError("A split is required for partitioned evaluation")
    x, y_true, _, _, sample_weight = _feature_target_frames(dataset, split_plan, partition)
    x = x[list(fit_result.feature_columns)]
    y_pred = fit_result.estimator.predict(x)
    sw = None if sample_weight is None else sample_weight.to_numpy(dtype=float)
    metrics: dict[str, float] = {}
    diagnostics: dict[str, Any] = {}
    tips: list[str] = []
    if sw is not None:
        diagnostics["sample_weight_column"] = fit_result.weight_column or weight_column(dataset)
        tips.append("Evaluation metrics use ColumnRole.WEIGHT as sample_weight where supported.")

    if fit_result.task == "regression":
        residuals = y_true.to_numpy(dtype=float) - np.asarray(y_pred, dtype=float)
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred, sample_weight=sw))
        metrics["mse"] = float(mean_squared_error(y_true, y_pred, sample_weight=sw))
        metrics["rmse"] = float(np.sqrt(metrics["mse"]))
        metrics["median_ae"] = float(median_absolute_error(y_true, y_pred, sample_weight=sw))
        metrics["r2"] = float(r2_score(y_true, y_pred, sample_weight=sw))
        try:
            metrics["mape"] = float(
                mean_absolute_percentage_error(y_true, y_pred, sample_weight=sw)
            )
        except ValueError:
            tips.append("MAPE unavailable (zeros/near-zeros in target).")
        mean_resid = float(
            np.average(residuals, weights=sw) if sw is not None else np.mean(residuals)
        )
        diagnostics["residual_summary"] = {
            "mean": mean_resid,
            "std": float(np.std(residuals)),
            "q05": float(np.quantile(residuals, 0.05)),
            "q50": float(np.quantile(residuals, 0.50)),
            "q95": float(np.quantile(residuals, 0.95)),
        }
        if metrics["r2"] < 0:
            tips.append("Negative R² — model underperforms a mean baseline on this partition.")
    else:
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred, sample_weight=sw))
        metrics["balanced_accuracy"] = float(
            balanced_accuracy_score(y_true, y_pred, sample_weight=sw)
        )
        metrics["precision_weighted"] = float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0, sample_weight=sw)
        )
        metrics["recall_weighted"] = float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0, sample_weight=sw)
        )
        metrics["f1_weighted"] = float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0, sample_weight=sw)
        )
        metrics["f1_macro"] = float(
            f1_score(y_true, y_pred, average="macro", zero_division=0, sample_weight=sw)
        )
        labels = sorted(
            pd.unique(pd.concat([y_true.astype(str), pd.Series(y_pred).astype(str)]))
        )
        cm = confusion_matrix(
            y_true.astype(str),
            pd.Series(y_pred).astype(str),
            labels=labels,
            sample_weight=sw,
        )
        diagnostics["confusion_matrix"] = {
            "labels": labels,
            "matrix": cm.tolist(),
        }
        report = classification_report(
            y_true.astype(str),
            pd.Series(y_pred).astype(str),
            output_dict=True,
            zero_division=0,
            sample_weight=sw,
        )
        diagnostics["classification_report"] = report
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true.astype(str),
            pd.Series(y_pred).astype(str),
            labels=labels,
            average=None,
            zero_division=0,
            sample_weight=sw,
        )
        diagnostics["per_class"] = {
            str(label): {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(s),
            }
            for label, p, r, f, s in zip(labels, precision, recall, f1, support, strict=True)
        }
        if hasattr(fit_result.estimator, "predict_proba"):
            try:
                proba = fit_result.estimator.predict_proba(x)
                metrics["log_loss"] = float(
                    log_loss(
                        y_true,
                        proba,
                        labels=fit_result.estimator.classes_,
                        sample_weight=sw,
                    )
                )
                if len(fit_result.estimator.classes_) == 2:
                    metrics["roc_auc"] = float(
                        roc_auc_score(y_true, proba[:, 1], sample_weight=sw)
                    )
                    metrics["average_precision"] = float(
                        average_precision_score(y_true, proba[:, 1], sample_weight=sw)
                    )
                else:
                    metrics["roc_auc_ovr_weighted"] = float(
                        roc_auc_score(
                            y_true,
                            proba,
                            multi_class="ovr",
                            average="weighted",
                            sample_weight=sw,
                        )
                    )
            except ValueError as exc:
                tips.append(f"Probability metrics unavailable: {exc}")
        if metrics.get("balanced_accuracy", 1) + 1e-9 < metrics.get("accuracy", 0):
            tips.append("Accuracy ≫ balanced accuracy — inspect class imbalance / majority bias.")

    if not tips:
        tips.append(
            "No urgent evaluation warnings — compare against baselines and other estimators."
        )

    return EvaluateResult(
        partition=partition,
        task=fit_result.task,
        metrics=metrics,
        diagnostics=diagnostics,
        n_rows=int(len(y_true)),
        recommendations=tips,
    )
