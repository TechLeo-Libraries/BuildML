"""The plumbing every NLP operation shares: column resolution, partitions, metrics.

Each surface in this package: classification, topics, sentiment, summarisation
: needs the same four things before it can start: find the text column, pull the
right partition, extract the documents, and refuse clearly when any of that is
ambiguous. Centralising it here is what makes the whole package behave
consistently, so ``text_column=None`` resolves the same way and an empty
partition fails with the same message wherever you hit it.

The metric helpers are here for the same reason. A classification report should
mean the same thing whichever surface produced it.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Literal

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset
from buildml.data.splits import PartitionName, SplitPlan, frame_for_partition

logger = logging.getLogger(__name__)

PartitionOrAll = PartitionName | Literal["all"]

HOLDOUT_PARTITIONS: tuple[str, ...] = ("validation", "test")


def candidate_text_columns(frame: pd.DataFrame) -> list[str]:
    """List the string-like columns of a frame, ignoring roles entirely.

    The role-blind counterpart to :func:`resolve_text_column`, for code that
    runs before a session has roles or a split: chiefly the explain resolver,
    which needs to say "this dataset looks like it has text in it" while the
    user is still deciding what to do.

    Parameters
    ----------
    frame:
        The dataframe to inspect.

    Returns
    -------
    list of str
        Column names with object or string dtype, in frame order. Dtype is the
        only test applied, so a column of identifiers stored as strings will
        appear here.

    See Also
    --------
    resolve_text_column : The role-aware version every NLP operation uses.
    """
    return [
        str(column)
        for column in frame.columns
        if pd.api.types.is_object_dtype(frame[column])
        or pd.api.types.is_string_dtype(frame[column])
    ]


def resolve_text_column(dataset: Dataset, text_column: str | None) -> str:
    """Work out which column holds the documents, or refuse to guess.

    Every NLP operation starts here. Given an explicit name it validates it;
    given nothing it looks for string columns carrying the feature role,
    falling back to any string column that is not the target or an identifier.

    When several candidates remain it does *not* pick one arbitrarily. A column
    only wins automatically if its documents average at least 20 characters and
    are at least twice as long as the runner-up: the signature of real prose
    sitting beside a short categorical column. Anything less clear-cut raises,
    listing the candidates with their mean lengths, because silently modelling
    the wrong column produces a plausible model of nothing.

    Parameters
    ----------
    dataset:
        The dataset to resolve against.
    text_column:
        An explicit column name, or ``None`` to infer.

    Returns
    -------
    str
        The resolved column name.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        The named column is absent or numeric; no string-like candidate exists;
        or several candidates are too similar to choose between.

    Notes
    -----
    Giving your text column the feature role removes the ambiguity for good and
    is worth doing once on any dataset with several string columns.

    See Also
    --------
    candidate_text_columns : The role-blind dtype scan underneath this.
    """
    frame = dataset._ensure_pandas()
    if text_column is not None:
        if text_column not in dataset.columns:
            raise ValidationError(
                f"text_column={text_column!r} is not a dataset column. "
                f"Available columns: {list(dataset.columns)[:25]}"
            )
        if pd.api.types.is_numeric_dtype(frame[text_column]):
            raise ValidationError(
                f"Column {text_column!r} is numeric; NLP operations expect "
                "string-like documents."
            )
        return text_column

    string_like = set(candidate_text_columns(frame))
    candidates = [
        column
        for column in dataset.role_columns(ColumnRole.FEATURE)
        if column in string_like
    ]
    if not candidates:
        candidates = [
            column
            for column in dataset.columns
            if column in string_like
            and dataset.roles.get(column) not in {ColumnRole.TARGET, ColumnRole.ID}
        ]
    if not candidates:
        raise ValidationError(
            "No text-like column found. Give a string column the 'feature' role "
            "or pass text_column=... explicitly."
        )
    if len(candidates) > 1:
        # Longest mean document wins only when it is clearly the text column;
        # otherwise refuse so the caller stays in control.
        lengths = {
            column: float(frame[column].astype("string").fillna("").str.len().mean())
            for column in candidates
        }
        ranked = sorted(lengths.items(), key=lambda item: -item[1])
        best, best_len = ranked[0]
        runner_len = ranked[1][1]
        if best_len < 20.0 or best_len < 2.0 * max(runner_len, 1e-9):
            raise ValidationError(
                "Multiple text-like columns found; pass text_column= explicitly. "
                f"Candidates (mean characters): "
                f"{ {name: round(value, 1) for name, value in ranked} }"
            )
        return best
    return candidates[0]


def frame_for(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    partition: PartitionOrAll,
    *,
    operation: str,
) -> pd.DataFrame:
    """Get the rows for one partition, failing loudly when there are none.

    The single place partition selection happens in this package, so every
    operation enforces the same rule: you need a split unless you explicitly
    asked for everything.

    Parameters
    ----------
    dataset:
        The dataset to slice.
    split_plan:
        The split defining partitions. May be ``None`` only when ``partition``
        is ``'all'``.
    partition:
        ``'train'``, ``'validation'``, ``'test'``, or ``'all'``.
    operation:
        The calling operation's name, used to make the error message say which
        call needs fixing.

    Returns
    -------
    ~pandas.DataFrame
        The rows for that partition.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        A partition other than ``'all'`` was requested without a split plan, or
        the requested partition has no rows.

    Notes
    -----
    An empty partition raises rather than returning nothing. Downstream, an
    empty frame produces a metric of zero or ``NaN`` that looks like a result,
    and the real cause: a split that put no rows on one side: would be
    several steps removed by the time anyone noticed.
    """
    if partition == "all":
        return dataset._ensure_pandas()
    if split_plan is None:
        raise ValidationError(
            f"{operation} requires a SplitPlan unless partition='all'. "
            "Call Session.split(...) first."
        )
    frame = frame_for_partition(dataset, split_plan, partition)
    if frame.empty:
        raise ValidationError(
            f"Partition {partition!r} is empty; {operation} has nothing to process."
        )
    return frame


def documents_for(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    partition: PartitionOrAll,
    text_column: str,
    *,
    operation: str,
) -> tuple[list[str], pd.DataFrame]:
    """Pull a partition's documents, and the frame they came from.

    The frame comes back alongside the documents because callers almost always
    need both: the text to model, and the other columns to read targets or
    identifiers from, positionally aligned with it.

    Parameters
    ----------
    dataset:
        The dataset holding the text.
    split_plan:
        The split defining partitions.
    partition:
        Which rows to take.
    text_column:
        The already-resolved text column name.
    operation:
        The calling operation's name, for error messages.

    Returns
    -------
    tuple
        ``(documents, frame)``. Documents are plain strings with nulls turned
        into ``''``, one per frame row and in the same order.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        No split plan where one is needed, an empty partition, or the text
        column is absent from the partition.

    Notes
    -----
    Blank documents are kept so the two returned objects stay aligned. Callers
    report them through :func:`empty_document_rate` rather than dropping them.
    """
    frame = frame_for(dataset, split_plan, partition, operation=operation)
    if text_column not in frame.columns:
        raise ValidationError(
            f"Text column {text_column!r} is missing from partition {partition!r}."
        )
    documents = frame[text_column].astype("string").fillna("").astype(str).tolist()
    return documents, frame


def targets_for(frame: pd.DataFrame, target_column: str, *, operation: str) -> pd.Series:
    """Pull the label column, refusing to proceed if any label is missing.

    Every supervised NLP operation reads its targets through here, so the
    treatment of missing labels is one decision made once rather than a
    per-operation accident.

    Parameters
    ----------
    frame:
        The partition frame, normally from :func:`documents_for`.
    target_column:
        The label column's name.
    operation:
        The calling operation's name, for error messages.

    Returns
    -------
    ~pandas.Series
        The labels, positionally aligned with the frame's documents.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        The column is absent, or any value in it is null.

    Notes
    -----
    Nulls raise rather than being dropped, and the strictness is deliberate.
    Dropping rows here would silently break alignment between documents and
    labels, and it would quietly change what the metric denominator means :
    a model evaluated on the subset of rows that happened to have labels is
    not a model evaluated on your holdout. Impute or filter explicitly instead.
    """
    if target_column not in frame.columns:
        raise ValidationError(
            f"Target column {target_column!r} is missing from the {operation} frame."
        )
    series = frame[target_column]
    if series.isna().any():
        raise ValidationError(
            f"{operation} found null targets in the evaluation partition; "
            "impute or filter them explicitly rather than dropping silently."
        )
    return series


def empty_document_rate(documents: list[str]) -> float:
    """Measure how much of a corpus has no text in it at all.

    A blank document produces no features, so the model can only fall back on
    its class priors: it returns a confident-looking prediction derived from
    nothing about that row. This rate is reported on fit and predict results
    for exactly that reason.

    Parameters
    ----------
    documents:
        The documents to check. Whitespace-only counts as blank.

    Returns
    -------
    float
        The blank share, from 0.0 to 1.0. An empty list returns 0.0.

    Notes
    -----
    A non-zero rate usually means the column was optional in whatever collected
    it. Decide whether those rows belong in the model at all before treating
    their predictions as real.
    """
    if not documents:
        return 0.0
    blank = sum(1 for item in documents if not str(item).strip())
    return float(blank / len(documents))


def class_counts(values: Any) -> dict[str, int]:
    """Count rows per class, in a stable, serialisable form.

    Reported on fit results because class balance decides how to read every
    metric that follows: 95% accuracy is excellent on a balanced corpus and
    worthless when 95% of documents share one label.

    Parameters
    ----------
    values:
        Any iterable of labels. Each is coerced to a string, so numeric and
        categorical labels key consistently.

    Returns
    -------
    dict
        Label to count, ordered by label so the output is stable across runs
        and safe to diff.

    See Also
    --------
    per_class_report : Per-class performance, once you have predictions.
    """
    counter = Counter(str(item) for item in values)
    return {key: int(counter[key]) for key in sorted(counter)}


def classification_metrics(
    y_true: Any,
    y_pred: Any,
    *,
    probabilities: np.ndarray | None = None,
    classes: tuple[Any, ...] = (),
) -> dict[str, float]:
    """Compute the headline metrics for single-label classification.

    Always returns six metrics from the predicted labels alone, and adds
    probability-based ones when calibrated scores are available.

    The six are chosen to be hard to misread together. Accuracy alone hides an
    ignored minority class; balanced accuracy averages recall across classes,
    so it collapses when one is never predicted. Macro F1 weights every class
    equally while weighted F1 weights by frequency: a large gap between them
    is itself the finding, telling you performance is concentrated in the
    common classes.

    Parameters
    ----------
    y_true:
        The known labels.
    y_pred:
        The predicted labels, aligned with ``y_true``. Both are coerced to
        strings, so label dtype does not have to match.
    probabilities:
        Predicted class probabilities as an array of shape
        ``(n_samples, n_classes)``, if the head produces them.
    classes:
        The class labels in the same column order as ``probabilities``.

    Returns
    -------
    dict
        Accuracy, balanced accuracy, macro and weighted F1, macro precision and
        macro recall. Plus log loss and ROC AUC when probabilities were
        supplied and are usable.

    Notes
    -----
    **The probability metrics are added opportunistically, never forced.** They
    are skipped silently when the matrix shape does not match the class list,
    when the true labels include something outside it, or when a class is
    missing from this partition: all situations where scikit-learn would
    either raise or return a number that means something other than what the
    name suggests. Their absence from the returned dict is the signal that they
    could not be computed honestly.

    **Log loss punishes confident mistakes**, which makes it the metric that
    notices an overconfident text classifier when accuracy does not.

    See Also
    --------
    per_class_report : The per-class breakdown behind these averages.
    confusion_rows : Which classes are being confused for which.
    """
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    true_list = [str(item) for item in y_true]
    pred_list = [str(item) for item in y_pred]
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(true_list, pred_list)),
        "balanced_accuracy": float(balanced_accuracy_score(true_list, pred_list)),
        "f1_macro": float(f1_score(true_list, pred_list, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(true_list, pred_list, average="weighted", zero_division=0)
        ),
        "precision_macro": float(
            precision_score(true_list, pred_list, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(true_list, pred_list, average="macro", zero_division=0)
        ),
    }
    if probabilities is None or not len(classes):
        return metrics

    labels = [str(item) for item in classes]
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(labels):
        return metrics
    observed = set(true_list)
    if not observed <= set(labels):
        return metrics
    try:
        metrics["log_loss"] = float(log_loss(true_list, matrix, labels=labels))
    except ValueError:
        # Optional metric: omit when probabilities are degenerate or labels mismatch.
        logger.debug(
            "nlp: log_loss unavailable for this prediction matrix",
            exc_info=True,
        )
    if len(labels) == 2 and len(observed) == 2:
        positive = matrix[:, 1]
        try:
            metrics["roc_auc"] = float(
                roc_auc_score([1 if item == labels[1] else 0 for item in true_list], positive)
            )
        except ValueError:
            # Optional binary AUC: omit when a class is absent or scores are invalid.
            logger.debug(
                "nlp: binary roc_auc unavailable for this prediction matrix",
                exc_info=True,
            )
    elif len(labels) > 2 and len(observed) == len(labels):
        try:
            metrics["roc_auc"] = float(
                roc_auc_score(
                    true_list,
                    matrix,
                    multi_class="ovr",
                    average="macro",
                    labels=labels,
                )
            )
        except ValueError:
            # Optional multiclass AUC: omit when OvR cannot be computed.
            logger.debug(
                "nlp: multiclass roc_auc unavailable for this prediction matrix",
                exc_info=True,
            )
    return metrics


def per_class_report(
    y_true: Any,
    y_pred: Any,
    classes: tuple[Any, ...],
) -> dict[str, dict[str, float]]:
    """Break performance down by class, which is usually where the story is.

    Overall metrics average away the thing you most need to know. A model can
    look strong while being useless on the one category you built it for, and
    only the per-class view shows that.

    Read precision and recall as a pair. Low recall with high precision means
    the model rarely predicts this class but is right when it does: it is too
    cautious. The reverse means it over-predicts the class. And always check
    support: precision of 1.0 on a class with three documents is noise, not
    performance.

    Parameters
    ----------
    y_true:
        The known labels.
    y_pred:
        The predicted labels, aligned with ``y_true``.
    classes:
        Which classes to report, and in what order. Include classes absent from
        the predictions: they score zero, and their absence is exactly the
        finding.

    Returns
    -------
    dict
        Class label to a dict of ``precision``, ``recall``, ``f1``, and
        ``support``, where support is how many documents genuinely belong to
        that class.

    See Also
    --------
    classification_metrics : The averaged headline numbers.
    """
    from sklearn.metrics import precision_recall_fscore_support

    labels = [str(item) for item in classes]
    precision, recall, f1, support = precision_recall_fscore_support(
        [str(item) for item in y_true],
        [str(item) for item in y_pred],
        labels=labels,
        zero_division=0,
    )
    return {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": float(support[index]),
        }
        for index, label in enumerate(labels)
    }


def confusion_rows(
    y_true: Any,
    y_pred: Any,
    classes: tuple[Any, ...],
) -> tuple[tuple[int, ...], ...]:
    """Build the confusion matrix: what got predicted as what.

    Row ``i``, column ``j`` counts documents whose true class is ``classes[i]``
    and whose predicted class is ``classes[j]``. The diagonal is correct
    predictions; everything off it is a specific, nameable mistake.

    This matters more for text than for most tabular problems, because text
    classifiers fail in structured ways. Two categories that share vocabulary
    get conflated with each other and with nothing else, which shows up as a
    single hot off-diagonal cell: and points directly at either a labelling
    boundary that is genuinely fuzzy or two categories that should be merged.

    Parameters
    ----------
    y_true:
        The known labels.
    y_pred:
        The predicted labels, aligned with ``y_true``.
    classes:
        The class order for both axes. The caller supplies it so the matrix
        lines up with :func:`per_class_report`.

    Returns
    -------
    tuple of tuple of int
        Row-major counts, in the given class order.

    See Also
    --------
    per_class_report : Precision and recall derived from these counts.
    """
    from sklearn.metrics import confusion_matrix

    labels = [str(item) for item in classes]
    matrix = confusion_matrix(
        [str(item) for item in y_true],
        [str(item) for item in y_pred],
        labels=labels,
    )
    return tuple(tuple(int(value) for value in row) for row in matrix)


def resolve_holdout_partition(
    split_plan: SplitPlan | None,
    partition: PartitionOrAll,
) -> PartitionOrAll:
    """Pick a usable holdout partition when validation may not exist.

    Operations default to evaluating on validation, but a two-way split has no
    validation partition. Rather than failing on a reasonable default, this
    falls through to test.

    Parameters
    ----------
    split_plan:
        The split to inspect. ``None`` passes the request through unchanged.
    partition:
        The requested partition.

    Returns
    -------
    PartitionOrAll
        ``'test'`` when validation was asked for but is empty; otherwise the
        request unchanged.

    Notes
    -----
    Only the default is redirected. An explicit request for any other partition
    is honoured, and asking for validation on a three-way split gets validation.

    Be aware of what the fallback costs: repeatedly evaluating on test while
    tuning erodes it as a holdout. On a two-way split, treat every look as one
    of a small budget.
    """
    if (
        partition == "validation"
        and split_plan is not None
        and not split_plan.validation_indices
    ):
        return "test"
    return partition


def token_stats(documents: list[str], normalize_plan: Any) -> dict[str, float]:
    """Summarise how long the documents are, measured in tokens.

    Token length is the number that decides whether bag-of-n-grams will work at
    all. Documents averaging two or three tokens produce a matrix so sparse
    that metrics swing wildly between splits, and n-grams have almost nothing
    to bind together. Documents averaging hundreds tend to be easy for this
    kind of model.

    Parameters
    ----------
    documents:
        The documents to measure.
    normalize_plan:
        The plan used to tokenise. Counts are of surviving tokens, so stopword
        removal and length filters are reflected: this measures what the model
        will actually see, not what the raw text contains.

    Returns
    -------
    dict
        ``mean``, ``median``, ``p95``, and ``max`` token counts. All zero for
        an empty list.

    Notes
    -----
    Compare mean against median. A mean far above the median means a few very
    long documents dominate, which affects both memory and whether truncation
    limits on the transformer backends will bite.

    See Also
    --------
    char_stats : The same summary in characters, needing no tokenisation.
    """
    from buildml.nlp.normalize import tokenize_document

    counts = [len(tokenize_document(item, normalize_plan)) for item in documents]
    if not counts:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    array = np.asarray(counts, dtype=float)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def char_stats(documents: list[str]) -> dict[str, float]:
    """Summarise how long the documents are, measured in characters.

    The cheap sibling of :func:`token_stats`: no tokenisation, no plan, and
    therefore usable during profiling before any normalisation has been
    decided. It also sees what tokenising hides: a document that is entirely
    punctuation or markup has characters but no tokens.

    Parameters
    ----------
    documents:
        The documents to measure. Values are coerced to strings.

    Returns
    -------
    dict
        ``mean``, ``median``, ``p95``, and ``max`` character counts. All zero
        for an empty list.

    See Also
    --------
    token_stats : Lengths in tokens, which is what the model actually sees.
    """
    counts = [len(str(item)) for item in documents]
    if not counts:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    array = np.asarray(counts, dtype=float)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


__all__ = [
    "HOLDOUT_PARTITIONS",
    "PartitionOrAll",
    "candidate_text_columns",
    "char_stats",
    "class_counts",
    "classification_metrics",
    "confusion_rows",
    "documents_for",
    "empty_document_rate",
    "frame_for",
    "per_class_report",
    "resolve_holdout_partition",
    "resolve_text_column",
    "targets_for",
    "token_stats",
]
