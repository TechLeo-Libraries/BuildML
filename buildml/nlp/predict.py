"""Assign labels to documents using an already-fitted text plan.

Scoring is strictly transform-only. The vocabulary, the IDF weights, and the
head all come from the training fit and are not updated by anything seen here :
which is what makes a holdout number honest and what makes production scoring
match the notebook.

The consequence worth understanding is that new documents will contain words the
model has never seen. Those words contribute nothing: they are not in the
vocabulary, so they have no weight, and the prediction is made from whatever
remains. A document made entirely of unseen words falls back to the head's class
prior: it gets a confident-looking answer derived from no evidence at all. The
out-of-vocabulary rate is reported for exactly this reason, and a high one means
the documents you are scoring are not drawn from the same population you trained
on.
"""

from __future__ import annotations

import numpy as np

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.nlp.features import (
    PartitionOrAll,
    documents_for,
    empty_document_rate,
)
from buildml.nlp.results import NlpPredictResult, NlpTextPlan
from buildml.nlp.vectorize import oov_token_rate


def transform_documents(plan: NlpTextPlan, documents: list[str]):
    """Turn raw strings into the numeric matrix the fitted head expects.

    The low-level step underneath :func:`predict_documents`, exposed because
    the vectors are sometimes useful on their own: for clustering documents,
    for a nearest-neighbour lookup, or for feeding another model.

    Parameters
    ----------
    plan:
        A fitted plan from
        :func:`~buildml.nlp.fit.fit_text_classifier`, or one restored from a
        saved bundle.
    documents:
        Raw document strings. Normalisation is applied here, so pass the text
        as it arrives rather than pre-processing it yourself: doing that twice
        gives a representation the head was not trained on.

    Returns
    -------
    Sparse matrix or ~numpy.ndarray
        One row per document, with the column layout fixed at fit time. Sparse
        for the bag-of-n-grams backends, dense for the embedding backends.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        The plan carries no fitted representation, which happens when a bundle
        was saved without its vectorizer.

    See Also
    --------
    predict_documents : Transform and classify in one step.
    """
    if plan.vectorizer_ is None:
        raise ValidationError(
            "The NLP plan has no fitted representation. Refit with "
            "fit_text_classifier or reload a complete buildml.nlp_bundle.v1."
        )
    return plan.vectorizer_.transform(documents)


def predict_documents(
    plan: NlpTextPlan,
    documents: list[str],
    *,
    return_probabilities: bool = True,
) -> tuple[tuple[str, ...], tuple[tuple[float, ...], ...]]:
    """Classify a list of raw document strings.

    The direct entry point when your documents are not in a dataset: scoring a
    request in a web service, or trying a handful of strings by hand.

    Parameters
    ----------
    plan:
        A fitted plan from :func:`~buildml.nlp.fit.fit_text_classifier`.
    documents:
        Raw strings, one per document.
    return_probabilities:
        Also return per-class probabilities. Silently produces nothing when the
        head does not support them: ``'linear_svm'`` has no probability
        output. Check ``plan.supports_proba`` if you need to know in advance.

    Returns
    -------
    tuple
        ``(predictions, probabilities)``. Predictions are class labels as
        strings, one per document. Probabilities is a tuple of rows aligned to
        ``plan.classes_``, or an empty tuple when unavailable.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        The plan carries no fitted head or no fitted representation.

    Notes
    -----
    Probabilities from a linear text classifier are usually poorly calibrated:
    high-dimensional sparse features push them toward 0 and 1, so a 0.97 does
    not mean the model is right 97% of the time. Treat them as a ranking unless
    you have calibrated them against a holdout.

    See Also
    --------
    predict_text : The dataset-level entry point, which also reports coverage.
    """
    if plan.estimator_ is None:
        raise ValidationError(
            "The NLP plan has no fitted head. Refit with fit_text_classifier."
        )
    matrix = transform_documents(plan, documents)
    predictions = tuple(str(item) for item in plan.estimator_.predict(matrix))
    probabilities: tuple[tuple[float, ...], ...] = ()
    if return_probabilities and plan.supports_proba:
        raw = plan.estimator_.predict_proba(matrix)
        array = np.asarray(raw, dtype=float)
        probabilities = tuple(tuple(float(value) for value in row) for row in array)
    return predictions, probabilities


def predict_text(
    dataset: Dataset,
    plan: NlpTextPlan,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll = "test",
    return_probabilities: bool = True,
) -> NlpPredictResult:
    """Score a dataset partition and report how well the model covers it.

    The dataset-level entry point. Beyond the predictions themselves it
    measures how much of the incoming text the model can actually see :
    the out-of-vocabulary rate and the blank-document rate: which is the
    difference between a prediction you can act on and one that came from the
    class prior.

    Parameters
    ----------
    dataset:
        The dataset holding the documents.
    plan:
        A fitted plan from :func:`~buildml.nlp.fit.fit_text_classifier`. It
        supplies the text column name, so the dataset need not match the one
        used at fit time.
    split_plan:
        The split defining partitions. Required unless ``partition`` is
        ``'all'``.
    partition:
        Which rows to score: ``'train'``, ``'validation'``, ``'test'``, or
        ``'all'``. Scoring ``'train'`` shows in-sample behaviour and will look
        better than the model really is.
    return_probabilities:
        Include per-class probabilities where the head supports them.

    Returns
    -------
    ~buildml.nlp.results.NlpPredictResult
        Predictions, optional probabilities, the class list, and the coverage
        diagnostics described above.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        The plan is incomplete, the text column is missing from the dataset, or
        the partition is empty.

    Notes
    -----
    **Nothing is refitted and no session state changes.** The same plan can
    score any number of partitions or datasets.

    **Watch the out-of-vocabulary rate.** Above roughly a third of tokens, a
    warning is recorded, and the predictions should be read as extrapolation.
    The usual causes are a training corpus too small to cover the language, a
    genuine shift in what people are writing about, or a different text source
    entirely.

    Examples
    --------
    >>> result = predict_text(dataset, plan, split_plan, partition="test")  # doctest: +SKIP
    >>> result.predictions[:3], result.oov_rate  # doctest: +SKIP

    See Also
    --------
    buildml.nlp.evaluate.evaluate_text_classifier : Adds metrics against known labels.
    """
    documents, _frame = documents_for(
        dataset, split_plan, partition, plan.text_column, operation="predict_text"
    )
    predictions, probabilities = predict_documents(
        plan, documents, return_probabilities=return_probabilities
    )

    warnings: list[str] = []
    unseen = oov_token_rate(documents, plan.vectorizer_, plan.normalize_plan)
    if unseen is not None and unseen > 0.35:
        warnings.append(
            f"{unseen:.1%} of tokens in partition {partition!r} are outside the "
            "train vocabulary; treat these predictions as extrapolation."
        )
    blank_rate = empty_document_rate(documents)
    if blank_rate > 0.0:
        warnings.append(
            f"{blank_rate:.1%} of scored documents are blank; those rows fall back "
            "to the head's prior."
        )
    if return_probabilities and not plan.supports_proba:
        warnings.append(
            f"estimator='{plan.estimator}' has no calibrated predict_proba; "
            "probabilities were not produced."
        )

    return NlpPredictResult(
        partition=str(partition),
        task=plan.task,
        n_rows=len(documents),
        predictions=predictions,
        probabilities=probabilities,
        classes=plan.classes_,
        oov_rate=unseen,
        empty_document_rate=blank_rate,
        disclosures=(
            "Scoring is transform-only: the representation, vocabulary, and head "
            "come from the train fit.",
            *plan.disclosures[:2],
        ),
        warnings=tuple(warnings),
    )


__all__ = ["predict_documents", "predict_text", "transform_documents"]
