"""Token-level attributions for linear document classifiers.

For a linear head the decision score of a document is exactly
``bias + sum_j coef[class, j] * x[class, j]``, so per-token contributions are an
identity rather than an approximation. That is why this surface refuses
representations without an invertible vocabulary (hashing, sentence embeddings,
pooled transformers) instead of returning a plausible-looking guess.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.nlp.features import PartitionOrAll, documents_for
from buildml.nlp.predict import predict_documents, transform_documents
from buildml.nlp.results import NlpInterpretResult, NlpTextPlan, TokenAttribution

LINEAR_ESTIMATORS: tuple[str, ...] = ("logistic", "linear_svm", "sgd")
NAIVE_BAYES_ESTIMATORS: tuple[str, ...] = ("complement_nb", "multinomial_nb")


def _require_invertible(plan: NlpTextPlan) -> tuple[str, ...]:
    if plan.backend != "sklearn":
        raise ValidationError(
            f"Token attributions need an invertible vocabulary, but backend="
            f"'{plan.backend}' produces latent dense vectors. Refit with "
            "backend='sklearn' and vectorizer='tfidf' to interpret tokens."
        )
    if str(plan.vectorize.get("kind")) == "hashing":
        raise ValidationError(
            "vectorizer='hashing' has no invertible vocabulary, so tokens cannot "
            "be recovered from feature positions. Refit with vectorizer='tfidf' "
            "or 'count' to interpret tokens."
        )
    if not plan.feature_names_:
        raise ValidationError(
            "The plan carries no feature names; refit with fit_text_classifier so "
            "the vocabulary is stored alongside the head."
        )
    return plan.feature_names_


def _coefficient_matrix(plan: NlpTextPlan) -> tuple[np.ndarray, str]:
    """Return ``(matrix[n_classes, n_features], method)`` for the fitted head."""
    head = plan.estimator_
    if head is None:
        raise ValidationError("The NLP plan has no fitted head.")
    if plan.estimator in LINEAR_ESTIMATORS:
        coef = np.asarray(head.coef_, dtype=float)
        if coef.ndim == 1:
            coef = coef.reshape(1, -1)
        if coef.shape[0] == 1 and len(plan.classes_) == 2:
            # Binary linear heads expose one row oriented at the positive class.
            coef = np.vstack([-coef[0], coef[0]])
        return coef, "linear-coefficient x feature-value"
    if plan.estimator in NAIVE_BAYES_ESTIMATORS:
        log_prob = np.asarray(head.feature_log_prob_, dtype=float)
        centred = log_prob - log_prob.mean(axis=0, keepdims=True)
        if plan.estimator == "complement_nb":
            # ComplementNB weights are already complement-oriented (higher means
            # more evidence *against* the class), so flip to keep the sign
            # convention "positive contribution supports the class".
            centred = -centred
        return centred, "centred naive-Bayes log-likelihood x feature-value"
    raise ValidationError(
        f"estimator='{plan.estimator}' does not expose per-feature weights."
    )


def _class_index(plan: NlpTextPlan, target_class: Any) -> int:
    labels = [str(item) for item in plan.classes_]
    if target_class is None:
        # Default to the last class so binary problems explain the positive class.
        return len(labels) - 1
    key = str(target_class)
    if key not in labels:
        raise ValidationError(
            f"target_class={target_class!r} is not one of the fitted classes "
            f"{labels}."
        )
    return labels.index(key)


def _row_attributions(
    row: np.ndarray | Any,
    weights: np.ndarray,
    names: tuple[str, ...],
    *,
    top_k: int,
) -> tuple[TokenAttribution, ...]:
    if sparse.issparse(row):
        coo = row.tocoo()
        indices = np.asarray(coo.col, dtype=int)
        values = np.asarray(coo.data, dtype=float)
    else:
        dense = np.asarray(row, dtype=float).ravel()
        indices = np.nonzero(dense)[0]
        values = dense[indices]
    if indices.size == 0:
        return ()
    contributions = weights[indices] * values
    order = np.argsort(-np.abs(contributions))[:top_k]
    out: list[TokenAttribution] = []
    for position in order:
        column = int(indices[position])
        token = names[column] if column < len(names) else f"feature_{column}"
        out.append(
            TokenAttribution(
                token=token,
                weight=float(weights[column]),
                value=float(values[position]),
                contribution=float(contributions[position]),
            )
        )
    return tuple(out)


def interpret_text_prediction(
    dataset: Dataset,
    plan: NlpTextPlan,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll = "test",
    target_class: Any = None,
    top_k: int = 12,
    max_documents: int = 10,
    include_global: bool = True,
) -> NlpInterpretResult:
    """Show which words drove each decision, and which words the model relies on.

    Produces two views. Per document, the tokens that pushed the prediction
    toward or away from a class, with the exact amount each contributed. Across
    the model, the vocabulary that carries the most weight overall — which is
    often where you discover that your classifier learned something
    embarrassing, like a ticket-routing model keying on the name of the agent
    who happened to handle every billing case.

    For a linear head these numbers are not estimates. The decision score is
    literally the bias plus the sum of each feature's weight times its value, so
    a token's contribution is a term in that sum. Nothing is being approximated
    or sampled, which is why this refuses to run rather than guess when the
    representation has no recoverable vocabulary.

    Parameters
    ----------
    dataset:
        The dataset holding the documents.
    plan:
        A fitted plan from :func:`~buildml.nlp.fit.fit_text_classifier`. It must
        use the sklearn backend with a count or TF-IDF vectorizer — see the
        notes.
    split_plan:
        The split defining partitions.
    partition:
        Which rows to explain.
    target_class:
        Which class the attributions are oriented toward. A positive
        contribution means the token pushed the document toward this class.
        Defaults to the last class, so a binary problem explains the positive
        one.
    top_k:
        How many tokens to report per document, taken by absolute
        contribution — so both the strongest evidence for and the strongest
        evidence against appear.
    max_documents:
        Cap on documents explained, since per-document output gets unwieldy
        fast. Set it to 0 to skip the per-document view and get only the global
        one.
    include_global:
        Also report the highest-weight vocabulary across the whole model, not
        just within these documents.

    Returns
    -------
    ~buildml.nlp.results.NlpInterpretResult
        Per-document attributions, the optional global vocabulary view, and the
        attribution method used, which differs between the linear and naive
        Bayes heads.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        ``top_k`` is below 1 or ``max_documents`` is negative; the backend is
        not sklearn; the vectorizer is ``'hashing'``; the plan stored no feature
        names; the head exposes no per-feature weights; or ``target_class`` is
        not one of the fitted classes.

    Notes
    -----
    **Why some representations are refused.** Hashing vectorizers map many
    tokens into one bucket with no way back, and sentence embeddings and pooled
    transformers produce dense latent dimensions that correspond to no
    particular word. In all three cases a token-level explanation would be
    fabricated. Refit with ``backend='sklearn'`` and ``vectorizer='tfidf'`` when
    you need this.

    **Contribution is weight times value, not weight alone.** A token with a
    large coefficient contributes nothing to a document that does not contain
    it. This is why the same model produces different explanations for
    different documents.

    **Naive Bayes attributions are centred log-likelihoods**, not coefficients,
    so they are comparable within a document but not on the same scale as the
    linear heads. The result records which method was used.

    **This explains the model, not the world.** A token with high weight is one
    the model relies on, which may be a genuine signal or an artefact of how the
    data was collected.

    Examples
    --------
    >>> result = interpret_text_prediction(  # doctest: +SKIP
    ...     dataset, plan, split_plan, top_k=8, max_documents=3
    ... )
    >>> first = result.document_attributions[0]  # doctest: +SKIP
    >>> [(item.token, round(item.contribution, 3)) for item in first[:3]]  # doctest: +SKIP

    See Also
    --------
    buildml.nlp.evaluate.evaluate_text_classifier : Where the model fails.
    buildml.nlp.keyphrases.extract_keyphrases : Salient terms without a model.
    """
    if top_k < 1:
        raise ValidationError("top_k must be >= 1.")
    if max_documents < 0:
        raise ValidationError("max_documents must be >= 0.")

    names = _require_invertible(plan)
    weights_matrix, method = _coefficient_matrix(plan)
    index = _class_index(plan, target_class)
    if index >= weights_matrix.shape[0]:
        raise ValidationError(
            "Fitted head exposes fewer weight rows than classes; refit the plan."
        )
    weights = weights_matrix[index]

    documents, frame = documents_for(
        dataset,
        split_plan,
        partition,
        plan.text_column,
        operation="interpret_text_prediction",
    )
    selected = documents[:max_documents] if max_documents else []
    row_labels = tuple(frame.index[: len(selected)])

    document_attributions: list[tuple[TokenAttribution, ...]] = []
    predictions: tuple[str, ...] = ()
    if selected:
        matrix = transform_documents(plan, selected)
        for position in range(len(selected)):
            row = matrix[position] if sparse.issparse(matrix) else matrix[position : position + 1]
            document_attributions.append(
                _row_attributions(row, weights, names, top_k=top_k)
            )
        predictions, _ = predict_documents(plan, selected, return_probabilities=False)

    global_top: dict[str, tuple[TokenAttribution, ...]] = {}
    if include_global:
        for class_position, label in enumerate(plan.classes_):
            if class_position >= weights_matrix.shape[0]:
                break
            row = weights_matrix[class_position]
            order = np.argsort(-row)[:top_k]
            global_top[str(label)] = tuple(
                TokenAttribution(
                    token=names[int(column)] if int(column) < len(names) else f"feature_{column}",
                    weight=float(row[int(column)]),
                    value=float("nan"),
                    contribution=float(row[int(column)]),
                )
                for column in order
            )

    warnings: list[str] = []
    if plan.estimator == "linear_svm":
        warnings.append(
            "LinearSVC contributions are margin contributions, not probabilities."
        )
    if plan.estimator in NAIVE_BAYES_ESTIMATORS:
        warnings.append(
            "Naive-Bayes attributions use centred log-likelihoods; they rank "
            "evidence but are not additive decision-function terms."
        )
    if max_documents and len(documents) > len(selected):
        warnings.append(
            f"Explained the first {len(selected)} of {len(documents)} documents "
            "in the partition (raise max_documents for more)."
        )

    return NlpInterpretResult(
        partition=str(partition),
        n_documents=len(selected),
        target_class=str(plan.classes_[index]),
        method=method,
        document_attributions=tuple(document_attributions),
        document_predictions=predictions,
        document_row_labels=row_labels,
        global_top_tokens=global_top,
        disclosures=(
            f"Attribution method: {method} for class "
            f"{str(plan.classes_[index])!r}.",
            "Per-document contributions are exact for linear heads; a positive "
            "contribution pushes the document toward the target class.",
            "Global top tokens rank the head's weights and ignore how often each "
            "token actually occurs.",
        ),
        warnings=tuple(warnings),
    )


__all__ = [
    "LINEAR_ESTIMATORS",
    "NAIVE_BAYES_ESTIMATORS",
    "interpret_text_prediction",
]
