"""Measure how well a text classifier does on documents it was not trained on.

The only number that means anything. A text classifier's training score is
close to useless: with tens of thousands of features and a few thousand
documents, a linear model can very nearly memorise the training set, so an
in-sample accuracy of 0.99 is the normal result rather than a good one.

Evaluation here reports overall metrics, a per-class breakdown, and the
confusion matrix. The per-class view usually matters more than the headline:
overall accuracy on an imbalanced corpus is dominated by the common category,
and a model that never once predicts your rare class can still score 0.94.
"""

from __future__ import annotations

import numpy as np

from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.nlp.features import (
    PartitionOrAll,
    classification_metrics,
    confusion_rows,
    documents_for,
    per_class_report,
    targets_for,
)
from buildml.nlp.predict import predict_documents
from buildml.nlp.results import NlpEvalResult, NlpTextPlan
from buildml.nlp.vectorize import oov_token_rate


def evaluate_text_classifier(
    dataset: Dataset,
    plan: NlpTextPlan,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll = "validation",
) -> NlpEvalResult:
    """Score a holdout partition against its known labels.

    Predicts every document in the partition and compares against the truth,
    without updating the vocabulary, the weights, or the head.

    Parameters
    ----------
    dataset:
        The dataset holding the documents and their labels.
    plan:
        A fitted plan from :func:`~buildml.nlp.fit.fit_text_classifier`, which
        supplies both the text and target column names.
    split_plan:
        The split defining partitions.
    partition:
        Which rows to evaluate. Defaults to ``'validation'``, which is the
        partition to iterate against. Keep ``'test'`` for a single final
        measurement — every time you look at it and change something in
        response, it becomes a little less of a holdout.

    Returns
    -------
    ~buildml.nlp.results.NlpEvalResult
        Overall metrics, per-class precision, recall and F1, the confusion
        matrix, the out-of-vocabulary rate, and any warnings.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        The plan is incomplete, a required column is missing, or the partition
        is empty.

    Notes
    -----
    **Read the confusion matrix, not just the metrics.** It tells you *which*
    categories the model conflates, and text classifiers usually fail in a
    structured way — two categories that genuinely share vocabulary — rather
    than uniformly. That is actionable in a way that a single number is not.

    **Labels present here but absent from training are flagged.** Those rows
    cannot be predicted correctly by construction, since the head has no output
    for a class it never saw, and they drag every metric down for a reason that
    has nothing to do with model quality. The warning names them.

    **A high out-of-vocabulary rate undermines the whole measurement.** If
    holdout documents are largely unseen words, this score is telling you about
    the model's priors more than its learning.

    Examples
    --------
    >>> result = evaluate_text_classifier(dataset, plan, split_plan)  # doctest: +SKIP
    >>> result.metrics["macro_f1"], result.per_class  # doctest: +SKIP

    See Also
    --------
    buildml.nlp.predict.predict_text : Predictions without labels to check against.
    buildml.nlp.interpret.interpret_text_prediction : Why one document was misread.
    """
    documents, frame = documents_for(
        dataset,
        split_plan,
        partition,
        plan.text_column,
        operation="evaluate_text_classifier",
    )
    y_true = targets_for(
        frame, plan.target_column, operation="evaluate_text_classifier"
    )
    predictions, probabilities = predict_documents(
        plan, documents, return_probabilities=True
    )

    proba_matrix = (
        np.asarray(probabilities, dtype=float) if probabilities else None
    )
    metrics = classification_metrics(
        y_true,
        predictions,
        probabilities=proba_matrix,
        classes=plan.classes_,
    )
    warnings: list[str] = []
    unseen_labels = sorted({str(item) for item in y_true} - set(plan.classes_))
    if unseen_labels:
        warnings.append(
            f"Partition {partition!r} contains class label(s) absent from train: "
            f"{unseen_labels}. Those rows can never be predicted correctly."
        )
    unseen_tokens = oov_token_rate(documents, plan.vectorizer_, plan.normalize_plan)
    if unseen_tokens is not None and unseen_tokens > 0.35:
        warnings.append(
            f"{unseen_tokens:.1%} of holdout tokens are outside the train "
            "vocabulary; consider a character analyzer or a larger train split."
        )

    classes = tuple(sorted(set(plan.classes_) | set(unseen_labels)))
    return NlpEvalResult(
        partition=str(partition),
        task=plan.task,
        n_rows=len(documents),
        metrics=metrics,
        per_class=per_class_report(y_true, predictions, classes),
        confusion=confusion_rows(y_true, predictions, classes),
        classes=classes,
        oov_rate=unseen_tokens,
        disclosures=(
            "Holdout evaluation only — the representation, vocabulary, and head "
            "were not updated from this partition.",
            "train_score on the fit result is in-sample and is not comparable to "
            "these holdout metrics.",
        ),
        warnings=tuple(warnings),
    )


__all__ = ["evaluate_text_classifier"]
