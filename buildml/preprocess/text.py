"""Turn free text into numeric columns a classical model can use.

A review, a description, a support ticket: none of it means anything to a
gradient booster. Vectorising converts each document into a row of numbers, one
per term, so text can sit alongside your other features in the same frame.

All three methods here are "bag of words": they count what appears and discard
the order it appeared in. "The film was good, not bad" and "the film was bad,
not good" produce identical features. That is a real limitation, and it is the
reason these methods lose to transformer models on tasks where nuance matters.
What they offer instead is speed, transparency: you can read which word drove
a prediction: and the fact that they work on a few thousand rows, where a
fine-tuned transformer would not.

**Count** records how many times each term occurs. Simple, and the raw numbers
mean something, but common words dominate purely by being common.

**TF-IDF** weighs each count down by how many documents the term appears in, so
a word appearing in every document contributes almost nothing while a
distinctive one stands out. It is the default and usually the best of the three.

**Hashing** maps terms into a fixed number of buckets with a hash function
instead of building a vocabulary. It uses constant memory regardless of corpus
size and handles unseen words without any special case, but two different words
can collide into the same bucket, and you cannot recover which word a feature
came from.

Count and TF-IDF learn their vocabulary from training documents only: a term
that appears only in test documents has no column, which is correct, since the
model could not have learned anything about it. For dense embeddings and
transformer models, see :mod:`buildml.nlp`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition, frame_for_partition
from buildml.explain.schemas import (
    Action,
    ActionPriority,
    Evidence,
    EvidenceKind,
    Finding,
    FindingSeverity,
    Recommendation,
)
from buildml.ingest.detect import schema_from_dataframe
from buildml.preprocess.columns import resolve_transform_columns
from buildml.preprocess.result import PreprocessResult

TextMethod = Literal["count", "tfidf", "hashing"]


@dataclass(slots=True)
class TextFeaturePlan:
    """The vocabulary learned from training documents, and the columns it produces.

    Fixing the vocabulary is what makes text features reproducible. A model
    trained with ``review_excellent`` in column 41 needs that same term in that
    same position forever after; re-deriving the vocabulary from a new batch
    would shuffle every column.

    Attributes
    ----------
    columns:
        The source text columns this plan vectorises.
    method:
        ``'count'``, ``'tfidf'``, or ``'hashing'``.
    max_features:
        The cap on vocabulary size per column, or ``None`` for uncapped.
    ngram_range:
        The ``(min_n, max_n)`` term lengths that were extracted.
    feature_names_:
        Every output column, in order, prefixed by its source column. This is
        the contract with the model.
    vectorizers_:
        The fitted scikit-learn vectorizer per column. Count and TF-IDF
        vectorizers carry the vocabulary and serialise with joblib for
        checkpoint and pipeline replay; a hashing vectorizer is stateless and
        needs nothing stored.
    n_features_per_column_:
        How many features each source column produced. Worth checking: this is
        where a frame unexpectedly grows by thousands of columns.
    drop_input_columns:
        Whether the original text columns were removed after vectorising.
    """

    columns: tuple[str, ...]
    method: TextMethod
    max_features: int | None
    ngram_range: tuple[int, int]
    feature_names_: tuple[str, ...]
    vectorizers_: dict[str, Any] = field(repr=False)
    n_features_per_column_: dict[str, int] = field(default_factory=dict)
    drop_input_columns: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return the plan's settings and output layout as JSON-safe values.

        The fitted vectorizers are omitted, since they do not serialise to
        JSON: save a pipeline to round-trip those.

        Returns
        -------
        dict
            Keys ``columns``, ``method``, ``max_features``, ``ngram_range``,
            ``feature_names_``, ``n_features_per_column_``, and
            ``drop_input_columns``.
        """
        return {
            "columns": list(self.columns),
            "method": self.method,
            "max_features": self.max_features,
            "ngram_range": list(self.ngram_range),
            "feature_names_": list(self.feature_names_),
            "n_features_per_column_": dict(self.n_features_per_column_),
            "drop_input_columns": self.drop_input_columns,
        }


def fit_text_features(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    columns: list[str] | None = None,
    method: TextMethod = "tfidf",
    max_features: int | None = 128,
    ngram_range: tuple[int, int] = (1, 1),
    drop_input_columns: bool = True,
) -> TextFeaturePlan:
    """Learn a term vocabulary from the training documents.

    Nothing is transformed here: pass the plan to
    :func:`transform_text_features` to apply it.

    Parameters
    ----------
    dataset:
        The full dataset. Only rows the split assigns to ``train`` are read.
    split_plan:
        The split defining the training documents. Required, because a
        vocabulary built from all documents tells the model which words the
        test set contains.
    columns:
        Which text columns to vectorise. Defaults to the text-typed
        ``feature`` columns. Each column gets its own independent vocabulary,
        so a term means something different depending on which field it came
        from: which is usually right, since "urgent" in a subject line is not
        "urgent" in a signature.
    method:
        ``'tfidf'`` (the default), ``'count'``, or ``'hashing'``. See the
        module docstring for the trade-offs.
    max_features:
        Keep only this many terms per column, chosen by frequency. This is the
        main defence against a frame that explodes: real text easily yields
        tens of thousands of distinct terms, most appearing once. The default
        of 128 is deliberately conservative: raise it into the low thousands
        when text is your primary signal. ``None`` keeps everything, which is
        rarely wise. Ignored by hashing, which is bounded by construction.
    ngram_range:
        The term lengths to extract, as ``(min_n, max_n)``. ``(1, 1)`` takes
        single words. ``(1, 2)`` adds adjacent pairs, which recovers a little
        of the word order that bag-of-words discards: "not good" becomes its
        own term: at a large cost in vocabulary size. Going beyond pairs
        rarely pays for itself.
    drop_input_columns:
        Remove the source text after vectorising. Usually correct, since the
        raw strings cannot be modelled. Keep them when you want to read the
        original text during error analysis.

    Returns
    -------
    TextFeaturePlan
        The learned vocabulary and output layout, ready to apply.

    Raises
    ------
    ~buildml.core.errors.LeakageError
        No split plan was supplied.
    ~buildml.core.errors.ValidationError
        ``method`` is unrecognised, ``max_features`` is below 1,
        ``ngram_range`` is malformed, or no text columns resolved.

    Notes
    -----
    **Check the resulting width.** Vectorising three text columns at 128
    features each adds 384 columns. Multiply by ``ngram_range`` and it grows
    quickly. ``n_features_per_column_`` on the returned plan tells you exactly
    what you are about to add.

    **The output is dense.** Text features are naturally sparse: most
    documents contain almost none of the vocabulary: but they are materialised
    as ordinary columns here so they can join the rest of the frame. Budget
    memory accordingly, and consider dimensionality reduction afterwards via
    :mod:`buildml.preprocess.reduce`.

    Examples
    --------
    >>> plan = fit_text_features(  # doctest: +SKIP
    ...     dataset, split_plan, columns=["review"], max_features=500
    ... )
    >>> plan.n_features_per_column_["review"]  # doctest: +SKIP
    500

    See Also
    --------
    transform_text_features : Applies the plan produced here.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None
    if method not in {"count", "tfidf", "hashing"}:
        raise ValidationError(f"Unsupported text feature method '{method}'")
    if max_features is not None and max_features < 1:
        raise ValidationError("max_features must be >= 1 when provided")
    if len(ngram_range) != 2 or ngram_range[0] < 1 or ngram_range[1] < ngram_range[0]:
        raise ValidationError("ngram_range must be a (min_n, max_n) pair with 1 <= min_n <= max_n")

    train = frame_for_partition(dataset, split_plan, "train")
    cols = _resolve_text_columns(dataset, train, columns)
    vectorizers: dict[str, Any] = {}
    feature_names: list[str] = []
    n_per_col: dict[str, int] = {}

    for column in cols:
        documents = _as_text(train[column])
        vectorizer = _build_vectorizer(method, max_features=max_features, ngram_range=ngram_range)
        matrix = vectorizer.fit_transform(documents)
        n_features = int(matrix.shape[1])
        n_per_col[column] = n_features
        names = _feature_names_for_column(column, vectorizer, n_features, method)
        feature_names.extend(names)
        vectorizers[column] = vectorizer

    return TextFeaturePlan(
        columns=tuple(cols),
        method=method,
        max_features=max_features,
        ngram_range=ngram_range,
        feature_names_=tuple(feature_names),
        vectorizers_=vectorizers,
        n_features_per_column_=n_per_col,
        drop_input_columns=drop_input_columns,
    )


def transform_text_features(
    dataset: Dataset,
    plan: TextFeaturePlan,
) -> tuple[Dataset, PreprocessResult]:
    """Convert text to numeric columns using an already-learned vocabulary.

    Runs across all partitions. Terms absent from the training vocabulary are
    simply not counted, which is the honest behaviour: the model has no
    parameter for a word it never saw.

    Parameters
    ----------
    dataset:
        The dataset to vectorise. Every column the plan names must be present.
    plan:
        A plan from :func:`fit_text_features`, or one restored from a saved
        pipeline.

    Returns
    -------
    tuple of (~buildml.data.dataset.Dataset, ~buildml.preprocess.result.PreprocessResult)
        The dataset with text replaced by numeric features, and a narrated
        record covering how many columns were added and how sparse they are.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        A column the plan expects is missing from the dataset.

    Notes
    -----
    **Out-of-vocabulary text yields an all-zero row.** A document made entirely
    of unseen terms produces zeros across every text feature, and the model
    will fall back to whatever it predicts in the absence of evidence. A high
    rate of this means the training text and the incoming text are drawn from
    different populations.

    **Missing values are treated as empty documents** rather than propagating
    as gaps, so they also produce zeros.

    See Also
    --------
    fit_text_features : Produces the plan this consumes.
    """
    missing = [c for c in plan.columns if c not in dataset.columns]
    if missing:
        raise ValidationError(f"Text plan columns missing from dataset: {missing}")

    frame = dataset._ensure_pandas().copy()
    roles = dict(dataset.roles)
    blocks: list[pd.DataFrame] = []
    for column in plan.columns:
        documents = _as_text(frame[column])
        matrix = plan.vectorizers_[column].transform(documents)
        dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
        n_features = plan.n_features_per_column_[column]
        names = _feature_names_for_column(
            column,
            plan.vectorizers_[column],
            n_features,
            plan.method,
        )
        block = pd.DataFrame(dense, columns=names, index=frame.index)
        blocks.append(block)
        roles.pop(column, None)
        for name in names:
            roles[name] = ColumnRole.FEATURE

    feature_frame = pd.concat(blocks, axis=1)
    if list(feature_frame.columns) != list(plan.feature_names_):
        # Keep a stable contract even if hashing name helpers change.
        feature_frame.columns = list(plan.feature_names_)
        for name in plan.feature_names_:
            roles[name] = ColumnRole.FEATURE

    if plan.drop_input_columns:
        frame = frame.drop(columns=list(plan.columns))
    out = pd.concat([frame, feature_frame], axis=1)
    new_dataset = Dataset.from_transformed(
        dataset,
        out,
        schema=schema_from_dataframe(out),
        roles=roles,
    )
    return new_dataset, _build_result(plan)


def _build_vectorizer(
    method: TextMethod,
    *,
    max_features: int | None,
    ngram_range: tuple[int, int],
) -> Any:
    common = {
        "ngram_range": ngram_range,
        "lowercase": True,
        "dtype": np.float64,
    }
    if method == "count":
        return CountVectorizer(max_features=max_features, **common)
    if method == "tfidf":
        return TfidfVectorizer(max_features=max_features, **common)
    # Hashing is stateless; max_features is the fixed output width.
    n_features = 128 if max_features is None else max_features
    return HashingVectorizer(n_features=n_features, alternate_sign=False, **common)


def _feature_names_for_column(
    column: str,
    vectorizer: Any,
    n_features: int,
    method: TextMethod,
) -> list[str]:
    if method != "hashing" and hasattr(vectorizer, "get_feature_names_out"):
        raw = [str(name) for name in vectorizer.get_feature_names_out()]
        return [f"{column}__{name}" for name in raw]
    return [f"{column}__hash_{i}" for i in range(n_features)]


def _as_text(series: pd.Series) -> list[str]:
    return series.astype("string").fillna("").tolist()


def _resolve_text_columns(
    dataset: Dataset,
    train: pd.DataFrame,
    columns: list[str] | None,
) -> list[str]:
    names = resolve_transform_columns(
        dataset,
        train,
        columns,
        kind="text",
        empty_message=(
            "No text/object feature columns available for text_features. "
            "Pass columns=... explicitly."
        ),
    )
    for column in names:
        if pd.api.types.is_numeric_dtype(train[column]):
            raise ValidationError(
                f"Column '{column}' is numeric; text_features expects string-like values."
            )
    return names


def _build_result(plan: TextFeaturePlan) -> PreprocessResult:
    evidence = [
        Evidence(
            key="text_features.width",
            kind=EvidenceKind.METRIC,
            summary="Train-fitted text feature width by source column.",
            value={
                "method": plan.method,
                "n_features_per_column": dict(plan.n_features_per_column_),
                "total_features": len(plan.feature_names_),
                "max_features": plan.max_features,
                "ngram_range": list(plan.ngram_range),
            },
            source="train.text_features",
            limitations=(
                "Bag features are bag-of-n-grams style; they ignore word order beyond n-grams.",
            ),
        )
    ]
    findings = [
        Finding(
            key="text_features.applied",
            title="Text features fitted on train",
            detail=(
                f"Method '{plan.method}' expanded {len(plan.columns)} text column(s) "
                f"into {len(plan.feature_names_)} numeric feature(s)."
            ),
            severity=FindingSeverity.INFO,
            evidence=tuple(evidence),
            affected_columns=plan.columns,
        )
    ]
    recommendations = [
        Recommendation(
            key="text_features.review-width",
            title="Review text feature width before scale-sensitive models",
            rationale=(
                "Wide sparse-style expansions can dominate linear models. Confirm "
                "max_features and holdout metrics before claiming improvement."
            ),
            priority=ActionPriority.NEXT,
            action=Action(
                key="text_features.eval-action",
                label="Session.evaluate(partition='validation')",
                operation="evaluate",
                parameters={"partition": "validation"},
            ),
            based_on=("text_features.applied",),
            caveats=(
                "Hashing collisions are irreversible; "
                "prefer TF-IDF when interpretability matters.",
            ),
        )
    ]
    return PreprocessResult(
        operation="text_features",
        plan=plan.to_dict(),
        evidence=evidence,
        findings=findings,
        interpretation=[
            f"Expanded {len(plan.columns)} text column(s) with method '{plan.method}'.",
            "Vectorizers were fitted on train documents only.",
        ],
        limitations=[
            "Missing text becomes empty strings before vectorization.",
            "Hashing has no invertible vocabulary; feature names are positional hashes.",
            "Dense materialization can be wide; keep max_features modest for tabular models.",
        ],
        recommendations=recommendations,
        methods=[
            f"method={plan.method}",
            f"max_features={plan.max_features}",
            f"ngram_range={plan.ngram_range}",
        ],
    )
