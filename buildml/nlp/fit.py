"""Train a classifier that assigns one label to each document.

This is the supervised half of :mod:`buildml.nlp`: you have a column of text and
a column of labels, and you want a model that reads the first and predicts the
second. Support tickets into routing categories, reviews into star ratings,
emails into spam or not.

Two things happen in one call, and it helps to keep them separate in your head.
First a *representation* is fitted — the machinery that turns a string into
numbers. Second a *head* is fitted — an ordinary classifier trained on those
numbers. Both are learned from training documents only, and both are stored in
the returned plan so new documents can be put through the identical pipeline.

The representation is where text modelling usually goes wrong. A vocabulary
built from every document in the dataset knows which words appear in the test
set, and the IDF weights encode how rare they are there. That is leakage, and it
is invisible: nothing errors, the holdout score is simply too high. Fitting
train-only means holdout documents will contain words the model has never seen,
which is exactly the situation the model will face in production — so the
out-of-vocabulary rate gets reported rather than hidden.

What this module does *not* do: multi-label assignment, span or token-level
labelling, or fine-tuning a transformer's weights. The transformer backends here
are frozen feature extractors; the Torch text path owns fine-tuning.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition
from buildml.nlp.catalog import resolve_backend_estimator
from buildml.nlp.features import (
    class_counts,
    documents_for,
    empty_document_rate,
    resolve_text_column,
    targets_for,
    token_stats,
)
from buildml.nlp.normalize import build_normalize_plan
from buildml.nlp.results import NlpFitResult, NlpTextPlan
from buildml.nlp.types import (
    DEFAULT_NORMALIZE_STEPS,
    NlpVectorizeConfig,
    TextNormalizeConfig,
)
from buildml.nlp.vectorize import (
    build_document_vectorizer,
    feature_names_for,
    matrix_width,
    vocabulary_size,
)

MIN_TRAIN_DOCUMENTS = 4
MAX_STORED_FEATURE_NAMES = 200_000


def build_estimator(
    estimator: str,
    *,
    class_weight: str | None,
    C: float,
    alpha: float,
    random_state: int | None,
) -> tuple[Any, bool]:
    """Construct the classifier that sits on top of the text representation.

    The "head" is an ordinary tabular classifier — text modelling has already
    happened by the time it sees anything. What makes these five suitable is
    that they cope with the shape of text features: tens of thousands of
    columns, almost all zero for any given document.

    Parameters
    ----------
    estimator:
        Which head to build.

        ``'logistic'`` is the default and the safest choice. It gives
        well-behaved probabilities, and its coefficients are per-token weights,
        which is what makes :func:`~buildml.nlp.interpret.interpret_text_prediction`
        able to say exactly why a document was classified as it was.

        ``'linear_svm'`` often edges out logistic regression on accuracy for
        high-dimensional sparse text, but produces no probabilities at all —
        only a decision. Choose it when you need the label and not a confidence.

        ``'sgd'`` fits by stochastic gradient descent, so it scales to corpora
        too large to hold in memory at once. Slightly less accurate on small
        data than the batch solvers.

        ``'complement_nb'`` and ``'multinomial_nb'`` are naive Bayes variants:
        extremely fast, effective on small corpora, and requiring non-negative
        features, so they work with count and TF-IDF vectors but not with
        embeddings. Complement is the better of the two when classes are
        imbalanced, since it corrects the bias multinomial naive Bayes has
        toward the majority class.
    class_weight:
        ``'balanced'`` weights each class inversely to its frequency, so a rare
        category is not ignored in favour of overall accuracy. ``None`` treats
        every document equally. Not supported by the naive Bayes heads, which
        handle priors their own way.
    C:
        Inverse regularisation strength for ``'logistic'`` and ``'linear_svm'``.
        Lower values regularise harder, shrinking coefficients toward zero,
        which helps when you have many more features than documents — the usual
        situation with text. Raise it when the model is underfitting.
    alpha:
        Additive smoothing for the naive Bayes heads and the regularisation
        term for ``'sgd'``. For naive Bayes this is what stops a word seen in
        only one class from making that class impossible for every document
        lacking it.
    random_state:
        Seed for the heads with a stochastic component, so fits reproduce.

    Returns
    -------
    tuple
        ``(estimator, supports_predict_proba)`` — the unfitted head, and
        whether it can produce class probabilities. The flag is recorded on the
        plan so prediction can warn rather than fail when probabilities are
        requested from a head that has none.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        ``C`` is not positive, ``alpha`` is negative, ``class_weight`` is
        neither ``None`` nor ``'balanced'``, or the estimator name is unknown.

    See Also
    --------
    fit_text_classifier : Fits the representation and this head together.
    """
    if C <= 0:
        raise ValidationError("C must be > 0.")
    if alpha < 0:
        raise ValidationError("alpha must be >= 0.")
    if class_weight not in (None, "balanced"):
        raise ValidationError("class_weight must be None or 'balanced'.")

    if estimator == "logistic":
        from sklearn.linear_model import LogisticRegression

        return (
            LogisticRegression(
                C=float(C),
                class_weight=class_weight,
                max_iter=2000,
                random_state=random_state,
            ),
            True,
        )
    if estimator == "linear_svm":
        from sklearn.svm import LinearSVC

        return (
            LinearSVC(C=float(C), class_weight=class_weight, random_state=random_state),
            False,
        )
    if estimator == "sgd":
        from sklearn.linear_model import SGDClassifier

        return (
            SGDClassifier(
                loss="modified_huber",
                alpha=max(float(alpha), 1e-6),
                class_weight=class_weight,
                max_iter=2000,
                tol=1e-4,
                random_state=random_state,
            ),
            True,
        )
    if estimator == "complement_nb":
        from sklearn.naive_bayes import ComplementNB

        return ComplementNB(alpha=max(float(alpha), 1e-10)), True
    if estimator == "multinomial_nb":
        from sklearn.naive_bayes import MultinomialNB

        return MultinomialNB(alpha=max(float(alpha), 1e-10)), True
    raise ValidationError(f"Unknown NLP estimator {estimator!r}.")


def fit_text_classifier(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    backend: str | None = None,
    estimator: str | None = None,
    text_column: str | None = None,
    vectorizer: str = "tfidf",
    analyzer: str = "word",
    ngram_range: tuple[int, int] = (1, 2),
    max_features: int | None = 20000,
    min_df: int | float = 1,
    max_df: int | float = 1.0,
    sublinear_tf: bool = True,
    binary: bool = False,
    n_hash_features: int = 2**18,
    normalize_steps: list[str] | None = None,
    stopwords: list[str] | None = None,
    stopword_language: str | None = None,
    min_token_length: int = 1,
    max_token_length: int = 40,
    stem: bool = False,
    lemmatize: bool = False,
    class_weight: str | None = None,
    C: float = 1.0,
    alpha: float = 1.0,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_seq_tokens: int = 256,
    device: str = "cpu",
    random_state: int | None = 0,
) -> tuple[NlpTextPlan, NlpFitResult]:
    """Learn to predict a document's label from its text, using train rows only.

    Normalises the training documents, fits a representation over them, fits a
    classifier head on the resulting vectors, and returns both as a replayable
    plan alongside a report of what the fit saw.

    Everything vocabulary-bearing — the tokenizer's output, document
    frequencies, IDF weights, class priors — comes from training documents.
    Holdout documents are only ever transformed, never fitted, so tokens the
    model has not seen are reported as an out-of-vocabulary rate instead of
    being quietly absorbed into the vocabulary.

    Parameters
    ----------
    dataset:
        The dataset holding the text and its labels.
    split_plan:
        The split defining the training documents. Required — a vocabulary
        fitted across the whole corpus produces a holdout score that cannot be
        trusted.
    backend:
        Which representation family to use. ``'sklearn'`` builds bag-of-n-grams
        in-process and needs no extra dependency. The sentence-embedding and
        transformer backends produce dense vectors that capture meaning rather
        than surface word matches, at the cost of a heavier install and no
        token-level interpretability. Left as ``None``, it is inferred from
        ``vectorizer``.
    estimator:
        The classifier head, as described in :func:`build_estimator`. Left as
        ``None``, a sensible head is chosen for the backend.
    text_column:
        Which column holds the documents. Inferred from column roles and dtype
        when omitted; name it explicitly if the dataset has several text
        columns.
    vectorizer:
        How tokens become numbers, for the sklearn backend.

        ``'tfidf'`` (the default) weighs each term down by how many documents
        contain it, so a word appearing everywhere contributes little. This is
        almost always the right starting point. ``'count'`` uses raw
        occurrences, which lets common words dominate. ``'hashing'`` maps terms
        into a fixed number of buckets, giving constant memory on an unbounded
        vocabulary — but it has no invertible vocabulary, so token attributions
        become impossible and a warning is recorded.
    analyzer:
        ``'word'`` splits on word boundaries and is what you want for ordinary
        prose. ``'char'`` builds features from character sequences instead,
        which is more robust to typos, handles languages without whitespace
        word boundaries, and works better on very short strings like product
        codes — at the cost of a much larger feature space.
    ngram_range:
        The term lengths to extract, as ``(min_n, max_n)``. The default
        ``(1, 2)`` takes single words and adjacent pairs, which recovers some
        of the word order that bag-of-words discards — "not good" becomes its
        own feature rather than dissolving into "not" and "good". Widening
        beyond pairs grows the vocabulary steeply for diminishing returns.
    max_features:
        Cap on vocabulary size, keeping the most frequent terms. The main
        control on memory: an uncapped word-bigram vocabulary over a large
        corpus runs to millions of features. ``None`` removes the cap.
    min_df:
        Ignore terms appearing in fewer than this many documents — an integer
        is a document count, a float a proportion. Raising it removes typos and
        one-off tokens that cannot generalise. Set it to at least 2 on a noisy
        corpus.
    max_df:
        Ignore terms appearing in more than this share of documents. A
        data-driven alternative to a stopword list: a word in 95% of your
        documents distinguishes nothing, whatever language it is in.
    sublinear_tf:
        Replace a raw term count with ``1 + log(count)``, so a word appearing
        twenty times counts as meaningfully more than one appearing twice but
        not ten times more. Usually improves results on documents of varying
        length.
    binary:
        Record only whether a term is present, discarding counts entirely.
        Worth trying on short documents, where a repeat says little.
    n_hash_features:
        Number of buckets for the hashing vectorizer. Larger means fewer
        collisions between unrelated terms, at proportional memory cost.
    normalize_steps:
        Which normalisation steps to apply — lowercasing, punctuation
        stripping, and so on. Defaults to a conservative sequence. See
        :mod:`buildml.nlp.normalize`.
    stopwords:
        Explicit terms to discard before vectorising.
    stopword_language:
        Use the shipped stopword list for this language instead of naming terms
        individually.
    min_token_length:
        Drop tokens shorter than this, which removes stray single characters.
    max_token_length:
        Drop tokens longer than this, which removes URLs and concatenated
        garbage that would each become their own useless feature.
    stem:
        Reduce words to a crude root, so "running" and "runs" collapse
        together. Shrinks the vocabulary and helps on small corpora; the roots
        are not real words, which makes attributions harder to read.
    lemmatize:
        Reduce words to their dictionary form. Slower than stemming and needs a
        language model, but the output stays readable.
    class_weight:
        Passed to the head. ``'balanced'`` when a category is rare and you care
        about catching it.
    C:
        Inverse regularisation strength for the linear heads.
    alpha:
        Smoothing or regularisation for the naive Bayes and SGD heads.
    embedding_model_name:
        Which sentence-transformer to load, for the embedding backend. The
        default is small and fast; larger models capture more nuance and cost
        proportionally more to run.
    max_seq_tokens:
        How much of each document the transformer backends read. Text beyond
        this is truncated, so raise it if your documents are long and the
        signal is not front-loaded — attention cost grows quadratically with
        this number.
    device:
        Where to run the neural backends. ``'cpu'`` works everywhere;
        ``'cuda'`` is dramatically faster if available.
    random_state:
        Seed for the head, so the fit reproduces.

    Returns
    -------
    tuple of (~buildml.nlp.results.NlpTextPlan, ~buildml.nlp.results.NlpFitResult)
        The plan carries the fitted representation and head for scoring new
        documents. The result reports what the fit saw — vocabulary size, class
        counts, mean document length, blank-document rate, and any warnings.

    Raises
    ------
    ~buildml.core.errors.LeakageError
        No split plan was supplied.
    ~buildml.core.errors.ValidationError
        The text column resolves to the target column; there are fewer than
        four training documents; the target has fewer than two classes in
        training; the representation came out empty; or a naive Bayes head was
        paired with a representation containing negative values.
    ~buildml.core.errors.MissingExtraError
        A neural backend was requested without its optional dependency.

    Notes
    -----
    **Read the warnings on the result.** They report the things that quietly
    ruin text models: blank documents that can only be predicted from the class
    prior, a mean document length so short that n-grams are too sparse to be
    stable, and a hashing vectorizer that has forfeited interpretability.

    **``train_score`` is in-sample.** It is the head scoring the documents it
    was fitted on, and on high-dimensional text it will look excellent
    regardless. Use :func:`~buildml.nlp.evaluate.evaluate_text_classifier` for
    a number that means something.

    **Start with the sklearn backend.** A TF-IDF bag of n-grams with logistic
    regression is a genuinely strong baseline on document classification, it
    trains in seconds, and it can tell you which words drove each decision.
    Reach for embeddings when word overlap is not enough — when documents mean
    the same thing in different words.

    Examples
    --------
    >>> plan, result = fit_text_classifier(  # doctest: +SKIP
    ...     dataset, split_plan, text_column="ticket_body", min_df=2
    ... )
    >>> result.vocabulary_size, result.classes  # doctest: +SKIP
    (8412, ('billing', 'technical'))

    See Also
    --------
    buildml.nlp.predict.predict_text : Score documents with the returned plan.
    buildml.nlp.evaluate.evaluate_text_classifier : Honest holdout metrics.
    buildml.nlp.interpret.interpret_text_prediction : Why a document scored as it did.
    buildml.nlp.profile.profile_text_corpus : Inspect the corpus before fitting.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None

    resolved_backend, resolved_estimator = resolve_backend_estimator(
        backend=backend, estimator=estimator, vectorizer=vectorizer
    )
    column = resolve_text_column(dataset, text_column)
    target_column = dataset.require_target()
    if column == target_column:
        raise ValidationError(
            f"text_column and target column are both {column!r}; "
            "pass text_column= explicitly."
        )

    normalize_config = TextNormalizeConfig(
        steps=tuple(normalize_steps) if normalize_steps else DEFAULT_NORMALIZE_STEPS,  # type: ignore[arg-type]
        stopwords=tuple(stopwords) if stopwords else None,
        stopword_language=stopword_language,
        min_token_length=int(min_token_length),
        max_token_length=int(max_token_length),
        stem=bool(stem),
        lemmatize=bool(lemmatize),
    )
    normalize_plan = build_normalize_plan(normalize_config)
    vectorize_config = NlpVectorizeConfig(
        kind=vectorizer,  # type: ignore[arg-type]
        analyzer=analyzer,  # type: ignore[arg-type]
        ngram_range=(int(ngram_range[0]), int(ngram_range[1])),
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=bool(sublinear_tf),
        binary=bool(binary),
        n_hash_features=int(n_hash_features),
    )

    documents, frame = documents_for(
        dataset, split_plan, "train", column, operation="fit_text_classifier"
    )
    if len(documents) < MIN_TRAIN_DOCUMENTS:
        raise ValidationError(
            f"fit_text_classifier needs at least {MIN_TRAIN_DOCUMENTS} train "
            f"documents; the train partition has {len(documents)}."
        )
    y_train = targets_for(frame, target_column, operation="fit_text_classifier")
    labels = tuple(sorted({str(item) for item in y_train}))
    if len(labels) < 2:
        raise ValidationError(
            f"Target {target_column!r} has {len(labels)} distinct train class(es); "
            "document classification needs at least 2."
        )

    warnings: list[str] = []
    blank_rate = empty_document_rate(documents)
    if blank_rate > 0.0:
        warnings.append(
            f"{blank_rate:.1%} of train documents are blank; they contribute no "
            "features and are predicted from class priors."
        )

    vector_obj, disclosures = build_document_vectorizer(
        backend=resolved_backend,  # type: ignore[arg-type]
        config=vectorize_config,
        normalize_plan=normalize_plan,
        embedding_model_name=embedding_model_name,
        max_seq_tokens=int(max_seq_tokens),
        device=device,
    )
    matrix = vector_obj.fit_transform(documents)
    n_features = matrix_width(matrix)
    if n_features == 0:
        raise ValidationError(
            "The train-fitted representation is empty (0 features). Loosen "
            "min_df / stopwords / min_token_length, or check the text column."
        )

    if resolved_estimator in {"complement_nb", "multinomial_nb"}:
        minimum = float(matrix.min()) if hasattr(matrix, "min") else float(np.min(matrix))
        if minimum < 0.0:
            raise ValidationError(
                f"estimator='{resolved_estimator}' requires non-negative features; "
                "the current representation contains negative values. Use "
                "estimator='logistic' or a count/tfidf vectorizer."
            )

    head, supports_proba = build_estimator(
        resolved_estimator,
        class_weight=class_weight,
        C=C,
        alpha=alpha,
        random_state=random_state,
    )
    y_encoded = np.asarray([str(item) for item in y_train], dtype=object)
    head.fit(matrix, y_encoded)
    fitted_classes = tuple(str(item) for item in getattr(head, "classes_", labels))

    train_score: float | None = None
    try:
        train_score = float(head.score(matrix, y_encoded))
    except Exception:  # pragma: no cover - defensive; sklearn heads all score
        train_score = None

    feature_names = feature_names_for(vector_obj, limit=MAX_STORED_FEATURE_NAMES)
    vocab_size = vocabulary_size(vector_obj)
    if vectorize_config.kind == "hashing" and resolved_backend == "sklearn":
        warnings.append(
            "vectorizer='hashing' has no invertible vocabulary; "
            "interpret_text_prediction cannot report token attributions."
        )
    stats = token_stats(documents, normalize_plan)
    if stats["mean"] < 3.0:
        warnings.append(
            f"Mean train document length is {stats['mean']:.1f} tokens; very short "
            "documents make n-gram features sparse and metrics unstable."
        )

    plan_disclosures = (
        *disclosures,
        *normalize_plan.disclosures,
        f"Representation and head were fitted on {len(documents)} train documents "
        "only; holdout documents are transform-only.",
        f"Head: {resolved_estimator} (class_weight={class_weight!r}).",
        "Honesty: single-label document classification. Not sequence labelling, "
        "not generation, not document retrieval for generation (buildml.rag).",
    )

    plan = NlpTextPlan(
        task="classification",
        backend=resolved_backend,
        estimator=resolved_estimator,
        text_column=column,
        target_column=target_column,
        normalize_plan=normalize_plan,
        vectorize=vectorize_config.to_dict(),
        n_train_rows=len(documents),
        n_features=n_features,
        classes_=fitted_classes,
        vectorizer_=vector_obj,
        estimator_=head,
        embedding_model_name=(
            None if resolved_backend == "sklearn" else embedding_model_name
        ),
        feature_names_=feature_names,
        supports_proba=supports_proba,
        disclosures=plan_disclosures,
        warnings=tuple(warnings),
        config={
            "backend": resolved_backend,
            "estimator": resolved_estimator,
            "class_weight": class_weight,
            "C": float(C),
            "alpha": float(alpha),
            "random_state": random_state,
            "max_seq_tokens": int(max_seq_tokens),
            "device": device,
        },
    )
    result = NlpFitResult(
        task="classification",
        backend=resolved_backend,
        estimator=resolved_estimator,
        text_column=column,
        target_column=target_column,
        n_train_rows=len(documents),
        n_features=n_features,
        vocabulary_size=vocab_size,
        classes=fitted_classes,
        class_counts=class_counts(y_encoded),
        train_score=train_score,
        mean_document_tokens=stats["mean"],
        empty_document_rate=blank_rate,
        disclosures=plan_disclosures,
        warnings=tuple(warnings),
    )
    return plan, result


__all__ = ["MIN_TRAIN_DOCUMENTS", "build_estimator", "fit_text_classifier"]
