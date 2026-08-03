"""Discover what a collection of documents is about, without being told.

Topic modelling is what you reach for when you have thousands of documents and
no labels. It finds groups of words that tend to appear together, and each group
is a "topic": not because anyone defined it, but because the arithmetic says
those words travel as a set.

The crucial thing to understand is that topics are found, not verified. Nothing
here can tell you whether a topic is meaningful; it can only tell you the terms
came out together. Reading them is the check, and it is not optional. Some
topics will obviously be "billing complaints"; others will be an artefact of a
shared email footer.

Two methods, differing in what they assume. NMF factorises the TF-IDF matrix
into non-negative parts, which tends to give sharp, readable topics and runs
fast. LDA models each document as a probabilistic mixture drawn from topic
distributions, which is a better fit when documents genuinely blend subjects and
which gives calibrated proportions rather than scores.

Both are fitted on training documents only, so assigning topics to a holdout is
a pure transform. That is what makes topic proportions safe to feed into a
downstream supervised model: a topic basis fitted across the whole corpus would
have seen the holdout text, and the model built on it would score too well.

Topic quality is reported through NPMI coherence, which measures whether a
topic's top terms genuinely co-occur rather than merely scoring together.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition
from buildml.nlp.features import PartitionOrAll, documents_for, resolve_text_column
from buildml.nlp.normalize import build_normalize_plan
from buildml.nlp.results import (
    NlpTopicAssignResult,
    NlpTopicPlan,
    NlpTopicResult,
    Topic,
)
from buildml.nlp.types import (
    DEFAULT_NORMALIZE_STEPS,
    NlpVectorizeConfig,
    TextNormalizeConfig,
)
from buildml.nlp.vectorize import build_sklearn_vectorizer, feature_names_for

MIN_TOPIC_DOCUMENTS = 6
VALID_TOPIC_METHODS: tuple[str, ...] = ("nmf", "lda")


def npmi_coherence(
    term_indices: list[int],
    document_term: sparse.csr_matrix,
) -> float | None:
    """Measure whether a topic's terms actually appear together in real documents.

    The problem this solves: a decomposition will happily group terms that
    score together arithmetically but never co-occur in any document. Such a
    topic looks fine in a list and means nothing. Coherence checks the terms
    against the corpus itself.

    Normalised pointwise mutual information compares how often each pair of
    terms appears in the same document against how often they would if they
    were independent, then averages across all pairs of the topic's top terms.

    Parameters
    ----------
    term_indices:
        Column indices of the topic's top terms. Indices beyond the matrix
        width are ignored rather than raising, since a truncated vocabulary is
        a normal condition.
    document_term:
        The document-term matrix the topics were fitted on. Only presence
        matters, so counts are binarised.

    Returns
    -------
    float or None
        Average pairwise NPMI, bounded in ``[-1, 1]``. Near 1 means the terms
        nearly always appear together; 0 means they are independent; negative
        means they actively avoid each other, which is a sign the topic is an
        artefact. ``None`` when fewer than two usable terms or fewer than two
        documents make the question unanswerable.

    Notes
    -----
    Coherence is the standard way to choose a topic count: fit several models
    and take the one where mean coherence peaks. Unlike reconstruction error,
    it does not improve monotonically with more topics, so it has a maximum
    worth finding.

    It is a proxy, not a verdict. A coherent topic is one whose words really do
    travel together: which is necessary for the topic to mean something, and
    not sufficient. Boilerplate is extremely coherent.
    """
    usable = [index for index in term_indices if index < document_term.shape[1]]
    if len(usable) < 2:
        return None
    n_docs = document_term.shape[0]
    if n_docs < 2:
        return None
    binary = document_term[:, usable]
    binary = (binary > 0).astype(np.float64)
    counts = np.asarray(binary.sum(axis=0)).ravel()
    co_counts = np.asarray((binary.T @ binary).todense(), dtype=float)

    scores: list[float] = []
    epsilon = 1e-12
    for left in range(len(usable)):
        for right in range(left + 1, len(usable)):
            p_left = counts[left] / n_docs
            p_right = counts[right] / n_docs
            p_both = co_counts[left, right] / n_docs
            if p_left <= 0 or p_right <= 0:
                continue
            if p_both <= 0:
                scores.append(-1.0)
                continue
            pmi = np.log(p_both / (p_left * p_right) + epsilon)
            denominator = -np.log(p_both + epsilon)
            if denominator <= epsilon:
                continue
            # The epsilon guards can push a perfect co-occurrence a hair past
            # 1.0; clamp so the reported score honours the documented bound.
            scores.append(float(min(1.0, max(-1.0, pmi / denominator))))
    if not scores:
        return None
    return float(np.mean(scores))


def _topic_label(terms: tuple[str, ...]) -> str:
    return " / ".join(terms[:3]) if terms else "empty"


def fit_topics(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    method: str = "nmf",
    n_topics: int = 6,
    text_column: str | None = None,
    top_terms: int = 10,
    max_features: int | None = 20000,
    min_df: int | float = 2,
    max_df: int | float = 0.95,
    ngram_range: tuple[int, int] = (1, 1),
    normalize_steps: list[str] | None = None,
    stopwords: list[str] | None = None,
    stopword_language: str | None = "en",
    stem: bool = False,
    max_iter: int = 300,
    random_state: int | None = 0,
) -> tuple[NlpTopicPlan, NlpTopicResult]:
    """Discover the recurring themes in a corpus, using training documents only.

    Vectorises the training documents, decomposes the result into the requested
    number of topics, and reports each topic's defining terms, its share of the
    corpus, and how coherent it is.

    Both the vectorizer and the decomposition are fitted on train. That makes
    :func:`assign_topics` on a holdout a pure transform, which is what lets
    topic proportions be used as features for a downstream supervised model
    without the topic basis having seen holdout text.

    Parameters
    ----------
    dataset:
        The dataset holding the documents.
    split_plan:
        The split defining the training documents. Required.
    method:
        ``'nmf'`` factorises the TF-IDF matrix into non-negative parts: fast,
        and usually the sharper and more readable of the two. ``'lda'`` models
        documents as probabilistic mixtures, which suits genuinely blended
        documents and yields proportions you can interpret as such.
    n_topics:
        How many topics to find. Your decision, not a discovery: there is no
        true number, and different counts give different valid views of the
        same corpus. Fit a range and compare ``mean_coherence``.
    text_column:
        Which column holds the documents. Inferred when omitted.
    top_terms:
        How many defining terms to report per topic. Ten is usually enough to
        recognise a topic; more starts including terms that barely belong.
    max_features:
        Vocabulary cap for the underlying vectorizer.
    min_df:
        Ignore terms appearing in fewer than this many documents. Defaults to 2
        here rather than 1, because a term appearing once cannot form part of a
        recurring theme by definition.
    max_df:
        Ignore terms appearing in more than this share of documents. Defaults
        to 0.95, since a near-universal term would otherwise appear in every
        topic and distinguish none of them.
    ngram_range:
        Term lengths to extract. Defaults to single words: topic models are
        read as word lists, and phrase features make that list harder to
        interpret without adding much.
    normalize_steps:
        Which normalisation steps to apply.
    stopwords:
        Extra terms to discard. Worth using for domain boilerplate: a product
        name in every document will otherwise anchor its own meaningless topic.
    stopword_language:
        Built-in stopword list to apply. Defaults to English, unlike the
        classification path, because unfiltered function words dominate topic
        term lists completely.
    stem:
        Collapse words to a crude root, merging "billing" and "billed". Tightens
        topics; makes the term lists less readable.
    max_iter:
        Iteration cap for the decomposition. Raise it if the fit warns about
        not converging.
    random_state:
        Seed. Both methods are randomised, and without a fixed seed the same
        corpus yields different topics every run.

    Returns
    -------
    tuple of (~buildml.nlp.results.NlpTopicPlan, ~buildml.nlp.results.NlpTopicResult)
        The plan assigns topics to new documents. The result reports the
        discovered topics and the quality signals.

    Raises
    ------
    ~buildml.core.errors.LeakageError
        No split plan was supplied.
    ~buildml.core.errors.ValidationError
        The method is unknown; ``n_topics`` is below 2; ``top_terms`` is below
        1; there are fewer than six training documents; or the text column
        cannot be resolved.

    Notes
    -----
    **Read the terms before using the topics for anything.** This is
    unsupervised: there is no accuracy to check, and the algorithm will always
    return exactly the number of topics you asked for whether or not the corpus
    contains that many themes.

    **Do not use ``reconstruction_error`` to choose ``n_topics``.** It falls
    monotonically as topics are added, so it always favours more. Coherence
    peaks, which is what makes it usable for the choice.

    **A topic with very low ``train_mass`` came from a handful of documents**
    and is unlikely to reappear in new text.

    Examples
    --------
    >>> plan, result = fit_topics(dataset, split_plan, n_topics=8)  # doctest: +SKIP
    >>> [topic.label for topic in result.topics]  # doctest: +SKIP

    See Also
    --------
    assign_topics : Apply the fitted topics to new documents.
    buildml.nlp.keyphrases.extract_keyphrases : Salient phrases, no model needed.
    """
    method_key = str(method).lower()
    if method_key not in VALID_TOPIC_METHODS:
        raise ValidationError(
            f"method={method!r} is not supported. Choose from {list(VALID_TOPIC_METHODS)}."
        )
    if n_topics < 2:
        raise ValidationError("n_topics must be >= 2.")
    if top_terms < 1:
        raise ValidationError("top_terms must be >= 1.")
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None

    column = resolve_text_column(dataset, text_column)
    documents, _frame = documents_for(
        dataset, split_plan, "train", column, operation="fit_topics"
    )
    if len(documents) < MIN_TOPIC_DOCUMENTS:
        raise ValidationError(
            f"fit_topics needs at least {MIN_TOPIC_DOCUMENTS} train documents; "
            f"the train partition has {len(documents)}."
        )
    if n_topics > len(documents):
        raise ValidationError(
            f"n_topics={n_topics} exceeds the {len(documents)} train documents "
            "available; reduce n_topics."
        )

    normalize_plan = build_normalize_plan(
        TextNormalizeConfig(
            steps=tuple(normalize_steps) if normalize_steps else DEFAULT_NORMALIZE_STEPS,  # type: ignore[arg-type]
            stopwords=tuple(stopwords) if stopwords else None,
            stopword_language=stopword_language,
            stem=bool(stem),
        )
    )
    # NMF factorizes TF-IDF weights; LDA is a generative count model and must
    # receive raw counts.
    kind = "tfidf" if method_key == "nmf" else "count"
    vectorize_config = NlpVectorizeConfig(
        kind=kind,  # type: ignore[arg-type]
        analyzer="word",
        ngram_range=(int(ngram_range[0]), int(ngram_range[1])),
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
    )
    vectorizer = build_sklearn_vectorizer(vectorize_config, normalize_plan)
    try:
        matrix = vectorizer.fit_transform(documents)
    except ValueError as exc:
        raise ValidationError(
            f"Topic vectorization produced no vocabulary ({exc}). Lower min_df, "
            "raise max_df, or relax stopword removal."
        ) from exc
    if matrix.shape[1] == 0:
        raise ValidationError(
            "Topic vectorization produced an empty vocabulary. Lower min_df, "
            "raise max_df, or relax stopword removal."
        )

    warnings: list[str] = []
    if n_topics > matrix.shape[1]:
        raise ValidationError(
            f"n_topics={n_topics} exceeds the {matrix.shape[1]}-term vocabulary; "
            "reduce n_topics or lower min_df."
        )

    if method_key == "nmf":
        from sklearn.decomposition import NMF

        model = NMF(
            n_components=int(n_topics),
            init="nndsvd",
            max_iter=int(max_iter),
            random_state=random_state,
        )
        weights = model.fit_transform(matrix)
        reconstruction_error = float(getattr(model, "reconstruction_err_", float("nan")))
        perplexity: float | None = None
    else:
        from sklearn.decomposition import LatentDirichletAllocation

        model = LatentDirichletAllocation(
            n_components=int(n_topics),
            max_iter=int(max_iter),
            learning_method="batch",
            random_state=random_state,
        )
        weights = model.fit_transform(matrix)
        reconstruction_error = None
        try:
            perplexity = float(model.perplexity(matrix))
        except Exception:  # pragma: no cover - sklearn guards this internally
            perplexity = None

    names = feature_names_for(vectorizer)
    components = np.asarray(model.components_, dtype=float)
    binary_counts = sparse.csr_matrix((matrix > 0).astype(np.float64))
    column_mass = np.asarray(weights, dtype=float).sum(axis=0)
    total_mass = float(column_mass.sum()) or 1.0

    topics: list[Topic] = []
    coherences: list[float] = []
    for index in range(components.shape[0]):
        row = components[index]
        order = np.argsort(-row)[: int(top_terms)]
        terms = tuple(
            names[int(position)] if int(position) < len(names) else f"term_{position}"
            for position in order
        )
        term_weights = tuple(float(row[int(position)]) for position in order)
        coherence = npmi_coherence([int(position) for position in order], binary_counts)
        if coherence is not None:
            coherences.append(coherence)
        topics.append(
            Topic(
                index=index,
                label=_topic_label(terms),
                terms=terms,
                weights=term_weights,
                train_mass=float(column_mass[index] / total_mass),
                coherence=coherence,
            )
        )

    tiny = [topic.index for topic in topics if topic.train_mass < 0.01]
    if tiny:
        warnings.append(
            f"Topic(s) {tiny} carry under 1% of train document mass; consider a "
            "smaller n_topics."
        )
    mean_coherence = float(np.mean(coherences)) if coherences else None
    if mean_coherence is not None and mean_coherence < 0.0:
        warnings.append(
            f"Mean NPMI coherence is {mean_coherence:.3f} (below zero); the top "
            "terms co-occur less than chance, so these topics are weak."
        )

    disclosures = (
        f"Topic model: {method_key.upper()} with {n_topics} topics over a "
        f"{matrix.shape[1]}-term {kind} vocabulary fitted on "
        f"{len(documents)} train documents.",
        "Coherence is NPMI over each topic's top terms, computed on the train "
        "partition only.",
        "Topics are unsupervised: labels are the top three terms, not validated "
        "human categories.",
        *normalize_plan.disclosures,
    )

    plan = NlpTopicPlan(
        method=method_key,
        text_column=column,
        n_topics=int(n_topics),
        n_train_rows=len(documents),
        normalize_plan=normalize_plan,
        vectorize=vectorize_config.to_dict(),
        topics=tuple(topics),
        vectorizer_=vectorizer,
        model_=model,
        random_state=random_state,
        disclosures=disclosures,
        warnings=tuple(warnings),
    )
    result = NlpTopicResult(
        method=method_key,
        n_topics=int(n_topics),
        n_train_rows=len(documents),
        text_column=column,
        topics=tuple(topics),
        mean_coherence=mean_coherence,
        reconstruction_error=reconstruction_error,
        perplexity=perplexity,
        disclosures=disclosures,
        warnings=tuple(warnings),
    )
    return plan, result


def assign_topics(
    dataset: Dataset,
    plan: NlpTopicPlan,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll = "test",
) -> NlpTopicAssignResult:
    """Score documents against topics that were already discovered.

    Each document gets a weight for every topic, not a single assignment,
    because documents genuinely mix subjects. The dominant topic is provided
    for when you need one answer, but read it alongside the weights: a
    document spread evenly across three topics has a dominant one that means
    very little.

    Parameters
    ----------
    dataset:
        The dataset holding the documents.
    plan:
        A fitted plan from :func:`fit_topics`, which supplies the text column.
    split_plan:
        The split defining partitions. Required unless ``partition`` is
        ``'all'``.
    partition:
        Which rows to assign.

    Returns
    -------
    ~buildml.nlp.results.NlpTopicAssignResult
        Per-document topic weights, the dominant topic per document, the topic
        share across the partition, and the topic labels.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        The plan has no fitted vectorizer or model, the partition is empty, or
        the text column is missing.

    Notes
    -----
    **Nothing is refitted.** New documents are projected onto the topics found
    during the fit, so a genuinely new theme in the incoming text has nowhere
    to go: it gets distributed across whichever existing topics fit least
    badly.

    **Compare the returned ``topic_share`` against the plan's ``train_mass``.**
    A large shift means the documents you are assigning are about different
    things than the ones the topics were built from, which is drift and a
    reason to refit.

    See Also
    --------
    fit_topics : Discover the topics in the first place.
    """
    if plan.vectorizer_ is None or plan.model_ is None:
        raise ValidationError(
            "The topic plan has no fitted vectorizer/model. Call fit_topics first."
        )
    documents, _frame = documents_for(
        dataset, split_plan, partition, plan.text_column, operation="assign_topics"
    )
    matrix = plan.vectorizer_.transform(documents)
    weights = np.asarray(plan.model_.transform(matrix), dtype=float)
    row_sums = weights.sum(axis=1, keepdims=True)
    shares = np.divide(
        weights,
        np.where(row_sums <= 0, 1.0, row_sums),
        out=np.zeros_like(weights),
        where=True,
    )
    dominant = tuple(int(index) for index in np.argmax(weights, axis=1))
    counts: dict[str, float] = {}
    for index in range(weights.shape[1]):
        counts[str(index)] = float(
            sum(1 for value in dominant if value == index) / max(1, len(dominant))
        )

    warnings: list[str] = []
    silent = int((weights.sum(axis=1) <= 0).sum())
    if silent:
        warnings.append(
            f"{silent} document(s) produced zero topic mass (no in-vocabulary "
            "terms); their dominant topic defaults to index 0."
        )

    return NlpTopicAssignResult(
        partition=str(partition),
        method=plan.method,
        n_rows=len(documents),
        n_topics=plan.n_topics,
        dominant_topics=dominant,
        topic_weights=tuple(tuple(float(value) for value in row) for row in shares),
        topic_share=counts,
        topic_labels=tuple(topic.label for topic in plan.topics),
        disclosures=(
            "Transform-only: the topic basis and vocabulary come from the train fit.",
            "topic_weights are row-normalized shares; topic_share is the fraction "
            "of documents whose dominant topic is that index.",
        ),
        warnings=tuple(warnings),
    )


__all__ = [
    "MIN_TOPIC_DOCUMENTS",
    "VALID_TOPIC_METHODS",
    "assign_topics",
    "fit_topics",
    "npmi_coherence",
]
