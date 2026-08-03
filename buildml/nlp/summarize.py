"""Shorten documents by choosing their most important sentences.

Sentences are **selected**, never written. A summary here is a subset of the
original document, which carries a guarantee worth having: it cannot state
anything the document did not, because every word came from the document. No
hallucination is possible when nothing is generated.

Generating prose needs a language model, and that lives in :mod:`buildml.ai`
where the provider, the cost, and the prompt are all disclosed. It is
deliberately not available here by accident.

Three methods.

**TextRank** builds a graph where sentences are nodes and edges weight how much
vocabulary two sentences share, then runs PageRank. A sentence is important if
it overlaps with other important sentences — which tends to surface the ones
expressing the document's central, repeated ideas.

**LexRank** works the same way but measures similarity with TF-IDF cosine rather
than raw overlap, and drops weak edges below a threshold. Weighting terms by
rarity makes it better at ignoring sentences that merely share common words.

**Lead-k** takes the first *k* sentences. Trivially simple, and a genuinely hard
baseline to beat on news and reports, where writers put the important
information first by convention. Try it before assuming a graph method is
better.

Selected sentences are always emitted in their original order, so the summary
reads as continuous text rather than a ranked list.
"""

from __future__ import annotations

import math

import numpy as np

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.nlp.catalog import SUMMARIZE_METHODS
from buildml.nlp.features import PartitionOrAll, documents_for, resolve_text_column
from buildml.nlp.lexicons import stopwords_for
from buildml.nlp.normalize import (
    TextNormalizePlan,
    build_normalize_plan,
    split_sentences,
    tokenize_document,
)
from buildml.nlp.results import NlpSummaryResult
from buildml.nlp.types import DEFAULT_NORMALIZE_STEPS, TextNormalizeConfig

LEXRANK_SIMILARITY_THRESHOLD = 0.1
PAGERANK_DAMPING = 0.85
PAGERANK_ITERATIONS = 40


def _pagerank(similarity: np.ndarray) -> np.ndarray:
    n = similarity.shape[0]
    if n == 0:
        return np.zeros(0, dtype=float)
    if n == 1:
        return np.ones(1, dtype=float)
    row_sums = similarity.sum(axis=1, keepdims=True)
    transition = np.divide(
        similarity,
        np.where(row_sums <= 0, 1.0, row_sums),
        out=np.zeros_like(similarity),
        where=True,
    )
    # Dangling rows redistribute uniformly so isolated sentences stay reachable.
    dangling = (row_sums.ravel() <= 0)
    if dangling.any():
        transition[dangling, :] = 1.0 / n
    scores = np.full(n, 1.0 / n, dtype=float)
    for _ in range(PAGERANK_ITERATIONS):
        updated = (1.0 - PAGERANK_DAMPING) / n + PAGERANK_DAMPING * (transition.T @ scores)
        total = updated.sum()
        if total <= 0:
            break
        updated = updated / total
        if np.abs(updated - scores).max() < 1e-8:
            scores = updated
            break
        scores = updated
    return scores


def _textrank_similarity(token_sets: list[list[str]]) -> np.ndarray:
    """Classic TextRank sentence similarity: overlap normalized by log lengths."""
    n = len(token_sets)
    matrix = np.zeros((n, n), dtype=float)
    unique = [set(tokens) for tokens in token_sets]
    for left in range(n):
        for right in range(left + 1, n):
            overlap = len(unique[left] & unique[right])
            if overlap == 0:
                continue
            denominator = math.log(len(unique[left]) + 1) + math.log(len(unique[right]) + 1)
            if denominator <= 0:
                continue
            value = overlap / denominator
            matrix[left, right] = value
            matrix[right, left] = value
    return matrix


def _lexrank_similarity(sentences: list[str], plan: TextNormalizePlan) -> np.ndarray:
    """LexRank similarity: cosine over TF-IDF sentence vectors above a threshold."""
    from buildml.nlp.types import NlpVectorizeConfig
    from buildml.nlp.vectorize import build_sklearn_vectorizer

    config = NlpVectorizeConfig(
        kind="tfidf",
        analyzer="word",
        ngram_range=(1, 1),
        max_features=None,
        min_df=1,
        max_df=1.0,
        sublinear_tf=True,
    )
    vectorizer = build_sklearn_vectorizer(config, plan)
    try:
        matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        return np.zeros((len(sentences), len(sentences)), dtype=float)
    if matrix.shape[1] == 0:
        return np.zeros((len(sentences), len(sentences)), dtype=float)
    from sklearn.metrics.pairwise import cosine_similarity

    similarity = np.asarray(cosine_similarity(matrix), dtype=float)
    np.fill_diagonal(similarity, 0.0)
    similarity[similarity < LEXRANK_SIMILARITY_THRESHOLD] = 0.0
    return similarity


def summarize_document(
    document: str,
    *,
    method: str,
    n_sentences: int,
    normalize_plan: TextNormalizePlan,
    max_input_sentences: int,
) -> tuple[str, tuple[int, ...], int]:
    """Summarise a single document by picking its best sentences.

    The unit of work behind :func:`summarize_text`, usable directly on a string
    that is not in a dataset.

    Parameters
    ----------
    document:
        The text to summarise.
    method:
        ``'textrank'``, ``'lexrank'``, or ``'lead'``.
    n_sentences:
        How many sentences to keep.
    normalize_plan:
        The plan used to tokenise sentences for the similarity computation. It
        governs which words count as shared, so the stopword list has real
        influence here — without it, sentences look similar because they both
        contain "the".
    max_input_sentences:
        Read at most this many sentences from the document. The graph methods
        cost time quadratic in sentence count, so this bounds a pathologically
        long document.

    Returns
    -------
    tuple
        ``(summary, selected_indices, n_input_sentences)``. The indices point
        into the document's own sentence list, in reading order, which is what
        lets a summary be highlighted in place rather than shown separately.

    Notes
    -----
    A document with no more sentences than requested is returned whole rather
    than treated as an error — the correct summary of a two-sentence document
    is those two sentences.

    Where the graph carries no signal at all, selection falls back to the
    leading sentences rather than picking arbitrarily.

    See Also
    --------
    summarize_text : Summarise a whole partition.
    """
    sentences = split_sentences(document, max_sentences=max_input_sentences)
    if not sentences:
        return "", (), 0
    if len(sentences) <= n_sentences:
        return " ".join(sentences), tuple(range(len(sentences))), len(sentences)

    if method == "lead":
        chosen = tuple(range(n_sentences))
        return " ".join(sentences[index] for index in chosen), chosen, len(sentences)

    if method == "lexrank":
        similarity = _lexrank_similarity(sentences, normalize_plan)
    else:
        token_sets = [tokenize_document(item, normalize_plan) for item in sentences]
        similarity = _textrank_similarity(token_sets)

    scores = _pagerank(similarity)
    if not scores.size or float(scores.max()) <= 0.0:
        chosen = tuple(range(n_sentences))
    else:
        ranked = np.argsort(-scores)[:n_sentences]
        # Emit in original reading order so the summary stays coherent.
        chosen = tuple(sorted(int(index) for index in ranked))
    return " ".join(sentences[index] for index in chosen), chosen, len(sentences)


def summarize_text(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll = "test",
    method: str = "textrank",
    text_column: str | None = None,
    n_sentences: int = 3,
    max_documents: int = 25,
    max_input_sentences: int = 200,
    stopword_language: str | None = "en",
    stopwords: list[str] | None = None,
) -> NlpSummaryResult:
    """Summarise every document in a partition by selecting sentences.

    Produces a short version of each document, built only from sentences the
    original contained.

    Parameters
    ----------
    dataset:
        The dataset holding the documents.
    split_plan:
        The split defining partitions. Required unless ``partition`` is
        ``'all'``.
    partition:
        Which rows to summarise. Summarisation learns nothing, so any partition
        is safe.
    method:
        ``'textrank'``, ``'lexrank'``, or ``'lead'``. Try ``'lead'`` as a
        baseline before assuming a graph method helps — on documents with a
        conventional structure it frequently wins.
    text_column:
        Which column holds the documents. Inferred when omitted.
    n_sentences:
        How many sentences each summary should contain.
    max_documents:
        Cap on documents summarised. The graph methods are not cheap, and a
        summary is something a person reads.
    max_input_sentences:
        Read at most this many sentences per document, bounding the quadratic
        cost of the similarity graph.
    stopword_language:
        Built-in stopword list to apply when measuring sentence similarity.
        Matters: without it, two sentences look similar for sharing "the" and
        "of".
    stopwords:
        Additional terms to ignore when comparing sentences.

    Returns
    -------
    ~buildml.nlp.results.NlpSummaryResult
        The summaries, which sentences were selected, row labels, the mean
        compression achieved, and any caveats.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        The method is unknown; ``n_sentences`` is below 1; the text column
        cannot be resolved; or the partition is empty.

    Notes
    -----
    **Nothing is generated.** Every sentence in the output appeared in the
    input, so a summary cannot invent a fact — but it can mislead by omission,
    which is what ``mean_compression`` is there to warn you about.

    **Short documents pass through unchanged.** A document with no more
    sentences than ``n_sentences`` is returned whole, which is why mean
    compression near 1.0 usually means the corpus did not need summarising.

    **Extractive summaries read like excerpts**, because they are. A document
    whose meaning is distributed across many sentences summarises poorly by
    selection, and no setting fixes that.

    Examples
    --------
    >>> result = summarize_text(dataset, split_plan, n_sentences=2)  # doctest: +SKIP
    >>> result.summaries[0], result.mean_compression  # doctest: +SKIP

    See Also
    --------
    summarize_document : Summarise a single string.
    buildml.nlp.keyphrases.extract_keyphrases : Phrases rather than sentences.
    """
    method_key = str(method).lower()
    if method_key not in SUMMARIZE_METHODS:
        raise ValidationError(
            f"method={method!r} is not supported. Choose from {list(SUMMARIZE_METHODS)}."
        )
    if n_sentences < 1:
        raise ValidationError("n_sentences must be >= 1.")
    if max_documents < 1:
        raise ValidationError("max_documents must be >= 1.")
    if max_input_sentences < n_sentences:
        raise ValidationError("max_input_sentences must be >= n_sentences.")

    column = resolve_text_column(dataset, text_column)
    documents, frame = documents_for(
        dataset, split_plan, partition, column, operation="summarize_text"
    )
    words = set()
    if stopword_language is not None:
        words |= set(stopwords_for(stopword_language))
    if stopwords:
        words |= {str(item).strip().lower() for item in stopwords if str(item).strip()}
    normalize_plan = build_normalize_plan(
        TextNormalizeConfig(
            steps=tuple(DEFAULT_NORMALIZE_STEPS),
            stopwords=tuple(sorted(words)) if words else None,
            min_token_length=2,
        )
    )

    selected = documents[:max_documents]
    summaries: list[str] = []
    indices: list[tuple[int, ...]] = []
    compressions: list[float] = []
    single_sentence = 0
    empty = 0
    for document in selected:
        summary, chosen, n_input = summarize_document(
            document,
            method=method_key,
            n_sentences=int(n_sentences),
            normalize_plan=normalize_plan,
            max_input_sentences=int(max_input_sentences),
        )
        summaries.append(summary)
        indices.append(chosen)
        if n_input == 0:
            empty += 1
            continue
        if n_input == 1:
            single_sentence += 1
        compressions.append(float(len(chosen) / n_input))

    warnings: list[str] = []
    if empty:
        warnings.append(f"{empty} document(s) had no detectable sentences.")
    if single_sentence:
        warnings.append(
            f"{single_sentence} document(s) contain a single sentence; the "
            "'summary' is the original text."
        )
    if len(documents) > len(selected):
        warnings.append(
            f"Summarized the first {len(selected)} of {len(documents)} documents "
            "(raise max_documents for more)."
        )

    return NlpSummaryResult(
        partition=str(partition),
        method=method_key,
        n_rows=len(selected),
        n_sentences=int(n_sentences),
        summaries=tuple(summaries),
        selected_sentence_indices=tuple(indices),
        document_row_labels=tuple(frame.index[: len(selected)]),
        mean_compression=float(np.mean(compressions)) if compressions else None,
        disclosures=(
            f"Extractive summarization with {method_key}: original sentences are "
            "selected and reordered into reading order; no text is generated.",
            "Sentence boundaries come from BuildML's abbreviation-aware splitter, "
            "so unusual formatting can change the candidate set.",
            "No reference summaries exist, so no ROUGE-style quality metric is "
            "claimed — only compression is reported.",
        ),
        warnings=tuple(warnings),
    )


__all__ = [
    "LEXRANK_SIMILARITY_THRESHOLD",
    "summarize_document",
    "summarize_text",
]
