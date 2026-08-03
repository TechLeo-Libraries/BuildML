"""Find the phrases that characterise a document or a corpus, without labels.

Keyphrases are the fastest way to understand text you have not read. Point this
at a folder of support tickets and you learn what people complain about in
seconds; point it at one document and you get something you could use as a tag.

Three methods, each with a different idea of what makes a phrase important.

**TF-IDF** asks which terms are frequent here and rare elsewhere. It needs a
corpus to compare against, and it is the most reliable when you have one.

**RAKE** looks at the words between stopwords. A phrase's score is the sum of
its words' scores, and each word scores by how many phrase-slots it occupies
relative to how often it appears: so words that consistently show up inside
longer, more specific phrases win. It works on a single document and favours
multi-word technical terms.

**TextRank** builds a graph of which words appear near which, then runs PageRank
over it. Words that co-occur with many other important words score highly. Also
single-document, and better than RAKE at finding phrases whose importance comes
from context rather than rarity.

None of these fits anything. Nothing is stored for reuse and no target is
consulted, which is why keyphrase extraction can be run on any partition :
though reading holdout text still informs you, and the disclosures say so.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.nlp.features import PartitionOrAll, documents_for, resolve_text_column
from buildml.nlp.lexicons import stopwords_for
from buildml.nlp.normalize import (
    TextNormalizePlan,
    build_normalize_plan,
    normalize_document,
)
from buildml.nlp.results import Keyphrase, NlpKeyphraseResult
from buildml.nlp.types import DEFAULT_NORMALIZE_STEPS, NlpVectorizeConfig, TextNormalizeConfig
from buildml.nlp.vectorize import build_sklearn_vectorizer, feature_names_for

VALID_KEYPHRASE_METHODS: tuple[str, ...] = ("tfidf", "rake", "textrank")
_PHRASE_SPLIT = re.compile(r"[^\w\s'\u2019\-]+|\d+")
_WORD = re.compile(r"[^\W\d_]+(?:['\u2019\-][^\W\d_]+)*")


def _candidate_phrases(
    document: str,
    stopwords: frozenset[str],
    *,
    max_words: int,
) -> list[list[str]]:
    """Split a normalized document into stopword-delimited candidate phrases."""
    phrases: list[list[str]] = []
    for chunk in _PHRASE_SPLIT.split(document):
        current: list[str] = []
        for word in _WORD.findall(chunk):
            token = word.lower()
            if token in stopwords or len(token) < 2:
                if current:
                    phrases.append(current)
                    current = []
                continue
            current.append(token)
            if len(current) >= max_words:
                phrases.append(current)
                current = []
        if current:
            phrases.append(current)
    return [phrase for phrase in phrases if phrase]


def rake_scores(
    documents: list[str],
    stopwords: frozenset[str],
    *,
    max_words: int = 3,
) -> tuple[dict[str, float], dict[str, int], list[dict[str, float]]]:
    """Score phrases by RAKE, which needs no corpus statistics to work.

    RAKE's insight is that stopwords mark phrase boundaries. Split a document
    at every stopword and punctuation mark, and what remains between the splits
    are candidate phrases: "the printer keeps jamming on thick paper" yields
    "printer keeps jamming" and "thick paper".

    Each word then scores as its degree divided by its frequency, where degree
    counts how many phrase-slots it participates in. A word that only ever
    appears alone scores 1; a word that consistently appears inside longer
    phrases scores higher. A phrase's score is the sum of its words', so longer
    and more distinctive phrases win.

    Parameters
    ----------
    documents:
        Already-normalised documents.
    stopwords:
        The terms that delimit phrases. Central to the method rather than
        incidental: too few and phrases run together into sentence fragments,
        too many and every phrase collapses to one word.
    max_words:
        Longest phrase to allow. Caps the bias toward length, which would
        otherwise let a single long run of non-stopwords dominate.

    Returns
    -------
    tuple
        ``(corpus_scores, document_frequency, per_document_scores)``. Corpus
        scores sum each phrase across documents; document frequency counts how
        many documents contain it; per-document scores keep each document's own
        phrases separately.

    Notes
    -----
    **RAKE needs no corpus and no training**, which is what makes it usable on
    a single document: the case where TF-IDF has nothing to compare against.

    **It is sensitive to the stopword list in a way the other methods are
    not.** The list defines the phrase boundaries, so a list missing your
    domain's connective words produces phrases that run on.

    See Also
    --------
    textrank_word_scores : Graph-based scoring, also single-document.
    extract_keyphrases : The user-facing entry point.
    """
    frequency: Counter[str] = Counter()
    degree: Counter[str] = Counter()
    per_document_phrases: list[list[list[str]]] = []
    document_frequency: Counter[str] = Counter()

    for document in documents:
        phrases = _candidate_phrases(document, stopwords, max_words=max_words)
        per_document_phrases.append(phrases)
        seen: set[str] = set()
        for phrase in phrases:
            span = len(phrase) - 1
            for word in phrase:
                frequency[word] += 1
                degree[word] += span
            seen.add(" ".join(phrase))
        for phrase_text in seen:
            document_frequency[phrase_text] += 1

    word_score = {
        word: float((degree[word] + frequency[word]) / frequency[word])
        for word in frequency
    }
    corpus: dict[str, float] = defaultdict(float)
    per_document: list[dict[str, float]] = []
    for phrases in per_document_phrases:
        local: dict[str, float] = {}
        for phrase in phrases:
            text = " ".join(phrase)
            score = float(sum(word_score.get(word, 0.0) for word in phrase))
            local[text] = max(local.get(text, 0.0), score)
        per_document.append(local)
        for text, score in local.items():
            corpus[text] += score
    return dict(corpus), dict(document_frequency), per_document


def textrank_word_scores(
    tokens_per_document: list[list[str]],
    *,
    window: int = 4,
    damping: float = 0.85,
    iterations: int = 40,
) -> dict[str, float]:
    """Score words by their position in a co-occurrence graph, using PageRank.

    Build a graph where words are nodes and an edge joins any two words that
    appear near each other, then run the algorithm that ranked web pages. A
    word is important if it sits near many important words: a recursive
    definition that PageRank resolves by iterating to a stable state.

    The practical difference from RAKE is what "important" means. RAKE rewards
    words that appear in distinctive phrases; TextRank rewards words that are
    central to how the document's vocabulary connects. A word can be common and
    still rank highly if it is the hub everything else attaches to.

    Parameters
    ----------
    tokens_per_document:
        Tokenised documents. Edges are only drawn within a document, so
        unrelated documents cannot link words together.
    window:
        How many tokens apart two words can be and still be joined. Wider
        windows capture looser association and produce a denser graph; narrow
        ones approximate direct adjacency.
    damping:
        PageRank's damping factor: the probability of continuing along an edge
        rather than jumping to a random node. The conventional 0.85 comes from
        the original PageRank work and rarely needs changing.
    iterations:
        How many rounds to run. The scores converge quickly; more iterations
        refine rather than change the ranking.

    Returns
    -------
    dict
        Word to score, normalised so all scores sum to 1. Empty when no
        co-occurrences were found, which happens on documents of one token.

    Notes
    -----
    Like RAKE, this needs no corpus: a single document has enough
    co-occurrence structure to build a graph from.

    See Also
    --------
    rake_scores : Phrase scoring from stopword boundaries.
    extract_keyphrases : The user-facing entry point.
    """
    adjacency: dict[str, Counter[str]] = defaultdict(Counter)
    for tokens in tokens_per_document:
        for position, token in enumerate(tokens):
            end = min(len(tokens), position + window + 1)
            for other in tokens[position + 1 : end]:
                if other == token:
                    continue
                adjacency[token][other] += 1
                adjacency[other][token] += 1
    if not adjacency:
        return {}
    nodes = sorted(adjacency)
    scores = {node: 1.0 / len(nodes) for node in nodes}
    out_weight = {node: float(sum(adjacency[node].values())) or 1.0 for node in nodes}
    for _ in range(max(1, iterations)):
        updated: dict[str, float] = {}
        for node in nodes:
            inbound = 0.0
            for neighbour, weight in adjacency[node].items():
                inbound += scores[neighbour] * (weight / out_weight[neighbour])
            updated[node] = (1.0 - damping) / len(nodes) + damping * inbound
        total = sum(updated.values()) or 1.0
        scores = {node: value / total for node, value in updated.items()}
    return scores


def extract_keyphrases(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll = "train",
    method: str = "tfidf",
    text_column: str | None = None,
    top_n: int = 15,
    max_phrase_words: int = 3,
    per_document: bool = True,
    max_documents: int = 25,
    stopword_language: str | None = "en",
    stopwords: list[str] | None = None,
    min_df: int | float = 1,
    max_df: int | float = 1.0,
    window: int = 4,
    random_state: int | None = 0,
) -> NlpKeyphraseResult:
    """Find the phrases that characterise a set of documents.

    Runs the chosen scorer and reports two views: the phrases that
    characterise the collection as a whole, and: optionally: the phrases that
    distinguish each individual document. The first is how you understand a
    corpus quickly; the second is how you tag or index it.

    Parameters
    ----------
    dataset:
        The dataset holding the documents.
    split_plan:
        The split defining partitions. Required unless ``partition`` is
        ``'all'``.
    partition:
        Which rows to analyse. Defaults to ``'train'``: not because holdout
        text is unsafe here, but because it is the habit worth keeping.
    method:
        ``'tfidf'``, ``'rake'``, or ``'textrank'``. See the module docstring
        for what each considers important; they will not agree, and the
        disagreement is often informative.
    text_column:
        Which column holds the documents. Inferred when omitted.
    top_n:
        How many phrases to keep per scope.
    max_phrase_words:
        Longest phrase to allow. Three captures most technical terms without
        drifting into sentence fragments.
    per_document:
        Also extract phrases for individual documents. Turn it off when you
        only want the corpus view and the corpus is large.
    max_documents:
        Cap on documents given the per-document treatment, since the output
        grows with every one.
    stopword_language:
        Built-in stopword list to apply. Defaults to English. For RAKE this is
        not cosmetic: stopwords define where phrases begin and end.
    stopwords:
        Additional terms to treat as stopwords. Add domain boilerplate here.
    min_df:
        Ignore phrases appearing in fewer than this many documents, for the
        TF-IDF method.
    max_df:
        Ignore phrases appearing in more than this share, for the TF-IDF
        method.
    window:
        Co-occurrence window for TextRank.
    random_state:
        Seed, for reproducibility where the method has a random component.

    Returns
    -------
    ~buildml.nlp.results.NlpKeyphraseResult
        Corpus keyphrases, optional per-document keyphrases, row labels, and
        the caveats.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        The method is unknown; ``top_n`` or ``max_phrase_words`` is below 1;
        ``max_documents`` is negative; the text column cannot be resolved; or
        the partition is empty.

    Notes
    -----
    **Nothing is fitted and nothing is stored.** This is description, not
    modelling: there is no plan to reuse and no model to deploy.

    **Scores are not comparable across methods or corpora.** Each method has
    its own scale. Use them to rank within one result.

    **Check ``document_frequency`` on each phrase.** A high score backed by one
    document is a phrase specific to that document, which may be exactly what
    you want or may be noise, and the score alone cannot tell you which.

    Examples
    --------
    >>> result = extract_keyphrases(dataset, split_plan, method="rake")  # doctest: +SKIP
    >>> top = result.corpus_keyphrases[:5]  # doctest: +SKIP
    >>> [(item.phrase, item.document_frequency) for item in top]  # doctest: +SKIP

    See Also
    --------
    buildml.nlp.topics.fit_topics : Recurring themes rather than salient phrases.
    buildml.nlp.profile.profile_text_corpus : Frequency and health, not salience.
    """
    method_key = str(method).lower()
    if method_key not in VALID_KEYPHRASE_METHODS:
        raise ValidationError(
            f"method={method!r} is not supported. "
            f"Choose from {list(VALID_KEYPHRASE_METHODS)}."
        )
    if top_n < 1:
        raise ValidationError("top_n must be >= 1.")
    if max_phrase_words < 1:
        raise ValidationError("max_phrase_words must be >= 1.")
    if max_documents < 0:
        raise ValidationError("max_documents must be >= 0.")

    column = resolve_text_column(dataset, text_column)
    raw_documents, frame = documents_for(
        dataset, split_plan, partition, column, operation="extract_keyphrases"
    )
    normalize_plan = build_normalize_plan(
        TextNormalizeConfig(
            steps=tuple(DEFAULT_NORMALIZE_STEPS),
            stopwords=None,
            stopword_language=None,
        )
    )
    normalized = [normalize_document(item, normalize_plan) for item in raw_documents]

    words: set[str] = set()
    if stopword_language is not None:
        words |= set(stopwords_for(stopword_language))
    if stopwords:
        words |= {str(item).strip().lower() for item in stopwords if str(item).strip()}
    stopword_set = frozenset(words)

    warnings: list[str] = []
    limit = max_documents if per_document else 0
    row_labels = tuple(frame.index[: min(limit, len(normalized))]) if limit else ()

    if method_key == "tfidf":
        corpus_scores, doc_frequency, per_doc = _tfidf_keyphrases(
            normalized,
            stopword_set,
            max_phrase_words=max_phrase_words,
            min_df=min_df,
            max_df=max_df,
        )
    elif method_key == "rake":
        corpus_scores, doc_frequency, per_doc = rake_scores(
            normalized, stopword_set, max_words=max_phrase_words
        )
    else:
        corpus_scores, doc_frequency, per_doc = _textrank_keyphrases(
            normalized,
            stopword_set,
            max_phrase_words=max_phrase_words,
            window=window,
        )

    if not corpus_scores:
        warnings.append(
            "No candidate keyphrases survived filtering; relax stopwords, "
            "min_df, or max_phrase_words."
        )

    ranked = sorted(corpus_scores.items(), key=lambda item: (-item[1], item[0]))[:top_n]
    corpus_keyphrases = tuple(
        Keyphrase(
            phrase=phrase,
            score=float(score),
            document_frequency=int(doc_frequency.get(phrase, 0)),
        )
        for phrase, score in ranked
    )

    document_keyphrases: list[tuple[Keyphrase, ...]] = []
    if limit:
        for local in per_doc[:limit]:
            local_ranked = sorted(
                local.items(), key=lambda item: (-item[1], item[0])
            )[:top_n]
            document_keyphrases.append(
                tuple(
                    Keyphrase(
                        phrase=phrase,
                        score=float(score),
                        document_frequency=int(doc_frequency.get(phrase, 0)),
                    )
                    for phrase, score in local_ranked
                )
            )
        if len(normalized) > limit:
            warnings.append(
                f"Returned per-document keyphrases for the first {limit} of "
                f"{len(normalized)} documents (raise max_documents for more)."
            )

    disclosures = [
        f"Scorer: {method_key}; candidate phrases hold at most "
        f"{max_phrase_words} content word(s) and are delimited by stopwords "
        "and punctuation.",
        f"Stopwords: {len(stopword_set)} term(s) "
        f"(language={stopword_language!r}).",
        "Bare numbers and punctuation are excluded from candidates; keyphrases "
        "are alphabetic content words.",
        "Unsupervised and descriptive: no target is consulted and nothing is "
        "persisted for later scoring.",
    ]
    if partition in {"validation", "test", "all"}:
        disclosures.append(
            f"partition={partition!r}: this describes holdout text. Do not use "
            "the result to choose features or hyperparameters."
        )
    if random_state is not None and method_key == "textrank":
        disclosures.append(
            "TextRank is deterministic here (fixed iteration count, no sampling)."
        )

    return NlpKeyphraseResult(
        partition=str(partition),
        method=method_key,
        n_rows=len(normalized),
        top_n=int(top_n),
        corpus_keyphrases=corpus_keyphrases,
        document_keyphrases=tuple(document_keyphrases),
        document_row_labels=row_labels,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )


def _tfidf_keyphrases(
    documents: list[str],
    stopwords: frozenset[str],
    *,
    max_phrase_words: int,
    min_df: int | float,
    max_df: int | float,
) -> tuple[dict[str, float], dict[str, int], list[dict[str, float]]]:
    plan = TextNormalizePlan(
        steps=(),
        stopwords=stopwords,
        min_token_length=2,
        keep_emoji=False,
        keep_numbers=False,
    )
    config = NlpVectorizeConfig(
        kind="tfidf",
        analyzer="word",
        ngram_range=(1, int(max_phrase_words)),
        max_features=None,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
    )
    vectorizer = build_sklearn_vectorizer(config, plan)
    try:
        matrix = vectorizer.fit_transform(documents)
    except ValueError:
        return {}, {}, [{} for _ in documents]
    if matrix.shape[1] == 0:
        return {}, {}, [{} for _ in documents]
    names = feature_names_for(vectorizer)
    means = np.asarray(matrix.mean(axis=0)).ravel()
    document_frequency = np.asarray((matrix > 0).sum(axis=0)).ravel()
    corpus = {
        str(names[index]): float(means[index])
        for index in range(len(names))
        if means[index] > 0
    }
    frequencies = {
        str(names[index]): int(document_frequency[index]) for index in range(len(names))
    }
    per_document: list[dict[str, float]] = []
    csr = matrix.tocsr()
    for row_index in range(csr.shape[0]):
        row = csr[row_index].tocoo()
        per_document.append(
            {
                str(names[int(column)]): float(value)
                for column, value in zip(row.col, row.data, strict=False)
                if value > 0
            }
        )
    return corpus, frequencies, per_document


def _textrank_keyphrases(
    documents: list[str],
    stopwords: frozenset[str],
    *,
    max_phrase_words: int,
    window: int,
) -> tuple[dict[str, float], dict[str, int], list[dict[str, float]]]:
    tokens_per_document = [
        [
            token
            for token in (word.lower() for word in _WORD.findall(document))
            if token not in stopwords and len(token) > 1
        ]
        for document in documents
    ]
    word_scores = textrank_word_scores(tokens_per_document, window=window)
    if not word_scores:
        return {}, {}, [{} for _ in documents]

    corpus: dict[str, float] = defaultdict(float)
    document_frequency: Counter[str] = Counter()
    per_document: list[dict[str, float]] = []
    for document in documents:
        phrases = _candidate_phrases(document, stopwords, max_words=max_phrase_words)
        local: dict[str, float] = {}
        for phrase in phrases:
            text = " ".join(phrase)
            score = float(sum(word_scores.get(word, 0.0) for word in phrase))
            local[text] = max(local.get(text, 0.0), score)
        per_document.append(local)
        for text, score in local.items():
            corpus[text] += score
            document_frequency[text] += 1
    return dict(corpus), dict(document_frequency), per_document


__all__ = [
    "VALID_KEYPHRASE_METHODS",
    "extract_keyphrases",
    "rake_scores",
    "textrank_word_scores",
]
