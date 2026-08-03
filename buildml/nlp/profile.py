"""Corpus health profiling and train/holdout text contamination screening.

Text leaks differently from tabular data. A duplicated support ticket, a
boilerplate template, or a quoted reply can put near-identical documents on both
sides of a split, which inflates holdout metrics without any obvious column-level
leak. This module measures that directly and reports it; it never silently drops
rows.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.nlp.features import char_stats, documents_for, resolve_text_column
from buildml.nlp.normalize import (
    build_normalize_plan,
    normalize_document,
    tokenize_document,
)
from buildml.nlp.results import NlpCorpusProfile
from buildml.nlp.types import DEFAULT_NORMALIZE_STEPS, NlpVectorizeConfig, TextNormalizeConfig
from buildml.nlp.vectorize import build_sklearn_vectorizer

DEFAULT_NEAR_DUPLICATE_THRESHOLD = 0.9
NEAR_DUPLICATE_MAX_FEATURES = 60_000
PARTITIONS: tuple[str, ...] = ("train", "validation", "test")


def _fingerprint(document: str) -> str:
    """Aggressively normalized key used for exact-duplicate grouping."""
    plan = build_normalize_plan(
        TextNormalizeConfig(
            steps=(
                "strip_html",
                "strip_urls",
                "strip_emails",
                "lowercase",
                "strip_punctuation",
                "collapse_whitespace",
            )
        )
    )
    return normalize_document(document, plan)


def near_duplicate_pairs(
    train_documents: list[str],
    holdout_documents: list[str],
    *,
    threshold: float,
) -> tuple[int, list[tuple[int, int, float]]]:
    """Find holdout documents that closely resemble a training document.

    Exact-duplicate detection misses the cases that matter most. A support
    ticket resubmitted with one word changed, an email quoting the message
    above it, a template with the customer name swapped — none of these match
    on equality, and all of them mean the model has effectively already seen
    the holdout document.

    Similarity is cosine distance over character 3-to-5-gram TF-IDF, fitted on
    the training documents only. Character n-grams are used deliberately: they
    survive the small edits, reorderings, and typos that defeat word-level
    comparison.

    Parameters
    ----------
    train_documents:
        The training documents, which define the representation and are the
        pool searched for matches.
    holdout_documents:
        The documents to check. Each is matched against its single nearest
        training neighbour.
    threshold:
        Cosine similarity at or above which a pair counts as a near duplicate.
        Around 0.9 catches genuine reuse while tolerating documents that merely
        share a topic; lowering it toward 0.7 starts flagging any two documents
        written in a similar style.

    Returns
    -------
    tuple
        ``(n_matches, matches)`` where each match is
        ``(holdout_index, train_index, similarity)``, indexed into the lists as
        passed. The indices are what let you inspect the offending pairs rather
        than just count them.

    Notes
    -----
    Each holdout document is compared against its nearest training neighbour
    only, so the count is documents-with-a-match rather than total pairs.

    An empty input list, or a corpus that yields no character features at all,
    returns zero matches rather than raising — an unprofilable corpus is not an
    error, and the caller reports the absence.

    The search is brute-force. On very large corpora it is the expensive part
    of profiling.

    See Also
    --------
    profile_text_corpus : Runs this as part of a fuller corpus screen.
    """
    if not train_documents or not holdout_documents:
        return 0, []
    normalize_plan = build_normalize_plan(
        TextNormalizeConfig(
            steps=("strip_html", "strip_urls", "strip_emails", "lowercase", "collapse_whitespace")
        )
    )
    config = NlpVectorizeConfig(
        kind="tfidf",
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=NEAR_DUPLICATE_MAX_FEATURES,
        min_df=1,
        max_df=1.0,
        sublinear_tf=True,
    )
    vectorizer = build_sklearn_vectorizer(config, normalize_plan)
    try:
        train_matrix = vectorizer.fit_transform(train_documents)
        holdout_matrix = vectorizer.transform(holdout_documents)
    except ValueError:
        return 0, []
    if train_matrix.shape[1] == 0:
        return 0, []

    from sklearn.neighbors import NearestNeighbors

    neighbours = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute")
    neighbours.fit(train_matrix)
    distances, indices = neighbours.kneighbors(holdout_matrix, n_neighbors=1)
    similarities = 1.0 - np.asarray(distances, dtype=float).ravel()
    train_positions = np.asarray(indices, dtype=int).ravel()

    matches: list[tuple[int, int, float]] = []
    for holdout_index, similarity in enumerate(similarities):
        if similarity >= threshold:
            matches.append(
                (int(holdout_index), int(train_positions[holdout_index]), float(similarity))
            )
    return len(matches), matches


def profile_text_corpus(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    text_column: str | None = None,
    top_tokens: int = 25,
    near_duplicate_threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
    detect_languages: bool = True,
    stopword_language: str | None = None,
) -> NlpCorpusProfile:
    """Understand a text column before modelling it, and check the split is clean.

    Two jobs in one pass.

    The **profile** describes the corpus: how many documents, how long they
    are, how much of the column is blank, the vocabulary size, the most common
    tokens, and which languages appear. This is where you find out that a
    fifth of your documents are empty, or that the corpus is a third German,
    or that the most frequent token is a template header appearing in every
    single row.

    The **contamination screen** compares partitions for duplicate and
    near-duplicate text. This is the leak that tabular checks cannot see: no
    column is shared, no statistic crosses the boundary, but the same document
    sits on both sides of the split and the holdout score is meaningless.
    Nothing is removed — the finding is reported and the decision is yours,
    because whether a duplicate is a data-collection artefact or a genuine
    repeated event depends on what you are modelling.

    Parameters
    ----------
    dataset:
        The dataset holding the text column.
    split_plan:
        The split to screen. Optional — without one you still get the corpus
        profile, but no contamination check, since there are no partitions to
        compare.
    text_column:
        Which column to profile. Inferred from roles and dtype when omitted.
    top_tokens:
        How many of the most frequent tokens to report. Scanning these is the
        fastest way to spot boilerplate that should be stripped or added to a
        stopword list.
    near_duplicate_threshold:
        Cosine similarity at or above which two documents count as near
        duplicates. Lower it to catch looser paraphrasing, at the cost of
        flagging documents that merely share a subject.
    detect_languages:
        Also identify the language of each document. Worth leaving on: a
        multilingual corpus modelled as if monolingual produces a vocabulary
        that is mostly one language plus noise.
    stopword_language:
        Exclude this language's stopwords from the frequency counts, so the top
        tokens are informative rather than "the", "of", "and".

    Returns
    -------
    ~buildml.nlp.results.NlpCorpusProfile
        Per-partition sizes, length and vocabulary statistics, top tokens,
        detected languages, exact and near-duplicate counts with the offending
        pairs, and warnings for anything worth acting on.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        ``top_tokens`` is below 1, ``near_duplicate_threshold`` is outside
        ``(0, 1]``, or the text column cannot be resolved.

    Notes
    -----
    **Run this before fitting, not after.** Every problem it surfaces —
    contamination, blank documents, an unexpected second language — changes
    what you should do next, and finding out afterwards means refitting.

    **Near-duplicate detection is the expensive step.** It vectorises the
    corpus and runs a brute-force nearest-neighbour search. On a very large
    corpus, profile a sample first.

    **Duplicates are reported, never dropped.** Two identical documents may be
    a collection artefact or may be two genuinely separate events that happen
    to read the same. Only you can tell.

    Examples
    --------
    >>> profile = profile_text_corpus(dataset, split_plan)  # doctest: +SKIP
    >>> profile.near_duplicate_count, profile.top_tokens[:3]  # doctest: +SKIP

    See Also
    --------
    near_duplicate_pairs : The contamination check on its own.
    buildml.nlp.fit.fit_text_classifier : The next step, once the corpus is understood.
    """
    if top_tokens < 1:
        raise ValidationError("top_tokens must be >= 1.")
    if not 0.0 < near_duplicate_threshold <= 1.0:
        raise ValidationError("near_duplicate_threshold must be within (0.0, 1.0].")

    column = resolve_text_column(dataset, text_column)
    all_documents, frame = documents_for(
        dataset, split_plan, "all", column, operation="profile_text_corpus"
    )
    normalize_plan = build_normalize_plan(
        TextNormalizeConfig(
            steps=tuple(DEFAULT_NORMALIZE_STEPS),
            stopword_language=stopword_language,
        )
    )

    partition_sizes: dict[str, int] = {"all": len(all_documents)}
    partition_documents: dict[str, list[str]] = {}
    if split_plan is not None:
        for name in PARTITIONS:
            indices = list(split_plan.indices_for(name))  # type: ignore[arg-type]
            partition_sizes[name] = len(indices)
            if indices:
                partition_documents[name] = (
                    frame.iloc[indices][column]
                    .astype("string")
                    .fillna("")
                    .astype(str)
                    .tolist()
                )

    blank = sum(1 for item in all_documents if not item.strip())
    token_lists = [tokenize_document(item, normalize_plan) for item in all_documents]
    counter: Counter[str] = Counter()
    for tokens in token_lists:
        counter.update(tokens)
    total_tokens = int(sum(counter.values()))
    vocabulary = len(counter)
    hapax = sum(1 for count in counter.values() if count == 1)
    token_counts = [len(tokens) for tokens in token_lists]
    token_array = np.asarray(token_counts, dtype=float) if token_counts else np.zeros(1)

    fingerprints = [_fingerprint(item) for item in all_documents]
    fingerprint_counts = Counter(item for item in fingerprints if item)
    duplicate_groups = sum(1 for count in fingerprint_counts.values() if count > 1)
    duplicated_rows = sum(count for count in fingerprint_counts.values() if count > 1)

    findings: list[str] = []
    warnings: list[str] = []
    disclosures: list[str] = [
        f"Profiled column {column!r} over {len(all_documents)} document(s).",
        "Duplicate detection compares lowercased, punctuation-stripped text.",
        "Near-duplicate screening fits character 3-5-gram TF-IDF on train only, "
        "then measures cosine similarity of holdout documents against train.",
        "This is a diagnostic: nothing is dropped, deduplicated, or rewritten.",
    ]

    exact_overlap = 0
    near_duplicates = 0
    holdout_oov: float | None = None
    train_documents = partition_documents.get("train", [])
    holdout_documents = [
        document
        for name in ("validation", "test")
        for document in partition_documents.get(name, [])
    ]
    if train_documents and holdout_documents:
        train_fingerprints = {
            _fingerprint(item) for item in train_documents if _fingerprint(item)
        }
        holdout_fingerprints = [_fingerprint(item) for item in holdout_documents]
        exact_overlap = sum(
            1
            for item in holdout_fingerprints
            if item and item in train_fingerprints
        )
        near_duplicates, _matches = near_duplicate_pairs(
            train_documents,
            holdout_documents,
            threshold=float(near_duplicate_threshold),
        )
        holdout_oov = _holdout_oov_rate(train_documents, holdout_documents, normalize_plan)

        holdout_size = max(1, len(holdout_documents))
        if exact_overlap:
            findings.append(
                f"{exact_overlap} of {holdout_size} holdout document(s) are exact "
                "duplicates of a train document; holdout metrics are optimistic by "
                "that amount."
            )
            warnings.append(
                "Exact train/holdout document overlap detected — deduplicate the "
                "corpus before splitting, or split by document group."
            )
        if near_duplicates > exact_overlap:
            findings.append(
                f"{near_duplicates} of {holdout_size} holdout document(s) have a "
                f"train document at cosine similarity >= {near_duplicate_threshold:.2f}."
            )
            warnings.append(
                "Near-duplicate text spans the split; consider grouping by source "
                "document or thread with Session.split(group_column=...)."
            )
        if holdout_oov is not None and holdout_oov > 0.35:
            findings.append(
                f"{holdout_oov:.1%} of holdout tokens are unseen in train; the "
                "vocabulary does not transfer well."
            )
    elif split_plan is None:
        disclosures.append(
            "No SplitPlan is attached, so train/holdout contamination could not be "
            "screened. Call Session.split(...) first for the leakage checks."
        )

    if blank:
        findings.append(
            f"{blank} document(s) are blank ({blank / max(1, len(all_documents)):.1%})."
        )
    if duplicate_groups:
        findings.append(
            f"{duplicate_groups} duplicate group(s) covering {duplicated_rows} row(s)."
        )
    if vocabulary and hapax / vocabulary > 0.6:
        findings.append(
            f"{hapax / vocabulary:.1%} of vocabulary terms appear in exactly one "
            "document; raise min_df or use a character analyzer."
        )
    if token_counts and float(token_array.mean()) < 5.0:
        findings.append(
            f"Mean document length is {float(token_array.mean()):.1f} tokens; word "
            "n-grams will be very sparse."
        )

    language_counts: dict[str, int] = {}
    if detect_languages:
        from buildml.nlp.language import detect_document_language

        language_counter: Counter[str] = Counter()
        for document in all_documents:
            code, _confidence = detect_document_language(document)
            language_counter[code] += 1
        language_counts = {key: int(language_counter[key]) for key in sorted(language_counter)}
        known = {key: value for key, value in language_counts.items() if key != "und"}
        if len(known) > 1:
            dominant = max(known, key=lambda key: known[key])
            share = known[dominant] / max(1, len(all_documents))
            if share < 0.9:
                findings.append(
                    f"Corpus is multilingual: dominant language {dominant!r} covers "
                    f"{share:.1%} of documents."
                )
        disclosures.append(
            "Language counts come from the native detector; codes for non-Latin "
            "scripts identify the script family."
        )

    return NlpCorpusProfile(
        text_column=column,
        partitions=partition_sizes,
        n_documents=len(all_documents),
        n_empty=blank,
        empty_rate=float(blank / max(1, len(all_documents))),
        document_length_chars=char_stats(all_documents),
        document_length_tokens={
            "mean": float(token_array.mean()),
            "median": float(np.median(token_array)),
            "p95": float(np.percentile(token_array, 95)),
            "max": float(token_array.max()),
        },
        vocabulary_size=vocabulary,
        hapax_rate=float(hapax / vocabulary) if vocabulary else 0.0,
        type_token_ratio=float(vocabulary / total_tokens) if total_tokens else 0.0,
        top_tokens=tuple(counter.most_common(int(top_tokens))),
        duplicate_document_groups=duplicate_groups,
        duplicate_document_rate=float(duplicated_rows / max(1, len(all_documents))),
        train_holdout_exact_overlap=exact_overlap,
        train_holdout_near_duplicate=near_duplicates,
        near_duplicate_threshold=float(near_duplicate_threshold),
        holdout_oov_token_rate=holdout_oov,
        language_counts=language_counts,
        findings=tuple(findings),
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )


def _holdout_oov_rate(
    train_documents: list[str],
    holdout_documents: list[str],
    normalize_plan,
) -> float | None:
    known: set[str] = set()
    for document in train_documents:
        known.update(tokenize_document(document, normalize_plan))
    total = 0
    unseen = 0
    for document in holdout_documents:
        for token in tokenize_document(document, normalize_plan):
            total += 1
            if token not in known:
                unseen += 1
    if total == 0:
        return None
    return float(unseen / total)


__all__ = [
    "DEFAULT_NEAR_DUPLICATE_THRESHOLD",
    "near_duplicate_pairs",
    "profile_text_corpus",
]
