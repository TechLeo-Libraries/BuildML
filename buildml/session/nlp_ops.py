"""Thin Session facades over buildml.nlp."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from buildml.core.errors import ValidationError
from buildml.data.splits import PartitionName
from buildml.nlp.checkpoint import load_nlp_bundle, save_nlp_bundle
from buildml.nlp.entities import extract_entities
from buildml.nlp.evaluate import evaluate_text_classifier
from buildml.nlp.explain_hooks import (
    entity_result_summary,
    eval_result_summary,
    fit_result_summary,
    interpret_result_summary,
    keyphrase_result_summary,
    language_result_summary,
    predict_result_summary,
    profile_result_summary,
    sentiment_result_summary,
    summary_result_summary,
    topic_assign_summary,
    topic_result_summary,
)
from buildml.nlp.fit import fit_text_classifier
from buildml.nlp.interpret import interpret_text_prediction
from buildml.nlp.keyphrases import extract_keyphrases
from buildml.nlp.language import detect_language
from buildml.nlp.predict import predict_text
from buildml.nlp.profile import profile_text_corpus
from buildml.nlp.sentiment import analyze_sentiment
from buildml.nlp.summarize import summarize_text
from buildml.nlp.topics import assign_topics, fit_topics

PartitionOrAll = PartitionName | Literal["all"]


def nlp_capability_matrix_op() -> dict[str, Any]:
    """Honest capability matrix for NLP backends and task surfaces.

    Thin Session facade over ``buildml.nlp``; records the call and stores NLP artifacts on the Session for follow-up steps.

    Returns
    -------
    dict[str, Any]
        Structured domain result recorded on the Session for follow-up evaluate/explain/export steps.
    """
    from buildml.nlp.catalog import nlp_capability_matrix

    return nlp_capability_matrix()


def _require_text_plan(session) -> Any:
    plan = getattr(session, "_nlp_text_plan", None)
    if plan is None:
        raise ValidationError(
            "No NLP text plan. Call fit_text_classifier(...) or "
            "load_nlp_bundle(...) first."
        )
    return plan


def _require_topic_plan(session) -> Any:
    plan = getattr(session, "_nlp_topic_plan", None)
    if plan is None:
        raise ValidationError(
            "No NLP topic plan. Call fit_topics(...) or load_nlp_bundle(...) first."
        )
    return plan


def profile_text_corpus_op(
    session,
    *,
    text_column: str | None = None,
    top_tokens: int = 25,
    near_duplicate_threshold: float = 0.9,
    detect_languages: bool = True,
    stopword_language: str | None = None,
) -> Any:
    """Profile corpus health and screen the split for text contamination.

    Thin Session facade over ``buildml.nlp``; records the call and stores NLP artifacts on the Session for follow-up steps.

    Parameters
    ----------
    session:
        Active Session with dataset and optional split plan attached.
    text_column:
        Text column name; defaults to the sole text-role column.
    top_tokens:
        How many highest-frequency tokens to surface in the corpus profile. Raise it for a broader vocabulary snapshot; lower it for a short health check.
    near_duplicate_threshold:
        Similarity cutoff above which two documents are flagged as near-duplicates. Closer to 1.0 is stricter; lower values catch paraphrases but add noise.
    detect_languages:
        When True, run language identification during profiling and report per-language counts. Turn off for monolingual corpora to skip that cost.
    stopword_language:
        Language key for the stopword list used in token stats (for example ``"english"``). ``None`` skips stopword filtering.

    Returns
    -------
    Any
        Structured domain result recorded on the Session for follow-up evaluate/explain/export steps.
    """
    result = profile_text_corpus(
        session.dataset,
        session._split_plan,
        text_column=text_column,
        top_tokens=top_tokens,
        near_duplicate_threshold=near_duplicate_threshold,
        detect_languages=detect_languages,
        stopword_language=stopword_language,
    )
    session._nlp_profile_result = result
    session._record(
        "profile_text_corpus",
        {
            "text_column": text_column,
            "top_tokens": top_tokens,
            "near_duplicate_threshold": near_duplicate_threshold,
            "detect_languages": detect_languages,
            "stopword_language": stopword_language,
        },
        warnings=tuple(result.warnings),
        result_summary=profile_result_summary(result),
    )
    return result


def detect_language_op(
    session,
    *,
    partition: PartitionOrAll = "all",
    backend: str | None = "native",
    text_column: str | None = None,
    min_characters: int = 12,
) -> Any:
    """Identify the language of every document in a partition.

    Thin Session facade over ``buildml.nlp``; records the call and stores NLP artifacts on the Session for follow-up steps.

    Parameters
    ----------
    session:
        Active Session with dataset and optional split plan attached.
    partition:
        Split partition to read or score.
    backend:
        Backend identifier; see capability matrix for valid values.
    text_column:
        Text column name; defaults to the sole text-role column.
    min_characters:
        Minimum character length before a document is language-classified. Short strings are skipped because language ID is unreliable on tiny text.

    Returns
    -------
    Any
        Structured domain result recorded on the Session for follow-up evaluate/explain/export steps.
    """
    result = detect_language(
        session.dataset,
        session._split_plan,
        partition=partition,
        backend=backend,
        text_column=text_column,
        min_characters=min_characters,
    )
    session._nlp_language_result = result
    session._record(
        "detect_language",
        {
            "partition": partition,
            "backend": backend,
            "text_column": text_column,
            "min_characters": min_characters,
        },
        warnings=tuple(result.warnings),
        result_summary=language_result_summary(result),
    )
    return result


def fit_text_classifier_op(
    session,
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
) -> Any:
    """Fit a single-label document classifier on Session train.

    Thin Session facade over ``buildml.nlp``; records the call and stores NLP artifacts on the Session for follow-up steps.

    Parameters
    ----------
    session:
        Active Session with dataset and optional split plan attached.
    backend:
        Backend identifier; see capability matrix for valid values.
    estimator:
        Unfitted sklearn-compatible estimator instance.
    text_column:
        Text column name; defaults to the sole text-role column.
    vectorizer:
        Bag vectorizer family for bag-of-n-grams paths (for example ``"tfidf"`` or ``"count"``). TF-IDF down-weights corpus-wide tokens; count keeps raw frequencies.
    analyzer:
        Tokenization unit for the vectorizer: ``"word"`` (default) or ``"char"`` / ``"char_wb"`` for character n-grams when morphology or misspellings matter.
    ngram_range:
        Inclusive ``(min_n, max_n)`` n-gram window. ``(1, 1)`` is unigrams only; ``(1, 2)`` adds bigrams and usually helps short-text classification at higher sparsity cost.
    max_features:
        Cap on vocabulary size after frequency ranking. ``None`` keeps all terms above ``min_df``; a finite cap controls memory and noise from rare tokens.
    min_df:
        Minimum document frequency for a term to enter the vocabulary (integer count or fraction of documents). Higher values drop rare/noisy tokens.
    max_df:
        Maximum document frequency for a term (count or fraction). Lower it to strip overly common tokens that behave like stopwords.
    sublinear_tf:
        When True (TF-IDF), replace raw term frequency with ``1 + log(tf)`` to dampen very frequent tokens.
    binary:
        When True, term values become 0/1 presence indicators instead of counts or TF-IDF weights.
    n_hash_features:
        Number of feature-hashing buckets for high-cardinality text/categorical hashing encoders.
    normalize_steps:
        Ordered text-normalization steps to apply before vectorization (lowercase, unicode cleanup, whitespace collapse, …).
    stopwords:
        Stopword list name, custom iterable, or ``None``. Removing common function words often helps bag-of-words models.
    stopword_language:
        Language key for the stopword list used in token stats (for example ``"english"``). ``None`` skips stopword filtering.
    min_token_length:
        Minimum token length retained after tokenization. Shorter tokens are discarded as noise.
    max_token_length:
        Maximum token length retained after tokenization. Longer tokens are discarded or truncated.
    stem:
        When True, apply stemming after tokenization. Useful for bag-of-words; usually off for embeddings.
    lemmatize:
        When True, lemmatize tokens (requires the NLP morphology extra/backend). Usually preferable to crude stemming when available.
    class_weight:
        Class reweighting strategy (``"balanced"``, a per-class dict, or ``None``). Use ``"balanced"`` when minority classes are underrepresented in train.
    C:
        Inverse regularization strength for linear / SVM-style models. Larger ``C`` fits training data more tightly; smaller ``C`` prefers simpler boundaries.
    alpha:
        Regularization strength for penalized linear models (larger = stronger penalty).
    embedding_model_name:
        Sentence-transformer / embedding model id used to encode documents.
    max_seq_tokens:
        Maximum tokens per document for transformer encode/fit loops. Longer inputs are truncated.
    device:
        Compute device string (``"cpu"``, ``"cuda"``, ``"mps"``, …) for torch-backed paths.
    random_state:
        Seed for randomized fitting steps so re-runs are comparable. ``None`` leaves RNG undeterministic.

    Returns
    -------
    Any
        Structured domain result recorded on the Session for follow-up evaluate/explain/export steps.

    Notes
    -----
    **Leakage:** Requires a split. Normalization, vocabulary, document
    frequencies, and the head are all fitted on train only. Honesty: document
    classification - not sequence labelling, not generation, not RAG.
    """
    session.assert_can_fit("train")
    plan, result = fit_text_classifier(
        session.dataset,
        session._split_plan,
        backend=backend,
        estimator=estimator,
        text_column=text_column,
        vectorizer=vectorizer,
        analyzer=analyzer,
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        binary=binary,
        n_hash_features=n_hash_features,
        normalize_steps=normalize_steps,
        stopwords=stopwords,
        stopword_language=stopword_language,
        min_token_length=min_token_length,
        max_token_length=max_token_length,
        stem=stem,
        lemmatize=lemmatize,
        class_weight=class_weight,
        C=C,
        alpha=alpha,
        embedding_model_name=embedding_model_name,
        max_seq_tokens=max_seq_tokens,
        device=device,
        random_state=random_state,
    )
    session._nlp_text_plan = plan
    session._nlp_fit_result = result
    session._nlp_eval_result = None
    session._nlp_predict_result = None
    session._nlp_interpret_result = None
    session._record(
        "fit_text_classifier",
        {
            "backend": backend,
            "estimator": estimator,
            "text_column": text_column,
            "vectorizer": vectorizer,
            "analyzer": analyzer,
            "ngram_range": list(ngram_range),
            "max_features": max_features,
            "min_df": min_df,
            "max_df": max_df,
            "sublinear_tf": sublinear_tf,
            "binary": binary,
            "n_hash_features": n_hash_features,
            "normalize_steps": normalize_steps,
            "stopwords": None if stopwords is None else len(stopwords),
            "stopword_language": stopword_language,
            "min_token_length": min_token_length,
            "max_token_length": max_token_length,
            "stem": stem,
            "lemmatize": lemmatize,
            "class_weight": class_weight,
            "C": C,
            "alpha": alpha,
            "embedding_model_name": embedding_model_name,
            "max_seq_tokens": max_seq_tokens,
            "device": device,
            "random_state": random_state,
        },
        warnings=tuple(result.warnings),
        result_summary=fit_result_summary(result),
    )
    return result


def predict_text_op(
    session,
    *,
    partition: PartitionOrAll = "test",
    return_probabilities: bool = True,
) -> Any:
    """Score a partition with the train-fitted text plan.

    Thin Session facade over ``buildml.nlp``; records the call and stores NLP artifacts on the Session for follow-up steps.

    Parameters
    ----------
    session:
        Active Session with dataset and optional split plan attached.
    partition:
        Split partition to read or score.
    return_probabilities:
        When True, include class probabilities in the prediction/result payload.

    Returns
    -------
    Any
        Structured domain result recorded on the Session for follow-up evaluate/explain/export steps.
    """
    plan = _require_text_plan(session)
    result = predict_text(
        session.dataset,
        plan,
        session._split_plan,
        partition=partition,
        return_probabilities=return_probabilities,
    )
    session._nlp_predict_result = result
    session._record(
        "predict_text",
        {"partition": partition, "return_probabilities": return_probabilities},
        warnings=tuple(result.warnings),
        result_summary=predict_result_summary(result),
    )
    return result


def evaluate_text_classifier_op(
    session,
    *,
    partition: PartitionOrAll = "validation",
) -> Any:
    """Evaluate the text classifier on a holdout partition.

    Thin Session facade over ``buildml.nlp``; records the call and stores NLP artifacts on the Session for follow-up steps.

    Parameters
    ----------
    session:
        Active Session with dataset and optional split plan attached.
    partition:
        Split partition to read or score.

    Returns
    -------
    Any
        Structured domain result recorded on the Session for follow-up evaluate/explain/export steps.
    """
    plan = _require_text_plan(session)
    from buildml.nlp.features import resolve_holdout_partition

    resolved = resolve_holdout_partition(session._split_plan, partition)
    result = evaluate_text_classifier(
        session.dataset,
        plan,
        session._split_plan,
        partition=resolved,
    )
    session._nlp_eval_result = result
    session._record(
        "evaluate_text_classifier",
        {"partition": resolved},
        warnings=tuple(result.warnings),
        result_summary=eval_result_summary(result),
    )
    return result


def interpret_text_prediction_op(
    session,
    *,
    partition: PartitionOrAll = "test",
    target_class: Any = None,
    top_k: int = 12,
    max_documents: int = 10,
    include_global: bool = True,
) -> Any:
    """Explain document decisions with per-token contributions.

    Thin Session facade over ``buildml.nlp``; records the call and stores NLP artifacts on the Session for follow-up steps.

    Parameters
    ----------
    session:
        Active Session with dataset and optional split plan attached.
    partition:
        Split partition to read or score.
    target_class:
        Class label treated as the positive / focus class for binary metrics or explanations.
    top_k:
        How many retrieved chunks / candidates to keep. Higher recall costs more context and noise.
    max_documents:
        Maximum documents to profile or process in one call. Lower it for a cheap health check on huge corpora.
    include_global:
        When True, attach global explanation summaries in addition to local ones.

    Returns
    -------
    Any
        Structured domain result recorded on the Session for follow-up evaluate/explain/export steps.
    """
    plan = _require_text_plan(session)
    result = interpret_text_prediction(
        session.dataset,
        plan,
        session._split_plan,
        partition=partition,
        target_class=target_class,
        top_k=top_k,
        max_documents=max_documents,
        include_global=include_global,
    )
    session._nlp_interpret_result = result
    session._record(
        "interpret_text_prediction",
        {
            "partition": partition,
            "target_class": target_class,
            "top_k": top_k,
            "max_documents": max_documents,
            "include_global": include_global,
        },
        warnings=tuple(result.warnings),
        result_summary=interpret_result_summary(result),
    )
    return result


def fit_topics_op(
    session,
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
) -> Any:
    """Fit an unsupervised topic model on Session train documents.

    Thin Session facade over ``buildml.nlp``; records the call and stores NLP artifacts on the Session for follow-up steps.

    Parameters
    ----------
    session:
        Active Session with dataset and optional split plan attached.
    method:
        Algorithm or method identifier for the resolved backend.
    n_topics:
        Number of topics for topic-model fits. Too many fragments themes; too few collapses them.
    text_column:
        Text column name; defaults to the sole text-role column.
    top_terms:
        How many top terms/features to surface in explanations or topic summaries.
    max_features:
        Cap on vocabulary size after frequency ranking. ``None`` keeps all terms above ``min_df``; a finite cap controls memory and noise from rare tokens.
    min_df:
        Minimum document frequency for a term to enter the vocabulary (integer count or fraction of documents). Higher values drop rare/noisy tokens.
    max_df:
        Maximum document frequency for a term (count or fraction). Lower it to strip overly common tokens that behave like stopwords.
    ngram_range:
        Inclusive ``(min_n, max_n)`` n-gram window. ``(1, 1)`` is unigrams only; ``(1, 2)`` adds bigrams and usually helps short-text classification at higher sparsity cost.
    normalize_steps:
        Ordered text-normalization steps to apply before vectorization (lowercase, unicode cleanup, whitespace collapse, …).
    stopwords:
        Stopword list name, custom iterable, or ``None``. Removing common function words often helps bag-of-words models.
    stopword_language:
        Language key for the stopword list used in token stats (for example ``"english"``). ``None`` skips stopword filtering.
    stem:
        When True, apply stemming after tokenization. Useful for bag-of-words; usually off for embeddings.
    max_iter:
        Hard cap on solver iterations or boosting rounds. Raise it when fits do not converge.
    random_state:
        Seed for randomized fitting steps so re-runs are comparable. ``None`` leaves RNG undeterministic.

    Returns
    -------
    Any
        Structured domain result recorded on the Session for follow-up evaluate/explain/export steps.

    Notes
    -----
    **Leakage:** Requires a split. The vectorizer and decomposition are fitted on
    train only, so ``assign_topics`` on holdout is a pure transform.
    """
    session.assert_can_fit("train")
    plan, result = fit_topics(
        session.dataset,
        session._split_plan,
        method=method,
        n_topics=n_topics,
        text_column=text_column,
        top_terms=top_terms,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        normalize_steps=normalize_steps,
        stopwords=stopwords,
        stopword_language=stopword_language,
        stem=stem,
        max_iter=max_iter,
        random_state=random_state,
    )
    session._nlp_topic_plan = plan
    session._nlp_topic_result = result
    session._nlp_topic_assign_result = None
    session._record(
        "fit_topics",
        {
            "method": method,
            "n_topics": n_topics,
            "text_column": text_column,
            "top_terms": top_terms,
            "max_features": max_features,
            "min_df": min_df,
            "max_df": max_df,
            "ngram_range": list(ngram_range),
            "normalize_steps": normalize_steps,
            "stopwords": None if stopwords is None else len(stopwords),
            "stopword_language": stopword_language,
            "stem": stem,
            "max_iter": max_iter,
            "random_state": random_state,
        },
        warnings=tuple(result.warnings),
        result_summary=topic_result_summary(result),
    )
    return result


def assign_topics_op(
    session,
    *,
    partition: PartitionOrAll = "test",
) -> Any:
    """Transform a partition into per-document topic weights.

    Thin Session facade over ``buildml.nlp``; records the call and stores NLP artifacts on the Session for follow-up steps.

    Parameters
    ----------
    session:
        Active Session with dataset and optional split plan attached.
    partition:
        Split partition to read or score.

    Returns
    -------
    Any
        Structured domain result recorded on the Session for follow-up evaluate/explain/export steps.
    """
    plan = _require_topic_plan(session)
    result = assign_topics(
        session.dataset,
        plan,
        session._split_plan,
        partition=partition,
    )
    session._nlp_topic_assign_result = result
    session._record(
        "assign_topics",
        {"partition": partition},
        warnings=tuple(result.warnings),
        result_summary=topic_assign_summary(result),
    )
    return result


def extract_keyphrases_op(
    session,
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
) -> Any:
    """Rank keyphrases for a partition with an unsupervised scorer.

    Thin Session facade over ``buildml.nlp``; records the call and stores NLP artifacts on the Session for follow-up steps.

    Parameters
    ----------
    session:
        Active Session with dataset and optional split plan attached.
    partition:
        Split partition to read or score.
    method:
        Algorithm or method identifier for the resolved backend.
    text_column:
        Text column name; defaults to the sole text-role column.
    top_n:
        How many top items/rows/terms to keep in the result table.
    max_phrase_words:
        Maximum words allowed in extracted keyphrases.
    per_document:
        When True, return per-document outputs instead of a corpus-level aggregate.
    max_documents:
        Maximum documents to profile or process in one call. Lower it for a cheap health check on huge corpora.
    stopword_language:
        Language key for the stopword list used in token stats (for example ``"english"``). ``None`` skips stopword filtering.
    stopwords:
        Stopword list name, custom iterable, or ``None``. Removing common function words often helps bag-of-words models.
    min_df:
        Minimum document frequency for a term to enter the vocabulary (integer count or fraction of documents). Higher values drop rare/noisy tokens.
    max_df:
        Maximum document frequency for a term (count or fraction). Lower it to strip overly common tokens that behave like stopwords.
    window:
        Sliding window size (in tokens/characters as documented by the NLP backend)
        used when scoring local keyphrase or co-occurrence context.
    random_state:
        Seed for randomized fitting steps so re-runs are comparable. ``None`` leaves RNG undeterministic.

    Returns
    -------
    Any
        Structured domain result recorded on the Session for follow-up evaluate/explain/export steps.
    """
    result = extract_keyphrases(
        session.dataset,
        session._split_plan,
        partition=partition,
        method=method,
        text_column=text_column,
        top_n=top_n,
        max_phrase_words=max_phrase_words,
        per_document=per_document,
        max_documents=max_documents,
        stopword_language=stopword_language,
        stopwords=stopwords,
        min_df=min_df,
        max_df=max_df,
        window=window,
        random_state=random_state,
    )
    session._nlp_keyphrase_result = result
    session._record(
        "extract_keyphrases",
        {
            "partition": partition,
            "method": method,
            "text_column": text_column,
            "top_n": top_n,
            "max_phrase_words": max_phrase_words,
            "per_document": per_document,
            "max_documents": max_documents,
            "stopword_language": stopword_language,
            "stopwords": None if stopwords is None else len(stopwords),
            "min_df": min_df,
            "max_df": max_df,
            "window": window,
            "random_state": random_state,
        },
        warnings=tuple(result.warnings),
        result_summary=keyphrase_result_summary(result),
    )
    return result


def analyze_sentiment_op(
    session,
    *,
    partition: PartitionOrAll = "test",
    backend: str = "lexicon",
    text_column: str | None = None,
    threshold: float = 0.05,
    compare_to_target: bool = False,
    transformer_model: str = "distilbert-base-uncased-finetuned-sst-2-english",
    device: str = "cpu",
) -> Any:
    """Score a partition's documents for sentiment.

    Thin Session facade over ``buildml.nlp``; records the call and stores NLP artifacts on the Session for follow-up steps.

    Parameters
    ----------
    session:
        Active Session with dataset and optional split plan attached.
    partition:
        Split partition to read or score.
    backend:
        Backend identifier; see capability matrix for valid values.
    text_column:
        Text column name; defaults to the sole text-role column.
    threshold:
        Decision cutoff on scores or probabilities. Raise it to flag fewer positives.
    compare_to_target:
        When True, compare predictions or retrieved labels against the Session target for metrics.
    transformer_model:
        Hugging Face transformer model id for sequence-classification / NER / generative NLP paths.
    device:
        Compute device string (``"cpu"``, ``"cuda"``, ``"mps"``, …) for torch-backed paths.

    Returns
    -------
    Any
        Structured domain result recorded on the Session for follow-up evaluate/explain/export steps.
    """
    text_plan = getattr(session, "_nlp_text_plan", None)
    result = analyze_sentiment(
        session.dataset,
        session._split_plan,
        partition=partition,
        backend=backend,
        text_column=text_column,
        threshold=threshold,
        text_plan=text_plan,
        compare_to_target=compare_to_target,
        transformer_model=transformer_model,
        device=device,
    )
    session._nlp_sentiment_result = result
    session._record(
        "analyze_sentiment",
        {
            "partition": partition,
            "backend": backend,
            "text_column": text_column,
            "threshold": threshold,
            "compare_to_target": compare_to_target,
            "transformer_model": transformer_model,
            "device": device,
        },
        warnings=tuple(result.warnings),
        result_summary=sentiment_result_summary(result),
    )
    return result


def extract_entities_op(
    session,
    *,
    partition: PartitionOrAll = "test",
    backend: str | None = "rules",
    text_column: str | None = None,
    labels: list[str] | None = None,
    gazetteers: dict[str, list[str]] | None = None,
    spacy_model: str = "en_core_web_sm",
    max_documents: int = 25,
    batch_size: int = 32,
) -> Any:
    """Extract entity mentions from a partition's documents.

    Thin Session facade over ``buildml.nlp``; records the call and stores NLP artifacts on the Session for follow-up steps.

    Parameters
    ----------
    session:
        Active Session with dataset and optional split plan attached.
    partition:
        Split partition to read or score.
    backend:
        Backend identifier; see capability matrix for valid values.
    text_column:
        Text column name; defaults to the sole text-role column.
    labels:
        Explicit label order for metrics/confusion matrices. ``None`` uses labels observed in the scored partition.
    gazetteers:
        Optional gazetteer dictionaries used by rule/NER backends to boost entity matches.
    spacy_model:
        spaCy pipeline name for industry NLP backends. Must be installed separately from BuildML.
    max_documents:
        Maximum documents to profile or process in one call. Lower it for a cheap health check on huge corpora.
    batch_size:
        Mini-batch size for transformer/embedding encode or train loops.

    Returns
    -------
    Any
        Structured domain result recorded on the Session for follow-up evaluate/explain/export steps.
    """
    result = extract_entities(
        session.dataset,
        session._split_plan,
        partition=partition,
        backend=backend,
        text_column=text_column,
        labels=labels,
        gazetteers=gazetteers,
        spacy_model=spacy_model,
        max_documents=max_documents,
        batch_size=batch_size,
    )
    session._nlp_entity_result = result
    session._record(
        "extract_entities",
        {
            "partition": partition,
            "backend": backend,
            "text_column": text_column,
            "labels": labels,
            "gazetteers": None if gazetteers is None else sorted(gazetteers),
            "spacy_model": spacy_model,
            "max_documents": max_documents,
            "batch_size": batch_size,
        },
        warnings=tuple(result.warnings),
        result_summary=entity_result_summary(result),
    )
    return result


def summarize_text_op(
    session,
    *,
    partition: PartitionOrAll = "test",
    method: str = "textrank",
    text_column: str | None = None,
    n_sentences: int = 3,
    max_documents: int = 25,
    max_input_sentences: int = 200,
    stopword_language: str | None = "en",
    stopwords: list[str] | None = None,
) -> Any:
    """Build extractive summaries for a partition's documents.

    Thin Session facade over ``buildml.nlp``; records the call and stores NLP artifacts on the Session for follow-up steps.

    Parameters
    ----------
    session:
        Active Session with dataset and optional split plan attached.
    partition:
        Split partition to read or score.
    method:
        Algorithm or method identifier for the resolved backend.
    text_column:
        Text column name; defaults to the sole text-role column.
    n_sentences:
        Number of sentences to keep in extractive summaries.
    max_documents:
        Maximum documents to profile or process in one call. Lower it for a cheap health check on huge corpora.
    max_input_sentences:
        Maximum sentences considered from each document before summarization.
    stopword_language:
        Language key for the stopword list used in token stats (for example ``"english"``). ``None`` skips stopword filtering.
    stopwords:
        Stopword list name, custom iterable, or ``None``. Removing common function words often helps bag-of-words models.

    Returns
    -------
    Any
        Structured domain result recorded on the Session for follow-up evaluate/explain/export steps.
    """
    result = summarize_text(
        session.dataset,
        session._split_plan,
        partition=partition,
        method=method,
        text_column=text_column,
        n_sentences=n_sentences,
        max_documents=max_documents,
        max_input_sentences=max_input_sentences,
        stopword_language=stopword_language,
        stopwords=stopwords,
    )
    session._nlp_summary_result = result
    session._record(
        "summarize_text",
        {
            "partition": partition,
            "method": method,
            "text_column": text_column,
            "n_sentences": n_sentences,
            "max_documents": max_documents,
            "max_input_sentences": max_input_sentences,
            "stopword_language": stopword_language,
            "stopwords": None if stopwords is None else len(stopwords),
        },
        warnings=tuple(result.warnings),
        result_summary=summary_result_summary(result),
    )
    return result


def save_nlp_bundle_op(session, path: str | Path) -> Path:
    """Persist the active NLP plan(s) as ``buildml.nlp_bundle.v1``.

    Thin Session facade over ``buildml.nlp``; records the call and stores NLP artifacts on the Session for follow-up steps.

    Parameters
    ----------
    session:
        Active Session with dataset and optional split plan attached.
    path:
        Filesystem path for load or save.

    Returns
    -------
    Path
        Resolved filesystem path written or loaded.

    Raises
    ------
    ValidationError
        When prerequisites are missing or inputs are invalid.
    """
    text_plan = getattr(session, "_nlp_text_plan", None)
    topic_plan = getattr(session, "_nlp_topic_plan", None)
    if text_plan is None and topic_plan is None:
        raise ValidationError(
            "No NLP plan to save. Call fit_text_classifier(...) or fit_topics(...) "
            "first."
        )
    out = save_nlp_bundle(
        path,
        text_plan,
        topic_plan=topic_plan,
        fit_result=getattr(session, "_nlp_fit_result", None),
        eval_result=getattr(session, "_nlp_eval_result", None),
    )
    session._record(
        "save_nlp_bundle",
        {"path": str(out)},
        result_summary={
            "path": str(out),
            "format": "buildml.nlp_bundle.v1",
            "has_text_plan": text_plan is not None,
            "has_topic_plan": topic_plan is not None,
        },
    )
    return out


def load_nlp_bundle_op(session, path: str | Path):
    """Load an NLP bundle into this Session.

    Thin Session facade over ``buildml.nlp``; records the call and stores NLP artifacts on the Session for follow-up steps.

    Parameters
    ----------
    session:
        Active Session with dataset and optional split plan attached.
    path:
        Filesystem path for load or save.
    """
    text_plan, topic_plan = load_nlp_bundle(path)
    session._nlp_text_plan = text_plan
    session._nlp_topic_plan = topic_plan
    session._nlp_fit_result = None
    session._nlp_eval_result = None
    session._nlp_predict_result = None
    session._nlp_interpret_result = None
    session._nlp_topic_result = None
    session._nlp_topic_assign_result = None
    session._record(
        "load_nlp_bundle",
        {"path": str(path)},
        result_summary={
            "path": str(path),
            "kind": "nlp",
            "has_text_plan": text_plan is not None,
            "has_topic_plan": topic_plan is not None,
            "text_column": (
                getattr(text_plan, "text_column", None)
                if text_plan is not None
                else getattr(topic_plan, "text_column", None)
            ),
        },
    )
    return session
