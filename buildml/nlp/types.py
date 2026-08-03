"""Configuration types for the Session-facing natural-language path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# Single-label document classification is the supervised NLP task BuildML ships.
# Multi-label / span-level supervision is an explicit non-goal (see catalog).
NlpTask = Literal["classification"]

# Document representation backends.
# - sklearn: train-fitted bag-of-n-grams (count / TF-IDF / hashing)
# - embedding: sentence-transformer document vectors (buildml[nlp])
# - transformer: frozen transformer encoder pooled vectors (buildml[nlp])
NlpBackend = Literal["sklearn", "embedding", "transformer"]

NlpVectorizerKind = Literal["tfidf", "count", "hashing"]
NlpAnalyzer = Literal["word", "char", "char_wb"]

# Linear / naive-Bayes heads that stay honest on sparse text features.
NlpEstimator = Literal[
    "logistic",
    "linear_svm",
    "complement_nb",
    "multinomial_nb",
    "sgd",
]

TopicMethod = Literal["nmf", "lda"]
KeyphraseMethod = Literal["tfidf", "rake", "textrank"]
SummarizeMethod = Literal["textrank", "lexrank", "lead"]
SentimentBackend = Literal["lexicon", "supervised", "transformer"]
EntityBackend = Literal["rules", "spacy"]
LanguageBackend = Literal["native", "langdetect"]

# Deterministic normalization steps applied before tokenization.
NormalizeStep = Literal[
    "lowercase",
    "strip_accents",
    "strip_html",
    "strip_urls",
    "strip_emails",
    "strip_numbers",
    "strip_punctuation",
    "collapse_whitespace",
    "collapse_repeats",
]

DEFAULT_NORMALIZE_STEPS: tuple[NormalizeStep, ...] = (
    "strip_html",
    "strip_urls",
    "strip_emails",
    "lowercase",
    "collapse_whitespace",
)


@dataclass(slots=True)
class TextNormalizeConfig:
    """Deterministic, stateless text-normalization knobs.

    Normalization never learns anything from the corpus, so applying it before a
    split cannot leak. Vocabulary-bearing steps (stopwords, vectorizer fitting)
    live in :class:`NlpVectorizeConfig` and are train-only.
    """

    steps: tuple[NormalizeStep, ...] = DEFAULT_NORMALIZE_STEPS
    stopwords: tuple[str, ...] | None = None
    stopword_language: str | None = None
    min_token_length: int = 1
    max_token_length: int = 40
    stem: bool = False
    lemmatize: bool = False
    keep_emoji: bool = True
    keep_numbers: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return the normalisation settings as plain JSON-safe values.

        Recorded on the plan and in model cards, because normalisation decides
        what a "token" is — two models with different settings are not
        comparable even on identical text.

        Returns
        -------
        dict
            Every field in plain-data form, with tuples flattened to lists.
        """
        return {
            "steps": list(self.steps),
            "stopwords": None if self.stopwords is None else list(self.stopwords),
            "stopword_language": self.stopword_language,
            "min_token_length": self.min_token_length,
            "max_token_length": self.max_token_length,
            "stem": self.stem,
            "lemmatize": self.lemmatize,
            "keep_emoji": self.keep_emoji,
            "keep_numbers": self.keep_numbers,
        }


@dataclass(slots=True)
class NlpVectorizeConfig:
    """How documents become numbers — the settings, not the fitted vocabulary.

    Unlike :class:`TextNormalizeConfig`, everything here feeds a step that
    *learns* from the corpus: which terms exist, how often each appears, how
    rare each is. That learning must happen on training documents only, which
    is why these settings are passed to a fit function rather than applied
    directly.

    Attributes
    ----------
    kind:
        ``'tfidf'``, ``'count'``, or ``'hashing'``. TF-IDF down-weights terms
        that appear everywhere and is the usual choice; hashing trades away an
        invertible vocabulary — and therefore token attributions — for bounded
        memory.
    analyzer:
        ``'word'`` or ``'char'``. Character n-grams survive typos and work on
        languages without whitespace word boundaries.
    ngram_range:
        ``(min_n, max_n)`` term lengths. Pairs recover a little word order;
        going wider grows the vocabulary steeply.
    max_features:
        Vocabulary cap, keeping the most frequent terms. The main memory
        control.
    min_df:
        Discard terms appearing in fewer than this many documents — an integer
        counts documents, a float is a proportion. The cheapest way to remove
        typos.
    max_df:
        Discard terms appearing in more than this share of documents. A
        corpus-driven stopword list.
    sublinear_tf:
        Use ``1 + log(count)`` instead of the raw count, so repetition counts
        for less. Helps when document lengths vary.
    binary:
        Record presence only, ignoring counts.
    n_hash_features:
        Bucket count for the hashing vectorizer. More buckets means fewer
        collisions between unrelated terms.
    """

    kind: NlpVectorizerKind = "tfidf"
    analyzer: NlpAnalyzer = "word"
    ngram_range: tuple[int, int] = (1, 2)
    max_features: int | None = 20000
    min_df: int | float = 1
    max_df: int | float = 1.0
    sublinear_tf: bool = True
    binary: bool = False
    n_hash_features: int = 2**18

    def to_dict(self) -> dict[str, Any]:
        """Return the vectorizer settings as plain JSON-safe values.

        Stored on the fitted plan so the representation can be described — and
        judged — without unpickling it.

        Returns
        -------
        dict
            Every field in plain-data form, with ``ngram_range`` as a list.
        """
        return {
            "kind": self.kind,
            "analyzer": self.analyzer,
            "ngram_range": list(self.ngram_range),
            "max_features": self.max_features,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "sublinear_tf": self.sublinear_tf,
            "binary": self.binary,
            "n_hash_features": self.n_hash_features,
        }


@dataclass(slots=True)
class NlpConfig:
    """The complete text-classifier configuration in one serialisable object.

    Bundles the normalisation and vectorisation settings with the head's
    hyperparameters, so a whole configuration can be recorded, compared, or
    passed around as a single value rather than as a dozen keyword arguments.

    Attributes
    ----------
    task:
        What the model does. Currently single-label document classification.
    backend:
        Which representation family produces the vectors.
    estimator:
        Which classifier head sits on top.
    text_column:
        The document column, or ``None`` to infer it from roles and dtype.
    normalize:
        Stateless text cleanup settings.
    vectorize:
        Settings for the corpus-fitted representation.
    class_weight:
        ``'balanced'`` to weight rare classes up, or ``None``.
    C:
        Inverse regularisation strength for the linear heads.
    alpha:
        Smoothing for naive Bayes, regularisation for SGD.
    embedding_model_name:
        Which sentence-transformer the embedding backend loads.
    max_seq_tokens:
        How much of each document the transformer backends read before
        truncating.
    random_state:
        Seed, so the fit reproduces.
    disclosures:
        Caveats carried alongside the configuration into reports.
    """

    task: NlpTask = "classification"
    backend: NlpBackend = "sklearn"
    estimator: NlpEstimator = "logistic"
    text_column: str | None = None
    normalize: TextNormalizeConfig = field(default_factory=TextNormalizeConfig)
    vectorize: NlpVectorizeConfig = field(default_factory=NlpVectorizeConfig)
    class_weight: Literal["balanced"] | None = None
    C: float = 1.0
    alpha: float = 1.0
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_seq_tokens: int = 256
    random_state: int | None = 0
    disclosures: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return the whole configuration as plain JSON-safe values.

        Belongs in a model card: two text classifiers are only comparable if
        their normalisation and vectorisation match, and this is the record
        that lets someone check.

        Returns
        -------
        dict
            Every field, with the nested normalise and vectorise
            configurations expanded in place.
        """
        return {
            "task": self.task,
            "backend": self.backend,
            "estimator": self.estimator,
            "text_column": self.text_column,
            "normalize": self.normalize.to_dict(),
            "vectorize": self.vectorize.to_dict(),
            "class_weight": self.class_weight,
            "C": self.C,
            "alpha": self.alpha,
            "embedding_model_name": self.embedding_model_name,
            "max_seq_tokens": self.max_seq_tokens,
            "random_state": self.random_state,
            "disclosures": list(self.disclosures),
        }
