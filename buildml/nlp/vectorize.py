"""Train-fitted document representations for the NLP domain.

Boundary with :mod:`buildml.preprocess.text`: ``Session.text_features`` expands a
text column into numeric *tabular* columns so classical tabular models can use
it. This module builds a document-level representation that stays inside the NLP
plan (sparse bag-of-n-grams or dense encoder vectors) and is never written back
onto the dataset. Both are train-only fits; they are not interchangeable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)

from buildml.core.errors import ValidationError
from buildml.nlp.normalize import TextNormalizePlan, build_analyzer
from buildml.nlp.types import NlpBackend, NlpVectorizeConfig

VALID_BACKENDS: tuple[str, ...] = ("sklearn", "embedding", "transformer")
VALID_KINDS: tuple[str, ...] = ("tfidf", "count", "hashing")
VALID_ANALYZERS: tuple[str, ...] = ("word", "char", "char_wb")


def validate_vectorize_config(config: NlpVectorizeConfig) -> NlpVectorizeConfig:
    """Check vectorizer settings before anything expensive happens.

    Validating up front matters more here than it looks. Several of these
    mistakes do not raise later — they silently produce an empty or degenerate
    vocabulary, and you discover it as a mysteriously useless model rather than
    as an error.

    Parameters
    ----------
    config:
        The settings to check.

    Returns
    -------
    ~buildml.nlp.types.NlpVectorizeConfig
        The same object, so this can be used inline.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        The vectorizer kind or analyzer is unrecognised; the n-gram range is
        not an ascending pair starting at 1 or more; ``max_features`` is below
        1; ``n_hash_features`` is below 16; or a float ``min_df`` or ``max_df``
        falls outside ``[0, 1]``.

    Notes
    -----
    Note the asymmetry in ``min_df`` and ``max_df``: as integers they are
    document counts, as floats they are proportions. Passing ``1.0`` where you
    meant "at least one document" gives you "in every document", which usually
    empties the vocabulary. Only the float range can be checked here, which is
    why the confusion is worth naming.

    See Also
    --------
    build_sklearn_vectorizer : Calls this before building.
    """
    if config.kind not in VALID_KINDS:
        raise ValidationError(
            f"vectorizer='{config.kind}' is not supported. Choose from {list(VALID_KINDS)}."
        )
    if config.analyzer not in VALID_ANALYZERS:
        raise ValidationError(
            f"analyzer='{config.analyzer}' is not supported. "
            f"Choose from {list(VALID_ANALYZERS)}."
        )
    low, high = config.ngram_range
    if int(low) < 1 or int(high) < int(low):
        raise ValidationError(
            "ngram_range must be a (min_n, max_n) pair with 1 <= min_n <= max_n."
        )
    if config.max_features is not None and int(config.max_features) < 1:
        raise ValidationError("max_features must be >= 1 when provided.")
    if config.n_hash_features < 16:
        raise ValidationError("n_hash_features must be >= 16.")
    if isinstance(config.min_df, float) and not 0.0 <= config.min_df <= 1.0:
        raise ValidationError("min_df as a float must be within [0.0, 1.0].")
    if isinstance(config.max_df, float) and not 0.0 <= config.max_df <= 1.0:
        raise ValidationError("max_df as a float must be within [0.0, 1.0].")
    return config


def build_sklearn_vectorizer(
    config: NlpVectorizeConfig,
    normalize_plan: TextNormalizePlan,
) -> Any:
    """Build a scikit-learn vectorizer that tokenises the BuildML way.

    Left to itself, scikit-learn applies its own tokenisation and lowercasing,
    which would mean the vocabulary a model learns differs from the tokens
    BuildML reports. Word analyzers therefore hand the plan's tokenizer to the
    vectorizer as its analyzer, so the two agree exactly. Character analyzers
    delegate n-gram extraction to scikit-learn — it has no notion of a "word"
    to disagree about — but still receive normalised input through the
    preprocessor hook.

    Parameters
    ----------
    config:
        Validated vectorizer settings.
    normalize_plan:
        The plan governing normalisation and tokenisation.

    Returns
    -------
    object
        An unfitted scikit-learn vectorizer: ``HashingVectorizer``,
        ``CountVectorizer``, or ``TfidfVectorizer`` according to
        ``config.kind``.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        The configuration fails :func:`validate_vectorize_config`.

    Notes
    -----
    ``lowercase=False`` is set deliberately. Lowercasing belongs to the
    normalisation plan, where it is recorded and reproducible; letting
    scikit-learn also do it would apply the step twice and hide it from the
    plan's record.

    Hashing vectorizers are stateless — nothing is learned at fit time, so they
    can transform without fitting. That is what makes them cheap and what makes
    their vocabulary unrecoverable.

    See Also
    --------
    build_document_vectorizer : Chooses between this and the neural backends.
    """
    validate_vectorize_config(config)
    ngram_range = (int(config.ngram_range[0]), int(config.ngram_range[1]))

    if config.analyzer == "word":
        shared: dict[str, Any] = {
            "analyzer": build_analyzer(normalize_plan, ngram_range=ngram_range),
            "lowercase": False,
        }
    else:
        from buildml.nlp.normalize import normalize_document

        shared = {
            "analyzer": config.analyzer,
            "ngram_range": ngram_range,
            "lowercase": False,
            "preprocessor": lambda doc: normalize_document(doc, normalize_plan),
        }

    if config.kind == "hashing":
        return HashingVectorizer(
            n_features=int(config.n_hash_features),
            alternate_sign=False,
            norm="l2",
            binary=bool(config.binary),
            dtype=np.float64,
            **shared,
        )

    bounded: dict[str, Any] = {
        "max_features": config.max_features,
        "min_df": config.min_df,
        "max_df": config.max_df,
        "binary": bool(config.binary),
        "dtype": np.float64,
    }
    if config.kind == "count":
        return CountVectorizer(**bounded, **shared)
    return TfidfVectorizer(sublinear_tf=bool(config.sublinear_tf), **bounded, **shared)


def build_document_vectorizer(
    *,
    backend: NlpBackend,
    config: NlpVectorizeConfig,
    normalize_plan: TextNormalizePlan,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_seq_tokens: int = 256,
    device: str = "cpu",
) -> tuple[Any, list[str]]:
    """Build the representation for a backend, along with what to disclose about it.

    The single dispatch point across the three ways this package turns
    documents into numbers, and the place where the honest caveats for each are
    attached.

    Bag-of-n-grams counts words. It cannot tell that "cancel my subscription"
    and "I want to unsubscribe" mean the same thing, because they share almost
    no tokens. Sentence embeddings and pooled transformers can, because they
    place documents in a space where meaning determines position — but they
    were trained on somebody else's corpus, they produce dimensions that
    correspond to no particular word, and they forfeit token attributions
    entirely.

    Parameters
    ----------
    backend:
        ``'sklearn'``, ``'embedding'``, or ``'transformer'``.
    config:
        Vectorizer settings. Used by the sklearn backend; the neural backends
        have their own architecture-fixed representation.
    normalize_plan:
        The normalisation plan, applied by the sklearn backend.
    embedding_model_name:
        Which pretrained model the neural backends load.
    max_seq_tokens:
        How much of each document the transformer backend reads before
        truncating.
    device:
        Where to run the neural backends.

    Returns
    -------
    tuple
        ``(vectorizer, disclosures)``. The vectorizer follows the scikit-learn
        fit/transform protocol whichever backend produced it. The disclosures
        are plain-language notes carried onto the plan and into reports.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        The backend name is unrecognised, or the sklearn configuration is
        invalid.
    ~buildml.core.errors.MissingExtraError
        A neural backend was requested without its optional dependency.

    Notes
    -----
    **The pretrained encoders are frozen and not fine-tuned.** Only the head is
    fitted on your training data. This keeps the operation cheap and keeps the
    train-only guarantee meaningful for everything that *is* fitted here — but
    it also means the encoder's own training data sits entirely outside your
    split, which is disclosed rather than assumed away.

    See Also
    --------
    build_sklearn_vectorizer : The bag-of-n-grams path in detail.
    """
    if backend not in VALID_BACKENDS:
        raise ValidationError(
            f"backend='{backend}' is not supported. Choose from {list(VALID_BACKENDS)}."
        )
    if backend == "sklearn":
        vectorizer = build_sklearn_vectorizer(config, normalize_plan)
        disclosures = [
            f"Representation: {config.kind} bag-of-{config.analyzer}-n-grams "
            f"{list(config.ngram_range)}, fitted on train documents only.",
        ]
        if config.kind == "hashing":
            disclosures.append(
                "Hashing has no invertible vocabulary; token attributions are "
                "unavailable and collisions are irreversible."
            )
        return vectorizer, disclosures

    if backend == "embedding":
        from buildml.nlp.adapters.sentence_embedding import SentenceEmbeddingVectorizer

        vectorizer = SentenceEmbeddingVectorizer(embedding_model_name)
        return vectorizer, [
            f"Representation: sentence-transformer document vectors "
            f"({embedding_model_name}); the encoder is pretrained and frozen.",
            "Pretrained encoders were trained outside this Session; their training "
            "data is not covered by the Session split.",
        ]

    from buildml.nlp.adapters.transformer_encoder import TransformerEncoderVectorizer

    vectorizer = TransformerEncoderVectorizer(
        embedding_model_name,
        max_seq_tokens=max_seq_tokens,
        device=device,
    )
    return vectorizer, [
        f"Representation: mean-pooled frozen transformer encoder "
        f"({embedding_model_name}, max_seq_tokens={max_seq_tokens}).",
        "The encoder is not fine-tuned here; only the linear head is fitted on "
        "train. Use the Torch text path for fine-tuning.",
    ]


def feature_names_for(vectorizer: Any, *, limit: int | None = None) -> tuple[str, ...]:
    """Recover the term behind each feature column, where that is possible.

    These names are what turn a coefficient vector into an explanation: column
    4,217 means nothing, but "refund" does. Storing them on the plan is what
    lets :func:`~buildml.nlp.interpret.interpret_text_prediction` work later.

    Parameters
    ----------
    vectorizer:
        A fitted vectorizer. Anything without ``get_feature_names_out`` is
        handled gracefully.
    limit:
        Truncate to this many names. A word-bigram vocabulary over a large
        corpus can run to millions of terms, and storing all of them would
        dominate the size of a saved bundle.

    Returns
    -------
    tuple of str
        Feature names in column order, possibly truncated. Empty when the
        representation has no recoverable vocabulary — hashing vectorizers and
        the neural backends both return nothing here.

    Notes
    -----
    Truncation keeps the first ``limit`` names, so attributions for features
    beyond the cut-off fall back to a positional placeholder rather than
    failing.

    See Also
    --------
    vocabulary_size : The count, when you do not need the names themselves.
    """
    getter = getattr(vectorizer, "get_feature_names_out", None)
    if getter is None:
        return ()
    try:
        names = [str(name) for name in getter()]
    except Exception:
        return ()
    if limit is not None and len(names) > limit:
        return tuple(names[:limit])
    return tuple(names)


def vocabulary_size(vectorizer: Any) -> int:
    """Count how many distinct terms the representation actually learned.

    Reported on fit results as a sanity check. A vocabulary far smaller than
    expected means ``min_df``, ``max_features``, or the stopword list is
    cutting harder than intended; one in the millions means the feature space
    is wider than the corpus can support and the model will overfit.

    Parameters
    ----------
    vectorizer:
        A fitted vectorizer.

    Returns
    -------
    int
        The learned vocabulary size, or 0 when there is no vocabulary to
        count.

    Notes
    -----
    Zero does not mean "learned nothing". Hashing vectorizers are stateless and
    embedding backends produce fixed-width dense vectors — in both cases there
    are features but no vocabulary behind them, and the count is not
    applicable. Read a zero alongside the backend, not on its own.
    """
    vocabulary = getattr(vectorizer, "vocabulary_", None)
    if isinstance(vocabulary, dict):
        return len(vocabulary)
    dim = getattr(vectorizer, "embedding_dim_", None)
    if isinstance(dim, int):
        return 0
    return 0


def matrix_width(matrix: Any) -> int:
    """Count a document matrix's columns, sparse or dense alike.

    Backends return different matrix types — SciPy sparse from bag-of-words,
    NumPy dense from the encoders — and callers need the feature count without
    caring which they have.

    Parameters
    ----------
    matrix:
        A document matrix from any backend.

    Returns
    -------
    int
        The number of feature columns.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        The object is not 2-dimensional, which means something other than a
        document matrix was passed.

    Notes
    -----
    A width of zero is possible and is caught by callers as an error: it means
    the filters removed every term, so there is nothing to model. That happens
    most often from a ``min_df`` set too high for a small corpus.
    """
    shape = getattr(matrix, "shape", None)
    if shape is None or len(shape) < 2:
        raise ValidationError("Document matrix must be 2-dimensional.")
    return int(shape[1])


def oov_token_rate(
    documents: list[str],
    vectorizer: Any,
    normalize_plan: TextNormalizePlan,
) -> float | None:
    """Measure how much of this text the model has simply never seen.

    The most useful single diagnostic for a deployed text model. A token
    outside the training vocabulary contributes nothing — it has no weight, so
    it is as if the word were not there. At a low rate that is harmless; at a
    high rate the model is predicting from a fraction of each document and its
    confidence is unearned.

    Parameters
    ----------
    documents:
        The documents to check.
    vectorizer:
        The fitted vectorizer whose vocabulary defines "seen".
    normalize_plan:
        The plan used at fit time. It must be the same one, or the tokens
        compared against the vocabulary will not be the tokens that built it.

    Returns
    -------
    float or None
        The unseen share of tokens, from 0.0 to 1.0. ``None`` when the question
        cannot be answered — hashing vectorizers and the neural backends have
        no vocabulary to check against, and a document with no tokens at all
        gives nothing to measure.

    Notes
    -----
    ``None`` and ``0.0`` mean different things. Zero is a measured result;
    ``None`` means the check was not possible, and callers report the absence
    rather than treating it as a clean bill of health.

    Above roughly a third, callers raise a warning. Rising drift over time is
    the signal to refit; a sudden jump usually means the upstream text source
    changed.
    """
    vocabulary = getattr(vectorizer, "vocabulary_", None)
    if not isinstance(vocabulary, dict) or not vocabulary:
        return None
    from buildml.nlp.normalize import tokenize_document

    known = set(vocabulary)
    total = 0
    unseen = 0
    for document in documents:
        for token in tokenize_document(document, normalize_plan):
            total += 1
            if token not in known:
                unseen += 1
    if total == 0:
        return None
    return float(unseen / total)


def densify(matrix: Any) -> np.ndarray:
    """Convert a document matrix to a dense float array.

    Some algorithms cannot accept sparse input. This makes the conversion
    explicit and uniform, rather than leaving each call site to guess whether
    it is holding a sparse matrix.

    Parameters
    ----------
    matrix:
        A sparse or dense document matrix.

    Returns
    -------
    ~numpy.ndarray
        A dense float array.

    Notes
    -----
    **Check the width before calling this on a bag-of-words matrix.** Text
    matrices are typically over 99% zeros, and densifying one costs
    ``n_documents × n_features × 8`` bytes — ten thousand documents over fifty
    thousand features is four gigabytes of mostly zeros.
    """
    if sparse.issparse(matrix):
        return np.asarray(matrix.toarray(), dtype=float)
    return np.asarray(matrix, dtype=float)


def reduce_for_similarity(
    matrix: Any,
    *,
    n_components: int,
    random_state: int | None = 0,
) -> tuple[np.ndarray, TruncatedSVD | None]:
    """Compress a wide sparse matrix into a few dense dimensions of meaning.

    Truncated SVD on a TF-IDF matrix is latent semantic analysis: it finds the
    directions along which documents vary most, and words that co-occur end up
    sharing a direction. The practical effect is that two documents about the
    same thing land near each other even when they use different words, which
    plain cosine similarity over raw term counts would miss.

    Used where a dense, lower-rank view is needed — near-duplicate screening
    and document similarity being the main cases.

    Parameters
    ----------
    matrix:
        The document matrix to project.
    n_components:
        How many dimensions to keep. Automatically clamped to what the data
        can support: never more than one below the feature count, and never
        more than one below the row count. More components retain more detail;
        fewer generalise harder.
    random_state:
        Seed, since the solver is randomised.

    Returns
    -------
    tuple
        ``(projected, reducer)``. The reducer is returned so the same basis can
        be applied to further documents — a projection fitted separately would
        put them in a different space, making the coordinates incomparable.

    Notes
    -----
    A dense or already-narrow matrix is returned densified with a ``None``
    reducer, since there is nothing to compress. Callers must handle that case
    rather than assuming a reducer comes back.

    Unlike PCA, truncated SVD does not centre the data, which is what lets it
    run on a sparse matrix without materialising it densely.
    """
    width = matrix_width(matrix)
    n_rows = int(matrix.shape[0])
    usable = max(1, min(int(n_components), width - 1, max(1, n_rows - 1)))
    if not sparse.issparse(matrix) or width <= usable:
        return densify(matrix), None
    reducer = TruncatedSVD(n_components=usable, random_state=random_state)
    projected = reducer.fit_transform(matrix)
    return np.asarray(projected, dtype=float), reducer


__all__ = [
    "VALID_ANALYZERS",
    "VALID_BACKENDS",
    "VALID_KINDS",
    "build_document_vectorizer",
    "build_sklearn_vectorizer",
    "densify",
    "feature_names_for",
    "matrix_width",
    "oov_token_rate",
    "reduce_for_similarity",
    "validate_vectorize_config",
    "vocabulary_size",
]
