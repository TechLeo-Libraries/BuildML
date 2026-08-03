"""What this NLP install can actually do, and what it would need to do more.

Half of this package's capabilities depend on optional dependencies, and the
worst way to discover that is a failed import in the middle of a workflow. The
functions here answer "can I do this?" before you try, and the capability matrix
answers "what would I have to install?": so a missing feature is a decision
rather than a surprise.

Defaults are chosen to always work. The bag-of-n-grams backend needs nothing
beyond scikit-learn, and it stays the default even when the heavier backends are
installed: it is reproducible, cheap, and the only representation that can
explain its own decisions.
"""

from __future__ import annotations

from typing import Any, Literal

from buildml.core.errors import MissingExtraError, ValidationError
from buildml.nlp.extras import (
    langdetect_available,
    nltk_available,
    sentence_transformers_available,
    spacy_available,
    spacy_model_available,
    transformers_available,
)
from buildml.nlp.lexicons import (
    RULE_ENTITY_LABELS,
    SENTIMENT_LEXICON,
    SUPPORTED_STOPWORD_LANGUAGES,
)

NlpBackendName = Literal["sklearn", "embedding", "transformer"]

SKLEARN_ESTIMATORS = (
    "logistic",
    "linear_svm",
    "complement_nb",
    "multinomial_nb",
    "sgd",
)
# Naive Bayes needs non-negative counts, so dense signed encoder vectors are out.
DENSE_ESTIMATORS = ("logistic", "linear_svm", "sgd")

VECTORIZERS = ("tfidf", "count", "hashing")
ANALYZERS = ("word", "char", "char_wb")
TOPIC_METHODS = ("nmf", "lda")
KEYPHRASE_METHODS = ("tfidf", "rake", "textrank")
SUMMARIZE_METHODS = ("textrank", "lexrank", "lead")
SENTIMENT_BACKENDS = ("lexicon", "supervised", "transformer")
ENTITY_BACKENDS = ("rules", "spacy")
LANGUAGE_BACKENDS = ("native", "langdetect")


def nlp_capability_matrix() -> dict[str, Any]:
    """Report every NLP capability, whether it is available, and what it costs.

    A full picture of the package as installed right now: which representation
    backends can run, which task surfaces are reachable, what each one needs,
    and: the part that is easy to skip: what each one honestly cannot do.

    That last part is the reason this exists rather than a simple availability
    check. Knowing that lexicon sentiment is available matters much less than
    knowing it is domain-blind and will misread your jargon.

    Returns
    -------
    dict
        Keyed by capability area. Backend entries give availability, the extra
        that provides them, the representation produced, compatible
        estimators, whether token attributions are possible, and notes on the
        trade-offs. Task entries cover topics, keyphrases, sentiment, entities,
        summarisation, and language detection in the same shape.

    Notes
    -----
    Availability is checked live by attempting imports, so this reflects the
    running environment rather than a static list. Call it after installing an
    extra to confirm the new backend is genuinely reachable.

    Examples
    --------
    >>> matrix = nlp_capability_matrix()
    >>> matrix["backends"]["sklearn"]["available"]
    True

    See Also
    --------
    list_nlp_backends : Just the backend names.
    backend_available : A single yes-or-no check.
    """
    embedding_ready = sentence_transformers_available()
    transformer_ready = transformers_available()
    spacy_ready = spacy_available()
    return {
        "backends": {
            "sklearn": {
                "available": True,
                "extra": None,
                "representation": "train-fitted bag-of-n-grams (tfidf / count / hashing)",
                "estimators": list(SKLEARN_ESTIMATORS),
                "analyzers": list(ANALYZERS),
                "token_attributions": True,
                "notes": (
                    "Always-available native path. Word analyzers use the BuildML "
                    "tokenizer so the learned vocabulary matches what the teaching "
                    "surface reports."
                ),
            },
            "embedding": {
                "available": embedding_ready,
                "extra": "nlp",
                "representation": "sentence-transformer document vectors (frozen)",
                "estimators": list(DENSE_ESTIMATORS),
                "analyzers": [],
                "token_attributions": False,
                "notes": (
                    "Dense document vectors from a pretrained sentence-transformer "
                    "plus a linear head fitted on train. No token-level "
                    "attributions because features are latent dimensions."
                ),
            },
            "transformer": {
                "available": transformer_ready,
                "extra": "nlp",
                "representation": "mean-pooled frozen transformer encoder",
                "estimators": list(DENSE_ESTIMATORS),
                "analyzers": [],
                "token_attributions": False,
                "requires": "buildml[nlp] (transformers + torch)",
                "notes": (
                    "The encoder is frozen; only the linear head is fitted. "
                    "Fine-tuning belongs to the Torch text path "
                    "(Session.make_text_torch_loaders / fit_torch)."
                ),
            },
        },
        "task_availability_disclosure": (
            "Task 'available' means at least one honest path works without optional "
            "extras; see backends_available for per-backend gating that matches "
            "runtime MissingExtraError refusal."
        ),
        "tasks": {
            "text_classification": {
                "available": True,
                "backends_available": {
                    "sklearn": True,
                    "embedding": embedding_ready,
                    "transformer": transformer_ready,
                },
                "kind": "single-label document classification",
                "metrics": [
                    "accuracy",
                    "balanced_accuracy",
                    "f1_macro",
                    "f1_weighted",
                    "precision_macro",
                    "recall_macro",
                    "log_loss",
                    "roc_auc",
                ],
                "notes": "Multi-label and span-level supervision are non-goals.",
            },
            "token_attribution": {
                "available": True,
                "kind": "linear coefficient x feature value per token",
                "requires_backend": "sklearn with tfidf/count (invertible vocabulary)",
                "backends_available": {
                    "sklearn_tfidf": True,
                    "sklearn_count": True,
                    "sklearn_hashing": False,
                    "embedding": False,
                    "transformer": False,
                },
                "notes": (
                    "Exact for linear heads; not a substitute for SHAP/LIME on "
                    "non-linear models."
                ),
            },
            "topic_modelling": {
                "available": True,
                "methods": list(TOPIC_METHODS),
                "metrics": ["npmi_coherence", "reconstruction_error", "perplexity"],
                "notes": (
                    "NMF on TF-IDF and LDA on counts. Coherence is NPMI computed "
                    "on the train partition only."
                ),
            },
            "keyphrase_extraction": {
                "available": True,
                "methods": list(KEYPHRASE_METHODS),
                "notes": "Unsupervised; no gold keyphrase metric is claimed.",
            },
            "sentiment": {
                "available": True,
                "backends_available": {
                    "lexicon": True,
                    "supervised": True,
                    "transformer": transformer_ready,
                },
                "backends": list(SENTIMENT_BACKENDS),
                "lexicon_terms": len(SENTIMENT_LEXICON),
                "notes": (
                    "The lexicon backend is rule-based (valence + negation + "
                    "intensifiers) and unsupervised. 'supervised' reuses a fitted "
                    "text classifier; 'transformer' needs buildml[nlp]."
                ),
            },
            "entity_extraction": {
                "available": True,
                "backends_available": {
                    "rules": True,
                    "spacy": spacy_model_available(),
                },
                "backends": list(ENTITY_BACKENDS),
                "rule_labels": list(RULE_ENTITY_LABELS),
                "spacy_model_present": spacy_model_available(),
                "notes": (
                    "The rules backend is precision-first regex/gazetteer matching "
                    "for structured mentions. Statistical NER needs "
                    "buildml[nlp-industry] plus a downloaded spaCy model."
                ),
            },
            "summarization": {
                "available": True,
                "methods": list(SUMMARIZE_METHODS),
                "kind": "extractive only",
                "notes": (
                    "Sentences are selected, never generated. Abstractive "
                    "summarization is an explicit non-goal; use buildml.ai for "
                    "LLM-generated prose with provider disclosure."
                ),
            },
            "language_detection": {
                "available": True,
                "backends_available": {
                    "native": True,
                    "langdetect": langdetect_available(),
                },
                "backends": list(LANGUAGE_BACKENDS),
                "native_languages": list(SUPPORTED_STOPWORD_LANGUAGES),
                "notes": (
                    "Native detection combines Unicode script probes with "
                    "function-word scoring for seven Latin-script languages. "
                    "Install buildml[nlp] for wide-coverage langdetect."
                ),
            },
            "corpus_profile": {
                "available": True,
                "kind": "corpus health + split-contamination screen",
                "checks": [
                    "empty_documents",
                    "length_distribution",
                    "vocabulary_and_hapax_rate",
                    "duplicate_documents",
                    "train_holdout_exact_overlap",
                    "train_holdout_near_duplicate",
                    "holdout_oov_token_rate",
                ],
                "notes": (
                    "Near-duplicate screening reports contamination it finds; it "
                    "does not silently drop rows."
                ),
            },
        },
        "morphology": {
            "stemming": {
                "native": "conservative English suffix rules (always available)",
                "nltk": nltk_available(),
                "extra": "nlp",
            },
            "lemmatization": {
                "native": False,
                "nltk_wordnet": nltk_available(),
                "extra": "nlp",
                "notes": "Requires the NLTK WordNet corpus to be downloaded once.",
            },
            "stopwords": {
                "builtin_languages": list(SUPPORTED_STOPWORD_LANGUAGES),
                "notes": "Pass stopwords=(...) for languages without a built-in list.",
            },
        },
        "nlp_vs_neighbours": {
            "nlp": (
                "Document-level modelling and analysis of a text column that lives "
                "on the Session dataset: classify, interpret, topic, keyphrase, "
                "sentiment, entities, extractive summary, language, profile."
            ),
            "rag": (
                "Corpus ingestion, chunking, indexing, and retrieval to ground "
                "generated answers with citations: a retrieval product, not a "
                "supervised text model."
            ),
            "preprocess_text_features": (
                "Session.text_features writes numeric columns back onto the "
                "dataset so tabular models can consume text. NLP keeps its "
                "representation inside the NLP plan."
            ),
            "dl_text": (
                "Session.make_text_torch_loaders / fit_torch fine-tune neural "
                "sequence models on token ids. NLP keeps encoders frozen."
            ),
            "ai": (
                "buildml.ai calls an external LLM provider under an operator "
                "policy. NLP never calls a network provider."
            ),
            "boundary": (
                "Sharing a text column does not merge these surfaces. NLP bundles "
                "(buildml.nlp_bundle.v1) are not RAG bundles or Torch bundles."
            ),
        },
        "default_backend_when_installed": _default_backend_when_installed(),
        "install_hints": {
            "nlp": (
                "pip install 'buildml[nlp]'  "
                "# NLTK morphology, langdetect language ID, sentence-transformer "
                "embeddings, frozen transformer encoders"
            ),
            "nlp-industry": (
                "pip install 'buildml[nlp-industry]'  "
                "# spaCy statistical NER (then: python -m spacy download en_core_web_sm)"
            ),
        },
        "non_goals": [
            "Abstractive / generative summarization and text generation",
            "Multi-label and span-level (sequence labelling) supervision",
            "Machine translation",
            "Transformer fine-tuning (Torch text path owns that)",
            "Document retrieval for generation (buildml.rag owns that)",
            "Coreference resolution and full dependency-parse products",
        ],
        "nltk_present": nltk_available(),
        "langdetect_present": langdetect_available(),
        "spacy_present": spacy_ready,
        "spacy_model_present": spacy_model_available(),
        "sentence_transformers_present": embedding_ready,
        "transformers_present": transformer_ready,
        "industry_extra_present": spacy_ready or transformer_ready,
    }


def _default_backend_when_installed() -> str:
    # Bag-of-n-grams stays the default even with extras installed: it is
    # reproducible, cheap, and the only backend with token attributions.
    return "sklearn"


def list_nlp_backends(*, available_only: bool = True) -> list[str]:
    """List the representation backends, by default only the usable ones.

    The quick check before offering a backend choice in a UI or a script, so
    users are not shown options that would fail on import.

    Parameters
    ----------
    available_only:
        Return only backends whose dependencies are installed. Set it ``False``
        to see everything the package supports in principle, which is what you
        want when telling someone what they could install.

    Returns
    -------
    list of str
        Backend names. Always includes ``'sklearn'``, which needs nothing
        beyond the core dependencies.

    See Also
    --------
    nlp_capability_matrix : The same information with the trade-offs attached.
    """
    matrix = nlp_capability_matrix()
    out: list[str] = []
    for name, entry in matrix["backends"].items():
        if available_only and not entry.get("available"):
            continue
        out.append(name)
    return out


def backend_available(name: str) -> bool:
    """Check whether one backend can run in this environment.

    Useful for branching: try embeddings when they are installed, fall back to
    bag-of-n-grams when they are not: rather than catching an import error
    after the fact.

    Parameters
    ----------
    name:
        The backend to check. An unrecognised name returns ``False`` rather
        than raising, since the question "can I use this?" has a sensible
        answer either way.

    Returns
    -------
    bool
        Whether the backend's dependencies are importable.

    See Also
    --------
    nlp_capability_matrix : Why you might not want a backend even when it is available.
    """
    entry = nlp_capability_matrix()["backends"].get(name)
    if entry is None:
        return False
    return bool(entry.get("available"))


def resolve_backend_estimator(
    *,
    backend: str | None,
    estimator: str | None,
    vectorizer: str = "tfidf",
) -> tuple[str, str]:
    """Fill in the backend and head, and reject combinations that cannot work.

    Not every head works with every representation, and the incompatibilities
    are not obvious from the names. Catching them here means a clear message
    instead of an error from deep inside scikit-learn: or worse, silently
    degraded results.

    Parameters
    ----------
    backend:
        The requested backend, or ``None`` for the default.
    estimator:
        The requested head, or ``None`` for the default.
    vectorizer:
        The requested vectorizer, checked for compatibility with the backend.

    Returns
    -------
    tuple
        ``(backend, estimator)``, both resolved and validated.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        The backend name is unknown; the head is incompatible with the backend;
        or ``vectorizer='hashing'`` was paired with a neural backend, which
        produces dense vectors and has no hashing step to configure.
    ~buildml.core.errors.MissingExtraError
        A neural backend was requested without its optional dependency.

    Notes
    -----
    **Naive Bayes cannot run on embeddings.** It models features as counts and
    requires them to be non-negative; encoder vectors contain negative values,
    so the pairing is rejected rather than silently producing nonsense.

    **The default stays ``'sklearn'`` even with extras installed.** It is
    reproducible, needs no downloads, and is the only backend that can explain
    its own decisions. Choose a neural backend deliberately, when word overlap
    genuinely is not enough.
    """
    resolved_backend = "sklearn" if backend is None else str(backend).lower()
    if resolved_backend not in {"sklearn", "embedding", "transformer"}:
        raise ValidationError(
            f"Unknown NLP backend {backend!r}. "
            "Choose from ['sklearn', 'embedding', 'transformer']."
        )

    if resolved_backend == "sklearn":
        allowed = SKLEARN_ESTIMATORS
        resolved_estimator = "logistic" if estimator is None else str(estimator).lower()
    else:
        allowed = DENSE_ESTIMATORS
        resolved_estimator = "logistic" if estimator is None else str(estimator).lower()

    if resolved_estimator not in allowed:
        raise ValidationError(
            f"estimator='{resolved_estimator}' is not valid for "
            f"backend='{resolved_backend}'. Choose from {list(allowed)}."
        )
    if resolved_backend != "sklearn" and str(vectorizer).lower() == "hashing":
        raise ValidationError(
            f"vectorizer='hashing' only applies to backend='sklearn'; "
            f"backend='{resolved_backend}' produces dense encoder vectors."
        )

    if resolved_backend == "embedding" and not sentence_transformers_available():
        raise MissingExtraError("nlp", "NLP backend='embedding'")
    if resolved_backend == "transformer" and not transformers_available():
        raise MissingExtraError("nlp", "NLP backend='transformer'")
    return resolved_backend, resolved_estimator


def resolve_entity_backend(backend: str | None) -> str:
    """Choose the entity extractor, defaulting to the one that always works.

    Resolves the name and checks its dependency up front, so a missing spaCy
    install surfaces before any documents are read.

    Parameters
    ----------
    backend:
        ``'rules'``, ``'spacy'``, or ``None`` for the default.

    Returns
    -------
    str
        The resolved backend name.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        The name is not a supported entity backend.
    ~buildml.core.errors.MissingExtraError
        spaCy was requested without ``buildml[nlp-industry]``.

    Notes
    -----
    The two extractors fail in genuinely different ways, so the choice matters
    beyond availability. Rules are precise on structured things: dates,
    amounts, reference codes: and blind to anything their patterns do not
    describe. spaCy's statistical model generalises to names and organisations
    it has never seen, and in exchange produces confident false positives on
    unusual text.

    Rules are the default because they need no download, run anywhere, and are
    fully inspectable.
    """
    name = "rules" if backend is None else str(backend).lower()
    if name not in ENTITY_BACKENDS:
        raise ValidationError(
            f"entity backend={backend!r} is not supported. "
            f"Choose from {list(ENTITY_BACKENDS)}."
        )
    if name == "spacy" and not spacy_available():
        raise MissingExtraError("nlp-industry", "entity backend='spacy'")
    return name


def resolve_language_backend(backend: str | None) -> str:
    """Choose the language detector, defaulting to the built-in one.

    Resolves the name and checks its dependency up front, before any documents
    are read.

    Parameters
    ----------
    backend:
        ``'native'``, ``'langdetect'``, or ``None`` for the default.

    Returns
    -------
    str
        The resolved backend name.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        The name is not a supported language backend.
    ~buildml.core.errors.MissingExtraError
        langdetect was requested without ``buildml[nlp]``.

    Notes
    -----
    The native detector scores character patterns against shipped profiles: no
    dependency, fast, and reliable on a paragraph of text. langdetect covers
    many more languages and does better on short strings, where a handful of
    characters must distinguish between close relatives.

    Both degrade on very short documents. A three-word string genuinely may not
    contain enough evidence to identify a language, and a confident answer to
    that question should be treated with suspicion whichever backend gave it.
    """
    name = "native" if backend is None else str(backend).lower()
    if name not in LANGUAGE_BACKENDS:
        raise ValidationError(
            f"language backend={backend!r} is not supported. "
            f"Choose from {list(LANGUAGE_BACKENDS)}."
        )
    if name == "langdetect" and not langdetect_available():
        raise MissingExtraError("nlp", "language backend='langdetect'")
    return name


__all__ = [
    "ANALYZERS",
    "DENSE_ESTIMATORS",
    "ENTITY_BACKENDS",
    "KEYPHRASE_METHODS",
    "LANGUAGE_BACKENDS",
    "SENTIMENT_BACKENDS",
    "SKLEARN_ESTIMATORS",
    "SUMMARIZE_METHODS",
    "TOPIC_METHODS",
    "VECTORIZERS",
    "backend_available",
    "list_nlp_backends",
    "nlp_capability_matrix",
    "resolve_backend_estimator",
    "resolve_entity_backend",
    "resolve_language_backend",
]
