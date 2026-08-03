"""Check for optional NLP dependencies, and demand them with a useful message.

Core NLP runs on numpy, pandas, and scikit-learn alone. Everything heavier —
NLTK, spaCy, transformers — is opt-in, so installing BuildML does not drag in
gigabytes of models you may never use.

Two kinds of function live here, and the distinction is the point.

The ``*_available`` predicates ask a question and never raise, which is what you
want for branching: use embeddings if they are here, otherwise fall back. The
``require_*`` functions demand a dependency and raise
:class:`~buildml.core.errors.MissingExtraError` naming the exact extra to install
— which is what you want at the moment a user has explicitly asked for something
that cannot run.

Each predicate does two checks rather than one. Finding the module's spec is
cheap and answers "is it installed"; actually importing it is slower but catches
the install that is present and broken — a compiled dependency built against the
wrong library version, say, which would otherwise fail much later and much less
clearly.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def _spec_present(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def nltk_spec_present() -> bool:
    """Check whether NLTK is installed, without importing it.

    The cheap check. Used where the cost of a real import is not warranted —
    building a capability matrix, for instance, which asks about every optional
    dependency at once.

    Returns
    -------
    bool
        Whether the module can be located on the import path. ``True`` here
        does not guarantee the import will succeed.

    See Also
    --------
    nltk_available : The stricter check that actually imports.
    """
    return _spec_present("nltk")


def nltk_available() -> bool:
    """Check whether NLTK can actually be imported.

    NLTK provides the Porter stemmer, WordNet lemmatisation, and wider stopword
    coverage. Stemming degrades gracefully to built-in suffix rules without it;
    lemmatisation does not, since approximating a dictionary would produce
    wrong roots rather than crude ones.

    Returns
    -------
    bool
        Whether ``import nltk`` succeeds. Install with ``buildml[nlp]``.

    Notes
    -----
    Even with NLTK importable, WordNet lemmatisation needs its corpus
    downloaded separately. That is checked at point of use, not here.
    """
    if not nltk_spec_present():
        return False
    try:
        import nltk  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


def langdetect_spec_present() -> bool:
    """Check whether langdetect is installed, without importing it.

    The cheap half of the pair, for capability reporting rather than for
    deciding whether a call will succeed.

    Returns
    -------
    bool
        Whether the module can be located on the import path.

    See Also
    --------
    langdetect_available : The stricter check that actually imports.
    """
    return _spec_present("langdetect")


def langdetect_available() -> bool:
    """Check whether langdetect can actually be imported.

    langdetect covers many more languages than the built-in detector and does
    better on short strings. The native detector remains the default and always
    works, so this is a quality upgrade rather than a prerequisite.

    Returns
    -------
    bool
        Whether ``import langdetect`` succeeds. Install with ``buildml[nlp]``.
    """
    if not langdetect_spec_present():
        return False
    try:
        import langdetect  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


def spacy_spec_present() -> bool:
    """Check whether spaCy is installed, without importing it.

    Worth using rather than the full check where speed matters: importing
    spaCy is noticeably slower than most libraries.

    Returns
    -------
    bool
        Whether the module can be located on the import path.

    See Also
    --------
    spacy_available : The stricter check that actually imports.
    """
    return _spec_present("spacy")


def spacy_available() -> bool:
    """Check whether spaCy can actually be imported.

    spaCy brings statistical named-entity recognition, which generalises to
    names and organisations no rule could enumerate. The rule-based extractor
    stays the default and needs nothing.

    Returns
    -------
    bool
        Whether ``import spacy`` succeeds. Install with
        ``buildml[nlp-industry]``.

    Notes
    -----
    The library alone is not enough. spaCy needs a language model downloaded
    separately, which is a distinct check — see :func:`spacy_model_available`.
    """
    if not spacy_spec_present():
        return False
    try:
        import spacy  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


def spacy_model_available(model: str = "en_core_web_sm") -> bool:
    """Check whether a specific spaCy language model is installed.

    Separate from :func:`spacy_available` because the two fail independently
    and are fixed differently. Having spaCy without a model is the common case:
    ``pip install spacy`` gives you the library, and the model is a further
    download.

    Parameters
    ----------
    model:
        The pipeline package name. The default is the small English pipeline;
        larger variants are more accurate and slower, and other languages have
        their own packages.

    Returns
    -------
    bool
        Whether both spaCy and this model are present and loadable.

    Notes
    -----
    Missing models are downloaded with ``python -m spacy download <model>``.
    Model choice matters for entity extraction: the small English pipeline will
    not recognise entities in French text, and will not say so — it will just
    find very few.
    """
    if not spacy_available():
        return False
    if not _spec_present(model):
        return False
    try:
        import spacy

        spacy.util.get_package_path(model)
    except Exception:
        return False
    return True


def sentence_transformers_spec_present() -> bool:
    """Check whether sentence-transformers is installed, without importing it.

    Worth preferring here in particular: importing sentence-transformers pulls
    PyTorch in with it, which is slow enough to notice.

    Returns
    -------
    bool
        Whether the module can be located on the import path.

    See Also
    --------
    sentence_transformers_available : The stricter check that actually imports.
    """
    return _spec_present("sentence_transformers")


def sentence_transformers_available() -> bool:
    """Check whether sentence-transformers can actually be imported.

    This is the embedding backend: documents become dense vectors positioned by
    meaning, so two texts saying the same thing in different words land near
    each other. Bag-of-n-grams cannot do that at any setting.

    Returns
    -------
    bool
        Whether ``import sentence_transformers`` succeeds. Install with
        ``buildml[nlp]``.

    Notes
    -----
    The library pulls in PyTorch, and the model weights download on first use.
    Neither is small.
    """
    if not sentence_transformers_spec_present():
        return False
    try:
        import sentence_transformers  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


def transformers_spec_present() -> bool:
    """Check whether Hugging Face transformers is installed, without importing it.

    The cheap check, which matters here because importing transformers is
    heavy — it initialises a backend framework on the way in.

    Returns
    -------
    bool
        Whether the module can be located on the import path.

    See Also
    --------
    transformers_available : The stricter check that actually imports.
    """
    return _spec_present("transformers")


def transformers_available() -> bool:
    """Check whether Hugging Face transformers can actually be imported.

    Powers the pooled-encoder representation and the pretrained sentiment
    backend. The encoders here are frozen feature extractors, not fine-tuned —
    the Torch text path owns fine-tuning.

    Returns
    -------
    bool
        Whether ``import transformers`` succeeds. Install with
        ``buildml[nlp]``.
    """
    if not transformers_spec_present():
        return False
    try:
        import transformers  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


def nlp_industry_available() -> bool:
    """Check whether any heavyweight NLP backend is available at all.

    A coarse capability question — "can this install do more than bag-of-words"
    — used in reporting rather than for choosing a specific backend.

    Returns
    -------
    bool
        Whether spaCy or transformers is importable.
    """
    return spacy_available() or transformers_available()


def require_nltk(*, feature: str = "NLTK stemming / lemmatization") -> Any:
    """Import NLTK, or explain exactly what to install and why.

    For the call sites with no fallback — chiefly WordNet lemmatisation, where
    approximating the dictionary would give wrong answers rather than rough
    ones.

    Parameters
    ----------
    feature:
        What the caller was trying to do. Named in the error, so the user sees
        which capability they lost rather than a bare import failure.

    Returns
    -------
    module
        The imported ``nltk`` module.

    Raises
    ------
    ~buildml.core.errors.MissingExtraError
        NLTK is absent or its install is broken. Install ``buildml[nlp]``.

    See Also
    --------
    nltk_available : The non-raising check, for when a fallback exists.
    """
    try:
        import nltk
    except ImportError as exc:
        raise MissingExtraError("nlp", feature) from exc
    except OSError as exc:  # pragma: no cover - broken install
        raise MissingExtraError("nlp", feature) from exc
    return nltk


def require_langdetect(*, feature: str = "wide-coverage language detection") -> Any:
    """Import langdetect, or explain exactly what to install and why.

    Reached only when a caller asked for this backend by name. The native
    detector is the default and never needs it.

    Parameters
    ----------
    feature:
        What the caller was trying to do, named in the error.

    Returns
    -------
    module
        The imported ``langdetect`` module.

    Raises
    ------
    ~buildml.core.errors.MissingExtraError
        langdetect is absent or broken. Install ``buildml[nlp]``.

    See Also
    --------
    langdetect_available : The non-raising check.
    """
    try:
        import langdetect
    except ImportError as exc:
        raise MissingExtraError("nlp", feature) from exc
    except OSError as exc:  # pragma: no cover - broken install
        raise MissingExtraError("nlp", feature) from exc
    return langdetect


def require_spacy(*, feature: str = "spaCy entity extraction") -> Any:
    """Import spaCy, or explain exactly what to install and why.

    Reached when statistical entity extraction was asked for by name. The
    rule-based extractor is the default and never needs it.

    Parameters
    ----------
    feature:
        What the caller was trying to do, named in the error.

    Returns
    -------
    module
        The imported ``spacy`` module.

    Raises
    ------
    ~buildml.core.errors.MissingExtraError
        spaCy is absent or broken. Install ``buildml[nlp-industry]``.

    Notes
    -----
    This only guarantees the library. A missing language model surfaces later,
    at load time, with its own message.

    See Also
    --------
    spacy_model_available : Check for a specific pipeline package.
    """
    try:
        import spacy
    except ImportError as exc:
        raise MissingExtraError("nlp-industry", feature) from exc
    except OSError as exc:  # pragma: no cover - broken install
        raise MissingExtraError("nlp-industry", feature) from exc
    return spacy


def require_sentence_transformers(
    *, feature: str = "sentence-transformer document embeddings"
) -> Any:
    """Import sentence-transformers, or explain exactly what to install and why.

    Reached when ``backend='embedding'`` was requested. There is no fallback:
    bag-of-n-grams is a different representation, not a degraded version of
    this one.

    Parameters
    ----------
    feature:
        What the caller was trying to do, named in the error.

    Returns
    -------
    module
        The imported ``sentence_transformers`` module.

    Raises
    ------
    ~buildml.core.errors.MissingExtraError
        The library is absent or broken. Install ``buildml[nlp]``.

    See Also
    --------
    sentence_transformers_available : The non-raising check.
    """
    try:
        import sentence_transformers
    except ImportError as exc:
        raise MissingExtraError("nlp", feature) from exc
    except OSError as exc:  # pragma: no cover - broken install
        raise MissingExtraError("nlp", feature) from exc
    return sentence_transformers


def require_transformers(*, feature: str = "transformer encoder pooling") -> Any:
    """Import Hugging Face transformers, or explain what to install and why.

    Reached when the pooled-encoder representation or the pretrained sentiment
    backend was requested by name.

    Parameters
    ----------
    feature:
        What the caller was trying to do, named in the error.

    Returns
    -------
    module
        The imported ``transformers`` module.

    Raises
    ------
    ~buildml.core.errors.MissingExtraError
        The library is absent or broken. Install ``buildml[nlp]``.

    See Also
    --------
    transformers_available : The non-raising check.
    """
    try:
        import transformers
    except ImportError as exc:
        raise MissingExtraError("nlp", feature) from exc
    except OSError as exc:  # pragma: no cover - broken install
        raise MissingExtraError("nlp", feature) from exc
    return transformers


__all__ = [
    "langdetect_available",
    "langdetect_spec_present",
    "nlp_industry_available",
    "nltk_available",
    "nltk_spec_present",
    "require_langdetect",
    "require_nltk",
    "require_sentence_transformers",
    "require_spacy",
    "require_transformers",
    "sentence_transformers_available",
    "sentence_transformers_spec_present",
    "spacy_available",
    "spacy_model_available",
    "spacy_spec_present",
    "transformers_available",
    "transformers_spec_present",
]
