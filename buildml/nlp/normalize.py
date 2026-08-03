"""Deterministic text normalization, tokenization, and sentence splitting.

Nothing in this module learns from data, so it is safe to apply before a split.
Anything that builds a vocabulary (vectorizers, topic models, classifiers) is
fitted on the train partition only — see :mod:`buildml.nlp.vectorize`.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from buildml.core.errors import ValidationError
from buildml.nlp.extras import nltk_available, require_nltk
from buildml.nlp.lexicons import SUFFIX_STEM_RULES, stopwords_for
from buildml.nlp.types import (
    DEFAULT_NORMALIZE_STEPS,
    NormalizeStep,
    TextNormalizeConfig,
)

VALID_NORMALIZE_STEPS: frozenset[str] = frozenset(
    {
        "lowercase",
        "strip_accents",
        "strip_html",
        "strip_urls",
        "strip_emails",
        "strip_numbers",
        "strip_punctuation",
        "collapse_whitespace",
        "collapse_repeats",
    }
)

_HTML_TAG = re.compile(r"<[^>]{1,200}>")
_HTML_ENTITY = re.compile(r"&(?:[a-zA-Z]{2,10}|#\d{2,5});")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_NUMBER = re.compile(r"\b\d[\d,._]*\b")
_WHITESPACE = re.compile(r"\s+")
_REPEATS = re.compile(r"(.)\1{2,}")
_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)

# Token pattern keeps intra-word apostrophes and hyphens ("don't", "state-of-art")
# and standalone emoji/symbol runs so sentiment rules can see them.
_TOKEN = re.compile(r"[^\W\d_]+(?:['\u2019\-][^\W\d_]+)*|\d+(?:[.,]\d+)*|[^\w\s]")
_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?\u2026])[\"')\]]*\s+(?=[\"'(\[]*[A-Z0-9\u00c0-\u024f])|[\r\n]{2,}"
)
_ABBREVIATIONS = (
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.", "etc.",
    "e.g.", "i.e.", "inc.", "ltd.", "co.", "fig.", "no.", "approx.",
    # Month abbreviations matter because "Jan. 4, 2024" is one of the date forms
    # the entity rules advertise; without these the splitter cuts mid-date.
    "jan.", "feb.", "mar.", "apr.", "jun.", "jul.", "aug.", "sep.", "sept.",
    "oct.", "nov.", "dec.",
)


@dataclass(slots=True)
class TextNormalizePlan:
    """A resolved normalisation recipe — the settings plus everything they resolved to.

    A :class:`~buildml.nlp.types.TextNormalizeConfig` says *what you asked for*;
    this says what you actually got. Requesting a language's stopwords becomes
    the materialised term set. Requesting stemming becomes a specific backend,
    which differs depending on whether NLTK is installed — and the plan records
    which one ran, because two stemmers do not produce the same tokens.

    Every NLP plan embeds one of these, which is what lets a saved model
    reproduce its exact preprocessing months later.

    Attributes
    ----------
    steps:
        The character-level cleanup steps, applied in order.
    stopwords:
        The resolved terms to drop after tokenising, merged from the language
        list and anything you supplied.
    stopword_language:
        Which built-in list contributed, or ``None``.
    min_token_length:
        Tokens shorter than this are dropped.
    max_token_length:
        Tokens longer than this are dropped, which removes URLs and
        concatenated garbage that survived the character steps.
    stem:
        Whether stemming was requested.
    lemmatize:
        Whether lemmatisation was requested.
    keep_emoji:
        Whether emoji survive cleanup. They carry real sentiment signal, so
        they are kept by default.
    keep_numbers:
        Whether numeric tokens survive.
    stem_backend:
        Which stemmer actually ran — ``'nltk-porter'`` when NLTK is available,
        ``'native-suffix'`` for the built-in conservative rules, or ``'none'``.
        The two produce different roots, so a plan stemmed one way cannot score
        documents stemmed the other.
    lemma_backend:
        Which lemmatiser ran, or ``'none'``.
    disclosures:
        Plain-language notes on what was resolved and any fallback taken,
        surfaced in reports.
    """

    steps: tuple[NormalizeStep, ...] = DEFAULT_NORMALIZE_STEPS
    stopwords: frozenset[str] = field(default_factory=frozenset)
    stopword_language: str | None = None
    min_token_length: int = 1
    max_token_length: int = 40
    stem: bool = False
    lemmatize: bool = False
    keep_emoji: bool = True
    keep_numbers: bool = True
    stem_backend: str = "none"
    lemma_backend: str = "none"
    disclosures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the resolved plan as plain JSON-safe values.

        The stopword set is summarised by its size rather than listed, since a
        language list runs to hundreds of terms and would swamp a model card.

        Returns
        -------
        dict
            Every setting plus ``n_stopwords``, the resolved backend names, and
            the disclosures.
        """
        return {
            "steps": list(self.steps),
            "n_stopwords": len(self.stopwords),
            "stopword_language": self.stopword_language,
            "min_token_length": self.min_token_length,
            "max_token_length": self.max_token_length,
            "stem": self.stem,
            "lemmatize": self.lemmatize,
            "keep_emoji": self.keep_emoji,
            "keep_numbers": self.keep_numbers,
            "stem_backend": self.stem_backend,
            "lemma_backend": self.lemma_backend,
            "disclosures": list(self.disclosures),
        }


def build_normalize_plan(config: TextNormalizeConfig | None = None) -> TextNormalizePlan:
    """Turn normalisation settings into a concrete, reproducible plan.

    Validates what you asked for, materialises the stopword set, and works out
    which stemming and lemmatisation backends are actually available in this
    environment — recording the answer, so a plan built on a machine with NLTK
    is distinguishable from one built without it.

    Parameters
    ----------
    config:
        The settings to resolve. ``None`` builds the default plan, which
        applies conservative cleanup and no morphology.

    Returns
    -------
    TextNormalizePlan
        The resolved plan, ready to apply to documents and to embed in a
        fitted model.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        A step name is unrecognised, ``min_token_length`` is below 1, or
        ``max_token_length`` is below ``min_token_length``.
    ~buildml.core.errors.MissingExtraError
        Lemmatisation was requested without NLTK. Unlike stemming there is no
        built-in fallback — a lemmatiser needs a dictionary, and approximating
        one would produce wrong roots rather than crude ones.

    Notes
    -----
    **Stemming degrades gracefully; lemmatisation does not.** Without NLTK,
    stemming falls back to conservative built-in suffix rules and records that
    it did. This keeps the library working on a bare install, but it means the
    same configuration can produce different tokens on different machines —
    check ``stem_backend`` on the plan when a model behaves differently after a
    deploy.

    **Requesting both runs lemmatisation first**, then stems the lemma. Rarely
    what you want: the two overlap heavily, and the combination produces roots
    that are neither readable nor obviously correct.

    See Also
    --------
    normalize_document : Applies a plan's character steps.
    tokenize_document : Applies the full plan, through to tokens.
    """
    cfg = config or TextNormalizeConfig()
    unknown = [step for step in cfg.steps if step not in VALID_NORMALIZE_STEPS]
    if unknown:
        raise ValidationError(
            f"Unknown normalization step(s) {unknown}. "
            f"Choose from {sorted(VALID_NORMALIZE_STEPS)}."
        )
    if cfg.min_token_length < 1:
        raise ValidationError("min_token_length must be >= 1.")
    if cfg.max_token_length < cfg.min_token_length:
        raise ValidationError("max_token_length must be >= min_token_length.")

    disclosures: list[str] = []
    words: set[str] = set()
    if cfg.stopword_language is not None:
        words |= set(stopwords_for(cfg.stopword_language))
        disclosures.append(
            f"Stopwords: built-in '{cfg.stopword_language}' list "
            f"({len(words)} terms) removed after tokenization."
        )
    if cfg.stopwords is not None:
        extra = {str(token).strip().lower() for token in cfg.stopwords if str(token).strip()}
        words |= extra
        disclosures.append(f"Stopwords: {len(extra)} caller-supplied term(s) added.")

    stem_backend = "none"
    if cfg.stem:
        if nltk_available():
            stem_backend = "nltk-porter"
            disclosures.append("Stemming: NLTK PorterStemmer (buildml[nlp]).")
        else:
            stem_backend = "native-suffix"
            disclosures.append(
                "Stemming: built-in conservative suffix rules "
                "(install buildml[nlp] for the NLTK Porter stemmer)."
            )

    lemma_backend = "none"
    if cfg.lemmatize:
        require_nltk(feature="WordNet lemmatization")
        lemma_backend = "nltk-wordnet"
        disclosures.append("Lemmatization: NLTK WordNetLemmatizer (buildml[nlp]).")
        if cfg.stem:
            disclosures.append(
                "Both stem=True and lemmatize=True requested; lemmatization runs "
                "first and stemming is applied to the lemma."
            )

    return TextNormalizePlan(
        steps=tuple(cfg.steps),
        stopwords=frozenset(words),
        stopword_language=cfg.stopword_language,
        min_token_length=cfg.min_token_length,
        max_token_length=cfg.max_token_length,
        stem=cfg.stem,
        lemmatize=cfg.lemmatize,
        keep_emoji=cfg.keep_emoji,
        keep_numbers=cfg.keep_numbers,
        stem_backend=stem_backend,
        lemma_backend=lemma_backend,
        disclosures=tuple(disclosures),
    )


def normalize_document(text: Any, plan: TextNormalizePlan) -> str:
    """Clean one document's raw characters, without splitting it into tokens.

    Runs the plan's character-level steps in order — stripping HTML, URLs, and
    email addresses, lowercasing, removing punctuation, collapsing runs of
    whitespace — and returns the cleaned string. Tokenisation, stopword
    removal, and morphology happen later, in :func:`tokenize_document`.

    Useful on its own when you want cleaned text rather than features: for
    display, for fingerprinting documents to detect duplicates, or as input to
    a tool that does its own tokenising.

    Parameters
    ----------
    text:
        The raw value. ``None`` becomes an empty string and non-strings are
        coerced, so a mixed-type column does not need cleaning first.
    plan:
        A plan from :func:`build_normalize_plan`.

    Returns
    -------
    str
        The cleaned document.

    Notes
    -----
    **This step is stateless and learns nothing**, so unlike almost everything
    else in the text path it cannot leak and is safe to run before splitting.

    **Order matters and is fixed by the plan's step order.** Stripping
    punctuation before stripping URLs, for instance, would leave the fragments
    of the URL behind as tokens.

    See Also
    --------
    tokenize_document : Cleaning plus tokenisation and morphology.
    """
    value = "" if text is None else str(text)
    if value != value.strip() and "collapse_whitespace" not in plan.steps:
        value = value.strip()
    for step in plan.steps:
        if step == "strip_html":
            value = _HTML_TAG.sub(" ", value)
            value = _HTML_ENTITY.sub(" ", value)
        elif step == "strip_urls":
            value = _URL.sub(" ", value)
        elif step == "strip_emails":
            value = _EMAIL.sub(" ", value)
        elif step == "lowercase":
            value = value.lower()
        elif step == "strip_accents":
            decomposed = unicodedata.normalize("NFKD", value)
            value = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
        elif step == "strip_numbers":
            value = _NUMBER.sub(" ", value)
        elif step == "strip_punctuation":
            value = _PUNCTUATION.sub(" ", value)
        elif step == "collapse_repeats":
            value = _REPEATS.sub(r"\1\1", value)
        elif step == "collapse_whitespace":
            value = _WHITESPACE.sub(" ", value).strip()
    return value


def _is_symbol(token: str) -> bool:
    """True for pictographic / symbol characters (emoji, currency, arrows)."""
    return all(unicodedata.category(char).startswith("S") for char in token)


def _native_stem(token: str) -> str:
    if len(token) < 4:
        return token
    for suffix, replacement, min_stem in SUFFIX_STEM_RULES:
        if not token.endswith(suffix):
            continue
        stem = token[: -len(suffix)] + replacement
        if len(stem) >= min_stem:
            return stem
        return token
    return token


class _Morphology:
    """Lazily-built stemmer / lemmatizer pair for a normalization plan."""

    __slots__ = ("_lemmatizer", "_plan", "_stemmer")

    def __init__(self, plan: TextNormalizePlan) -> None:
        """Prepare a morphology pair without loading either backend yet.

        Construction is cheap by design: NLTK's stemmer and lemmatiser are only
        instantiated on first use, so a plan that never touches a document
        never pays for them.

        Parameters
        ----------
        plan:
            The resolved plan, whose ``stem_backend`` and ``lemma_backend``
            decide what gets built.
        """
        self._plan = plan
        self._stemmer: Any = None
        self._lemmatizer: Any = None

    def _ensure_stemmer(self) -> Any:
        if self._stemmer is None and self._plan.stem_backend == "nltk-porter":
            nltk = require_nltk(feature="Porter stemming")
            self._stemmer = nltk.stem.PorterStemmer()
        return self._stemmer

    def _ensure_lemmatizer(self) -> Any:
        if self._lemmatizer is None and self._plan.lemma_backend == "nltk-wordnet":
            nltk = require_nltk(feature="WordNet lemmatization")
            try:
                nltk.data.find("corpora/wordnet.zip")
            except LookupError as exc:
                raise ValidationError(
                    "NLTK WordNet data is not installed. Run "
                    "python -c \"import nltk; nltk.download('wordnet')\" "
                    "or use stem=True instead."
                ) from exc
            self._lemmatizer = nltk.stem.WordNetLemmatizer()
        return self._lemmatizer

    def apply(self, token: str) -> str:
        """Reduce one token to its root form.

        Lemmatisation runs before stemming when both are configured, so a
        stemmer sees the dictionary form rather than the surface one.

        Parameters
        ----------
        token:
            A single token, already normalised at the character level.

        Returns
        -------
        str
            The reduced token, or the input unchanged when neither backend is
            active.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            WordNet lemmatisation was configured but NLTK's WordNet corpus has
            not been downloaded. The message gives the command to fix it.
        ~buildml.core.errors.MissingExtraError
            A backend needs NLTK and it is not installed.
        """
        out = token
        if self._plan.lemma_backend == "nltk-wordnet":
            out = str(self._ensure_lemmatizer().lemmatize(out))
        if self._plan.stem_backend == "nltk-porter":
            out = str(self._ensure_stemmer().stem(out))
        elif self._plan.stem_backend == "native-suffix":
            out = _native_stem(out)
        return out


def tokenize_document(
    text: Any,
    plan: TextNormalizePlan,
    *,
    morphology: _Morphology | None = None,
    normalize: bool = True,
) -> list[str]:
    """Turn one document into the list of tokens a vectorizer will count.

    The full pipeline: character cleanup, splitting into tokens, dropping
    punctuation while keeping emoji, optional stemming or lemmatisation, then
    filtering by length and against the stopword set. What comes out is exactly
    what the vocabulary is built from, which makes this the function to call
    when a feature name surprises you.

    Parameters
    ----------
    text:
        The raw value. ``None`` and non-strings are coerced.
    plan:
        A plan from :func:`build_normalize_plan`.
    morphology:
        A reusable stemmer and lemmatiser pair. Pass one when tokenising many
        documents in a loop, so the backends load once rather than per
        document; leave it ``None`` for one-off calls.
    normalize:
        Whether to run character cleanup first. Set it ``False`` only when the
        text has already been through :func:`normalize_document` — cleaning
        twice can change the result, since some steps are not idempotent.

    Returns
    -------
    list of str
        The surviving tokens in document order. Empty for a blank document or
        one made entirely of stopwords.

    Notes
    -----
    **Punctuation is dropped but emoji are kept** when ``keep_emoji`` is set.
    Punctuation contributes nothing to a bag of words; emoji carry genuine
    sentiment. Sentiment scoring reads exclamation and question marks from the
    raw string rather than from this token stream, which is why stripping them
    here costs nothing.

    **Stopword removal happens after morphology**, so the stopword list must
    contain stemmed forms if you are stemming. The shipped lists contain
    surface forms, which means aggressive stemming can let a stopword through
    in reduced form.

    Examples
    --------
    >>> plan = build_normalize_plan()
    >>> tokenize_document("Visit https://example.com -- it's GREAT!", plan)
    ['visit', "it's", 'great']

    See Also
    --------
    normalize_document : Character cleanup only.
    build_analyzer : Wraps this for scikit-learn vectorizers.
    """
    value = normalize_document(text, plan) if normalize else ("" if text is None else str(text))
    raw = _TOKEN.findall(value)
    morph = morphology
    if morph is None and (plan.stem or plan.lemmatize):
        morph = _Morphology(plan)
    out: list[str] = []
    for token in raw:
        if not token:
            continue
        is_word = token[0].isalnum()
        if not is_word:
            # Punctuation carries no bag-of-words signal, so only pictographic
            # symbols survive; sentiment scoring reads punctuation from the raw
            # text itself rather than from this token stream.
            if plan.keep_emoji and _is_symbol(token):
                out.append(token)
            continue
        if not plan.keep_numbers and token[0].isdigit():
            continue
        if morph is not None:
            token = morph.apply(token)
        length = len(token)
        if length < plan.min_token_length or length > plan.max_token_length:
            continue
        if token in plan.stopwords:
            continue
        out.append(token)
    return out


class TextAnalyzer:
    """The callable scikit-learn uses to turn a document into terms.

    A scikit-learn vectorizer delegates "what counts as a term" to an analyzer
    callable. Supplying this one is what makes BuildML's normalisation plan
    govern the vocabulary, instead of scikit-learn's own default tokeniser.

    It is a class rather than a closure for a specific reason: a fitted
    vectorizer is saved inside an NLP bundle with ``joblib.dump``, and closures
    do not pickle. The morphology cache is deliberately dropped on pickling and
    rebuilt on first use after loading, so a bundle stays portable rather than
    carrying NLTK objects across machines.

    Attributes
    ----------
    plan:
        The normalisation plan governing tokenisation.
    low, high:
        The n-gram bounds, also available together as :attr:`ngram_range`.
    """

    __slots__ = ("_morphology", "high", "low", "plan")

    def __init__(
        self, plan: TextNormalizePlan, *, ngram_range: tuple[int, int] = (1, 1)
    ) -> None:
        """Bind a normalisation plan and n-gram range into an analyzer.

        Validates the range up front rather than at first call, so a bad
        configuration fails while you can still see where it came from instead
        of somewhere inside a vectorizer's fit.

        Parameters
        ----------
        plan:
            The resolved plan from :func:`build_normalize_plan`.
        ngram_range:
            ``(min_n, max_n)`` term lengths to emit. The default emits single
            tokens only.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            ``min_n`` is below 1 or ``max_n`` is below ``min_n``.
        """
        low, high = int(ngram_range[0]), int(ngram_range[1])
        if low < 1 or high < low:
            raise ValidationError(
                "ngram_range must be (min_n, max_n) with 1 <= min_n <= max_n."
            )
        self.plan = plan
        self.low = low
        self.high = high
        self._morphology: _Morphology | None = None

    @property
    def ngram_range(self) -> tuple[int, int]:
        """The ``(min_n, max_n)`` term lengths this analyzer emits."""
        return (self.low, self.high)

    def __call__(self, document: Any) -> list[str]:
        """Turn one document into the terms the vectorizer will count.

        Tokenises through the plan, then joins adjacent tokens into n-grams
        across the configured range. Word order survives only within an n-gram
        — this is why ``(1, 2)`` is a common default: "not good" stays intact
        as a term rather than dissolving into two independent words.

        Parameters
        ----------
        document:
            The raw value, coerced to a string.

        Returns
        -------
        list of str
            Terms in document order, with multi-token n-grams space-joined.
        """
        plan = self.plan
        morphology = self._morphology
        if morphology is None and (plan.stem or plan.lemmatize):
            morphology = _Morphology(plan)
            self._morphology = morphology
        tokens = tokenize_document(document, plan, morphology=morphology)
        if self.low == 1 and self.high == 1:
            return tokens
        grams: list[str] = []
        n_tokens = len(tokens)
        for n in range(self.low, self.high + 1):
            if n == 1:
                grams.extend(tokens)
                continue
            for start in range(n_tokens - n + 1):
                grams.append(" ".join(tokens[start : start + n]))
        return grams

    def __getstate__(self) -> dict[str, Any]:
        return {"plan": self.plan, "low": self.low, "high": self.high}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.plan = state["plan"]
        self.low = int(state["low"])
        self.high = int(state["high"])
        self._morphology = None

    def __repr__(self) -> str:
        return f"TextAnalyzer(ngram_range=({self.low}, {self.high}))"


def build_analyzer(
    plan: TextNormalizePlan, *, ngram_range: tuple[int, int] = (1, 1)
) -> TextAnalyzer:
    """Wrap a normalisation plan as a scikit-learn analyzer callable.

    Sharing one analyzer is what keeps
    :func:`~buildml.nlp.fit.fit_text_classifier`,
    :func:`~buildml.nlp.topics.fit_topics`, and
    :func:`~buildml.nlp.keyphrases.extract_keyphrases` on the same
    preprocessing contract — a token means the same thing across all three, so
    their outputs can be compared.

    Parameters
    ----------
    plan:
        The resolved plan from :func:`build_normalize_plan`.
    ngram_range:
        ``(min_n, max_n)`` term lengths to emit.

    Returns
    -------
    TextAnalyzer
        A picklable callable to pass as a vectorizer's ``analyzer``.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        The n-gram range is invalid.

    See Also
    --------
    TextAnalyzer : What the returned object does.
    """
    return TextAnalyzer(plan, ngram_range=ngram_range)


def split_sentences(text: Any, *, max_sentences: int | None = None) -> list[str]:
    """Split a document into sentences without breaking on abbreviations.

    Splitting on every full stop gets "Dr. Smith arrived." wrong three
    different ways. This splits on sentence-ending punctuation, then rejoins
    fragments that ended in a known abbreviation, so titles, "e.g.", and
    "Inc." stay attached to the sentence they belong to.

    Sentences are the unit for extractive summarisation, and they are often the
    right unit for sentiment too — a review that is positive overall can
    contain a sharply negative sentence worth surfacing on its own.

    Parameters
    ----------
    text:
        The raw document. ``None`` and non-strings are coerced; whitespace and
        line endings are normalised before splitting.
    max_sentences:
        Return at most this many, taken from the start. Useful for a preview,
        or for capping cost on very long documents. ``None`` returns all.

    Returns
    -------
    list of str
        Sentences with surrounding whitespace stripped. Empty for a blank
        document.

    Notes
    -----
    The abbreviation list is finite and English-oriented, so an unusual
    abbreviation will still cause a split. Sentence boundaries in text without
    reliable punctuation — transcripts, chat logs, OCR output — are unreliable
    for any rule-based splitter, this one included.

    Examples
    --------
    >>> split_sentences("Dr. Smith arrived. The meeting started late.")
    ['Dr. Smith arrived.', 'The meeting started late.']

    See Also
    --------
    buildml.nlp.summarize.summarize_text : Selects sentences to build a summary.
    """
    value = "" if text is None else str(text)
    value = _WHITESPACE.sub(" ", value.replace("\r\n", "\n")).strip()
    if not value:
        return []
    pieces = _SENTENCE_BOUNDARY.split(value)
    merged: list[str] = []
    for piece in pieces:
        chunk = (piece or "").strip()
        if not chunk:
            continue
        if merged and merged[-1].lower().endswith(_ABBREVIATIONS):
            merged[-1] = f"{merged[-1]} {chunk}"
            continue
        merged.append(chunk)
    if max_sentences is not None and max_sentences > 0:
        return merged[:max_sentences]
    return merged


def normalize_series(values: Any, plan: TextNormalizePlan) -> list[str]:
    """Clean many documents with the same plan.

    The bulk form of :func:`normalize_document`, for a column or any other
    iterable of raw values.

    Parameters
    ----------
    values:
        An iterable of raw documents. Each is coerced, so nulls and mixed types
        are handled.
    plan:
        The plan to apply to every document.

    Returns
    -------
    list of str
        Cleaned documents, in input order and the same length as the input.

    See Also
    --------
    normalize_document : The per-document form.
    """
    return [normalize_document(item, plan) for item in values]


def documents_from_frame(frame: Any, column: str) -> list[str]:
    """Pull a text column out of a frame as plain strings.

    Handles the coercion that a raw pandas column needs before text tooling
    will accept it: mixed types become strings and nulls become empty
    documents rather than the string ``'nan'``, which would otherwise turn into
    a spuriously frequent vocabulary term.

    Parameters
    ----------
    frame:
        The dataframe holding the column.
    column:
        Which column to extract.

    Returns
    -------
    list of str
        One string per row, in frame order. Rows that were null become ``''``,
        which downstream code counts toward the blank-document rate.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        The column is not in the frame.

    Notes
    -----
    Blank documents are kept rather than dropped, so positions stay aligned
    with the frame's rows — dropping them here would silently break the
    correspondence between predictions and the rows they describe.
    """
    if column not in frame.columns:
        raise ValidationError(f"Text column {column!r} is missing from the frame.")
    return frame[column].astype("string").fillna("").astype(str).tolist()


__all__ = [
    "TextAnalyzer",
    "TextNormalizePlan",
    "VALID_NORMALIZE_STEPS",
    "build_analyzer",
    "build_normalize_plan",
    "documents_from_frame",
    "normalize_document",
    "normalize_series",
    "split_sentences",
    "tokenize_document",
]
