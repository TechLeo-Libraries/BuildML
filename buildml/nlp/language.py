"""Work out what language documents are written in.

Worth doing before anything else with a text column. A corpus that is 80%
English and 20% something else will quietly produce a vocabulary dominated by
one language, a stopword list that only cleans part of it, and a model whose
accuracy varies by language in a way nothing in the metrics reveals.

The built-in detector works in two stages. If the text is in a non-Latin script
— Cyrillic, Greek, Arabic, CJK — the Unicode block share settles it almost
immediately, because scripts do not overlap. Latin-script languages need more
care, so they are scored on function words: the little grammatical words that
appear constantly and differ between languages. Each marker is weighted by how
many languages share it, so "de" (Spanish, French, Portuguese) counts for much
less than a word unique to one.

When the evidence is thin, the answer is ``'und'`` rather than a guess. A
five-word string genuinely may not contain enough to distinguish Spanish from
Portuguese, and saying so is more useful than a confident wrong answer.

Nothing here learns from your corpus, so it is safe to run on any partition.
"""

from __future__ import annotations

import re
from collections import Counter

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.nlp.catalog import resolve_language_backend
from buildml.nlp.features import PartitionOrAll, documents_for, resolve_text_column
from buildml.nlp.lexicons import LANGUAGE_MARKERS, SCRIPT_LABELS, SCRIPT_RANGES
from buildml.nlp.results import NlpLanguageResult

UNDETERMINED = "und"
MIN_CHARACTERS = 12
SCRIPT_SHARE_THRESHOLD = 0.30
MIN_MARKER_SCORE = 0.04

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

# Inverse index: a marker shared by many languages carries less evidence, so its
# weight is 1/n. This is what keeps Romance-language overlap from dominating.
_MARKER_WEIGHTS: dict[str, dict[str, float]] = {}
_MARKER_OWNERS: Counter[str] = Counter()
for _language, _markers in LANGUAGE_MARKERS.items():
    for _marker in _markers:
        _MARKER_OWNERS[_marker] += 1
for _language, _markers in LANGUAGE_MARKERS.items():
    _MARKER_WEIGHTS[_language] = {
        marker: 1.0 / _MARKER_OWNERS[marker] for marker in _markers
    }


def script_shares(text: str) -> dict[str, float]:
    """Measure which writing systems a string uses.

    The first stage of detection, and by far the strongest signal available.
    Scripts do not overlap the way vocabularies do — a document in Cyrillic
    characters is not going to turn out to be English, whereas a document
    containing "the" might still be Dutch.

    Parameters
    ----------
    text:
        The string to examine. Only alphabetic characters count, so digits and
        punctuation cannot skew the proportions.

    Returns
    -------
    dict
        Script code to its share of the letters, summing to at most 1. Empty
        for a string with no letters at all.

    Notes
    -----
    Latin script is not probed, so an all-English document returns an empty
    dict. That is the signal to fall through to function-word scoring rather
    than an indication of failure.

    A mixed result — say 60% Cyrillic and 40% Latin — usually means quoted
    text, transliteration, or embedded product names, and is worth noticing
    before modelling.
    """
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return {}
    counts: Counter[str] = Counter()
    for char in letters:
        point = ord(char)
        for code, low, high in SCRIPT_RANGES:
            if low <= point <= high:
                counts[code] += 1
                break
    total = float(len(letters))
    return {code: counts[code] / total for code in counts}


def detect_document_language(
    text: str,
    *,
    min_characters: int = MIN_CHARACTERS,
) -> tuple[str, float]:
    """Identify one string's language with the built-in detector.

    Tries the script check first, since a non-Latin script is close to
    conclusive. Failing that, scores function words across the built-in
    languages, weighting each marker by how many languages share it — which is
    what stops the heavy overlap between Romance languages from swamping the
    signal.

    Parameters
    ----------
    text:
        The string to identify. ``None`` and non-strings are coerced.
    min_characters:
        Below this length, return ``'und'`` without trying. Short strings do
        not contain enough evidence, and a detector that answers anyway is
        producing noise that looks like data.

    Returns
    -------
    tuple
        ``(code, confidence)``. The code is a language code or ``'und'``.
        Confidence is the winning script's share, or the winner's share of the
        total marker score — in both cases a relative measure of how clearly
        this language beat the others, not a probability.

    Notes
    -----
    **Coverage is limited to the built-in languages** for Latin script. A
    document in a language with no marker list scores near zero everywhere and
    comes back ``'und'`` — correctly reporting "I don't know" rather than
    guessing the nearest available answer.

    **``'und'`` with a non-zero confidence** means markers were found but none
    strongly enough. That usually indicates a mixed-language document or one
    consisting mostly of names and numbers.

    Examples
    --------
    >>> code, confidence = detect_document_language(
    ...     "The quick brown fox jumps over the lazy dog and the cat."
    ... )
    >>> code
    'en'

    See Also
    --------
    detect_language : Run across a whole partition, with aggregates.
    """
    value = "" if text is None else str(text)
    stripped = value.strip()
    if len(stripped) < max(1, int(min_characters)):
        return UNDETERMINED, 0.0

    shares = script_shares(stripped)
    if shares:
        code, share = max(shares.items(), key=lambda item: item[1])
        if share >= SCRIPT_SHARE_THRESHOLD:
            return code, float(min(1.0, share))

    tokens = [token.lower() for token in _WORD.findall(stripped)]
    if not tokens:
        return UNDETERMINED, 0.0
    scores: dict[str, float] = {}
    for language, weights in _MARKER_WEIGHTS.items():
        hit = sum(weights.get(token, 0.0) for token in tokens)
        scores[language] = hit / len(tokens)
    total = sum(scores.values())
    if total <= 0.0:
        return UNDETERMINED, 0.0
    best = max(scores.items(), key=lambda item: item[1])
    if best[1] < MIN_MARKER_SCORE:
        return UNDETERMINED, float(best[1] / total) if total else 0.0
    return best[0], float(best[1] / total)


def detect_language(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll = "all",
    backend: str | None = "native",
    text_column: str | None = None,
    min_characters: int = MIN_CHARACTERS,
) -> NlpLanguageResult:
    """Identify the language of every document, and report what the corpus mixes.

    Per-document detection plus the aggregate view: which languages appear, how
    often, and how much of the corpus could not be determined.

    Parameters
    ----------
    dataset:
        The dataset holding the documents.
    split_plan:
        The split defining partitions. Required unless ``partition`` is
        ``'all'``.
    partition:
        Which rows to check. Defaults to ``'all'``, because the question is
        about the corpus rather than about a model, and detection learns
        nothing that could leak.
    backend:
        ``'native'`` for the built-in detector, or ``'langdetect'`` for wider
        language coverage and better short-string handling.
    text_column:
        Which column holds the documents. Inferred when omitted.
    min_characters:
        Below this length, documents are reported as undetermined rather than
        guessed at.

    Returns
    -------
    ~buildml.nlp.results.NlpLanguageResult
        Per-document languages and confidences, counts per language, the
        dominant language, the undetermined rate, and warnings when the corpus
        turns out to be multilingual.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        ``min_characters`` is below 1, the text column cannot be resolved, or
        the partition is empty.
    ~buildml.core.errors.MissingExtraError
        langdetect was requested without ``buildml[nlp]``.

    Notes
    -----
    **Run this before modelling.** A multilingual corpus needs a decision:
    filter to one language, model each separately, or accept that a single
    vocabulary will serve the majority language best. Discovering the mix after
    training means the decision was made for you.

    **A high undetermined rate usually means short documents**, not exotic
    languages. Check the length statistics from
    :func:`~buildml.nlp.profile.profile_text_corpus` before concluding the
    detector is at fault.

    Examples
    --------
    >>> result = detect_language(dataset, split_plan)  # doctest: +SKIP
    >>> result.dominant_language, result.language_counts  # doctest: +SKIP

    See Also
    --------
    detect_document_language : Identify a single string.
    buildml.nlp.profile.profile_text_corpus : Runs this as part of a fuller screen.
    """
    backend_key = resolve_language_backend(backend)
    if min_characters < 1:
        raise ValidationError("min_characters must be >= 1.")

    column = resolve_text_column(dataset, text_column)
    documents, _frame = documents_for(
        dataset, split_plan, partition, column, operation="detect_language"
    )

    disclosures: list[str] = []
    if backend_key == "native":
        detected = [
            detect_document_language(item, min_characters=min_characters)
            for item in documents
        ]
        disclosures.extend(
            [
                f"Native detector: Unicode script probes for "
                f"{len(SCRIPT_LABELS)} non-Latin script(s) plus "
                f"distinctiveness-weighted function words for "
                f"{list(LANGUAGE_MARKERS)}.",
                f"Documents under {min_characters} characters, or with no known "
                f"marker, are reported as {UNDETERMINED!r} instead of guessed.",
                "Non-Latin codes identify the script family (for example 'zh' means "
                "Han characters), not a verified language.",
                "Confidence is the winning language's share of total marker "
                "evidence, not a calibrated probability.",
            ]
        )
    else:
        detected = _langdetect_languages(documents, min_characters=min_characters)
        disclosures.extend(
            [
                "langdetect backend: wide-coverage n-gram profiles "
                "(buildml[nlp]); seeded for reproducibility.",
                f"Documents under {min_characters} characters are reported as "
                f"{UNDETERMINED!r}.",
            ]
        )

    languages = tuple(item[0] for item in detected)
    confidences = tuple(float(item[1]) for item in detected)
    counter = Counter(languages)
    counts = {key: int(counter[key]) for key in sorted(counter)}
    known = {key: value for key, value in counts.items() if key != UNDETERMINED}
    dominant = max(known, key=lambda key: known[key]) if known else None
    undetermined_rate = (
        float(counts.get(UNDETERMINED, 0) / len(languages)) if languages else 0.0
    )

    warnings: list[str] = []
    if undetermined_rate > 0.25:
        warnings.append(
            f"{undetermined_rate:.1%} of documents are undetermined; they are "
            "likely too short or use vocabulary outside the built-in markers. "
            "Install buildml[nlp] for langdetect coverage."
        )
    if len(known) > 1 and dominant is not None:
        dominant_share = known[dominant] / max(1, len(languages))
        if dominant_share < 0.9:
            warnings.append(
                f"Corpus is multilingual: {dominant!r} covers only "
                f"{dominant_share:.1%} of documents. A single stopword list and "
                "one shared vocabulary will underfit the minority languages."
            )

    return NlpLanguageResult(
        partition=str(partition),
        backend=backend_key,
        n_rows=len(documents),
        languages=languages,
        confidences=confidences,
        language_counts=counts,
        dominant_language=dominant,
        undetermined_rate=undetermined_rate,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )


def _langdetect_languages(
    documents: list[str],
    *,
    min_characters: int,
) -> list[tuple[str, float]]:
    from buildml.nlp.extras import require_langdetect

    langdetect = require_langdetect(feature="language backend='langdetect'")
    langdetect.DetectorFactory.seed = 0
    out: list[tuple[str, float]] = []
    for document in documents:
        text = ("" if document is None else str(document)).strip()
        if len(text) < min_characters:
            out.append((UNDETERMINED, 0.0))
            continue
        try:
            ranked = langdetect.detect_langs(text)
        except Exception:
            out.append((UNDETERMINED, 0.0))
            continue
        if not ranked:
            out.append((UNDETERMINED, 0.0))
            continue
        best = ranked[0]
        out.append((str(best.lang), float(best.prob)))
    return out


__all__ = [
    "MIN_CHARACTERS",
    "UNDETERMINED",
    "detect_document_language",
    "detect_language",
    "script_shares",
]
