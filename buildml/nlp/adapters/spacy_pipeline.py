"""Bridge to spaCy's statistical entity recogniser.

spaCy finds entities a rule cannot: person and organisation names it has never
seen, recognised from grammatical context rather than from a pattern. The
trade-off is that it will also confidently label things that are not entities,
particularly on text unlike what it was trained on.

The labels are remapped onto BuildML's own vocabulary here, so results from
spaCy and from the rules backend can be compared without the caller re-keying
anything. The mapping is approximate — spaCy's categories are finer-grained than
BuildML's, and several collapse into one.
"""

from __future__ import annotations

from typing import Any

from buildml.core.errors import ValidationError
from buildml.nlp.extras import require_spacy

DEFAULT_SPACY_MODEL = "en_core_web_sm"

# Map spaCy's ONTONOTES labels onto BuildML's rule-path label vocabulary so both
# backends can be compared without the caller re-keying anything.
SPACY_LABEL_ALIASES: dict[str, str] = {
    "PERSON": "PERSON",
    "ORG": "ORG",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "FAC": "LOCATION",
    "NORP": "GROUP",
    "PRODUCT": "PRODUCT",
    "EVENT": "EVENT",
    "WORK_OF_ART": "WORK",
    "LAW": "LAW",
    "LANGUAGE": "LANGUAGE",
    "DATE": "DATE",
    "TIME": "TIME",
    "PERCENT": "PERCENT",
    "MONEY": "MONEY",
    "QUANTITY": "QUANTITY",
    "ORDINAL": "ORDINAL",
    "CARDINAL": "CARDINAL",
}


def load_spacy_pipeline(model: str = DEFAULT_SPACY_MODEL) -> Any:
    """Load a spaCy model with only the parts entity extraction needs.

    A full spaCy pipeline does lemmatisation and text classification alongside
    entity recognition. Those components are excluded here because they cost
    time on every document and contribute nothing to the entity output.

    Parameters
    ----------
    model:
        The pipeline package to load. The default is the small English model;
        larger variants are more accurate and slower, and other languages have
        their own packages.

    Returns
    -------
    object
        The loaded spaCy ``Language`` pipeline.

    Raises
    ------
    ~buildml.core.errors.MissingExtraError
        spaCy itself is not installed. Install ``buildml[nlp-industry]``.
    ~buildml.core.errors.ValidationError
        spaCy is installed but this model is not. The message gives the exact
        download command — these are two separate failures with two separate
        fixes, which is why they raise differently.

    Notes
    -----
    **Match the model to your text's language.** An English model on French
    text does not error; it finds very few entities and reports that as a
    result.

    See Also
    --------
    extract_spacy_entities : Run a loaded pipeline over documents.
    """
    spacy = require_spacy(feature="spaCy entity extraction")
    try:
        return spacy.load(model, exclude=["lemmatizer", "textcat"])
    except OSError as exc:
        raise ValidationError(
            f"spaCy model {model!r} is not installed. Install it with: "
            f"python -m spacy download {model}"
        ) from exc


def extract_spacy_entities(
    documents: list[str],
    *,
    model: str = DEFAULT_SPACY_MODEL,
    labels: tuple[str, ...] | None = None,
    batch_size: int = 32,
) -> list[list[tuple[str, str, int, int]]]:
    """Find entities across documents with spaCy, relabelled to BuildML's vocabulary.

    Streams documents through the pipeline in batches and returns each
    mention's text, label, and character span.

    Parameters
    ----------
    documents:
        Raw document strings.
    model:
        Which spaCy pipeline to use.
    labels:
        Keep only these labels, matched against the *remapped* names rather
        than spaCy's originals — so filter for ``'LOCATION'``, not ``'GPE'``.
    batch_size:
        How many documents spaCy processes at once. Larger is faster and uses
        more memory.

    Returns
    -------
    list of list of tuple
        Per document, a list of ``(text, label, start_char, end_char)``. Offsets
        index into the raw document, so mentions can be highlighted or redacted
        in place.

    Raises
    ------
    ~buildml.core.errors.MissingExtraError
        spaCy is not installed.
    ~buildml.core.errors.ValidationError
        The requested model is not installed.

    Notes
    -----
    **The remapping loses distinctions.** spaCy separates countries and cities
    (``GPE``) from other locations (``LOC``) and from buildings (``FAC``); all
    three arrive as ``LOCATION``. Labels with no BuildML equivalent pass through
    unchanged.

    **Statistical recognition is confident when it is wrong.** On text unlike
    the model's training data — internal jargon, logs, transcripts — it
    produces plausible-looking false positives with no signal that anything is
    amiss.

    See Also
    --------
    buildml.nlp.entities.extract_rule_entities : The deterministic alternative.
    """
    pipeline = load_spacy_pipeline(model)
    wanted = None if labels is None else {str(item).upper() for item in labels}
    out: list[list[tuple[str, str, int, int]]] = []
    for doc in pipeline.pipe(documents, batch_size=max(1, int(batch_size))):
        mentions: list[tuple[str, str, int, int]] = []
        for span in doc.ents:
            label = SPACY_LABEL_ALIASES.get(span.label_, span.label_)
            if wanted is not None and label not in wanted:
                continue
            mentions.append((span.text, label, int(span.start_char), int(span.end_char)))
        out.append(mentions)
    return out


__all__ = [
    "DEFAULT_SPACY_MODEL",
    "SPACY_LABEL_ALIASES",
    "extract_spacy_entities",
    "load_spacy_pipeline",
]
